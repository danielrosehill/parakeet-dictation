#!/usr/bin/env python3
"""Parakeet Dictation — On-device voice typing with punctuation via sherpa-onnx.

Supports multiple ASR model profiles (Parakeet, Canary, Nemotron) with
configurable hotkeys and VAD-segmented or true streaming transcription.
"""

import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import gi
import numpy as np
import sounddevice as sd
from ten_vad import TenVad

gi.require_version("Gtk", "3.0")
gi.require_version("AyatanaAppIndicator3", "0.1")
from gi.repository import AyatanaAppIndicator3, GLib, Gtk, Gdk

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

APP_NAME = "Parakeet Dictation"
APP_ID = "parakeet-dictation"
CONFIG_DIR = Path.home() / ".config" / APP_ID
CONFIG_FILE = CONFIG_DIR / "config.json"
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / APP_ID
MODELS_DIR = DATA_DIR / "models"
MODELS_JSON = APP_DIR / "models.json"
SAMPLE_RATE = 16000
CHUNK_SECS = 0.1  # mic callback granularity
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_SECS)
PREROLL_SECS = 0.5  # pre-press audio spliced into a new session


def _migrate_legacy_models():
    """Move models from APP_DIR/models to the XDG data directory if needed."""
    legacy = APP_DIR / "models"
    if not legacy.is_dir() or legacy == MODELS_DIR:
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for item in legacy.iterdir():
        dest = MODELS_DIR / item.name
        if dest.exists():
            continue
        try:
            shutil.move(str(item), str(dest))
        except OSError:
            # Installed read-only — copy instead
            if item.is_dir():
                shutil.copytree(str(item), str(dest))
            else:
                shutil.copy2(str(item), str(dest))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    # Model
    model_profile: str = "desktop"
    num_threads: int = min(os.cpu_count() or 4, 8)
    vad_threshold: float = 0.5

    # Max seconds of continuous speech before a chunk is force-transcribed.
    # 0 = no limit: transcribe only on a pause or when recording stops.
    max_speech_secs: float = 30.0

    # Seconds of silence that end a chunk and trigger transcription.
    # 0 = never: everything is one chunk, transcribed when recording stops.
    pause_secs: float = 0.25

    # Audio
    beep_volume: float = 0.5
    audio_device: str = ""  # Empty = system default; otherwise device name or index

    # Typing method: "clipboard" (wl-copy+Ctrl+V, works on all compositors),
    #   "wtype" (needs virtual-keyboard protocol), "ydotool" (needs daemon+uinput)
    typer: str = "clipboard"

    # Hotkey mode: "toggle" (one key) or "start_stop" (separate keys)
    hotkey_mode: str = "toggle"

    # Night mode — suppress beeps between these hours (24h format)
    night_mode: bool = True
    night_start: int = 22  # 10 PM
    night_end: int = 9     # 9 AM

    # Streaming partial-overwrite: type partials into active window and
    # backspace-retype when the model revises.  When False, partials are
    # shown only in the status bar and text is typed on final endpoint.
    partial_overwrite: bool = True

    # Strip filler words ("um", "uh", "ehm" …) before injecting text
    filter_fillers: bool = True

    # Personal vocabulary: whole-word replacements applied to transcribed
    # text, case-insensitive, e.g. {"herder": "herdr"}
    word_replacements: dict = field(default_factory=dict)

    # Language (for models that support it, e.g. Canary)
    language: str = "en"

    # Hotkey bindings (pynput format, e.g. "<ctrl>+0", "<alt>+d")
    hotkey_toggle: str = "<ctrl>+0"
    hotkey_start: str = "<ctrl>+9"
    hotkey_stop: str = "<ctrl>+8"
    hotkey_pause: str = "<ctrl>+<alt>+0"

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load() -> "AppConfig":
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                return AppConfig(**{
                    k: v for k, v in data.items()
                    if k in AppConfig.__dataclass_fields__
                })
            except Exception as e:
                print(f"WARNING: ignoring bad config {CONFIG_FILE}: {e}",
                      file=sys.stderr)
        return AppConfig()


def load_model_profiles() -> dict:
    with open(MODELS_JSON) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _generate_tone(freq: float, duration: float, volume: float) -> np.ndarray:
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), dtype=np.float32)
    tone = (volume * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    fade = min(int(SAMPLE_RATE * 0.01), len(tone) // 2)
    if fade > 0:
        tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return tone


def _is_night_mode(config: "AppConfig") -> bool:
    """Check if current time falls within night mode hours."""
    if not config.night_mode:
        return False
    hour = datetime.now().hour
    if config.night_start > config.night_end:
        # Wraps midnight: e.g. 22-9 means 22,23,0,1,...,8
        return hour >= config.night_start or hour < config.night_end
    else:
        return config.night_start <= hour < config.night_end


# canberra-gtk-play plays named events from the user's desktop sound theme,
# honoring the theme choice and alert volume in system Settings -> Sound.
_CANBERRA = shutil.which("canberra-gtk-play")


def _play_event(event_id: str, config: AppConfig, fallback_tone) -> None:
    """Play an XDG sound-theme event; sine-tone fallback without canberra."""
    volume = config.beep_volume
    if volume <= 0 or _is_night_mode(config):
        return
    if _CANBERRA:
        # Beep-volume slider becomes attenuation relative to the alert volume
        gain_db = 20 * np.log10(min(volume, 1.0))
        # ponytail: one Popen per beep (~tens of ms latency); switch to
        # ctypes libcanberra with a cached context if it feels laggy
        subprocess.Popen(
            [_CANBERRA, "-i", event_id, "-V", f"{gain_db:.1f}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    else:
        sd.play(fallback_tone(volume), samplerate=SAMPLE_RATE)


def play_beep_start(config: AppConfig):
    """Dictation started — theme 'device-added' sound (rising tone fallback)."""
    _play_event("device-added", config, lambda v: _generate_tone(880, 0.15, v))


def play_beep_stop(config: AppConfig):
    """Dictation stopped — theme 'device-removed' sound (falling tone fallback)."""
    _play_event("device-removed", config, lambda v: _generate_tone(440, 0.15, v))


def play_beep_pause(config: AppConfig):
    """Paused/resumed — theme 'message' sound (double beep fallback)."""
    def double_beep(volume):
        t = _generate_tone(660, 0.07, volume)
        gap = np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32)
        return np.concatenate([t, gap, t])
    _play_event("message", config, double_beep)


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def list_input_devices() -> list[dict]:
    """Return a list of input-capable audio devices."""
    result = []
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            result.append({"index": i, "name": dev["name"], "channels": dev["max_input_channels"]})
    return result


def resolve_audio_device(config_value: str):
    """Convert config audio_device string to a sounddevice device index or None."""
    if not config_value:
        return None
    try:
        return int(config_value)
    except ValueError:
        for dev in list_input_devices():
            if config_value in dev["name"]:
                return dev["index"]
        return None


# ---------------------------------------------------------------------------
# Filler word filter
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"\b(?:um|uh|uhm|ehm|hmm|er|ah|erm|hm)\b",
    re.IGNORECASE,
)
_MULTI_SPACE_RE = re.compile(r"  +")


def strip_fillers(text: str) -> str:
    """Remove common filler words, collapse resulting double-spaces."""
    text = _FILLER_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Word replacements (personal vocabulary)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _word_re(words: tuple[str, ...]) -> re.Pattern:
    # Longest-first so "note book" wins over "note" when both are keys.
    alts = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{alts})\b", re.IGNORECASE)


def apply_word_replacements(text: str, mapping: dict) -> str:
    """Replace whole words case-insensitively, keeping a leading capital."""
    if not mapping or not text:
        return text
    lower = {k.lower(): v for k, v in mapping.items()}

    def _sub(m: re.Match) -> str:
        rep = lower[m.group(0).lower()]
        if rep and m.group(0)[0].isupper():
            return rep[0].upper() + rep[1:]
        return rep

    return _word_re(tuple(sorted(lower))).sub(_sub, text)


# ---------------------------------------------------------------------------
# Text typer
# ---------------------------------------------------------------------------

class TextTyper:
    def __init__(self, method: str = "clipboard"):
        self._method = method
        # The current partial as typed on-screen, so a revision only needs
        # to erase/retype the differing tail instead of the whole string.
        self._partial = ""
        self._xdisplay = None
        self._mod_keycodes = ()
        if method in ("xdotool", "clipboard"):
            self._heal_modifiers()

    @staticmethod
    def _heal_modifiers():
        """Release any phantom modifier a crashed/killed injector left
        latched.  keyup-only: releases an XTEST-held modifier, no-op for
        one the user is physically holding (core state is the union of
        all device states)."""
        try:
            subprocess.run(["xdotool", "keyup", "ctrl", "shift", "alt",
                            "super"], timeout=5, start_new_session=True)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _wait_modifiers_up(self, timeout: float = 2.0) -> bool:
        """Wait until no modifier key is held; True if they were released.

        Typing while the user still holds the hotkey's Ctrl garbles the
        target app.  xdotool's --clearmodifiers "solves" that by releasing
        and then RE-PRESSING the modifier — and on Xorg that synthetic
        re-press outlives the process, latching a phantom Ctrl once the
        user's physical release has already happened.  So: never touch
        modifiers, just wait for the user's hand to lift.
        """
        try:
            from Xlib.display import Display
        except ImportError:
            return True
        if self._xdisplay is None:
            self._xdisplay = Display()
            self._mod_keycodes = {
                kc for row in self._xdisplay.get_modifier_mapping()
                for kc in row if kc}
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            keys = self._xdisplay.query_keymap()
            if not any(keys[kc >> 3] & (1 << (kc & 7))
                       for kc in self._mod_keycodes):
                return True
            time.sleep(0.02)
        return False

    def _modifiers_clear(self) -> bool:
        """True once no modifier is held; never inject on False.

        A modifier down while XTEST types turns every character into a
        hotkey chord (a held Ctrl made each 't' a Ctrl+T = new terminal
        in the focused app).  On timeout, heal a possible phantom latch
        and re-wait; a modifier still down then is physically held, so
        the caller must drop its keystrokes, not send them.
        """
        if self._wait_modifiers_up():
            return True
        self._heal_modifiers()
        return self._wait_modifiers_up()

    def _drop(self, text: str):
        print(f"WARNING: modifier key held; dropped instead of typing: "
              f"{text!r}", file=sys.stderr)

    # -- low-level helpers --------------------------------------------------

    def _type_raw(self, text: str):
        """Type a string into the active window (no newline safety)."""
        # ponytail: if/elif on self._method here, in _send_backspaces and in
        # __init__ — switch to per-method strategy objects if a 5th typer lands
        try:
            if self._method == "wtype":
                subprocess.run(["wtype", "--", text], timeout=5)
            elif self._method == "ydotool":
                subprocess.run(["ydotool", "type", "--", text], timeout=5)
            elif self._method == "xdotool":
                # X11: type directly, no clipboard involved.  No
                # --clearmodifiers (see _wait_modifiers_up); new session so
                # a terminal Ctrl+C on the app can't kill it mid-keystroke.
                # Small pieces with a modifier re-check between them: a
                # Ctrl pressed mid-injection chords at most one piece,
                # then the rest is dropped instead of typed.
                for i in range(0, len(text), 24):
                    if not self._modifiers_clear():
                        self._drop(text[i:])
                        return
                    subprocess.run(["xdotool", "type", "--delay", "2", "--",
                                    text[i:i + 24]],
                                   timeout=30, start_new_session=True)
            else:  # "clipboard" and unknown methods
                subprocess.run(["wl-copy", "--", text], timeout=5)
                time.sleep(0.05)
                if not self._modifiers_clear():
                    self._drop(text)
                    return
                subprocess.run(["xdotool", "key", "ctrl+v"], timeout=5,
                               start_new_session=True)
        except FileNotFoundError:
            print(f"ERROR: {self._method} not found.", file=sys.stderr)
        except subprocess.TimeoutExpired:
            pass

    def _send_backspaces(self, count: int):
        """Erase *count* characters via repeated BackSpace key presses."""
        if count <= 0:
            return
        try:
            if self._method in ("ydotool",):
                # ydotool key accepts X11 keycodes; BackSpace = 14
                for _ in range(count):
                    subprocess.run(["ydotool", "key", "14:1", "14:0"], timeout=5)
            elif self._method == "wtype":
                for _ in range(count):
                    subprocess.run(["wtype", "-k", "BackSpace"], timeout=5)
            else:
                # X11 (xdotool/clipboard): one process for the whole burst.
                # A held Ctrl would make every one a Ctrl+BackSpace
                # (delete-word) in the focused app — drop, never send.
                if not self._modifiers_clear():
                    self._drop(f"<{count} backspaces>")
                    return
                subprocess.run(["xdotool", "key", "--delay", "2",
                                "--repeat", str(count), "BackSpace"],
                               timeout=10, start_new_session=True)
        except FileNotFoundError:
            print(f"ERROR: backspace helper not found for {self._method}.", file=sys.stderr)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _sanitize(text: str) -> str:
        """Strip newlines/carriage-returns — never inject Enter."""
        return text.replace("\n", " ").replace("\r", " ").strip()

    @staticmethod
    def _diff(old: str, new: str) -> tuple[int, str]:
        """Return (backspaces, suffix) that edits *old* into *new*."""
        i = len(os.path.commonprefix((old, new)))
        return len(old) - i, new[i:]

    # -- public API ---------------------------------------------------------

    def type_text(self, text: str):
        """Type final (committed) text — adds trailing space."""
        text = self._sanitize(text)
        if not text:
            return
        self._type_raw(text + " ")

    def type_partial(self, text: str):
        """Type a streaming partial, erasing only what the revision changed."""
        text = self._sanitize(text)
        if not text or text == self._partial:
            return
        if not self._wait_modifiers_up():
            return  # user is holding a modifier; drop — a fresher partial follows
        backspaces, suffix = self._diff(self._partial, text)
        self._send_backspaces(backspaces)
        if suffix:
            self._type_raw(suffix)
        self._partial = text

    def commit_partial(self, text: str):
        """Commit (finalize) a partial: fix up the tail, add trailing space."""
        text = self._sanitize(text)
        backspaces, suffix = self._diff(self._partial, text)
        self._send_backspaces(backspaces)
        if text:
            self._type_raw(suffix + " ")
        self._partial = ""

    def reset_partial(self):
        """Discard partial tracking without erasing anything on screen."""
        self._partial = ""


# ---------------------------------------------------------------------------
# TEN VAD wrapper — lightweight voice activity detection (~306 KB)
# Provides segment-based interface compatible with the ASR engine.
# ---------------------------------------------------------------------------

class _SpeechSegment:
    """A completed speech segment with audio samples."""
    __slots__ = ("samples",)

    def __init__(self, samples: list[float]):
        self.samples = samples


class TenVadDetector:
    """TEN VAD wrapper that accumulates speech and yields segments on silence."""

    def __init__(self, threshold: float = 0.5, min_silence_duration: float = 0.25,
                 min_speech_duration: float = 0.25, max_speech_duration: float = 30.0,
                 sample_rate: int = 16000):
        self._hop_size = 256  # ~16ms at 16kHz — TEN VAD optimal
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._min_silence_samples = int(min_silence_duration * sample_rate)
        self._min_speech_samples = int(min_speech_duration * sample_rate)
        # <= 0 means unlimited: segment only on silence or explicit flush()
        self._max_speech_samples = (
            int(max_speech_duration * sample_rate) if max_speech_duration > 0 else 0
        )

        self._vad = TenVad(hop_size=self._hop_size, threshold=threshold)

        # Internal state
        self._buffer: list[float] = []  # float32 samples for ASR
        # Sub-hop leftover, kept as float samples so the audio buffered for
        # ASR and the audio the VAD classifies are sliced from the same
        # array — a separate int16 remainder desynced them and dropped ~64
        # samples of real audio per call at chunk boundaries.
        self._remainder: list[float] = []
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._segments: list[_SpeechSegment] = []
        self._is_speech = False
        # Pre-roll: keep ~0.3s of the most recent non-speech audio and
        # prepend it when speech starts, so the soft onset of the first
        # word (before the detector crosses threshold) isn't clipped.
        self._preroll: deque[list[float]] = deque(
            maxlen=max(1, int(0.3 * sample_rate) // self._hop_size))

    def accept_waveform(self, samples: list[float]):
        """Feed float32 audio samples (matching sounddevice output)."""
        # Prepend any leftover from previous call
        data = self._remainder + list(samples)
        # Convert to int16 for TEN VAD
        int16_data = (np.asarray(data, dtype=np.float32) * 32767).astype(np.int16)

        # Process in hop_size chunks
        i = 0
        while i + self._hop_size <= len(int16_data):
            chunk = int16_data[i:i + self._hop_size]
            prob, _flag = self._vad.process(chunk)
            is_speech = prob >= self._threshold

            float_chunk = data[i:i + self._hop_size]

            if is_speech:
                self._is_speech = True
                self._silence_samples = 0
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_samples = 0
                    for pre in self._preroll:
                        self._buffer.extend(pre)
                    self._preroll.clear()
                self._buffer.extend(float_chunk)
                self._speech_samples += self._hop_size

                # Force segment if max duration reached (0 = unlimited)
                if self._max_speech_samples and self._speech_samples >= self._max_speech_samples:
                    self._emit_segment()
            else:
                if self._in_speech:
                    self._buffer.extend(float_chunk)
                    self._silence_samples += self._hop_size
                    if self._silence_samples >= self._min_silence_samples:
                        self._emit_segment()
                else:
                    self._is_speech = False
                    self._preroll.append(float_chunk)

            i += self._hop_size

        # Save leftover
        self._remainder = data[i:]

    def _emit_segment(self):
        """Finalize current speech buffer into a segment."""
        # Gate on actual speech samples — the buffer also holds pre-roll
        # and trailing silence, which must not qualify a blip as speech.
        if self._speech_samples >= self._min_speech_samples:
            self._segments.append(_SpeechSegment(list(self._buffer)))
        self._buffer.clear()
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speech = False

    def is_speech_detected(self) -> bool:
        return self._is_speech

    def empty(self) -> bool:
        return len(self._segments) == 0

    @property
    def front(self) -> _SpeechSegment:
        return self._segments[0]

    def pop(self):
        self._segments.pop(0)

    def flush(self):
        """Emit any remaining buffered speech."""
        if self._buffer:
            self._emit_segment()


# ---------------------------------------------------------------------------
# ASR Engine — supports offline (VAD-segmented) and streaming modes
# ---------------------------------------------------------------------------

class ASREngine:
    def __init__(self, config: AppConfig, profile: dict, on_text, on_partial, on_error,
                 on_partial_type=None, on_commit_partial=None):
        self._config = config
        self._profile = profile
        self._on_text = on_text
        self._on_partial = on_partial
        self._on_error = on_error
        self._on_partial_type = on_partial_type or (lambda t: None)
        self._on_commit_partial = on_commit_partial or (lambda t: None)
        self._running = False
        self._paused = False
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = NOT paused
        self._pause_event.set()
        # Monotonic stamp for streaming partials.  The GTK idle queue can
        # lag behind the 100 ms partial cadence; stale queued partials must
        # be dropped, not typed, or the app keeps typing after stop.
        self._partial_seq = 0
        # The model is cached across starts and loaded in the background at
        # engine build — a cold load takes seconds and used to run between
        # the hotkey press and the mic opening, eating the first sentence.
        self._recognizer = None
        self._recognizer_lock = threading.Lock()
        # Persistent mic: opened once (in _warm) and never closed between
        # sessions — opening the stream per press cost 100-300 ms of lost
        # speech.  While idle the callback keeps the newest PREROLL_SECS
        # in a ring; _begin_capture splices that in front of the live
        # queue, so words spoken at (or just before) the hotkey press
        # are transcribed too.
        self._mic_stream = None
        self._mic_dev = None
        self._mic_lock = threading.Lock()
        self._route_lock = threading.Lock()
        self._capturing = False
        self._audio_q: queue.Queue = queue.Queue()
        self._preroll_ring: deque = deque(
            maxlen=max(1, int(PREROLL_SECS / CHUNK_SECS)))
        threading.Thread(target=self._warm, daemon=True).start()

    def _get_recognizer(self):
        with self._recognizer_lock:
            if self._recognizer is None:
                if self._profile.get("streaming", False):
                    self._recognizer = self._build_online_recognizer()
                else:
                    self._recognizer = self._build_offline_recognizer()
            return self._recognizer

    def _warm(self):
        try:
            # Mic first: the stream must be live before any model work so
            # the ring holds audio from the moment the app starts.
            self._ensure_mic()
        except Exception:
            pass  # no input device — reported when dictation starts
        try:
            self._ensure_models()
            recognizer = self._get_recognizer()
            # First inference allocates lazily; push 0.5s of silence through
            # so the first real segment decodes at full speed.
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, [0.0] * (SAMPLE_RATE // 2))
            if self._profile.get("streaming", False):
                while recognizer.is_ready(s):
                    recognizer.decode_stream(s)
            else:
                recognizer.decode_stream(s)
        except Exception:
            pass  # missing models etc. are reported when dictation starts

    def _get_model_dir(self) -> Path:
        return MODELS_DIR / self._config.model_profile

    def _ensure_models(self):
        model_dir = self._get_model_dir()
        profile_files = self._profile.get("files", {})
        missing = []
        for key, info in profile_files.items():
            fp = model_dir / info["filename"]
            if not fp.exists():
                missing.append(info["filename"])
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {', '.join(missing)}\n"
                f"Run: python download_models.py {self._config.model_profile}"
            )

    def _build_offline_recognizer(self):
        import sherpa_onnx
        model_dir = self._get_model_dir()
        files = self._profile["files"]
        decoder_type = self._profile.get("decoder_type", "transducer")

        if decoder_type == "transducer":
            return sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=str(model_dir / files["encoder"]["filename"]),
                decoder=str(model_dir / files["decoder"]["filename"]),
                joiner=str(model_dir / files["joiner"]["filename"]),
                tokens=str(model_dir / files["tokens"]["filename"]),
                num_threads=self._config.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=self._profile.get("feature_dim", 128),
                provider="cpu",
                model_type=self._profile.get("model_type", "nemo_transducer"),
                decoding_method="greedy_search",
            )
        elif decoder_type == "canary":
            return sherpa_onnx.OfflineRecognizer.from_nemo_canary(
                encoder=str(model_dir / files["encoder"]["filename"]),
                decoder=str(model_dir / files["decoder"]["filename"]),
                tokens=str(model_dir / files["tokens"]["filename"]),
                src_lang=self._config.language,
                tgt_lang=self._config.language,
                num_threads=self._config.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=self._profile.get("feature_dim", 128),
                provider="cpu",
                decoding_method="greedy_search",
            )
        else:
            return sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                model=str(model_dir / files["model"]["filename"]),
                tokens=str(model_dir / files["tokens"]["filename"]),
                num_threads=self._config.num_threads,
                sample_rate=SAMPLE_RATE,
                feature_dim=self._profile.get("feature_dim", 128),
                provider="cpu",
                decoding_method="greedy_search",
            )

    def _build_online_recognizer(self):
        import sherpa_onnx
        model_dir = self._get_model_dir()
        files = self._profile["files"]
        return sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=str(model_dir / files["encoder"]["filename"]),
            decoder=str(model_dir / files["decoder"]["filename"]),
            joiner=str(model_dir / files["joiner"]["filename"]),
            tokens=str(model_dir / files["tokens"]["filename"]),
            num_threads=self._config.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=self._profile.get("feature_dim", 128),
            provider="cpu",
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=300,
        )

    def _build_vad(self):
        # ponytail: pause_secs 0 maps to an unreachably-large silence threshold
        # = never emit on pause — the whole recording buffers in RAM and
        # flush() on stop sends one chunk.  ~27 MB/min of audio; fine for
        # dictation.
        return TenVadDetector(
            threshold=self._config.vad_threshold,
            min_silence_duration=self._config.pause_secs or 10**9,
            min_speech_duration=0.25,
            max_speech_duration=self._config.max_speech_secs,
            sample_rate=SAMPLE_RATE,
        )

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_paused(self) -> bool:
        return self._paused

    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._pause_event.set()
        self._paused = False
        # Set before the thread runs: toggle() must see "running" the
        # moment start() returns, or a second hotkey fire during init
        # starts a second capture thread instead of stopping this one.
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._partial_seq += 1  # invalidate any queued partial typing
        self._stop_event.set()
        self._pause_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        self._paused = False

    def _emit_partial_type(self, seq: int, text: str):
        """Type a partial only if it is still the newest one produced."""
        if seq == self._partial_seq:
            self._on_partial_type(text)

    def toggle_pause(self):
        if not self._running:
            return
        if self._paused:
            self._paused = False
            self._pause_event.set()
            play_beep_pause(self._config)
            GLib.idle_add(self._on_partial, "Resumed")
        else:
            self._paused = True
            self._pause_event.clear()
            play_beep_pause(self._config)
            GLib.idle_add(self._on_partial, "Paused")

    def _run(self):
        try:
            self._ensure_models()
        except Exception as e:
            self._running = False
            GLib.idle_add(self._on_error, str(e))
            return

        is_streaming = self._profile.get("streaming", False)

        try:
            if is_streaming:
                self._run_streaming()
            else:
                self._run_offline()
        except Exception as e:
            GLib.idle_add(self._on_error, str(e))
        finally:
            self._running = False
            play_beep_stop(self._config)

    def _on_audio(self, indata, _frames, _time, _status):
        """PortAudio callback — never blocked by decoding; the queue
        absorbs any backlog.  (The blocking read() path held only ~0.4s
        and dropped overrun silently, eating the words after a decode
        stall.)  The lock only guards routing flips; it is held for
        microseconds."""
        chunk = indata.reshape(-1).copy()
        with self._route_lock:
            if self._capturing:
                self._audio_q.put(chunk)
            else:
                self._preroll_ring.append(chunk)

    def _ensure_mic(self):
        """Open the persistent input stream; reopen on device change."""
        with self._mic_lock:
            device = resolve_audio_device(self._config.audio_device)
            if self._mic_stream is not None:
                if self._mic_dev == device and self._mic_stream.active:
                    return
                try:
                    self._mic_stream.close()
                except Exception:
                    pass
                self._mic_stream = None
            stream = sd.InputStream(
                device=device, channels=1, dtype="float32",
                samplerate=SAMPLE_RATE, blocksize=CHUNK_SAMPLES,
                callback=self._on_audio,
            )
            stream.start()
            self._mic_stream = stream
            self._mic_dev = device

    def _begin_capture(self) -> queue.Queue:
        """Route mic audio into a fresh queue, pre-press ring first."""
        self._ensure_mic()
        q: queue.Queue = queue.Queue()
        with self._route_lock:
            while self._preroll_ring:
                q.put(self._preroll_ring.popleft())
            self._audio_q = q
            self._capturing = True
        return q

    def _end_capture(self):
        with self._route_lock:
            self._capturing = False

    def close(self):
        """Release the mic — call before dropping the engine."""
        self.stop()
        with self._mic_lock:
            if self._mic_stream is not None:
                try:
                    self._mic_stream.close()
                except Exception:
                    pass
                self._mic_stream = None

    def _drain(self, audio_q) -> list:
        """Pop everything currently queued (used for pause/stop)."""
        chunks = []
        while True:
            try:
                chunks.append(audio_q.get_nowait())
            except queue.Empty:
                return chunks

    def _decode_pending(self, recognizer, vad):
        """Transcribe every completed VAD segment and emit its text."""
        while not vad.empty():
            s = recognizer.create_stream()
            s.accept_waveform(SAMPLE_RATE, vad.front.samples)
            recognizer.decode_stream(s)
            text = s.result.text.strip()
            if text:
                GLib.idle_add(self._on_text, text)
            vad.pop()

    def _run_offline(self):
        audio_q = self._begin_capture()
        try:
            play_beep_start(self._config)
            GLib.idle_add(self._on_partial, "")
            # Fetched after capture starts: a cold model load (seconds)
            # then delays the first text, but loses no audio.
            recognizer = self._get_recognizer()
            vad = self._build_vad()

            while not self._stop_event.is_set():
                self._pause_event.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                if self._paused:
                    self._drain(audio_q)  # discard mic audio while paused
                    continue

                try:
                    audio = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                vad.accept_waveform(audio.tolist())

                if vad.is_speech_detected():
                    GLib.idle_add(self._on_partial, "Listening...")

                self._decode_pending(recognizer, vad)

            # Feed audio captured but not yet consumed, then flush
            self._end_capture()
            for audio in self._drain(audio_q):
                vad.accept_waveform(audio.tolist())
            vad.flush()
            self._decode_pending(recognizer, vad)
        finally:
            self._end_capture()

    def _run_streaming(self):
        audio_q = self._begin_capture()
        try:
            play_beep_start(self._config)
            GLib.idle_add(self._on_partial, "")
            recognizer = self._get_recognizer()
            stream = recognizer.create_stream()
            partial_overwrite = self._config.partial_overwrite

            while not self._stop_event.is_set():
                self._pause_event.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                if self._paused:
                    self._drain(audio_q)  # discard mic audio while paused
                    continue

                try:
                    audio = audio_q.get(timeout=0.1)
                except queue.Empty:
                    continue
                stream.accept_waveform(SAMPLE_RATE, audio.tolist())

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                partial = recognizer.get_result(stream).strip()
                if partial:
                    GLib.idle_add(self._on_partial, partial)
                    if partial_overwrite:
                        self._partial_seq += 1
                        GLib.idle_add(self._emit_partial_type,
                                      self._partial_seq, partial)

                if recognizer.is_endpoint(stream):
                    text = recognizer.get_result(stream).strip()
                    if text:
                        if partial_overwrite:
                            GLib.idle_add(self._on_commit_partial, text)
                        else:
                            GLib.idle_add(self._on_text, text)
                    recognizer.reset(stream)
        finally:
            self._end_capture()


# ---------------------------------------------------------------------------
# Dictation controller
# ---------------------------------------------------------------------------

class DictationController:
    def __init__(self, config: AppConfig):
        self._config = config
        self._typer = TextTyper(config.typer)
        self._profiles_data = load_model_profiles()
        self._engine = None
        self._status_callback = None
        self._rebuild_engine()

    def _rebuild_engine(self):
        if self._engine is not None:
            self._engine.close()  # release the persistent mic before swapping
        profile = self._profiles_data["profiles"].get(self._config.model_profile)
        if not profile:
            profile = self._profiles_data["profiles"]["desktop"]
        # Callers mutate the shared config object before apply_config, so a
        # change can only be detected against the profile the engine holds.
        self._engine_profile = self._config.model_profile
        self._engine = ASREngine(
            self._config, profile,
            on_text=self._on_final_text,
            on_partial=self._on_partial,
            on_error=self._on_error,
            on_partial_type=self._on_partial_type,
            on_commit_partial=self._on_commit_partial,
        )

    def set_status_callback(self, cb):
        self._status_callback = cb

    @property
    def is_running(self) -> bool:
        return self._engine.is_running

    @property
    def is_paused(self) -> bool:
        return self._engine.is_paused

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def profiles(self) -> dict:
        return self._profiles_data["profiles"]

    @property
    def profiles_data(self) -> dict:
        return self._profiles_data

    def start(self):
        if not self._engine.is_running:
            self._engine.start()

    def stop(self):
        self._engine.stop()
        self._typer.reset_partial()
        if self._status_callback:
            self._status_callback("")

    def toggle(self):
        if self._engine.is_running:
            self.stop()
        else:
            self.start()

    def toggle_pause(self):
        self._engine.toggle_pause()

    def switch_model(self, model_id: str):
        self._config.model_profile = model_id
        self.apply_config(self._config)

    def apply_config(self, new_config: AppConfig):
        was_running = self._engine.is_running
        if was_running:
            self._engine.stop()
        self._config = new_config
        self._config.save()
        self._typer = TextTyper(new_config.typer)
        if new_config.model_profile != self._engine_profile or was_running:
            self._rebuild_engine()

    def _postprocess(self, text: str) -> str:
        if self._config.filter_fillers:
            text = strip_fillers(text)
        return apply_word_replacements(text, self._config.word_replacements)

    def _on_final_text(self, text: str):
        text = self._postprocess(text)
        if not text:
            return
        self._typer.type_text(text)
        if self._status_callback:
            self._status_callback("")

    def _on_partial_type(self, text: str):
        """Type a streaming partial into the active window (overwrite prev)."""
        text = self._postprocess(text)
        if text:
            self._typer.type_partial(text)

    def _on_commit_partial(self, text: str):
        """Commit the streaming partial as final text."""
        text = self._postprocess(text)
        self._typer.commit_partial(text)
        if self._status_callback:
            self._status_callback("")

    def _on_partial(self, text: str):
        if self._status_callback:
            self._status_callback(text)

    def _on_error(self, msg: str):
        print(f"ERROR: {msg}", file=sys.stderr)
        if self._status_callback:
            self._status_callback(f"Error: {msg[:60]}")


# ---------------------------------------------------------------------------
# Hotkey manager
# ---------------------------------------------------------------------------

class HotkeyManager:
    def __init__(self, config: AppConfig, on_toggle, on_start, on_stop, on_pause):
        self._config = config
        self._on_toggle = on_toggle
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_pause = on_pause
        self._listener = None

    @staticmethod
    def _debounce(fn, gap: float = 1.0):
        """One activation per hotkey press, however long it is held.

        X11 auto-repeat re-delivers a held key as release+press pulses
        (~33/s after a ~0.5s delay) and pynput re-activates on every
        pulse — each re-fire toggled dictation off right after it
        started.  Fire only when the previous pulse is at least *gap*
        old; every suppressed pulse re-arms the window, so pulses 30ms
        apart never fire no matter how long the key is held.
        """
        last = [None]

        def wrapper():
            now = time.monotonic()
            prev = last[0]
            last[0] = now
            if prev is None or now - prev >= gap:
                fn()

        return wrapper

    def start(self):
        from pynput import keyboard
        bindings = {}
        if self._config.hotkey_mode == "toggle":
            bindings[self._config.hotkey_toggle] = lambda: GLib.idle_add(self._on_toggle)
        else:
            bindings[self._config.hotkey_start] = lambda: GLib.idle_add(self._on_start)
            bindings[self._config.hotkey_stop] = lambda: GLib.idle_add(self._on_stop)

        if self._config.hotkey_pause:
            bindings[self._config.hotkey_pause] = lambda: GLib.idle_add(self._on_pause)

        bindings = {key: self._debounce(fn) for key, fn in bindings.items()}
        self._listener = keyboard.GlobalHotKeys(bindings)
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None

    def rebuild(self, config: AppConfig):
        self._config = config
        self.stop()
        self.start()


# ---------------------------------------------------------------------------
# Hotkey capture widget
# ---------------------------------------------------------------------------

class HotkeyCaptureButton(Gtk.Button):
    def __init__(self, current_binding: str):
        super().__init__(label=self._display(current_binding))
        self._binding = current_binding
        self._capturing = False
        self._key_handler = None
        self.connect("clicked", self._on_clicked)

    @property
    def binding(self) -> str:
        return self._binding

    @staticmethod
    def _display(binding: str) -> str:
        return binding.replace("<", "").replace(">", "").replace("+", " + ").title()

    def _on_clicked(self, _btn):
        if self._capturing:
            return
        self._capturing = True
        self.set_label("Press a key combo...")
        self._key_handler = self.get_toplevel().connect("key-press-event", self._on_key)

    def _on_key(self, _widget, event):
        if not self._capturing:
            return False

        # A modifier press alone must NOT end capture — keep waiting
        # for the main key (fixes capturing e.g. Ctrl+E).
        keyname = Gdk.keyval_name(event.keyval).lower()
        if keyname in ("control_l", "control_r", "alt_l", "alt_r",
                       "shift_l", "shift_r", "super_l", "super_r",
                       "meta_l", "meta_r"):
            return True

        self._capturing = False
        self.get_toplevel().disconnect(self._key_handler)

        if keyname == "escape":
            # Escape cancels capture and keeps the old binding
            self.set_label(self._display(self._binding))
            return True

        parts = []
        if event.state & Gdk.ModifierType.CONTROL_MASK:
            parts.append("<ctrl>")
        if event.state & Gdk.ModifierType.MOD1_MASK:
            parts.append("<alt>")
        if event.state & Gdk.ModifierType.SHIFT_MASK:
            parts.append("<shift>")
        if event.state & Gdk.ModifierType.SUPER_MASK:
            parts.append("<cmd>")  # pynput's name for the Super/Win key

        # GDK key names pynput spells differently
        keyname = {"return": "enter", "kp_enter": "enter", "prior": "page_up",
                   "next": "page_down", "print": "print_screen"}.get(keyname, keyname)
        # pynput wants special keys angle-bracketed ("<f9>", "<pause>");
        # a bare multi-char name makes GlobalHotKeys raise and kills all
        # hotkeys on the next rebuild.
        parts.append(keyname if len(keyname) == 1 else f"<{keyname}>")

        binding = "+".join(parts)
        from pynput.keyboard import HotKey
        try:
            HotKey.parse(binding)
        except ValueError:
            # Key unknown to pynput — reject capture, keep old binding
            self.set_label(self._display(self._binding))
            return True

        self._binding = binding
        self.set_label(self._display(self._binding))
        return True


# ---------------------------------------------------------------------------
# Settings dialog (tabbed: Models, Hotkeys, About)
# ---------------------------------------------------------------------------

def _any_model_downloaded(profiles: dict) -> bool:
    """Check if at least one model is downloaded."""
    for mid in profiles:
        if _is_model_downloaded(mid, profiles):
            return True
    return False


def _is_model_downloaded(model_id: str, profiles: dict) -> bool:
    """Check if all files for a model are present on disk."""
    profile = profiles.get(model_id)
    if not profile:
        return False
    model_dir = MODELS_DIR / model_id
    for key, info in profile.get("files", {}).items():
        if not (model_dir / info["filename"]).exists():
            return False
    return True


def _download_file(url: str, dest: Path, on_progress_bytes=None):
    """Download a single file with progress reporting. Raises on failure."""
    import requests

    resp = requests.get(url, stream=True, timeout=(15, 60))
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if on_progress_bytes:
                    on_progress_bytes(downloaded, total)
        tmp.rename(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _download_models(model_ids: list, profiles_data: dict, on_progress, on_done):
    """Download the given models sequentially in a background thread."""

    def _worker():
        try:
            profiles = profiles_data["profiles"]
            for mid in model_ids:
                profile = profiles[mid]
                model_dir = MODELS_DIR / mid
                model_dir.mkdir(parents=True, exist_ok=True)
                files = profile["files"]
                total_files = len(files)
                prefix = f"{profile['name'][:18]}: " if len(model_ids) > 1 else ""
                for i, info in enumerate(files.values(), 1):
                    dest = model_dir / info["filename"]
                    if dest.exists() and dest.stat().st_size > 0:
                        continue

                    def _file_progress(done, total, p=prefix,
                                       fname=info["filename"], idx=i):
                        mb = done / 1024 / 1024
                        if total:
                            GLib.idle_add(
                                on_progress,
                                f"{p}{fname} ({idx}/{total_files}): "
                                f"{mb:.0f}/{total / 1024 / 1024:.0f} MB",
                                done / total,
                            )
                        else:
                            GLib.idle_add(on_progress, f"{p}{fname}: {mb:.0f} MB", -1.0)

                    _download_file(info["url"], dest, _file_progress)
            GLib.idle_add(on_done, True, "")
        except Exception as e:
            GLib.idle_add(on_done, False, str(e))

    threading.Thread(target=_worker, daemon=True).start()


LANG_LABELS = {"en": "English", "es": "Spanish", "de": "German", "fr": "French"}


class WelcomeDialog(Gtk.Dialog):
    """First-run dialog — downloads all models automatically."""

    def __init__(self, profiles_data: dict, config: AppConfig, on_model_ready):
        super().__init__(title=f"Welcome to {APP_NAME}", flags=0)
        self._profiles_data = profiles_data
        self._profiles = profiles_data["profiles"]
        self._config = config
        self._on_model_ready = on_model_ready
        self.set_default_size(440, 280)
        self.set_deletable(False)

        box = self.get_content_area()
        box.set_spacing(12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(16)
        box.set_margin_bottom(16)

        header = Gtk.Label()
        header.set_markup(
            f"<span size='x-large' weight='bold'>Welcome to {APP_NAME}</span>"
        )
        header.set_halign(Gtk.Align.START)
        box.pack_start(header, False, False, 0)

        total_mb = sum(m["size_mb"] for m in self._profiles.values())
        subtitle = Gtk.Label()
        subtitle.set_markup(
            f"Three speech recognition models will be downloaded\n"
            f"so you can switch between them freely.\n\n"
            f"Total download: <b>~{total_mb} MB</b>"
        )
        subtitle.set_halign(Gtk.Align.START)
        subtitle.set_line_wrap(True)
        box.pack_start(subtitle, False, False, 0)

        # Model summary (read-only)
        for mid, mdata in self._profiles.items():
            lbl = Gtk.Label()
            tag = "Streaming" if mdata.get("streaming") else "VAD-segmented"
            lbl.set_markup(
                f"  \u2022 <b>{mdata['name']}</b>  ({mdata['size_mb']} MB, {tag})"
            )
            lbl.set_halign(Gtk.Align.START)
            lbl.get_style_context().add_class("dim-label")
            box.pack_start(lbl, False, False, 0)

        # Download button
        self._dl_btn = Gtk.Button()
        dl_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        dl_hbox.set_halign(Gtk.Align.CENTER)
        dl_hbox.pack_start(
            Gtk.Image.new_from_icon_name("folder-download-symbolic", Gtk.IconSize.BUTTON),
            False, False, 0,
        )
        dl_hbox.pack_start(Gtk.Label(label="Download All Models"), False, False, 0)
        self._dl_btn.add(dl_hbox)
        self._dl_btn.get_style_context().add_class("suggested-action")
        self._dl_btn.set_margin_top(8)
        self._dl_btn.connect("clicked", self._on_download_all)
        box.pack_start(self._dl_btn, False, False, 0)

        # Progress bar (hidden until download starts)
        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_show_text(True)
        self._progress_bar.set_no_show_all(True)
        box.pack_start(self._progress_bar, False, False, 0)

        # Error label (hidden until error)
        self._error_label = Gtk.Label()
        self._error_label.set_line_wrap(True)
        self._error_label.set_max_width_chars(60)
        self._error_label.set_halign(Gtk.Align.START)
        self._error_label.set_no_show_all(True)
        box.pack_start(self._error_label, False, False, 0)

        self.show_all()

    def _update_progress(self, msg, fraction):
        self._progress_bar.set_text(msg)
        if fraction >= 0:
            self._progress_bar.set_fraction(min(fraction, 1.0))
        else:
            self._progress_bar.pulse()

    def _on_download_all(self, btn):
        btn.set_sensitive(False)
        self._progress_bar.show()
        self._error_label.hide()

        def on_progress(msg, fraction):
            self._update_progress(msg, fraction)

        def on_done(success, err):
            if success:
                self._config.model_profile = "desktop"
                self._config.save()
                self.destroy()
                if self._on_model_ready:
                    self._on_model_ready(self._config)
            else:
                btn.set_sensitive(True)
                self._progress_bar.hide()
                self._error_label.set_markup(f"<span color='red'>Download failed: {GLib.markup_escape_text(err)}</span>")
                self._error_label.show()
                print(f"Download error: {err}", file=sys.stderr)

        _download_models(list(self._profiles), self._profiles_data,
                         on_progress, on_done)


class SettingsDialog(Gtk.Dialog):
    def __init__(self, config: AppConfig, profiles_data: dict, on_save):
        super().__init__(title=f"{APP_NAME} — Settings", flags=0)
        self._config = config
        self._profiles_data = profiles_data
        self._profiles = profiles_data["profiles"]
        self._on_save = on_save
        self.set_default_size(520, 560)

        notebook = Gtk.Notebook()
        self.get_content_area().pack_start(notebook, True, True, 0)

        notebook.append_page(self._build_models_tab(), Gtk.Label(label="Models"))
        notebook.append_page(self._build_hotkeys_tab(), Gtk.Label(label="Hotkeys"))
        notebook.append_page(self._build_general_tab(), Gtk.Label(label="General"))
        notebook.append_page(self._build_about_tab(), Gtk.Label(label="About"))

        self.show_all()

    # --- Models tab ---

    def _build_models_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(16)
        box.set_margin_end(16)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        self._model_status_label = Gtk.Label()
        self._model_status_label.set_halign(Gtk.Align.START)
        box.pack_start(self._model_status_label, False, False, 0)

        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self._model_list = Gtk.ListBox()
        self._model_list.set_selection_mode(Gtk.SelectionMode.NONE)
        sw.add(self._model_list)
        box.pack_start(sw, True, True, 0)

        # Download All button
        self._dl_all_btn = Gtk.Button()
        dl_all_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        dl_all_hbox.set_halign(Gtk.Align.CENTER)
        dl_all_hbox.pack_start(
            Gtk.Image.new_from_icon_name("folder-download-symbolic", Gtk.IconSize.BUTTON),
            False, False, 0,
        )
        total_mb = sum(m["size_mb"] for m in self._profiles.values())
        self._dl_all_label = Gtk.Label(label=f"Download All Models ({total_mb} MB)")
        dl_all_hbox.pack_start(self._dl_all_label, False, False, 0)
        self._dl_all_btn.add(dl_all_hbox)
        self._dl_all_btn.connect("clicked", self._on_download_all)
        box.pack_start(self._dl_all_btn, False, False, 0)

        # Progress bar (hidden until download starts)
        self._dl_progress = Gtk.ProgressBar()
        self._dl_progress.set_show_text(True)
        self._dl_progress.set_no_show_all(True)
        box.pack_start(self._dl_progress, False, False, 0)

        # Error label (hidden until error)
        self._dl_error = Gtk.Label()
        self._dl_error.set_line_wrap(True)
        self._dl_error.set_max_width_chars(60)
        self._dl_error.set_halign(Gtk.Align.START)
        self._dl_error.set_no_show_all(True)
        box.pack_start(self._dl_error, False, False, 0)

        self._populate_models()
        return box

    def _populate_models(self):
        for child in self._model_list.get_children():
            self._model_list.remove(child)

        active = self._config.model_profile
        self._model_status_label.set_markup(
            f"Active: <b>{self._profiles.get(active, {}).get('name', active)}</b>"
        )

        for mid, mdata in self._profiles.items():
            row = Gtk.ListBoxRow()
            row.set_activatable(False)
            hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            hbox.set_margin_start(8)
            hbox.set_margin_end(8)
            hbox.set_margin_top(6)
            hbox.set_margin_bottom(6)

            # Info column
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            name_label = Gtk.Label()
            name_label.set_markup(f"<b>{mdata['name']}</b>")
            name_label.set_halign(Gtk.Align.START)
            vbox.pack_start(name_label, False, False, 0)

            desc = mdata.get("description", "")
            desc_label = Gtk.Label(label=desc)
            desc_label.set_halign(Gtk.Align.START)
            desc_label.set_line_wrap(True)
            desc_label.set_max_width_chars(50)
            desc_label.get_style_context().add_class("dim-label")
            vbox.pack_start(desc_label, False, False, 0)

            rec = mdata.get("recommended_for", "")
            hw = mdata.get("hardware_label", "CPU")
            langs = mdata.get("languages")
            tag_parts = [f"{mdata['params']} params", f"{mdata['size_mb']} MB", hw]
            if rec:
                tag_parts.append(rec)
            if langs:
                tag_parts.append("/".join(l.upper() for l in langs))
            if mdata.get("streaming"):
                tag_parts.append("Streaming")
            tag_label = Gtk.Label()
            tag_label.set_markup(f"<small>{' · '.join(tag_parts)}</small>")
            tag_label.set_halign(Gtk.Align.START)
            tag_label.get_style_context().add_class("dim-label")
            vbox.pack_start(tag_label, False, False, 0)

            hbox.pack_start(vbox, True, True, 0)

            # Buttons column
            btn_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            btn_box.set_valign(Gtk.Align.CENTER)

            downloaded = _is_model_downloaded(mid, self._profiles)
            is_active = (mid == active)

            if downloaded:
                if is_active:
                    active_label = Gtk.Label(label="Active")
                    active_label.get_style_context().add_class("dim-label")
                    btn_box.pack_start(active_label, False, False, 0)
                else:
                    use_btn = Gtk.Button(label="Use")
                    use_btn.connect("clicked", self._on_use_model, mid)
                    btn_box.pack_start(use_btn, False, False, 0)
            else:
                dl_btn = Gtk.Button()
                dl_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
                dl_hbox.pack_start(
                    Gtk.Image.new_from_icon_name("folder-download-symbolic", Gtk.IconSize.BUTTON),
                    False, False, 0,
                )
                dl_hbox.pack_start(Gtk.Label(label="Download"), False, False, 0)
                dl_btn.add(dl_hbox)
                dl_btn.connect("clicked", self._on_download_model, mid, dl_btn)
                btn_box.pack_start(dl_btn, False, False, 0)

            hbox.pack_end(btn_box, False, False, 0)
            row.add(hbox)
            self._model_list.add(row)

        self._model_list.show_all()

    def _on_use_model(self, _btn, model_id):
        self._config.model_profile = model_id
        self._config.save()
        if self._on_save:
            self._on_save(self._config)
        self._populate_models()

    def _update_dl_progress(self, msg, fraction):
        self._dl_progress.set_text(msg)
        if fraction >= 0:
            self._dl_progress.set_fraction(min(fraction, 1.0))
        else:
            self._dl_progress.pulse()

    def _on_download_model(self, _btn, model_id, btn_widget):
        btn_widget.set_sensitive(False)
        self._dl_progress.show()
        self._dl_error.hide()

        def on_progress(msg, fraction):
            self._update_dl_progress(msg, fraction)

        def on_done(success, err):
            self._dl_progress.hide()
            if success:
                self._populate_models()
            else:
                btn_widget.set_sensitive(True)
                self._dl_error.set_markup(f"<span color='red'>Download failed: {GLib.markup_escape_text(err)}</span>")
                self._dl_error.show()
                print(f"Download error: {err}", file=sys.stderr)

        _download_models([model_id], self._profiles_data, on_progress, on_done)

    def _on_download_all(self, _btn):
        self._dl_all_btn.set_sensitive(False)
        self._dl_progress.show()
        self._dl_error.hide()

        def on_progress(msg, fraction):
            self._update_dl_progress(msg, fraction)

        def on_done(success, err):
            self._dl_progress.hide()
            if success:
                self._dl_all_label.set_text("All models downloaded")
                self._populate_models()
            else:
                self._dl_all_btn.set_sensitive(True)
                self._dl_error.set_markup(f"<span color='red'>Download failed: {GLib.markup_escape_text(err)}</span>")
                self._dl_error.show()
                print(f"Download error: {err}", file=sys.stderr)

        _download_models(list(self._profiles), self._profiles_data,
                         on_progress, on_done)

    # --- Hotkeys tab ---

    def _build_hotkeys_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(16)
        box.set_margin_end(16)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        # Mode
        self._mode_toggle = Gtk.RadioButton.new_with_label(
            None, "Toggle (one key starts and stops)")
        self._mode_startstop = Gtk.RadioButton.new_with_label_from_widget(
            self._mode_toggle, "Start/Stop (separate keys)")
        if self._config.hotkey_mode == "start_stop":
            self._mode_startstop.set_active(True)
        box.pack_start(self._mode_toggle, False, False, 0)
        box.pack_start(self._mode_startstop, False, False, 4)

        # Bindings
        hint = Gtk.Label(label="Click a button, then press your desired key combo.")
        hint.set_halign(Gtk.Align.START)
        hint.set_margin_top(8)
        box.pack_start(hint, False, False, 0)

        grid = Gtk.Grid(column_spacing=12, row_spacing=8)
        grid.set_margin_top(4)

        grid.attach(Gtk.Label(label="Toggle:", halign=Gtk.Align.END), 0, 0, 1, 1)
        self._hk_toggle = HotkeyCaptureButton(self._config.hotkey_toggle)
        grid.attach(self._hk_toggle, 1, 0, 1, 1)

        grid.attach(Gtk.Label(label="Start:", halign=Gtk.Align.END), 0, 1, 1, 1)
        self._hk_start = HotkeyCaptureButton(self._config.hotkey_start)
        grid.attach(self._hk_start, 1, 1, 1, 1)

        grid.attach(Gtk.Label(label="Stop:", halign=Gtk.Align.END), 0, 2, 1, 1)
        self._hk_stop = HotkeyCaptureButton(self._config.hotkey_stop)
        grid.attach(self._hk_stop, 1, 2, 1, 1)

        grid.attach(Gtk.Label(label="Pause:", halign=Gtk.Align.END), 0, 3, 1, 1)
        self._hk_pause = HotkeyCaptureButton(self._config.hotkey_pause)
        grid.attach(self._hk_pause, 1, 3, 1, 1)

        box.pack_start(grid, False, False, 0)

        # Save
        save_btn = Gtk.Button(label="Save Hotkeys")
        save_btn.get_style_context().add_class("suggested-action")
        save_btn.connect("clicked", self._save_hotkeys)
        save_btn.set_margin_top(12)
        box.pack_start(save_btn, False, False, 0)

        return box

    def _save_hotkeys(self, _btn):
        self._config.hotkey_mode = "start_stop" if self._mode_startstop.get_active() else "toggle"
        self._config.hotkey_toggle = self._hk_toggle.binding
        self._config.hotkey_start = self._hk_start.binding
        self._config.hotkey_stop = self._hk_stop.binding
        self._config.hotkey_pause = self._hk_pause.binding
        self._config.save()
        if self._on_save:
            self._on_save(self._config)

    # --- General tab ---

    def _build_general_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(16)
        box.set_margin_end(16)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        # Microphone selector
        hbox_mic = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox_mic.pack_start(Gtk.Label(label="Microphone:"), False, False, 0)
        self._mic_combo = Gtk.ComboBoxText()
        self._mic_combo.append("", "System Default")
        devices = list_input_devices()
        for dev in devices:
            self._mic_combo.append(str(dev["index"]), dev["name"])
        self._mic_combo.set_active_id(self._config.audio_device or "")
        hbox_mic.pack_start(self._mic_combo, True, True, 0)
        box.pack_start(hbox_mic, False, False, 0)

        # Typing method
        hbox_typer = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox_typer.pack_start(Gtk.Label(label="Typing method:"), False, False, 0)
        self._typer_combo = Gtk.ComboBoxText()
        self._typer_combo.append("xdotool", "xdotool type (X11)")
        self._typer_combo.append("clipboard", "Clipboard paste (recommended)")
        self._typer_combo.append("wtype", "wtype (GNOME/Sway only)")
        self._typer_combo.append("ydotool", "ydotool (needs daemon+uinput)")
        self._typer_combo.set_active_id(self._config.typer)
        hbox_typer.pack_start(self._typer_combo, False, False, 0)
        box.pack_start(hbox_typer, False, False, 0)

        sep0 = Gtk.Separator()
        sep0.set_margin_top(4)
        box.pack_start(sep0, False, False, 0)

        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.pack_start(Gtk.Label(label="Beep volume:"), False, False, 0)
        self._vol_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, 0.05)
        self._vol_scale.set_value(self._config.beep_volume)
        hbox.pack_start(self._vol_scale, True, True, 0)
        box.pack_start(hbox, False, False, 0)

        hbox2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox2.pack_start(Gtk.Label(label="CPU threads:"), False, False, 0)
        self._threads_spin = Gtk.SpinButton.new_with_range(1, 16, 1)
        self._threads_spin.set_value(self._config.num_threads)
        hbox2.pack_start(self._threads_spin, False, False, 0)
        box.pack_start(hbox2, False, False, 0)

        # Language (applies to Canary model)
        hbox_lang = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox_lang.pack_start(Gtk.Label(label="Language:"), False, False, 0)
        self._lang_combo = Gtk.ComboBoxText()
        for code, label in LANG_LABELS.items():
            self._lang_combo.append(code, label)
        self._lang_combo.set_active_id(self._config.language)
        hbox_lang.pack_start(self._lang_combo, False, False, 0)
        lang_hint = Gtk.Label()
        lang_hint.set_markup("<small>Used by Canary model. Parakeet auto-detects.</small>")
        lang_hint.get_style_context().add_class("dim-label")
        hbox_lang.pack_start(lang_hint, False, False, 0)
        box.pack_start(hbox_lang, False, False, 0)

        # Pause length that ends a chunk and triggers transcription
        hbox_pause = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox_pause.pack_start(Gtk.Label(label="Transcribe after a pause of:"), False, False, 0)
        self._pause_spin = Gtk.SpinButton.new_with_range(0, 30, 0.25)
        self._pause_spin.set_value(self._config.pause_secs)
        hbox_pause.pack_start(self._pause_spin, False, False, 0)
        hbox_pause.pack_start(Gtk.Label(label="seconds"), False, False, 0)
        box.pack_start(hbox_pause, False, False, 0)
        pause_hint = Gtk.Label()
        pause_hint.set_markup(
            "<small>0 = never — everything you say is transcribed as one "
            "chunk when you stop recording.</small>")
        pause_hint.get_style_context().add_class("dim-label")
        pause_hint.set_halign(Gtk.Align.START)
        pause_hint.set_line_wrap(True)
        pause_hint.set_max_width_chars(60)
        box.pack_start(pause_hint, False, False, 0)

        # Max continuous-speech duration before a forced transcription cut
        hbox_chunk = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox_chunk.pack_start(Gtk.Label(label="Transcribe after:"), False, False, 0)
        self._chunk_spin = Gtk.SpinButton.new_with_range(0, 3600, 5)
        self._chunk_spin.set_value(self._config.max_speech_secs)
        hbox_chunk.pack_start(self._chunk_spin, False, False, 0)
        hbox_chunk.pack_start(Gtk.Label(label="seconds of nonstop speech"), False, False, 0)
        box.pack_start(hbox_chunk, False, False, 0)
        chunk_hint = Gtk.Label()
        chunk_hint.set_markup(
            "<small>Speaking without pausing is cut into chunks this long. "
            "Set 0 to never cut — text appears when you pause or stop recording.</small>")
        chunk_hint.get_style_context().add_class("dim-label")
        chunk_hint.set_halign(Gtk.Align.START)
        chunk_hint.set_line_wrap(True)
        chunk_hint.set_max_width_chars(60)
        box.pack_start(chunk_hint, False, False, 0)

        # Streaming options
        sep_stream = Gtk.Separator()
        sep_stream.set_margin_top(4)
        box.pack_start(sep_stream, False, False, 0)

        self._partial_overwrite_check = Gtk.CheckButton(
            label="Streaming partial-overwrite (type text as you speak)")
        self._partial_overwrite_check.set_active(self._config.partial_overwrite)
        self._partial_overwrite_check.set_tooltip_text(
            "When enabled, streaming models type partial results into the active window "
            "and revise them in place.  When disabled, text only appears on final endpoint.")
        box.pack_start(self._partial_overwrite_check, False, False, 4)

        self._filter_fillers_check = Gtk.CheckButton(
            label="Filter filler words (um, uh, ehm …)")
        self._filter_fillers_check.set_active(self._config.filter_fillers)
        box.pack_start(self._filter_fillers_check, False, False, 4)

        repl_label = Gtk.Label(label="Word replacements (one per line, spoken=typed):",
                               xalign=0)
        box.pack_start(repl_label, False, False, 0)
        self._repl_view = Gtk.TextView()
        self._repl_view.set_tooltip_text(
            "Whole-word, case-insensitive replacements applied to transcribed "
            "text.  Example line:  herder=herdr")
        self._repl_view.get_buffer().set_text(
            "\n".join(f"{k}={v}" for k, v in self._config.word_replacements.items()))
        repl_scroll = Gtk.ScrolledWindow()
        repl_scroll.set_min_content_height(70)
        repl_scroll.set_shadow_type(Gtk.ShadowType.IN)
        repl_scroll.add(self._repl_view)
        box.pack_start(repl_scroll, False, False, 4)

        # Night mode
        sep = Gtk.Separator()
        sep.set_margin_top(8)
        box.pack_start(sep, False, False, 0)

        self._night_check = Gtk.CheckButton(label="Night mode (suppress beeps)")
        self._night_check.set_active(self._config.night_mode)
        box.pack_start(self._night_check, False, False, 4)

        hbox3 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox3.pack_start(Gtk.Label(label="Quiet hours:"), False, False, 0)
        self._night_start_spin = Gtk.SpinButton.new_with_range(0, 23, 1)
        self._night_start_spin.set_value(self._config.night_start)
        hbox3.pack_start(self._night_start_spin, False, False, 0)
        hbox3.pack_start(Gtk.Label(label="to"), False, False, 0)
        self._night_end_spin = Gtk.SpinButton.new_with_range(0, 23, 1)
        self._night_end_spin.set_value(self._config.night_end)
        hbox3.pack_start(self._night_end_spin, False, False, 0)
        box.pack_start(hbox3, False, False, 0)

        save_btn = Gtk.Button(label="Save General")
        save_btn.get_style_context().add_class("suggested-action")
        save_btn.connect("clicked", self._save_general)
        save_btn.set_margin_top(12)
        box.pack_start(save_btn, False, False, 0)

        return box

    def _save_general(self, _btn):
        self._config.audio_device = self._mic_combo.get_active_id() or ""
        self._config.typer = self._typer_combo.get_active_id() or "wtype"
        self._config.beep_volume = self._vol_scale.get_value()
        self._config.num_threads = int(self._threads_spin.get_value())
        self._config.language = self._lang_combo.get_active_id() or "en"
        self._config.max_speech_secs = self._chunk_spin.get_value()
        self._config.pause_secs = self._pause_spin.get_value()
        self._config.partial_overwrite = self._partial_overwrite_check.get_active()
        self._config.filter_fillers = self._filter_fillers_check.get_active()
        buf = self._repl_view.get_buffer()
        raw = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        self._config.word_replacements = {
            k.strip(): v.strip()
            for k, _, v in (line.partition("=") for line in raw.splitlines())
            if k.strip() and v.strip()
        }
        self._config.night_mode = self._night_check.get_active()
        self._config.night_start = int(self._night_start_spin.get_value())
        self._config.night_end = int(self._night_end_spin.get_value())
        self._config.save()
        if self._on_save:
            self._on_save(self._config)

    # --- About tab ---

    def _build_about_tab(self):
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(16)
        box.set_margin_end(16)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        about_text = Gtk.Label()
        about_text.set_markup(
            f"<b>{APP_NAME}</b>\n\n"
            "On-device voice typing with punctuation.\n"
            "Powered by sherpa-onnx + NVIDIA NeMo models.\n\n"
            "<b>Model recommendations:</b>\n\n"
            "<b>Parakeet TDT 0.6B v3</b> (639 MB)\n"
            "Best overall accuracy. Ideal for desktops and workstations.\n"
            "Works on CPU at ~30x real-time. Even faster with GPU.\n"
            "Supports 25 European languages.\n\n"
            "<b>Canary 180M Flash</b> (198 MB)\n"
            "Lightweight model for laptops and low-RAM machines.\n"
            "Good accuracy for its size. Supports EN/ES/DE/FR.\n"
            "Only 198 MB download — ideal for travel.\n\n"
            "<b>Nemotron Streaming 0.6B</b> (631 MB)\n"
            "True real-time streaming — text appears as you speak\n"
            "with no pause needed. English only.\n"
            "Higher latency tradeoff: slightly less accurate on\n"
            "sentence boundaries vs. VAD-segmented models.\n\n"
            "<b>General tips:</b>\n"
            "• All models include punctuation and capitalization\n"
            "• Non-streaming models wait for a brief pause, then transcribe\n"
            "• Streaming model transcribes continuously but may revise text\n"
            "• More CPU threads = faster transcription (4-8 recommended)\n"
            "• Pause hotkey mutes mic without unloading model (fast resume)"
        )
        about_text.set_halign(Gtk.Align.START)
        about_text.set_valign(Gtk.Align.START)
        about_text.set_line_wrap(True)
        about_text.set_selectable(True)
        about_text.set_max_width_chars(60)

        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        sw.add(about_text)
        box.pack_start(sw, True, True, 0)

        return box


# ---------------------------------------------------------------------------
# Main window (undockable full-size UI)
# ---------------------------------------------------------------------------

class MainWindow(Gtk.Window):
    def __init__(self, controller: DictationController):
        super().__init__(title=APP_NAME)
        self._controller = controller
        self.tray_refresh = lambda: None  # set by main once the tray exists
        self.set_default_size(480, -1)
        self.set_resizable(False)
        self.set_icon_name("audio-input-microphone")
        self.connect("delete-event", self._on_delete)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        vbox.set_margin_start(16)
        vbox.set_margin_end(16)
        vbox.set_margin_top(16)
        vbox.set_margin_bottom(16)
        self.add(vbox)

        # --- Status ---
        self._status_label = Gtk.Label()
        self._status_label.set_markup("<span size='large'>Idle</span>")
        self._status_label.set_halign(Gtk.Align.CENTER)
        vbox.pack_start(self._status_label, False, False, 0)

        # --- Controls ---
        btn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        btn_box.set_halign(Gtk.Align.CENTER)

        self._toggle_btn = Gtk.Button()
        self._toggle_btn.get_style_context().add_class("suggested-action")
        self._toggle_btn.connect("clicked", self._on_toggle)
        btn_box.pack_start(self._toggle_btn, False, False, 0)

        self._pause_btn = Gtk.Button(label="Pause")
        self._pause_btn.set_sensitive(False)
        self._pause_btn.connect("clicked", lambda _: self._controller.toggle_pause())
        btn_box.pack_start(self._pause_btn, False, False, 0)

        vbox.pack_start(btn_box, False, False, 0)

        vbox.pack_start(Gtk.Separator(), False, False, 0)

        # --- Microphone selector ---
        mic_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        mic_box.pack_start(Gtk.Label(label="Microphone:"), False, False, 0)
        self._mic_combo = Gtk.ComboBoxText()
        self._mic_combo.append("", "System Default")
        for dev in list_input_devices():
            self._mic_combo.append(str(dev["index"]), dev["name"])
        self._mic_combo.set_active_id(self._controller.config.audio_device or "")
        self._mic_combo.connect("changed", self._on_mic_changed)
        mic_box.pack_start(self._mic_combo, True, True, 0)
        vbox.pack_start(mic_box, False, False, 0)

        # --- Model selector ---
        model_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        model_box.pack_start(Gtk.Label(label="Model:"), False, False, 0)
        self._model_combo = Gtk.ComboBoxText()
        for mid, mdata in self._controller.profiles.items():
            downloaded = _is_model_downloaded(mid, self._controller.profiles)
            label = mdata["name"]
            if not downloaded:
                label += " (not downloaded)"
            self._model_combo.append(mid, label)
        self._model_combo.set_active_id(self._controller.config.model_profile)
        self._model_combo.connect("changed", self._on_model_changed)
        model_box.pack_start(self._model_combo, True, True, 0)
        vbox.pack_start(model_box, False, False, 0)

        # --- Streaming toggle ---
        self._streaming_check = Gtk.CheckButton(label="Streaming mode (type as you speak)")
        self._streaming_check.set_tooltip_text(
            "Switch between real-time streaming (Nemotron) and "
            "VAD-segmented transcription (waits for pause, higher accuracy).")
        is_streaming = self._controller.profiles.get(
            self._controller.config.model_profile, {}).get("streaming", False)
        self._streaming_check.set_active(is_streaming)
        # Remember the non-streaming model so we can restore it
        if is_streaming:
            self._non_streaming_model = "desktop"
        else:
            self._non_streaming_model = self._controller.config.model_profile
        self._streaming_check.connect("toggled", self._on_streaming_toggled)
        vbox.pack_start(self._streaming_check, False, False, 0)

        self._update_controls()

    def _on_delete(self, _win, _event):
        self.hide()
        return True

    def _on_toggle(self, _btn):
        self._controller.toggle()
        self._update_controls()

    def _on_mic_changed(self, combo):
        dev_id = combo.get_active_id() or ""
        cfg = self._controller.config
        cfg.audio_device = dev_id
        cfg.save()

    def _on_model_changed(self, combo):
        model_id = combo.get_active_id()
        if not model_id or model_id == self._controller.config.model_profile:
            return
        if not _is_model_downloaded(model_id, self._controller.profiles):
            self._model_combo.set_active_id(self._controller.config.model_profile)
            return
        self._controller.switch_model(model_id)
        # Keep streaming checkbox in sync
        is_streaming = self._controller.profiles.get(model_id, {}).get("streaming", False)
        self._streaming_check.handler_block_by_func(self._on_streaming_toggled)
        self._streaming_check.set_active(is_streaming)
        self._streaming_check.handler_unblock_by_func(self._on_streaming_toggled)
        if not is_streaming:
            self._non_streaming_model = model_id
        self.tray_refresh()

    def _on_streaming_toggled(self, check):
        if check.get_active():
            # Remember current non-streaming model, switch to streaming
            cur = self._controller.config.model_profile
            cur_profile = self._controller.profiles.get(cur, {})
            if not cur_profile.get("streaming", False):
                self._non_streaming_model = cur
            target = "streaming"
        else:
            # Restore previous non-streaming model
            target = getattr(self, "_non_streaming_model", "desktop")

        if not _is_model_downloaded(target, self._controller.profiles):
            # Can't switch — revert checkbox
            check.handler_block_by_func(self._on_streaming_toggled)
            check.set_active(not check.get_active())
            check.handler_unblock_by_func(self._on_streaming_toggled)
            return

        self._controller.switch_model(target)
        # Sync the model combo
        self._model_combo.set_active_id(target)
        self.tray_refresh()

    def _update_controls(self):
        running = self._controller.is_running
        paused = self._controller.is_paused
        cfg = self._controller.config

        if running:
            if paused:
                self._toggle_btn.set_label("Resume")
                self._status_label.set_markup("<span size='large'>Paused</span>")
            else:
                self._toggle_btn.set_label("Stop")
                self._status_label.set_markup("<span size='large'>Listening...</span>")
            self._pause_btn.set_sensitive(True)
        else:
            self._toggle_btn.set_label("Start Dictation")
            self._status_label.set_markup("<span size='large'>Idle</span>")
            self._pause_btn.set_sensitive(False)

        # Sync model combo and streaming checkbox if changed externally
        if self._model_combo.get_active_id() != cfg.model_profile:
            self._model_combo.set_active_id(cfg.model_profile)
        is_streaming = self._controller.profiles.get(
            cfg.model_profile, {}).get("streaming", False)
        if self._streaming_check.get_active() != is_streaming:
            self._streaming_check.handler_block_by_func(self._on_streaming_toggled)
            self._streaming_check.set_active(is_streaming)
            self._streaming_check.handler_unblock_by_func(self._on_streaming_toggled)

    def on_status_update(self, text: str):
        self._update_controls()
        if text and text not in ("Listening...", "Ready", "Resumed", "Paused"):
            display = text[:60] + "\u2026" if len(text) > 60 else text
            self._status_label.set_markup(f"<span size='large'>\u25b6 {GLib.markup_escape_text(display)}</span>")


# ---------------------------------------------------------------------------
# System tray
# ---------------------------------------------------------------------------

class TrayIcon:
    def __init__(self, controller: DictationController, hotkey_mgr: HotkeyManager,
                 main_window: MainWindow):
        self._controller = controller
        self._hotkey_mgr = hotkey_mgr
        self._main_window = main_window

        self._indicator = AyatanaAppIndicator3.Indicator.new(
            APP_ID,
            "audio-input-microphone-muted",
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        self._indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        self._indicator.set_title(APP_NAME)

        self._build_menu()
        controller.set_status_callback(self._on_status_update)

    def _build_menu(self):
        menu = Gtk.Menu()
        cfg = self._controller.config

        show_window_item = Gtk.MenuItem(label="Show Window")
        show_window_item.connect("activate", lambda _: (self._main_window.show_all(), self._main_window.present()))
        menu.append(show_window_item)

        menu.append(Gtk.SeparatorMenuItem())

        self._toggle_item = Gtk.MenuItem(label=f"Start Dictation ({cfg.hotkey_toggle})")
        self._toggle_item.connect("activate", self._on_toggle)
        menu.append(self._toggle_item)

        self._pause_item = Gtk.MenuItem(label=f"Pause ({cfg.hotkey_pause})")
        self._pause_item.connect("activate", lambda _: self._controller.toggle_pause())
        self._pause_item.set_sensitive(False)
        menu.append(self._pause_item)

        self._status_item = Gtk.MenuItem(label="Idle")
        self._status_item.set_sensitive(False)
        menu.append(self._status_item)

        menu.append(Gtk.SeparatorMenuItem())

        # Model switcher submenu
        model_menu_item = Gtk.MenuItem(label="Model")
        model_submenu = Gtk.Menu()
        active_profile = cfg.model_profile
        for mid, mdata in self._controller.profiles.items():
            downloaded = _is_model_downloaded(mid, self._controller.profiles)
            label = mdata["name"]
            if mid == active_profile:
                label = f"\u2713 {label}"
            elif not downloaded:
                label = f"  {label} (not downloaded)"
            else:
                label = f"  {label}"
            item = Gtk.MenuItem(label=label)
            if downloaded and mid != active_profile:
                item.connect("activate", self._on_switch_model, mid)
            else:
                item.set_sensitive(False)
            model_submenu.append(item)
        model_menu_item.set_submenu(model_submenu)
        menu.append(model_menu_item)

        menu.append(Gtk.SeparatorMenuItem())

        settings_item = Gtk.MenuItem(label="Settings")
        settings_item.connect("activate", self._on_settings)
        menu.append(settings_item)

        menu.append(Gtk.SeparatorMenuItem())

        quit_item = Gtk.MenuItem(label="Quit")
        quit_item.connect("activate", self._on_quit)
        menu.append(quit_item)

        menu.show_all()
        self._indicator.set_menu(menu)

    def refresh(self):
        """Rebuild the menu and re-sync labels/icon after a model change."""
        self._build_menu()
        self.update_ui()

    def _on_switch_model(self, _item, model_id):
        self._controller.switch_model(model_id)
        self.refresh()

    def _on_toggle(self, _item=None):
        self._controller.toggle()
        self.update_ui()

    def update_ui(self):
        running = self._controller.is_running
        paused = self._controller.is_paused
        cfg = self._controller.config

        if running:
            if paused:
                self._toggle_item.set_label("Resume Dictation")
                self._indicator.set_icon_full("audio-input-microphone-muted", "Paused")
            else:
                key = cfg.hotkey_toggle if cfg.hotkey_mode == "toggle" else cfg.hotkey_stop
                self._toggle_item.set_label(f"Stop Dictation ({key})")
                self._indicator.set_icon_full("audio-input-microphone", "Listening")
            self._pause_item.set_sensitive(True)
        else:
            key = cfg.hotkey_toggle if cfg.hotkey_mode == "toggle" else cfg.hotkey_start
            self._toggle_item.set_label(f"Start Dictation ({key})")
            self._indicator.set_icon_full("audio-input-microphone-muted", "Idle")
            self._pause_item.set_sensitive(False)
            self._status_item.set_label("Idle")

    def _on_status_update(self, text: str):
        self.update_ui()
        self._main_window.on_status_update(text)
        if text:
            display = text[:60] + "\u2026" if len(text) > 60 else text
            self._status_item.set_label(f"\u25b6 {display}")
        else:
            if self._controller.is_running:
                self._status_item.set_label("Ready")
            else:
                self._status_item.set_label("Idle")

    def _on_settings(self, _item):
        SettingsDialog(
            self._controller.config,
            self._controller.profiles_data,
            on_save=self._apply_settings,
        )

    def _apply_settings(self, new_config: AppConfig):
        self._controller.apply_config(new_config)
        self._hotkey_mgr.rebuild(new_config)
        self.refresh()

    def _on_quit(self, _item):
        self._controller.stop()
        Gtk.main_quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    _migrate_legacy_models()
    config = AppConfig.load()

    # Ensure typer is a valid method
    if config.typer not in ("xdotool", "clipboard", "wtype", "ydotool"):
        config.typer = "clipboard"

    controller = DictationController(config)

    # tray referenced in hotkey lambdas — assigned after creation
    tray = None

    hotkey_mgr = HotkeyManager(
        config,
        on_toggle=lambda: (controller.toggle(), tray and tray.update_ui()),
        on_start=lambda: (controller.start(), tray and tray.update_ui()),
        on_stop=lambda: (controller.stop(), tray and tray.update_ui()),
        on_pause=lambda: controller.toggle_pause(),
    )

    main_window = MainWindow(controller)

    tray = TrayIcon(controller, hotkey_mgr, main_window)
    main_window.tray_refresh = tray.refresh
    hotkey_mgr.start()

    signal.signal(signal.SIGINT, lambda *_: (controller.stop(), Gtk.main_quit()))

    # First-run: show welcome dialog if no models are downloaded
    profiles_data = controller.profiles_data
    if not _any_model_downloaded(profiles_data["profiles"]):
        def _on_model_ready(new_config):
            controller.apply_config(new_config)
            hotkey_mgr.rebuild(new_config)
            tray.refresh()
        WelcomeDialog(profiles_data, config, _on_model_ready)

    profile_name = controller.profiles.get(
        config.model_profile, {}
    ).get("name", config.model_profile)
    if config.hotkey_mode == "toggle":
        mode_desc = f"Toggle: {config.hotkey_toggle}"
    else:
        mode_desc = f"Start: {config.hotkey_start}, Stop: {config.hotkey_stop}"
    print(f"{APP_NAME} running. {mode_desc}. Pause: {config.hotkey_pause}")
    print(f"Model: {profile_name} | Typer: {config.typer} | Threads: {config.num_threads}")

    Gtk.main()
    hotkey_mgr.stop()


if __name__ == "__main__":
    main()
