#!/usr/bin/env python3
"""Parakeet Dictation — On-device voice typing with punctuation via sherpa-onnx.

Supports multiple ASR model profiles (Parakeet, Canary, Nemotron) with
configurable hotkeys and VAD-segmented or true streaming transcription.
"""

import json
import os
import signal
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
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


def _migrate_legacy_models():
    """Move models from APP_DIR/models to the XDG data directory if needed."""
    import shutil

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


_migrate_legacy_models()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    # Model
    model_profile: str = "desktop"
    num_threads: int = min(os.cpu_count() or 4, 8)
    vad_threshold: float = 0.5

    # Audio
    beep_volume: float = 0.5
    audio_device: str = ""  # Empty = system default; otherwise device name or index

    # Typing method: "wtype" (recommended), "ydotool" (needs daemon), "clipboard" (wl-copy+wtype paste)
    typer: str = "wtype"

    # Hotkey mode: "toggle" (one key) or "start_stop" (separate keys)
    hotkey_mode: str = "toggle"

    # Night mode — suppress beeps between these hours (24h format)
    night_mode: bool = True
    night_start: int = 22  # 10 PM
    night_end: int = 9     # 9 AM

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
            except Exception:
                pass
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


# Global config ref for beep functions (set in main)
_active_config: "AppConfig | None" = None


def play_beep_start(volume: float = 0.5):
    """Rising tone — dictation started."""
    if _active_config and _is_night_mode(_active_config):
        return
    sd.play(_generate_tone(880, 0.15, volume), samplerate=SAMPLE_RATE)


def play_beep_stop(volume: float = 0.5):
    """Falling tone — dictation stopped."""
    if _active_config and _is_night_mode(_active_config):
        return
    sd.play(_generate_tone(440, 0.15, volume), samplerate=SAMPLE_RATE)


def play_beep_pause(volume: float = 0.5):
    """Double short beep — paused/resumed."""
    if _active_config and _is_night_mode(_active_config):
        return
    t1 = _generate_tone(660, 0.07, volume)
    gap = np.zeros(int(SAMPLE_RATE * 0.05), dtype=np.float32)
    t2 = _generate_tone(660, 0.07, volume)
    sd.play(np.concatenate([t1, gap, t2]), samplerate=SAMPLE_RATE)


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
# Text typer
# ---------------------------------------------------------------------------

class TextTyper:
    def __init__(self, method: str = "wtype"):
        self._method = method

    def type_text(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            if self._method == "wtype":
                subprocess.run(["wtype", "--", text + " "], timeout=5)
            elif self._method == "ydotool":
                subprocess.run(["ydotool", "type", "--", text + " "], timeout=5)
            elif self._method == "clipboard":
                subprocess.run(["wl-copy", "--", text + " "], timeout=5)
                subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], timeout=5)
            else:
                subprocess.run(["wtype", "--", text + " "], timeout=5)
        except FileNotFoundError:
            print(f"ERROR: {self._method} not found. Install wtype: sudo apt install wtype", file=sys.stderr)
        except subprocess.TimeoutExpired:
            pass


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
        self._max_speech_samples = int(max_speech_duration * sample_rate)

        self._vad = TenVad(hop_size=self._hop_size, threshold=threshold)

        # Internal state
        self._buffer: list[float] = []  # float32 samples for ASR
        self._int16_remainder = np.array([], dtype=np.int16)  # leftover for VAD
        self._in_speech = False
        self._speech_samples = 0
        self._silence_samples = 0
        self._segments: list[_SpeechSegment] = []
        self._is_speech = False

    def accept_waveform(self, samples: list[float]):
        """Feed float32 audio samples (matching sounddevice output)."""
        # Convert to int16 for TEN VAD
        arr = np.array(samples, dtype=np.float32)
        int16_data = (arr * 32767).astype(np.int16)

        # Prepend any leftover from previous call
        if len(self._int16_remainder) > 0:
            int16_data = np.concatenate([self._int16_remainder, int16_data])

        # Process in hop_size chunks
        i = 0
        while i + self._hop_size <= len(int16_data):
            chunk = int16_data[i:i + self._hop_size]
            prob, _flag = self._vad.process(chunk)
            is_speech = prob >= self._threshold

            float_chunk = samples[i:i + self._hop_size] if i + self._hop_size <= len(samples) else arr[i:i + self._hop_size].tolist()

            if is_speech:
                self._is_speech = True
                self._silence_samples = 0
                if not self._in_speech:
                    self._in_speech = True
                    self._speech_samples = 0
                self._buffer.extend(float_chunk)
                self._speech_samples += self._hop_size

                # Force segment if max duration reached
                if self._speech_samples >= self._max_speech_samples:
                    self._emit_segment()
            else:
                if self._in_speech:
                    self._buffer.extend(float_chunk)
                    self._silence_samples += self._hop_size
                    if self._silence_samples >= self._min_silence_samples:
                        self._emit_segment()
                else:
                    self._is_speech = False

            i += self._hop_size

        # Save leftover
        self._int16_remainder = int16_data[i:]

    def _emit_segment(self):
        """Finalize current speech buffer into a segment."""
        if len(self._buffer) >= self._min_speech_samples:
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
    def __init__(self, config: AppConfig, profile: dict, on_text, on_partial, on_error):
        self._config = config
        self._profile = profile
        self._on_text = on_text
        self._on_partial = on_partial
        self._on_error = on_error
        self._running = False
        self._paused = False
        self._thread = None
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # set = NOT paused
        self._pause_event.set()

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
        return TenVadDetector(
            threshold=self._config.vad_threshold,
            min_silence_duration=0.25,
            min_speech_duration=0.25,
            max_speech_duration=30.0,
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
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        self._pause_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False
        self._paused = False

    def pause(self):
        if not self._running:
            return
        if self._paused:
            self._paused = False
            self._pause_event.set()
            play_beep_pause(self._config.beep_volume)
            GLib.idle_add(self._on_partial, "Resumed")
        else:
            self._paused = True
            self._pause_event.clear()
            play_beep_pause(self._config.beep_volume)
            GLib.idle_add(self._on_partial, "Paused")

    def _run(self):
        try:
            self._ensure_models()
        except Exception as e:
            GLib.idle_add(self._on_error, str(e))
            return

        is_streaming = self._profile.get("streaming", False)
        self._running = True

        try:
            if is_streaming:
                self._run_streaming()
            else:
                self._run_offline()
        except Exception as e:
            GLib.idle_add(self._on_error, str(e))
        finally:
            self._running = False
            play_beep_stop(self._config.beep_volume)

    def _run_offline(self):
        recognizer = self._build_offline_recognizer()
        vad = self._build_vad()
        chunk_duration = 0.1
        samples_per_chunk = int(SAMPLE_RATE * chunk_duration)

        device = resolve_audio_device(self._config.audio_device)
        with sd.InputStream(
            device=device, channels=1, dtype="float32", samplerate=SAMPLE_RATE,
            blocksize=samples_per_chunk,
        ) as stream:
            play_beep_start(self._config.beep_volume)
            GLib.idle_add(self._on_partial, "")

            while not self._stop_event.is_set():
                self._pause_event.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                if self._paused:
                    continue

                audio, overflowed = stream.read(samples_per_chunk)
                if overflowed:
                    continue
                samples = audio.reshape(-1).tolist()
                vad.accept_waveform(samples)

                if vad.is_speech_detected():
                    GLib.idle_add(self._on_partial, "Listening...")

                while not vad.empty():
                    segment = vad.front
                    s = recognizer.create_stream()
                    s.accept_waveform(SAMPLE_RATE, segment.samples)
                    recognizer.decode_stream(s)
                    text = s.result.text.strip()
                    if text:
                        GLib.idle_add(self._on_text, text)
                    vad.pop()

            # Flush remaining
            vad.flush()
            while not vad.empty():
                segment = vad.front
                s = recognizer.create_stream()
                s.accept_waveform(SAMPLE_RATE, segment.samples)
                recognizer.decode_stream(s)
                text = s.result.text.strip()
                if text:
                    GLib.idle_add(self._on_text, text)
                vad.pop()

    def _run_streaming(self):
        recognizer = self._build_online_recognizer()
        stream = recognizer.create_stream()
        chunk_duration = 0.1
        samples_per_chunk = int(SAMPLE_RATE * chunk_duration)

        device = resolve_audio_device(self._config.audio_device)
        with sd.InputStream(
            device=device, channels=1, dtype="float32", samplerate=SAMPLE_RATE,
            blocksize=samples_per_chunk,
        ) as mic:
            play_beep_start(self._config.beep_volume)
            GLib.idle_add(self._on_partial, "")

            while not self._stop_event.is_set():
                self._pause_event.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                if self._paused:
                    continue

                audio, overflowed = mic.read(samples_per_chunk)
                if overflowed:
                    continue
                samples = audio.reshape(-1).tolist()
                stream.accept_waveform(SAMPLE_RATE, samples)

                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)

                partial = recognizer.get_result(stream).strip()
                if partial:
                    GLib.idle_add(self._on_partial, partial)

                if recognizer.is_endpoint(stream):
                    text = recognizer.get_result(stream).strip()
                    if text:
                        GLib.idle_add(self._on_text, text)
                    recognizer.reset(stream)


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
        self._transcript_callback = None
        self._rebuild_engine()

    def _rebuild_engine(self):
        profile = self._profiles_data["profiles"].get(self._config.model_profile)
        if not profile:
            profile = self._profiles_data["profiles"]["desktop"]
        self._engine = ASREngine(
            self._config, profile,
            on_text=self._on_final_text,
            on_partial=self._on_partial,
            on_error=self._on_error,
        )

    def set_status_callback(self, cb):
        self._status_callback = cb

    def set_transcript_callback(self, cb):
        self._transcript_callback = cb

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
        if self._status_callback:
            self._status_callback("")

    def toggle(self):
        if self._engine.is_running:
            self.stop()
        else:
            self.start()

    def pause(self):
        self._engine.pause()

    def apply_config(self, new_config: AppConfig):
        was_running = self._engine.is_running
        if was_running:
            self._engine.stop()
        old_profile = self._config.model_profile
        self._config = new_config
        self._config.save()
        self._typer = TextTyper(new_config.typer)
        if new_config.model_profile != old_profile or was_running:
            self._rebuild_engine()

    def _on_final_text(self, text: str):
        self._typer.type_text(text)
        if self._transcript_callback:
            self._transcript_callback(text)
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
        self._capturing = False
        self.get_toplevel().disconnect(self._key_handler)

        parts = []
        if event.state & Gdk.ModifierType.CONTROL_MASK:
            parts.append("<ctrl>")
        if event.state & Gdk.ModifierType.MOD1_MASK:
            parts.append("<alt>")
        if event.state & Gdk.ModifierType.SHIFT_MASK:
            parts.append("<shift>")

        keyname = Gdk.keyval_name(event.keyval).lower()
        if keyname in ("control_l", "control_r", "alt_l", "alt_r",
                       "shift_l", "shift_r", "super_l", "super_r",
                       "meta_l", "meta_r"):
            self.set_label(self._display(self._binding))
            return True

        parts.append(keyname)
        self._binding = "+".join(parts)
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


def _ensure_vad(profiles_data: dict, on_progress_bytes=None):
    """Download the VAD model if not present."""
    vad = profiles_data.get("vad")
    if not vad:
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / vad["filename"]
    if dest.exists() and dest.stat().st_size > 0:
        return
    _download_file(vad["url"], dest, on_progress_bytes)


def _download_model(model_id: str, profiles_data: dict, on_progress, on_done):
    """Download a model in a background thread."""

    def _worker():
        try:
            profile = profiles_data["profiles"][model_id]
            model_dir = MODELS_DIR / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download VAD first
            def _vad_progress(done, total):
                if total:
                    mb = done / 1024 / 1024
                    total_mb = total / 1024 / 1024
                    GLib.idle_add(on_progress, f"VAD: {mb:.0f}/{total_mb:.0f} MB", -1.0)
            _ensure_vad(profiles_data, _vad_progress)

            # Download model files
            files = profile["files"]
            total_files = len(files)
            for i, (key, info) in enumerate(files.items(), 1):
                dest = model_dir / info["filename"]
                if dest.exists() and dest.stat().st_size > 0:
                    continue

                def _file_progress(done, total, fname=info["filename"], idx=i):
                    if total:
                        frac = done / total
                        mb = done / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        GLib.idle_add(
                            on_progress,
                            f"{fname} ({idx}/{total_files}): {mb:.0f}/{total_mb:.0f} MB",
                            frac,
                        )
                    else:
                        mb = done / 1024 / 1024
                        GLib.idle_add(on_progress, f"{fname}: {mb:.0f} MB", -1.0)

                _download_file(info["url"], dest, _file_progress)

            GLib.idle_add(on_done, True, "")
        except Exception as e:
            GLib.idle_add(on_done, False, str(e))

    threading.Thread(target=_worker, daemon=True).start()


def _download_all_models(profiles_data: dict, on_progress, on_done):
    """Download all models sequentially in a background thread."""

    def _worker():
        try:
            # Download VAD first
            def _vad_progress(done, total):
                if total:
                    mb = done / 1024 / 1024
                    total_mb = total / 1024 / 1024
                    GLib.idle_add(on_progress, f"VAD: {mb:.0f}/{total_mb:.0f} MB", -1.0)
            _ensure_vad(profiles_data, _vad_progress)

            profiles = profiles_data["profiles"]
            for mid, profile in profiles.items():
                model_dir = MODELS_DIR / mid
                model_dir.mkdir(parents=True, exist_ok=True)
                files = profile["files"]
                total_files = len(files)
                for i, (key, info) in enumerate(files.items(), 1):
                    dest = model_dir / info["filename"]
                    if dest.exists() and dest.stat().st_size > 0:
                        continue
                    short_name = profile["name"][:18]

                    def _file_progress(done, total, sn=short_name, fname=info["filename"], idx=i, tf=total_files):
                        if total:
                            frac = done / total
                            mb = done / 1024 / 1024
                            total_mb = total / 1024 / 1024
                            GLib.idle_add(on_progress, f"{sn}: {fname} {mb:.0f}/{total_mb:.0f} MB", frac)
                        else:
                            mb = done / 1024 / 1024
                            GLib.idle_add(on_progress, f"{sn}: {fname} {mb:.0f} MB", -1.0)

                    _download_file(info["url"], dest, _file_progress)
            GLib.idle_add(on_done, True, "")
        except Exception as e:
            GLib.idle_add(on_done, False, str(e))

    threading.Thread(target=_worker, daemon=True).start()


LANG_LABELS = {"en": "English", "es": "Spanish", "de": "German", "fr": "French"}


class WelcomeDialog(Gtk.Dialog):
    """First-run dialog shown when no models are downloaded yet."""

    def __init__(self, profiles_data: dict, config: AppConfig, on_model_ready):
        super().__init__(title=f"Welcome to {APP_NAME}", flags=0)
        self._profiles_data = profiles_data
        self._profiles = profiles_data["profiles"]
        self._config = config
        self._on_model_ready = on_model_ready
        self.set_default_size(520, 480)
        self.set_deletable(False)

        box = self.get_content_area()
        box.set_spacing(12)
        box.set_margin_start(20)
        box.set_margin_end(20)
        box.set_margin_top(16)
        box.set_margin_bottom(16)

        # Welcome header
        header = Gtk.Label()
        header.set_markup(
            f"<span size='x-large' weight='bold'>Welcome to {APP_NAME}</span>"
        )
        header.set_halign(Gtk.Align.START)
        box.pack_start(header, False, False, 0)

        subtitle = Gtk.Label()
        subtitle.set_markup(
            "To get started, download a speech recognition model.\n"
            "Choose the one that best fits your hardware:"
        )
        subtitle.set_halign(Gtk.Align.START)
        subtitle.set_line_wrap(True)
        box.pack_start(subtitle, False, False, 0)

        # Model cards
        for mid, mdata in self._profiles.items():
            frame = Gtk.Frame()
            frame.set_shadow_type(Gtk.ShadowType.ETCHED_IN)

            fbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
            fbox.set_margin_start(12)
            fbox.set_margin_end(12)
            fbox.set_margin_top(10)
            fbox.set_margin_bottom(10)

            # Info
            vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
            name_label = Gtk.Label()
            name_label.set_markup(f"<b>{mdata['name']}</b>")
            name_label.set_halign(Gtk.Align.START)
            vbox.pack_start(name_label, False, False, 0)

            # Recommendation tag
            rec = mdata.get("recommended_for", "")
            hw = mdata.get("hardware_label", "CPU")
            langs = mdata.get("languages")
            tag_parts = []
            if rec:
                tag_parts.append(rec)
            tag_parts.append(f"{hw}")
            tag_parts.append(f"{mdata['size_mb']} MB")
            if langs:
                tag_parts.append("/".join(l.upper() for l in langs))
            if mdata.get("streaming"):
                tag_parts.append("Streaming")
            tag_label = Gtk.Label()
            tag_label.set_markup(
                f"<small>{' · '.join(tag_parts)}</small>"
            )
            tag_label.set_halign(Gtk.Align.START)
            tag_label.get_style_context().add_class("dim-label")
            vbox.pack_start(tag_label, False, False, 0)

            desc_label = Gtk.Label(label=mdata.get("description", ""))
            desc_label.set_halign(Gtk.Align.START)
            desc_label.set_line_wrap(True)
            desc_label.set_max_width_chars(45)
            desc_label.get_style_context().add_class("dim-label")
            vbox.pack_start(desc_label, False, False, 0)

            fbox.pack_start(vbox, True, True, 0)

            # Download button with icon
            dl_btn = Gtk.Button()
            btn_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
            btn_hbox.pack_start(
                Gtk.Image.new_from_icon_name("folder-download-symbolic", Gtk.IconSize.BUTTON),
                False, False, 0,
            )
            btn_hbox.pack_start(Gtk.Label(label="Download"), False, False, 0)
            dl_btn.add(btn_hbox)
            dl_btn.get_style_context().add_class("suggested-action")
            dl_btn.set_valign(Gtk.Align.CENTER)
            dl_btn.connect("clicked", self._on_download, mid, dl_btn)
            fbox.pack_end(dl_btn, False, False, 0)

            frame.add(fbox)
            box.pack_start(frame, False, False, 0)

        # Download All button
        total_mb = sum(m["size_mb"] for m in self._profiles.values())
        dl_all_btn = Gtk.Button()
        dl_all_hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        dl_all_hbox.set_halign(Gtk.Align.CENTER)
        dl_all_hbox.pack_start(
            Gtk.Image.new_from_icon_name("folder-download-symbolic", Gtk.IconSize.BUTTON),
            False, False, 0,
        )
        dl_all_hbox.pack_start(
            Gtk.Label(label=f"Download All Models ({total_mb} MB)"), False, False, 0,
        )
        dl_all_btn.add(dl_all_hbox)
        dl_all_btn.set_margin_top(4)
        dl_all_btn.connect("clicked", self._on_download_all)
        box.pack_start(dl_all_btn, False, False, 0)

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
                self._config.model_profile = "laptop"
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

        _download_all_models(self._profiles_data, on_progress, on_done)

    def _on_download(self, _btn, model_id, btn_widget):
        btn_widget.set_sensitive(False)
        self._progress_bar.show()
        self._error_label.hide()

        def on_progress(msg, fraction):
            self._update_progress(msg, fraction)

        def on_done(success, err):
            if success:
                self._config.model_profile = model_id
                self._config.save()
                self.destroy()
                if self._on_model_ready:
                    self._on_model_ready(self._config)
            else:
                btn_widget.set_sensitive(True)
                self._progress_bar.hide()
                self._error_label.set_markup(f"<span color='red'>Download failed: {GLib.markup_escape_text(err)}</span>")
                self._error_label.show()
                print(f"Download error: {err}", file=sys.stderr)

        _download_model(model_id, self._profiles_data, on_progress, on_done)


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

        _download_model(model_id, self._profiles_data, on_progress, on_done)

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

        _download_all_models(self._profiles_data, on_progress, on_done)

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
        self._typer_combo.append("wtype", "wtype (recommended)")
        self._typer_combo.append("ydotool", "ydotool (needs daemon)")
        self._typer_combo.append("clipboard", "Clipboard paste (wl-copy)")
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
        global _active_config
        self._config.audio_device = self._mic_combo.get_active_id() or ""
        self._config.typer = self._typer_combo.get_active_id() or "wtype"
        self._config.beep_volume = self._vol_scale.get_value()
        self._config.num_threads = int(self._threads_spin.get_value())
        self._config.language = self._lang_combo.get_active_id() or "en"
        self._config.night_mode = self._night_check.get_active()
        self._config.night_start = int(self._night_start_spin.get_value())
        self._config.night_end = int(self._night_end_spin.get_value())
        _active_config = self._config
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
    def __init__(self, controller: DictationController, hotkey_mgr: HotkeyManager):
        super().__init__(title=APP_NAME)
        self._controller = controller
        self._hotkey_mgr = hotkey_mgr
        self.set_default_size(600, 450)
        self.set_icon_name("audio-input-microphone")
        self.connect("delete-event", self._on_delete)

        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.add(vbox)

        # --- Header bar with controls ---
        header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        header.set_margin_start(12)
        header.set_margin_end(12)
        header.set_margin_top(8)
        header.set_margin_bottom(8)

        self._toggle_btn = Gtk.Button()
        self._toggle_btn.get_style_context().add_class("suggested-action")
        self._toggle_btn.connect("clicked", self._on_toggle)
        header.pack_start(self._toggle_btn, False, False, 0)

        self._pause_btn = Gtk.Button(label="Pause")
        self._pause_btn.set_sensitive(False)
        self._pause_btn.connect("clicked", lambda _: self._controller.pause())
        header.pack_start(self._pause_btn, False, False, 0)

        self._status_label = Gtk.Label(label="Idle")
        self._status_label.set_halign(Gtk.Align.START)
        self._status_label.set_hexpand(True)
        self._status_label.set_margin_start(12)
        header.pack_start(self._status_label, True, True, 0)

        # Microphone selector in header
        self._mic_combo = Gtk.ComboBoxText()
        self._mic_combo.append("", "Default mic")
        for dev in list_input_devices():
            self._mic_combo.append(str(dev["index"]), dev["name"])
        self._mic_combo.set_active_id(self._controller.config.audio_device or "")
        self._mic_combo.connect("changed", self._on_mic_changed)
        header.pack_end(self._mic_combo, False, False, 0)
        header.pack_end(Gtk.Label(label="Mic:"), False, False, 0)

        vbox.pack_start(header, False, False, 0)
        vbox.pack_start(Gtk.Separator(), False, False, 0)

        # --- Transcript area ---
        sw = Gtk.ScrolledWindow()
        sw.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        sw.set_vexpand(True)

        self._textview = Gtk.TextView()
        self._textview.set_editable(False)
        self._textview.set_wrap_mode(Gtk.WrapMode.WORD_CHAR)
        self._textview.set_left_margin(12)
        self._textview.set_right_margin(12)
        self._textview.set_top_margin(8)
        self._textview.set_bottom_margin(8)
        self._textbuf = self._textview.get_buffer()
        sw.add(self._textview)
        vbox.pack_start(sw, True, True, 0)

        # --- Bottom bar ---
        bottom = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        bottom.set_margin_start(12)
        bottom.set_margin_end(12)
        bottom.set_margin_top(6)
        bottom.set_margin_bottom(6)

        clear_btn = Gtk.Button(label="Clear")
        clear_btn.connect("clicked", lambda _: self._textbuf.set_text(""))
        bottom.pack_start(clear_btn, False, False, 0)

        copy_btn = Gtk.Button(label="Copy All")
        copy_btn.connect("clicked", self._on_copy_all)
        bottom.pack_start(copy_btn, False, False, 0)

        self._model_label = Gtk.Label()
        self._model_label.set_halign(Gtk.Align.END)
        self._model_label.get_style_context().add_class("dim-label")
        bottom.pack_end(self._model_label, False, False, 0)

        vbox.pack_start(bottom, False, False, 0)

        self._update_controls()

    def _on_delete(self, _win, _event):
        self.hide()
        return True  # Don't destroy, just hide

    def _on_toggle(self, _btn):
        self._controller.toggle()
        self._update_controls()

    def _on_mic_changed(self, combo):
        dev_id = combo.get_active_id() or ""
        cfg = self._controller.config
        cfg.audio_device = dev_id
        cfg.save()

    def _on_copy_all(self, _btn):
        start, end = self._textbuf.get_bounds()
        text = self._textbuf.get_text(start, end, False)
        if text:
            subprocess.run(["wl-copy", "--", text], timeout=5)

    def _update_controls(self):
        running = self._controller.is_running
        paused = self._controller.is_paused
        cfg = self._controller.config

        if running:
            if paused:
                self._toggle_btn.set_label("Resume")
                self._status_label.set_text("Paused")
            else:
                self._toggle_btn.set_label("Stop")
                self._status_label.set_text("Listening...")
            self._pause_btn.set_sensitive(True)
        else:
            self._toggle_btn.set_label("Start Dictation")
            self._status_label.set_text("Idle")
            self._pause_btn.set_sensitive(False)

        profile_name = self._controller.profiles.get(
            cfg.model_profile, {}
        ).get("name", cfg.model_profile)
        self._model_label.set_markup(f"<small>{profile_name}</small>")

    def on_status_update(self, text: str):
        """Called by the controller status callback (on the GTK main thread)."""
        self._update_controls()
        if text and text not in ("Listening...", "Ready", "Resumed", "Paused"):
            display = text[:80] + "\u2026" if len(text) > 80 else text
            self._status_label.set_text(f"\u25b6 {display}")

    def append_transcript(self, text: str):
        """Append final transcribed text to the transcript view."""
        end_iter = self._textbuf.get_end_iter()
        existing = self._textbuf.get_char_count()
        if existing > 0:
            self._textbuf.insert(end_iter, " ")
            end_iter = self._textbuf.get_end_iter()
        self._textbuf.insert(end_iter, text)
        # Auto-scroll to bottom
        mark = self._textbuf.get_insert()
        self._textbuf.place_cursor(self._textbuf.get_end_iter())
        self._textview.scroll_mark_onscreen(mark)


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
        self._pause_item.connect("activate", lambda _: self._controller.pause())
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

    def _on_switch_model(self, _item, model_id):
        new_config = self._controller.config
        new_config.model_profile = model_id
        self._controller.apply_config(new_config)
        self._hotkey_mgr.rebuild(new_config)
        self._build_menu()
        self.update_ui()

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
        self._build_menu()
        self.update_ui()

    def _on_quit(self, _item):
        self._controller.stop()
        Gtk.main_quit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _active_config
    config = AppConfig.load()
    _active_config = config

    # Ensure typer is a valid Wayland method
    if config.typer not in ("wtype", "ydotool", "clipboard"):
        config.typer = "wtype"

    controller = DictationController(config)

    # tray referenced in hotkey lambdas — assigned after creation
    tray = None

    hotkey_mgr = HotkeyManager(
        config,
        on_toggle=lambda: (controller.toggle(), tray and tray.update_ui()),
        on_start=lambda: (controller.start(), tray and tray.update_ui()),
        on_stop=lambda: (controller.stop(), tray and tray.update_ui()),
        on_pause=lambda: controller.pause(),
    )

    main_window = MainWindow(controller, hotkey_mgr)
    controller.set_transcript_callback(main_window.append_transcript)

    tray = TrayIcon(controller, hotkey_mgr, main_window)
    hotkey_mgr.start()

    signal.signal(signal.SIGINT, lambda *_: (controller.stop(), Gtk.main_quit()))

    # First-run: show welcome dialog if no models are downloaded
    profiles_data = controller.profiles_data
    if not _any_model_downloaded(profiles_data["profiles"]):
        def _on_model_ready(new_config):
            controller.apply_config(new_config)
            hotkey_mgr.rebuild(new_config)
            tray._build_menu()
            tray.update_ui()
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
