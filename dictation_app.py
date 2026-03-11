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
MODELS_DIR = APP_DIR / "models"
MODELS_JSON = APP_DIR / "models.json"
SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class AppConfig:
    # Model
    model_profile: str = "desktop"
    num_threads: int = 4
    vad_threshold: float = 0.5

    # Audio
    beep_volume: float = 0.5

    # Typing method: "ydotool" for Wayland, "xdotool" for X11
    typer: str = "ydotool"

    # Hotkey mode: "toggle" (one key) or "start_stop" (separate keys)
    hotkey_mode: str = "toggle"

    # Night mode — suppress beeps between these hours (24h format)
    night_mode: bool = True
    night_start: int = 22  # 10 PM
    night_end: int = 9     # 9 AM

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
# Text typer
# ---------------------------------------------------------------------------

class TextTyper:
    def __init__(self, method: str = "ydotool"):
        self._method = method

    def type_text(self, text: str):
        text = text.strip()
        if not text:
            return
        try:
            if self._method == "ydotool":
                subprocess.run(["ydotool", "type", "--", text + " "], timeout=5)
            else:
                subprocess.run(
                    ["xdotool", "type", "--clearmodifiers", "--", text + " "],
                    timeout=5,
                )
        except FileNotFoundError:
            print(f"ERROR: {self._method} not found.", file=sys.stderr)
        except subprocess.TimeoutExpired:
            pass


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
        vad_path = MODELS_DIR / "silero_vad.onnx"
        if not vad_path.exists():
            missing.append("silero_vad.onnx")
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
        import sherpa_onnx
        vad_config = sherpa_onnx.VadModelConfig(
            sherpa_onnx.SileroVadModelConfig(
                model=str(MODELS_DIR / "silero_vad.onnx"),
                threshold=self._config.vad_threshold,
                min_silence_duration=0.25,
                min_speech_duration=0.25,
                max_speech_duration=30.0,
            ),
            sample_rate=SAMPLE_RATE,
        )
        return sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

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

        with sd.InputStream(
            channels=1, dtype="float32", samplerate=SAMPLE_RATE,
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

        with sd.InputStream(
            channels=1, dtype="float32", samplerate=SAMPLE_RATE,
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

def _is_model_downloaded(model_id: str, profiles: dict) -> bool:
    """Check if all files for a model are present on disk."""
    profile = profiles.get(model_id)
    if not profile:
        return False
    model_dir = MODELS_DIR / model_id
    for key, info in profile.get("files", {}).items():
        if not (model_dir / info["filename"]).exists():
            return False
    vad_path = MODELS_DIR / "silero_vad.onnx"
    return vad_path.exists()


def _download_model(model_id: str, profiles_data: dict, on_progress, on_done):
    """Download a model in a background thread."""
    import requests

    def _worker():
        try:
            profile = profiles_data["profiles"][model_id]
            model_dir = MODELS_DIR / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Download VAD if missing
            vad_path = MODELS_DIR / "silero_vad.onnx"
            if not vad_path.exists():
                GLib.idle_add(on_progress, "Downloading VAD model...")
                vad = profiles_data["vad"]
                resp = requests.get(vad["url"], stream=True, timeout=30)
                resp.raise_for_status()
                with open(vad_path, "wb") as f:
                    for chunk in resp.iter_content(1024 * 1024):
                        f.write(chunk)

            # Download model files
            files = profile["files"]
            total_files = len(files)
            for i, (key, info) in enumerate(files.items(), 1):
                dest = model_dir / info["filename"]
                if dest.exists() and dest.stat().st_size > 0:
                    continue
                GLib.idle_add(on_progress, f"Downloading {info['filename']} ({i}/{total_files})...")
                resp = requests.get(info["url"], stream=True, timeout=30)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(1024 * 1024):
                        f.write(chunk)

            GLib.idle_add(on_done, True, "")
        except Exception as e:
            GLib.idle_add(on_done, False, str(e))

    threading.Thread(target=_worker, daemon=True).start()


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

            tags = f"{mdata['params']} params · {mdata['size_mb']} MB"
            if mdata.get("streaming"):
                tags += " · streaming"
            tag_label = Gtk.Label(label=tags)
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
                dl_btn = Gtk.Button(label="Download")
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

    def _on_download_model(self, _btn, model_id, btn_widget):
        btn_widget.set_sensitive(False)
        btn_widget.set_label("...")

        def on_progress(msg):
            btn_widget.set_label(msg[:20])

        def on_done(success, err):
            if success:
                self._populate_models()
            else:
                btn_widget.set_label("Failed")
                btn_widget.set_sensitive(True)
                print(f"Download error: {err}", file=sys.stderr)

        _download_model(model_id, self._profiles_data, on_progress, on_done)

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
        self._config.beep_volume = self._vol_scale.get_value()
        self._config.num_threads = int(self._threads_spin.get_value())
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
# System tray
# ---------------------------------------------------------------------------

class TrayIcon:
    def __init__(self, controller: DictationController, hotkey_mgr: HotkeyManager):
        self._controller = controller
        self._hotkey_mgr = hotkey_mgr

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

        profile_name = self._controller.profiles.get(
            cfg.model_profile, {}
        ).get("name", cfg.model_profile)
        self._model_item = Gtk.MenuItem(label=f"Model: {profile_name}")
        self._model_item.set_sensitive(False)
        menu.append(self._model_item)

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

    # Auto-detect Wayland vs X11
    session = os.environ.get("XDG_SESSION_TYPE", "")
    if session == "wayland":
        config.typer = "ydotool"
    elif session == "x11":
        config.typer = "xdotool"

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

    tray = TrayIcon(controller, hotkey_mgr)
    hotkey_mgr.start()

    signal.signal(signal.SIGINT, lambda *_: (controller.stop(), Gtk.main_quit()))

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
