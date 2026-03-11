#!/usr/bin/env python3
"""Local Dictation — On-device voice typing with punctuation via sherpa-onnx.

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

APP_NAME = "Local Dictation"
APP_ID = "local-dictation"
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


def play_beep_start(volume: float = 0.5):
    """Rising tone — dictation started."""
    sd.play(_generate_tone(880, 0.15, volume), samplerate=SAMPLE_RATE)


def play_beep_stop(volume: float = 0.5):
    """Falling tone — dictation stopped."""
    sd.play(_generate_tone(440, 0.15, volume), samplerate=SAMPLE_RATE)


def play_beep_pause(volume: float = 0.5):
    """Double short beep — paused/resumed."""
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
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(Gtk.Dialog):
    def __init__(self, config: AppConfig, profiles: dict, on_save):
        super().__init__(title=f"{APP_NAME} — Settings", flags=0)
        self._config = config
        self._profiles = profiles
        self._on_save = on_save
        self.set_default_size(450, 520)

        box = self.get_content_area()
        box.set_spacing(8)
        box.set_margin_start(16)
        box.set_margin_end(16)
        box.set_margin_top(16)
        box.set_margin_bottom(16)

        # --- Model profile ---
        self._add_section(box, "Model")
        self._profile_combo = Gtk.ComboBoxText()
        for pid, pdata in profiles.items():
            label = f"{pdata['name']} ({pdata['params']}, {pdata['size_mb']} MB)"
            if pdata.get("streaming"):
                label += " [streaming]"
            self._profile_combo.append(pid, label)
        self._profile_combo.set_active_id(config.model_profile)
        box.pack_start(self._profile_combo, False, False, 0)

        # --- Hotkey mode ---
        self._add_section(box, "Hotkey Mode")
        self._mode_toggle = Gtk.RadioButton.new_with_label(
            None, "Toggle (one key starts and stops)")
        self._mode_startstop = Gtk.RadioButton.new_with_label_from_widget(
            self._mode_toggle, "Start/Stop (separate keys)")
        if config.hotkey_mode == "start_stop":
            self._mode_startstop.set_active(True)
        box.pack_start(self._mode_toggle, False, False, 0)
        box.pack_start(self._mode_startstop, False, False, 0)

        # --- Hotkey bindings ---
        self._add_section(box, "Key Bindings (click to rebind)")
        grid = Gtk.Grid(column_spacing=12, row_spacing=6)

        grid.attach(Gtk.Label(label="Toggle:", halign=Gtk.Align.END), 0, 0, 1, 1)
        self._hk_toggle = HotkeyCaptureButton(config.hotkey_toggle)
        grid.attach(self._hk_toggle, 1, 0, 1, 1)

        grid.attach(Gtk.Label(label="Start:", halign=Gtk.Align.END), 0, 1, 1, 1)
        self._hk_start = HotkeyCaptureButton(config.hotkey_start)
        grid.attach(self._hk_start, 1, 1, 1, 1)

        grid.attach(Gtk.Label(label="Stop:", halign=Gtk.Align.END), 0, 2, 1, 1)
        self._hk_stop = HotkeyCaptureButton(config.hotkey_stop)
        grid.attach(self._hk_stop, 1, 2, 1, 1)

        grid.attach(Gtk.Label(label="Pause:", halign=Gtk.Align.END), 0, 3, 1, 1)
        self._hk_pause = HotkeyCaptureButton(config.hotkey_pause)
        grid.attach(self._hk_pause, 1, 3, 1, 1)

        box.pack_start(grid, False, False, 0)

        # --- General ---
        self._add_section(box, "General")
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.pack_start(Gtk.Label(label="Beep volume:"), False, False, 0)
        self._vol_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, 0.05)
        self._vol_scale.set_value(config.beep_volume)
        hbox.pack_start(self._vol_scale, True, True, 0)
        box.pack_start(hbox, False, False, 0)

        hbox2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox2.pack_start(Gtk.Label(label="CPU threads:"), False, False, 0)
        self._threads_spin = Gtk.SpinButton.new_with_range(1, 16, 1)
        self._threads_spin.set_value(config.num_threads)
        hbox2.pack_start(self._threads_spin, False, False, 0)
        box.pack_start(hbox2, False, False, 0)

        # --- Save ---
        save_btn = Gtk.Button(label="Save & Apply")
        save_btn.get_style_context().add_class("suggested-action")
        save_btn.connect("clicked", self._save)
        box.pack_end(save_btn, False, False, 8)

        self.show_all()

    def _add_section(self, box, title):
        label = Gtk.Label()
        label.set_markup(f"<b>{title}</b>")
        label.set_halign(Gtk.Align.START)
        label.set_margin_top(8)
        box.pack_start(label, False, False, 0)

    def _save(self, _btn):
        new = AppConfig(
            model_profile=self._profile_combo.get_active_id() or self._config.model_profile,
            num_threads=int(self._threads_spin.get_value()),
            vad_threshold=self._config.vad_threshold,
            beep_volume=self._vol_scale.get_value(),
            typer=self._config.typer,
            hotkey_mode="start_stop" if self._mode_startstop.get_active() else "toggle",
            hotkey_toggle=self._hk_toggle.binding,
            hotkey_start=self._hk_start.binding,
            hotkey_stop=self._hk_stop.binding,
            hotkey_pause=self._hk_pause.binding,
        )
        if self._on_save:
            self._on_save(new)
        self.destroy()


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
            self._controller.profiles,
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
    config = AppConfig.load()

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
