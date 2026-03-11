#!/usr/bin/env python3
"""Parakeet Dictation — On-device voice typing using NVIDIA Parakeet TDT via sherpa-onnx.

Press Ctrl+0 to toggle dictation. Speech is transcribed with punctuation and
typed into the active window via ydotool (Wayland) or xdotool (X11).
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
from gi.repository import AyatanaAppIndicator3, GLib, Gtk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APP_NAME = "Parakeet Dictation"
APP_ID = "parakeet-dictation"
CONFIG_DIR = Path.home() / ".config" / APP_ID
CONFIG_FILE = CONFIG_DIR / "config.json"
MODELS_DIR = Path(__file__).resolve().parent / "models"
SAMPLE_RATE = 16000


@dataclass
class AppConfig:
    beep_volume: float = 0.5
    num_threads: int = 4
    vad_threshold: float = 0.5
    # Typing method: "ydotool" for Wayland, "xdotool" for X11
    typer: str = "ydotool"

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load() -> "AppConfig":
        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text())
                return AppConfig(**{k: v for k, v in data.items() if k in AppConfig.__dataclass_fields__})
            except Exception:
                pass
        return AppConfig()


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


def play_beep_on(volume: float = 0.5):
    sd.play(_generate_tone(880, 0.15, volume), samplerate=SAMPLE_RATE)


def play_beep_off(volume: float = 0.5):
    sd.play(_generate_tone(440, 0.15, volume), samplerate=SAMPLE_RATE)


# ---------------------------------------------------------------------------
# Text typer — injects text into the active window
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
# ASR Engine — sherpa-onnx Parakeet TDT + Silero VAD
# ---------------------------------------------------------------------------

class ASREngine:
    """Manages the recognizer and VAD. Runs audio capture in a thread."""

    def __init__(self, config: AppConfig, on_text, on_partial, on_error):
        self._config = config
        self._on_text = on_text        # callback(final_text: str)
        self._on_partial = on_partial  # callback(partial_text: str) — for status display
        self._on_error = on_error      # callback(error_msg: str)
        self._running = False
        self._thread = None
        self._stop_event = threading.Event()
        self._recognizer = None
        self._vad = None

    def _ensure_models(self):
        """Check that model files exist."""
        required = [
            MODELS_DIR / "encoder.int8.onnx",
            MODELS_DIR / "decoder.int8.onnx",
            MODELS_DIR / "joiner.int8.onnx",
            MODELS_DIR / "tokens.txt",
            MODELS_DIR / "silero_vad.onnx",
        ]
        missing = [f for f in required if not f.exists()]
        if missing:
            names = ", ".join(f.name for f in missing)
            raise FileNotFoundError(
                f"Missing model files in {MODELS_DIR}: {names}\n"
                "Run: python download_models.py"
            )

    def _init_recognizer(self):
        import sherpa_onnx

        self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=str(MODELS_DIR / "encoder.int8.onnx"),
            decoder=str(MODELS_DIR / "decoder.int8.onnx"),
            joiner=str(MODELS_DIR / "joiner.int8.onnx"),
            tokens=str(MODELS_DIR / "tokens.txt"),
            num_threads=self._config.num_threads,
            sample_rate=SAMPLE_RATE,
            feature_dim=128,
            provider="cpu",
            model_type="nemo_transducer",
            decoding_method="greedy_search",
        )

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
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=60)

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self):
        if self._running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        if not self._running:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        self._running = False

    def _run(self):
        try:
            self._ensure_models()
            self._init_recognizer()
        except Exception as e:
            GLib.idle_add(self._on_error, str(e))
            return

        self._running = True
        GLib.idle_add(self._on_partial, "")  # signal ready

        chunk_duration = 0.1  # 100ms chunks
        samples_per_chunk = int(SAMPLE_RATE * chunk_duration)

        try:
            with sd.InputStream(
                channels=1,
                dtype="float32",
                samplerate=SAMPLE_RATE,
                blocksize=samples_per_chunk,
            ) as stream:
                play_beep_on(self._config.beep_volume)
                while not self._stop_event.is_set():
                    audio, overflowed = stream.read(samples_per_chunk)
                    if overflowed:
                        continue
                    samples = audio.reshape(-1).tolist()
                    self._vad.accept_waveform(samples)

                    # Show "listening" indicator when speech is detected
                    if self._vad.is_speech_detected():
                        GLib.idle_add(self._on_partial, "Listening...")

                    # Process completed speech segments
                    while not self._vad.empty():
                        segment = self._vad.front
                        self._transcribe_segment(segment.samples)
                        self._vad.pop()

        except Exception as e:
            GLib.idle_add(self._on_error, str(e))
        finally:
            self._running = False
            play_beep_off(self._config.beep_volume)
            # Flush any remaining speech
            if self._vad:
                self._vad.flush()
                while not self._vad.empty():
                    segment = self._vad.front
                    self._transcribe_segment(segment.samples)
                    self._vad.pop()

    def _transcribe_segment(self, samples):
        """Run offline recognition on a VAD segment."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._recognizer.decode_stream(stream)
        text = stream.result.text.strip()
        if text:
            GLib.idle_add(self._on_text, text)


# ---------------------------------------------------------------------------
# Dictation controller
# ---------------------------------------------------------------------------

class DictationController:
    def __init__(self, config: AppConfig):
        self._config = config
        self._typer = TextTyper(config.typer)
        self._engine = ASREngine(
            config,
            on_text=self._on_final_text,
            on_partial=self._on_partial,
            on_error=self._on_error,
        )
        self._status_callback = None

    def set_status_callback(self, cb):
        self._status_callback = cb

    @property
    def is_running(self) -> bool:
        return self._engine.is_running

    @property
    def config(self) -> AppConfig:
        return self._config

    def toggle(self):
        if self._engine.is_running:
            self._engine.stop()
            if self._status_callback:
                self._status_callback("")
        else:
            self._engine.start()

    def stop(self):
        self._engine.stop()

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
# Global hotkey (pynput)
# ---------------------------------------------------------------------------

class HotkeyManager:
    def __init__(self, toggle_callback):
        self._toggle_callback = toggle_callback
        self._listener = None

    def start(self):
        from pynput import keyboard
        self._listener = keyboard.GlobalHotKeys({"<ctrl>+0": self._on_toggle})
        self._listener.daemon = True
        self._listener.start()

    def _on_toggle(self):
        GLib.idle_add(self._toggle_callback)

    def stop(self):
        if self._listener:
            self._listener.stop()
            self._listener = None


# ---------------------------------------------------------------------------
# Settings dialog
# ---------------------------------------------------------------------------

class SettingsDialog(Gtk.Dialog):
    def __init__(self, config: AppConfig, on_save):
        super().__init__(title="Parakeet Dictation — Settings", flags=0)
        self._config = config
        self._on_save = on_save
        self.set_default_size(350, 200)
        self.set_resizable(False)

        box = self.get_content_area()
        box.set_spacing(8)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        # Beep volume
        hbox = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox.pack_start(Gtk.Label(label="Beep volume:"), False, False, 0)
        self._vol_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 1, 0.05)
        self._vol_scale.set_value(config.beep_volume)
        hbox.pack_start(self._vol_scale, True, True, 0)
        box.pack_start(hbox, False, False, 0)

        # Threads
        hbox2 = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        hbox2.pack_start(Gtk.Label(label="CPU threads:"), False, False, 0)
        self._threads_spin = Gtk.SpinButton.new_with_range(1, 16, 1)
        self._threads_spin.set_value(config.num_threads)
        hbox2.pack_start(self._threads_spin, False, False, 0)
        box.pack_start(hbox2, False, False, 0)

        # Save button
        save_btn = Gtk.Button(label="Save")
        save_btn.connect("clicked", self._save)
        box.pack_start(save_btn, False, False, 4)

        self.show_all()

    def _save(self, _btn):
        self._config.beep_volume = self._vol_scale.get_value()
        self._config.num_threads = int(self._threads_spin.get_value())
        self._config.save()
        if self._on_save:
            self._on_save(self._config)
        self.destroy()


# ---------------------------------------------------------------------------
# System tray
# ---------------------------------------------------------------------------

class TrayIcon:
    def __init__(self, controller: DictationController):
        self._controller = controller

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

        self._toggle_item = Gtk.MenuItem(label="Start Dictation (Ctrl+0)")
        self._toggle_item.connect("activate", self._on_toggle)
        menu.append(self._toggle_item)

        self._status_item = Gtk.MenuItem(label="Idle")
        self._status_item.set_sensitive(False)
        menu.append(self._status_item)

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
        self._update_ui()

    def _update_ui(self):
        running = self._controller.is_running
        label = "Stop Dictation (Ctrl+0)" if running else "Start Dictation (Ctrl+0)"
        icon = "audio-input-microphone" if running else "audio-input-microphone-muted"
        self._toggle_item.set_label(label)
        self._indicator.set_icon_full(icon, "Dictation status")
        if not running:
            self._status_item.set_label("Idle")

    def _on_status_update(self, text: str):
        self._update_ui()
        if text:
            display = text[:60] + "\u2026" if len(text) > 60 else text
            self._status_item.set_label(f"\u25b6 {display}")
        else:
            if self._controller.is_running:
                self._status_item.set_label("Ready")
            else:
                self._status_item.set_label("Idle")

    def _on_settings(self, _item):
        SettingsDialog(self._controller.config, on_save=lambda c: None)

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
    if session == "wayland" and config.typer != "ydotool":
        config.typer = "ydotool"
    elif session == "x11" and config.typer != "xdotool":
        config.typer = "xdotool"

    controller = DictationController(config)
    tray = TrayIcon(controller)

    hotkey = HotkeyManager(lambda: controller.toggle() or tray._update_ui())
    hotkey.start()

    signal.signal(signal.SIGINT, lambda *_: (controller.stop(), Gtk.main_quit()))

    print(f"Parakeet Dictation running. Ctrl+0 to toggle. Ctrl+C to quit.")
    print(f"Typer: {config.typer} | Threads: {config.num_threads}")

    Gtk.main()
    hotkey.stop()


if __name__ == "__main__":
    main()
