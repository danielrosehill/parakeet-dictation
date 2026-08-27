# Parakeet Dictation

On-device voice typing for **Linux / Wayland** with **built-in punctuation and capitalization** — no cloud API, no GPU required.

Uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) to run NVIDIA NeMo ASR models locally, including [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), one of the highest-accuracy open-source speech recognition models available. Unlike Whisper-based dictation tools, Parakeet produces **natively punctuated and capitalized** output with no post-processing — making it ideal for live typing workflows.

Validated on **Ubuntu 25.04** with **KDE Plasma 6 / Wayland**.

> **Note**: This is a Wayland-only tool. X11 is not supported.

## Why not Whisper?

Whisper is excellent for batch transcription but has drawbacks for live dictation:

- **No native punctuation control** — requires separate punctuation models or heuristics
- **High latency** — designed for processing complete audio files, not real-time segments
- **Heavy** — even `whisper-small` uses more RAM than Parakeet TDT 0.6B (int8) while being less accurate for English

The NeMo family (Parakeet, Canary, Nemotron) was designed for production speech pipelines and outputs punctuated text natively. Parakeet TDT 0.6B v3 achieves state-of-the-art word error rates on English benchmarks while running ~30x real-time on CPU.

## Why sherpa-onnx?

[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) provides a clean, optimized runtime for running ONNX-exported NeMo models. It handles feature extraction, decoding, and streaming — all in a single Python package with no external dependencies beyond ONNX Runtime. This avoids the overhead of full NeMo/PyTorch and enables efficient CPU-only inference even on laptops.

## Model Profiles

| Profile | Model | Type | Params | Download | Best for |
|---|---|---|---|---|---|
| **desktop** | Parakeet TDT 0.6B v3 (int8) | Offline (VAD-segmented) | 600M | 639 MB | Desktop/workstation — best accuracy |
| **desktop-fp32** | Parakeet TDT 0.6B v3 (fp32) | Offline (VAD-segmented) | 600M | 2432 MB | Same model unquantized — cleanest wording |
| **laptop** | Canary 180M Flash (int8) | Offline (VAD-segmented) | 180M | 198 MB | Laptop, low RAM, travel |
| **streaming** | Nemotron Streaming 0.6B (int8) | Online (frame-by-frame) | 600M | 631 MB | True real-time — lowest latency |

### Model types explained

- **Offline (VAD-segmented)**: TEN VAD detects when you pause speaking, then sends the completed speech segment to the model. You get punctuated text ~1–2 seconds after each pause. Best accuracy.
- **Online (streaming)**: The model processes audio frame-by-frame as you speak, outputting partial results in real time. Lower latency, but slightly different sentence boundary behavior.

All models output punctuated, capitalized text natively.

### Choosing a model

- **Desktop/workstation with plenty of RAM**: Use `desktop` (Parakeet TDT 0.6B). Best accuracy, handles accents and technical vocabulary well. ~2 GB RAM.
- **RAM to spare**: Use `desktop-fp32` — the same model without int8 quantization, at the same decode speed (~190 ms per 10 s of audio). Slightly cleaner wording on technical terms. ~2.6 GB RAM.
- **Laptop or low-RAM machine**: Use `laptop` (Canary 180M Flash). Only 198 MB download, ~500 MB RAM. Supports English, Spanish, German, and French.
- **Lowest possible latency**: Use `streaming` (Nemotron Streaming 0.6B). Text appears as you speak rather than after pauses. English only.

## Install

### Option A: .deb package (recommended)

```bash
git clone https://github.com/danielrosehill/parakeet-dictation.git
cd parakeet-dictation
chmod +x build-deb.sh
./build-deb.sh

sudo dpkg -i parakeet-dictation_1.0.0.deb
sudo apt-get install -f          # resolve any missing deps
sudo /opt/parakeet-dictation/setup-pip-deps.sh

parakeet-dictation
```

### Option B: Run from source

```bash
git clone https://github.com/danielrosehill/parakeet-dictation.git
cd parakeet-dictation
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download models (or use the in-app model manager)
uv pip install requests tqdm
python download_models.py desktop    # 639 MB — best accuracy
python download_models.py desktop-fp32  # 2432 MB — unquantized, needs ~2.6 GB RAM
python download_models.py laptop     # 198 MB — lightweight
python download_models.py streaming  # 631 MB — real-time
python download_models.py all        # all profiles

# System dependencies (Ubuntu/Debian)
sudo apt install wtype gir1.2-ayatanaappindicator3-0.1 libportaudio2 libgirepository-2.0-dev libc++1

python dictation_app.py
```

### Text input method

Text is typed into the focused application via **wtype** (Wayland-native). This is the default and recommended method.

Alternative methods available in **Settings > General**:

| Method | Command | Notes |
|---|---|---|
| **wtype** (default) | `wtype` | Native Wayland keystroke injection. Just works. |
| **ydotool** | `ydotool` | Requires `ydotoold` daemon + `/dev/uinput` permissions (see below) |
| **clipboard** | `wl-copy` + `wtype Ctrl+V` | Pastes via clipboard. Works everywhere but overwrites clipboard. |

#### ydotool setup (only if using ydotool method)

```bash
sudo groupadd -f input
sudo usermod -aG input $USER
echo 'KERNEL=="uinput", GROUP="input", MODE="0660"' | sudo tee /etc/udev/rules.d/80-uinput.rules
sudo udevadm control --reload-rules
sudo udevadm trigger /dev/uinput
# Log out and back in for group membership to take effect
```

### TEN VAD system dependency

[TEN VAD](https://github.com/TEN-framework/ten-vad) requires `libc++1` (C++ standard library). The .deb package installs this automatically. If running from source:

```bash
sudo apt install libc++1
```

On some systems, the `libc++.so.1` symlink may be missing. If you get an error about `libc++.so.1`, create the symlink:

```bash
sudo ln -sf /lib/x86_64-linux-gnu/libc++.so.1.0.* /lib/x86_64-linux-gnu/libc++.so.1
sudo ln -sf /lib/x86_64-linux-gnu/libc++abi.so.1.0.* /lib/x86_64-linux-gnu/libc++abi.so.1
sudo ldconfig
```

## Usage

The app runs as a **system tray indicator** with an optional **full-size main window**.

- **Tray icon**: Right-click for start/stop, model switching, settings, and to open the main window.
- **Main window**: Open from the tray menu ("Show Window"). Provides a transcript view, start/stop/pause controls, and a microphone selector. Closing the window hides it back to tray.

### Default hotkeys

| Action | Default | Description |
|---|---|---|
| Toggle | `Ctrl+0` | Start/stop dictation |
| Start | `Ctrl+9` | Start only (start/stop mode) |
| Stop | `Ctrl+8` | Stop only (start/stop mode) |
| Pause | `Ctrl+Alt+0` | Pause/resume without stopping engine |

### Hotkey modes

- **Toggle mode** (default): One key starts and stops dictation
- **Start/Stop mode**: Separate keys for starting and stopping

All hotkeys are rebindable from **Settings > Hotkeys**.

### Microphone selection

Choose your input device from:
- **Settings > General > Microphone** (persisted)
- **Main window** header bar mic selector (quick switch)

Leave set to "System Default" to use your desktop's default input device.

The input stream stays open while the app runs (your desktop may show a
mic-in-use indicator). While idle, only the last 0.5 seconds is kept in
memory; nothing is transcribed or stored. That half-second is included
when you press the hotkey, so your first words are never clipped.

### Model manager

Open **Settings > Models** to browse available model profiles, download them in-app, and select which one to use.

### Audio feedback

Cues play from your desktop sound theme (via `canberra-gtk-play`), so they follow the theme and alert volume you set in **System Settings > Sound**:

- **`device-added`** — dictation started
- **`device-removed`** — dictation stopped
- **`message`** — paused/resumed

If `canberra-gtk-play` isn't installed (package `gnome-session-canberra` on Debian/Ubuntu), the app falls back to simple sine tones. The in-app beep-volume slider attenuates relative to the system alert volume.

### Night mode

Automatically suppresses audio feedback between configurable hours (default 22:00–09:00). Enable from **Settings > General**.

### How it works

1. **[TEN VAD](https://github.com/TEN-framework/ten-vad)** detects speech segments in real time (~306 KB, lightweight)
2. When you pause speaking, the completed segment is sent to the ASR model
3. Transcribed text (with punctuation) is typed into the focused window via **wtype**

The streaming profile uses frame-by-frame processing instead of VAD segmentation.

## Configuration

Settings stored in `~/.config/parakeet-dictation/config.json`:

```json
{
  "model_profile": "desktop",
  "num_threads": 4,
  "vad_threshold": 0.5,
  "beep_volume": 0.5,
  "audio_device": "",
  "typer": "wtype",
  "hotkey_mode": "toggle",
  "hotkey_toggle": "<ctrl>+0",
  "hotkey_start": "<ctrl>+9",
  "hotkey_stop": "<ctrl>+8",
  "hotkey_pause": "<ctrl>+<alt>+0",
  "night_mode": false,
  "night_start": 22,
  "night_end": 9
}
```

## Requirements

- Python 3.10+
- Linux with Wayland (validated on Ubuntu 25.04, KDE Plasma 6)
- ~500 MB – 2 GB RAM depending on model
- wtype (Wayland text input)
- libc++1 (for TEN VAD)

## License

MIT
