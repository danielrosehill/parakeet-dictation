# Local Dictation

On-device voice typing for Linux with **built-in punctuation and capitalization**. Uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) to run NVIDIA NeMo ASR models locally — no cloud API, no GPU required.

## Model Profiles

| Profile | Model | Params | Download | Best for |
|---|---|---|---|---|
| **desktop** | Parakeet TDT 0.6B v3 (int8) | 600M | 639 MB | Desktop/workstation, best accuracy |
| **laptop** | Canary 180M Flash (int8) | 180M | 198 MB | Laptop, low RAM, travel |
| **streaming** | Nemotron Streaming 0.6B (int8) | 600M | 631 MB | True real-time (lowest latency) |

All models output punctuated, capitalized text natively — no post-processing needed.

## Setup

```bash
git clone https://github.com/danielrosehill/local-dictation.git
cd local-dictation
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download models (choose your profile)
uv pip install requests tqdm
python download_models.py desktop    # 639 MB — best accuracy
python download_models.py laptop     # 198 MB — lightweight
python download_models.py streaming  # 631 MB — real-time
python download_models.py all        # all profiles

# System dependencies (Ubuntu/Debian)
sudo apt install ydotool gir1.2-ayatanaappindicator3-0.1 libportaudio2 libgirepository-2.0-dev
```

## Usage

```bash
source .venv/bin/activate
python dictation_app.py
```

### Default hotkeys

| Action | Default | Description |
|---|---|---|
| Toggle | `Ctrl+0` | Start/stop dictation |
| Start | `Ctrl+9` | Start only (start/stop mode) |
| Stop | `Ctrl+8` | Stop only (start/stop mode) |
| Pause | `Ctrl+Alt+0` | Pause/resume without stopping engine |

### Hotkey modes

- **Toggle mode** (default): One key starts and stops dictation
- **Start/Stop mode**: Separate keys for starting and stopping — useful if you want a dedicated "I'm done" key

All hotkeys are configurable from Settings (system tray menu).

### Audio feedback

- **Rising tone** (880 Hz) — dictation started
- **Falling tone** (440 Hz) — dictation stopped
- **Double beep** (660 Hz) — paused/resumed

### How it works

1. **Silero VAD** detects speech segments in real time
2. When you pause speaking, the completed segment is sent to the ASR model
3. Transcribed text (with punctuation) is typed into the focused window
4. Text injection uses **ydotool** (Wayland) or **xdotool** (X11), auto-detected

The streaming profile (Nemotron) uses true frame-by-frame streaming instead of VAD segmentation, giving lower latency at the cost of slightly different sentence boundary behavior.

## Configuration

Settings stored in `~/.config/local-dictation/config.json`:

```json
{
  "model_profile": "desktop",
  "num_threads": 4,
  "vad_threshold": 0.5,
  "beep_volume": 0.5,
  "typer": "ydotool",
  "hotkey_mode": "toggle",
  "hotkey_toggle": "<ctrl>+0",
  "hotkey_start": "<ctrl>+9",
  "hotkey_stop": "<ctrl>+8",
  "hotkey_pause": "<ctrl>+<alt>+0"
}
```

## Requirements

- Python 3.10+
- Linux (tested on Ubuntu 25.04, KDE Plasma / Wayland)
- ~500 MB - 2 GB RAM depending on model
- ydotool + ydotoold (Wayland) or xdotool (X11)
