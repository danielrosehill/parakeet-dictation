# Parakeet Dictation

On-device voice typing for Linux with **built-in punctuation and capitalization** — no cloud API, no GPU required.

Uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) to run NVIDIA NeMo ASR models locally, including [Parakeet TDT 0.6B](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), one of the highest-accuracy open-source speech recognition models available. Unlike Whisper-based dictation tools, Parakeet produces **natively punctuated and capitalized** output with no post-processing — making it ideal for live typing workflows.

Validated on **Ubuntu 25.04** with **KDE Plasma 6 / Wayland**.

## Why not Whisper?

Whisper is excellent for batch transcription but has drawbacks for live dictation:

- **No native punctuation control** — requires separate punctuation models or heuristics
- **High latency** — designed for processing complete audio files, not real-time segments
- **Heavy** — even `whisper-small` uses more RAM than Parakeet TDT 0.6B (int8) while being less accurate for English

The NeMo family (Parakeet, Canary, Nemotron) was designed for production speech pipelines and outputs punctuated text natively. Parakeet TDT 0.6B v3 achieves state-of-the-art word error rates on English benchmarks while running ~30x real-time on CPU.

## Model Profiles

| Profile | Model | Type | Params | Download | Best for |
|---|---|---|---|---|---|
| **desktop** | Parakeet TDT 0.6B v3 (int8) | Offline (VAD-segmented) | 600M | 639 MB | Desktop/workstation — best accuracy |
| **laptop** | Canary 180M Flash (int8) | Offline (VAD-segmented) | 180M | 198 MB | Laptop, low RAM, travel |
| **streaming** | Nemotron Streaming 0.6B (int8) | Online (frame-by-frame) | 600M | 631 MB | True real-time — lowest latency |

### Model types explained

- **Offline (VAD-segmented)**: Silero VAD detects when you pause speaking, then sends the completed speech segment to the model. You get punctuated text ~1–2 seconds after each pause. Best accuracy.
- **Online (streaming)**: The model processes audio frame-by-frame as you speak, outputting partial results in real time. Lower latency, but slightly different sentence boundary behavior.

All models output punctuated, capitalized text natively.

### Choosing a model

- **Desktop/workstation with plenty of RAM**: Use `desktop` (Parakeet TDT 0.6B). Best accuracy, handles accents and technical vocabulary well. ~2 GB RAM.
- **Laptop or low-RAM machine**: Use `laptop` (Canary 180M Flash). Only 198 MB download, ~500 MB RAM. Supports English, Spanish, German, and French.
- **Lowest possible latency**: Use `streaming` (Nemotron Streaming 0.6B). Text appears as you speak rather than after pauses. English only.

## Install

### Option A: .deb package (recommended)

```bash
git clone https://github.com/danielrosehill/parakeet-dictation.git
cd parakeet-dictation
chmod +x build-deb.sh
./build-deb.sh

sudo dpkg -i parakeet-dictation_0.1.0.deb
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
python download_models.py laptop     # 198 MB — lightweight
python download_models.py streaming  # 631 MB — real-time
python download_models.py all        # all profiles

# System dependencies (Ubuntu/Debian)
sudo apt install ydotool gir1.2-ayatanaappindicator3-0.1 libportaudio2 libgirepository-2.0-dev

python dictation_app.py
```

## Usage

The app runs as a **system tray indicator**. Right-click the tray icon to access Settings, switch models, or quit.

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

All hotkeys are rebindable from **Settings → Hotkeys**.

### Model manager

Open **Settings → Models** to browse available model profiles, download them in-app, and select which one to use. The **About** tab has recommendations for choosing a model.

### Audio feedback

- **Rising tone** (880 Hz) — dictation started
- **Falling tone** (440 Hz) — dictation stopped
- **Double beep** (660 Hz) — paused/resumed

### Night mode

Automatically suppresses audio feedback between configurable hours (default 22:00–09:00). Enable from **Settings → General**.

### How it works

1. **Silero VAD** detects speech segments in real time
2. When you pause speaking, the completed segment is sent to the ASR model
3. Transcribed text (with punctuation) is typed into the focused window via **ydotool** (Wayland) or **xdotool** (X11), auto-detected

The streaming profile uses frame-by-frame processing instead of VAD segmentation.

## Configuration

Settings stored in `~/.config/parakeet-dictation/config.json`:

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
  "hotkey_pause": "<ctrl>+<alt>+0",
  "night_mode": false,
  "night_start": 22,
  "night_end": 9
}
```

## Requirements

- Python 3.10+
- Linux (validated on Ubuntu 25.04, KDE Plasma 6 / Wayland)
- ~500 MB – 2 GB RAM depending on model
- ydotool + ydotoold (Wayland) or xdotool (X11)

## License

MIT
