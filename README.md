# Parakeet Dictation

On-device voice typing for Linux using [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) via [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Press **Ctrl+0** to toggle live dictation — transcribed speech is typed into your active window **with punctuation and capitalization**.

## Why Parakeet?

- **Built-in punctuation and capitalization** — no post-processing needed
- **~6% WER** — beats Whisper Large v3 in accuracy at a fraction of the size
- **Runs on CPU** — int8 quantized, works on laptops without a GPU
- **~30x real-time** on modern CPUs (i7-12700F class)
- **600M parameters** — good accuracy/speed tradeoff

## How it works

1. **Silero VAD** detects speech segments in real time
2. When you pause speaking, the completed segment is sent to **Parakeet TDT** for transcription
3. The transcribed text (with punctuation) is typed into the focused window via **ydotool** (Wayland) or **xdotool** (X11)

This is a VAD-segmented offline approach, not true streaming. There's ~1-2s latency after you pause speaking before text appears. The tradeoff is much better accuracy and punctuation than streaming models.

## Setup

```bash
# Clone and create venv
git clone <this-repo>
cd parakeet-dictation
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Download models (~670 MB)
uv pip install requests tqdm
python download_models.py

# System dependencies
sudo apt install ydotool gir1.2-ayatanaappindicator3-0.1 libportaudio2 libgirepository-2.0-dev

# Ensure ydotoold is running (Wayland)
systemctl --user start ydotoold
# or just run: ydotoold &
```

## Usage

```bash
source .venv/bin/activate
python dictation_app.py
```

- **Ctrl+0** — Toggle dictation on/off
- System tray menu: Start/Stop, Settings, Quit
- Audio beeps confirm start/stop
- Auto-detects Wayland (ydotool) vs X11 (xdotool)

## Configuration

Settings are stored in `~/.config/parakeet-dictation/config.json`:

- **beep_volume**: 0.0 - 1.0
- **num_threads**: CPU threads for inference (default: 4)
- **vad_threshold**: Voice activity detection sensitivity (default: 0.5)
- **typer**: "ydotool" or "xdotool" (auto-detected)

## Requirements

- Python 3.10+
- Linux (tested on Ubuntu 25.04, KDE Plasma / Wayland)
- ~2 GB RAM for model inference
- ydotool (Wayland) or xdotool (X11) for text injection
- ydotoold daemon running (for Wayland)

## Model Info

- **ASR**: Parakeet TDT 0.6B v3, int8 quantized (~670 MB)
- **VAD**: Silero VAD (~2.3 MB)
- Models are downloaded to `./models/` and git-ignored
