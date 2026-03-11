#!/bin/bash
# Post-install script to set up Python dependencies in a venv.
# Run as root after installing the .deb package:
#   sudo /opt/parakeet-dictation/setup-pip-deps.sh

set -e

INSTALL_DIR="/opt/parakeet-dictation"
VENV_DIR="$INSTALL_DIR/.venv"

echo "Setting up Parakeet Dictation Python environment..."

# Create venv with access to system packages (for PyGObject/gi)
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# Install/upgrade pip dependencies (PyGObject comes from system apt)
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install sherpa-onnx sounddevice pynput numpy requests tqdm

# Make venv world-readable so any user can run the app
chmod -R a+rX "$VENV_DIR"

echo ""
echo "Setup complete. Run 'parakeet-dictation' to start the app."
echo "On first use, download a model from Settings → Models tab."
