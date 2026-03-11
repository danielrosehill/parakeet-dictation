#!/bin/bash
# Launcher for Parakeet Dictation
# Uses a dedicated venv at /opt/parakeet-dictation/.venv

INSTALL_DIR="/opt/parakeet-dictation"
VENV_DIR="$INSTALL_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Python virtual environment not found at $VENV_DIR"
    echo "Run: sudo /opt/parakeet-dictation/setup-pip-deps.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"
exec python3 "$INSTALL_DIR/dictation_app.py" "$@"
