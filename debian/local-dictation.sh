#!/bin/bash
# Launcher for Local Dictation
# Uses a dedicated venv at /opt/local-dictation/.venv

INSTALL_DIR="/opt/local-dictation"
VENV_DIR="$INSTALL_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Python virtual environment not found at $VENV_DIR"
    echo "Run: sudo /opt/local-dictation/setup-pip-deps.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"
exec python3 "$INSTALL_DIR/dictation_app.py" "$@"
