#!/bin/bash
# Build the .deb package for Parakeet Dictation
# Usage: ./build-deb.sh
set -e

VERSION="0.1.0"
PKG_NAME="parakeet-dictation"
BUILD_DIR="$(mktemp -d)"
PKG_DIR="$BUILD_DIR/${PKG_NAME}_${VERSION}"

echo "Building ${PKG_NAME} ${VERSION}..."

# Create directory structure
mkdir -p "$PKG_DIR/DEBIAN"
mkdir -p "$PKG_DIR/opt/parakeet-dictation"
mkdir -p "$PKG_DIR/usr/bin"
mkdir -p "$PKG_DIR/usr/share/applications"

# DEBIAN control file
cat > "$PKG_DIR/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: sound
Priority: optional
Architecture: all
Depends: python3 (>= 3.10), python3-gi, python3-numpy, gir1.2-ayatanaappindicator3-0.1, libportaudio2, python3-venv, ydotool
Maintainer: Daniel Rosehill <daniel@danielrosehill.co.il>
Homepage: https://github.com/danielrosehill/parakeet-dictation
Description: On-device voice typing using Parakeet and NeMo ASR models
 Parakeet Dictation provides on-device voice typing for Linux using
 sherpa-onnx and NVIDIA NeMo ASR models (Parakeet, Canary, Nemotron).
 Features built-in punctuation and capitalization, multiple model profiles
 (desktop/laptop/streaming), configurable hotkeys, and a system tray
 interface. No cloud API or GPU required.
EOF

# Post-install message
cat > "$PKG_DIR/DEBIAN/postinst" << 'EOF'
#!/bin/bash
set -e
case "$1" in
    configure)
        echo ""
        echo "============================================"
        echo " Parakeet Dictation installed successfully"
        echo "============================================"
        echo ""
        echo " Next step — set up Python dependencies:"
        echo "   sudo /opt/parakeet-dictation/setup-pip-deps.sh"
        echo ""
        echo " Then run:"
        echo "   parakeet-dictation"
        echo ""
        ;;
esac
exit 0
EOF
chmod 755 "$PKG_DIR/DEBIAN/postinst"

# Application files
cp dictation_app.py "$PKG_DIR/opt/parakeet-dictation/"
chmod 755 "$PKG_DIR/opt/parakeet-dictation/dictation_app.py"

cp models.json "$PKG_DIR/opt/parakeet-dictation/"
cp requirements.txt "$PKG_DIR/opt/parakeet-dictation/"
cp download_models.py "$PKG_DIR/opt/parakeet-dictation/"

cp debian/setup-pip-deps.sh "$PKG_DIR/opt/parakeet-dictation/"
chmod 755 "$PKG_DIR/opt/parakeet-dictation/setup-pip-deps.sh"

# Launcher script
cp debian/parakeet-dictation.sh "$PKG_DIR/usr/bin/parakeet-dictation"
chmod 755 "$PKG_DIR/usr/bin/parakeet-dictation"

# Desktop file
cp debian/parakeet-dictation.desktop "$PKG_DIR/usr/share/applications/"

# Build the .deb
dpkg-deb --root-owner-group --build "$PKG_DIR"

# Move to project directory
DEB_FILE="${PKG_NAME}_${VERSION}.deb"
mv "$PKG_DIR.deb" "./$DEB_FILE"
rm -rf "$BUILD_DIR"

echo ""
echo "Built: ./$DEB_FILE"
echo ""
echo "Install with:"
echo "  sudo dpkg -i $DEB_FILE"
echo "  sudo apt-get install -f   # resolve any missing deps"
echo "  sudo /opt/parakeet-dictation/setup-pip-deps.sh"
