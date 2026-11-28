#!/data/data/com.termux/files/usr/bin/bash
set -eu

if [ -z "${PREFIX:-}" ] || ! command -v pkg >/dev/null 2>&1; then
    echo "Run this script inside Termux." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    pkg update -y
    pkg install -y python
fi

if ! python3 -c 'import numpy' >/dev/null 2>&1; then
    pkg update -y
    pkg install -y python-numpy
fi

if ! command -v sshd >/dev/null 2>&1; then
    pkg update -y
    pkg install -y openssh
fi

if ! python3 -c 'import cv2' >/dev/null 2>&1; then
    pkg update -y
    pkg install -y x11-repo
    pkg update -y
    if ! pkg install -y opencv-python; then
        pkg install -y opencv
    fi
fi

python3 -m pip install flask

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(dirname "$SCRIPT_DIR")
TARGET_DIR=${VISIONGRAPH_DIR:-$HOME/VisionGraph}
mkdir -p "$TARGET_DIR"
cp "$REPO_ROOT"/*.py "$TARGET_DIR"/ 2>/dev/null || true
if [ -d "$REPO_ROOT/termux" ]; then
    cp "$REPO_ROOT"/termux/*.py "$TARGET_DIR"/ 2>/dev/null || true
fi

python3 - <<'PY'
import numpy
import cv2
import flask

print("numpy", numpy.__version__)
print("cv2", cv2.__version__)
if not hasattr(cv2, "aruco"):
    raise SystemExit("cv2.aruco is missing: install an OpenCV build with contrib modules")
print("aruco available")
print("flask", flask.__version__)
PY

install -m 0755 "$REPO_ROOT/termux/visiongraph-service.sh" "$PREFIX/bin/visiongraph-service"
mkdir -p "$HOME/.termux/boot" "$HOME/.config/visiongraph"
install -m 0755 "$REPO_ROOT/termux/boot.sh" "$HOME/.termux/boot/visiongraph"
CONFIG_FILE="$HOME/.config/visiongraph/termux_service.conf"
if [ ! -f "$CONFIG_FILE" ]; then
    printf 'WIDTH=600\nHEIGHT=400\nFPS=2\nCAMERA_ID=\nRES=\nHOST=127.0.0.1\nPORT=5000\n' > "$CONFIG_FILE"
fi

echo "VisionGraph Termux files installed in $TARGET_DIR"
echo "Boot autostart installed at ~/.termux/boot/visiongraph; install Termux:Boot from F-Droid to use it."
"$PREFIX/bin/visiongraph-service" start
