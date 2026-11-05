#!/usr/bin/env sh
# Install Pai into a per-user directory for Linux.
#
# Usage:
#   sh install.sh [--prefix DIR] [--setup] [--uninstall]
#
#   --prefix DIR  install directory (default: $HOME/.local/share/pai).
#                 Must be user-writable: Pai creates its private .venv there.
#   --setup       install the Python dependencies now instead of on first
#                 launch (runs setup_env.py --setup-only).
#   --uninstall   remove the install directory and the desktop entry.
#
# Re-running without --uninstall upgrades an existing install in place: the
# application files are overwritten, while .venv and the user's settings in
# ~/.config/Pai are left untouched. Only a changed requirements file triggers a
# dependency reinstall (see setup_env.py's hash marker).
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
PREFIX="${HOME}/.local/share/pai"
APPS_DIR="${HOME}/.local/share/applications"
DESKTOP_FILE="${APPS_DIR}/pai.desktop"
SETUP=0
UNINSTALL=0

if [ ! -f "$APP_ROOT/setup_env.py" ] || [ ! -f "$APP_ROOT/run.sh" ]; then
    echo "Could not find the Pai app tree at $APP_ROOT." >&2
    echo "Run install.sh from inside the extracted release (installer/linux/install.sh)." >&2
    exit 1
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --prefix) shift; PREFIX="$1" ;;
        --setup) SETUP=1 ;;
        --uninstall) UNINSTALL=1 ;;
        -h|--help) sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ "$UNINSTALL" -eq 1 ]; then
    rm -rf "$PREFIX"
    rm -f "$DESKTOP_FILE"
    if command -v update-desktop-database >/dev/null 2>&1; then
        update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
    fi
    echo "Removed Pai from $PREFIX and $DESKTOP_FILE."
    exit 0
fi

find_python() {
    for candidate in python3.12 python3.13 python3.14 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && \
           "$candidate" -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)' >/dev/null 2>&1; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

app_version() {
    sed -n 's/^APP_VERSION *= *"\([^"]*\)".*/\1/p' "$1" 2>/dev/null | head -n 1
}

PY=$(find_python || true)
if [ -z "$PY" ]; then
    echo "Warning: Python 3.12-3.14 was not found on PATH." >&2
    echo "Pai offers to install it on first launch; or install it now (e.g. sudo apt install python3.12 python3.12-venv python3-tk)." >&2
elif ! "$PY" -c 'import tkinter' >/dev/null 2>&1; then
    echo "Warning: this Python has no tkinter. Install the Tk package (e.g. python3-tk)." >&2
fi

if [ -f "$PREFIX/paintapp/constants.py" ]; then
    OLD_VERSION=$(app_version "$PREFIX/paintapp/constants.py")
    NEW_VERSION=$(app_version "$APP_ROOT/paintapp/constants.py")
    echo "Updating Pai ${OLD_VERSION:-?} -> ${NEW_VERSION:-?} (keeping .venv and settings)."
fi

mkdir -p "$PREFIX"
tar -C "$APP_ROOT" \
    --exclude='./.venv' --exclude='./.venv/*' \
    --exclude='./.git' --exclude='./.git/*' \
    --exclude='./dist' --exclude='./dist/*' \
    --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
    --exclude='*.zip' \
    -cf - . | tar -C "$PREFIX" -xf -

chmod +x "$PREFIX/run.sh"
cp "$HERE/launch.sh" "$PREFIX/launch.sh"
chmod +x "$PREFIX/launch.sh"

mkdir -p "$APPS_DIR"
sed -e "s|@EXEC@|$PREFIX/launch.sh|g" \
    -e "s|@ICON@|$PREFIX/pai_icon.png|g" \
    "$HERE/pai.desktop" > "$DESKTOP_FILE"
chmod 644 "$DESKTOP_FILE"
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi

echo "Installed Pai to $PREFIX"
echo "Registered $DESKTOP_FILE"

if [ "$SETUP" -eq 1 ]; then
    sh "$PREFIX/run.sh" --setup-only
fi

echo "Launch Pai from your application menu, or run: $PREFIX/run.sh"
