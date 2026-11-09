#!/usr/bin/env sh
# Launch Pai. On the first run this creates a private .venv next to the app
# and installs the pinned dependencies; later runs reuse it. No other Python
# environment (such as a ComfyUI venv) is required or used.
#
# POSIX counterpart of run.bat for Linux and macOS (also works under Git Bash
# or WSL on Windows). Run with:  sh run.sh [--core-only] [--reinstall] [--setup-only] [--rocm] [--] [app args]
#
# When no supported Python (3.12-3.14) is on PATH the launcher offers to
# download and install one (Homebrew/apt/dnf/zypper/pacman, then the official
# python.org installer on macOS).
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASEPY=""

# Prefer a 3.12 interpreter, then 3.13/3.14 or any supported Python on PATH.
for candidate in python3.12 python3.13 python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if (3, 12) <= sys.version_info[:2] < (3, 15) else 1)' >/dev/null 2>&1; then
        BASEPY="$candidate"
        break
    fi
done

if [ -z "$BASEPY" ]; then
    echo "Pai needs Python 3.12-3.14, which was not found on PATH." >&2
    installer="$HERE/installer/unix/install_python.sh"
    # Only offer when the helper ships with the app and we have a terminal to
    # answer the prompt on (a desktop double-click or a pipeline must not hang).
    if [ -f "$installer" ] && [ -t 0 ]; then
        printf 'Download and install Python now? [y/N] ' >&2
        read -r answer || answer=""
        case "$answer" in
            [yY]|[yY][eE][sS])
                if ! BASEPY=$(sh "$installer"); then
                    BASEPY=""
                fi
                ;;
        esac
    fi
fi

if [ -z "$BASEPY" ]; then
    echo "Pai needs Python 3.12-3.14 on PATH." >&2
    echo "Install one from https://www.python.org/downloads/ and run this again." >&2
    exit 1
fi

exec "$BASEPY" "$HERE/setup_env.py" "$@"
