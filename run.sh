#!/usr/bin/env sh
# Launch Pai. On the first run this creates a private .venv next to the app
# and installs the pinned dependencies; later runs reuse it. No other Python
# environment (such as a ComfyUI venv) is required or used.
#
# POSIX counterpart of run.bat for Linux and macOS (also works under Git Bash
# or WSL on Windows). Run with:  sh run.sh [--core-only] [--reinstall] [--] [app args]
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BASEPY=""

# Prefer a 3.12 interpreter, then 3.13/3.14 or any Python 3.12+ on PATH.
for candidate in python3.12 python3.13 python3.14 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && \
       "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; then
        BASEPY="$candidate"
        break
    fi
done

if [ -z "$BASEPY" ]; then
    echo "Pai needs Python 3.12 or newer on PATH." >&2
    echo "Install it from https://www.python.org/downloads/ and run this again." >&2
    exit 1
fi

exec "$BASEPY" "$HERE/setup_env.py" "$@"
