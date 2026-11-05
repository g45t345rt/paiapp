#!/bin/sh
# Desktop launcher: show the first-run dependency install in a terminal, then
# start Pai. Once .venv exists, Pai launches without a console window.
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ ! -x "$HERE/.venv/bin/python" ] && command -v x-terminal-emulator >/dev/null 2>&1; then
    exec x-terminal-emulator -e "$HERE/run.sh" "$@"
fi
exec "$HERE/run.sh" "$@"
