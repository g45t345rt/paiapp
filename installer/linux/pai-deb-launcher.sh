#!/bin/sh
# Launcher installed by the Debian package.
#
# The app tree is read-only under /usr/lib/pai, so the private environment is
# created in the user's data directory instead (PAI_VENV is honored by
# setup_env.py). The first launch installs the pinned dependencies there.
export PAI_VENV="${XDG_DATA_HOME:-$HOME/.local/share}/pai/.venv"
exec /bin/sh /usr/lib/pai/run.sh "$@"
