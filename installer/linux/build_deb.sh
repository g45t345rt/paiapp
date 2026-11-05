#!/bin/sh
# Build a Debian package for Pai on Debian/Ubuntu.
#
# Usage:
#   sh build_deb.sh [--version VERSION] [--out DIR]
#
# The package installs the app tree read-only under /usr/lib/pai and a launcher
# at /usr/bin/pai. The private .venv is created in the user's data directory on
# first launch (PAI_VENV, see installer/linux/pai-deb-launcher.sh), so nothing
# is written under /usr. The package therefore depends on Python 3.12+ and is
# intended for Ubuntu 24.04+ / Debian 13+.
set -eu

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
OUT_DIR="$APP_ROOT/dist"
VERSION=""

while [ $# -gt 0 ]; do
    case "$1" in
        --version) shift; VERSION="$1" ;;
        --out) shift; OUT_DIR="$1" ;;
        -h|--help) sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ ! -f "$APP_ROOT/setup_env.py" ]; then
    echo "Could not find the Pai app tree at $APP_ROOT." >&2
    exit 1
fi
if ! command -v dpkg-deb >/dev/null 2>&1; then
    echo "dpkg-deb is required; build the package on Debian or Ubuntu." >&2
    exit 1
fi
if ! command -v rsync >/dev/null 2>&1; then
    echo "rsync is required." >&2
    exit 1
fi

if [ -z "$VERSION" ]; then
    VERSION=$(sed -n 's/^APP_VERSION *= *"\([^"]*\)".*/\1/p' "$APP_ROOT/paintapp/constants.py" | head -n 1)
fi
if [ -z "$VERSION" ]; then
    echo "Could not determine the version; pass --version." >&2
    exit 1
fi

PKG="pai_${VERSION}_all"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
ROOT="$WORK/$PKG"

mkdir -p \
    "$ROOT/DEBIAN" \
    "$ROOT/usr/lib/pai" \
    "$ROOT/usr/bin" \
    "$ROOT/usr/share/applications" \
    "$ROOT/usr/share/icons/hicolor/256x256/apps" \
    "$ROOT/usr/share/doc/pai"

rsync -a \
    --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
    --exclude '.github' --exclude 'dist' --exclude 'tests' --exclude 'installer' \
    --exclude '*.pyc' --exclude '*.pyo' --exclude '*.zip' \
    "$APP_ROOT"/ "$ROOT/usr/lib/pai"/

chmod 755 "$ROOT/usr/lib/pai/run.sh"
cp "$HERE/pai-deb-launcher.sh" "$ROOT/usr/bin/pai"
chmod 755 "$ROOT/usr/bin/pai"
cp "$HERE/pai-deb.desktop" "$ROOT/usr/share/applications/pai.desktop"
chmod 644 "$ROOT/usr/share/applications/pai.desktop"
cp "$APP_ROOT/pai_icon.png" "$ROOT/usr/share/icons/hicolor/256x256/apps/pai.png"
cp "$APP_ROOT/LICENSE" "$ROOT/usr/share/doc/pai/copyright"

cat > "$ROOT/DEBIAN/control" <<CONTROL
Package: pai
Version: $VERSION
Section: graphics
Priority: optional
Architecture: all
Depends: python3 (>= 3.12), python3-venv, python3-tk
Maintainer: Pai <pai@pai.app>
Description: AI paint studio
 Pai is a paint application with local AI image generation, upscaling and
 background removal. Dependencies are downloaded into a per-user environment
 on first launch.
CONTROL

cat > "$ROOT/DEBIAN/postinst" <<'POSTINST'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
POSTINST
cat > "$ROOT/DEBIAN/postrm" <<'POSTRM'
#!/bin/sh
set -e
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database -q /usr/share/applications || true
fi
POSTRM
chmod 755 "$ROOT/DEBIAN/postinst" "$ROOT/DEBIAN/postrm"

mkdir -p "$OUT_DIR"
OUT_DIR=$(CDPATH= cd -- "$OUT_DIR" && pwd)
dpkg-deb --root-owner-group --build "$ROOT" "$OUT_DIR/$PKG.deb"
echo "Built $OUT_DIR/$PKG.deb"
