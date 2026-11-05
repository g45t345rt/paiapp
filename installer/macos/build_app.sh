#!/bin/sh
# Build a Pai.app bundle (and optionally a .dmg) on macOS.
#
# Usage:
#   sh build_app.sh [--src DIR] [--out DIR] [--dmg]
#
#   --src DIR  app tree to bundle (default: the repository root).
#   --out DIR  output directory (default: ../../dist).
#   --dmg      also write a compressed disk image.
#
# The bundle carries the app tree in Contents/Resources/app. Pai's first-run
# bootstrap creates .venv inside that folder, so keep the app in a user-writable
# location (e.g. ~/Applications), not a system-wide /Applications owned by root.
#
# macOS has no CUDA: the launcher exports PAI_TORCH_INDEX=none, so torch comes
# from PyPI and the engine runs on the CPU.
#
# Re-running against an existing Pai.app updates the tree in place and keeps
# the existing .venv, so the wheels are not downloaded again.
set -eu

if [ "$(uname -s)" != "Darwin" ]; then
    echo "build_app.sh must run on macOS." >&2
    exit 1
fi

HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP_ROOT=$(CDPATH= cd -- "$HERE/../.." && pwd)
OUT_DIR="$HERE/../../dist"
MAKE_DMG=0

while [ $# -gt 0 ]; do
    case "$1" in
        --src) shift; APP_ROOT=$(CDPATH= cd -- "$1" && pwd) ;;
        --out) shift; OUT_DIR="$1" ;;
        --dmg) MAKE_DMG=1 ;;
        -h|--help) sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

if [ ! -f "$APP_ROOT/setup_env.py" ] || [ ! -f "$APP_ROOT/run.sh" ]; then
    echo "Could not find the Pai app tree at $APP_ROOT." >&2
    echo "Use --src DIR, or run build_app.sh from the extracted release." >&2
    exit 1
fi

version=$(sed -n 's/^APP_VERSION *= *"\([^"]*\)".*/\1/p' "$APP_ROOT/paintapp/constants.py" | head -n 1)
version=${version:-1.0.2}
mkdir -p "$OUT_DIR"
OUT_DIR=$(CDPATH= cd -- "$OUT_DIR" && pwd)
APP="$OUT_DIR/Pai.app"
CONTENTS="$APP/Contents"
RES="$CONTENTS/Resources"
BUNDLED="$RES/app"

old_version=$(sed -n 's/.*<key>CFBundleShortVersionString<\/key>.*<string>\([^<]*\)<.*/\1/p' "$CONTENTS/Info.plist" 2>/dev/null | head -n 1 || true)
if [ -n "$old_version" ]; then
    echo "Updating Pai ${old_version} -> ${version} (keeping .venv and settings)."
fi

mkdir -p "$CONTENTS/MacOS" "$BUNDLED"

# rsync (no --delete) leaves a .venv already present in the bundle untouched.
rsync -a \
    --exclude '.venv' --exclude '__pycache__' --exclude '.git' \
    --exclude 'dist' --exclude '*.pyc' --exclude '*.pyo' --exclude '*.zip' \
    "$APP_ROOT"/ "$BUNDLED"/

cat > "$BUNDLED/bootstrap.command" <<'LAUNCH'
#!/bin/sh
export PAI_TORCH_INDEX="${PAI_TORCH_INDEX:-none}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$HERE/run.sh" "$@"
LAUNCH
chmod +x "$BUNDLED/bootstrap.command" "$BUNDLED/run.sh"

cat > "$CONTENTS/MacOS/Pai" <<'LAUNCH'
#!/bin/sh
export PAI_TORCH_INDEX="${PAI_TORCH_INDEX:-none}"
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
APP="$HERE/../Resources/app"
if [ ! -x "$APP/.venv/bin/python" ]; then
    exec /usr/bin/open -a Terminal "$APP/bootstrap.command" "$@"
fi
exec "$APP/bootstrap.command" "$@"
LAUNCH
chmod +x "$CONTENTS/MacOS/Pai"

ICON_LINE=""
if [ -f "$APP_ROOT/pai_icon.png" ] && command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
    ICONSET="$OUT_DIR/Pai.iconset"
    rm -rf "$ICONSET"
    mkdir -p "$ICONSET"
    for size in 16 32 128 256 512; do
        sips -z "$size" "$size" "$APP_ROOT/pai_icon.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
        sips -z "$((size * 2))" "$((size * 2))" "$APP_ROOT/pai_icon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
    done
    iconutil -c icns "$ICONSET" -o "$CONTENTS/Resources/Pai.icns"
    rm -rf "$ICONSET"
    ICON_LINE="<key>CFBundleIconFile</key><string>Pai</string>"
fi

cat > "$CONTENTS/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>Pai</string>
    <key>CFBundleDisplayName</key><string>Pai</string>
    <key>CFBundleIdentifier</key><string>app.pai.pai</string>
    <key>CFBundleExecutable</key><string>Pai</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>NSHighResolutionCapable</key><true/>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    $ICON_LINE
</dict>
</plist>
PLIST

echo "Built $APP"

if [ "$MAKE_DMG" -eq 1 ]; then
    DMG="$OUT_DIR/pai-$version-macos.dmg"
    rm -f "$DMG"
    hdiutil create -volname Pai -srcfolder "$APP" -ov -format UDZO "$DMG" >/dev/null
    echo "Built $DMG"
fi
