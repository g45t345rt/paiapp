"""Assemble the per-OS source archives the installers unpack.

Usage:
    python installer/build_release.py [--out DIR]

Writes, under ``dist/`` by default::

    pai-<version>-windows.zip
    pai-<version>-linux.tar.gz
    pai-<version>-macos.tar.gz
    SHA256SUMS

The private ``.venv``, caches, VCS metadata and previous output are excluded,
so ``setup_env.py`` builds a fresh environment on the target machine.
``pai_icon.ico`` is generated from ``pai_icon.png`` when Pillow is available,
for the Windows installer and shortcuts.
"""

import argparse
import hashlib
import os
import re
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSTANTS = ROOT / "paintapp" / "constants.py"

EXCLUDE_DIRS = {
    ".git", ".venv", "__pycache__", "dist", "build",
    ".pytest_cache", ".mypy_cache", ".ruff_cache",
}
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".zip", ".tar.gz")
EXCLUDE_NAMES = {".DS_Store", "Thumbs.db"}


def app_version():
    """Read ``APP_VERSION`` from the app constants without importing the app."""
    text = CONSTANTS.read_text(encoding="utf-8")
    match = re.search(r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not match:
        sys.exit("Could not read APP_VERSION from paintapp/constants.py")
    return match.group(1)


def source_files():
    """Every file in the app tree that belongs in a release archive."""
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            if name in EXCLUDE_NAMES or name.endswith(EXCLUDE_SUFFIXES):
                continue
            files.append(Path(dirpath) / name)
    return files


def write_zip(files, top, out):
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, f"{top}/{path.relative_to(ROOT).as_posix()}")


def write_tar(files, top, out):
    with tarfile.open(out, "w:gz") as archive:
        for path in files:
            archive.add(path, f"{top}/{path.relative_to(ROOT).as_posix()}")


def make_windows_icon():
    """Render ``pai_icon.ico`` from ``pai_icon.png`` for Windows shortcuts."""
    source = ROOT / "pai_icon.png"
    target = ROOT / "pai_icon.ico"
    if not source.exists():
        return None
    try:
        from PIL import Image
    except ImportError:
        print("Pillow is not installed; skipping pai_icon.ico "
              "(Windows shortcuts will use a default icon).")
        return None
    image = Image.open(source).convert("RGBA")
    image.save(target, sizes=[(16, 16), (24, 24), (32, 32), (48, 48),
                              (64, 64), (128, 128), (256, 256)])
    return target


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "dist"),
                        help="output directory (default: dist/)")
    args = parser.parse_args(argv)

    version = app_version()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    icon = make_windows_icon()
    if icon is not None:
        print(f"wrote {icon}")

    files = source_files()
    top = f"pai-{version}"
    archives = [
        (out_dir / f"pai-{version}-windows.zip", write_zip),
        (out_dir / f"pai-{version}-linux.tar.gz", write_tar),
        (out_dir / f"pai-{version}-macos.tar.gz", write_tar),
    ]
    for path, writer in archives:
        writer(files, top, path)
        print(f"wrote {path}")

    sums = out_dir / "SHA256SUMS"
    with open(sums, "w", encoding="ascii") as handle:
        for path, _writer in archives:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(f"wrote {sums}")

    print("\nNext steps:")
    print(f"  Windows  ISCC.exe /DAppVersion={version} installer\\windows\\pai.iss")
    print("  Linux    extract the tar.gz and run installer/linux/install.sh")
    print("  macOS    extract the tar.gz and run installer/macos/build_app.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
