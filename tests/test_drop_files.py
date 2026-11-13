"""Tests for the OS drag-and-drop file classification
(run: .venv\\Scripts\\python.exe tests\\test_drop_files.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import _dropped_file_kind  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_kinds():
    check("png is an image", _dropped_file_kind("photo.png"), "image")
    check("uppercase jpeg is an image", _dropped_file_kind("PHOTO.JPEG"), "image")
    check("svg is an image", _dropped_file_kind("art.svg"), "image")
    check("tiff is an image", _dropped_file_kind("scan.tiff"), "image")
    check("pai is a project", _dropped_file_kind("scene.pai"), "project")
    check("uppercase pai is a project", _dropped_file_kind("SCENE.PAI"), "project")
    check("text is ignored", _dropped_file_kind("notes.txt"), None)
    check("no extension is ignored", _dropped_file_kind("README"), None)


def main():
    test_kinds()
    if FAILED:
        print()
        for failure in FAILED:
            print(f"FAIL {failure}")
        return 1
    print()
    print("all drop file tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
