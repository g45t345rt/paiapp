"""Checks the canvas-origin helpers used by resize and generative fill.

Run: .venv\\Scripts\\python.exe tests\\test_canvas_resize.py

The helpers are called unbound with minimal stand-ins, so the geometry (mask
remap, object shift, guide shift) is exercised without a Tk root or a loaded
model.
"""

import os
import sys
from types import SimpleNamespace

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import path, textrender  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r}, want {want!r}")
        print(f"FAIL {name}: got {got!r}, want {want!r}")
    else:
        print(f"ok  {name}")


def test_remap_keeps_existing_mask():
    old = SimpleNamespace(size=(100, 80))
    old.inpaint_mask = Image.new("L", old.size, 0)
    ImageDraw.Draw(old.inpaint_mask).rectangle((10, 10, 20, 20), fill=255)
    old._inpaint_mask_shared = False

    PaintApp._remap_inpaint_mask(None, old, (200, 160), 50, 40)

    mask = old.inpaint_mask
    check("mask size", mask.size, (200, 160))
    check("painted mask moved", mask.getpixel((65, 55)), 255)
    check("old position cleared", mask.getpixel((15, 15)), 0)
    check("new border unmarked", mask.getpixel((0, 0)), 0)
    check("not shared", old._inpaint_mask_shared, False)


def test_remap_clears_empty_mask():
    doc = SimpleNamespace(size=(100, 80), inpaint_mask=None,
                          _inpaint_mask_shared=True)
    PaintApp._remap_inpaint_mask(None, doc, (120, 100), 10, 10)
    check("empty mask stays empty", doc.inpaint_mask, None)
    check("not shared", doc._inpaint_mask_shared, False)


def test_shift_layer_objects():
    layer = SimpleNamespace(
        is_link=False,
        texts=[textrender.new_spec("hi", x=5, y=6)],
        paths=[path.new_spec([[1, 0], [3, 0]], [[1, 0], [3, 0]],
                             [[1, 0], [3, 0]])],
        invalidate_texts=lambda: None,
        invalidate_paths=lambda: None,
    )
    moved = {}
    fake = SimpleNamespace(
        _shift_pose_scene=lambda item, dx, dy: moved.update(pose=(dx, dy)),
    )
    PaintApp._shift_layer_objects(fake, layer, 10, 20)

    check("text shifted", (layer.texts[0]["x"], layer.texts[0]["y"]), (15, 26))
    check("path shifted", layer.paths[0]["anchors"][0], [11, 20])
    check("pose shifted", moved["pose"], (10, 20))


def test_shift_layer_objects_skips_links():
    layer = SimpleNamespace(
        is_link=True,
        texts=[textrender.new_spec("hi", x=5, y=6)],
        paths=[],
        invalidate_texts=lambda: None,
        invalidate_paths=lambda: None,
    )
    fake = SimpleNamespace(_shift_pose_scene=lambda *a: None)
    PaintApp._shift_layer_objects(fake, layer, 10, 20)
    check("link text untouched", (layer.texts[0]["x"], layer.texts[0]["y"]),
          (5, 6))


def test_shift_guides():
    doc = SimpleNamespace(guides=[("v", 10), ("h", 20), ("v", 5)])
    fake = SimpleNamespace(guides_shared=False)
    PaintApp._shift_guides(fake, doc, 10, 20, 100, 100)
    check("guides shifted and kept", doc.guides,
          [("v", 20), ("h", 40), ("v", 15)])

    shared = SimpleNamespace(guides_shared=True)
    doc = SimpleNamespace(guides=[("v", 10)])
    PaintApp._shift_guides(shared, doc, 10, 20, 100, 100)
    check("shared guides untouched", doc.guides, [("v", 10)])


if __name__ == "__main__":
    test_remap_keeps_existing_mask()
    test_remap_clears_empty_mask()
    test_shift_layer_objects()
    test_shift_layer_objects_skips_links()
    test_shift_guides()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
