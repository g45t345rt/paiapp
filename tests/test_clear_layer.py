"""Checks that clearing a layer wipes off-canvas pixels too.

Run: .venv\\Scripts\\python.exe tests\\test_clear_layer.py

``PaintApp.clear`` is called unbound with a minimal stand-in coordinator, so
the whole-buffer clear is exercised without a Tk root.
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import app as app_module  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_app(doc):
    return SimpleNamespace(
        current=doc,
        root=None,
        active_layer=doc.active_layer,
        drawing=SimpleNamespace(commit_text=lambda: None, finish_path=lambda: None),
        _active_layer_locked=lambda: False,
        _push_undo=lambda *_args, **_kwargs: None,
        _refresh_layer_list=lambda: None,
        set_result=lambda *_args, **_kwargs: None,
        render=lambda: None,
    )


def run_clear(doc):
    original = app_module.dialogs.ask_yes_no
    app_module.dialogs.ask_yes_no = lambda *_args, **_kwargs: True
    try:
        PaintApp.clear(fake_app(doc))
    finally:
        app_module.dialogs.ask_yes_no = original


def test_clear_wipes_off_canvas_pixels():
    doc = Canvas("c", (64, 64))
    doc.add_layer(name="paint")
    layer = doc.active_layer()
    # A moved layer that has been painted keeps a buffer larger than the canvas
    # with ``offset`` reset to ``(0, 0)``; ``bake_offsets`` then leaves it alone.
    layer.offset = (20, 20)
    doc.align_for_painting(layer)
    layer.draw.point((5, 5), fill=(255, 0, 0, 255))
    layer.draw.point((70, 70), fill=(0, 255, 0, 255))
    check("buffer larger than canvas", layer.image.size, (84, 84))
    check("offset baked to zero", layer.offset, (0, 0))

    run_clear(doc)

    check("off-canvas pixel cleared", layer.image.getpixel((70, 70))[3], 0)
    check("on-canvas pixel cleared", layer.image.getpixel((5, 5))[3], 0)
    check("layer fully transparent", layer.image.getbbox(), None)


def test_clear_canvas_sized_layer():
    doc = Canvas("c", (48, 48))
    doc.add_layer(name="plain")
    layer = doc.active_layer()
    layer.draw.point((10, 10), fill=(0, 0, 255, 255))

    run_clear(doc)

    check("plain layer cleared", layer.image.getbbox(), None)


if __name__ == "__main__":
    test_clear_wipes_off_canvas_pixels()
    test_clear_canvas_sized_layer()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
