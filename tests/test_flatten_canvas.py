"""Checks the Flatten Canvas operation.

Run: .venv\\Scripts\\python.exe tests\\test_flatten_canvas.py

``Canvas.flatten`` is exercised directly for the compositing semantics, and
``PaintApp.flatten_canvas`` unbound with a minimal stand-in coordinator for the
lock guard and undo/message bookkeeping.
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
    results = []
    pushes = []
    return SimpleNamespace(
        current=doc,
        root=None,
        drawing=SimpleNamespace(
            commit_text=lambda: None,
            finish_path=lambda: None,
            sync_transform=lambda: None,
        ),
        _push_undo=lambda label="Edit": pushes.append(label),
        _unlink_links_to=lambda *_args, **_kwargs: None,
        _refresh_layer_list=lambda *_args, **_kwargs: None,
        _transform_mode_active=lambda: False,
        set_result=lambda text, kind=None: results.append(text),
        render=lambda: None,
    ), pushes, results


def run_flatten(app, answer=True):
    original = app_module.dialogs.ask_yes_no
    app_module.dialogs.ask_yes_no = lambda *_args, **_kwargs: answer
    try:
        PaintApp.flatten_canvas(app)
    finally:
        app_module.dialogs.ask_yes_no = original


def test_flatten_composites_all_visible_content():
    doc = Canvas("c", (40, 30))
    red = doc.add_layer(name="red")
    red.draw.point((5, 5), fill=(255, 0, 0, 255))
    blue = doc.add_layer(name="blue")
    blue.draw.point((10, 10), fill=(0, 0, 255, 255))
    hidden = doc.add_layer(name="hidden")
    hidden.draw.point((20, 20), fill=(0, 255, 0, 255))
    hidden.visible = False
    group = doc.new_group(members=[blue])
    group.opacity = 0.5

    layer = doc.flatten()

    check("one layer remains", len(doc.layers), 1)
    check("groups dropped", doc.groups, [])
    check("active layer reset", doc.active, 0)
    check("flattened name", layer.name, "Flattened")
    check("flattened at origin", layer.offset, (0, 0))
    check("flattened canvas-sized", layer.size, (40, 30))
    check("red kept", layer.image.getpixel((5, 5)), (255, 0, 0, 255))
    check("group opacity baked", layer.image.getpixel((10, 10))[:3], (0, 0, 255))
    check("group opacity alpha",
          layer.image.getpixel((10, 10))[3] in (127, 128), True)
    check("hidden layer dropped", layer.image.getpixel((20, 20))[3], 0)


def test_flatten_app_pushes_undo_and_reports():
    doc = Canvas("c", (16, 16))
    doc.add_layer().draw.point((1, 1), fill=(255, 0, 0, 255))
    doc.add_layer().draw.point((2, 2), fill=(0, 255, 0, 255))
    app, pushes, results = fake_app(doc)

    run_flatten(app)

    check("undo pushed", pushes, ["Flatten canvas"])
    check("single layer", len(doc.layers), 1)
    check("active reset", app.active, 0)
    check("status reported", results[-1], "Flattened 2 layer(s) into one.")
    check("first pixel kept", doc.layers[0].image.getpixel((1, 1))[3], 255)
    check("second pixel kept", doc.layers[0].image.getpixel((2, 2))[3], 255)


def test_flatten_refuses_locked_visible_layer():
    doc = Canvas("c", (8, 8))
    doc.add_layer(name="base")
    locked = doc.add_layer(name="locked")
    locked.locked = True
    app, pushes, results = fake_app(doc)

    run_flatten(app)

    check("nothing pushed", pushes, [])
    check("stack untouched", len(doc.layers), 2)
    check("lock reported",
          results[-1], "Cannot flatten canvas: layer 'locked' is locked.")


def test_flatten_can_be_cancelled():
    doc = Canvas("c", (8, 8))
    doc.add_layer(name="base")
    doc.add_layer(name="top")
    app, pushes, results = fake_app(doc)

    run_flatten(app, answer=False)

    check("cancel pushes nothing", pushes, [])
    check("cancel keeps the stack", len(doc.layers), 2)
    check("cancel reports nothing", results, [])


if __name__ == "__main__":
    test_flatten_composites_all_visible_content()
    test_flatten_app_pushes_undo_and_reports()
    test_flatten_refuses_locked_visible_layer()
    test_flatten_can_be_cancelled()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
