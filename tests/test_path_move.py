"""Regression: the Move tool must carry a layer's document-space paths.

Path objects are stored in document coordinates, and the compositor places them
at ``anchors - layer.offset`` while the canvas re-adds ``offset``. So a move has
to shift both the offset *and* the path geometry; shifting only the offset leaves
the path pinned to the page and clips it as the layer window slides over it
(run: .venv\\Scripts\\python.exe tests\\test_path_move.py).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import ImageChops  # noqa: E402

from paintapp import path  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _spec():
    outline = path.Path()
    outline.add(20, 20)
    outline.add(60, 20)
    outline.add(60, 60)
    outline.add(20, 60)
    outline.closed = True
    return path.from_path(
        outline, mode="fill", fill_color="#000000", stroke_color="#000000",
        width=1, outline=0, feather=0, opacity=100, antialias=True,
        cap="round", join="round",
    )


def _move(layer, dx, dy):
    """Emulate what the Move tool does to a layer's paths."""
    snapshots = [(spec, path.clone(spec)) for spec in list(layer.paths)]
    ox, oy = layer.offset
    layer.offset = (ox + dx, oy + dy)
    for spec, snapshot in snapshots:
        path.apply_geometry(spec, path.to_path(snapshot))
        path.translate(spec, dx, dy)
    layer.invalidate_paths()


def test_moving_a_layer_moves_its_paths():
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="p")
    layer.paths = [_spec()]
    layer.invalidate_paths()
    before = canvas.composite_transparent().getchannel("A").getbbox()
    _move(layer, 30, 10)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("path travels with the layer",
          (after[0] - before[0], after[1] - before[1]), (30, 10))


def test_moving_does_not_clip_path_art():
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="p")
    layer.paths = [_spec()]
    layer.invalidate_paths()
    before = canvas.composite_transparent().getchannel("A").getbbox()
    _move(layer, 20, 20)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("width preserved", after[2] - after[0], before[2] - before[0])
    check("height preserved", after[3] - after[1], before[3] - before[1])


def _drag_frame(layer, dx, dy):
    """One Move-tool frame: slide ``offset`` and record the delta, specs untouched."""
    ox, oy = layer.offset
    layer.offset = (ox + dx, oy + dy)
    layer._content_delta = (dx, dy)


def test_drag_delta_slides_path_without_rerendering():
    """A Move drag must slide the layer's placement without re-rasterising its
    path overlay: ``_content_delta`` keeps ``content_offset`` fixed, so the
    layer's own image is unchanged all drag (the path-move lag fix)."""
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="p")
    layer.paths = [_spec()]
    layer.invalidate_paths()
    layer.offset = (10, 5)
    still = layer.composited_image()
    cached = layer._path_cache
    before = canvas.composite_transparent().getchannel("A").getbbox()

    _drag_frame(layer, 30, 10)

    during = layer.composited_image()
    check("path overlay not re-rasterised during a drag",
          layer._path_cache is cached, True)
    check("layer content unchanged during a drag",
          ImageChops.difference(during.convert("RGBA"),
                                still.convert("RGBA")).getbbox(), None)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("layer placement slides with the drag",
          (after[0] - before[0], after[1] - before[1]), (30, 10))


def test_baking_drag_delta_keeps_the_frame():
    """Releasing (bake the delta into the spec, clear it) must not shift the
    frame the drag was showing."""
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="p")
    spec = _spec()
    layer.paths = [spec]
    layer.invalidate_paths()
    layer.offset = (10, 5)
    _drag_frame(layer, 30, 10)
    during = layer.composited_image()

    # Bake, as ``_end_move`` does: shift the spec by the delta (the offset is
    # already at its dragged position) and clear the delta.
    path.translate(spec, 30, 10)
    layer.invalidate_paths()
    layer._content_delta = (0, 0)
    baked = layer.composited_image()
    check("baked frame matches the dragged frame",
          ImageChops.difference(baked.convert("RGBA"),
                                during.convert("RGBA")).getbbox(), None)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")

if __name__ == "__main__":
    main()
