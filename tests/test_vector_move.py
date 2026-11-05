"""Regression: the Move tool must carry a vector layer's document-space art.

Vector strokes and fills are stored in document coordinates, and the compositor
places them at ``points - layer.offset`` while the canvas re-adds ``offset``.
So a move has to shift both the offset *and* the geometry; shifting only the
offset leaves the art pinned to the page and clips it as the layer window slides
over it (run: .venv\\Scripts\\python.exe tests\\test_vector_move.py).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import vector  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _square(x0, y0, x1, y1):
    points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return {
        "points": points,
        "handles_in": [list(p) for p in points],
        "handles_out": [list(p) for p in points],
    }


def _move(layer, dx, dy):
    """Emulate what the Move tool does to a vector layer."""
    snapshots = [
        (spec, vector.item_snapshot(spec))
        for spec in list(layer.strokes) + list(layer.fills)
    ]
    ox, oy = layer.offset
    layer.offset = (ox + dx, oy + dy)
    for spec, snapshot in snapshots:
        vector.item_restore(spec, snapshot)
        vector.item_translate(spec, dx, dy)
    layer.invalidate_vectors()


def test_moving_a_fill_layer_moves_the_art():
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="v", vector=True)
    layer.fills = [vector.new_fill([_square(20, 20, 60, 60)], "#000000")]
    before = canvas.composite_transparent().getchannel("A").getbbox()
    _move(layer, 30, 10)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("fill starts where drawn", before, (20, 20, 60, 60))
    check("fill travels with the layer", after, (50, 30, 90, 70))


def test_moving_a_stroke_layer_moves_the_art():
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="v", vector=True)
    layer.strokes = [vector.new_stroke([[20, 20], [60, 20]], 4, "#000000")]
    before = canvas.composite_transparent().getchannel("A").getbbox()
    _move(layer, 10, 25)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("stroke moves with the layer",
          (after[0] - before[0], after[1] - before[1]), (10, 25))


def test_moving_does_not_clip_vector_art():
    canvas = Canvas("c", (128, 128))
    layer = canvas.add_layer(name="v", vector=True)
    layer.fills = [vector.new_fill([_square(50, 50, 90, 90)], "#000000")]
    _move(layer, 20, 20)
    # The full width/height of the shape survives the move (no buffer clipping).
    left, top, right, bottom = canvas.composite_transparent().getchannel("A").getbbox()
    check("width preserved", right - left, 40)
    check("height preserved", bottom - top, 40)


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
