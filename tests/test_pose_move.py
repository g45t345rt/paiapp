"""Regression: the Move tool must carry a layer's document-space pose figures.

A pose figure's ``x``/``y`` are recomputed from the shared scene anchor before
every render, so the compositor places the figure from the scene anchor (minus
``layer.offset``) while the canvas re-adds ``offset``. A move must therefore
shift both the offset *and* the scene anchor; shifting only the offset leaves the
figures pinned to the page and clips them as the layer window slides over it
(run: .venv\\Scripts\\python.exe tests\\test_pose_move.py).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import poses  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _move(layer, dx, dy):
    """Emulate what the Move tool does to a layer's poses scene."""
    scene = getattr(layer, "pose_scene", None)
    if isinstance(scene, dict):
        scene["x"] = float(scene.get("x", 0.0)) + dx
        scene["y"] = float(scene.get("y", 0.0)) + dy
        layer.invalidate_poses()
    ox, oy = layer.offset
    layer.offset = (ox + dx, oy + dy)


def test_moving_a_layer_moves_its_pose_figure():
    canvas = Canvas("c", (400, 400))
    layer = canvas.add_layer(name="p")
    layer.poses.append(poses.new(160, 220, scale=160, body="male"))
    layer.invalidate_poses()
    before = canvas.composite_transparent().getchannel("A").getbbox()
    _move(layer, 30, 10)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("figure travels with the layer",
          (after[0] - before[0], after[1] - before[1]), (30, 10))


def test_moving_does_not_clip_pose_art():
    canvas = Canvas("c", (400, 400))
    layer = canvas.add_layer(name="p")
    layer.poses.append(poses.new(200, 200, scale=160, body="male"))
    layer.invalidate_poses()
    before = canvas.composite_transparent().getchannel("A").getbbox()
    _move(layer, 20, 20)
    after = canvas.composite_transparent().getchannel("A").getbbox()
    check("width preserved", after[2] - after[0], before[2] - before[0])
    check("height preserved", after[3] - after[1], before[3] - before[1])


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
