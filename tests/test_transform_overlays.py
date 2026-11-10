"""Regression: the Transform tool must carry a layer's document-space overlays.

Path objects, vector strokes/fills, text and pose figures are stored in document
coordinates and drawn over the layer's pixels at composite time. The transform
gizmo resamples the base pixels, so before this the overlays stayed pinned to the
page and slid out of alignment (the base image moved, they did not). The gizmo
now maps their geometry through the same document-space delta, so they travel
with the pixels - and a vector-only layer, whose base image is empty, can be
transformed at all (run: .venv\\Scripts\\python.exe tests\\test_transform_overlays.py).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import gizmo, path, poses, textrender, vector  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def approx(name, got, want, tol=1e-6):
    ok = all(abs(a - b) <= tol for a, b in zip(got, want))
    if not ok:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class FakeApp:
    def __init__(self, layer):
        self.layer = layer
        self.current = type("Canvas", (), {"size": layer.size})()

    def active_layer(self):
        return self.layer

    def _drop_undo(self):
        pass


def _square_spec(mode="fill"):
    outline = path.Path()
    outline.add(20, 20)
    outline.add(60, 20)
    outline.add(60, 60)
    outline.add(20, 60)
    outline.closed = True
    return path.from_path(
        outline, mode=mode, fill_color="#000000", stroke_color="#000000",
        width=4 if mode == "stroke" else 1, outline=0, feather=0, opacity=100,
        antialias=True, cap="round", join="round",
    )


def _begin(layer):
    original = gizmo.gpuimage.texture
    gizmo.gpuimage.texture = lambda *a, **k: None  # deterministic Pillow path
    app = FakeApp(layer)
    instance = gizmo.TransformGizmo()
    if not instance.begin(app, None):
        gizmo.gpuimage.texture = original
        return None, app, original
    return instance, app, original


def _center_doc(instance):
    w, h = instance.source.size
    return gizmo.apply(instance.matrix, w / 2.0, h / 2.0)


def _drag_move(layer, dx, dy):
    instance, app, original = _begin(layer)
    try:
        if instance is None:
            return False
        press = _center_doc(instance)
        instance.start("move", *press)
        instance.update(press[0] + dx, press[1] + dy)
        instance.finish()
        return True
    finally:
        gizmo.gpuimage.texture = original


def test_path_travels_with_the_transform():
    canvas = Canvas("c", (256, 256))
    layer = canvas.add_layer(name="p")
    layer.paths = [_square_spec()]
    layer.invalidate_paths()
    before = [list(a) for a in layer.paths[0]["anchors"]]
    check("vector-only layer can be transformed", _drag_move(layer, 30, 10), True)
    after = [list(a) for a in layer.paths[0]["anchors"]]
    for (bx, by), (ax, ay) in zip(before, after):
        approx("path anchor travels", (ax, ay), (bx + 30, by + 10))


def test_vector_stroke_travels_with_the_transform():
    canvas = Canvas("c", (256, 256))
    layer = canvas.add_layer(name="v", vector=True)
    layer.strokes = [vector.new_stroke([[20, 20], [60, 20]], 4, "#000000")]
    layer.invalidate_vectors()
    before = [list(p) for p in layer.strokes[0]["points"]]
    check("vector layer can be transformed", _drag_move(layer, 15, 25), True)
    after = [list(p) for p in layer.strokes[0]["points"]]
    for (bx, by), (ax, ay) in zip(before, after):
        approx("stroke point travels", (ax, ay), (bx + 15, by + 25))
    check("stroke width is untouched by a translation", layer.strokes[0]["width"], 4)


def test_text_travels_with_the_transform():
    canvas = Canvas("c", (256, 256))
    layer = canvas.add_layer(name="t")
    spec = textrender.new_spec("Hi", 40, 40, size=24)
    layer.texts = [spec]
    layer.invalidate_texts()
    before = (spec["x"], spec["y"])
    check("text layer can be transformed", _drag_move(layer, 12, 8), True)
    approx("text anchor travels", (spec["x"], spec["y"]),
           (before[0] + 12, before[1] + 8))
    check("text stays editable", bool(layer.texts), True)


def test_pose_figure_travels_with_the_transform():
    canvas = Canvas("c", (400, 400))
    layer = canvas.add_layer(name="p")
    layer.poses = [poses.new(200, 200, scale=160, body="male")]
    layer.invalidate_poses()
    scene = layer.pose_scene
    before = (scene["x"], scene["y"])
    check("pose layer can be transformed", _drag_move(layer, 30, 10), True)
    approx("pose scene anchor travels", (scene["x"], scene["y"]),
           (before[0] + 30, before[1] + 10))
    # The derived per-figure positions must follow on the next render.
    poses.sync_scene(layer.poses, scene)
    approx("figure follows the scene", (layer.poses[0]["x"], layer.poses[0]["y"]),
           (230.0, 210.0))


def test_scale_transform_scales_path_geometry():
    canvas = Canvas("c", (256, 256))
    layer = canvas.add_layer(name="p")
    layer.paths = [_square_spec(mode="stroke")]
    layer.invalidate_paths()
    before = [list(a) for a in layer.paths[0]["anchors"]]
    instance, app, original = _begin(layer)
    try:
        if instance is None:
            check("session for scaling", False, True)
            return
        w, h = instance.source.size
        base = instance.matrix
        pivot = gizmo.apply(base, 0.0, 0.0)      # "se" scales about "nw"
        instance.start("se", *gizmo.apply(base, w, h))
        instance.update(*gizmo.apply(base, 2 * w, 2 * h))
        instance.finish()
    finally:
        gizmo.gpuimage.texture = original
    after = [list(a) for a in layer.paths[0]["anchors"]]
    for (bx, by), (ax, ay) in zip(before, after):
        approx("path anchor scales about the pivot", (ax, ay),
               (pivot[0] + 2 * (bx - pivot[0]), pivot[1] + 2 * (by - pivot[1])))
    check("stroke width scales with the transform", layer.paths[0]["width"], 8)


def test_cancel_restores_the_overlay():
    canvas = Canvas("c", (256, 256))
    layer = canvas.add_layer(name="p")
    layer.paths = [_square_spec()]
    layer.invalidate_paths()
    before = [list(a) for a in layer.paths[0]["anchors"]]
    instance, app, original = _begin(layer)
    try:
        if instance is None:
            check("session for cancel", False, True)
            return
        press = _center_doc(instance)
        instance.start("move", *press)
        instance.update(press[0] + 40, press[1] + 40)
        instance.cancel_drag(app)
    finally:
        gizmo.gpuimage.texture = original
    after = [list(a) for a in layer.paths[0]["anchors"]]
    check("cancel restores the path geometry", after, before)


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
