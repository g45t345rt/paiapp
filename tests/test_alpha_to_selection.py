"""Regression: "Alpha to Selection" must keep the real outline after a transform.

The transform tool grows the layer's buffer to cover the canvas, so a resized
shape can have a boundary longer than ``selection.MAX_EDGES``. That used to make
``mask_edges`` give up and the marching ants fall back to the selection's
bounding box, which for a scaled-up layer covers (almost) the whole layer - so
"Alpha to Selection" looked like it selected everything. The outline is now
traced from the mask itself when the segment list is too large, and a mask
selection is drawn from its boundary rather than its polygon.

Run: .venv\\Scripts\\python.exe tests\\test_alpha_to_selection.py
"""

import os
import sys

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import gizmo, selection  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _View:
    def __init__(self):
        self.selection = None

    def set_selection(self, value):
        self.selection = value


class FakeApp:
    def __init__(self, layer, size):
        self.layer = layer
        self.current = type("Canvas", (), {"size": size})()
        self._view = _View()
        self.tools = type("Tools", (), {"select": lambda _self, _n: None})()
        self.result = None

    @property
    def area(self):
        return type("Area", (), {"current_view": lambda _self: self._view})()

    def active_layer(self):
        return self.layer

    def set_result(self, text, kind=None):
        self.result = text


def _shape_layer(size, box):
    from paintapp.document import Layer
    layer = Layer("shape", size)
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse(box, fill=(255, 255, 255, 255))
    layer.set_image(image)
    return layer


def _alpha_to_selection(layer, size):
    app = FakeApp(layer, size)
    from paintapp.app import PaintApp
    PaintApp.alpha_to_selection(app)
    return app._view.selection


def _drag_se_to(layer, size, target):
    app = FakeApp(layer, size)
    instance = gizmo.TransformGizmo()
    if not instance.begin(app, None):
        return False
    se = instance._handle_points()["se"]
    instance.start("se", se[0], se[1])
    instance.update(float(target[0]), float(target[1]))
    instance.finish()
    return True


def test_alpha_selection_region_matches_the_shape():
    size = (256, 256)
    layer = _shape_layer(size, (80, 80, 160, 160))
    sel = _alpha_to_selection(layer, size)
    check("selection is a mask selection", sel.masked, True)
    check("region is the shape, not the canvas", sel.bbox, (80, 80, 161, 161))
    check("corner is outside the ellipse", sel.contains(80, 80), False)
    check("centre is inside the ellipse", sel.contains(120, 120), True)


def test_outline_follows_the_shape_after_a_large_resize():
    # A canvas large enough that the resized boundary exceeds MAX_EDGES, so the
    # segment list is capped and the outline must come from the mask itself.
    size = (4096, 4096)
    layer = _shape_layer(size, (1700, 1700, 2400, 2400))
    check("resize applied", _drag_se_to(layer, size, (4096, 4096)), True)
    sel = _alpha_to_selection(layer, size)
    check("selection is a mask selection", sel.masked, True)
    check("segment list capped", sel.edges, None)
    boundary = np.asarray(sel.boundary_image().getchannel("A")) > 0
    left, top, right, bottom = sel.bbox
    # The old bug drew the bounding-box rectangle: its top-left corner and the
    # rectangle edges would be inked. The traced ellipse leaves the corner and
    # the bbox mid-edges empty and only inks a thin outline.
    check("bbox corner is not inked", bool(boundary[top, left]), False)
    check("bbox quarter-edge is not inked",
          bool(boundary[top, left + (right - left) // 4]), False)
    check("outline is thin, not a filled box", bool(boundary.mean() < 0.1), True)


def test_polygon_marquee_is_unaffected():
    sel = selection.Selection.rectangle((256, 256), (40, 40, 120, 120))
    check("marquee is not a mask selection", sel.masked, False)
    check("marquee keeps its polygon", len(sel.outline()), 4)
    check("moved marquee stays a polygon", sel.moved(5, 5).masked, False)
    lasso = selection.Selection.lasso((256, 256), [(10, 10), (100, 10), (100, 100)])
    check("lasso is not a mask selection", lasso.masked, False)


def test_mask_ops_stay_mask_selections():
    size = (256, 256)
    layer = _shape_layer(size, (80, 80, 160, 160))
    sel = _alpha_to_selection(layer, size)
    for name, result in (
        ("inverted", sel.inverted()),
        ("grown", sel.grown(2)),
        ("shrunk", sel.shrunk(2)),
        ("feathered", sel.feathered(2)),
        ("moved", sel.moved(5, 5)),
        ("transformed", sel.transformed((1, 0, 0, 1, 5, 5))),
    ):
        check(f"{name} stays a mask selection", result.masked, True)


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
