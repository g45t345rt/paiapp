"""Tests for the pointer stabiliser styles
(run: .venv\\Scripts\\python.exe tests\\test_stabilizer.py)."""

import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.drawing import DrawingController, STABILIZE_MODES  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _Tools:
    mask_edit = False
    inpaint_mask_edit = False
    eraser = False


class _App:
    pass


def controller(style, stabilize):
    ctrl = DrawingController.__new__(DrawingController)
    tools = _Tools()
    tools.brush_stabilize = stabilize
    tools.brush_stabilize_mode = style
    tools.eraser_stabilize = stabilize
    tools.eraser_stabilize_mode = style
    app = _App()
    app.tools = tools
    ctrl.app = app
    ctrl._smooth_x = ctrl._smooth_y = None
    ctrl._raw_x = ctrl._raw_y = None
    ctrl._stable_history = []
    return ctrl


def feed(ctrl, points):
    return [ctrl._stable_point(x, y, "brush") for x, y in points]


def test_modes_present():
    check("three stabiliser styles", sorted(STABILIZE_MODES), ["simple", "velocity", "weighted"])


def test_zero_strength_follows_the_pointer():
    points = [(0.0, 0.0), (10.0, 5.0), (25.0, 9.0)]
    out = feed(controller("simple", 0), points)
    for got, want in zip(out, points):
        check("zero smooth is the raw pointer", (got[0], got[1]), want)


def test_weighted_resists_a_spike():
    # A steady line with one outlying sample: the moving average should move
    # toward it less than the simple ease does, i.e. it rejects the outlier.
    points = [(0.0, 0.0)] * 6 + [(100.0, 0.0)] + [(0.0, 0.0)] * 6
    simple = feed(controller("simple", 80), points)
    weighted = feed(controller("weighted", 80), points)
    check("weighted resists a spike more than simple",
          max(p[0] for p in weighted) < max(p[0] for p in simple), True)


def test_velocity_keeps_up_with_fast_motion():
    fast = [(i * 40.0, 0.0) for i in range(7)]
    simple = feed(controller("simple", 90), fast)
    velocity = feed(controller("velocity", 90), fast)
    raw_x = fast[-1][0]
    check("velocity trails the pointer less than simple",
          abs(velocity[-1][0] - raw_x) < abs(simple[-1][0] - raw_x), True)



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
