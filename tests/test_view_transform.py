"""Checks the non-destructive view transform (rotate/flip) math.

Run: .venv\\Scripts\\python.exe tests\\test_view_transform.py

The coordinate maths in ``CanvasView`` is exercised on an uninitialised
instance (no Tk root), so the document<->canvas mapping, its inverse, and the
rigid matrix can be checked without a display. These are the pieces the raster
warp and every overlay wrapper rely on.
"""

import math
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.canvasarea import CanvasView  # noqa: E402

FAILED = []


def check(name, got, want, tol=1e-6):
    ok = got == want
    if isinstance(got, float) or isinstance(want, float):
        ok = abs(got - want) <= tol
    if not ok:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_close(name, got, want, tol=1e-6):
    if any(abs(a - b) > tol for a, b in zip(got, want)):
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def view(angle=0.0, flip=False, flip_v=False, zoom=1.0, pivot=None,
         width=100, height=80):
    v = CanvasView.__new__(CanvasView)
    v._layout = None
    v.view_angle = angle
    v.view_flip = flip
    v.view_flip_v = flip_v
    v.zoom = zoom
    v._view_pivot = pivot
    v.doc = SimpleNamespace(width=width, height=height)
    return v


def test_identity_without_transform():
    v = view(zoom=2.0)
    check("transformed off", v.transformed, False)
    check_close("doc to view scales", v._doc_to_view(3.0, 4.0), (6.0, 8.0))
    check_close("doc to canvas scales", v._doc_to_canvas(3.0, 4.0), (6.0, 8.0))
    check_close("view to doc unscales", v._view_to_doc_float(6.0, 8.0), (3.0, 4.0))


def test_doc_to_view_stays_untransformed():
    # The overlay draw wrappers apply the view transform, so ``_doc_to_view``
    # must stay canonical; otherwise overlays get rotated twice (the border bug).
    v = view(angle=1.0, flip=True, zoom=2.0, pivot=(20.0, 15.0))
    check_close("doc to view is just scaled", v._doc_to_view(3.0, 4.0), (6.0, 8.0))
    check_close(
        "doc to canvas applies the transform",
        v._doc_to_canvas(3.0, 4.0),
        v._xform_pt(6.0, 8.0),
    )


def test_round_trip_rotation_and_flip():
    for angle in (0.3, math.pi / 2, 2.4, -1.1):
        for flip in (False, True):
            for flip_v in (False, True):
                v = view(angle=angle, flip=flip, flip_v=flip_v, zoom=1.7,
                         pivot=(20.0, 15.0))
                for point in ((0.0, 0.0), (40.0, 5.0), (-10.0, 33.0)):
                    canvas = v._doc_to_canvas(*point)
                    back = v._view_to_doc_float(*canvas)
                    check_close(
                        f"round trip a={angle:.2f} flip={flip} "
                        f"flip_v={flip_v} p={point}", back, point,
                    )


def test_flip_mirrors_about_pivot():
    v = view(flip=True, pivot=(10.0, 10.0))
    check_close("flip mirrors x", v._doc_to_canvas(0.0, 10.0), (20.0, 10.0))
    check_close("flip keeps y", v._doc_to_canvas(4.0, 3.0), (16.0, 3.0))


def test_vertical_flip_mirrors_about_pivot():
    v = view(flip_v=True, pivot=(10.0, 10.0))
    check_close("flip_v mirrors y", v._doc_to_canvas(10.0, 0.0), (10.0, 20.0))
    check_close("flip_v keeps x", v._doc_to_canvas(4.0, 3.0), (4.0, 17.0))


def test_both_flips_equal_half_turn():
    # Two independent mirrors compose to a 180 degree turn (det +1), exactly as
    # GIMP's both-checked Flip Horizontally/Vertically state.
    v = view(flip=True, flip_v=True, pivot=(10.0, 10.0))
    check_close("both flips turn 180", v._doc_to_canvas(0.0, 10.0), (20.0, 10.0))
    check_close("both flips turn 180 other axis",
                v._doc_to_canvas(10.0, 0.0), (10.0, 20.0))
    check_close("a point off both axes", v._doc_to_canvas(3.0, 4.0), (17.0, 16.0))


def test_rotation_is_clockwise_on_screen():
    # Screen y points down, so the +x axis turning onto +y reads as clockwise.
    v = view(angle=math.pi / 2, pivot=(0.0, 0.0))
    check_close("quarter turn clockwise", v._doc_to_canvas(10.0, 0.0), (0.0, 10.0))


def test_matrix_is_orthonormal_and_transpose_inverse():
    for flip, flip_v, det in ((True, False, -1.0), (False, True, -1.0),
                              (True, True, 1.0)):
        v = view(angle=1.234, flip=flip, flip_v=flip_v, pivot=(5.0, 6.0))
        m00, m01, m10, m11, i00, i01, i10, i11 = v._matrix()
        check(f"det flip={flip} flip_v={flip_v}",
              round(m00 * m11 - m01 * m10, 6), det)
        # M @ M^-1 == I.
        check(f"inverse 00 flip={flip} flip_v={flip_v}",
              round(m00 * i00 + m01 * i10, 6), 1.0)
        check(f"inverse 01 flip={flip} flip_v={flip_v}",
              round(m00 * i01 + m01 * i11, 6), 0.0)
        check(f"inverse 10 flip={flip} flip_v={flip_v}",
              round(m10 * i00 + m11 * i10, 6), 0.0)
        check(f"inverse 11 flip={flip} flip_v={flip_v}",
              round(m10 * i01 + m11 * i11, 6), 1.0)


def test_doc_rect_bbox_rotates_square():
    v = view(angle=math.pi / 2, pivot=(0.0, 0.0))
    box = v._doc_rect_to_canvas_bbox(0.0, 0.0, 10.0, 20.0)
    # A 10x20 doc rect turned a quarter clockwise still spans 10x20.
    check_close("rotated bbox", box, (-20.0, 0.0, 0.0, 10.0))


def test_rotated_distinguishes_flip_from_rotation():
    # A flip keeps the document axes aligned, so the rulers still make sense
    # (mirrored) and must not be blanked; only a rotation hides them.
    check("identity is not rotated", view().rotated, False)
    check("flip only is not rotated", view(flip=True).rotated, False)
    check("vertical flip only is not rotated", view(flip_v=True).rotated, False)
    check("both flips are not rotated", view(flip=True, flip_v=True).rotated, False)
    check("rotation is rotated", view(angle=0.5).rotated, True)
    check("flip plus rotation is rotated", view(angle=0.5, flip=True).rotated, True)
    check("angle is transformed", view(flip=True).transformed, True)
    check("vertical flip is transformed", view(flip_v=True).transformed, True)


if __name__ == "__main__":
    test_identity_without_transform()
    test_doc_to_view_stays_untransformed()
    test_round_trip_rotation_and_flip()
    test_flip_mirrors_about_pivot()
    test_vertical_flip_mirrors_about_pivot()
    test_both_flips_equal_half_turn()
    test_rotation_is_clockwise_on_screen()
    test_matrix_is_orthonormal_and_transpose_inverse()
    test_doc_rect_bbox_rotates_square()
    test_rotated_distinguishes_flip_from_rotation()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
