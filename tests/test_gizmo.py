"""Unit tests for the transform gizmo (run: .venv\\Scripts\\python.exe tests\\test_gizmo.py)."""

import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import gizmo  # noqa: E402
from paintapp.document import Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class FakeApp:
    """The little of the app a gizmo reads: the active layer and its canvas."""

    def __init__(self, layer, size):
        self.layer = layer
        self.current = type("Canvas", (), {"size": size})()

    def active_layer(self):
        return self.layer

    def _drop_undo(self):
        """The one app call a cancelled drag makes."""


def photo_layer(width=320, height=240, seed=4, mask=False):
    """An opaque RGBA layer of smooth content, the way a drag resamples it."""
    yy, xx = np.mgrid[0:height, 0:width].astype("float32")
    r = 120 + 90 * np.sin(xx / 37.0) * np.cos(yy / 41.0)
    g = 140 + 80 * np.sin((xx + yy) / 53.0)
    b = 100 + 70 * np.cos(xx / 23.0)
    pixels = np.clip(np.stack([r, g, b, np.full((height, width), 255.0)],
                              axis=2), 0, 255).astype(np.uint8)
    layer = Layer("photos", (width, height))
    layer.set_image(Image.fromarray(pixels))
    if mask:
        layer.mask = Image.fromarray(
            np.clip(np.full((height, width), 0.0) + xx, 0, 255).astype(np.uint8))
    return layer


def run_drag(handle, start_xy, end_xy, layer=None, gpu=True, uniform=False):
    """Begin a session, drag ``handle``, release; return the gizmo and layer."""
    layer = layer if layer is not None else photo_layer()
    app = FakeApp(layer, layer.size)
    original_texture = gizmo.gpuimage.texture
    if not gpu:
        gizmo.gpuimage.texture = lambda *a, **k: None
    try:
        instance = gizmo.TransformGizmo()
        if not instance.begin(app, None):
            return instance, layer
        instance.start(handle, *start_xy)
        instance.update(end_xy[0], end_xy[1], uniform)
        instance.finish()
    finally:
        gizmo.gpuimage.texture = original_texture
    return instance, layer


def test_session_needs_content():
    layer = Layer("empty", (64, 48))
    check("empty layer cannot be transformed",
          gizmo.TransformGizmo().begin(FakeApp(layer, layer.size), None), False)


class FakeView:
    """The bit of a canvas view the gizmo talks to: it stores the geometry."""

    def __init__(self):
        self.geometry = "unset"

    def set_gizmo(self, geometry):
        self.geometry = geometry


def test_failed_begin_drops_the_previous_session():
    # Switching to a hidden/locked layer (a link's master, say) while the
    # transform tool is on used to leave the previous layer's frame drawn with
    # no live gizmo behind it, so the overlay looked "still active".
    view = FakeView()
    instance = gizmo.TransformGizmo()
    check("session for the first layer",
          instance.begin(FakeApp(photo_layer(), (320, 240)), view), True)
    instance.clear()
    check("a cleared session releases the view overlay", view.geometry, None)

    hidden = photo_layer()
    hidden.visible = False
    check("a hidden layer cannot start a session",
          instance.begin(FakeApp(hidden, hidden.size), view), False)
    check("a failed begin is not reported as active", instance.active, False)
    check("a failed begin keeps no layer", instance.layer, None)


def test_switching_view_releases_the_previous_overlay():
    # Each canvas view draws its own copy of the frame while the session lives
    # on the controller. Moving the session to another view (a canvas switch)
    # must clear the frame on the view it left, or that canvas keeps redrawing
    # a stale transform frame when the user switches back.
    first = FakeView()
    second = FakeView()
    instance = gizmo.TransformGizmo()
    check("session on the first view",
          instance.begin(FakeApp(photo_layer(), (320, 240)), first), True)
    first.set_gizmo("frame")
    check("the first view draws the frame", first.geometry, "frame")
    check("session moves to the second view",
          instance.begin(FakeApp(photo_layer(), (320, 240)), second), True)
    check("the view it left drops its frame", first.geometry, None)
    instance.clear()
    check("the current view drops its frame", second.geometry, None)


def test_switching_view_ends_the_previous_drag_preview():
    # A live drag preview is stored on the session's layer. Beginning a session
    # on another canvas must clear it while the old layer is still current,
    # otherwise that canvas returns showing the drag-time snapshot forever.
    first_layer = photo_layer()
    second_layer = photo_layer(seed=5)
    instance = gizmo.TransformGizmo()
    check("session on the first layer",
          instance.begin(FakeApp(first_layer, first_layer.size), FakeView()), True)
    instance.start("se", 320.0, 240.0)
    instance.update(360.0, 270.0, draft=True)
    check("a live preview is painted on the first layer",
          first_layer._live_preview is not None, True)
    check("session moves to the second layer",
          instance.begin(FakeApp(second_layer, second_layer.size), FakeView()), True)
    check("the layer it left drops the live preview",
          first_layer._live_preview, None)


def test_destroyed_view_is_not_touched():
    # A project load or undo rebuilds the canvas tabs, destroying the widget the
    # gizmo last drew on. Moving away from it (or clearing) must not call into
    # the dead widget.
    class DeadView(FakeView):
        def winfo_exists(self):
            return False

        def set_gizmo(self, geometry):
            raise AssertionError("a destroyed view must not be drawn on")

    instance = gizmo.TransformGizmo()
    check("session on a destroyed view",
          instance.begin(FakeApp(photo_layer(), (320, 240)), DeadView()), True)
    check("moving to a live view skips the destroyed one",
          instance.begin(FakeApp(photo_layer(), (320, 240)), FakeView()), True)
    instance.view = DeadView()
    instance.clear()
    check("clear drops the destroyed view", instance.view, None)



def test_move_commits_the_source_shifted():
    # A drag at whole-pixel offsets samples the source at its own texel centres,
    # so the resample must reproduce the pixels it moved exactly - the check
    # that catches a half-pixel error in the shader's coordinate convention.
    # A move defers that resample to release (like the Move tool), so the pixels
    # are untouched mid-drag and only shift when the drag finishes.
    layer = photo_layer()
    source = np.asarray(layer.image)
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    check("session starts", instance.begin(app, None), True)
    instance.start("move", 40.0, 30.0)
    instance.update(60.0, 40.0)
    check("a move drag defers the resample",
          np.array_equal(np.asarray(layer.image), source), True)
    check("the drag delta is recorded", layer._content_delta, (20.0, 10.0))
    instance.finish()
    moved = np.asarray(layer.image)
    check("buffer spans the moved content",
          layer.image.size, (source.shape[1] + 20, source.shape[0] + 10))
    check("buffer offset", layer.offset, (0, 0))
    check("the drag moved the pixels exactly",
          np.array_equal(moved[10:, 20:], source), True)
    check("release clears the drag delta", layer._content_delta, (0, 0))


def test_release_does_not_resample_again():
    # The drag's own result is the transformed layer: releasing only ends the
    # drag, so a large layer pays for the interactive resamples and nothing more.
    instance, layer = run_drag("se", (320.0, 240.0), (480.0, 360.0))
    committed = np.asarray(layer.image).copy()
    check("release has nothing to finish", instance.finish(), None)
    check("release leaves the pixels alone",
          np.array_equal(np.asarray(layer.image), committed), True)


def test_gpu_scaling_matches_the_pillow_path():
    _, gpu_layer = run_drag("se", (320.0, 240.0), (480.0, 360.0))
    _, pillow_layer = run_drag("se", (320.0, 240.0), (480.0, 360.0), gpu=False)
    check("both paths agree on the buffer",
          (gpu_layer.image.size, gpu_layer.offset),
          (pillow_layer.image.size, pillow_layer.offset))
    difference = np.abs(np.asarray(gpu_layer.image, dtype="int16")
                        - np.asarray(pillow_layer.image, dtype="int16"))
    check("gpu scaling is within a few counts of Pillow",
          bool(difference.mean() < 2.0), True)


def test_scaled_result_scales_the_source():
    # Scaling the 320x240 source by 1.5 leaves a 480x360 frame of ink.
    _, layer = run_drag("se", (320.0, 240.0), (480.0, 360.0))
    ink = np.asarray(layer.image)[:, :, 3] > 0
    check("scaled frame width", int(ink.any(axis=0).sum()), 480)
    check("scaled frame height", int(ink.any(axis=1).sum()), 360)


def test_result_lands_where_the_pillow_path_lands():
    # Pillow's affine maps pixel *centres* through the matrix, so an output
    # pixel reads (o + 0.5) / scale - 0.5. Getting that half pixel wrong costs
    # nothing on smooth content and shifts texture by up to half a source
    # pixel, so scale a hard edge and require the GPU result to put it on the
    # same output pixel Pillow's own resample does.
    def stripe(use_gpu):
        width, height = 120, 90
        pixels = np.zeros((height, width, 4), dtype=np.uint8)
        pixels[:, :, 3] = 255
        pixels[:, 61, :3] = 255          # a one-pixel stripe, half way along
        layer = Layer("edge", (width, height))
        layer.set_image(Image.fromarray(pixels))
        app = FakeApp(layer, layer.size)
        instance = gizmo.TransformGizmo()
        original = gizmo.gpuimage.texture
        if not use_gpu:
            gizmo.gpuimage.texture = lambda *a, **k: None
        try:
            if not instance.begin(app, None):
                return None
            instance.start("se", float(width), float(height))
            instance.update(width * 3.0, height * 3.0)
        finally:
            gizmo.gpuimage.texture = original
        return np.asarray(layer.image)[:, :, 0].sum(axis=0)

    gpu = stripe(True)
    pillow = stripe(False)
    check("both paths scaled the stripe",
          (gpu is not None, pillow is not None), (True, True))
    check("gpu puts the stripe on Pillow's column",
          int(np.argmax(gpu)), int(np.argmax(pillow)))
    check("gpu keeps the stripe as sharp as Pillow",
          int(gpu.max()), int(pillow.max()))


def test_masked_layer_commits_the_mask():
    _, with_gl = run_drag("move", (40.0, 30.0), (60.0, 40.0),
                          layer=photo_layer(mask=True))
    _, without_gl = run_drag("move", (40.0, 30.0), (60.0, 40.0),
                             layer=photo_layer(mask=True), gpu=False)
    check("mask transformed with the pixels",
          np.array_equal(np.asarray(with_gl.mask), np.asarray(without_gl.mask)),
          True)
    check("mask follows the pixels", with_gl.mask.size, with_gl.image.size)


def test_source_is_uploaded_once_per_session():
    calls = []
    original = gizmo.gpuimage.texture

    def counting(image, *args, **kwargs):
        calls.append(image.size)
        return original(image, *args, **kwargs)

    layer = photo_layer()
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    gizmo.gpuimage.texture = counting
    try:
        if not instance.begin(app, None):
            check("session for upload count", False, True)
            return
        instance.start("se", 320.0, 240.0)
        instance.update(360.0, 270.0)
        instance.update(400.0, 300.0)
        instance.update(440.0, 330.0)
        instance.cancel_drag(app)
        uploaded = len(calls)
        instance.clear()
        check("one upload serves the whole drag", uploaded, 1)
        check("clear drops the upload", instance._gpu, None)
    finally:
        gizmo.gpuimage.texture = original


def test_cancel_restores_the_pixels():
    layer = photo_layer()
    before = np.asarray(layer.image).copy()
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    check("session for cancel", instance.begin(app, None), True)
    instance.start("move", 40.0, 30.0)
    instance.update(90.0, 70.0)
    instance.cancel_drag(app)
    check("cancel restores the source", np.array_equal(np.asarray(layer.image),
                                                       before), True)


def test_rotation_origin_is_configurable():
    layer = photo_layer(120, 90)
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    check("session for pivot", instance.begin(app, None), True)
    check("default pivot is the frame centre",
          instance._rotation_pivot_local(), (60.0, 45.0))
    instance.start("pivot", 60.0, 45.0)
    instance.update(0.0, 0.0)
    instance.finish()
    check("pivot moved to the grabbed point",
          instance.rot_pivot_local, (0.0, 0.0))
    check("pivot is hittable where it was dropped",
          instance.hit(0.0, 0.0, 4.0), "pivot")
    instance.start("rot", 60.0, -20.0)
    instance.update(20.0, 60.0)  # exactly 90 degrees about the origin
    pinned = gizmo.apply(instance.matrix, 0.0, 0.0)
    check("rotating keeps the custom origin fixed",
          (round(pinned[0], 3), round(pinned[1], 3)), (0.0, 0.0))
    corner = gizmo.apply(instance.matrix, 60.0, 45.0)
    check("a point rotates about the custom origin",
          (round(corner[0], 3), round(corner[1], 3)), (-45.0, 60.0))


def test_rotation_origin_snaps_to_anchors():
    layer = photo_layer(120, 90)
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    check("session for snapping", instance.begin(app, None), True)
    instance.start("pivot", 60.0, 45.0)
    instance.update(3.0, 3.0)
    check("origin snaps to the top-left corner",
          instance.rot_pivot_local, (0.0, 0.0))
    instance.finish()
    instance.start("pivot", 0.0, 0.0)
    instance.update(118.0, 88.0)
    check("origin snaps to the bottom-right corner",
          instance.rot_pivot_local, (120.0, 90.0))
    instance.finish()
    instance.start("pivot", 120.0, 90.0)
    instance.update(60.0, 45.0)
    check("origin snaps to the centre",
          instance.rot_pivot_local, (60.0, 45.0))
    instance.finish()
    instance.reset_rotation_pivot()
    check("reset returns to the default centre",
          instance._rotation_pivot_local(), (60.0, 45.0))


def _rotation_angle(instance):
    """The gizmo matrix's rotation in degrees (the linear part is a rotation)."""
    return math.degrees(math.atan2(instance.matrix[1], instance.matrix[0]))


def test_rotation_snap_step_is_configurable():
    # The Transform tool's rotation snap step replaces the old hard-coded 15
    # degrees; a step of 0 turns snapping off entirely.
    def drag_to(angle_deg, snap):
        layer = photo_layer(120, 90)
        app = FakeApp(layer, layer.size)
        instance = gizmo.TransformGizmo()
        check("session for snap step", instance.begin(app, None), True)
        instance.rotation_snap = snap
        cx, cy = instance._rotation_pivot_local()
        radius = 100.0
        instance.start("rot", cx, cy - radius)
        x = cx + radius * math.sin(math.radians(angle_deg))
        y = cy - radius * math.cos(math.radians(angle_deg))
        instance.update(x, y, uniform=True)
        return _rotation_angle(instance)

    check("rotation snaps to the configured step",
          round(drag_to(50.0, 45.0), 3), 45.0)
    check("zero step leaves the angle free",
          abs(drag_to(50.0, 0.0) - 50.0) < 0.5, True)


def test_snap_anchors_can_be_disabled():
    layer = photo_layer(120, 90)
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    check("session for snap toggle", instance.begin(app, None), True)
    instance.snap_anchors = False
    instance.start("pivot", 60.0, 45.0)
    instance.update(3.0, 3.0)
    check("no snap keeps the dragged pivot point",
          (round(instance.rot_pivot_local[0], 1),
           round(instance.rot_pivot_local[1], 1)), (3.0, 3.0))


def test_interpolation_selects_the_kernel():
    # A non-bicubic kernel has no shader, so the session must skip the GPU
    # upload and commit through Pillow instead.
    calls = []
    original = gizmo.gpuimage.texture

    def counting(image, *args, **kwargs):
        calls.append(image.size)
        return original(image, *args, **kwargs)

    layer = photo_layer()
    app = FakeApp(layer, layer.size)
    instance = gizmo.TransformGizmo()
    instance.interpolation = "bilinear"
    gizmo.gpuimage.texture = counting
    try:
        check("session for interpolation", instance.begin(app, None), True)
        instance.start("se", 320.0, 240.0)
        instance.update(480.0, 360.0)
        instance.finish()
    finally:
        gizmo.gpuimage.texture = original
    check("a non-bicubic kernel skips the GPU upload", calls, [])
    check("bilinear still commits a scaled buffer", layer.image.size, (480, 360))


def test_selection_gizmo_touches_no_pixels():
    from paintapp import selection
    mask = Image.new("L", (40, 30), 0)
    mask.paste(255, (10, 10, 30, 20))
    selected = selection.Selection.from_mask((40, 30), mask)
    instance = gizmo.SelectionGizmo()
    check("selection session starts",
          instance.begin_selection(selected, (40, 30)), True)
    instance.start("se", 30.0, 20.0)
    instance.update(40.0, 26.0)
    check("selection never resamples", instance.source, None)
    check("selection has nothing to finish", instance.finish(), None)


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
