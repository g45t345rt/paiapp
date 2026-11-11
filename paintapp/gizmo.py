"""Interactive free-transform gizmo for the active layer.

A layer's content is held under an affine transform that maps *source* pixels
(the layer's cropped content) into *document* pixels. The gizmo exposes draggable
handles drawn over the canvas:

    * corner handles  scale the frame (hold Shift for uniform scaling)
    * edge handles    shear/skew along that edge
    * the top handle  rotates about the rotation origin (Shift snaps to 15°)
    * the pivot handle moves that rotation origin (the frame centre by default)
    * dragging inside moves the frame

Every drag re-samples the original cropped source, so repeated edits do not
accumulate interpolation error. The transform itself is a 6-tuple
``(a, b, c, d, e, f)`` meaning::

    x' = a*x + c*y + e
    y' = b*x + d*y + f

That re-sample is the whole cost of a drag, and it grows with the buffer the
dragged content is rendered into - hundreds of milliseconds per event on a large
layer, growing as the frame does, which is what made the tool lag. So the drag
resamples through the GPU (:data:`_AFFINE_FRAGMENT`) and *that* result is the
transformed layer: nothing is re-done on release, so the drag stays interactive
and what you see is what is committed.

The shader is Catmull-Rom bicubic with Pillow's ``BICUBIC`` kernel and Pillow's
pixel-centre mapping, minified through a mip chain. On smooth content it lands
within a count of Pillow's own resample; on fine texture the two differ by a
couple of counts either way, since Pillow accumulates its weights in 22-bit
fixed point and widens its filter when scaling down where the shader reads a mip
level. A machine with no GL context falls back to the Pillow resample of old
(see :meth:`TransformGizmo._apply_pixels`).
"""

import math

import numpy as np
from PIL import Image

from . import gpuimage
from . import path
from . import poses
from . import textrender
from . import vector

IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

CORNER_OPPOSITE = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}
CORNERS = ("nw", "ne", "se", "sw")
EDGES = ("n", "e", "s", "w")

#: Resample kernels a session may commit with, chosen from the Transform tool's
#: settings. The GPU shader implements bicubic only, so any other kernel falls
#: back to the Pillow resample (see :meth:`TransformGizmo._resample_base`).
RESAMPLE = {
    "bicubic": Image.BICUBIC,
    "bilinear": Image.BILINEAR,
    "nearest": Image.NEAREST,
}

# The affine resample as a fragment shader: one output pixel maps back to a
# source coordinate by the same matrix ``Image.transform`` is handed, and reads
# the source there with a Catmull-Rom (a = -0.5) bicubic - the kernel behind
# Pillow's ``BICUBIC``.
#
# The one wrinkle is where a pixel sits. Pillow's ``affine`` maps *pixel centres*
# through the matrix - output pixel ``o`` reads source coordinate
# ``(o + 0.5) / scale - 0.5`` - while the filter taps below want it in index
# space, where pixel *i* covers ``[i, i + 1]`` and its centre is ``i + 0.5``
# (which is what ``tap``'s half-texel offset converts to). Getting that half
# wrong costs nothing on a smooth gradient and shifts textured content by up to
# half a source pixel, so it is checked by ``tests/test_gizmo.py``.
# Samples outside the source come back transparent, as Pillow's affine fill does.
_AFFINE_FRAGMENT = """#version 330
uniform sampler2D u_tex0;
uniform vec2 u_tex_size;
uniform vec2 u_mx;
uniform vec2 u_my;
uniform vec2 u_mt;
out vec4 f_color;

vec4 tap(vec2 p) {
    return texture(u_tex0, (p + 0.5) / u_tex_size);
}

void main() {
    vec2 o = gl_FragCoord.xy;
    vec2 s = vec2(u_mx.x * o.x + u_my.x * o.y + u_mt.x,
                  u_mx.y * o.x + u_my.y * o.y + u_mt.y) - vec2(0.5);
    if (s.x < -0.5 || s.y < -0.5 ||
        s.x > u_tex_size.x - 0.5 || s.y > u_tex_size.y - 0.5) {
        f_color = vec4(0.0);
        return;
    }
    vec2 base = floor(s);
    vec2 f = s - base;
    float wx[4];
    float wy[4];
    wx[0] = -0.5 * f.x * f.x * f.x + f.x * f.x - 0.5 * f.x;
    wx[1] = 1.5 * f.x * f.x * f.x - 2.5 * f.x * f.x + 1.0;
    wx[2] = -1.5 * f.x * f.x * f.x + 2.0 * f.x * f.x + 0.5 * f.x;
    wx[3] = 0.5 * f.x * f.x * f.x - 0.5 * f.x * f.x;
    wy[0] = -0.5 * f.y * f.y * f.y + f.y * f.y - 0.5 * f.y;
    wy[1] = 1.5 * f.y * f.y * f.y - 2.5 * f.y * f.y + 1.0;
    wy[2] = -1.5 * f.y * f.y * f.y + 2.0 * f.y * f.y + 0.5 * f.y;
    wy[3] = 0.5 * f.y * f.y * f.y - 0.5 * f.y * f.y;
    vec4 acc = vec4(0.0);
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            vec2 p = base + vec2(float(i) - 1.0, float(j) - 1.0);
            acc += wx[i] * wy[j] * tap(p);
        }
    }
    f_color = clamp(acc, 0.0, 1.0);
}
"""


# --------------------------------------------------------------------- matrix
def apply(matrix, x, y):
    """Map a point through ``matrix``."""
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def mul(first, second):
    """Compose two matrices; ``second`` is applied first."""
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def invert(matrix):
    """Return the inverse of an affine matrix."""
    a, b, c, d, e, f = matrix
    det = a * d - b * c
    if abs(det) < 1e-12:
        det = 1e-12
    return (
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    )


def translation(tx, ty):
    return (1.0, 0.0, 0.0, 1.0, tx, ty)


def rotation(theta):
    cos = math.cos(theta)
    sin = math.sin(theta)
    return (cos, sin, -sin, cos, 0.0, 0.0)


def scaling(sx, sy):
    return (sx, 0.0, 0.0, sy, 0.0, 0.0)


def shearing(kx, ky):
    """Shear matrix: ``x' = x + kx*y`` and ``y' = ky*x + y``."""
    return (1.0, ky, kx, 1.0, 0.0, 0.0)


def about(matrix, px, py):
    """Conjugate ``matrix`` so it acts about the pivot ``(px, py)``."""
    return mul(mul(translation(px, py), matrix), translation(-px, -py))


def point_in_polygon(x, y, points):
    """Even-odd test for a point against a polygon given as ``(x, y)`` pairs."""
    inside = False
    count = len(points)
    j = count - 1
    for i in range(count):
        xi, yi = points[i]
        xj, yj = points[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def _ratio(current, original):
    """Scale factor from a pivot deltas, defaulting to 1 for degenerate axes."""
    if abs(original) < 1e-6:
        return 1.0
    return current / original


class TransformGizmo:
    """State machine for one free-transform session on one layer."""

    HANDLE_HIT = 8.0  # document-pixel pickup radius at zoom 1

    def __init__(self):
        self.active = False
        self.view = None
        self.layer = None
        self.source = None
        self.source_mask = None
        self.canvas_size = (1, 1)
        self.matrix = IDENTITY
        # The session's starting placement: the document-space delta of a drag is
        # ``matrix . inverse(_base_matrix)``, which is what the live document-space
        # overlays (paths, vector items, text, poses) are mapped through so they
        # travel with the transformed pixels instead of staying pinned.
        self._base_matrix = IDENTITY
        # The overlays' geometry as captured at session start, so every drag can
        # re-derive them from the original rather than accumulating error.
        self._overlay_backup = None
        self._handle = None
        self._press = None
        self._press_local = None
        self._pivot_local = None
        self._pivot_doc = None
        self._angle0 = 0.0
        # Rotation origin in source/local space; ``None`` means the frame
        # centre. Set by dragging the "pivot" handle and reset per session.
        self.rot_pivot_local = None
        self._pivot_grab = None
        self._start_matrix = None
        self._drag_image = None
        self._drag_mask = None
        self._drag_offset = (0, 0)
        # A link layer has no pixels of its own to resample: a session records
        # its per-instance transform instead (see :meth:`_apply_pixels`).
        self._is_link = False
        self._link_box = (0, 0, 0, 0)
        self._drag_transform = None
        # A masked link's own mask, held in the buffer frame of the transform
        # the session started from, so every drag frame resamples it from the
        # original instead of accumulating interpolation error.
        self._link_mask = None
        self._link_mask_matrix = IDENTITY
        # The resident GPU uploads of ``source``/``source_mask`` for the session
        # (see :meth:`_gpu_textures`).
        self._gpu = None
        self._gpu_tried = False
        # Interactive drag throttling state: the caller may skip the expensive
        # per-event resample/overlay work (``apply=False``) while the pointer
        # keeps moving, and paint overlays in a cheap non-antialiased *draft*
        # while it does. Either leaves the committed frame stale, so a release
        # lands the final, full-quality resample (see :meth:`finish`).
        self._pending_apply = False
        self._draft_used = False
        self._preview_image = None
        self._preview_origin = None
        self._link_mask_frame = None
        # Options mirrored from the Transform tool's settings (see the
        # controller's ``_apply_transform_options``): the rotation snap step in
        # degrees (0 disables snapping), whether the rotation origin snaps to a
        # frame anchor, and the resample kernel a commit uses.
        self.rotation_snap = 15.0
        self.snap_anchors = True
        self.interpolation = "bicubic"

    # -------------------------------------------------------------- lifecycle
    def begin(self, app, view):
        """Start a session for ``app``'s active layer. False when unavailable."""
        layer = app.active_layer()
        # A failed begin (no layer, locked/empty content) must not leave a
        # previous session reported as active.
        self.active = False
        self.layer = None
        self.view = view
        if layer is None or (layer.is_locked() and not layer.is_link):
            return False
        self._release_textures()
        self.layer = layer
        self.canvas_size = app.current.size
        if layer.is_link:
            # A link renders its master's content; its own buffer is empty, so
            # the session transforms that content per instance (stored back as
            # ``layer.transform``) rather than resampling the layer's pixels.
            source = layer.content_source()
            base = source._composited_image_at(source.offset)
            box = base.getbbox()
            if box is None:
                self.layer = None
                return False
            self.source = base.crop(box)
            self.source_mask = None
            self._is_link = True
            self._link_box = box
            # A link's own instance mask rides along with the content: it is
            # stored in the buffer frame the layer's transform defines, so the
            # drag resamples it through the same frame change as the pixels.
            self._link_mask = layer.mask
            self._link_mask_matrix = (
                layer.transform if layer.transform is not None else IDENTITY
            )
            # Current content-to-document matrix: the link's stored transform
            # maps content into its buffer, and ``offset`` places that buffer.
            content_to_doc = mul(
                translation(*layer.offset),
                layer.transform if layer.transform is not None else IDENTITY,
            )
            self.matrix = mul(content_to_doc, translation(box[0], box[1]))
            self._overlay_backup = None
        else:
            # Document-space overlays (paths, vector items, text, poses) travel
            # with a transform like they do with a move; capture their geometry
            # now so every drag re-derives them from the original.
            self._overlay_backup = self._capture_overlays(layer)
            box = self._content_box(layer)
            if box is None:
                self.layer = None
                self._overlay_backup = None
                return False
            self.source = layer.image.crop(box)
            self.source_mask = (
                layer.mask.crop(box) if layer.mask is not None else None
            )
            self._is_link = False
            ox, oy = layer.offset
            self.matrix = (1.0, 0.0, 0.0, 1.0,
                           float(box[0] + ox), float(box[1] + oy))
        self._base_matrix = self.matrix
        self.rot_pivot_local = None
        self._end_drag()
        self.active = True
        return True

    def clear(self):
        self.active = False
        if self.layer is not None:
            self.layer._transform_draft = False
        self.layer = None
        self.source = None
        self.source_mask = None
        self._is_link = False
        self._link_box = (0, 0, 0, 0)
        self._link_mask = None
        self._link_mask_matrix = IDENTITY
        self._drag_transform = None
        self._overlay_backup = None
        self._base_matrix = IDENTITY
        self._release_textures()
        self._end_drag()
        if self.view is not None:
            self.view.set_gizmo(None)
        self.view = None

    # -------------------------------------------------------------- overlays
    def _capture_overlays(self, layer):
        """Snapshot a layer's document-space overlay geometry, or ``None``.

        A link's overlays belong to its master and are placed by the master's
        render, so a link captures nothing.
        """
        if layer.is_link:
            return None
        backup = {}
        if layer.paths:
            backup["paths"] = [(spec, path.clone(spec)) for spec in layer.paths]
        items = list(getattr(layer, "strokes", [])) + list(
            getattr(layer, "fills", []))
        if items:
            backup["vector"] = [
                (spec, vector.item_snapshot(spec)) for spec in items
            ]
        if layer.texts:
            backup["texts"] = [(spec, dict(spec)) for spec in layer.texts]
        scene = getattr(layer, "pose_scene", None)
        if layer.poses and isinstance(scene, dict):
            backup["poses"] = (scene, dict(scene))
        return backup or None

    def _content_box(self, layer):
        """The layer's content bounds in its own pixel space, overlays included.

        The base pixels' ink box alone would leave a vector/text/pose-only layer
        with no frame (and clip an overlay that spills past the pixels), so the
        union with every overlay's document bounds (shifted into buffer space)
        frames the whole thing.
        """
        boxes = []
        base = layer.image.getbbox()
        if base is not None:
            boxes.append(base)
        ox, oy = layer.offset

        def add(box):
            if not box:
                return
            left, top, right, bottom = box
            if right > left and bottom > top:
                boxes.append((left - ox, top - oy, right - ox, bottom - oy))

        for spec in layer.paths:
            add(path.bounds(spec))
        for spec in getattr(layer, "strokes", []):
            add(vector.bounds(spec))
        for spec in getattr(layer, "fills", []):
            add(vector.fill_bounds(spec))
        for spec in layer.texts:
            points = textrender.corners(spec)
            if points:
                xs = [point[0] for point in points]
                ys = [point[1] for point in points]
                add((min(xs), min(ys), max(xs), max(ys)))
        if layer.poses:
            scene = getattr(layer, "pose_scene", None)
            if isinstance(scene, dict):
                poses.sync_scene(layer.poses, scene)
            for spec in layer.poses:
                add(poses.bounds(spec))
        if not boxes:
            return None
        return (
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        )

    def _apply_overlays(self, delta):
        """Map every captured overlay through the document-space ``delta``.

        Each spec is restored from its session-start snapshot first, so repeated
        drag events re-derive it from the original rather than accumulating the
        transform (and a release leaves exactly the last dragged state).
        """
        backup = self._overlay_backup
        if not backup or self.layer is None:
            return
        layer = self.layer
        if "paths" in backup:
            for spec, snapshot in backup["paths"]:
                path.apply_geometry(spec, path.to_path(snapshot))
                path.transform(spec, delta)
            layer.invalidate_paths()
        if "vector" in backup:
            for spec, snapshot in backup["vector"]:
                vector.item_restore(spec, snapshot)
                vector.item_transform(spec, delta)
            layer.invalidate_vectors()
        if "texts" in backup:
            for spec, snapshot in backup["texts"]:
                spec.clear()
                spec.update(snapshot)
                textrender.transform(spec, delta)
            layer.invalidate_texts()
        if "poses" in backup:
            scene, snapshot = backup["poses"]
            scene.clear()
            scene.update(snapshot)
            poses.transform_scene(scene, delta)
            layer.invalidate_poses()

    def cancel_drag(self, app):
        """Revert the in-progress drag and drop its undo entry."""
        if self.active and self.dragging:
            if self._is_link:
                self.layer.transform = self._drag_transform
                self.layer.offset = self._drag_offset
                self.layer._transform_cache = None
                self.layer._transform_key = None
                app._drop_undo()
            elif self._drag_image is not None:
                self.layer.set_image(self._drag_image)
                self.layer.mask = self._drag_mask
                self.layer.offset = self._drag_offset
                self.layer._content_delta = (0, 0)
                # Put the overlays back where the restored pixels show them: the
                # drag-start placement, i.e. the total delta up to ``start``.
                self._apply_overlays(
                    mul(self._start_matrix, invert(self._base_matrix)))
                app._drop_undo()
        self._end_drag()

    def _end_drag(self):
        self._handle = None
        self._press = None
        self._press_local = None
        self._pivot_local = None
        self._pivot_doc = None
        self._pivot_grab = None
        self._start_matrix = None
        self._drag_image = None
        self._drag_mask = None
        self._pending_apply = False
        self._draft_used = False
        self._link_mask_frame = None
        self._clear_preview()
        if self.layer is not None:
            # A draft frame's cheap overlay render must not outlive the drag.
            self.layer._transform_draft = False

    # ------------------------------------------------------------------- gpu
    def _release_textures(self):
        """Drop the uploaded source and mask, if any."""
        if self._gpu is not None:
            for texture in self._gpu:
                if texture is not None:
                    texture.release()
        self._gpu = None
        self._gpu_tried = False

    def _gpu_textures(self):
        """The resident uploads for ``source`` and its mask, or ``None``.

        Uploaded on the first drag event rather than in :meth:`begin`, so
        activating the tool - or switching layers with it active - costs nothing
        for a session that never drags. ``None`` means no GL context, a failed
        upload, or a mask that could not join it, and the caller keeps its
        Pillow resample. ``_gpu_tried`` remembers the attempt either way, so a
        machine without GL does not retry it on every event.
        """
        if self._gpu_tried:
            return self._gpu
        self._gpu_tried = True
        source = gpuimage.texture(self.source, mipmaps=True, storage="f1")
        if source is None:
            return None
        mask = None
        if self.source_mask is not None:
            mask = gpuimage.texture(self.source_mask, mipmaps=True, storage="f1")
            if mask is None:
                source.release()
                return None
        self._gpu = (source, mask)
        return self._gpu

    def _resample_gpu(self, texture, source, size, data):
        """Resample the uploaded ``texture`` into ``size``; ``None`` if no GL.

        ``source`` is the image being resampled, for its mode: the layer's
        pixels come back in it, and an ``"L"`` mask keeps only the channel its
        upload replicated. ``data`` is the same Pillow ``AFFINE`` tuple the CPU
        path is handed, one output pixel mapping through it to a source pixel.
        """
        if texture is None:
            return None
        a, b, c, d, e, f = data
        pixels = gpuimage.render(
            _AFFINE_FRAGMENT, size, textures=[texture],
            uniforms={
                "u_tex_size": (float(texture.size[0]), float(texture.size[1])),
                "u_mx": (a, d),
                "u_my": (b, e),
                "u_mt": (c, f),
            },
            target_dtype="f1",
        )
        if pixels is None:
            return None
        if source.mode == "L":
            return Image.fromarray(np.ascontiguousarray(pixels[:, :, 0]))
        result = Image.fromarray(pixels)
        if result.mode != source.mode:
            result = result.convert(source.mode)
        return result

    # --------------------------------------------------------------- geometry
    def _size(self):
        return self.source.size

    def _rot_offset(self):
        width, height = self._size()
        return max(18.0, 0.08 * max(width, height))

    def _handles_local(self):
        width, height = self._size()
        return {
            "nw": (0.0, 0.0),
            "ne": (float(width), 0.0),
            "se": (float(width), float(height)),
            "sw": (0.0, float(height)),
            "n": (width / 2.0, 0.0),
            "e": (float(width), height / 2.0),
            "s": (width / 2.0, float(height)),
            "w": (0.0, height / 2.0),
            "rot": (width / 2.0, -self._rot_offset()),
            "pivot": self._rotation_pivot_local(),
        }

    def _center_local(self):
        width, height = self._size()
        return (width / 2.0, height / 2.0)

    def _rotation_pivot_local(self):
        """The rotation origin in source space; the frame centre by default."""
        if self.rot_pivot_local is None:
            return self._center_local()
        return self.rot_pivot_local

    def _anchor_local(self):
        """The frame's snap anchors: corners, edge midpoints and centre."""
        width, height = self._size()
        return {
            "nw": (0.0, 0.0), "n": (width / 2.0, 0.0), "ne": (width, 0.0),
            "w": (0.0, height / 2.0), "c": (width / 2.0, height / 2.0),
            "e": (width, height / 2.0),
            "sw": (0.0, height), "s": (width / 2.0, height),
            "se": (width, height),
        }

    def _snap_anchor_local(self, doc_x, doc_y, base):
        """Nearest frame anchor to document ``(doc_x, doc_y)``, or ``None``."""
        zoom = max(getattr(self.view, "zoom", 1.0) or 1.0, 1e-6)
        tolerance = self.HANDLE_HIT / zoom
        best = None
        best_distance = tolerance
        for local in self._anchor_local().values():
            ax, ay = apply(base, *local)
            distance = math.hypot(ax - doc_x, ay - doc_y)
            if distance <= best_distance:
                best_distance = distance
                best = local
        return best

    def reset_rotation_pivot(self):
        """Put the rotation origin back at the frame centre."""
        self.rot_pivot_local = None

    def _handle_points(self):
        return {
            key: apply(self.matrix, x, y)
            for key, (x, y) in self._handles_local().items()
        }

    def geometry(self):
        """Overlay geometry in document pixels, or ``None`` when inactive."""
        if not self.active:
            return None
        points = self._handle_points()
        frame = [points[key] for key in CORNERS]
        handles = [points[key] for key in CORNERS + EDGES]
        top = points["n"]
        return {
            "frame": frame,
            "handles": handles,
            "rotate": points["rot"],
            "stem": (top, points["rot"]),
            "pivot": points["pivot"],
        }

    # ------------------------------------------------------------------ input
    def hit(self, x, y, tolerance):
        """Return the handle under ``(x, y)``, ``"move"`` inside, or ``None``."""
        points = self._handle_points()
        hx, hy = points["pivot"]
        if abs(hx - x) <= tolerance and abs(hy - y) <= tolerance:
            return "pivot"
        for key in ("rot",) + CORNERS + EDGES:
            hx, hy = points[key]
            if abs(hx - x) <= tolerance and abs(hy - y) <= tolerance:
                return key
        if point_in_polygon(x, y, [points[key] for key in CORNERS]):
            return "move"
        return None

    def start(self, handle, x, y):
        self._handle = handle
        self._press = (x, y)
        self._preview_image = None
        self._start_matrix = self.matrix
        self._press_local = apply(invert(self.matrix), x, y)
        self._drag_offset = tuple(self.layer.offset)
        if self._is_link:
            # Nothing to resample; cancel restores the transform instead.
            self._drag_image = None
            self._drag_mask = None
            self._drag_transform = self.layer.transform
        else:
            self._drag_image = self.layer.image.copy()
            self._drag_mask = (
                self.layer.mask.copy() if self.layer.mask is not None else None
            )
            self._drag_transform = None
        if handle == "pivot":
            px, py = self._rotation_pivot_local()
            self._pivot_grab = (px - self._press_local[0],
                                py - self._press_local[1])
        elif handle == "rot":
            self._pivot_doc = apply(self.matrix, *self._rotation_pivot_local())
            self._angle0 = math.atan2(y - self._pivot_doc[1],
                                      x - self._pivot_doc[0])
        elif handle in CORNERS:
            local = self._handles_local()[CORNER_OPPOSITE[handle]]
            self._pivot_local = local
        elif handle in EDGES:
            self._pivot_local = self._center_local()

    def update(self, x, y, uniform=False, apply_pixels=True, draft=False):
        """Apply a drag to the transform and re-sample the layer. True if moved.

        ``apply_pixels=False`` skips the resample and overlay work for this event
        while still updating :attr:`matrix` (so the gizmo frame follows the
        pointer at full rate); the caller repaints less often during a drag and a
        release lands the final frame. ``draft=True`` paints the overlays without
        anti-aliasing, which is far cheaper over a large or scaled buffer; the
        release restores it (see :meth:`finish`).
        """
        if not self.active or self._handle is None:
            return False
        base = self._start_matrix
        handle = self._handle
        if handle == "pivot":
            # Moving the rotation origin only repositions the marker; it does
            # not resample any pixels. It snaps to a frame anchor (corners,
            # edge midpoints, centre) when dragged close to one.
            local = apply(invert(base), x, y)
            gx, gy = self._pivot_grab if self._pivot_grab is not None else (0.0, 0.0)
            marker_local = (local[0] + gx, local[1] + gy)
            marker_doc = apply(base, *marker_local)
            snapped = None
            if self.snap_anchors:
                snapped = self._snap_anchor_local(
                    marker_doc[0], marker_doc[1], base)
            self.rot_pivot_local = snapped if snapped is not None else marker_local
            return True
        if handle == "move":
            px, py = self._press
            dx = x - px
            dy = y - py
            self.matrix = mul(translation(dx, dy), base)
            # A pure translation is deferred like the Move tool: slide the layer
            # placement (offset) with the matching ``_content_delta`` instead of
            # re-sampling the whole buffer every event, which is the cost that
            # made dragging a transform lag. The single resample happens once on
            # release (see :meth:`_bake_move`). A link records its own transform,
            # so it keeps the per-event path.
            if self._is_link:
                # A link records its own transform. Snap the live move to whole
                # pixels so the normalised transform (and with it the link's
                # cached resample) stays unchanged and only the offset slides;
                # the release lands the exact sub-pixel position.
                self.matrix = mul(translation(round(dx), round(dy)), base)
                self._apply_pixels(draft=True)
                self.matrix = mul(translation(dx, dy), base)
                self._pending_apply = True
                self._draft_used = True
            else:
                self._apply_move(dx, dy)
            return True
        if handle == "rot":
            cx, cy = self._pivot_doc
            delta = math.atan2(y - cy, x - cx) - self._angle0
            if uniform and self.rotation_snap > 0:
                step = math.radians(self.rotation_snap)
                delta = round(delta / step) * step
            self.matrix = mul(about(rotation(delta), cx, cy), base)
        elif handle in CORNERS:
            local = apply(invert(base), x, y)
            pl = self._press_local
            ox, oy = self._pivot_local
            fx = _ratio(local[0] - ox, pl[0] - ox)
            fy = _ratio(local[1] - oy, pl[1] - oy)
            if uniform:
                d0 = math.hypot(pl[0] - ox, pl[1] - oy)
                d1 = math.hypot(local[0] - ox, local[1] - oy)
                factor = d1 / d0 if d0 > 1e-6 else 1.0
                fx = fy = factor
            self.matrix = mul(base, about(scaling(fx, fy), ox, oy))
        else:  # edge handles shear
            local = apply(invert(base), x, y)
            pl = self._press_local
            cx, cy = self._pivot_local
            width, height = self._size()
            if handle in ("n", "s"):
                shear = about(shearing((local[0] - pl[0]) / width, 0.0), cx, cy)
            else:
                shear = about(shearing(0.0, (local[1] - pl[1]) / height), cx, cy)
            self.matrix = mul(base, shear)
        if apply_pixels and draft:
            # A live frame: show the drag-start snapshot through the new matrix,
            # sampled at display resolution by the renderer, instead of
            # resampling the whole layer every frame. The release resamples once.
            self._show_preview()
            self._pending_apply = False
            self._draft_used = True
        elif apply_pixels:
            self._apply_pixels(draft=draft)
            self._pending_apply = False
            self._draft_used = draft
        else:
            # The matrix (and so the gizmo frame) is current, but the pixels are
            # not: remember to land them on release.
            self._pending_apply = True
        return True

    def flush_draft(self):
        """Land a throttled drag's skipped frame as a draft. True if it painted.

        The drag throttle skips resampling between frames; when the pointer then
        holds still, this catches the pixels up to the gizmo frame without
        waiting for the release.
        """
        if not (self.active and self.dragging and self._pending_apply):
            return False
        if self._handle == "move" and self._is_link:
            return False  # already painted (snapped) every event
        self._show_preview()
        self._pending_apply = False
        self._draft_used = True
        return True

    def _show_preview(self):
        """Point the layer's live preview at the current :attr:`matrix`.

        The snapshot is the layer's composite (filters, overlays, masks) taken
        once at the first live frame of the drag; the matrix maps its pixels
        through the drag's delta into document space.
        """
        if self._preview_image is None:
            image = self.layer.composited_image()
            if image.mode != "RGBA":
                image = image.convert("RGBA")
            self._preview_image = image
            ox, oy = self._drag_offset
            self._preview_origin = translation(ox, oy)
        delta = mul(self.matrix, invert(self._start_matrix))
        self.layer._live_preview = (
            self._preview_image, mul(delta, self._preview_origin))

    def _clear_preview(self):
        self._preview_image = None
        if self.layer is not None:
            self.layer._live_preview = None
            self.layer._live_preview_mips = None

    def finish(self):
        # A move drag deferred its resample (see :meth:`_apply_move`); commit it
        # once here so the released state matches what a per-event resample would
        # have produced.
        if self._handle == "move" and self.active and not self._is_link:
            self._bake_move()
        elif self.active and (self._pending_apply or self._draft_used):
            # A throttled drag skipped its last resample, or painted a draft
            # (non-antialiased overlays): land the exact, full-quality frame.
            self._apply_pixels(draft=False)
        self._end_drag()

    @property
    def dragging(self):
        return self._handle is not None

    @property
    def moving_pivot(self):
        """True while the rotation-origin marker is being dragged."""
        return self._handle == "pivot"

    # ----------------------------------------------------------------- render
    def _apply_move(self, dx, dy):
        """Slide the layer for a pure translation, without re-sampling.

        The Move tool's trick: shift ``offset`` and set the matching
        ``_content_delta`` so ``content_offset`` is unchanged. The layer's own
        image (and every overlay's cached raster, keyed on the image token and
        the content offset) then stays warm for the whole drag - only the
        placement slides - so dragging a large layer is as cheap as a move.
        """
        ox, oy = self._drag_offset
        self.layer.offset = (ox + dx, oy + dy)
        self.layer._content_delta = (dx, dy)

    def _bake_move(self):
        """Commit a deferred move: one resample, then clear the drag delta."""
        self._apply_pixels()
        self.layer._content_delta = (0, 0)

    def _apply_pixels(self, draft=False):
        """Re-sample the layer through :attr:`matrix`, into its own buffer.

        The GPU does the resample when it can (see :meth:`_gpu_textures`): the
        Pillow call below costs hundreds of milliseconds on a large layer, which
        is what made every drag event lag. Without a GL context the drag keeps
        that Pillow resample, which is also what every version before this one
        committed, so a machine without GL still transforms correctly.
        """
        if self._is_link:
            # A link owns no pixels: record the per-instance transform that
            # places the master's content. ``self.matrix`` maps the session's
            # source (a crop of the master content) to the document; shift it
            # back to the content's own pixel space for the layer to store.
            box = self._link_box
            content_matrix = mul(
                self.matrix, translation(-box[0], -box[1])
            )
            self.layer._transform_draft = draft
            self.layer.set_transform(content_matrix)
            if self._link_mask is not None:
                # The instance mask lives in the buffer frame the transform
                # defines. Map it from the frame the session started in into the
                # new one, resampling from the original every frame.
                new_matrix = self.layer.transform or IDENTITY
                frame = mul(self._link_mask_matrix, invert(new_matrix))
                frame_key = (tuple(round(v, 9) for v in frame),
                             tuple(self.layer.size), self.interpolation)
                if getattr(self, "_link_mask_frame", None) == frame_key:
                    # A pure move keeps the same buffer frame: nothing to redo.
                    return
                self._link_mask_frame = frame_key
                self.layer.mask = self._link_mask.transform(
                    self.layer.size, Image.AFFINE,
                    (frame[0], frame[2], frame[4], frame[1], frame[3], frame[5]),
                    resample=RESAMPLE.get(self.interpolation, Image.BICUBIC),
                )
            return
        self._resample_base(draft=draft)
        # Carry the document-space overlays through the same gesture, from the
        # session's original placement, so a path/vector/text/pose travels with
        # the transformed pixels instead of staying pinned to the page.
        self._apply_overlays(mul(self.matrix, invert(self._base_matrix)))
        # Mark the layer so the *render* paints its overlays without the costly
        # anti-aliased re-raster while the drag is live (see ``document``); a
        # release resamples with ``draft=False`` and clears it again.
        self.layer._transform_draft = draft

    def _resample_base(self, draft=False):
        """Re-sample the base pixels and mask through :attr:`matrix`."""
        # Render into a buffer big enough to hold the transformed content as well
        # as the canvas, then record where that buffer sits in document space via
        # ``layer.offset``. This keeps pixels scaled or moved past the canvas edge
        # instead of clipping them to the canvas rectangle.
        width, height = self.source.size
        corners = [apply(self.matrix, x, y) for x, y in (
            (0.0, 0.0), (float(width), 0.0),
            (float(width), float(height)), (0.0, float(height)),
        )]
        doc_w, doc_h = self.canvas_size
        if draft:
            # A live drag frame only needs the transformed content itself: the
            # layer offset places it. Growing the buffer to the canvas made every
            # frame resample (and invalidate) a canvas-sized image; the release
            # (``draft=False``) restores the canvas-covering buffer.
            doc_w = doc_h = float("-inf")
            left = math.floor(min(p[0] for p in corners))
            top = math.floor(min(p[1] for p in corners))
        else:
            left = min(0, math.floor(min(p[0] for p in corners)))
            top = min(0, math.floor(min(p[1] for p in corners)))
        right = max(doc_w, math.ceil(max(p[0] for p in corners)))
        bottom = max(doc_h, math.ceil(max(p[1] for p in corners)))
        size = (max(1, right - left), max(1, bottom - top))
        # PIL AFFINE data maps output pixels back to source pixels, so invert the
        # source-to-buffer transform (matrix shifted to the buffer's origin).
        inverse = invert(mul(translation(-left, -top), self.matrix))
        data = (inverse[0], inverse[2], inverse[4],
                inverse[1], inverse[3], inverse[5])
        resample = RESAMPLE.get(self.interpolation, Image.BICUBIC)
        # The GPU shader is bicubic only; any other kernel keeps the Pillow path.
        textures = self._gpu_textures() if resample == Image.BICUBIC else None
        result = None
        if textures is not None:
            result = self._resample_gpu(textures[0], self.source, size, data)
        if result is None:
            result = self.source.transform(
                size, Image.AFFINE, data, resample=resample
            )
        self.layer.set_image(result)
        if self.source_mask is not None:
            mask = None
            if textures is not None:
                mask = self._resample_gpu(
                    textures[1], self.source_mask, size, data)
            if mask is None:
                mask = self.source_mask.transform(
                    size, Image.AFFINE, data, resample=resample
                )
            self.layer.mask = mask
        self.layer.offset = (left, top)


class SelectionGizmo(TransformGizmo):
    """Free-transform gizmo for a selection's outline.

    Reuses :class:`TransformGizmo`'s handle geometry and input mapping, but the
    transform is applied to a :class:`selection.Selection` (its mask and
    outline) instead of a layer image, so it works on any layer - even an empty
    one. ``on_change`` receives the transformed selection while dragging.
    """

    def __init__(self):
        super().__init__()
        self.on_change = None
        self.source_selection = None
        self.source_box = None
        self._source_size = (1, 1)

    def begin_selection(self, selection, document_size):
        """Start a session for ``selection``. False when it is empty."""
        box = selection.bbox if selection is not None else None
        if box is None:
            return False
        self.source_selection = selection
        self.source_box = box
        self.canvas_size = document_size
        self._source_size = (
            max(1, box[2] - box[0]), max(1, box[3] - box[1]),
        )
        self.matrix = (1.0, 0.0, 0.0, 1.0, float(box[0]), float(box[1]))
        self.rot_pivot_local = None
        self._end_drag()
        self.active = True
        return True

    def clear(self):
        self.source_selection = None
        self.source_box = None
        super().clear()

    def _size(self):
        return self._source_size

    def start(self, handle, x, y):
        """Begin a drag; like the base, minus the layer backup."""
        self._handle = handle
        self._press = (x, y)
        self._preview_image = None
        self._start_matrix = self.matrix
        self._press_local = apply(invert(self.matrix), x, y)
        if handle == "pivot":
            px, py = self._rotation_pivot_local()
            self._pivot_grab = (px - self._press_local[0],
                                py - self._press_local[1])
        elif handle == "rot":
            self._pivot_doc = apply(self.matrix, *self._rotation_pivot_local())
            self._angle0 = math.atan2(y - self._pivot_doc[1],
                                      x - self._pivot_doc[0])
        elif handle in CORNERS:
            self._pivot_local = self._handles_local()[CORNER_OPPOSITE[handle]]
        elif handle in EDGES:
            self._pivot_local = self._center_local()

    def cancel_drag(self, app):
        self._end_drag()

    def _apply_pixels(self, draft=False):
        # A selection has no layer pixels to resample; its mask and outline are
        # the transformed state, so there is nothing for the GPU path to do.
        if self.on_change is not None and self.source_selection is not None:
            self.on_change(self.source_selection.transformed(self.matrix))
