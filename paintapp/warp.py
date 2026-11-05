"""Pixel-warping engine shared by the Liquify, Mesh Warp and Puppet Warp tools.

Every tool deforms a layer by resampling its original pixels through an
accumulated displacement field - liquify grows the field under a brush, mesh
warp sets it from a dragged control grid, and puppet warp blends it from
dragged pins - so they share the one resampler here. A warp keeps a
premultiplied copy of the source pixels and a per-pixel displacement
``(dx, dy)``; output pixel ``p`` samples the source at ``p - (dx, dy)``,
bilinearly, so the tools preview and commit the same way.

Colours are premultiplied by alpha before sampling (and divided back out after),
so the transparent black around a shape cannot bleed into a warped edge and
darken it - the same trick :func:`antialias.downscale` uses. Sampling outside
the source yields transparent pixels, leaving a clean gap when content is pushed
off the edge rather than a smeared border.
"""

import math

import numpy as np
from PIL import Image

from . import gpuimage
from . import puppetmesh

# Liquify brush modes. ``twirl_cw``/``twirl_ccw`` rotate the field around the
# brush; the others push along the drag, or pull/push radially.
LIQUIFY_MODES = ("push", "expand", "pinch", "twirl_cw", "twirl_ccw")

# Guard on the source size (pixels) a single warp will hold, so an enormous
# canvas does not allocate gigabytes for the premultiplied copy and field.
MAX_SOURCE_PIXELS = 64_000_000

# Most pins a single puppet-warp session accepts, so the control-region blend
# stays bounded.
MAX_PINS = 64


# The mesh-warp field and resample as a fragment shader: it mirrors ``_field``
# and ``_bilinear``/``_unpremultiply`` exactly, so a GPU patch and the numpy one
# agree pixel for pixel (bar the final byte rounding). The node displacements
# arrive in a small RGBA float texture (``r``/``g``) so the grid size is not
# capped by the uniform-array limit; ``gl_FragCoord`` maps to the local source
# pixel index, matching the numpy ``mgrid`` rect.
_MESH_FRAGMENT = """#version 330
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform vec2 u_tex_size;
uniform vec2 u_rect;
uniform vec2 u_cell;
uniform int u_cols;
uniform int u_rows;
out vec4 f_color;

vec2 node_disp(int row, int col) {
    return texelFetch(u_tex1, ivec2(col, row), 0).rg;
}

vec2 field(vec2 local) {
    vec2 g = (local + vec2(0.5)) / u_cell;
    int col0 = clamp(int(floor(g.x)), 0, u_cols - 1);
    int row0 = clamp(int(floor(g.y)), 0, u_rows - 1);
    vec2 f = clamp(g - vec2(float(col0), float(row0)), 0.0, 1.0);
    vec2 d00 = node_disp(row0, col0);
    vec2 d10 = node_disp(row0, col0 + 1);
    vec2 d01 = node_disp(row0 + 1, col0);
    vec2 d11 = node_disp(row0 + 1, col0 + 1);
    vec2 top_edge = d00 + (d10 - d00) * f.x;
    vec2 bottom_edge = d01 + (d11 - d01) * f.x;
    return top_edge + (bottom_edge - top_edge) * f.y;
}

vec4 tap(vec2 p) {
    vec2 base = floor(p);
    vec2 f = p - base;
    ivec2 i0 = ivec2(base);
    ivec2 hi = ivec2(u_tex_size) - 1;
    ivec2 ia = clamp(i0, ivec2(0), hi);
    ivec2 ib = clamp(i0 + 1, ivec2(0), hi);
    vec4 v00 = texelFetch(u_tex0, ivec2(ia.x, ia.y), 0);
    vec4 v10 = texelFetch(u_tex0, ivec2(ib.x, ia.y), 0);
    vec4 v01 = texelFetch(u_tex0, ivec2(ia.x, ib.y), 0);
    vec4 v11 = texelFetch(u_tex0, ivec2(ib.x, ib.y), 0);
    vec4 top = v00 + (v10 - v00) * f.x;
    vec4 bottom = v01 + (v11 - v01) * f.x;
    return top + (bottom - top) * f.y;
}

void main() {
    vec2 local = floor(gl_FragCoord.xy) + u_rect;
    vec2 src = local - field(local);
    vec4 s = vec4(0.0);
    if (src.x >= 0.0 && src.y >= 0.0 &&
        src.x <= u_tex_size.x - 1.0 && src.y <= u_tex_size.y - 1.0) {
        s = tap(src);
    }
    float a = clamp(s.a, 0.0, 1.0);
    vec3 rgb = (a > 1e-6) ? clamp(s.rgb / max(a, 1e-6), 0.0, 1.0) : vec3(0.0);
    f_color = vec4(rgb, a);
}
"""


def _delaunay(points):
    """Compatibility wrapper for the shared puppet triangulator."""
    return puppetmesh.delaunay(points)


def _run_edges(line):
    """``(start, end)`` indices of each opaque run in a boolean 1-D array."""
    indices = np.nonzero(line)[0]
    if indices.size == 0:
        return []
    breaks = np.nonzero(np.diff(indices) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    return [(int(indices[s]), int(indices[e])) for s, e in zip(starts, ends)]


def _premultiplied(image):
    """``image`` as a float RGBA array in ``0..1`` with colour premultiplied."""
    rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:4]
    return rgba


def _bilinear(source, xs, ys):
    """Sample ``source`` (float RGBA) at float coordinates ``xs``/``ys``.

    A sample is valid while it lands within the pixel-centre range; the
    neighbours are clamped at the border so the last row/column survives instead
    of being dropped. Genuinely beyond the source samples transparent, so a warp
    that pulls content off an edge leaves a gap instead of smearing.
    """
    height, width = source.shape[:2]
    x0 = np.floor(xs)
    y0 = np.floor(ys)
    fx = (xs - x0)[..., None].astype(np.float32)
    fy = (ys - y0)[..., None].astype(np.float32)
    xi0 = x0.astype(np.int64)
    yi0 = y0.astype(np.int64)
    inside = (xs >= 0) & (xs <= width - 1) & (ys >= 0) & (ys <= height - 1)
    xa = np.clip(xi0, 0, width - 1)
    xb = np.clip(xi0 + 1, 0, width - 1)
    ya = np.clip(yi0, 0, height - 1)
    yb = np.clip(yi0 + 1, 0, height - 1)
    v00 = source[ya, xa]
    v10 = source[ya, xb]
    v01 = source[yb, xa]
    v11 = source[yb, xb]
    top = v00 + (v10 - v00) * fx
    bottom = v01 + (v11 - v01) * fx
    out = top + (bottom - top) * fy
    out[~inside] = 0.0
    return out


def _unpremultiply(array):
    """Turn a premultiplied float RGBA array back into an 8-bit ``RGBA`` image."""
    alpha = np.clip(array[..., 3:4], 0.0, 1.0)
    rgb = np.where(alpha > 1e-6, array[..., :3] / np.maximum(alpha, 1e-6), 0.0)
    out = np.empty(array.shape, dtype=np.float32)
    out[..., :3] = np.clip(rgb, 0.0, 1.0)
    out[..., 3:4] = alpha
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8), "RGBA")


def _falloff(distance, radius):
    """Smooth brush weight: 1 at the centre, easing to 0 at ``radius``."""
    t = np.clip(1.0 - distance / max(float(radius), 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Liquify:
    """A liquify session on one layer: brush dabs grow a displacement field."""

    def __init__(self, layer):
        self.layer = layer
        self.width, self.height = layer.size
        self.source = _premultiplied(layer.image)
        self.dx = np.zeros((self.height, self.width), dtype=np.float32)
        self.dy = np.zeros((self.height, self.width), dtype=np.float32)
        self.image = layer.image.copy()

    def _region(self, cx, cy, radius):
        left = max(0, int(np.floor(cx - radius - 1.0)))
        top = max(0, int(np.floor(cy - radius - 1.0)))
        right = min(self.width, int(np.ceil(cx + radius + 1.0)))
        bottom = min(self.height, int(np.ceil(cy + radius + 1.0)))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    def _refresh(self, rect):
        """Re-sample ``rect`` from the source through the field into ``image``."""
        left, top, right, bottom = rect
        ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
        sample_x = xs - self.dx[top:bottom, left:right]
        sample_y = ys - self.dy[top:bottom, left:right]
        patch = _unpremultiply(_bilinear(self.source, sample_x, sample_y))
        self.image.paste(patch, (left, top))

    def stamp(self, cx, cy, radius, mode, strength, step_x=0.0, step_y=0.0):
        """Apply one brush dab centred at ``(cx, cy)`` and refresh the preview.

        ``mode`` is a :data:`LIQUIFY_MODES` value; ``strength`` is ``0..1``.
        ``step_x``/``step_y`` is the pointer motion since the last dab, used by
        the push mode. Returns the layer's new image, or ``None`` if off-canvas.
        """
        if mode not in LIQUIFY_MODES:
            mode = "push"
        radius = max(1.0, float(radius))
        rect = self._region(cx, cy, radius)
        if rect is None:
            return None
        left, top, right, bottom = rect
        ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
        rel_x = xs - cx
        rel_y = ys - cy
        distance = np.hypot(rel_x, rel_y)
        weight = _falloff(distance, radius)
        safe = np.maximum(distance, 1e-6)
        if mode == "push":
            move_x = float(step_x) * weight
            move_y = float(step_y) * weight
        elif mode in ("expand", "pinch"):
            amount = strength * radius * 0.15
            sign = 1.0 if mode == "expand" else -1.0
            move_x = sign * amount * weight * rel_x / safe
            move_y = sign * amount * weight * rel_y / safe
        else:  # twirl
            angle = strength * 0.25 * (1.0 if mode == "twirl_cw" else -1.0)
            move_x = -rel_y * angle * weight
            move_y = rel_x * angle * weight
        self.dx[top:bottom, left:right] += move_x.astype(np.float32)
        self.dy[top:bottom, left:right] += move_y.astype(np.float32)
        self._refresh(rect)
        return self.image

    def commit(self):
        """The warped image; the caller stores it on the layer."""
        return self.image


class MeshWarp:
    """A mesh-warp session: a control grid drives a bilinear displacement field.

    The grid is laid over the layer's content box as ``(rows + 1) x (cols + 1)``
    nodes. Dragging a node records a displacement there; the field at any pixel
    is the bilinear blend of the surrounding node displacements, holding the
    nearest edge node's displacement past the box (never extrapolating, so a far
    drag cannot fold source pixels into a smear). ``render`` re-samples the
    source through that field, growing the region by :meth:`_margin` so content a
    node pushes past the box edge is kept instead of reading as transparent
    (which would look like the warp was cut along the box).
    """

    def __init__(self, layer, box, cols=4, rows=4):
        self.layer = layer
        self.box = box
        self.cols = max(1, int(cols))
        self.rows = max(1, int(rows))
        left, top, right, bottom = box
        self.origin = (left, top)
        # The box is in layer pixels; the layer's document offset places the
        # grid on the canvas, so the overlay and hit-testing work in document
        # space while the pixel maths stays in layer space.
        offset = tuple(getattr(layer, "offset", (0, 0)))
        self.doc_origin = (left + offset[0], top + offset[1])
        self.cell_w = max(1.0, (right - left) / self.cols)
        self.cell_h = max(1.0, (bottom - top) / self.rows)
        self.source = _premultiplied(layer.image.crop(box))
        self.nodes = np.zeros((self.rows + 1, self.cols + 1, 2), dtype=np.float32)
        # The pristine layer image, so Reset (and a rebuild) can drop every node
        # offset without sampling through the field.
        self.original = layer.image.copy()
        self.image = layer.image.copy()
        # Bounding box (local pixels) of every region written since the image was
        # last rebuilt from the source, so a partial re-render can also clear the
        # wider band a larger earlier drag wrote into.
        self._dirty = None
        # The resident upload of ``source`` for the whole session, so a node drag
        # resamples a patch on the GPU without re-uploading the layer each event.
        self._gpu_texture = None
        self._gpu_tried = False

    def node_position(self, row, col):
        """The node's current position in document pixels (grid + displacement)."""
        left, top = self.doc_origin
        return (
            left + col * self.cell_w + float(self.nodes[row, col, 0]),
            top + row * self.cell_h + float(self.nodes[row, col, 1]),
        )

    def node_rest(self, row, col):
        """The node's rest position (its undisplaced grid point), document pixels."""
        left, top = self.doc_origin
        return (left + col * self.cell_w, top + row * self.cell_h)

    def grid_geometry(self):
        """The deformed grid as a list of horizontal and vertical node polylines."""
        horizontal = []
        for row in range(self.rows + 1):
            horizontal.append([
                self.node_position(row, col) for col in range(self.cols + 1)
            ])
        vertical = []
        for col in range(self.cols + 1):
            vertical.append([
                self.node_position(row, col) for row in range(self.rows + 1)
            ])
        return {"horizontal": horizontal, "vertical": vertical}

    def hit_node(self, x, y, tolerance):
        """``(row, col)`` of the node within ``tolerance`` of ``(x, y)``, or ``None``."""
        best = None
        best_distance = tolerance
        for row in range(self.rows + 1):
            for col in range(self.cols + 1):
                nx, ny = self.node_position(row, col)
                distance = np.hypot(nx - x, ny - y)
                if distance <= best_distance:
                    best_distance = distance
                    best = (row, col)
        return best

    def _field(self, rect):
        """The displacement field over the local pixel ``rect``, ``(h, w, 2)``.

        The blend weight is clamped to ``0..1``, so past the box the field holds
        the nearest edge node's displacement instead of extrapolating it. Without
        the clamp an edge node dragged far would fold source pixels across a huge
        area, smearing them well outside the grid.
        """
        left, top, right, bottom = rect
        ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
        gx = (xs + 0.5) / self.cell_w
        gy = (ys + 0.5) / self.cell_h
        col0 = np.clip(np.floor(gx).astype(np.int64), 0, self.cols - 1)
        row0 = np.clip(np.floor(gy).astype(np.int64), 0, self.rows - 1)
        fx = np.clip(gx - col0, 0.0, 1.0)[..., None]
        fy = np.clip(gy - row0, 0.0, 1.0)[..., None]
        d00 = self.nodes[row0, col0]
        d10 = self.nodes[row0, col0 + 1]
        d01 = self.nodes[row0 + 1, col0]
        d11 = self.nodes[row0 + 1, col0 + 1]
        top_edge = d00 + (d10 - d00) * fx
        bottom_edge = d01 + (d11 - d01) * fx
        return top_edge + (bottom_edge - top_edge) * fy

    def affected_rect(self, row, col):
        """The local pixel rect that changes when node ``(row, col)`` moves.

        Bilinear interpolation makes a node affect only the cells touching it
        (up to four), so a drag re-samples just that patch rather than the whole
        layer - the difference between a laggy drag and a smooth one on a large
        canvas.
        """
        height, width = self.source.shape[:2]
        row0 = max(0, row - 1)
        row1 = min(self.rows, row + 1)
        col0 = max(0, col - 1)
        col1 = min(self.cols, col + 1)
        left = max(0, int(np.floor(col0 * self.cell_w)) - 1)
        top = max(0, int(np.floor(row0 * self.cell_h)) - 1)
        right = min(width, int(np.ceil(col1 * self.cell_w)) + 1)
        bottom = min(height, int(np.ceil(row1 * self.cell_h)) + 1)
        return (left, top, right, bottom)

    def _margin(self):
        """Pixels to grow the render region by so node pushes are not clipped.

        The grid sits on the content box, so the shared edge nodes have no source
        pixels just outside them. Without a margin, content a node pushes past the
        box edge is sampled from outside the source and reads as transparent - the
        warp looks cut along the box. Grow by the furthest any node has moved (plus
        a bilinear pad), clamped the same way :meth:`PuppetWarp._margin` is.
        """
        if not self.nodes.any():
            return 0
        move = float(np.abs(self.nodes).max())
        if move <= 0.0:
            return 0
        height, width = self.source.shape[:2]
        margin = int(math.ceil(move)) + 2
        margin = min(margin, max(width, height))
        while (width + 2 * margin) * (height + 2 * margin) > MAX_SOURCE_PIXELS \
                and margin > 0:
            margin = int(margin * 0.8)
        return margin

    def render(self, rect=None):
        """Re-sample ``rect`` (local pixels) through the node field.

        ``None`` re-samples the whole content box; a rect re-samples just that
        patch, which is what a single-node drag needs. The region is grown by
        :meth:`_margin` so content a node pushes past the box edge is kept rather
        than cut, and unioned with every region written since the last rebuild so
        a later, smaller drag still clears the wider band an earlier large drag
        left (otherwise the far smear of a node dragged out and back would
        persist). The image is rebuilt from the pristine source each time, so a
        region outside the union is always the untouched original.

        The GPU renders the patch when it can (see :meth:`_render_patch_gpu`),
        which is what makes a node drag interactive on a large layer; without a GL
        context the numpy path does the same work, so the two agree bar the final
        rounding.
        """
        height, width = self.source.shape[:2]
        if rect is None:
            rect = (0, 0, width, height)
        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return self.image
        margin = self._margin()
        region = (int(left) - margin, int(top) - margin,
                  int(right) + margin, int(bottom) + margin)
        if self._dirty is not None:
            dl, dt, dr, db = self._dirty
            region = (min(region[0], dl), min(region[1], dt),
                      max(region[2], dr), max(region[3], db))
        self._dirty = region
        patch = self._render_patch_gpu(region)
        if patch is None:
            patch = self._render_patch_cpu(region)
        self.image = self.original.copy()
        self.image.paste(patch, (self.origin[0] + region[0],
                                 self.origin[1] + region[1]))
        return self.image

    def _render_patch_cpu(self, rect):
        """The numpy resample of ``rect`` (the fallback when no GL context)."""
        left, top, right, bottom = rect
        field = self._field(rect)
        ys, xs = np.mgrid[top:bottom, left:right].astype(np.float32)
        return _unpremultiply(
            _bilinear(self.source, xs - field[..., 0], ys - field[..., 1])
        )

    def _source_texture(self):
        """The resident upload of ``source``, or ``None`` when GL is unavailable.

        Uploaded on the first drag event rather than in ``__init__``, so selecting
        the tool - or rebuilding the grid after Apply - costs nothing until a node
        actually moves. ``_gpu_tried`` remembers a failed attempt so a machine
        without GL does not retry it on every event.
        """
        if self._gpu_texture is not None:
            return self._gpu_texture
        if self._gpu_tried:
            return None
        self._gpu_tried = True
        self._gpu_texture = gpuimage.texture(self.source, storage="f4")
        return self._gpu_texture

    def _render_patch_gpu(self, rect):
        """Render ``rect`` with :data:`_MESH_FRAGMENT`, or ``None``.

        ``None`` (no context, a failed upload, or any GL failure) hands the caller
        back to the numpy patch rather than raising.
        """
        left, top, right, bottom = rect
        texture = self._source_texture()
        if texture is None:
            return None
        nodes = np.zeros((self.rows + 1, self.cols + 1, 4), dtype="f4")
        nodes[..., :2] = self.nodes
        pixels = gpuimage.render(
            _MESH_FRAGMENT, (right - left, bottom - top),
            textures=[texture, nodes],
            uniforms={
                "u_tex_size": (
                    float(self.source.shape[1]), float(self.source.shape[0])),
                "u_rect": (float(left), float(top)),
                "u_cell": (float(self.cell_w), float(self.cell_h)),
                "u_cols": int(self.cols),
                "u_rows": int(self.rows),
            },
            target_dtype="f1",
        )
        if pixels is None:
            return None
        return Image.fromarray(pixels, "RGBA")

    def release(self):
        """Free the resident source texture; safe to call more than once."""
        if self._gpu_texture is not None:
            self._gpu_texture.release()
        self._gpu_texture = None
        self._gpu_tried = False

    def nudge_node(self, row, col, dx, dy):
        """Add ``(dx, dy)`` to node ``(row, col)`` and re-render its cells."""
        self.nodes[row, col, 0] += float(dx)
        self.nodes[row, col, 1] += float(dy)
        return self.render(self.affected_rect(row, col))

    def set_node_offset(self, row, col, dx, dy):
        """Set node ``(row, col)``'s displacement to ``(dx, dy)`` and re-render."""
        self.nodes[row, col, 0] = float(dx)
        self.nodes[row, col, 1] = float(dy)
        return self.render(self.affected_rect(row, col))

    def reset(self):
        self.nodes[:] = 0.0
        self.image = self.original.copy()
        self._dirty = None
        return self.image

    def commit(self):
        return self.image


class PuppetWarp:
    """A puppet-warp session: movable, rotatable pins deform an ARAP mesh.

    The layer content is covered by a triangular mesh (:mod:`paintapp.puppetmesh`).
    Dragging a pin moves its nearest mesh vertex; unmoved pins anchor their
    vertices, and a two-step Igarashi ARAP solve determines all other vertices.
    A single pin applies a rigid transform to its connected component. Pin
    rotation constrains a small local neighbourhood to the rotated pin frame.
    The warped region is bounded by the destination layer; an optional ``clip``
    mask limits the warp to the active selection; ``render``
    transfers the source through the deformed mesh.
    """

    #: The longer mesh side is split into roughly this many cells, so the mesh
    #: (and the solver) stays a bounded size on a large layer.
    MESH_CELLS = 45
    #: Source pixels the live drag preview aims for; a bigger layer previews at a
    #: lower resolution and the exact :meth:`render` sharpens it on release.
    PREVIEW_TARGET = 20_000

    def __init__(self, layer, box, influence=150, expansion=20, clip=None):
        self.layer = layer
        self.box = box
        left, top, right, bottom = box
        self.origin = (left, top)
        # The box is in layer pixels; the layer's document offset places it on
        # the canvas, so the overlay and hit-testing work in document space while
        # the pixel maths stays in layer space.
        offset = tuple(getattr(layer, "offset", (0, 0)))
        self.doc_origin = (left + offset[0], top + offset[1])
        self.source = _premultiplied(layer.image.crop(box))
        #: Optional layer-sized ``L`` mask limiting the warp to the active
        #: selection (255 where pixels may move, 0 where they are kept as-is).
        self.clip = clip
        #: Upper bound for the local rotation frame and pin-order region. Plain
        #: drags use point constraints and are not limited by this radius.
        self.influence = max(1.0, float(influence))
        self.expansion = max(0.0, float(expansion))
        # The pristine layer image, so Reset (and Clear) can drop the warp.
        self.original = layer.image.copy()
        self.image = layer.image.copy()
        # Each pin is a dict: ``rest``/``pos`` are ``[x, y]`` document pixels,
        # ``angle`` is the rotation (radians) about the rest point, and
        # ``selected`` drives the overlay and multi-pin drags.
        self.pins = []
        # True while ``image`` holds a reduced-resolution drag preview, so the
        # caller knows to finish with an exact :meth:`render` when the drag ends.
        self.previewed = False
        height, width = self.source.shape[:2]
        self.mesh_delta = max(6.0, max(width, height) / float(self.MESH_CELLS))
        self.support_alpha = self.source[..., 3].copy()
        if self.clip is not None:
            clip_alpha = np.asarray(self.clip.crop(box), dtype=np.float32) / 255.0
            self.support_alpha *= clip_alpha
        self.support = self.support_alpha > 0.01
        self._preview_source = None
        self._build_mesh()

    def _build_mesh(self):
        """Triangulate the silhouette with every pin as an exact vertex."""
        extra = [np.asarray(pin["rest"]) - self.doc_origin for pin in self.pins]
        self.vertices, self.faces = puppetmesh.content_mesh(
            self.support_alpha, self.mesh_delta, extra_points=extra)
        self.arap = puppetmesh.TwoStepARAP(self.vertices, self.faces)
        self.edges = self.arap.edges
        parent = list(range(len(self.vertices)))

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        for a, b in self.edges:
            ra, rb = root(int(a)), root(int(b))
            if ra != rb:
                parent[rb] = ra
        roots = [root(index) for index in range(len(parent))]
        labels = {value: index for index, value in enumerate(sorted(set(roots)))}
        self.components = np.asarray([labels[value] for value in roots])
        self.deformed = self.vertices.copy()
        self._dirty = True

    # ------------------------------------------------------------------- pins
    def _rest_point(self, x, y):
        """Map a click on the visible mesh back to its original source point."""
        point = np.asarray((x, y), dtype=float) - self.doc_origin
        if self._static():
            candidates = [point]
        else:
            triangles = self._ensure_solved()[self.faces]
            v0 = triangles[:, 1] - triangles[:, 0]
            v1 = triangles[:, 2] - triangles[:, 0]
            offset = point - triangles[:, 0]
            den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
            valid = np.abs(den) > 1e-9
            safe_den = np.where(valid, den, 1)
            b = (offset[:, 0] * v1[:, 1] - offset[:, 1] * v1[:, 0]) / safe_den
            c = (v0[:, 0] * offset[:, 1] - v0[:, 1] * offset[:, 0]) / safe_den
            a = 1 - b - c
            hit = np.flatnonzero(valid & (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6))
            priorities = self._face_priorities()
            hit = sorted(hit, key=lambda i: (priorities[i], i), reverse=True)
            candidates = [np.array((a[i], b[i], c[i])) @ self.vertices[self.faces[i]]
                          for i in hit]
        height, width = self.support.shape
        for point in candidates:
            px, py = point
            if 0 <= px <= width and 0 <= py <= height:
                col = min(width - 1, int(px))
                row = min(height - 1, int(py))
                if self.support[row, col]:
                    return point
        return None

    def contains(self, x, y):
        """Whether a document point is on the rendered/selected puppet mesh."""
        return self._rest_point(x, y) is not None

    def add_pin(self, x, y):
        """Add a selected pin at document ``(x, y)``; returns its index."""
        rest = self._rest_point(x, y)
        rest = np.asarray((x, y)) if rest is None else rest + self.doc_origin
        self.pins.append({
            "rest": [float(rest[0]), float(rest[1])],
            "pos": [float(x), float(y)],
            "angle": 0.0,
            "selected": True,
            # ``correct`` lets auto rotation-correction turn this pin; a pin with
            # it off stays put, anchoring the areas you do not want altered.
            "correct": True,
            # Higher order draws overlapping triangles on top, without changing
            # the deformation or other pins' positional constraints.
            "order": 0,
        })
        self._build_mesh()
        return len(self.pins) - 1

    def remove_pin(self, index):
        """Drop pin ``index``; returns whether it existed."""
        if 0 <= index < len(self.pins):
            del self.pins[index]
            self._dirty = True
            self.render()
            return True
        return False

    def clear_pins(self):
        """Remove every pin and restore the pristine image."""
        self.previewed = False
        self.pins = []
        self.deformed = self.vertices.copy()
        self._dirty = True
        self.image = self.original.copy()
        return self.image

    def hit_pin(self, x, y, tolerance):
        """Index of the pin within ``tolerance`` of ``(x, y)``, or ``None``."""
        best = None
        best_distance = tolerance
        for index, pin in enumerate(self.pins):
            px, py = pin["pos"]
            distance = math.hypot(px - x, py - y)
            if distance <= best_distance:
                best_distance = distance
                best = index
        return best

    def handle_pos(self, index, radius):
        """The rotation handle's document position for pin ``index``."""
        px, py = self.pins[index]["pos"]
        angle = self.pins[index]["angle"]
        return (px + math.sin(angle) * radius, py - math.cos(angle) * radius)

    def hit_handle(self, x, y, tolerance, radius):
        """Index of a selected pin whose rotation handle is near ``(x, y)``."""
        for index, pin in enumerate(self.pins):
            if not pin["selected"]:
                continue
            hx, hy = self.handle_pos(index, radius)
            if math.hypot(hx - x, hy - y) <= tolerance:
                return index
        return None

    def hit_ring(self, x, y, tolerance, radius):
        """Index of a selected pin whose outer rotation ring passes near ``(x, y)``.

        The ring is the circle of ``radius`` around the pin, so grabbing it
        (rather than the little knob on top) rotates the pin - the outer-handle
        behaviour of a puppet warp. Only selected pins expose a ring, so a new
        pin can still be dropped close to an existing one.
        """
        best = None
        best_distance = tolerance + 1e-6
        for index, pin in enumerate(self.pins):
            if not pin["selected"]:
                continue
            px, py = pin["pos"]
            distance = abs(math.hypot(px - x, py - y) - radius)
            if distance <= best_distance:
                best_distance = distance
                best = index
        return best

    # -------------------------------------------------------------- selection
    def selected_indices(self):
        return [i for i, pin in enumerate(self.pins) if pin["selected"]]

    def select_only(self, index):
        for i, pin in enumerate(self.pins):
            pin["selected"] = i == index

    def toggle_select(self, index):
        self.pins[index]["selected"] = not self.pins[index]["selected"]

    def clear_selection(self):
        for pin in self.pins:
            pin["selected"] = False

    def select_rect(self, box, additive=False):
        """Select the pins inside ``box`` (adding to any already selected)."""
        left, top, right, bottom = box
        for pin in self.pins:
            px, py = pin["pos"]
            inside = left <= px <= right and top <= py <= bottom
            pin["selected"] = bool(inside or (additive and pin["selected"]))

    # -------------------------------------------------------------- transforms
    def set_pin_pos(self, index, x, y):
        """Move pin ``index`` to document ``(x, y)`` without re-rendering."""
        self.pins[index]["pos"][0] = float(x)
        self.pins[index]["pos"][1] = float(y)
        self._dirty = True

    def set_pin_angle(self, index, angle):
        """Set pin ``index``'s rotation (radians) without re-rendering."""
        self.pins[index]["angle"] = float(angle)
        self._dirty = True

    def toggle_correct(self, index):
        """Toggle whether auto rotation-correction may turn pin ``index``."""
        self.pins[index]["correct"] = not self.pins[index]["correct"]

    def raise_pins(self, indices):
        """Move the given pins to the top of the overlap order."""
        order = max((pin["order"] for pin in self.pins), default=0) + 1
        for index in indices:
            self.pins[index]["order"] = order
        self._dirty = True

    def lower_pins(self, indices):
        """Move the given pins to the bottom of the overlap order."""
        order = min((pin["order"] for pin in self.pins), default=0) - 1
        for index in indices:
            self.pins[index]["order"] = order
        self._dirty = True

    def correct_rotations(self, hold=(), iterations=8, rate=0.5):
        """Retained for UI compatibility; pins keep their user-set frames.

        The two-step ARAP solver already derives local mesh rotations. Moving
        neighbouring pin handles afterwards would fight the user's anchors and
        is therefore not part of this deformation model.
        """
        return

    def move_pin(self, index, x, y):
        """Move pin ``index`` to document ``(x, y)`` and preview the result.

        The blend reaches every pixel, so a drag warps the whole region and a
        full-resolution render each event would lag. Instead the live drag
        previews at reduced resolution through :meth:`_render_preview`; the
        caller runs the exact :meth:`render` when the drag ends.
        """
        pin = self.pins[index]
        if x == pin["pos"][0] and y == pin["pos"][1]:
            return self.image
        pin["pos"][0] = float(x)
        pin["pos"][1] = float(y)
        self._dirty = True
        return self._render_preview()

    def preview(self):
        """Re-render a live drag at reduced resolution (see :meth:`render`)."""
        return self._render_preview()

    def reset(self):
        """Return every pin to rest and restore the pristine image."""
        self.previewed = False
        for pin in self.pins:
            pin["pos"][0] = pin["rest"][0]
            pin["pos"][1] = pin["rest"][1]
            pin["angle"] = 0.0
        self.deformed = self.vertices.copy()
        self._dirty = True
        self.image = self.original.copy()
        return self.image

    def pin_geometry(self):
        """The pins as dicts of ``rest``/``pos``/``angle``/``selected``/flags."""
        return [
            {
                "rest": (pin["rest"][0], pin["rest"][1]),
                "pos": (pin["pos"][0], pin["pos"][1]),
                "angle": pin["angle"],
                "selected": pin["selected"],
                "correct": pin["correct"],
                "order": pin["order"],
            }
            for pin in self.pins
        ]

    def _static(self):
        """Whether every pin still sits at its rest frame (no warp to apply)."""
        for pin in self.pins:
            if pin["angle"] != 0.0:
                return False
            if pin["pos"][0] != pin["rest"][0] or pin["pos"][1] != pin["rest"][1]:
                return False
        return True

    # -------------------------------------------------------------- mesh view
    def mesh_geometry(self):
        """The deformed mesh over the drawing, as triangle edges.

        The mesh is generated from the rendered/selected silhouette, so its
        edges follow the drawing rather than covering the whole bounding box.
        Returns ``(start, end)`` document-pixel edges.
        """
        if self._static():
            deformed = self.vertices
            edges = self.edges
        else:
            deformed = self._ensure_solved(iterations=6)
            edges = self.edges
        left, top = self.doc_origin
        # Thousands of Tk line items are far more expensive than the sparse
        # solve. A representative subset keeps overlay redraws interactive; the
        # full triangle set is still used by the renderer.
        stride = max(1, int(math.ceil(len(edges) / 600.0)))
        return [
            (
                (left + float(deformed[i, 0]), top + float(deformed[i, 1])),
                (left + float(deformed[j, 0]), top + float(deformed[j, 1])),
            )
            for i, j in edges[::stride]
        ]

    def _pin_frame(self, pin, scale=1.0):
        """Local rest/target positions and rotation for one pin."""
        rest = (np.asarray(pin["rest"], dtype=np.float64) - self.doc_origin) * scale
        pos = (np.asarray(pin["pos"], dtype=np.float64) - self.doc_origin) * scale
        angle = float(pin["angle"])
        ca, sa = math.cos(angle), math.sin(angle)
        rotation = np.array(((ca, -sa), (sa, ca)))
        return rest, pos, rotation

    def _constraints_for(self, vertices, components, mesh_delta, scale=1.0):
        """Build point-pin constraints for the final or preview mesh."""
        count = len(vertices)
        target = vertices.copy()
        claim_distance = np.full(count, np.inf)
        claimed = np.zeros(count, dtype=bool)
        anchors = []
        pin_components = np.zeros(len(self.pins), dtype=np.int64)
        rotation_radius = max(
            mesh_delta * 1.5, min(self.influence * scale, mesh_delta * 3.0))

        for pin_index, pin in enumerate(self.pins):
            rest, pos, rotation = self._pin_frame(pin, scale)
            offset = vertices - rest
            distance = np.hypot(offset[:, 0], offset[:, 1])
            nearest = int(np.argmin(distance))
            anchors.append((nearest, pos))
            pin_components[pin_index] = components[nearest]
            if pin["angle"] != 0.0:
                inside = distance <= rotation_radius
            else:
                inside = np.zeros(count, dtype=bool)
                inside[nearest] = True
            moved = offset @ rotation.T + pos
            inside &= components == components[nearest]
            better = inside & (distance < claim_distance)
            target[better] = moved[better]
            claim_distance[better] = distance[better]
            claimed[better] = True

        for component in range(int(components.max()) + 1):
            members = components == component
            pin_indices = np.nonzero(pin_components == component)[0]
            if not len(pin_indices):
                target[members] = vertices[members]
                claimed[members] = True
            elif len(pin_indices) == 1:
                rest, pos, rotation = self._pin_frame(
                    self.pins[pin_indices[0]], scale)
                target[members] = (vertices[members] - rest) @ rotation.T + pos
                claimed[members] = True
        # A rotation neighbourhood must never displace another pin's anchor.
        for nearest, pos in anchors:
            target[nearest] = pos
            claimed[nearest] = True
        constrained = np.nonzero(claimed)[0]
        return constrained, target[constrained]

    def _constraints(self):
        return self._constraints_for(
            self.vertices, self.components, self.mesh_delta)

    def _solve(self, iterations):
        """Recompute the deformed mesh from the current pins."""
        constrained, targets = self._constraints()
        if len(constrained) == self.arap.n:
            self.deformed = self.vertices.copy()
            self.deformed[constrained] = targets
        else:
            self.deformed = self.arap.deformation(constrained, targets)
        self._dirty = False

    def _ensure_solved(self, iterations=8):
        """The deformed vertices, solving only when the pins have changed."""
        if self._dirty:
            self._solve(iterations)
        return self.deformed

    # ----------------------------------------------------------------- render
    def _clip_image(self, image):
        """Keep the warp inside the clip mask, if there is one."""
        if self.clip is None:
            return image
        return Image.composite(image, self.original, self.clip)

    def _face_priorities(self):
        """Triangle stacking values derived from the pins that claim them."""
        priorities = np.zeros(len(self.faces), dtype=np.float64)
        if not self.pins or not any(pin["order"] for pin in self.pins):
            return priorities
        claimed = np.full(len(self.faces), -np.inf)
        centroids = self.vertices[self.faces].mean(axis=1)
        for pin in self.pins:
            rest = np.asarray(pin["rest"], dtype=np.float64) - self.doc_origin
            distance = np.hypot(*(centroids - rest).T)
            better = (distance <= self.influence) & (pin["order"] > claimed)
            claimed[better] = pin["order"]
        priorities[np.isfinite(claimed)] = claimed[np.isfinite(claimed)]
        return priorities

    def _render_solved(self):
        """Transfer the source through the current deformed mesh."""
        return self._render_at_scale(1, self.source)

    def _render_at_scale(self, factor, source):
        """Rasterize only visible deformed bounds using the same mesh at any scale."""
        image = self.original.copy()
        image.paste((0, 0, 0, 0), self.box)
        low = np.floor((self.deformed.min(axis=0) - 2) / factor).astype(int)
        high = np.ceil((self.deformed.max(axis=0) + 2) / factor).astype(int)
        canvas_low = np.floor(-np.asarray(self.origin) / factor).astype(int)
        canvas_high = np.ceil(
            (np.asarray(image.size) - self.origin) / factor).astype(int)
        low = np.maximum(low, canvas_low)
        high = np.minimum(high, canvas_high)
        if np.any(high <= low):
            return self._clip_image(image)
        bounds = (*low, *high)
        patch = puppetmesh.warp_triangles(
            source, self.vertices / factor, self.deformed / factor, self.faces, 0,
            priorities=self._face_priorities(), bounds=bounds)
        patch_image = _unpremultiply(patch)
        if factor != 1:
            scale = 1.0 / factor
            patch_image = patch_image.transform(
                (patch_image.width * factor, patch_image.height * factor),
                Image.AFFINE, (scale, 0, (1 - scale) / 2,
                               0, scale, (1 - scale) / 2), Image.BILINEAR)
        image.alpha_composite(patch_image, dest=(
            int(self.origin[0] + low[0] * factor),
            int(self.origin[1] + low[1] * factor)))
        return self._clip_image(image)

    def _render_exact(self):
        """Solve the mesh and transfer the source through it at full resolution."""
        self._ensure_solved(iterations=8)
        return self._render_solved()

    def render(self):
        """Deform the mesh for the current pins and re-sample the source.

        The As-Rigid-As-Possible solve resists stretching and shearing, so the
        warp bends smoothly around dragged pins and preserves thin appendages.
        """
        self.previewed = False
        if not self.pins or self._static():
            self.deformed = self.vertices.copy()
            self._dirty = True
            self.image = self.original.copy()
            return self.image
        self.image = self._render_exact()
        return self.image

    def _preview_factor(self):
        """The power-of-two downscale that keeps a drag preview affordable."""
        height, width = self.source.shape[:2]
        factor = 1
        while math.ceil(width / factor) * math.ceil(height / factor) > self.PREVIEW_TARGET:
            factor *= 2
        return factor

    def _render_preview(self):
        """A quicker render for a live drag; :meth:`render` is the exact one.

        Only the texture resolution changes; sharing the exact deformation keeps
        anchors, thin connections and overlap order stable on mouse release.
        """
        factor = self._preview_factor()
        if factor <= 1 or not self.pins or self._static():
            return self.render()
        self.previewed = True
        self._ensure_solved()
        if self._preview_source is None:
            height, width = self.source.shape[:2]
            # Filter premultiplied channels and preserve mesh pixel coordinates.
            shift = -(factor - 1) / (2.0 * factor)
            channels = [Image.fromarray(self.source[..., i]).reduce(factor).transform(
                (math.ceil(width / factor), math.ceil(height / factor)),
                Image.AFFINE, (1, 0, shift, 0, 1, shift), Image.BILINEAR)
                for i in range(4)]
            self._preview_source = np.stack(
                [np.asarray(channel) for channel in channels], axis=-1)
        self.image = self._render_at_scale(factor, self._preview_source)
        return self.image

    def commit(self):
        return self.image
