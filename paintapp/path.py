"""Editable bezier path used by the Path tool.

A path is a list of anchor points, each with an incoming and outgoing cubic
bezier control point. Clicking places a corner anchor (handles coincide with the
anchor); dragging while placing pulls a smooth handle out, mirrored on the other
side. :meth:`Path.flatten` samples the segments into a plain polygon so the
shape can be filled with :meth:`PIL.ImageDraw.ImageDraw.polygon`.

The module also owns the *committed* path object: a plain-dict spec
(:func:`normalize`/:func:`clone`) that a finished path is stored as on its layer,
the geometry helpers the canvas uses to pick and reshape it
(:func:`hit`/:func:`anchor_hit`/:func:`handle_hit`/:func:`move_anchor` and
friends), and :func:`paste`, which renders the spec into a layer's pixels. The
layer keeps the spec (like text and vector strokes) so the path stays editable
until it is rasterized.
"""

import math

from PIL import Image, ImageDraw, ImageFilter

from . import antialias
from .constants import PATH_OUTLINE_MAX, rgba

STEPS = 16

# Fill/stroke output modes and the cap/join spellings a saved path may carry.
MODES = ("fill", "stroke", "both")
CAPS = ("round", "butt", "square")
JOINS = ("round", "miter")

MAX_WIDTH = 2000



def _cubic(p0, p1, p2, p3, steps=STEPS):
    """Sample the cubic bezier ``p0 -> p3`` (control points ``p1``/``p2``)."""
    points = []
    for i in range(1, steps + 1):
        t = i / steps
        mt = 1.0 - t
        a = mt * mt * mt
        b = 3.0 * mt * mt * t
        c = 3.0 * mt * t * t
        d = t * t * t
        points.append((
            a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
        ))
    return points


class Path:
    def __init__(self):
        self.anchors = []
        self.handles_in = []
        self.handles_out = []
        self.closed = False

    def __len__(self):
        return len(self.anchors)

    def add(self, x, y):
        """Append a corner anchor at ``(x, y)`` and return its index."""
        x = float(x)
        y = float(y)
        self.anchors.append((x, y))
        self.handles_in.append((x, y))
        self.handles_out.append((x, y))
        return len(self.anchors) - 1

    def remove_last(self):
        if self.anchors:
            self.anchors.pop()
            self.handles_in.pop()
            self.handles_out.pop()

    def remove(self, index):
        """Delete the anchor (and its handles) at ``index``."""
        if 0 <= index < len(self.anchors):
            del self.anchors[index]
            del self.handles_in[index]
            del self.handles_out[index]

    def move(self, index, x, y):
        """Move anchor ``index`` and its handles by the delta to ``(x, y)``."""
        ax, ay = self.anchors[index]
        dx = float(x) - ax
        dy = float(y) - ay
        self.anchors[index] = (float(x), float(y))
        hix, hiy = self.handles_in[index]
        self.handles_in[index] = (hix + dx, hiy + dy)
        hox, hoy = self.handles_out[index]
        self.handles_out[index] = (hox + dx, hoy + dy)

    def set_out_handle(self, index, x, y):
        """Pull anchor ``index``'s outgoing handle to ``(x, y)``.

        The incoming handle is mirrored through the anchor so the curve stays
        smooth at that point.
        """
        ax, ay = self.anchors[index]
        x = float(x)
        y = float(y)
        self.handles_out[index] = (x, y)
        self.handles_in[index] = (2.0 * ax - x, 2.0 * ay - y)

    def flatten(self, extra=None, closed=False, steps=STEPS):
        """Return the path as a polygon in document pixels.

        ``extra`` appends one or more raw points to the sampled polygon. When
        ``closed`` is true a segment is added from the last anchor back to the
        first.
        """
        anchors = self.anchors
        if not anchors:
            return []
        points = [anchors[0]]
        for i in range(len(anchors) - 1):
            points.extend(_cubic(
                anchors[i], self.handles_out[i],
                self.handles_in[i + 1], anchors[i + 1], steps,
            ))
        if closed and len(anchors) >= 2:
            points.extend(_cubic(
                anchors[-1], self.handles_out[-1],
                self.handles_in[0], anchors[0], steps,
            ))
        if extra:
            points.extend(extra)
        return points

    def preview(self, cursor=None):
        """A lightweight dict describing the path for the on-screen overlay.

        It carries the geometry the tool's fill and stroke settings preview:
        ``shape`` is the closed fill outline and ``line`` the stroke polyline
        (both from the placed anchors only), while ``rubber`` is the raw cursor
        tip so the view can draw the pending segment to the pointer.
        """
        return {
            "shape": self.flatten(closed=True) if len(self.anchors) >= 2 else [],
            "line": self.flatten(closed=self.closed),
            "rubber": (float(cursor[0]), float(cursor[1]))
                      if cursor is not None else None,
            "anchors": list(self.anchors),
            "handles": list(zip(self.handles_in, self.handles_out)),
            "closed": self.closed,
        }


# --------------------------------------------------------------- spec helpers
def _clean_points(raw):
    """Coerce ``raw`` to a list of ``(x, y)`` float pairs, dropping bad rows."""
    points = []
    for point in raw or []:
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return points


def _pad_handles(handles, anchors):
    """Pad ``handles`` to ``len(anchors)`` with the anchor positions."""
    padded = list(handles[:len(anchors)])
    while len(padded) < len(anchors):
        padded.append(list(anchors[len(padded)]))
    return padded


def normalize(spec):
    """Coerce ``spec`` to a complete, valid path object (or ``None``).

    A path needs at least two anchors to describe a shape; anything shorter is
    treated as empty. Handles are padded with the anchor positions, so a
    corner anchor (whose handles coincide with it) round-trips as a corner.
    """
    if not isinstance(spec, dict):
        return None
    anchors = _clean_points(spec.get("anchors"))
    if len(anchors) < 2:
        return None
    handles_in = _pad_handles(_clean_points(spec.get("handles_in")), anchors)
    handles_out = _pad_handles(_clean_points(spec.get("handles_out")), anchors)
    mode = spec.get("mode") if spec.get("mode") in MODES else "fill"
    cap = spec.get("cap") if spec.get("cap") in CAPS else "round"
    join = spec.get("join") if spec.get("join") in JOINS else "round"
    try:
        width = int(round(float(spec.get("width", 1))))
    except (TypeError, ValueError):
        width = 1
    width = max(1, min(MAX_WIDTH, width))
    try:
        outline = int(round(float(spec.get("outline", 0) or 0)))
    except (TypeError, ValueError):
        outline = 0
    outline = max(0, min(PATH_OUTLINE_MAX, outline))
    try:
        feather = int(round(float(spec.get("feather", 0) or 0)))
    except (TypeError, ValueError):
        feather = 0
    feather = max(0, feather)
    try:
        opacity = int(round(float(spec.get("opacity", 100))))
    except (TypeError, ValueError):
        opacity = 100
    opacity = max(1, min(100, opacity))
    return {
        "anchors": anchors,
        "handles_in": handles_in,
        "handles_out": handles_out,
        "closed": bool(spec.get("closed", False)),
        "mode": mode,
        "fill_color": str(spec.get("fill_color") or "#000000"),
        "stroke_color": str(spec.get("stroke_color") or "#000000"),
        "width": width,
        "outline": outline,
        "feather": feather,
        "opacity": opacity,
        "antialias": bool(spec.get("antialias", True)),
        "cap": cap,
        "join": join,
    }


def clone(spec):
    """A deep copy of ``spec`` safe to keep in a snapshot/undo entry."""
    normalized = normalize(spec)
    if normalized is None:
        return None
    normalized["anchors"] = [list(point) for point in normalized["anchors"]]
    normalized["handles_in"] = [list(point) for point in normalized["handles_in"]]
    normalized["handles_out"] = [list(point) for point in normalized["handles_out"]]
    return normalized


def new_spec(anchors, handles_in, handles_out, closed=False, mode="fill",
             fill_color="#000000", stroke_color="#000000", width=1, outline=0,
             feather=0, opacity=100, antialias=True, cap="round", join="round"):
    """Build a normalised path object from its geometry and style."""
    return normalize({
        "anchors": anchors, "handles_in": handles_in, "handles_out": handles_out,
        "closed": closed, "mode": mode, "fill_color": fill_color,
        "stroke_color": stroke_color, "width": width, "outline": outline,
        "feather": feather, "opacity": opacity, "antialias": antialias,
        "cap": cap, "join": join,
    })


def from_path(path, **style):
    """A spec built from an in-edit :class:`Path` and the tool's style options."""
    return new_spec(
        [list(point) for point in path.anchors],
        [list(point) for point in path.handles_in],
        [list(point) for point in path.handles_out],
        closed=path.closed, **style,
    )


def to_path(spec):
    """An in-edit :class:`Path` loaded from ``spec``'s geometry."""
    path = Path()
    path.anchors = [tuple(point) for point in spec.get("anchors", [])]
    path.handles_in = [tuple(point) for point in spec.get("handles_in", [])]
    path.handles_out = [tuple(point) for point in spec.get("handles_out", [])]
    path.closed = bool(spec.get("closed", False))
    return path


def apply_geometry(spec, path):
    """Write ``path``'s geometry into ``spec`` (style is left untouched)."""
    spec["anchors"] = [list(point) for point in path.anchors]
    spec["handles_in"] = [list(point) for point in path.handles_in]
    spec["handles_out"] = [list(point) for point in path.handles_out]
    spec["closed"] = bool(path.closed)
    return spec


def describe(spec):
    """A short one-line label for a path, for the layer list."""
    count = len(spec.get("anchors", []))
    return f"{count} points  ({spec.get('mode', 'fill')})"


# -------------------------------------------------------------------- geometry
def flatten(spec, closed=None, steps=STEPS):
    """Return ``spec``'s geometry as a sampled polygon in document pixels."""
    anchors = spec.get("anchors") or []
    handles_in = spec.get("handles_in") or []
    handles_out = spec.get("handles_out") or []
    if not anchors:
        return []
    if closed is None:
        closed = bool(spec.get("closed", False))
    points = [tuple(anchors[0])]
    for index in range(len(anchors) - 1):
        points.extend(_cubic(
            anchors[index], handles_out[index],
            handles_in[index + 1], anchors[index + 1], steps,
        ))
    if closed and len(anchors) >= 2:
        points.extend(_cubic(
            anchors[-1], handles_out[-1], handles_in[0], anchors[0], steps,
        ))
    return points


def _segments(spec):
    """The path's flat segments as ``(a, b)`` point pairs (document pixels)."""
    points = flatten(spec)
    if bool(spec.get("closed", False)) and len(points) >= 2:
        points = points + [points[0]]
    return list(zip(points, points[1:]))


def bounds(spec):
    """The path's document-space bounding box padded for its style, or ``None``."""
    geometry = flatten(spec, closed=True)
    if not geometry:
        return None
    xs = [px for px, _py in geometry]
    ys = [py for _px, py in geometry]
    width = float(spec.get("width", 1))
    outline = float(spec.get("outline", 0) or 0)
    pad = max(width, outline) / 2.0 + float(spec.get("feather", 0) or 0) + 2.0
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _point_in_polygon(x, y, points):
    inside = False
    count = len(points)
    for index in range(count):
        ax, ay = points[index]
        bx, by = points[(index + 1) % count]
        if (ay > y) != (by > y):
            cross = (bx - ax) * (y - ay) / (by - ay) + ax
            if x < cross:
                inside = not inside
    return inside


def hit(spec, x, y, tolerance=0.0):
    """True when ``(x, y)`` lands on ``spec`` (its interior or its ink)."""
    mode = spec.get("mode", "fill")
    if mode in ("fill", "both"):
        polygon = flatten(spec, closed=True)
        if len(polygon) >= 3 and _point_in_polygon(x, y, polygon):
            return True
    if mode in ("stroke", "both"):
        radius = float(spec.get("width", 1)) / 2.0 + tolerance
        for (ax, ay), (bx, by) in _segments(spec):
            distance = _distance_to_segment(x, y, ax, ay, bx, by)
            if distance <= radius:
                return True
    return False


def anchor_hit(spec, x, y, tolerance):
    """Index of the anchor nearest ``(x, y)`` within ``tolerance``, or ``None``."""
    best = None
    best_distance = tolerance
    for index, (ax, ay) in enumerate(spec.get("anchors") or []):
        distance = math.hypot(x - ax, y - ay)
        if distance <= best_distance:
            best_distance = distance
            best = index
    return best


def handle_hit(spec, x, y, tolerance):
    """``(side, index)`` of the bezier handle under ``(x, y)``, or ``None``.

    ``side`` is ``"in"`` or ``"out"``. A handle that coincides with its anchor
    (a corner) is ignored, so it is the anchor - not a hidden handle - that the
    pointer grabs.
    """
    anchors = spec.get("anchors") or []
    for index, anchor in enumerate(anchors):
        for side, handles in (("in", spec.get("handles_in") or []),
                              ("out", spec.get("handles_out") or [])):
            if index >= len(handles):
                continue
            hx, hy = handles[index]
            if math.hypot(hx - anchor[0], hy - anchor[1]) < 0.5:
                continue
            if math.hypot(hx - x, hy - y) <= tolerance:
                return side, index
    return None


def _distance_to_segment(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def nearest_segment(spec, x, y):
    """``(index, t)`` of the segment nearest ``(x, y)``, or ``None``."""
    best = None
    best_distance = None
    for index, ((ax, ay), (bx, by)) in enumerate(_segments(spec)):
        dx = bx - ax
        dy = by - ay
        length_sq = dx * dx + dy * dy
        if length_sq <= 1e-9:
            t = 0.0
        else:
            t = max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / length_sq))
        distance = math.hypot(x - (ax + t * dx), y - (ay + t * dy))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best = (index, t)
    return best


# --------------------------------------------------------------------- editing
def move_anchor(spec, index, x, y):
    """Move anchor ``index`` (and its handles) to ``(x, y)`` (in place)."""
    anchors = spec.get("anchors") or []
    if not (0 <= index < len(anchors)):
        return
    ax, ay = anchors[index]
    dx = float(x) - ax
    dy = float(y) - ay
    anchors[index] = [float(x), float(y)]
    for handles in (spec.get("handles_in") or [], spec.get("handles_out") or []):
        if index < len(handles):
            handles[index] = [handles[index][0] + dx, handles[index][1] + dy]


def set_out_handle(spec, index, x, y):
    """Pull anchor ``index``'s outgoing handle to ``(x, y)``, mirroring the in."""
    anchors = spec.get("anchors") or []
    if not (0 <= index < len(anchors)):
        return
    ax, ay = anchors[index]
    out = spec.setdefault("handles_out", [])
    inn = spec.setdefault("handles_in", [])
    while len(out) <= index:
        out.append([ax, ay])
    while len(inn) <= index:
        inn.append([ax, ay])
    out[index] = [float(x), float(y)]
    inn[index] = [2.0 * ax - float(x), 2.0 * ay - float(y)]


def set_in_handle(spec, index, x, y):
    """Pull anchor ``index``'s incoming handle to ``(x, y)``, mirroring the out."""
    anchors = spec.get("anchors") or []
    if not (0 <= index < len(anchors)):
        return
    ax, ay = anchors[index]
    out = spec.setdefault("handles_out", [])
    inn = spec.setdefault("handles_in", [])
    while len(out) <= index:
        out.append([ax, ay])
    while len(inn) <= index:
        inn.append([ax, ay])
    inn[index] = [float(x), float(y)]
    out[index] = [2.0 * ax - float(x), 2.0 * ay - float(y)]


def remove_anchor(spec, index):
    """Delete anchor ``index`` and its handles (in place).``"""
    for key in ("anchors", "handles_in", "handles_out"):
        entries = spec.get(key) or []
        if 0 <= index < len(entries):
            del entries[index]


def insert_anchor(spec, index, x, y):
    """Insert a corner anchor after ``index`` at ``(x, y)`` and return its index."""
    at = index + 1
    for key in ("anchors", "handles_in", "handles_out"):
        entries = spec.setdefault(key, [])
        at = min(at, len(entries))
        entries.insert(at, [float(x), float(y)])
    return at


def translate(spec, dx, dy):
    """Offset every anchor and handle of ``spec`` by ``(dx, dy)`` (in place)."""
    for key in ("anchors", "handles_in", "handles_out"):
        for point in spec.get(key) or []:
            point[0] += dx
            point[1] += dy
    return spec


# ------------------------------------------------------------------- rendering
def _cap_quad(draw, fill, point, dx, dy, nx, ny):
    """Extend a stroke past ``point`` along ``(dx, dy)`` for a square cap."""
    x, y = point
    r = math.hypot(nx, ny)
    tx = x + dx * r
    ty = y + dy * r
    draw.polygon(
        (
            (x + nx, y + ny),
            (tx + nx, ty + ny),
            (tx - nx, ty - ny),
            (x - nx, y - ny),
        ),
        fill=fill,
    )


def stroke_polyline(draw, points, fill, width, cap="round", join="round"):
    """Rasterise a stroked polyline without the seams of a wide ``line`` call.

    Pillow draws every segment of a wide line as its own quadrilateral and
    rounds each segment's geometry independently, so a dense or gently curved
    polyline (exactly what :func:`flatten` produces) keeps thin transparent
    slivers at the joins. Drawing the quads ourselves makes consecutive
    segments share their vertices, and a stamp at every point fills the wedge
    left at each corner (and caps the open ends). ``cap`` is ``round``,
    ``butt`` or ``square``; ``join`` is ``round`` or ``bevel``.
    """
    unique = []
    for point in points:
        x = float(point[0])
        y = float(point[1])
        if not unique or abs(x - unique[-1][0]) > 1e-6 \
                or abs(y - unique[-1][1]) > 1e-6:
            unique.append((x, y))
    if not unique:
        return
    if width <= 1:
        if len(unique) == 1:
            draw.point(unique[0], fill=fill)
        else:
            draw.line(unique, fill=fill)
        return
    radius = width / 2.0
    segments = []
    for start, end in zip(unique, unique[1:]):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        nx = -dy / length * radius
        ny = dx / length * radius
        draw.polygon(
            (
                (start[0] + nx, start[1] + ny),
                (end[0] + nx, end[1] + ny),
                (end[0] - nx, end[1] - ny),
                (start[0] - nx, start[1] - ny),
            ),
            fill=fill,
        )
        segments.append((start, end, dx / length, dy / length, nx, ny))
    if not segments:
        x, y = unique[0]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        return
    if cap == "round":
        for x, y in (unique[0], unique[-1]):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=fill
            )
    elif cap == "square":
        first = segments[0]
        _cap_quad(draw, fill, first[0], -first[2], -first[3], first[4], first[5])
        last = segments[-1]
        _cap_quad(draw, fill, last[1], last[2], last[3], last[4], last[5])
    if join == "round":
        for x, y in unique[1:-1]:
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius), fill=fill
            )
    else:
        for index in range(1, len(unique) - 1):
            incoming = segments[index - 1]
            outgoing = segments[index]
            vx, vy = unique[index]
            cross = incoming[2] * outgoing[3] - incoming[3] * outgoing[2]
            side = 1.0 if cross < 0 else -1.0
            draw.polygon(
                (
                    (vx, vy),
                    (vx + incoming[4] * side, vy + incoming[5] * side),
                    (vx + outgoing[4] * side, vy + outgoing[5] * side),
                ),
                fill=fill,
            )


def _paint(target, spec, origin, scale):
    """Paint ``spec`` onto a draw surface at ``origin`` with ``scale``."""
    mode = spec.get("mode", "fill")
    fill = mode in ("fill", "both")
    stroke = mode in ("stroke", "both")
    fill_color = rgba(spec.get("fill_color", "#000000"))
    stroke_color = rgba(spec.get("stroke_color", "#000000"))
    width = spec.get("width", 1)
    outline = spec.get("outline", 0)
    cap = spec.get("cap", "round")
    join = spec.get("join", "round")

    def tx(px, py):
        return ((px - origin[0]) * scale, (py - origin[1]) * scale)

    if fill:
        points = [tx(px, py) for px, py in flatten(spec, closed=True)]
        if len(points) >= 3:
            target.polygon(points, fill=fill_color)
            if mode == "fill" and outline > 0:
                stroke_polyline(
                    target, points + [points[0]], fill_color, outline * scale,
                    cap, join,
                )
    if stroke:
        points = [tx(px, py) for px, py in flatten(spec)]
        if spec.get("closed") and len(points) >= 2:
            points = points + [points[0]]
        if len(points) >= 2:
            stroke_polyline(target, points, stroke_color, width * scale, cap, join)


def paste(image, spec, offset=(0, 0)):
    """Composite ``spec``'s path onto ``image``, minus the layer ``offset``.

    ``offset`` is the layer's document offset, so anchors in document space land
    correctly in the layer buffer. The geometry's bounding box sizes the
    supersampled canvas, and any part outside ``image`` is clipped rather than
    raising. Returns the image (mutated in place).
    """
    if not spec.get("anchors"):
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    box = bounds(spec)
    if box is None:
        return image
    ox, oy = offset
    mode = spec.get("mode", "fill")
    antialias_on = bool(spec.get("antialias", True))
    feather = int(spec.get("feather", 0) or 0)
    opacity = int(spec.get("opacity", 100) or 100)
    width = float(spec.get("width", 1))
    outline = float(spec.get("outline", 0) or 0)
    geometry = flatten(
        spec, closed=True if mode in ("fill", "both") else bool(spec.get("closed"))
    )
    if not geometry:
        return image
    xs = [px - ox for px, _py in geometry]
    ys = [py - oy for _px, py in geometry]
    pad = int(math.ceil(max(width, outline) / 2.0)) + feather + 2
    left = max(0, int(math.floor(min(xs))) - pad)
    top = max(0, int(math.floor(min(ys))) - pad)
    right = min(image.width, int(math.ceil(max(xs))) + pad)
    bottom = min(image.height, int(math.ceil(max(ys))) + pad)
    if right <= left or bottom <= top:
        return image
    if not feather and opacity == 100 and not antialias_on:
        _paint(ImageDraw.Draw(image), spec, (ox, oy), 1)
        return image
    scale = (antialias.scale_for(right - left, bottom - top)
             if antialias_on else 1)
    temp = Image.new(
        "RGBA", ((right - left) * scale, (bottom - top) * scale), (0, 0, 0, 0)
    )
    _paint(ImageDraw.Draw(temp), spec, (ox + left, oy + top), scale)
    if scale > 1:
        temp = antialias.downscale(temp, scale)
    mask = temp.getchannel("A")
    if feather:
        mask = mask.filter(ImageFilter.GaussianBlur(feather))
    if opacity < 100:
        mask = mask.point(lambda value: value * opacity // 100)
    image.paste(temp, (left, top), mask)
    return image

