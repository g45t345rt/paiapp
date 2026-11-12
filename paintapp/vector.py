"""Editable vector strokes for vector layers.

A vector layer keeps its content as a list of continuous strokes - each a path
of document-space anchors with a uniform width, colour and opacity - rather
than as baked pixels. A stroke is either a straight polyline or a cubic bezier
curve: every anchor may carry an incoming and outgoing control handle, and a
handle that sits on its anchor is a corner. An old polyline is therefore just a
curve whose handles are all corners, so both kinds share one representation.
Keeping the geometry is what makes the vector tools possible: anchor points and
handles can be edited, one line's width can be corrected without touching its
neighbours, and the vector eraser can cut across a stroke and split it where the
eraser crossed. The layer's pixels are rebuilt from the specs at composite time,
exactly the way ``textrender`` keeps text editable over a layer's pixels (see
``Layer.composited_image``).

A stroke spec is a plain dict::

    {"points": [[x, y], ...],
     "handles_in": [[x, y], ...], "handles_out": [[x, y], ...],
     "width": 6, "color": "#000000", "opacity": 1.0,
     "closed": False, "cap": "round", "join": "round"}

``points`` are document pixels, ``handles_in``/``handles_out`` are the cubic
control points parallel to them (a handle equal to its anchor is a corner, and a
spec without handle arrays reads back as a plain polyline), and ``width`` is the
stroke's diameter in pixels, uniform along the stroke (so "correct line width"
is a single assignment). :func:`flatten` samples the geometry back into a plain
polyline for rendering and hit-testing, and ``paste`` is the one path that turns
a spec into pixels, shared by compositing and the destructive rasterise.
"""

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw

from . import antialias
from .constants import rgba
from .path import _cubic

MIN_WIDTH = 1
MAX_WIDTH = 2000
DEFAULT_WIDTH = 6

CAPS = ("round", "butt", "square")
JOINS = ("round", "miter")

# End-of-stroke markers. ``none`` draws nothing; the others are centred on the
# stroke end and point outwards along its tangent.
ARROW_STYLES = ("none", "triangle", "stealth", "circle", "square", "bar")

# Dash patterns are absolute lengths in document pixels (SVG's stroke-dasharray).
# A lone entry is repeated to make an even on/off pattern; values below this
# epsilon are treated as gaps and dropped.
DASH_EPSILON = 0.01

# Gradient fill types for a closed stroke or a compound fill.
GRADIENT_TYPES = ("linear", "radial")

# Blend modes an item may carry. The actual maths lives in ``brush``; this list
# only validates the stored id (kept in sync with brush.BLEND_MODES below).
BLEND_MODES = (
    "normal", "darken", "multiply", "color_burn", "linear_burn",
    "darker_color", "lighten", "screen", "color_dodge", "add",
    "lighter_color", "overlay", "soft_light", "hard_light", "vivid_light",
    "linear_light", "pin_light", "hard_mix", "difference", "exclusion",
    "subtract", "divide", "hue", "saturation", "color", "luminosity",
)

# Points closer than this to the previously kept point are dropped while a
# freehand stroke is recorded, so a vector line stays a handful of nodes rather
# than one node per motion event. Small enough that a curve keeps its shape.
SIMPLIFY_DISTANCE = 1.5

# How finely a curved segment is sampled back into a polyline for rendering and
# hit-testing. Straight (corner-to-corner) segments are not sampled at all, so a
# polyline keeps the exact segment list it always had.
CURVE_STEPS = 16

# A handle within this distance of its anchor is treated as a corner.
CORNER_EPSILON = 1e-6

# Upper bound on a fill's supersampled mask area (pixels). Keeps a large fill's
# rasterisation cheap enough that editing a layered canvas stays responsive.
FILL_SUPERSAMPLE_AREA = 2_000_000

# Rendered sprites are cached by content. Re-compositing an unchanged stroke or
# fill then reuses its raster instead of re-scanning it - which is what makes
# dragging a vector layer smooth, since moving the layer changes its offset and
# re-composites the layer every frame but leaves the geometry untouched. The
# cache is bounded by entry count and total pixels. Keys are translation
# invariant (see ``_shift_key``), so moving a vector layer - which translates
# every spec by a whole-pixel delta on release - reuses the rasters too.
_SPRITE_CACHE = {}
_SPRITE_CACHE_PIXELS = 0
_SPRITE_CACHE_BUDGET = 40_000_000
_SPRITE_CACHE_MAX = 256


def _shift_key(points):
    """The integer origin a sprite key is made relative to.

    Shifting every coordinate by ``floor`` of the first point makes the key of
    a spec translated by whole pixels identical, and the sprite rendered for it
    is the same raster at a shifted ``left``/``top``.
    """
    for point in points:
        return (math.floor(float(point[0])), math.floor(float(point[1])))
    return (0, 0)


def _rel(points, shift):
    sx, sy = shift
    return tuple((float(p[0]) - sx, float(p[1]) - sy) for p in points)


def _sprite_lookup(key):
    key, (sx, sy) = key
    entry = _SPRITE_CACHE.pop(key, None)
    if entry is None:
        return None
    # Re-insert so eviction is least-recently-used, not first-in.
    _SPRITE_CACHE[key] = entry
    sprite, left, top = entry[0]
    return sprite, left + sx, top + sy


def _sprite_store(key, sprite, left, top):
    global _SPRITE_CACHE_PIXELS
    key, (sx, sy) = key
    if key in _SPRITE_CACHE:
        return
    _SPRITE_CACHE[key] = ((sprite, left - sx, top - sy), sprite.width * sprite.height)
    _SPRITE_CACHE_PIXELS += sprite.width * sprite.height
    while len(_SPRITE_CACHE) > _SPRITE_CACHE_MAX:
        oldest = next(iter(_SPRITE_CACHE))
        _SPRITE_CACHE_PIXELS -= _SPRITE_CACHE.pop(oldest)[1]
    while _SPRITE_CACHE_PIXELS > _SPRITE_CACHE_BUDGET and len(_SPRITE_CACHE) > 1:
        oldest = next(iter(_SPRITE_CACHE))
        _SPRITE_CACHE_PIXELS -= _SPRITE_CACHE.pop(oldest)[1]


def _stroke_sprite_key(spec):
    handles_in, handles_out = handles(spec)
    points = spec.get("points") or []
    shift = _shift_key(points)
    return (
        spec.get("color"), spec.get("width"),
        round(float(spec.get("opacity", 1.0)), 4),
        bool(spec.get("closed", False)), spec.get("cap"), spec.get("join"),
        bool(spec.get("antialias", True)),
        repr(spec.get("fill")), repr(spec.get("fill_gradient")),
        tuple(round(float(value), 4) for value in anchor_widths(spec)),
        tuple(round(float(value), 4) for value in dash_pattern(spec)),
        round(float(spec.get("dash_offset", 0.0)), 4),
        spec.get("arrow_start"), spec.get("arrow_end"),
        round(float(spec.get("arrow_size", 0.0)), 4),
        _rel(points, shift),
        _rel(handles_in, shift),
        _rel(handles_out, shift),
    ), shift


def _fill_sprite_key(spec):
    contours = []
    shift = None
    for contour in spec.get("contours") or []:
        handles_in, handles_out = handles(contour)
        points = contour.get("points") or []
        if shift is None and points:
            shift = _shift_key(points)
        s = shift or (0, 0)
        contours.append((
            _rel(points, s),
            _rel(handles_in, s),
            _rel(handles_out, s),
        ))
    shift = shift or (0, 0)
    return (
        spec.get("color"), round(float(spec.get("opacity", 1.0)), 4),
        bool(spec.get("antialias", True)), bool(spec.get("even_odd", True)),
        repr(spec.get("gradient")),
        tuple(contours),
    ), shift


# ------------------------------------------------------------------- specs
def _opacity(spec):
    try:
        value = float(spec.get("opacity", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(1.0, value))


def _clean_points(raw):
    points = []
    for point in raw or []:
        try:
            points.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return points


def _clean_handles(raw, anchors):
    """Coerce ``raw`` to handle pairs, padding missing ones onto their anchors."""
    cleaned = []
    for point in raw or []:
        try:
            cleaned.append([float(point[0]), float(point[1])])
        except (TypeError, ValueError, IndexError):
            continue
    while len(cleaned) < len(anchors):
        cleaned.append(list(anchors[len(cleaned)]))
    return cleaned[:len(anchors)]


def handles(spec):
    """``(handles_in, handles_out)`` padded to the anchor count.

    A handle coinciding with its anchor is a corner, so a spec with no handle
    arrays - every stroke recorded before curves existed - reads back as the
    polyline it always was.
    """
    points = spec.get("points") or []
    return (
        _clean_handles(spec.get("handles_in"), points),
        _clean_handles(spec.get("handles_out"), points),
    )


def _clamp_width(value):
    return max(float(MIN_WIDTH), min(float(MAX_WIDTH), float(value)))


def _clean_widths(raw, anchors, base):
    """Per-anchor stroke widths, padded with the uniform ``base`` width."""
    widths = []
    for value in raw or []:
        try:
            widths.append(_clamp_width(value))
        except (TypeError, ValueError):
            continue
    while len(widths) < len(anchors):
        widths.append(float(base))
    return widths[:len(anchors)]


def anchor_widths(spec):
    """The stroke's width at each anchor (uniform when no profile is stored)."""
    points = spec.get("points") or []
    try:
        base = _clamp_width(spec.get("width", DEFAULT_WIDTH))
    except (TypeError, ValueError):
        base = float(DEFAULT_WIDTH)
    return _clean_widths(spec.get("widths"), points, base)


def has_width_profile(spec):
    """True when the stroke's width varies along its anchors."""
    widths = anchor_widths(spec)
    if len(widths) < 2:
        return False
    first = widths[0]
    return any(abs(width - first) > 1e-6 for width in widths[1:])


def _clean_dash(raw):
    """A non-negative dash pattern with an even number of entries."""
    pattern = []
    for value in raw or []:
        try:
            length = float(value)
        except (TypeError, ValueError):
            continue
        if length > DASH_EPSILON:
            pattern.append(length)
    if not pattern:
        return []
    if len(pattern) % 2:
        pattern = pattern + pattern
    return pattern


def dash_pattern(spec):
    """The stroke's dash pattern (empty for a solid line)."""
    pattern = spec.get("dash")
    if pattern is None:
        pattern = spec.get("dasharray")
    return _clean_dash(pattern)


def normalize_gradient(value):
    """Coerce ``value`` to a valid gradient paint, or ``None``.

    A gradient is a plain dict::

        {"type": "linear", "stops": [[0.0, "#000000"], [1.0, "#ffffff"]],
         "start": [x, y], "end": [x, y]}
        {"type": "radial", ..., "center": [x, y], "radius": r}

    Stops are sorted and clamped to ``0..1``; at least two are required. The
    start/end (or centre/radius) live in document pixels, so gradients travel
    with the geometry through moves, scales and rotations.
    """
    if not isinstance(value, dict):
        return None
    kind = str(value.get("type") or "linear").lower()
    if kind not in GRADIENT_TYPES:
        kind = "linear"
    stops = []
    for entry in value.get("stops") or []:
        try:
            offset = max(0.0, min(1.0, float(entry[0])))
            color = str(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        stops.append([offset, color])
    if len(stops) < 2:
        return None
    stops.sort(key=lambda entry: entry[0])

    def point(name, fallback):
        raw = value.get(name)
        try:
            return [float(raw[0]), float(raw[1])]
        except (TypeError, ValueError, IndexError):
            return list(fallback)

    if kind == "radial":
        try:
            radius = max(0.0, float(value.get("radius", 1.0)))
        except (TypeError, ValueError):
            radius = 1.0
        return {
            "type": "radial", "stops": stops,
            "center": point("center", (0.0, 0.0)), "radius": radius,
        }
    start = point("start", (0.0, 0.0))
    end = point("end", (start[0] + 1.0, start[1]))
    return {"type": "linear", "stops": stops, "start": start, "end": end}


def _clone_gradient(gradient):
    if gradient is None:
        return None
    copy = {
        "type": gradient.get("type", "linear"),
        "stops": [[float(entry[0]), str(entry[1])]
                  for entry in gradient.get("stops") or []],
    }
    for key in ("start", "end", "center"):
        if key in gradient:
            copy[key] = [float(gradient[key][0]), float(gradient[key][1])]
    if "radius" in gradient:
        copy["radius"] = float(gradient["radius"])
    return copy



def normalize(spec):
    """Coerce ``spec`` to a complete, valid stroke (or ``None`` when empty)."""
    if not isinstance(spec, dict):
        return None
    points = _clean_points(spec.get("points"))
    if not points:
        return None
    handles_in = _clean_handles(spec.get("handles_in"), points)
    handles_out = _clean_handles(spec.get("handles_out"), points)
    try:
        width = int(round(float(spec.get("width", DEFAULT_WIDTH))))
    except (TypeError, ValueError):
        width = DEFAULT_WIDTH
    width = max(MIN_WIDTH, min(MAX_WIDTH, width))
    color = str(spec.get("color") or "#000000")
    fill = spec.get("fill")
    fill = str(fill) if fill else None
    cap = spec.get("cap") if spec.get("cap") in CAPS else "round"
    join = spec.get("join") if spec.get("join") in JOINS else "round"
    arrow_start = (spec.get("arrow_start")
                   if spec.get("arrow_start") in ARROW_STYLES else "none")
    arrow_end = (spec.get("arrow_end")
                 if spec.get("arrow_end") in ARROW_STYLES else "none")
    try:
        arrow_size = max(0.0, float(spec.get("arrow_size", 0.0)))
    except (TypeError, ValueError):
        arrow_size = 0.0
    blend = spec.get("blend")
    blend = blend if blend in BLEND_MODES else "normal"
    try:
        dash_offset = float(spec.get("dash_offset", 0.0))
    except (TypeError, ValueError):
        dash_offset = 0.0
    widths = _clean_widths(spec.get("widths"), points, width)
    return {
        "points": points,
        "handles_in": handles_in,
        "handles_out": handles_out,
        "width": width,
        "widths": widths,
        "color": color,
        "fill": fill,
        "fill_gradient": normalize_gradient(spec.get("fill_gradient")),
        "opacity": _opacity(spec),
        "closed": bool(spec.get("closed", False)),
        "cap": cap,
        "join": join,
        "dash": _clean_dash(spec.get("dash")),
        "dash_offset": dash_offset,
        "arrow_start": arrow_start,
        "arrow_end": arrow_end,
        "arrow_size": arrow_size,
        "blend": blend,
        "antialias": bool(spec.get("antialias", True)),
    }


def clone(spec):
    """A deep copy of ``spec`` safe to keep in a snapshot/undo entry."""
    normalized = normalize(spec)
    if normalized is None:
        return None
    normalized["points"] = [[x, y] for x, y in normalized["points"]]
    normalized["handles_in"] = [[x, y] for x, y in normalized["handles_in"]]
    normalized["handles_out"] = [[x, y] for x, y in normalized["handles_out"]]
    normalized["widths"] = [float(value) for value in normalized["widths"]]
    normalized["dash"] = [float(value) for value in normalized["dash"]]
    normalized["fill_gradient"] = _clone_gradient(
        normalized.get("fill_gradient"))
    return normalized


def new_stroke(points, width, color, opacity=1.0, closed=False,
               cap="round", join="round", antialias=True,
               handles_in=None, handles_out=None, fill=None,
               widths=None, dash=None, dash_offset=0.0,
               arrow_start="none", arrow_end="none", arrow_size=0.0,
               fill_gradient=None, blend="normal"):
    """Build a normalised stroke from ``points`` (document pixels).

    ``handles_in``/``handles_out`` are the optional cubic control points; omitting
    them builds the straight polyline a freehand stroke always was. ``fill`` is
    an optional colour painted inside a closed stroke. ``widths`` is an optional
    per-anchor width profile, ``dash`` an SVG-style dash pattern, and the arrow
    options draw a marker at either end.
    """
    return normalize({
        "points": points, "width": width, "color": color, "opacity": opacity,
        "closed": closed, "cap": cap, "join": join, "antialias": antialias,
        "handles_in": handles_in, "handles_out": handles_out, "fill": fill,
        "widths": widths, "dash": dash, "dash_offset": dash_offset,
        "arrow_start": arrow_start, "arrow_end": arrow_end,
        "arrow_size": arrow_size, "fill_gradient": fill_gradient,
        "blend": blend,
    })


def describe(spec):
    """A short one-line label for a stroke, for the layer list."""
    count = len(spec.get("points", []))
    width = int(spec.get("width", DEFAULT_WIDTH))
    kind = "closed shape" if spec.get("closed") else "line"
    return f"{count} pt {kind} ({width}px)"


def simplify(points, distance=SIMPLIFY_DISTANCE):
    """Drop points closer than ``distance`` to the last point kept.

    The first point is always kept and the last recorded point is always kept,
    so a stroke keeps both ends even when the pointer barely moved.
    """
    if len(points) < 3:
        return [list(point) for point in points]
    kept = [list(points[0])]
    for point in points[1:-1]:
        if math.hypot(point[0] - kept[-1][0], point[1] - kept[-1][1]) >= distance:
            kept.append(list(point))
    kept.append(list(points[-1]))
    return kept


def _rdp(points, tolerance):
    """Ramer-Douglas-Peucker: the points a polyline needs to stay within
    ``tolerance`` of the original."""
    if len(points) < 3:
        return [list(point) for point in points]
    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]
    while stack:
        start, end = stack.pop()
        ax, ay = points[start]
        bx, by = points[end]
        worst = -1.0
        pivot = -1
        for index in range(start + 1, end):
            px, py = points[index]
            distance, _t = _distance_to_segment(px, py, ax, ay, bx, by)
            if distance > worst:
                worst = distance
                pivot = index
        if worst > tolerance and pivot != -1:
            keep[pivot] = True
            stack.append((start, pivot))
            stack.append((pivot, end))
    return [list(points[i]) for i in range(len(points)) if keep[i]]


def simplify_spec(spec, tolerance=SIMPLIFY_DISTANCE):
    """Decimate ``spec``'s anchors while keeping its shape within ``tolerance``.

    The stroke is flattened and simplified, then rebuilt as a corner polyline,
    so a curve being simplified is straightened - the "convert to straight line"
    result the Simplify tool gives. Both ends are always kept.
    """
    points = flatten(spec)
    if len(points) < 3:
        return spec
    profile = anchor_widths(spec)
    kept = _rdp(points, tolerance)
    spec["points"] = [[float(x), float(y)] for x, y in kept]
    spec["handles_in"] = [list(point) for point in spec["points"]]
    spec["handles_out"] = [list(point) for point in spec["points"]]
    average = (sum(profile) / len(profile)) if profile else float(
        spec.get("width", DEFAULT_WIDTH))
    spec["widths"] = [average] * len(spec["points"])
    return spec


# ---------------------------------------------------------------- geometry
def _is_corner(anchor, handle):
    return (abs(anchor[0] - handle[0]) < CORNER_EPSILON
            and abs(anchor[1] - handle[1]) < CORNER_EPSILON)


def _segment(p0, h0, h1, p1, steps):
    """Sample the segment ``p0 -> p1`` (controls ``h0``/``h1``), excluding ``p0``.

    A segment whose two handles sit on their anchors is a straight line, so it
    contributes only its endpoint - a polyline stroke stays as light as it was
    before curves existed.
    """
    if _is_corner(p0, h0) and _is_corner(p1, h1):
        return [[float(p1[0]), float(p1[1])]]
    return [[float(x), float(y)] for x, y in _cubic(
        tuple(p0), tuple(h0), tuple(h1), tuple(p1), steps)]


def flatten(spec, closed=None, steps=CURVE_STEPS):
    """Sample ``spec`` into a plain polyline in document pixels.

    Each pair of anchors is joined by a cubic bezier through its handles (a
    straight line when they are corners), so both kinds of stroke render and
    hit-test through one path. ``closed`` overrides the spec's own flag; when
    closed, the returned polyline ends on its first point.
    """
    points = spec.get("points") or []
    if not points:
        return []
    if closed is None:
        closed = bool(spec.get("closed", False))
    handles_in, handles_out = handles(spec)
    result = [[float(points[0][0]), float(points[0][1])]]
    for index in range(len(points) - 1):
        result.extend(_segment(
            points[index], handles_out[index],
            handles_in[index + 1], points[index + 1], steps))
    if closed and len(points) >= 2:
        result.extend(_segment(
            points[-1], handles_out[-1], handles_in[0], points[0], steps))
    return result


def bounds(spec):
    """The stroke's document-space bounding box, padded for width, or ``None``."""
    points = flatten(spec)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    widest = max(anchor_widths(spec) or [spec.get("width", DEFAULT_WIDTH)])
    pad = float(widest) / 2.0 + 1.0
    marker = max(_arrow_size(spec), 0.0)
    if marker > 0.0 and (spec.get("arrow_start", "none") != "none"
                         or spec.get("arrow_end", "none") != "none"):
        pad = max(pad, marker + 1.0)
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def _arrow_size(spec):
    """The marker size in document pixels (automatic when unset)."""
    try:
        size = float(spec.get("arrow_size", 0.0))
    except (TypeError, ValueError):
        size = 0.0
    if size > 0.0:
        return size
    widest = max(anchor_widths(spec) or [spec.get("width", DEFAULT_WIDTH)])
    return max(4.0, float(widest)) * 3.0


def _centerline(spec, steps=CURVE_STEPS):
    """Sample a stroke's centreline as ``(x, y, width)`` triples.

    Straight segments contribute only their endpoint (as :func:`flatten` does),
    while curved segments are sampled at ``steps``; the width is interpolated
    between the two anchors' profile values.
    """
    points = spec.get("points") or []
    if not points:
        return [], bool(spec.get("closed", False))
    widths = anchor_widths(spec)
    closed = bool(spec.get("closed", False))
    handles_in, handles_out = handles(spec)
    samples = [(float(points[0][0]), float(points[0][1]), widths[0])]

    def add_curve(p0, h0, h1, p1, w0, w1):
        if _is_corner(p0, h0) and _is_corner(p1, h1):
            samples.append((float(p1[0]), float(p1[1]), w1))
            return
        curve = _cubic(tuple(p0), tuple(h0), tuple(h1), tuple(p1), steps)
        for index, point in enumerate(curve, start=1):
            t = index / steps
            samples.append((float(point[0]), float(point[1]),
                            w0 + (w1 - w0) * t))

    for index in range(len(points) - 1):
        add_curve(points[index], handles_out[index],
                  handles_in[index + 1], points[index + 1],
                  widths[index], widths[index + 1])
    if closed and len(points) >= 2:
        add_curve(points[-1], handles_out[-1], handles_in[0], points[0],
                  widths[-1], widths[0])
        if len(samples) > 1 and math.hypot(
                samples[-1][0] - samples[0][0],
                samples[-1][1] - samples[0][1]) < 1e-9:
            samples.pop()
    return samples, closed


def _lerp_sample(a, b, t):
    return (a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t)


def _split_dash_samples(samples, pattern, offset=0.0):
    """Cut a width-tagged centreline into the "on" runs of a dash pattern."""
    if not pattern or len(samples) < 2:
        return [samples]
    cumulative = [0.0]
    for first, second in zip(samples, samples[1:]):
        cumulative.append(cumulative[-1] + math.hypot(
            second[0] - first[0], second[1] - first[1]))
    total = cumulative[-1]
    if total <= 1e-9:
        return [samples]

    def at(distance):
        distance = max(0.0, min(total, distance))
        lo, hi = 0, len(cumulative) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if cumulative[mid] <= distance:
                lo = mid
            else:
                hi = mid
        span = cumulative[hi] - cumulative[lo]
        t = 0.0 if span <= 1e-12 else (distance - cumulative[lo]) / span
        return _lerp_sample(samples[lo], samples[hi], t)

    period = sum(pattern)
    position = float(offset) % period
    index = 0
    while position >= pattern[index]:
        position -= pattern[index]
        index = (index + 1) % len(pattern)
    remaining = pattern[index] - position
    drawing = index % 2 == 0

    runs = []
    distance = 0.0
    while distance < total - 1e-9:
        step = min(remaining, total - distance)
        end = distance + step
        if drawing:
            run = [at(distance)]
            for point_index in range(1, len(samples) - 1):
                if distance < cumulative[point_index] < end:
                    run.append(samples[point_index])
            run.append(at(end))
            if len(run) >= 2:
                runs.append(run)
        distance = end
        remaining -= step
        if remaining <= 1e-9:
            index = (index + 1) % len(pattern)
            remaining = pattern[index]
            drawing = index % 2 == 0
    return runs


def _dot_outline(x, y, width, cap):
    """A closed polygon for a single-point (dot) stroke."""
    radius = max(0.5, width / 2.0)
    if cap == "square":
        return [(x - radius, y - radius), (x + radius, y - radius),
                (x + radius, y + radius), (x - radius, y + radius)]
    points = []
    for index in range(16):
        angle = 2.0 * math.pi * index / 16.0
        points.append((x + math.cos(angle) * radius,
                       y + math.sin(angle) * radius))
    return points


def _cap_arc(center, tangent, radius, at_start, steps=7):
    """Points around a round end cap, from the right side to the left side."""
    tx, ty = tangent
    length = math.hypot(tx, ty)
    if length <= 1e-9:
        tx, ty, length = 1.0, 0.0, 1.0
    tx, ty = tx / length, ty / length
    nx, ny = -ty, tx
    points = []
    for index in range(steps + 1):
        if at_start:
            angle = -math.pi / 2.0 + math.pi * index / steps
            ax = nx * math.sin(angle) - tx * math.cos(angle)
            ay = ny * math.sin(angle) - ty * math.cos(angle)
        else:
            angle = math.pi / 2.0 - math.pi * index / steps
            ax = nx * math.sin(angle) + tx * math.cos(angle)
            ay = ny * math.sin(angle) + ty * math.cos(angle)
        points.append((center[0] + ax * radius, center[1] + ay * radius))
    return points


def _outline_run(samples, closed, cap="round"):
    """A single outline polygon around a width-tagged polyline."""
    if len(samples) < 2:
        if samples:
            x, y, width = samples[0]
            return _dot_outline(x, y, width, cap)
        return []
    count = len(samples)
    left = []
    right = []
    tangents = []
    for index, (x, y, width) in enumerate(samples):
        if closed:
            previous = samples[(index - 1) % count]
            following = samples[(index + 1) % count]
        else:
            previous = samples[max(0, index - 1)]
            following = samples[min(count - 1, index + 1)]
        dx = following[0] - previous[0]
        dy = following[1] - previous[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            dx, dy, length = 1.0, 0.0, 1.0
        tx, ty = dx / length, dy / length
        tangents.append((tx, ty))
        radius = max(0.25, width / 2.0)
        nx, ny = -ty * radius, tx * radius
        left.append((x + nx, y + ny))
        right.append((x - nx, y - ny))
    if closed:
        return left + list(reversed(right))

    start_radius = max(0.25, samples[0][2] / 2.0)
    end_radius = max(0.25, samples[-1][2] / 2.0)
    start_tangent = tangents[0]
    end_tangent = tangents[-1]
    if cap == "square":
        sx = start_tangent[0] * start_radius
        sy = start_tangent[1] * start_radius
        ex = end_tangent[0] * end_radius
        ey = end_tangent[1] * end_radius
        polygon = [(left[0][0] - sx, left[0][1] - sy)]
        polygon.extend(left)
        polygon.append((left[-1][0] + ex, left[-1][1] + ey))
        polygon.append((right[-1][0] + ex, right[-1][1] + ey))
        polygon.extend(reversed(right))
        polygon.append((right[0][0] - sx, right[0][1] - sy))
        return polygon
    if cap == "round":
        polygon = list(left)
        polygon.extend(_cap_arc(samples[-1][:2], end_tangent, end_radius, False))
        polygon.extend(reversed(right))
        polygon.extend(_cap_arc(samples[0][:2], start_tangent, start_radius, True))
        return polygon
    polygon = list(left)
    polygon.extend(reversed(right))
    return polygon


def outline_polygons(spec):
    """The stroke's painted area as one closed polygon per dash run."""
    samples, closed = _centerline(spec)
    if not samples:
        return []
    if len(samples) == 1:
        x, y, width = samples[0]
        return [_dot_outline(x, y, width, spec.get("cap", "round"))]
    cap = spec.get("cap", "round")
    pattern = dash_pattern(spec)
    if pattern:
        runs = _split_dash_samples(
            samples, pattern, float(spec.get("dash_offset", 0.0)))
        return [_outline_run(run, False, cap) for run in runs if len(run) >= 2]
    return [_outline_run(samples, closed, cap)]


def stroke_area_paths(spec):
    """Closed polygons covering what a stroke paints (for boolean operations)."""
    polygons = list(outline_polygons(spec))
    if bool(spec.get("closed", False)) and (
            spec.get("fill") or spec.get("fill_gradient")):
        interior = flatten(spec, closed=True)
        if len(interior) >= 3:
            polygons.append(interior)
    return polygons


def _distance_to_segment(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-9:
        return math.hypot(px - ax, py - ay), 0.0
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy)), t


def path_points(spec):
    """The stroke's flattened segments as ``(ax, ay, bx, by)``."""
    points = flatten(spec)
    return [
        (a[0], a[1], b[0], b[1]) for a, b in zip(points, points[1:])
    ]


def _nearest_in(samples, x, y, index, best):
    """The nearest ``(index, fraction)`` over sampled ``(a, b)`` sub-segments."""
    best_distance = None if best is None else best[2]
    for step in range(len(samples) - 1):
        ax, ay = samples[step]
        bx, by = samples[step + 1]
        distance, local_t = _distance_to_segment(x, y, ax, ay, bx, by)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            fraction = (step + local_t) / (len(samples) - 1)
            best = (index, fraction, distance)
    return best


def _nearest_segment_detail(spec, x, y):
    """``(index, t, distance)`` of the nearest anchor segment, or ``None``."""
    points = spec.get("points") or []
    if len(points) < 2:
        return None
    closed = bool(spec.get("closed", False))
    handles_in, handles_out = handles(spec)
    best = None
    for index in range(len(points) - 1):
        samples = [points[index]] + _segment(
            points[index], handles_out[index],
            handles_in[index + 1], points[index + 1], CURVE_STEPS)
        best = _nearest_in(samples, x, y, index, best)
    if closed and len(points) >= 2:
        index = len(points) - 1
        samples = [points[-1]] + _segment(
            points[-1], handles_out[-1], handles_in[0], points[0], CURVE_STEPS)
        best = _nearest_in(samples, x, y, index, best)
    return best


def nearest_segment(spec, x, y):
    """``(index, t)`` of the anchor segment nearest ``(x, y)``, or ``None``.

    ``index`` names the anchor segment ``points[index] -> points[index + 1]``
    (or the closing segment when it is the last one) and ``t`` is the position
    along it, so a curve can be split exactly where the pointer landed.
    """
    best = _nearest_segment_detail(spec, x, y)
    if best is None:
        return None
    return best[0], best[1]


def hit_stroke(spec, x, y, tolerance=0.0):
    """True when ``(x, y)`` lies within the stroke's ink (plus ``tolerance``)."""
    points = spec.get("points") or []
    radius = float(spec.get("width", DEFAULT_WIDTH)) / 2.0 + tolerance
    if len(points) == 1:
        return math.hypot(x - points[0][0], y - points[0][1]) <= radius
    for ax, ay, bx, by in path_points(spec):
        distance, _t = _distance_to_segment(x, y, ax, ay, bx, by)
        if distance <= radius:
            return True
    return False


def anchor_hit(spec, x, y, tolerance):
    """Index of the anchor nearest ``(x, y)`` within ``tolerance``, or ``None``."""
    best = None
    best_distance = tolerance
    for index, (ax, ay) in enumerate(spec.get("points") or []):
        distance = math.hypot(x - ax, y - ay)
        if distance <= best_distance:
            best_distance = distance
            best = index
    return best


# ----------------------------------------------------------------- editing
def insert_point(spec, segment_index, x, y):
    """Insert a corner anchor on segment ``segment_index`` at ``(x, y)``."""
    points = spec.get("points") or []
    handles_in, handles_out = handles(spec)
    widths = anchor_widths(spec)
    at = min(len(points), segment_index + 1)
    point = [float(x), float(y)]
    points.insert(at, point)
    handles_in.insert(at, list(point))
    handles_out.insert(at, list(point))
    if at < len(widths):
        width = (widths[at - 1] + widths[at]) / 2.0
    else:
        width = widths[min(at - 1, len(widths) - 1)] if widths else DEFAULT_WIDTH
    widths.insert(at, width)
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out
    spec["widths"] = widths
    return at


def _lerp(a, b, t):
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def split_segment(spec, index, t):
    """Split anchor segment ``index`` at ``t`` (0..1), preserving its shape.

    The split runs de Casteljau's algorithm on the cubic, so the two halves draw
    exactly the curve the original did. Returns the new anchor's index, or
    ``None`` when the segment does not exist.
    """
    points = spec.get("points") or []
    if len(points) < 2:
        return None
    closed = bool(spec.get("closed", False))
    handles_in, handles_out = handles(spec)
    count = len(points)
    if index < count - 1:
        after = index + 1
        p0, p3 = points[index], points[index + 1]
        h0, h1 = handles_out[index], handles_in[index + 1]
    elif closed and index == count - 1:
        after = count
        p0, p3 = points[-1], points[0]
        h0, h1 = handles_out[-1], handles_in[0]
    else:
        return None
    t = max(0.0, min(1.0, float(t)))
    a = _lerp(p0, h0, t)
    b = _lerp(h0, h1, t)
    c = _lerp(h1, p3, t)
    d = _lerp(a, b, t)
    e = _lerp(b, c, t)
    mid = _lerp(d, e, t)
    points.insert(after, mid)
    handles_in.insert(after, d)
    handles_out.insert(after, e)
    handles_out[index] = a
    widths = anchor_widths(spec)
    widths.insert(after, widths[index] + (widths[index + 1] - widths[index]) * t
                 if after < len(widths) else widths[index])
    spec["widths"] = widths
    if after < count:
        handles_in[after + 1] = c
    else:
        handles_in[0] = c
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out
    return after


def move_point(spec, index, x, y):
    """Move anchor ``index`` and its handles to ``(x, y)`` (in place)."""
    points = spec.get("points") or []
    if not (0 <= index < len(points)):
        return
    dx = float(x) - points[index][0]
    dy = float(y) - points[index][1]
    points[index] = [float(x), float(y)]
    handles_in, handles_out = handles(spec)
    for entries in (handles_in, handles_out):
        entries[index] = [entries[index][0] + dx, entries[index][1] + dy]
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out


def set_handle(spec, index, side, x, y, mirror=True):
    """Pull anchor ``index``'s ``side`` handle to ``(x, y)`` (in place).

    With ``mirror`` the opposite handle is kept collinear through the anchor,
    so a smooth point stays smooth; without it only that side moves, leaving a
    corner.
    """
    points = spec.get("points") or []
    if not (0 <= index < len(points)):
        return
    handles_in, handles_out = handles(spec)
    entries = handles_in if side == "in" else handles_out
    entries[index] = [float(x), float(y)]
    if mirror:
        other = handles_out if side == "in" else handles_in
        anchor = points[index]
        other[index] = [2.0 * anchor[0] - float(x), 2.0 * anchor[1] - float(y)]
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out


def handle_hit(spec, x, y, tolerance):
    """``(side, index)`` of the bezier handle under ``(x, y)``, or ``None``.

    A handle sitting on its anchor (a corner) is ignored, so the anchor - not a
    hidden handle - is what the pointer grabs.
    """
    points = spec.get("points") or []
    handles_in, handles_out = handles(spec)
    for index, anchor in enumerate(points):
        for side, entries in (("in", handles_in), ("out", handles_out)):
            hx, hy = entries[index]
            if math.hypot(hx - anchor[0], hy - anchor[1]) < 0.5:
                continue
            if math.hypot(hx - x, hy - y) <= tolerance:
                return side, index
    return None


def set_corner(spec, index):
    """Collapse anchor ``index``'s handles onto the anchor (a sharp corner)."""
    points = spec.get("points") or []
    if not (0 <= index < len(points)):
        return
    handles_in, handles_out = handles(spec)
    handles_in[index] = list(points[index])
    handles_out[index] = list(points[index])
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out


def set_smooth(spec, index):
    """Make anchor ``index`` smooth: handles collinear along the neighbour chord."""
    points = spec.get("points") or []
    count = len(points)
    if not (0 <= index < count):
        return
    closed = bool(spec.get("closed", False))
    if closed and count >= 2:
        previous = points[(index - 1) % count]
        following = points[(index + 1) % count]
    else:
        previous = points[index - 1] if index > 0 else None
        following = points[index + 1] if index < count - 1 else None
    anchor = points[index]
    handles_in, handles_out = handles(spec)
    if previous is None or following is None:
        neighbour = following if following is not None else previous
        if neighbour is None:
            set_corner(spec, index)
            return
        dx, dy = neighbour[0] - anchor[0], neighbour[1] - anchor[1]
        length = math.hypot(dx, dy)
        if length < 1e-6:
            set_corner(spec, index)
            return
        ux, uy = dx / length, dy / length
        if following is not None:
            handles_out[index] = [anchor[0] + ux * length / 3.0,
                                  anchor[1] + uy * length / 3.0]
            handles_in[index] = list(anchor)
        else:
            handles_in[index] = [anchor[0] - ux * length / 3.0,
                                 anchor[1] - uy * length / 3.0]
            handles_out[index] = list(anchor)
        spec["handles_in"] = handles_in
        spec["handles_out"] = handles_out
        return
    tx, ty = following[0] - previous[0], following[1] - previous[1]
    length = math.hypot(tx, ty)
    if length < 1e-6:
        set_corner(spec, index)
        return
    ux, uy = tx / length, ty / length
    d_in = math.hypot(anchor[0] - previous[0], anchor[1] - previous[1]) / 3.0
    d_out = math.hypot(following[0] - anchor[0], following[1] - anchor[1]) / 3.0
    handles_in[index] = [anchor[0] - ux * d_in, anchor[1] - uy * d_in]
    handles_out[index] = [anchor[0] + ux * d_out, anchor[1] + uy * d_out]
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out


def toggle_closed(spec):
    """Flip the stroke's closed flag (in place) and return the new value."""
    spec["closed"] = not bool(spec.get("closed", False))
    return spec["closed"]


def split_at(spec, index):
    """Split ``spec`` at anchor ``index``.

    An open stroke becomes two strokes that meet at the point; a closed stroke
    is opened there instead (it stays one stroke). Returns the resulting list.
    """
    points = spec.get("points") or []
    count = len(points)
    if count < 2 or not (0 <= index < count):
        return [spec]
    handles_in, handles_out = handles(spec)
    widths = anchor_widths(spec)
    if bool(spec.get("closed", False)):
        order = list(range(index, count)) + list(range(0, index)) + [index]
        opened = clone(spec)
        opened["closed"] = False
        opened["points"] = [list(points[i]) for i in order]
        opened["handles_in"] = [list(handles_in[i]) for i in order]
        opened["handles_out"] = [list(handles_out[i]) for i in order]
        opened["widths"] = [float(widths[i]) for i in order]
        opened["handles_in"][0] = list(points[index])
        opened["handles_out"][-1] = list(points[index])
        return [opened]
    if index <= 0 or index >= count - 1:
        return [spec]
    first = clone(spec)
    first["points"] = [list(points[i]) for i in range(0, index + 1)]
    first["handles_in"] = [list(handles_in[i]) for i in range(0, index + 1)]
    first["handles_out"] = [list(handles_out[i]) for i in range(0, index + 1)]
    first["widths"] = [float(widths[i]) for i in range(0, index + 1)]
    second = clone(spec)
    second["points"] = [list(points[i]) for i in range(index, count)]
    second["handles_in"] = [list(handles_in[i]) for i in range(index, count)]
    second["handles_out"] = [list(handles_out[i]) for i in range(index, count)]
    second["widths"] = [float(widths[i]) for i in range(index, count)]
    return [first, second]


def remove_point(spec, index):
    """Delete anchor ``index`` and its handles (in place).

    Returns ``True`` when it was removed.
    """
    points = spec.get("points") or []
    if not (0 <= index < len(points)):
        return False
    handles_in, handles_out = handles(spec)
    del points[index]
    del handles_in[index]
    del handles_out[index]
    widths = anchor_widths(spec)
    if 0 <= index < len(widths):
        del widths[index]
    spec["handles_in"] = handles_in
    spec["handles_out"] = handles_out
    spec["widths"] = widths
    return True


def correct_width(spec, width):
    """Set the stroke's uniform width (the line-width correction), in place."""
    value = max(MIN_WIDTH, min(MAX_WIDTH, int(round(width))))
    spec["width"] = value
    if spec.get("widths"):
        spec["widths"] = [float(value)] * len(spec.get("points") or [])
    return value


def scale_width(spec, factor, minimum=MIN_WIDTH, maximum=MAX_WIDTH):
    """Multiply the stroke's width by ``factor`` (thicken/narrow), in place."""
    current = float(spec.get("width", DEFAULT_WIDTH))
    spec["width"] = max(minimum, min(maximum, int(round(current * factor))))
    if spec.get("widths"):
        spec["widths"] = [
            max(float(minimum), min(float(maximum), float(value) * factor))
            for value in spec["widths"]
        ]
    return spec["width"]


def set_anchor_width(spec, index, width):
    """Set the width at one anchor, building a profile from the uniform width."""
    points = spec.get("points") or []
    if not (0 <= index < len(points)):
        return False
    widths = anchor_widths(spec)
    widths[index] = _clamp_width(width)
    spec["widths"] = widths
    spec["width"] = int(round(max(widths)))
    return True


def width_at(spec, x, y):
    """``(anchor_index, width)`` of the nearest anchor to ``(x, y)``."""
    points = spec.get("points") or []
    widths = anchor_widths(spec)
    if not points:
        return None, None
    best = min(range(len(points)),
               key=lambda index: math.hypot(
                   x - points[index][0], y - points[index][1]))
    return best, widths[best]


def smooth(points, strength):
    """Smooth a recorded polyline in place-free fashion (``strength`` 0..100).

    Each pass replaces every interior point with a weighted average of itself
    and its neighbours while pinning the two ends, so a freehand line loses its
    pointer jitter without shrinking away from where it was drawn. ``strength``
    picks how many passes run.
    """
    result = [[float(x), float(y)] for x, y in points]
    passes = max(0, int(strength)) // 20
    for _pass in range(passes):
        if len(result) < 3:
            break
        smoothed = [result[0]]
        for index in range(1, len(result) - 1):
            px, py = result[index - 1]
            cx, cy = result[index]
            nx, ny = result[index + 1]
            smoothed.append([
                (px + 2.0 * cx + nx) / 4.0,
                (py + 2.0 * cy + ny) / 4.0,
            ])
        smoothed.append(result[-1])
        result = smoothed
    return result


def translate(spec, dx, dy):
    """Offset every anchor and handle of ``spec`` by ``(dx, dy)`` (in place)."""
    for key in ("points", "handles_in", "handles_out"):
        for point in spec.get(key) or []:
            point[0] += dx
            point[1] += dy
    gradient = spec.get("fill_gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                point[0] += dx
                point[1] += dy
    return spec


def rotate(spec, angle, cx, cy):
    """Rotate ``spec``'s anchors and handles by ``angle`` radians about ``(cx, cy)``."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for key in ("points", "handles_in", "handles_out"):
        for point in spec.get(key) or []:
            ox = point[0] - cx
            oy = point[1] - cy
            point[0] = cx + ox * cos_a - oy * sin_a
            point[1] = cy + ox * sin_a + oy * cos_a
    gradient = spec.get("fill_gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                ox = point[0] - cx
                oy = point[1] - cy
                point[0] = cx + ox * cos_a - oy * sin_a
                point[1] = cy + ox * sin_a + oy * cos_a
    return spec


def scale(spec, sx, sy, cx, cy):
    """Scale ``spec``'s anchors and handles by ``(sx, sy)`` about ``(cx, cy)``."""
    for key in ("points", "handles_in", "handles_out"):
        for point in spec.get(key) or []:
            point[0] = cx + (point[0] - cx) * sx
            point[1] = cy + (point[1] - cy) * sy
    gradient = spec.get("fill_gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                point[0] = cx + (point[0] - cx) * sx
                point[1] = cy + (point[1] - cy) * sy
        if "radius" in gradient:
            gradient["radius"] = float(gradient["radius"]) * math.sqrt(
                abs(sx * sy))
    factor = math.sqrt(abs(sx * sy))
    if abs(factor - 1.0) > 1e-9:
        if spec.get("widths"):
            spec["widths"] = [_clamp_width(value * factor)
                              for value in spec["widths"]]
        spec["width"] = max(MIN_WIDTH, min(
            MAX_WIDTH, int(round(float(spec.get("width", DEFAULT_WIDTH)) * factor))))
    return spec


def union_bounds(specs):
    """The union of :func:`bounds` over ``specs``, or ``None`` when empty."""
    boxes = [box for box in (bounds(spec) for spec in specs) if box]
    if not boxes:
        return None
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def erase(stroke, cx, cy, radius, mode="touch"):
    """Cut the eraser disc ``(cx, cy, radius)`` out of ``stroke``.

    ``mode`` is ``"touch"`` (remove the anchors the disc covers, splitting the
    stroke where a run is removed from the middle or an end) or ``"whole"``
    (remove the entire stroke as soon as the disc touches it). Returns the list
    of strokes left behind: the original when nothing was touched, an empty list
    when the stroke was swallowed whole, and one or two strokes when a run was
    removed from the middle or an end. Anchors are the unit of removal, so a
    stroke is cut at its nodes rather than mid-segment - the same trade-off the
    anchor editor makes.
    """
    points = stroke.get("points") or []
    if not points:
        return []
    inside = [
        math.hypot(x - cx, y - cy) <= radius for x, y in points
    ]
    if not any(inside):
        return [stroke]
    if mode == "whole":
        return []
    survivors = [index for index, covered in enumerate(inside) if not covered]
    if not survivors:
        return []
    handles_in, handles_out = handles(stroke)
    widths = anchor_widths(stroke)
    runs = []
    current = [survivors[0]] if survivors else []
    for index in survivors[1:]:
        if index == current[-1] + 1:
            current.append(index)
        else:
            runs.append(current)
            current = [index]
    if current:
        runs.append(current)
    # A lone surviving node is not a line; drop it unless the original was a dot.
    pieces = []
    for run in runs:
        if len(run) == 1 and len(points) > 1:
            continue
        piece = clone(stroke)
        piece["points"] = [list(points[index]) for index in run]
        piece["handles_in"] = [list(handles_in[index]) for index in run]
        piece["handles_out"] = [list(handles_out[index]) for index in run]
        piece["widths"] = [float(widths[index]) for index in run]
        pieces.append(piece)
    return pieces


def _reverse(spec):
    """A clone of ``spec`` drawn from its other end (its handles are swapped)."""
    out = clone(spec)
    out["points"] = list(reversed(out["points"]))
    old_in = out["handles_in"]
    old_out = out["handles_out"]
    out["handles_in"] = list(reversed(old_out))
    out["handles_out"] = list(reversed(old_in))
    out["widths"] = list(reversed(out.get("widths") or []))
    return out


def _join(a, b, tolerance):
    """Join ``a`` and ``b`` at their nearest open ends, or ``None``."""
    for first in (a, _reverse(a)):
        for second in (b, _reverse(b)):
            end = first["points"][-1]
            start = second["points"][0]
            if math.hypot(end[0] - start[0], end[1] - start[1]) > tolerance:
                continue
            merged = clone(first)
            merged["points"] = first["points"] + second["points"][1:]
            merged["handles_in"] = first["handles_in"] + second["handles_in"][1:]
            merged["handles_out"] = first["handles_out"] + second["handles_out"][1:]
            merged["widths"] = ((first.get("widths") or [])
                                + (second.get("widths") or [])[1:])
            merged["closed"] = False
            return merged
    return None


def connect(strokes, tolerance=4.0):
    """Join open strokes whose nearest ends meet within ``tolerance``.

    Chains are merged repeatedly until no two ends are close enough, so several
    selected stubs become one line. Closed strokes are left untouched. Returns a
    new list; the originals are not modified.
    """
    result = [clone(spec) for spec in strokes if spec.get("points")]
    merged = True
    while merged:
        merged = False
        for i in range(len(result)):
            if result[i] is None or bool(result[i].get("closed")):
                continue
            for j in range(len(result)):
                if i == j or result[j] is None or bool(result[j].get("closed")):
                    continue
                joined = _join(result[i], result[j], tolerance)
                if joined is not None:
                    result[i] = joined
                    result[j] = None
                    result = [spec for spec in result if spec is not None]
                    merged = True
                    break
            if merged:
                break
    return result


# ------------------------------------------------------- booleans and objects
def _item_paint_paths(spec):
    """Closed polygons covering an item's painted area."""
    if is_fill(spec):
        return fill_contour_paths(spec)
    return stroke_area_paths(spec)


def _trace_mask(mask):
    """Link the boundary of a binary (``L``) mask into closed polygons.

    Marching squares over pixel cells, joined at shared edge midpoints. The
    result is in mask pixel coordinates and may contain several loops (outer
    boundaries plus holes); callers convert and simplify.
    """
    width, height = mask.size
    pixels = mask.load()

    def inside(x, y):
        return 0 <= x < width and 0 <= y < height and pixels[x, y] >= 128

    segments = []
    for y in range(-1, height):
        for x in range(-1, width):
            tl = inside(x, y)
            tr = inside(x + 1, y)
            br = inside(x + 1, y + 1)
            bl = inside(x, y + 1)
            code = (8 if tl else 0) | (4 if tr else 0) | (2 if br else 0) | (
                1 if bl else 0)
            if code in (0, 15):
                continue
            top = (x + 0.5, y)
            right = (x + 1.0, y + 0.5)
            bottom = (x + 0.5, y + 1.0)
            left = (x, y + 0.5)
            table = {
                1: [(left, bottom)],
                2: [(bottom, right)],
                3: [(left, right)],
                4: [(right, top)],
                5: [(left, top), (bottom, right)],
                6: [(bottom, top)],
                7: [(left, top)],
                8: [(top, left)],
                9: [(top, bottom)],
                10: [(top, right), (bottom, left)],
                11: [(top, right)],
                12: [(left, right)],
                13: [(bottom, right)],
                14: [(left, bottom)],
            }
            segments.extend(table.get(code, ()))

    def key(point):
        return (round(point[0], 6), round(point[1], 6))

    adjacency = {}
    for index, (start, end) in enumerate(segments):
        adjacency.setdefault(key(start), []).append(index)
        adjacency.setdefault(key(end), []).append(index)

    used = [False] * len(segments)
    loops = []
    for index in range(len(segments)):
        if used[index]:
            continue
        start, current = segments[index]
        used[index] = True
        loop = [start, current]
        closed = False
        while True:
            if key(current) == key(loop[0]):
                closed = True
                break
            next_index = None
            for candidate in adjacency.get(key(current), ()):
                if not used[candidate]:
                    next_index = candidate
                    break
            if next_index is None:
                break
            used[next_index] = True
            first, second = segments[next_index]
            current = second if key(first) == key(current) else first
            loop.append(current)
        if closed and len(loop) >= 4:
            loops.append(loop[:-1])
    return loops


def _polygon_area(points):
    total = 0.0
    for first, second in zip(points, points[1:] + points[:1]):
        total += first[0] * second[1] - second[0] * first[1]
    return abs(total) / 2.0


def boolean(items, operation="union", max_dimension=1600, resolution=None):
    """Combine selected items into one editable compound fill.

    ``operation`` is ``"union"``, ``"subtract"``, ``"intersect"`` or
    ``"exclude"``. The shapes are flattened, rasterised together at a bounded
    resolution, then traced back into polyline contours, so the result stays
    editable (anchors can be moved, the fill re-styled) if not mathematically
    exact. Returns a fill spec, or ``None`` when nothing remains.
    """
    specs = [spec for spec in items if spec]
    if len(specs) < 2:
        return None
    paths = [_item_paint_paths(spec) for spec in specs]
    paths = [polygons for polygons in paths if polygons]
    if len(paths) < 2:
        return None
    xs = [point[0] for polygons in paths for poly in polygons for point in poly]
    ys = [point[1] for polygons in paths for poly in polygons for point in poly]
    if not xs:
        return None
    left = math.floor(min(xs)) - 1
    top = math.floor(min(ys)) - 1
    span_x = max(1.0, math.ceil(max(xs)) + 1 - left)
    span_y = max(1.0, math.ceil(max(ys)) + 1 - top)
    if resolution is None:
        resolution = min(2.0, max(0.5, max_dimension / max(span_x, span_y)))
    size = (max(1, int(span_x * resolution)),
            max(1, int(span_y * resolution)))
    masks = []
    for polygons in paths:
        mask = Image.new("L", size, 0)
        pen = ImageDraw.Draw(mask)
        for polygon in polygons:
            if len(polygon) >= 3:
                pen.polygon(
                    [((point[0] - left) * resolution,
                      (point[1] - top) * resolution) for point in polygon],
                    fill=255)
        masks.append(mask)
    combined = masks[0]
    for mask in masks[1:]:
        if operation == "union":
            combined = ImageChops.lighter(combined, mask)
        elif operation == "intersect":
            combined = ImageChops.multiply(combined, mask)
        elif operation == "subtract":
            combined = ImageChops.subtract(combined, mask)
        elif operation == "exclude":
            combined = ImageChops.difference(combined, mask)
        else:
            return None
    contours = []
    for loop in _trace_mask(combined):
        polygon = [[left + x / resolution, top + y / resolution]
                   for x, y in loop]
        polygon = _rdp(polygon, 0.75 / resolution)
        if len(polygon) >= 3 and _polygon_area(polygon) >= 0.5:
            contours.append({
                "points": [[float(x), float(y)] for x, y in polygon],
                "handles_in": [[float(x), float(y)] for x, y in polygon],
                "handles_out": [[float(x), float(y)] for x, y in polygon],
            })
    if not contours:
        return None
    first = specs[0]
    if is_fill(first):
        color = first.get("color")
        gradient = first.get("gradient")
        opacity = first.get("opacity", 1.0)
        blend = first.get("blend", "normal")
    else:
        color = first.get("color")
        gradient = first.get("fill_gradient")
        opacity = first.get("opacity", 1.0)
        blend = first.get("blend", "normal")
    return new_fill(contours, color, opacity=opacity, even_odd=True,
                    gradient=gradient, blend=blend)


def duplicate_item(spec, dx=0.0, dy=0.0):
    """A translated deep copy of a vector item."""
    copy = clone_fill(spec) if is_fill(spec) else clone(spec)
    if copy is not None and (dx or dy):
        item_translate(copy, dx, dy)
    return copy


def align_items(specs, edge):
    """Align items to one edge of their combined bounds (in place)."""
    specs = [spec for spec in specs if spec]
    boxes = [(spec, item_bounds(spec)) for spec in specs]
    boxes = [(spec, box) for spec, box in boxes if box]
    if not boxes:
        return specs
    left = min(box[0] for _spec, box in boxes)
    top = min(box[1] for _spec, box in boxes)
    right = max(box[2] for _spec, box in boxes)
    bottom = max(box[3] for _spec, box in boxes)
    for spec, box in boxes:
        dx = dy = 0.0
        if edge == "left":
            dx = left - box[0]
        elif edge == "right":
            dx = right - box[2]
        elif edge == "top":
            dy = top - box[1]
        elif edge == "bottom":
            dy = bottom - box[3]
        elif edge == "hcenter":
            dx = (left + right) / 2.0 - (box[0] + box[2]) / 2.0
        elif edge == "vcenter":
            dy = (top + bottom) / 2.0 - (box[1] + box[3]) / 2.0
        if dx or dy:
            item_translate(spec, dx, dy)
    return specs


def distribute_items(specs, axis="h"):
    """Space items evenly between the first and last along ``axis`` (in place)."""
    boxes = [(spec, item_bounds(spec)) for spec in specs if spec]
    boxes = [(spec, box) for spec, box in boxes if box]
    if len(boxes) < 3:
        return [spec for spec, _box in boxes]
    if axis == "v":
        boxes.sort(key=lambda entry: (entry[1][1] + entry[1][3]) / 2.0)
        start = (boxes[0][1][1] + boxes[0][1][3]) / 2.0
        end = (boxes[-1][1][1] + boxes[-1][1][3]) / 2.0
    else:
        boxes.sort(key=lambda entry: (entry[1][0] + entry[1][2]) / 2.0)
        start = (boxes[0][1][0] + boxes[0][1][2]) / 2.0
        end = (boxes[-1][1][0] + boxes[-1][1][2]) / 2.0
    step = (end - start) / (len(boxes) - 1)
    for index, (spec, box) in enumerate(boxes[1:-1], start=1):
        target = start + step * index
        if axis == "v":
            current = (box[1] + box[3]) / 2.0
            if abs(target - current) > 1e-9:
                item_translate(spec, 0.0, target - current)
        else:
            current = (box[0] + box[2]) / 2.0
            if abs(target - current) > 1e-9:
                item_translate(spec, target - current, 0.0)
    return [spec for spec, _box in boxes]


def reorder_items(items, indices, position):
    """Return ``items`` with ``indices`` moved to ``position``.

    ``position`` is ``"front"``, ``"back"``, ``"forward"`` or ``"backward"``.
    ``items`` is any ordered list of specs (a layer's strokes or fills).
    """
    items = list(items)
    selected = [index for index in sorted(set(indices))
                if 0 <= index < len(items)]
    if not selected:
        return items
    chosen = [items[index] for index in selected]
    if position in ("front", "back"):
        rest = [spec for index, spec in enumerate(items) if index not in selected]
        return (rest + chosen) if position == "front" else (chosen + rest)
    if position == "forward":
        for index in reversed(selected):
            if index + 1 < len(items) and (index + 1) not in selected:
                items[index], items[index + 1] = items[index + 1], items[index]
    elif position == "backward":
        for index in selected:
            if index - 1 >= 0 and (index - 1) not in selected:
                items[index], items[index - 1] = items[index - 1], items[index]
    return items


# ------------------------------------------------------------- filled areas
def normalize_fill(spec):
    """Coerce ``spec`` to a valid compound fill spec (or ``None``).

    A fill spec paints one or more closed contours with a single colour. Where
    contours overlap, ``even_odd`` picks the rule: ``True`` cuts a hole wherever
    the coverage is even (SVG's ``fill-rule="evenodd"``), which is how a filled
    glyph draws its knocked-out details.
    """
    if not isinstance(spec, dict):
        return None
    contours = []
    for entry in spec.get("contours") or []:
        if not isinstance(entry, dict):
            continue
        points = _clean_points(entry.get("points"))
        if len(points) < 2:
            continue
        contours.append({
            "points": points,
            "handles_in": _clean_handles(entry.get("handles_in"), points),
            "handles_out": _clean_handles(entry.get("handles_out"), points),
        })
    if not contours:
        return None
    blend = spec.get("blend")
    return {
        "contours": contours,
        "color": str(spec.get("color") or "#000000"),
        "gradient": normalize_gradient(spec.get("gradient")),
        "opacity": _opacity(spec),
        "antialias": bool(spec.get("antialias", True)),
        "even_odd": bool(spec.get("even_odd", True)),
        "blend": blend if blend in BLEND_MODES else "normal",
    }


def clone_fill(spec):
    """A deep copy of a fill spec, safe for a snapshot/undo entry."""
    normalized = normalize_fill(spec)
    if normalized is None:
        return None
    for contour in normalized["contours"]:
        contour["points"] = [[x, y] for x, y in contour["points"]]
        contour["handles_in"] = [[x, y] for x, y in contour["handles_in"]]
        contour["handles_out"] = [[x, y] for x, y in contour["handles_out"]]
    normalized["gradient"] = _clone_gradient(normalized.get("gradient"))
    return normalized


def new_fill(contours, color, opacity=1.0, antialias=True, even_odd=True,
             gradient=None, blend="normal"):
    """Build a normalised compound fill from ``contours``."""
    return normalize_fill({
        "contours": contours, "color": color, "opacity": opacity,
        "antialias": antialias, "even_odd": even_odd,
        "gradient": gradient, "blend": blend,
    })


def fill_contour_paths(spec):
    """The fill's flattened contours as closed polylines."""
    paths = []
    for contour in spec.get("contours") or []:
        points = flatten(contour, closed=True)
        if len(points) >= 3:
            paths.append(points)
    return paths


def fill_bounds(spec):
    """The fill's document-space bounding box, or ``None``."""
    xs = []
    ys = []
    for points in fill_contour_paths(spec):
        xs.extend(point[0] for point in points)
        ys.extend(point[1] for point in points)
    if not xs:
        return None
    return min(xs) - 1.0, min(ys) - 1.0, max(xs) + 1.0, max(ys) + 1.0


def translate_fill(spec, dx, dy):
    """Offset every contour point and handle of ``spec`` (in place)."""
    for contour in spec.get("contours") or []:
        for key in ("points", "handles_in", "handles_out"):
            for point in contour.get(key) or []:
                point[0] += dx
                point[1] += dy
    gradient = spec.get("gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                point[0] += dx
                point[1] += dy
    return spec


def scale_fill(spec, sx, sy, cx, cy):
    """Scale every contour point and handle of ``spec`` about ``(cx, cy)``."""
    for contour in spec.get("contours") or []:
        for key in ("points", "handles_in", "handles_out"):
            for point in contour.get(key) or []:
                point[0] = cx + (point[0] - cx) * sx
                point[1] = cy + (point[1] - cy) * sy
    gradient = spec.get("gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                point[0] = cx + (point[0] - cx) * sx
                point[1] = cy + (point[1] - cy) * sy
        if "radius" in gradient:
            gradient["radius"] = float(gradient["radius"]) * math.sqrt(
                abs(sx * sy))
    return spec


def rotate_fill(spec, angle, cx, cy):
    """Rotate every contour point and handle of ``spec`` about ``(cx, cy)``."""
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    for contour in spec.get("contours") or []:
        for key in ("points", "handles_in", "handles_out"):
            for point in contour.get(key) or []:
                ox = point[0] - cx
                oy = point[1] - cy
                point[0] = cx + ox * cos_a - oy * sin_a
                point[1] = cy + ox * sin_a + oy * cos_a
    gradient = spec.get("gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                ox = point[0] - cx
                oy = point[1] - cy
                point[0] = cx + ox * cos_a - oy * sin_a
                point[1] = cy + ox * sin_a + oy * cos_a
    return spec


def describe_fill(spec):
    """A short one-line label for a fill, for the layer list."""
    count = len(spec.get("contours") or [])
    return f"{count} contour fill"


# --------------------------------------------------------------- item editing
# A vector "item" is either a stroke (a single contour whose geometry lives on
# the spec itself) or a compound fill (a list of contours). These helpers give
# the anchor-editing tools one interface over both, addressing anchors by a
# global index that runs through the contours in order.
def is_fill(spec):
    """True when ``spec`` is a compound fill (it carries ``contours``)."""
    return bool(spec.get("contours"))


def item_contours(spec):
    """The editable contours of ``spec`` (a stroke is its own single contour)."""
    contours = spec.get("contours")
    if contours:
        return contours
    return [spec]


def item_anchor(spec, index):
    """``(contour, local_index)`` for global anchor ``index``, or ``None``."""
    remaining = index
    for contour in item_contours(spec):
        count = len(contour.get("points") or [])
        if remaining < count:
            return contour, remaining
        remaining -= count
    return None


def item_anchor_points(spec):
    """Every anchor of ``spec`` in :func:`item_anchor` order."""
    points = []
    for contour in item_contours(spec):
        for point in contour.get("points") or []:
            points.append([float(point[0]), float(point[1])])
    return points


def item_anchor_hit(spec, x, y, tolerance):
    """Global index of the nearest anchor within ``tolerance``, or ``None``."""
    best = None
    best_distance = tolerance
    index = 0
    for contour in item_contours(spec):
        for ax, ay in contour.get("points") or []:
            distance = math.hypot(x - ax, y - ay)
            if distance <= best_distance:
                best_distance = distance
                best = index
            index += 1
    return best


def item_handle_hit(spec, x, y, tolerance):
    """``(side, global_index)`` of the handle under ``(x, y)``, or ``None``."""
    index = 0
    for contour in item_contours(spec):
        points = contour.get("points") or []
        handles_in, handles_out = handles(contour)
        for local, anchor in enumerate(points):
            for side, entries in (("in", handles_in), ("out", handles_out)):
                hx, hy = entries[local]
                if math.hypot(hx - anchor[0], hy - anchor[1]) < 0.5:
                    continue
                if math.hypot(hx - x, hy - y) <= tolerance:
                    return side, index + local
        index += len(points)
    return None


def item_handle_pairs(spec):
    """Bezier handle stubs ``[anchor_x, anchor_y, handle_x, handle_y]``."""
    stubs = []
    for contour in item_contours(spec):
        handles_in, handles_out = handles(contour)
        for anchor, handle_in, handle_out in zip(
                contour.get("points") or [], handles_in, handles_out):
            for handle in (handle_in, handle_out):
                if (abs(anchor[0] - handle[0]) > 0.5
                        or abs(anchor[1] - handle[1]) > 0.5):
                    stubs.append([anchor[0], anchor[1], handle[0], handle[1]])
    return stubs


def item_flatten(spec):
    """The item's flattened polylines (one for a stroke, many for a fill)."""
    if is_fill(spec):
        return fill_contour_paths(spec)
    points = flatten(spec)
    return [points] if points else []


def item_anchor_count(spec):
    return sum(len(contour.get("points") or []) for contour in item_contours(spec))


def item_set_corner(spec, index):
    resolved = item_anchor(spec, index)
    if resolved is not None:
        set_corner(resolved[0], resolved[1])


def item_set_smooth(spec, index):
    resolved = item_anchor(spec, index)
    if resolved is not None:
        set_smooth(resolved[0], resolved[1])


def item_move_point(spec, index, x, y):
    resolved = item_anchor(spec, index)
    if resolved is not None:
        move_point(resolved[0], resolved[1], x, y)


def item_set_handle(spec, index, side, x, y, mirror=True):
    resolved = item_anchor(spec, index)
    if resolved is not None:
        set_handle(resolved[0], resolved[1], side, x, y, mirror=mirror)


def item_remove_point(spec, index):
    """Delete global anchor ``index`` (dropping an emptied fill contour)."""
    resolved = item_anchor(spec, index)
    if resolved is None:
        return False
    contour, local = resolved
    remove_point(contour, local)
    if is_fill(spec):
        spec["contours"] = [
            entry for entry in spec["contours"]
            if len(entry.get("points") or []) >= 3
        ]
        return bool(spec["contours"])
    return bool(spec.get("points"))


def item_nearest_segment(spec, x, y):
    """``(contour_index, segment_index, t)`` nearest ``(x, y)``, or ``None``."""
    best = None
    for contour_index, contour in enumerate(item_contours(spec)):
        detail = _nearest_segment_detail(contour, x, y)
        if detail is None:
            continue
        segment, fraction, distance = detail
        if best is None or distance < best[3]:
            best = (contour_index, segment, fraction, distance)
    if best is None:
        return None
    return best[0], best[1], best[2]


def item_split(spec, contour_index, segment, t):
    """Split ``spec``'s ``contour_index`` at ``(segment, t)`` (shape-preserving)."""
    contours = item_contours(spec)
    if not (0 <= contour_index < len(contours)):
        return None
    return split_segment(contours[contour_index], segment, t)


def _point_in_paths(x, y, polygons):
    inside = False
    for polygon in polygons:
        count = len(polygon)
        for index in range(count):
            ax, ay = polygon[index]
            bx, by = polygon[(index + 1) % count]
            if (ay > y) != (by > y):
                cross = (bx - ax) * (y - ay) / (by - ay) + ax
                if x < cross:
                    inside = not inside
    return inside


def item_hit(spec, x, y, tolerance=0.0):
    """True when ``(x, y)`` is on a stroke or inside/near a fill."""
    if not is_fill(spec):
        return hit_stroke(spec, x, y, tolerance)
    polygons = fill_contour_paths(spec)
    if len(polygons) >= 1 and _point_in_paths(x, y, polygons):
        return True
    for polygon in polygons:
        for a, b in zip(polygon, polygon[1:]):
            distance, _t = _distance_to_segment(x, y, a[0], a[1], b[0], b[1])
            if distance <= tolerance:
                return True
    return False


def item_bounds(spec):
    if is_fill(spec):
        return fill_bounds(spec)
    return bounds(spec)


def item_translate(spec, dx, dy):
    return translate_fill(spec, dx, dy) if is_fill(spec) else translate(spec, dx, dy)


def item_rotate(spec, angle, cx, cy):
    return (rotate_fill(spec, angle, cx, cy) if is_fill(spec)
            else rotate(spec, angle, cx, cy))


def item_scale(spec, sx, sy, cx, cy):
    return (scale_fill(spec, sx, sy, cx, cy) if is_fill(spec)
            else scale(spec, sx, sy, cx, cy))


def matrix_scale(matrix):
    """The geometric-mean scale of an affine ``(a, b, c, d, e, f)`` matrix.

    A stroke width is a scalar, so a non-uniform transform cannot be represented
    exactly; ``sqrt(|det|)`` is the area-preserving scale used to keep a line's
    weight proportional (matching the vector group-scale behaviour).
    """
    a, b, c, d, _e, _f = matrix
    det = abs(a * d - b * c)
    return math.sqrt(det) if det > 0.0 else 1.0


def _matrix_points(container, matrix, keys=("points", "handles_in", "handles_out")):
    a, b, c, d, e, f = matrix
    for key in keys:
        for point in container.get(key) or []:
            x = point[0]
            y = point[1]
            point[0] = a * x + c * y + e
            point[1] = b * x + d * y + f


def transform(spec, matrix):
    """Map a stroke's anchors and handles through an affine ``matrix`` (in place)."""
    _matrix_points(spec, matrix)
    gradient = spec.get("fill_gradient")
    if gradient:
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                x = point[0]
                y = point[1]
                a, b, c, d, e, f = matrix
                point[0] = a * x + c * y + e
                point[1] = b * x + d * y + f
        if "radius" in gradient:
            gradient["radius"] = float(gradient["radius"]) * matrix_scale(matrix)
    factor = matrix_scale(matrix)
    if abs(factor - 1.0) > 1e-9:
        scale_width(spec, factor)
    return spec


def transform_fill(spec, matrix):
    """Map every contour point and handle of a fill through ``matrix`` (in place)."""
    for contour in spec.get("contours") or []:
        _matrix_points(contour, matrix)
    gradient = spec.get("gradient")
    if gradient:
        a, b, c, d, e, f = matrix
        for key in ("start", "end", "center"):
            point = gradient.get(key)
            if point:
                x = point[0]
                y = point[1]
                point[0] = a * x + c * y + e
                point[1] = b * x + d * y + f
        if "radius" in gradient:
            gradient["radius"] = float(gradient["radius"]) * matrix_scale(matrix)
    return spec


def item_transform(spec, matrix):
    return transform_fill(spec, matrix) if is_fill(spec) else transform(spec, matrix)


def item_snapshot(spec):
    """A restorable copy of the item's geometry (for a transform gesture)."""
    if is_fill(spec):
        return {
            "contours": [
                {
                    "points": [[p[0], p[1]] for p in contour.get("points") or []],
                    "handles_in": [[p[0], p[1]] for p in handles(contour)[0]],
                    "handles_out": [[p[0], p[1]] for p in handles(contour)[1]],
                }
                for contour in spec.get("contours") or []
            ],
            "gradient": _clone_gradient(spec.get("gradient")),
        }
    handles_in, handles_out = handles(spec)
    return {
        "points": [[p[0], p[1]] for p in spec.get("points") or []],
        "handles_in": [[p[0], p[1]] for p in handles_in],
        "handles_out": [[p[0], p[1]] for p in handles_out],
        "widths": [float(value) for value in anchor_widths(spec)],
        "fill_gradient": _clone_gradient(spec.get("fill_gradient")),
    }


def item_restore(spec, snapshot):
    """Write a snapshot from :func:`item_snapshot` back onto ``spec``."""
    if is_fill(spec):
        spec["contours"] = [
            {
                "points": [[p[0], p[1]] for p in contour["points"]],
                "handles_in": [[p[0], p[1]] for p in contour["handles_in"]],
                "handles_out": [[p[0], p[1]] for p in contour["handles_out"]],
            }
            for contour in snapshot["contours"]
        ]
        if "gradient" in snapshot:
            spec["gradient"] = _clone_gradient(snapshot["gradient"])
    else:
        spec["points"] = [[p[0], p[1]] for p in snapshot["points"]]
        spec["handles_in"] = [[p[0], p[1]] for p in snapshot["handles_in"]]
        spec["handles_out"] = [[p[0], p[1]] for p in snapshot["handles_out"]]
        if "widths" in snapshot:
            spec["widths"] = [float(value) for value in snapshot["widths"]]
        if "fill_gradient" in snapshot:
            spec["fill_gradient"] = _clone_gradient(snapshot["fill_gradient"])


def item_width(spec):
    """The item's width (a fill has none; used for width-preserving scaling)."""
    return spec.get("width")


def item_set_anchor_width(spec, index, width):
    """Set the width profile value at global anchor ``index``."""
    if is_fill(spec):
        return False
    return set_anchor_width(spec, index, width)


def item_width_at(spec, x, y):
    """``(global_index, width)`` of the nearest anchor, or ``(None, None)``."""
    if is_fill(spec):
        return None, None
    return width_at(spec, x, y)


# ---------------------------------------------------------------- rendering
def _square_cap(pen, x, y, dx, dy, radius, color):
    """Draw a square cap centred on ``(x, y)``, aligned to ``(dx, dy)``."""
    length = math.hypot(dx, dy)
    ux, uy = (dx / length, dy / length) if length > 1e-9 else (1.0, 0.0)
    px, py = -uy, ux
    pen.polygon(
        [
            (x + (px + ux) * radius, y + (py + uy) * radius),
            (x + (px - ux) * radius, y + (py - uy) * radius),
            (x - (px + ux) * radius, y - (py + uy) * radius),
            (x - (px - ux) * radius, y - (py - uy) * radius),
        ],
        fill=color,
    )


def _draw_polyline(pen, points, color, width, closed, cap, join):
    """Draw ``points`` (already in pen space) with round caps and joins."""
    radius = width / 2.0
    if len(points) == 1:
        x, y = points[0]
        if cap == "square":
            _square_cap(pen, x, y, 1.0, 0.0, radius, color)
        elif cap == "round":
            pen.ellipse((x - radius, y - radius, x + radius, y + radius),
                        fill=color)
        else:
            pen.rectangle((x - radius, y - radius, x + radius, y + radius),
                          fill=color)
        return
    line = list(points)
    if closed and (len(line) < 2 or line[0] != line[-1]):
        line.append(line[0])
    try:
        pen.line(line, fill=color, width=int(round(width)),
                 joint="curve" if join == "round" else None)
    except TypeError:
        pen.line(line, fill=color, width=int(round(width)))
    if closed:
        return
    if cap == "round":
        for x, y in (line[0], line[-1]):
            pen.ellipse((x - radius, y - radius, x + radius, y + radius),
                        fill=color)
    elif cap == "square":
        ends = (
            (line[0], (line[0][0] - line[1][0], line[0][1] - line[1][1])),
            (line[-1], (line[-1][0] - line[-2][0], line[-1][1] - line[-2][1])),
        )
        for (x, y), (dx, dy) in ends:
            _square_cap(pen, x, y, dx, dy, radius, color)


def _sample_stops(stops, t):
    """The RGB colour at ``t`` (0..1) interpolated between gradient stops."""
    if t <= stops[0][0]:
        return rgba(stops[0][1])[:3]
    if t >= stops[-1][0]:
        return rgba(stops[-1][1])[:3]
    for index in range(1, len(stops)):
        left_offset, left_color = stops[index - 1]
        right_offset, right_color = stops[index]
        if t <= right_offset:
            span = right_offset - left_offset
            local = 0.0 if span <= 1e-9 else (t - left_offset) / span
            first = rgba(left_color)
            second = rgba(right_color)
            return tuple(
                first[channel] + (second[channel] - first[channel]) * local
                for channel in range(3)
            )
    return rgba(stops[-1][1])[:3]


def _gradient_image(gradient, size, left, top, scale=1.0):
    """Render a gradient paint over a sprite-sized RGB image.

    Pixel centres are mapped back to document coordinates through ``left`` /
    ``top`` and ``scale`` so a linear or radial gradient stays locked to the
    artwork, not to the sprite's clipped box.
    """
    width, height = size
    if width <= 0 or height <= 0 or not gradient:
        return Image.new("RGB", (max(1, width), max(1, height)), (0, 0, 0))
    xs = (np.arange(width, dtype=np.float32) + 0.5) / float(scale) + left
    ys = (np.arange(height, dtype=np.float32) + 0.5) / float(scale) + top
    grid_x, grid_y = np.meshgrid(xs, ys)
    if gradient.get("type") == "radial":
        center = gradient.get("center") or (left, top)
        radius = max(1e-6, float(gradient.get("radius", 1.0)))
        t = np.sqrt((grid_x - center[0]) ** 2
                    + (grid_y - center[1]) ** 2) / radius
    else:
        start = gradient.get("start") or (left, top)
        end = gradient.get("end") or (start[0] + 1.0, start[1])
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length_sq = max(1e-9, dx * dx + dy * dy)
        t = ((grid_x - start[0]) * dx + (grid_y - start[1]) * dy) / length_sq
    t = np.clip(t, 0.0, 1.0)
    offsets = np.array([stop[0] for stop in gradient["stops"]], dtype=np.float32)
    colors = np.array([rgba(stop[1])[:3] for stop in gradient["stops"]],
                      dtype=np.float32)
    index = np.clip(np.searchsorted(offsets, t, side="right") - 1,
                    0, len(offsets) - 2)
    span = np.maximum(1e-6, offsets[index + 1] - offsets[index])
    local = ((t - offsets[index]) / span)[..., None]
    rgb = colors[index] + (colors[index + 1] - colors[index]) * local
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _arrow_polygon(kind, tip, tangent, size):
    """A marker polygon (or circle bounds) at ``tip`` pointing along ``tangent``."""
    length = math.hypot(tangent[0], tangent[1])
    if length <= 1e-9:
        tx, ty = 1.0, 0.0
    else:
        tx, ty = tangent[0] / length, tangent[1] / length
    nx, ny = -ty, tx
    half = size / 2.0
    if kind == "triangle":
        return [
            (tip[0], tip[1]),
            (tip[0] - tx * size + nx * half, tip[1] - ty * size + ny * half),
            (tip[0] - tx * size - nx * half, tip[1] - ty * size - ny * half),
        ]
    if kind == "stealth":
        return [
            (tip[0], tip[1]),
            (tip[0] - tx * size + nx * half, tip[1] - ty * size + ny * half),
            (tip[0] - tx * size * 0.62, tip[1] - ty * size * 0.62),
            (tip[0] - tx * size - nx * half, tip[1] - ty * size - ny * half),
        ]
    if kind == "bar":
        thickness = max(1.0, size * 0.25)
        return [
            (tip[0] + nx * size - tx * thickness / 2.0,
             tip[1] + ny * size - ty * thickness / 2.0),
            (tip[0] + nx * size + tx * thickness / 2.0,
             tip[1] + ny * size + ty * thickness / 2.0),
            (tip[0] - nx * size + tx * thickness / 2.0,
             tip[1] - ny * size + ty * thickness / 2.0),
            (tip[0] - nx * size - tx * thickness / 2.0,
             tip[1] - ny * size - ty * thickness / 2.0),
        ]
    center = (tip[0] - tx * half, tip[1] - ty * half)
    if kind == "circle":
        return ("ellipse", (center[0] - half, center[1] - half,
                            center[0] + half, center[1] + half))
    return ("rect", (center[0] - half, center[1] - half,
                     center[0] + half, center[1] + half))


def _draw_arrows(pen, spec, left, top, scale, color):
    """Draw the start/end markers onto a supersampled sprite."""
    samples, closed = _centerline(spec)
    if closed or len(samples) < 2:
        return
    size = _arrow_size(spec) * scale

    def sprite_point(sample):
        return ((sample[0] - left) * scale, (sample[1] - top) * scale)

    start = sprite_point(samples[0])
    after = sprite_point(samples[1])
    end = sprite_point(samples[-1])
    before = sprite_point(samples[-2])
    markers = (
        (spec.get("arrow_start", "none"), start,
         (start[0] - after[0], start[1] - after[1])),
        (spec.get("arrow_end", "none"), end,
         (end[0] - before[0], end[1] - before[1])),
    )
    for kind, tip, tangent in markers:
        if kind == "none":
            continue
        shape = _arrow_polygon(kind, tip, tangent, max(2.0, size))
        if isinstance(shape, tuple):
            label, box = shape
            if label == "ellipse":
                pen.ellipse(box, fill=color)
            else:
                pen.rectangle(box, fill=color)
        else:
            pen.polygon(shape, fill=color)


def _render_sprite(spec):
    """Render ``spec`` to an RGBA sprite; return ``(sprite, left, top)`` in doc px."""
    key = _stroke_sprite_key(spec)
    cached = _sprite_lookup(key)
    if cached is not None:
        return cached
    box = bounds(spec)
    left = int(math.floor(box[0]))
    top = int(math.floor(box[1]))
    right = int(math.ceil(box[2]))
    bottom = int(math.ceil(box[3]))
    width_px = max(1, right - left)
    height_px = max(1, bottom - top)
    width = max(MIN_WIDTH, int(round(float(spec.get("width", DEFAULT_WIDTH)))))
    scale = antialias.scale_for(width_px, height_px)
    if not bool(spec.get("antialias", True)):
        scale = 1
    color = rgba(spec.get("color", "#000000"), int(round(255 * _opacity(spec))))
    points = [
        ((x - left) * scale, (y - top) * scale)
        for x, y in flatten(spec)
    ]
    sprite = Image.new("RGBA", (width_px * scale, height_px * scale), (0, 0, 0, 0))
    pen = ImageDraw.Draw(sprite)
    fill = spec.get("fill")
    gradient = spec.get("fill_gradient")
    closed = bool(spec.get("closed", False))
    fill_points = [
        ((x - left) * scale, (y - top) * scale)
        for x, y in (points if closed else flatten(spec, closed=True))
    ]
    if (fill or gradient) and len(fill_points) >= 3:
        if gradient:
            mask = Image.new("L", sprite.size, 0)
            ImageDraw.Draw(mask).polygon(fill_points, fill=255)
            paint = _gradient_image(
                gradient, sprite.size, left, top, scale).convert("RGBA")
            paint.putalpha(mask)
            sprite.alpha_composite(paint)
        else:
            pen.polygon(
                fill_points,
                fill=rgba(fill, int(round(255 * _opacity(spec)))))
    variable = has_width_profile(spec)
    dashed = bool(dash_pattern(spec))
    arrows = (spec.get("arrow_start", "none") != "none"
              or spec.get("arrow_end", "none") != "none")
    if variable or dashed:
        for polygon in outline_polygons(spec):
            if len(polygon) >= 3:
                pen.polygon(
                    [((x - left) * scale, (y - top) * scale)
                     for x, y in polygon],
                    fill=color)
    else:
        _draw_polyline(
            pen, points, color, max(1, width * scale), closed,
            spec.get("cap", "round"), spec.get("join", "round"),
        )
    if arrows:
        _draw_arrows(pen, spec, left, top, scale, color)
    if scale > 1:
        sprite = antialias.downscale(sprite, scale)
    _sprite_store(key, sprite, left, top)
    return sprite, left, top


def _blend_over(base, overlay, mode):
    """Source-over ``overlay`` onto ``base`` (RGBA) using a blend ``mode``."""
    if mode in (None, "", "normal"):
        return None
    from . import brush
    base = base.convert("RGBA")
    overlay = overlay.convert("RGBA")
    base_rgb = base.convert("RGB")
    overlay_rgb = overlay.convert("RGB")
    mixed = np.asarray(brush._blend(mode, base_rgb, overlay_rgb),
                       dtype=np.float32)
    base_arr = np.asarray(base_rgb, dtype=np.float32)
    base_alpha = (np.asarray(base.getchannel("A"), dtype=np.float32)
                  / 255.0)[..., None]
    src_alpha = (np.asarray(overlay.getchannel("A"), dtype=np.float32)
                 / 255.0)[..., None]
    out_alpha = src_alpha + base_alpha * (1.0 - src_alpha)
    rgb = (mixed * src_alpha + base_arr * base_alpha * (1.0 - src_alpha))
    rgb = np.clip(rgb / np.maximum(out_alpha, 1e-6), 0.0, 255.0)
    out = np.empty((base.height, base.width, 4), dtype=np.uint8)
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = np.clip(out_alpha[..., 0] * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _composite_sprite(image, sprite, left, top, offset, blend="normal"):
    """Composite a sprite onto ``image`` at its document origin."""
    ox, oy = offset
    dest_x = left - int(round(ox))
    dest_y = top - int(round(oy))
    ix0 = max(0, dest_x)
    iy0 = max(0, dest_y)
    ix1 = min(image.width, dest_x + sprite.width)
    iy1 = min(image.height, dest_y + sprite.height)
    if ix1 <= ix0 or iy1 <= iy0:
        return image
    crop = sprite.crop((ix0 - dest_x, iy0 - dest_y,
                        ix1 - dest_x, iy1 - dest_y))
    if blend in (None, "", "normal"):
        image.alpha_composite(crop, dest=(ix0, iy0))
        return image
    region = image.crop((ix0, iy0, ix1, iy1))
    blended = _blend_over(region, crop, blend)
    if blended is not None:
        image.paste(blended, (ix0, iy0))
    else:
        image.alpha_composite(crop, dest=(ix0, iy0))
    return image


def paste(image, spec, offset=(0, 0)):
    """Composite ``spec``'s stroke onto ``image``, minus the layer ``offset``.

    ``offset`` is the layer's document offset, so points in document space land
    correctly in the layer buffer. The sprite is cropped to the part that falls
    inside ``image``, so a stroke running off the layer edge does not raise.
    Returns the image (mutated in place).
    """
    if not spec.get("points"):
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    sprite, left, top = _render_sprite(spec)
    return _composite_sprite(image, sprite, left, top, offset,
                             spec.get("blend", "normal"))


def _fill_mask(polygons, size, scale, left, top, even_odd):
    """A coverage mask (0/255) for ``polygons`` in supersampled sprite space.

    Contours are resolved with a scanline fill so the rule is honoured:
    ``even_odd`` toggles coverage per crossing (SVG's ``fill-rule="evenodd"``),
    while the default nonzero rule counts winding direction, so a nested contour
    wound the other way (an icon's knocked-out hole) is left unpainted. The
    per-contour XOR/union shortcut that preceded this could not express a
    nonzero hole and filled such shapes solid.
    """
    width, height = size
    mask = Image.new("L", size, 0)
    if width <= 0 or height <= 0:
        return mask
    # Bucket every non-horizontal edge into the scanlines it spans.
    buckets = [[] for _ in range(height)]
    for poly in polygons:
        points = [((x - left) * scale, (y - top) * scale) for x, y in poly]
        if len(points) < 3:
            continue
        if points[0] != points[-1]:
            points.append(points[0])
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if y0 == y1:
                continue
            low, high = (y0, y1) if y0 < y1 else (y1, y0)
            first = max(0, int(math.floor(low)))
            last = min(height - 1, int(math.ceil(high)) - 1)
            edge = (x0, y0, x1, y1, 1 if y1 > y0 else -1)
            for row in range(first, last + 1):
                buckets[row].append(edge)
    draw = ImageDraw.Draw(mask)
    for row in range(height):
        edges = buckets[row]
        if len(edges) < 2:
            continue
        scan = row + 0.5
        crossings = []
        for x0, y0, x1, y1, direction in edges:
            low, high = (y0, y1) if y0 < y1 else (y1, y0)
            if not low <= scan < high:
                continue
            t = (scan - y0) / (y1 - y0)
            crossings.append((x0 + t * (x1 - x0), direction))
        if len(crossings) < 2:
            continue
        crossings.sort()
        winding = 0
        for index in range(len(crossings) - 1):
            winding += crossings[index][1]
            inside = (winding != 0) if not even_odd else (index % 2 == 0)
            if not inside:
                continue
            start = max(0, int(math.floor(crossings[index][0])))
            stop = min(width - 1, int(math.ceil(crossings[index + 1][0])) - 1)
            if stop >= start:
                draw.line([(start, row), (stop, row)], fill=255)
    return mask


def _fill_sprite(spec):
    """Render ``spec``'s compound fill; returns ``(sprite, left, top)``."""
    key = _fill_sprite_key(spec)
    cached = _sprite_lookup(key)
    if cached is not None:
        return cached
    polygons = fill_contour_paths(spec)
    if not polygons:
        return None
    xs = [point[0] for poly in polygons for point in poly]
    ys = [point[1] for poly in polygons for point in poly]
    left = int(math.floor(min(xs)))
    top = int(math.floor(min(ys)))
    width_px = max(1, int(math.ceil(max(xs))) - left)
    height_px = max(1, int(math.ceil(max(ys))) - top)
    scale = antialias.scale_for(width_px, height_px)
    if not bool(spec.get("antialias", True)):
        scale = 1
    # Cap the supersampled area: a large fill would otherwise allocate and
    # scan-convert a huge mask on every composite, which stalls editing. Small
    # fills keep full supersampling; big ones fall back to a lower factor (edge
    # softness matters less when the shape is that large on screen).
    if scale > 1:
        area = width_px * height_px
        if area * scale * scale > FILL_SUPERSAMPLE_AREA:
            scale = max(1, int(math.sqrt(FILL_SUPERSAMPLE_AREA / max(1, area))))
    size = (width_px * scale, height_px * scale)
    even_odd = bool(spec.get("even_odd", True))
    mask = _fill_mask(polygons, size, scale, left, top, even_odd)
    if scale > 1:
        # BOX (area average) keeps a binary mask's edges soft without the
        # ringing a Lanczos kernel's negative lobes would leave inside holes.
        mask = mask.resize((width_px, height_px), Image.BOX)
    opacity = _opacity(spec)
    if opacity < 1.0:
        mask = mask.point(lambda value: int(value * opacity))
    gradient = spec.get("gradient")
    if gradient:
        sprite = _gradient_image(
            gradient, (width_px, height_px), left, top, 1.0).convert("RGBA")
        sprite.putalpha(mask)
    else:
        sprite = Image.new("RGBA", (width_px, height_px),
                           rgba(spec.get("color", "#000000")))
        sprite.putalpha(mask)
    _sprite_store(key, sprite, left, top)
    return sprite, left, top


def paste_fill(image, spec, offset=(0, 0)):
    """Composite ``spec``'s compound fill onto ``image`` (see :func:`paste`)."""
    if not spec.get("contours"):
        return image
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    rendered = _fill_sprite(spec)
    if rendered is None:
        return image
    sprite, left, top = rendered
    return _composite_sprite(image, sprite, left, top, offset,
                             spec.get("blend", "normal"))
