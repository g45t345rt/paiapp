"""Mirror and radial symmetry guides for painting.

A symmetry guide reflects every dab a painting tool makes across one or more
axes, so a stroke drawn on one side of the canvas is mirrored onto the others -
the "symmetry ruler" that makes faces, patterns and radial motifs quick to
draw. A guide is a small dict::

    {"mode": "off", "count": 6, "mirror": True}

``mode`` is ``"off"``, ``"vertical"``, ``"horizontal"``, ``"both"`` (a four-way
mirror) or ``"radial"`` (``count`` spokes, optionally ``mirror``ed between
spokes). The axes all pass through the canvas centre, so the guide needs no
stored position: the caller passes the document size and :func:`transforms`
builds the point mappings about its midpoint.

:func:`transforms` is the one entry point the tools use: it returns a list of
``(x, y) -> (x, y)`` functions in document space, identity first, so a tool can
feed the same pointer position to one stroke per mapping.
"""

import math

MODES = ("off", "vertical", "horizontal", "both", "radial")
RADIAL_MIN = 2
RADIAL_MAX = 16
DEFAULT_COUNT = 6


def normalize(guide):
    """Coerce ``guide`` to a complete, valid guide (or ``None`` when off)."""
    if not isinstance(guide, dict):
        return None
    mode = guide.get("mode", "off")
    if mode not in MODES or mode == "off":
        return None
    try:
        count = int(round(float(guide.get("count", DEFAULT_COUNT))))
    except (TypeError, ValueError):
        count = DEFAULT_COUNT
    count = max(RADIAL_MIN, min(RADIAL_MAX, count))
    return {"mode": mode, "count": count, "mirror": bool(guide.get("mirror", True))}


def is_active(guide):
    """True when ``guide`` asks for at least one reflection."""
    return normalize(guide) is not None


def _identity(x, y):
    return (x, y)


def _flip_x(cx):
    return lambda x, y: (2.0 * cx - x, y)


def _flip_y(cy):
    return lambda x, y: (x, 2.0 * cy - y)


def _rotate(cx, cy, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    def transform(x, y):
        dx = x - cx
        dy = y - cy
        return (cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a)

    return transform


def _reflect(cx, cy, angle):
    """Reflect across the line through ``(cx, cy)`` at ``angle``."""
    cos_2a = math.cos(2.0 * angle)
    sin_2a = math.sin(2.0 * angle)

    def transform(x, y):
        dx = x - cx
        dy = y - cy
        return (cx + dx * cos_2a + dy * sin_2a, cy + dx * sin_2a - dy * cos_2a)

    return transform


def transforms(guide, size):
    """The point mappings for ``guide`` about the centre of ``size``.

    Returns ``[identity]`` when the guide is off, so callers can always iterate
    the result. ``size`` is ``(width, height)`` in document pixels; the guide is
    centred on its midpoint.
    """
    guide = normalize(guide)
    if guide is None:
        return [_identity]
    cx = size[0] / 2.0
    cy = size[1] / 2.0
    mode = guide["mode"]
    if mode == "vertical":
        return [_identity, _flip_x(cx)]
    if mode == "horizontal":
        return [_identity, _flip_y(cy)]
    if mode == "both":
        return [
            _identity, _flip_x(cx), _flip_y(cy),
            lambda x, y: (2.0 * cx - x, 2.0 * cy - y),
        ]
    # radial: N rotations, plus N reflections across the spoke lines when the
    # guide is mirrored - together the dihedral group of the N-fold rosette.
    count = guide["count"]
    result = []
    for index in range(count):
        result.append(_rotate(cx, cy, 2.0 * math.pi * index / count))
    if guide["mirror"]:
        for index in range(count):
            result.append(_reflect(cx, cy, math.pi * index / count))
    return result


def axes(guide, size):
    """Guide lines for the overlay, as ``[((x0, y0), (x1, y1)), ...]``.

    The lines are clipped to the document rectangle, so the overlay can draw
    them without knowing the view transform. Empty when the guide is off.
    """
    guide = normalize(guide)
    if guide is None:
        return []
    width, height = size
    cx = width / 2.0
    cy = height / 2.0
    mode = guide["mode"]
    if mode == "vertical":
        return [((cx, 0.0), (cx, float(height)))]
    if mode == "horizontal":
        return [((0.0, cy), (float(width), cy))]
    if mode == "both":
        return [
            ((cx, 0.0), (cx, float(height))),
            ((0.0, cy), (float(width), cy)),
        ]
    count = guide["count"]
    radius = math.hypot(width, height)
    lines = []
    # Half-turn coverage is enough: a line through the centre covers both
    # opposite spokes, so N lines at pi/N spacing give the full rosette.
    for index in range(count):
        angle = math.pi * index / count
        dx = math.cos(angle) * radius
        dy = math.sin(angle) * radius
        lines.append(((cx - dx, cy - dy), (cx + dx, cy + dy)))
    return lines
