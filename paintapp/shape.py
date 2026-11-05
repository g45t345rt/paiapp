"""Primitive shape drawing behind the Shape tool.

A drag defines a bounding box in document pixels; :func:`draw` renders the
chosen primitive into a layer's ``ImageDraw``. Shapes are filled or stroked with
the current colour. Holding Shift constrains a rectangle/ellipse to a square or
circle and snaps a line to 45-degree increments.
"""

import math

SHAPES = [
    ("rectangle", "Rectangle"),
    ("ellipse", "Ellipse"),
    ("line", "Line"),
    ("triangle", "Triangle"),
    ("diamond", "Diamond"),
    ("pentagon", "Pentagon"),
    ("hexagon", "Hexagon"),
    ("star", "Star"),
    ("arrow", "Arrow"),
]

# Shapes rendered as a single closed polygon inscribed in the drag box.
POLYGONS = ("triangle", "diamond", "pentagon", "hexagon", "star", "arrow")

# Number of sides for the regular polygons; the star alternates its radii.
_REGULAR_SIDES = {"pentagon": 5, "hexagon": 6}
_STAR_POINTS = 5
_STAR_INNER = 0.382

# Fraction of the box height taken by an arrow's shaft.
_ARROW_SHAFT = 0.44


def constrain(box, kind, shift):
    """Apply Shift constraints to ``box`` ``(left, top, right, bottom)``."""
    left, top, right, bottom = box
    if not shift:
        return box
    if kind == "line":
        dx = right - left
        dy = bottom - top
        angle = round(math.atan2(dy, dx) / (math.pi / 4.0)) * (math.pi / 4.0)
        length = math.hypot(dx, dy)
        return (
            left, top,
            left + math.cos(angle) * length,
            top + math.sin(angle) * length,
        )
    width = right - left
    height = bottom - top
    size = max(abs(width), abs(height))
    right = left + (size if width >= 0 else -size)
    bottom = top + (size if height >= 0 else -size)
    return (left, top, right, bottom)


def corner_radius(box, radius):
    """Clamp ``radius`` to what ``box`` ``(left, top, right, bottom)`` allows."""
    span = min(abs(box[2] - box[0]), abs(box[3] - box[1])) / 2.0
    return max(0.0, min(float(radius), span))


def polygon_points(kind, box):
    """The corner list for a polygon ``kind`` inscribed in ``box``.

    Returns ``[]`` for a kind that is not a polygon. The points are in the same
    coordinate space as ``box``, so the caller can use this for the layer image
    or for an on-screen preview alike.
    """
    if kind not in POLYGONS:
        return []
    left, top, right, bottom = box
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    cx = (left + right) / 2.0
    cy = (top + bottom) / 2.0
    rx = (right - left) / 2.0
    ry = (bottom - top) / 2.0
    if kind == "triangle":
        return [(cx, top), (right, bottom), (left, bottom)]
    if kind == "diamond":
        return [(cx, top), (right, cy), (cx, bottom), (left, cy)]
    if kind == "arrow":
        mid = left + (right - left) * 0.55
        half = (bottom - top) * _ARROW_SHAFT / 2.0
        return [
            (left, cy - half), (mid, cy - half), (mid, top),
            (right, cy), (mid, bottom), (mid, cy + half), (left, cy + half),
        ]
    if kind == "star":
        points = []
        for index in range(_STAR_POINTS * 2):
            outer = index % 2 == 0
            angle = -math.pi / 2.0 + math.pi * index / _STAR_POINTS
            factor = 1.0 if outer else _STAR_INNER
            points.append((
                cx + math.cos(angle) * rx * factor,
                cy + math.sin(angle) * ry * factor,
            ))
        return points
    sides = _REGULAR_SIDES.get(kind, 5)
    return [
        (
            cx + math.cos(-math.pi / 2.0 + 2.0 * math.pi * i / sides) * rx,
            cy + math.sin(-math.pi / 2.0 + 2.0 * math.pi * i / sides) * ry,
        )
        for i in range(sides)
    ]


def draw(pen, kind, box, fill=None, outline=None, width=1, shift=False, radius=0):
    """Draw ``kind`` into ``pen`` (an ``ImageDraw``) using ``box``."""
    left, top, right, bottom = constrain(box, kind, shift)
    if kind == "line":
        pen.line((left, top, right, bottom), fill=outline or fill, width=width)
        return
    if right < left:
        left, right = right, left
    if bottom < top:
        top, bottom = bottom, top
    box = (left, top, right, bottom)
    if kind == "ellipse":
        pen.ellipse(box, fill=fill, outline=outline, width=width)
    elif kind in POLYGONS:
        pen.polygon(
            polygon_points(kind, box), fill=fill, outline=outline, width=width,
        )
    else:
        corner = corner_radius(box, radius)
        if corner > 0:
            pen.rounded_rectangle(
                box, radius=corner, fill=fill, outline=outline, width=width,
            )
        else:
            pen.rectangle(box, fill=fill, outline=outline, width=width)
