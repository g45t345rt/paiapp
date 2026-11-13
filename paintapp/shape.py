"""Primitive shape drawing behind the Shape tool.

A drag defines a bounding box in document pixels; :func:`draw` renders the
chosen primitive into a layer's ``ImageDraw``. Shapes are filled or stroked with
the current colour. Holding Shift constrains a rectangle/ellipse to a square or
circle and snaps a line to 45-degree increments.
"""

import math

from . import svgimport

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


def vector_geometry(kind, box, shift=False, radius=0):
    """``box``'s geometry as ``{points, handles_in, handles_out, closed}``.

    The primitive is converted through the SVG path importer, so ellipses and
    rounded rectangles come back as true cubic segments and every shape can be
    stored as an editable vector item instead of being baked into pixels.
    """
    left, top, right, bottom = constrain(box, kind, shift)
    if kind == "line":
        data = f"M {left} {top} L {right} {bottom}"
    elif kind == "ellipse":
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        rx = abs(right - left) / 2.0
        ry = abs(bottom - top) / 2.0
        if rx <= 0 or ry <= 0:
            return None
        data = (f"M {cx - rx} {cy} A {rx} {ry} 0 1 0 {cx + rx} {cy} "
                f"A {rx} {ry} 0 1 0 {cx - rx} {cy} Z")
    elif kind in POLYGONS:
        points = polygon_points(kind, box)
        if len(points) < 3:
            return None
        data = "M " + " L ".join(f"{x} {y}" for x, y in points) + " Z"
    else:
        if right < left:
            left, right = right, left
        if bottom < top:
            top, bottom = bottom, top
        corner = corner_radius((left, top, right, bottom), radius)
        if corner <= 0:
            data = (f"M {left} {top} L {right} {top} L {right} {bottom} "
                    f"L {left} {bottom} Z")
        else:
            data = (
                f"M {left + corner} {top} L {right - corner} {top} "
                f"A {corner} {corner} 0 0 1 {right} {top + corner} "
                f"L {right} {bottom - corner} "
                f"A {corner} {corner} 0 0 1 {right - corner} {bottom} "
                f"L {left + corner} {bottom} "
                f"A {corner} {corner} 0 0 1 {left} {bottom - corner} "
                f"L {left} {top + corner} "
                f"A {corner} {corner} 0 0 1 {left + corner} {top} Z")
    subpaths = svgimport.parse_path_data(data)
    if not subpaths:
        return None
    geometry = subpaths[0]
    points = geometry["points"]
    # A closed path ends on a duplicate of its first anchor; drop it so the
    # shape has one editable node per corner (the closing segment supplies it).
    # The duplicate carries the curve control the closing segment arrives with,
    # so fold it into the first anchor instead of losing it - otherwise a true
    # curve (an ellipse, a rounded corner) closes with a sharp cusp.
    if (geometry["closed"] and len(points) >= 3
            and math.hypot(points[-1][0] - points[0][0],
                           points[-1][1] - points[0][1]) < 1e-6):
        geometry["handles_in"][0] = geometry["handles_in"][-1]
        geometry["points"] = points[:-1]
        geometry["handles_in"] = geometry["handles_in"][:-1]
        geometry["handles_out"] = geometry["handles_out"][:-1]
    return geometry
