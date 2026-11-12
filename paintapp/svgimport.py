"""Import SVG artwork as editable vector strokes.

This is the "convert to vector lines" half of vector import: an SVG file (or a
pasted SVG string) is read with the standard library only, its geometry is
converted to editable :mod:`paintapp.vector` strokes, and the result can be
dropped onto a vector layer.

What is understood, in the order it is applied:

* The ``viewBox`` and ``width``/``height`` of the root ``<svg>``, mapped so the
  artwork lands in viewport pixels with the origin at ``(0, 0)``.
* ``transform`` on any element or group (``matrix``, ``translate``, ``scale``,
  ``rotate``, ``skewX``, ``skewY``), composed down the tree.
* Path data (``d``) with every command - ``M/L/H/V/C/S/Q/T/A/Z``, absolute and
  relative - including quadratic and elliptical-arc conversion to cubics.
* The shape elements ``rect`` (with rounded corners), ``circle``, ``ellipse``,
  ``line``, ``polyline`` and ``polygon``.
* Presentation attributes and the ``style`` attribute, inherited from groups:
  ``color``, ``fill``, ``stroke``, ``stroke-width``, ``opacity``,
  ``fill-opacity``, ``stroke-opacity``, ``stroke-linecap``, ``stroke-linejoin``,
  ``fill-rule``, ``display`` and ``visibility``.

A stroke is imported from its outline: a stroked element uses its stroke colour
and width, while a fill-only element (an area, not a line) becomes a closed
stroke in the fill colour at :data:`FILL_WIDTH` pixels - SVG's "line shape only"
rule. Colours support ``#rgb``/``#rrggbb``/``#rrggbbaa``, ``rgb()``, the
``currentColor`` keyword (resolved against the element's ``color``, black by
default) and the common named colours; anything else falls back to black.

Only geometry a vector stroke can hold is imported, so filters, gradients,
clipping and text are not reproduced.
"""

import math
import re
import xml.etree.ElementTree as ElementTree

from . import vector

# Width, in pixels, given to a fill-only element that becomes a closed stroke.
FILL_WIDTH = 1.0

MAX_COORDINATE = 1e7

_NUMBERS = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
_COMMAND = re.compile(r"[MmLlHhVvCcSsQqTtAaZz]")
_TRANSFORM = re.compile(r"(matrix|translate|scale|rotate|skewX|skewY)\s*\(([^)]*)\)")
_SPLIT = re.compile(r"[\s,]+")

_NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000",
    "lime": "#00ff00", "green": "#008000", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "aqua": "#00ffff",
    "magenta": "#ff00ff", "fuchsia": "#ff00ff", "gray": "#808080",
    "grey": "#808080", "silver": "#c0c0c0", "maroon": "#800000",
    "olive": "#808000", "navy": "#000080", "teal": "#008080",
    "purple": "#800080", "orange": "#ffa500", "pink": "#ffc0cb",
    "brown": "#a52a2a", "transparent": None,
}


def _finite(value):
    """Coerce a parsed coordinate to a finite number inside ``±MAX_COORDINATE``.

    Malformed or extreme path data (a stray ``1e400``, a runaway transform)
    would otherwise land as ``inf``/``nan`` and make the geometry - and its
    rendered sprite - unbounded.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(number):
        return 0.0
    if number > MAX_COORDINATE:
        return MAX_COORDINATE
    if number < -MAX_COORDINATE:
        return -MAX_COORDINATE
    return number


def _sanitize_subpath(subpath):
    """A copy of ``subpath`` with all coordinates finite and in range."""
    return {
        "points": [[_finite(p[0]), _finite(p[1])] for p in subpath["points"]],
        "handles_in": [[_finite(p[0]), _finite(p[1])] for p in subpath["handles_in"]],
        "handles_out": [[_finite(p[0]), _finite(p[1])] for p in subpath["handles_out"]],
        "closed": bool(subpath["closed"]),
    }


# ------------------------------------------------------------------ matrices
def _identity():
    return (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def _compose(outer, inner):
    """Return ``outer`` applied after ``inner`` (both 2x3 affine matrices)."""
    a1, b1, c1, d1, e1, f1 = outer
    a2, b2, c2, d2, e2, f2 = inner
    return (
        a1 * a2 + c1 * b2, b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2, b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1, b1 * e2 + d1 * f2 + f1,
    )


def _apply(matrix, x, y):
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _parse_transform(text):
    """Compose the ``transform`` attribute's operations into one matrix."""
    matrix = _identity()
    if not text:
        return matrix
    for name, args in _TRANSFORM.findall(text):
        values = [float(v) for v in _SPLIT.split(args.strip()) if v]
        if name == "matrix" and len(values) >= 6:
            local = tuple(values[:6])
        elif name == "translate":
            tx = values[0] if values else 0.0
            ty = values[1] if len(values) > 1 else 0.0
            local = (1.0, 0.0, 0.0, 1.0, tx, ty)
        elif name == "scale":
            sx = values[0] if values else 1.0
            sy = values[1] if len(values) > 1 else sx
            local = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate":
            angle = math.radians(values[0] if values else 0.0)
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotate = (cos_a, sin_a, -sin_a, cos_a, 0.0, 0.0)
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                local = _compose(
                    _compose((1.0, 0.0, 0.0, 1.0, cx, cy), rotate),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy))
            else:
                local = rotate
        elif name == "skewX":
            local = (1.0, 0.0, math.tan(math.radians(values[0] if values else 0.0)),
                     1.0, 0.0, 0.0)
        elif name == "skewY":
            local = (1.0, math.tan(math.radians(values[0] if values else 0.0)),
                     0.0, 1.0, 0.0, 0.0)
        else:
            continue
        matrix = _compose(matrix, local)
    return matrix


# -------------------------------------------------------------------- colors
def _parse_color(value, default=None, current=None):
    """A ``#rrggbb`` colour from an SVG paint value, or ``None`` for no paint.

    ``current`` is the element's resolved ``color`` property, used to replace
    the ``currentColor`` keyword (which is resolved at parse time, since the
    importer does not keep live paint servers); it defaults to black, SVG's
    initial ``color``.
    """
    if value is None:
        return default
    text = str(value).strip().lower()
    if not text or text == "none" or text == "transparent":
        return None
    if text == "currentcolor":
        return current or "#000000"
    if text in _NAMED_COLORS:
        return _NAMED_COLORS[text]
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            return "#" + "".join(ch * 2 for ch in digits)
        if len(digits) == 6:
            return "#" + digits
        if len(digits) == 8:
            return "#" + digits[:6]
    match = re.match(r"rgba?\(([^)]*)\)", text)
    if match:
        parts = [p for p in _SPLIT.split(match.group(1).strip()) if p]
        if len(parts) >= 3:
            channels = []
            for part in parts[:3]:
                if part.endswith("%"):
                    channels.append(round(float(part[:-1]) * 2.55))
                else:
                    channels.append(round(float(part)))
            return "#" + "".join(f"{max(0, min(255, c)):02x}" for c in channels)
    return default


def _parse_length(value, default=0.0):
    """The leading number of an SVG length (units are treated as user units)."""
    if value is None:
        return default
    match = _NUMBERS.search(str(value))
    if not match:
        return default
    try:
        return float(match.group())
    except ValueError:
        return default


# --------------------------------------------------------------- path data
class _Builder:
    """Collects path subpaths in document pixels (already matrix-mapped)."""

    def __init__(self, matrix):
        self.matrix = matrix
        self.subpaths = []
        self.current = None
        # The current point in *user* space, kept separately because the
        # stored geometry is matrix-mapped: command arithmetic (relative
        # moves, ``H``/``V``, reflected controls) must run on raw coordinates
        # and let :meth:`_point` apply the matrix exactly once.
        self.raw_current = (0.0, 0.0)
        self._raw_start = (0.0, 0.0)

    def _point(self, x, y):
        return list(_apply(self.matrix, float(x), float(y)))

    def move_to(self, x, y):
        self.raw_current = (float(x), float(y))
        self._raw_start = self.raw_current
        point = self._point(x, y)
        self.current = {
            "points": [point], "handles_in": [list(point)],
            "handles_out": [list(point)], "closed": False,
        }
        self.subpaths.append(self.current)

    def line_to(self, x, y):
        if self.current is None:
            self.move_to(x, y)
            return
        self.raw_current = (float(x), float(y))
        point = self._point(x, y)
        self.current["handles_out"][-1] = list(self.current["points"][-1])
        self.current["points"].append(point)
        self.current["handles_in"].append(list(point))
        self.current["handles_out"].append(list(point))

    def curve_to(self, c1, c2, x, y):
        if self.current is None:
            self.move_to(x, y)
            return
        self.raw_current = (float(x), float(y))
        self.current["handles_out"][-1] = self._point(c1[0], c1[1])
        point = self._point(x, y)
        self.current["points"].append(point)
        self.current["handles_in"].append(self._point(c2[0], c2[1]))
        self.current["handles_out"].append(list(point))

    def close(self):
        if self.current is None:
            return
        self.current["closed"] = True
        self.current["handles_out"][-1] = list(self.current["points"][-1])
        self.current["handles_in"][0] = list(self.current["points"][0])
        # A closed subpath leaves the current point at its start.
        self.raw_current = self._raw_start


def _tokenize(data):
    tokens = []
    position = 0
    length = len(data)
    while position < length:
        char = data[position]
        if char in " ,\t\r\n":
            position += 1
            continue
        command = _COMMAND.match(data, position)
        if command:
            tokens.append(command.group())
            position = command.end()
            continue
        number = _NUMBERS.match(data, position)
        if number:
            tokens.append(float(number.group()))
            position = number.end()
            continue
        raise ValueError(f"bad SVG path data near {data[position:position + 16]!r}")
    return tokens


def _arc_cubics(x1, y1, rx, ry, phi, large_arc, sweep, x2, y2):
    """Convert an SVG elliptical arc to a list of ``(c1, c2, end)`` cubics."""
    if rx == 0 or ry == 0 or (x1 == x2 and y1 == y2):
        return [((x1, y1), (x2, y2), (x2, y2))]
    rx, ry = abs(rx), abs(ry)
    phi = math.radians(phi)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p = cos_phi * dx + sin_phi * dy
    y1p = -sin_phi * dx + cos_phi * dy
    lam = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry)
    if lam > 1:
        scale = math.sqrt(lam)
        rx *= scale
        ry *= scale
    numerator = rx * rx * ry * ry - rx * rx * y1p * y1p - ry * ry * x1p * x1p
    denominator = rx * rx * y1p * y1p + ry * ry * x1p * x1p
    factor = 0.0
    if denominator > 1e-12:
        factor = math.sqrt(max(0.0, numerator / denominator))
    if large_arc == sweep:
        factor = -factor
    cxp = factor * rx * y1p / ry
    cyp = -factor * ry * x1p / rx
    cx = cos_phi * cxp - sin_phi * cyp + (x1 + x2) / 2.0
    cy = sin_phi * cxp + cos_phi * cyp + (y1 + y2) / 2.0

    def angle(ux, uy, vx, vy):
        dot = ux * vx + uy * vy
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        if length == 0:
            return 0.0
        value = max(-1.0, min(1.0, dot / length))
        result = math.acos(value)
        if ux * vy - uy * vx < 0:
            result = -result
        return result

    ux = (x1p - cxp) / rx
    uy = (y1p - cyp) / ry
    vx = (-x1p - cxp) / rx
    vy = (-y1p - cyp) / ry
    theta1 = angle(1.0, 0.0, ux, uy)
    delta = angle(ux, uy, vx, vy)
    if not sweep and delta > 0:
        delta -= 2 * math.pi
    elif sweep and delta < 0:
        delta += 2 * math.pi
    count = max(1, int(math.ceil(abs(delta) / (math.pi / 2.0))))
    step = delta / count
    segments = []
    for index in range(count):
        t1 = theta1 + index * step
        t2 = t1 + step
        segments.append(_ellipse_cubic(cx, cy, rx, ry, phi, t1, t2))
    return segments


def _ellipse_cubic(cx, cy, rx, ry, phi, t1, t2):
    """A cubic approximating the ellipse arc from parameter ``t1`` to ``t2``."""
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)

    def point(t):
        return (
            cx + rx * math.cos(t) * cos_phi - ry * math.sin(t) * sin_phi,
            cy + rx * math.cos(t) * sin_phi + ry * math.sin(t) * cos_phi,
        )

    def derivative(t):
        return (
            -rx * math.sin(t) * cos_phi - ry * math.cos(t) * sin_phi,
            -rx * math.sin(t) * sin_phi + ry * math.cos(t) * cos_phi,
        )

    alpha = 4.0 / 3.0 * math.tan((t2 - t1) / 4.0)
    start = point(t1)
    end = point(t2)
    d1 = derivative(t1)
    d2 = derivative(t2)
    c1 = (start[0] + alpha * d1[0], start[1] + alpha * d1[1])
    c2 = (end[0] - alpha * d2[0], end[1] - alpha * d2[1])
    return c1, c2, end


def parse_path_data(data, matrix=None):
    """Parse one ``d`` string into subpaths (points/handles/closed), matrix-mapped."""
    if not data:
        return []
    if matrix is None:
        matrix = _identity()
    tokens = _tokenize(data)
    builder = _Builder(matrix)
    index = 0
    command = None
    last_cubic = None
    last_quad = None

    def numbers(count):
        nonlocal index
        values = tokens[index:index + count]
        if len(values) < count or any(isinstance(v, str) for v in values):
            raise ValueError("SVG path data ended mid-command")
        index += count
        return values

    while index < len(tokens):
        token = tokens[index]
        if isinstance(token, str):
            command = token
            index += 1
            if command in "Zz":
                builder.close()
                continue
        elif command is None:
            raise ValueError("SVG path data must start with a command")
        elif command in "Mm":
            command = "L" if command == "M" else "l"
        relative = command.islower()
        upper = command.upper()
        cx, cy = _last_point(builder)

        if upper == "M":
            x, y = numbers(2)
            last_cubic = last_quad = None
            builder.move_to(cx + x, cy + y) if relative else builder.move_to(x, y)
        elif upper == "L":
            x, y = numbers(2)
            last_cubic = last_quad = None
            builder.line_to(cx + x, cy + y) if relative else builder.line_to(x, y)
        elif upper == "H":
            x = numbers(1)[0]
            last_cubic = last_quad = None
            builder.line_to(cx + x, cy) if relative else builder.line_to(x, cy)
        elif upper == "V":
            y = numbers(1)[0]
            last_cubic = last_quad = None
            builder.line_to(cx, cy + y) if relative else builder.line_to(cx, y)
        elif upper == "C":
            x1, y1, x2, y2, x, y = numbers(6)
            if relative:
                c1 = (cx + x1, cy + y1)
                c2 = (cx + x2, cy + y2)
                end = (cx + x, cy + y)
            else:
                c1, c2, end = (x1, y1), (x2, y2), (x, y)
            builder.curve_to(c1, c2, *end)
            last_cubic = c2
            last_quad = None
        elif upper == "S":
            x2, y2, x, y = numbers(4)
            if last_cubic is not None:
                c1 = (2 * cx - last_cubic[0], 2 * cy - last_cubic[1])
            else:
                c1 = (cx, cy)
            if relative:
                c2 = (cx + x2, cy + y2)
                end = (cx + x, cy + y)
            else:
                c2, end = (x2, y2), (x, y)
            builder.curve_to(c1, c2, *end)
            last_cubic = c2
            last_quad = None
        elif upper == "Q":
            qx, qy, x, y = numbers(4)
            if relative:
                control = (cx + qx, cy + qy)
                end = (cx + x, cy + y)
            else:
                control, end = (qx, qy), (x, y)
            c1, c2 = _quad_to_cubic((cx, cy), control, end)
            builder.curve_to(c1, c2, *end)
            last_quad = control
            last_cubic = None
        elif upper == "T":
            x, y = numbers(2)
            if last_quad is not None:
                control = (2 * cx - last_quad[0], 2 * cy - last_quad[1])
            else:
                control = (cx, cy)
            if relative:
                end = (cx + x, cy + y)
            else:
                end = (x, y)
            c1, c2 = _quad_to_cubic((cx, cy), control, end)
            builder.curve_to(c1, c2, *end)
            last_quad = control
            last_cubic = None
        elif upper == "A":
            rx, ry, phi, large, sweep, x, y = numbers(7)
            end = (cx + x, cy + y) if relative else (x, y)
            for c1, c2, point in _arc_cubics(
                    cx, cy, rx, ry, phi, bool(large), bool(sweep), *end):
                builder.curve_to(c1, c2, *point)
            last_cubic = last_quad = None
        else:  # pragma: no cover - tokenizer only accepts known commands
            raise ValueError(f"unsupported SVG path command {command!r}")
    return builder.subpaths


def _last_point(builder):
    """The current point in user space (the builder maps it once on store)."""
    return builder.raw_current


def _quad_to_cubic(start, control, end):
    c1 = (start[0] + 2.0 / 3.0 * (control[0] - start[0]),
          start[1] + 2.0 / 3.0 * (control[1] - start[1]))
    c2 = (end[0] + 2.0 / 3.0 * (control[0] - end[0]),
          end[1] + 2.0 / 3.0 * (control[1] - end[1]))
    return c1, c2


# -------------------------------------------------------------------- shapes
def _shape_to_path(element, tag):
    """An SVG path-data string for a basic shape element (or ``None``)."""
    get = element.get

    def length(name, default=0.0):
        return _parse_length(get(name), default)

    if tag == "rect":
        x, y = length("x"), length("y")
        width, height = length("width"), length("height")
        if width <= 0 or height <= 0:
            return None
        rx = length("rx", -1.0)
        ry = length("ry", -1.0)
        if rx < 0 and ry < 0:
            rx = ry = 0.0
        elif rx < 0:
            rx = ry
        elif ry < 0:
            ry = rx
        rx = min(rx, width / 2.0)
        ry = min(ry, height / 2.0)
        if rx <= 0 or ry <= 0:
            return (f"M {x} {y} L {x + width} {y} L {x + width} {y + height} "
                    f"L {x} {y + height} Z")
        return (
            f"M {x + rx} {y} L {x + width - rx} {y} "
            f"A {rx} {ry} 0 0 1 {x + width} {y + ry} "
            f"L {x + width} {y + height - ry} "
            f"A {rx} {ry} 0 0 1 {x + width - rx} {y + height} "
            f"L {x + rx} {y + height} "
            f"A {rx} {ry} 0 0 1 {x} {y + height - ry} "
            f"L {x} {y + ry} A {rx} {ry} 0 0 1 {x + rx} {y} Z")
    if tag == "circle":
        cx, cy, r = length("cx"), length("cy"), length("r")
        if r <= 0:
            return None
        return (f"M {cx - r} {cy} A {r} {r} 0 1 0 {cx + r} {cy} "
                f"A {r} {r} 0 1 0 {cx - r} {cy} Z")
    if tag == "ellipse":
        cx, cy = length("cx"), length("cy")
        rx, ry = length("rx"), length("ry")
        if rx <= 0 or ry <= 0:
            return None
        return (f"M {cx - rx} {cy} A {rx} {ry} 0 1 0 {cx + rx} {cy} "
                f"A {rx} {ry} 0 1 0 {cx - rx} {cy} Z")
    if tag == "line":
        x1, y1 = length("x1"), length("y1")
        x2, y2 = length("x2"), length("y2")
        return f"M {x1} {y1} L {x2} {y2}"
    if tag in ("polyline", "polygon"):
        raw = get("points")
        if not raw:
            return None
        numbers = [float(v) for v in _SPLIT.split(raw.strip()) if v]
        if len(numbers) < 4:
            return None
        pairs = list(zip(numbers[0::2], numbers[1::2]))
        parts = [f"M {pairs[0][0]} {pairs[0][1]}"]
        parts.extend(f"L {px} {py}" for px, py in pairs[1:])
        if tag == "polygon":
            parts.append("Z")
        return " ".join(parts)
    return None


# -------------------------------------------------------------------- style
_STYLE_KEYS = (
    "color", "fill", "stroke", "stroke-width", "opacity", "fill-opacity",
    "stroke-opacity", "stroke-linecap", "stroke-linejoin", "fill-rule",
    "stroke-dasharray", "stroke-dashoffset", "display", "visibility",
)


def _parse_dash(value):
    """A dash pattern from ``stroke-dasharray`` (empty for a solid line)."""
    if value is None:
        return []
    text = str(value).strip().lower()
    if not text or text == "none":
        return []
    values = []
    for token in _SPLIT.split(text):
        try:
            values.append(float(token))
        except ValueError:
            continue
    return values


def _child_style(parent, element):
    style = dict(parent)
    for key in _STYLE_KEYS:
        value = element.get(key)
        if value is not None:
            style[key] = value
    inline = element.get("style")
    if inline:
        for declaration in inline.split(";"):
            if ":" not in declaration:
                continue
            key, _, value = declaration.partition(":")
            key = key.strip().lower()
            if key in _STYLE_KEYS:
                style[key] = value.strip()
    return style


def _float(style, key, default):
    try:
        return float(style.get(key, default))
    except (TypeError, ValueError):
        return default


def _specs_from_subpaths(subpaths, style, fill_width):
    """Turn matrix-mapped subpaths into strokes and/or a compound fill.

    Returns ``(strokes, fills)``. A stroked element yields one stroke per
    subpath; a fill-only element yields a single compound fill spec so the
    fill rule can knock out holes (a filled glyph's or icon's cut-outs).
    """
    # ``currentColor`` resolves against the element's ``color`` (inherited),
    # which is black unless the document says otherwise.
    current = _parse_color(style.get("color"), default="#000000") or "#000000"
    stroke = _parse_color(style.get("stroke"), current=current)
    fill = _parse_color(style.get("fill", "black"), default="#000000",
                        current=current)
    opacity = _float(style, "opacity", 1.0)
    if stroke is not None:
        color = stroke
        width = _float(style, "stroke-width", 1.0)
        alpha = opacity * _float(style, "stroke-opacity", 1.0)
        fill_color = fill
        fill_only = False
    elif fill is not None:
        color = fill
        width = fill_width
        alpha = opacity * _float(style, "fill-opacity", 1.0)
        fill_color = None
        fill_only = True
    else:
        return [], []
    cap = (style.get("stroke-linecap") or "butt").strip().lower()
    cap = cap if cap in vector.CAPS else "butt"
    join = (style.get("stroke-linejoin") or "miter").strip().lower()
    join = "round" if join == "round" else "miter"
    dash = _parse_dash(style.get("stroke-dasharray"))
    try:
        dash_offset = float(style.get("stroke-dashoffset") or 0.0)
    except (TypeError, ValueError):
        dash_offset = 0.0
    width = _finite(width)
    width = max(vector.MIN_WIDTH, min(vector.MAX_WIDTH, int(round(width))))
    opacity = max(0.0, min(1.0, alpha))
    subpaths = [_sanitize_subpath(subpath) for subpath in subpaths]

    if fill_only:
        contours = [
            {
                "points": subpath["points"],
                "handles_in": subpath["handles_in"],
                "handles_out": subpath["handles_out"],
            }
            for subpath in subpaths if len(subpath["points"]) >= 2
        ]
        even_odd = str(style.get("fill-rule", "nonzero")).strip().lower() == "evenodd"
        spec = vector.new_fill(contours, color, opacity=opacity, even_odd=even_odd)
        return ([], [spec]) if spec is not None else ([], [])

    strokes = []
    for subpath in subpaths:
        points = subpath["points"]
        if len(points) < 2:
            continue
        closed = bool(subpath["closed"])
        spec = vector.new_stroke(
            points, width, color, opacity=opacity, closed=closed, cap=cap,
            join=join, handles_in=subpath["handles_in"],
            handles_out=subpath["handles_out"],
            fill=fill_color if closed else None,
            dash=dash, dash_offset=dash_offset)
        if spec is not None:
            strokes.append(spec)
    return strokes, []


def _tag(element):
    tag = element.tag
    if isinstance(tag, str) and "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _viewport(root):
    """``(matrix, (width, height))`` mapping user units to viewport pixels."""
    width = _parse_length(root.get("width"), 0.0)
    height = _parse_length(root.get("height"), 0.0)
    view_box = root.get("viewBox")
    if not view_box:
        return _identity(), (width or 0.0, height or 0.0)
    values = [float(v) for v in _SPLIT.split(view_box.strip()) if v]
    if len(values) != 4:
        return _identity(), (width or 0.0, height or 0.0)
    min_x, min_y, box_w, box_h = values
    if box_w == 0 or box_h == 0:
        return _identity(), (width or 0.0, height or 0.0)
    scale_x = width / box_w if width else 1.0
    scale_y = height / box_h if height else 1.0
    matrix = (scale_x, 0.0, 0.0, scale_y, -min_x * scale_x, -min_y * scale_y)
    return matrix, (width or box_w, height or box_h)


def parse(text, fill_width=FILL_WIDTH):
    """Parse SVG ``text`` into ``{"strokes": [...], "size": (w, h)}``.

    Strokes are editable :mod:`paintapp.vector` specs in viewport pixels, ready
    to be placed on a vector layer. Raises ``ValueError`` when the text is not
    valid SVG.
    """
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError as error:
        raise ValueError(f"not valid SVG: {error}") from error
    if _tag(root) != "svg":
        raise ValueError("not valid SVG: root element is not <svg>")
    matrix, size = _viewport(root)
    strokes = []
    fills = []
    _walk(root, matrix, {}, strokes, fills, fill_width)
    return {"strokes": strokes, "fills": fills, "size": size}


def _walk(element, matrix, style, out_strokes, out_fills, fill_width):
    tag = _tag(element)
    if tag in ("defs", "symbol", "clipPath", "mask", "marker", "metadata",
               "title", "desc", "style", "script"):
        return
    style = _child_style(style, element)
    if str(style.get("display", "")).strip().lower() == "none":
        return
    if str(style.get("visibility", "")).strip().lower() in ("hidden", "collapse"):
        return
    local = _compose(matrix, _parse_transform(element.get("transform")))
    if tag in ("g", "svg", "a", "switch"):
        for child in element:
            _walk(child, local, style, out_strokes, out_fills, fill_width)
        return
    if tag == "path":
        data = element.get("d")
        subpaths = parse_path_data(data, local) if data else []
    elif tag in ("rect", "circle", "ellipse", "line", "polyline", "polygon"):
        data = _shape_to_path(element, tag)
        subpaths = parse_path_data(data, local) if data else []
    else:
        return
    strokes, fills = _specs_from_subpaths(subpaths, style, fill_width)
    out_strokes.extend(strokes)
    out_fills.extend(fills)


def load(path, fill_width=FILL_WIDTH):
    """Read an SVG file from ``path`` and parse it (see :func:`parse`)."""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return parse(handle.read(), fill_width=fill_width)


def _content_bounds(strokes, fills):
    """The union of stroke and fill geometry, ignoring stroke width padding."""
    xs = []
    ys = []
    for spec in strokes:
        for x, y in vector.flatten(spec):
            xs.append(x)
            ys.append(y)
    for spec in fills:
        for poly in vector.fill_contour_paths(spec):
            for x, y in poly:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def fit_to(strokes, canvas_size, margin=0.0, fills=None):
    """Clone ``strokes`` and ``fills`` scaled and centred to fit ``canvas_size``.

    Returns ``(placed_strokes, placed_fills)``; both share one transform so a
    document mixing lines and filled areas stays aligned. The aspect ratio is
    preserved and artwork is never enlarged past 1:1, matching how an opened
    image sits on a fresh layer.
    """
    strokes = list(strokes or [])
    fills = list(fills or [])
    if not strokes and not fills:
        return [], []
    box = _content_bounds(strokes, fills)
    if box is None:
        return list(strokes), list(fills)
    left, top, right, bottom = box
    art_w = max(1e-6, right - left)
    art_h = max(1e-6, bottom - top)
    canvas_w, canvas_h = canvas_size
    usable_w = max(1.0, canvas_w - 2.0 * margin)
    usable_h = max(1.0, canvas_h - 2.0 * margin)
    scale = min(1.0, usable_w / art_w, usable_h / art_h)
    offset_x = (canvas_w - art_w * scale) / 2.0 - left * scale
    offset_y = (canvas_h - art_h * scale) / 2.0 - top * scale
    placed_strokes = []
    for spec in strokes:
        copy = vector.clone(spec)
        vector.scale(copy, scale, scale, 0.0, 0.0)
        vector.translate(copy, offset_x, offset_y)
        placed_strokes.append(copy)
    placed_fills = []
    for spec in fills:
        copy = vector.clone_fill(spec)
        vector.scale_fill(copy, scale, scale, 0.0, 0.0)
        vector.translate_fill(copy, offset_x, offset_y)
        placed_fills.append(copy)
    return placed_strokes, placed_fills


def fit_within(strokes, fills, canvas_size):
    """Scale and centre artwork only when it is larger than ``canvas_size``.

    A pasted SVG carries its own viewport, which need not match the canvas, so
    an oversized shape would otherwise run off the edge; artwork already inside
    the canvas is returned untouched, keeping an in-app copy/paste at 1:1.
    """
    strokes = list(strokes or [])
    fills = list(fills or [])
    box = _content_bounds(strokes, fills)
    if box is None:
        return strokes, fills
    left, top, right, bottom = box
    if (right - left) <= canvas_size[0] and (bottom - top) <= canvas_size[1]:
        return strokes, fills
    return fit_to(strokes, canvas_size, fills=fills)


# -------------------------------------------------------------------- export
def _number(value):
    """A compact decimal string for an SVG coordinate."""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _cubic_is_corner(anchor, handle):
    return abs(anchor[0] - handle[0]) < 1e-6 and abs(anchor[1] - handle[1]) < 1e-6


def _segment_data(p0, h0, h1, p1):
    """``L`` for a straight segment, ``C`` when its handles bow it."""
    if _cubic_is_corner(p0, h0) and _cubic_is_corner(p1, h1):
        return f"L {_number(p1[0])} {_number(p1[1])}"
    return (f"C {_number(h0[0])} {_number(h0[1])} "
            f"{_number(h1[0])} {_number(h1[1])} "
            f"{_number(p1[0])} {_number(p1[1])}")


def _geometry_data(points, handles_in, handles_out, closed):
    parts = [f"M {_number(points[0][0])} {_number(points[0][1])}"]
    for index in range(len(points) - 1):
        parts.append(_segment_data(
            points[index], handles_out[index],
            handles_in[index + 1], points[index + 1]))
    if closed and len(points) >= 2:
        # A straight closing segment needs no explicit command: ``Z`` draws it.
        # Only a curved close is written out, so a straight polygon keeps its
        # anchor count across an export/import round trip.
        if not (_cubic_is_corner(points[-1], handles_out[-1])
                and _cubic_is_corner(points[0], handles_in[0])):
            parts.append(_segment_data(
                points[-1], handles_out[-1], handles_in[0], points[0]))
        parts.append("Z")
    return " ".join(parts)


def path_data(spec):
    """The SVG ``d`` attribute for one vector stroke."""
    points = spec.get("points") or []
    if not points:
        return ""
    handles_in, handles_out = vector.handles(spec)
    return _geometry_data(
        points, handles_in, handles_out, bool(spec.get("closed", False)))


def fill_data(spec):
    """The SVG ``d`` attribute for a compound fill (all contours, closed)."""
    parts = []
    for contour in spec.get("contours") or []:
        points = contour.get("points") or []
        if len(points) < 2:
            continue
        handles_in, handles_out = vector.handles(contour)
        parts.append(_geometry_data(points, handles_in, handles_out, True))
    return " ".join(parts)


def _polygon_data(points):
    """An SVG round-tripable ``d`` for a closed polyline."""
    if len(points) < 3:
        return ""
    parts = [f"M {_number(points[0][0])} {_number(points[0][1])}"]
    for point in points[1:]:
        parts.append(f"L {_number(point[0])} {_number(point[1])}")
    parts.append("Z")
    return " ".join(parts)


def _gradient_markup(identifier, gradient):
    """A ``<linearGradient>``/``<radialGradient>`` def for a vector gradient."""
    stops = "".join(
        f'<stop offset="{_number(offset)}" stop-color="{color}"/>'
        for offset, color in gradient.get("stops") or []
    )
    if gradient.get("type") == "radial":
        center = gradient.get("center") or (0.0, 0.0)
        return (
            f'<radialGradient id="{identifier}" gradientUnits="userSpaceOnUse" '
            f'cx="{_number(center[0])}" cy="{_number(center[1])}" '
            f'r="{_number(gradient.get("radius", 1.0))}">{stops}'
            f'</radialGradient>'
        )
    start = gradient.get("start") or (0.0, 0.0)
    end = gradient.get("end") or (start[0] + 1.0, start[1])
    return (
        f'<linearGradient id="{identifier}" gradientUnits="userSpaceOnUse" '
        f'x1="{_number(start[0])}" y1="{_number(start[1])}" '
        f'x2="{_number(end[0])}" y2="{_number(end[1])}">{stops}'
        f'</linearGradient>'
    )


_MARKER_BODIES = {
    "triangle": '<path d="M0,0 L10,5 L0,10 Z"/>',
    "stealth": '<path d="M0,0 L10,5 L0,10 L3,5 Z"/>',
    "circle": '<circle cx="5" cy="5" r="5"/>',
    "square": '<rect x="0" y="0" width="10" height="10"/>',
    "bar": '<rect x="4" y="0" width="2" height="10"/>',
}


def _marker_markup(identifier, kind, size):
    body = _MARKER_BODIES.get(kind)
    if body is None:
        return None
    ref = 5 if kind in ("circle", "square", "bar") else 10
    return (
        f'<marker id="{identifier}" markerUnits="userSpaceOnUse" '
        f'markerWidth="{_number(size)}" markerHeight="{_number(size)}" '
        f'refX="{ref}" refY="5" viewBox="0 0 10 10" '
        f'orient="auto-start-reverse">{body}</marker>'
    )


def export(strokes, fills=None, size=None, padding=2.0):
    """Serialise vector ``strokes`` (and compound ``fills``) as SVG.

    With ``size`` the artwork is placed at the canvas origin with that viewport;
    otherwise the viewport hugs the geometry with ``padding`` pixels of margin.
    Colours, widths, caps, joins, dashes, gradients, arrow markers, opacity and
    variable-width profiles round-trip where SVG can express them (a width
    profile is written as its filled outline).
    """
    strokes = [spec for spec in strokes if spec.get("points")]
    fills = [spec for spec in (fills or []) if spec.get("contours")]
    if size is not None:
        width, height = float(size[0]), float(size[1])
        min_x = min_y = 0.0
    else:
        box = _content_bounds(strokes, fills) or (0.0, 0.0, 1.0, 1.0)
        min_x = box[0] - padding
        min_y = box[1] - padding
        width = max(1.0, box[2] - box[0] + 2.0 * padding)
        height = max(1.0, box[3] - box[1] + 2.0 * padding)

    defs = []
    gradients = {}
    markers = {}

    def gradient_id(gradient):
        key = repr(gradient)
        if key not in gradients:
            identifier = f"gradient{len(gradients)}"
            markup = _gradient_markup(identifier, gradient)
            gradients[key] = identifier
            defs.append(markup)
        return gradients[key]

    def marker_id(kind, marker_size):
        key = (kind, round(float(marker_size), 3))
        if key not in markers:
            identifier = f"marker_{kind}_{len(markers)}"
            markup = _marker_markup(identifier, kind, marker_size)
            if markup is None:
                return None
            markers[key] = identifier
            defs.append(markup)
        return markers[key]

    body = []
    for spec in fills:
        data = fill_data(spec)
        if not data:
            continue
        gradient = spec.get("gradient")
        paint = (f"url(#{gradient_id(gradient)})" if gradient
                 else spec.get("color") or "#000000")
        attrs = [
            f'd="{data}"',
            f'fill="{paint}"',
            f'fill-rule="{"evenodd" if spec.get("even_odd", True) else "nonzero"}"',
            'stroke="none"',
        ]
        opacity = float(spec.get("opacity", 1.0))
        if opacity < 1.0:
            attrs.append(f'opacity="{_number(opacity)}"')
        body.append("  <path " + " ".join(attrs) + "/>")

    for spec in strokes:
        if vector.has_width_profile(spec):
            for polygon in vector.outline_polygons(spec):
                data = _polygon_data(polygon)
                if not data:
                    continue
                attrs = [
                    f'd="{data}"',
                    f'fill="{spec.get("color") or "#000000"}"',
                    'stroke="none"',
                ]
                opacity = float(spec.get("opacity", 1.0))
                if opacity < 1.0:
                    attrs.append(f'opacity="{_number(opacity)}"')
                body.append("  <path " + " ".join(attrs) + "/>")
            continue
        data = path_data(spec)
        if not data:
            continue
        gradient = spec.get("fill_gradient")
        if gradient and spec.get("closed"):
            fill = f"url(#{gradient_id(gradient)})"
        else:
            fill = spec.get("fill") or "none"
        attrs = [
            f'd="{data}"',
            f'fill="{fill}"',
            f'stroke="{spec.get("color") or "#000000"}"',
            f'stroke-width="{int(spec.get("width", vector.DEFAULT_WIDTH))}"',
            f'stroke-linecap="{spec.get("cap", "round")}"',
            f'stroke-linejoin="{spec.get("join", "round")}"',
        ]
        dash = vector.dash_pattern(spec)
        if dash:
            attrs.append(
                'stroke-dasharray="'
                + " ".join(_number(value) for value in dash) + '"')
            offset = float(spec.get("dash_offset", 0.0))
            if offset:
                attrs.append(f'stroke-dashoffset="{_number(offset)}"')
        marker_size = vector._arrow_size(spec)
        start_kind = spec.get("arrow_start", "none")
        end_kind = spec.get("arrow_end", "none")
        if start_kind != "none":
            identifier = marker_id(start_kind, marker_size)
            if identifier:
                attrs.append(f'marker-start="url(#{identifier})"')
        if end_kind != "none":
            identifier = marker_id(end_kind, marker_size)
            if identifier:
                attrs.append(f'marker-end="url(#{identifier})"')
        opacity = float(spec.get("opacity", 1.0))
        if opacity < 1.0:
            attrs.append(f'opacity="{_number(opacity)}"')
        body.append("  <path " + " ".join(attrs) + "/>")

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (f'<svg xmlns="http://www.w3.org/2000/svg" '
         f'width="{_number(width)}" height="{_number(height)}" '
         f'viewBox="{_number(min_x)} {_number(min_y)} '
         f'{_number(width)} {_number(height)}">'),
    ]
    if defs:
        lines.append("  <defs>")
        lines.extend("    " + entry for entry in defs)
        lines.append("  </defs>")
    lines.extend(body)
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
