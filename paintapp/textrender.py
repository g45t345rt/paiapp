"""Editable text objects: normalisation, rasterising and layout metrics.

A text object is a plain dict stored on a layer (see ``Layer.texts``). Keeping
the description rather than only the pixels lets the text tool edit, transform
and re-render it, so the on-canvas preview and the committed result share one
renderer.

The spec uses document pixels for its anchor, the top-left of the text block,
plus a clockwise ``rotation`` in degrees::

    {"text", "family", "size", "bold", "italic", "color", "align",
     "rotation", "line_spacing", "letter_spacing", "stroke", "shadow",
     "width", "wrap", "x", "y", "case", "underline", "strikethrough",
     "paragraph_spacing", "opacity"}

``line_spacing`` is extra leading as a fraction of the size, ``letter_spacing``
is extra pixels between glyphs, ``paragraph_spacing`` is extra pixels between
paragraphs, ``stroke`` is ``{"color", "width"}`` or ``None`` and ``shadow`` is
``{"color", "dx", "dy", "blur"}`` or ``None``. ``case`` is one of
:data:`CASE_MODES` and changes how glyphs are drawn without altering the stored
text (caret offsets stay valid). ``align`` also accepts ``"justify"``, which
stretches every row but the last of a paragraph across the layout box.
``opacity`` is a percentage applied to the fill, outline and shadow.

``width`` is the fixed layout box in document pixels (``0`` hugs the longest
line). When a box is set, ``align`` centres or right-aligns within it, and with
``wrap`` on, lines wider than the box soft-wrap at word boundaries.

Text is laid out in a local, unrotated space whose origin is the anchor; the
rotation spins that space about the anchor. Layout helpers work in document
coordinates so the canvas overlay can draw them directly, and ``paste`` is the
single path that turns a spec into pixels (used for compositing and for the
destructive rasterise).
"""

import math
import unicodedata

from PIL import Image, ImageDraw, ImageFilter

from . import fonts
from .constants import rgba

MIN_SIZE = 4
MAX_SIZE = 1000
DEFAULT_SIZE = 48
ALIGNMENTS = ("left", "center", "right", "justify")
CASE_MODES = ("normal", "upper", "lower", "title")

# Extra pixels between lines, as a fraction of the font size, when a spec does
# not ask for its own leading.
DEFAULT_LINE_SPACING = 0.15
MAX_STROKE = 100
MAX_BLUR = 100
MAX_OFFSET = 2000
MAX_PARAGRAPH_SPACING = 1000
MAX_OPACITY = 100

# Fixed layout box limits. ``width`` of 0 means the box hugs the longest line.
MAX_WIDTH = 20000.0
MIN_BOX_WIDTH = 8.0

# Corner order used by the transform box and its handles.
CORNER_ORDER = ("nw", "ne", "se", "sw")
CORNER_OPPOSITE = {"nw": "se", "ne": "sw", "se": "nw", "sw": "ne"}

# Mid-edge handles that set the layout box width.
EDGE_ORDER = ("e", "w")


# ------------------------------------------------------------------- specs
def _norm_stroke(value):
    if not isinstance(value, dict):
        return None
    try:
        width = int(round(float(value.get("width", 0) or 0)))
    except (TypeError, ValueError):
        width = 0
    width = max(0, min(MAX_STROKE, width))
    if width <= 0:
        return None
    return {"color": str(value.get("color") or "#000000"), "width": width}


def _norm_shadow(value):
    if not isinstance(value, dict):
        return None

    def number(key, default):
        try:
            return int(round(float(value.get(key, default) or 0)))
        except (TypeError, ValueError):
            return default

    return {
        "color": str(value.get("color") or "#000000"),
        "dx": max(-MAX_OFFSET, min(MAX_OFFSET, number("dx", 2))),
        "dy": max(-MAX_OFFSET, min(MAX_OFFSET, number("dy", 2))),
        "blur": max(0, min(MAX_BLUR, number("blur", 0))),
    }


def normalize(spec):
    """Coerce ``spec`` to a complete, valid text object (or ``None``)."""
    if not isinstance(spec, dict):
        return None
    try:
        size = int(round(float(spec.get("size", DEFAULT_SIZE))))
    except (TypeError, ValueError):
        size = DEFAULT_SIZE
    size = max(MIN_SIZE, min(MAX_SIZE, size))
    family = str(spec.get("family") or fonts.default_family() or "Default")
    align = str(spec.get("align") or "left").lower()
    if align not in ALIGNMENTS:
        align = "left"
    color = str(spec.get("color") or "#000000")
    try:
        rotation = float(spec.get("rotation", 0.0) or 0.0)
    except (TypeError, ValueError):
        rotation = 0.0
    try:
        leading = float(spec.get("line_spacing", DEFAULT_LINE_SPACING))
    except (TypeError, ValueError):
        leading = DEFAULT_LINE_SPACING
    leading = max(-0.5, min(3.0, leading))
    try:
        track = float(spec.get("letter_spacing", 0.0) or 0.0)
    except (TypeError, ValueError):
        track = 0.0
    track = max(-0.5 * size, min(2.0 * size, track))
    case = str(spec.get("case") or "normal").lower()
    if case not in CASE_MODES:
        case = "normal"
    try:
        paragraph = int(round(float(spec.get("paragraph_spacing", 0) or 0)))
    except (TypeError, ValueError):
        paragraph = 0
    paragraph = max(0, min(MAX_PARAGRAPH_SPACING, paragraph))
    try:
        opacity = int(round(float(spec.get("opacity", MAX_OPACITY))))
    except (TypeError, ValueError):
        opacity = MAX_OPACITY
    opacity = max(0, min(MAX_OPACITY, opacity))
    try:
        width = float(spec.get("width", 0.0) or 0.0)
    except (TypeError, ValueError):
        width = 0.0
    width = max(0.0, min(MAX_WIDTH, width))
    try:
        x = float(spec.get("x", 0.0))
        y = float(spec.get("y", 0.0))
    except (TypeError, ValueError):
        x = y = 0.0
    result = {
        "text": str(spec.get("text", "") or ""),
        "family": family,
        "size": size,
        "bold": bool(spec.get("bold", False)),
        "italic": bool(spec.get("italic", False)),
        "color": color,
        "align": align,
        "rotation": rotation,
        "line_spacing": leading,
        "letter_spacing": track,
        "paragraph_spacing": paragraph,
        "case": case,
        "underline": bool(spec.get("underline", False)),
        "strikethrough": bool(spec.get("strikethrough", False)),
        "opacity": opacity,
        "stroke": _norm_stroke(spec.get("stroke")),
        "shadow": _norm_shadow(spec.get("shadow")),
        "width": width,
        "wrap": bool(spec.get("wrap", False)),
        "x": x,
        "y": y,
    }
    # Hidden objects stay on their layer but are skipped when drawn; the key
    # is only written when set so visible specs keep their original shape.
    if spec.get("hidden"):
        result["hidden"] = True
    return result


def clone(spec):
    """A copy of ``spec`` safe to keep in a snapshot/undo entry."""
    normalized = normalize(spec)
    return dict(normalized) if normalized is not None else None


def new_spec(text, x, y, family=None, size=DEFAULT_SIZE, bold=False,
             italic=False, color="#000000", align="left", rotation=0.0,
             line_spacing=DEFAULT_LINE_SPACING, letter_spacing=0.0,
             stroke=None, shadow=None, width=0.0, wrap=False,
             case="normal", underline=False, strikethrough=False,
             paragraph_spacing=0, opacity=MAX_OPACITY):
    """Build a normalised text object anchored at document ``(x, y)``."""
    return normalize({
        "text": text, "family": family, "size": size, "bold": bold,
        "italic": italic, "color": color, "align": align, "rotation": rotation,
        "line_spacing": line_spacing, "letter_spacing": letter_spacing,
        "stroke": stroke, "shadow": shadow, "width": width, "wrap": wrap,
        "case": case, "underline": underline, "strikethrough": strikethrough,
        "paragraph_spacing": paragraph_spacing, "opacity": opacity,
        "x": x, "y": y,
    })


def describe(spec):
    """A short one-line label for a text object, for the layer list."""
    text = " ".join((spec.get("text") or "").split())
    if not text:
        text = "(empty)"
    elif len(text) > 24:
        text = text[:23] + "\u2026"
    return f'{text}  ({int(spec.get("size", DEFAULT_SIZE))}px)'


# ------------------------------------------------------------------- fonts
def load_font(spec):
    """The ``ImageFont`` for ``spec``'s family, size and style."""
    return fonts.load(
        spec.get("family"), spec.get("size", DEFAULT_SIZE),
        bold=bool(spec.get("bold")), italic=bool(spec.get("italic")),
    )


def line_spacing(spec):
    """Extra pixels inserted between lines."""
    try:
        ratio = float(spec.get("line_spacing", DEFAULT_LINE_SPACING))
    except (TypeError, ValueError):
        ratio = DEFAULT_LINE_SPACING
    return int(round(int(spec.get("size", DEFAULT_SIZE)) * ratio))


def letter_spacing(spec):
    """Extra pixels inserted between glyphs."""
    try:
        return float(spec.get("letter_spacing", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def paragraph_spacing(spec):
    """Extra pixels inserted between paragraphs."""
    try:
        return max(0, min(MAX_PARAGRAPH_SPACING,
                          int(round(float(spec.get("paragraph_spacing", 0) or 0)))))
    except (TypeError, ValueError):
        return 0


def opacity(spec):
    """The text's opacity as a 0-1 fraction."""
    try:
        value = int(round(float(spec.get("opacity", MAX_OPACITY))))
    except (TypeError, ValueError):
        value = MAX_OPACITY
    return max(0, min(MAX_OPACITY, value)) / float(MAX_OPACITY)


def case_mode(spec):
    """The spec's case transform, normalised to one of :data:`CASE_MODES`."""
    mode = str(spec.get("case") or "normal").lower()
    return mode if mode in CASE_MODES else "normal"


def _case_char(char, mode):
    """Map one character, keeping it unchanged when the mapping changes length.

    Upper-casing can expand a character (``ß`` becomes ``SS``); keeping the
    original then means the rendered string stays the same length as the stored
    text, so caret offsets and selections still line up.
    """
    mapped = char.upper() if mode == "upper" else char.lower()
    return mapped if len(mapped) == 1 else char


def display_text(spec):
    """``spec``'s text with its case transform applied, preserving length."""
    text = spec.get("text", "")
    mode = case_mode(spec)
    if mode == "normal":
        return text
    if mode == "title":
        result = []
        word_start = True
        for char in text:
            if char.isalnum():
                result.append(_case_char(char, "upper" if word_start else "lower"))
                word_start = False
            else:
                result.append(char)
                # Keep apostrophes inside the word ("don't", "O'Neill") while
                # punctuation and spaces start a new one.
                if char not in "'\u2019" and not _is_combining(char):
                    word_start = True
        return "".join(result)
    return "".join(_case_char(char, mode) for char in text)


def box_width(spec):
    """The fixed layout width in pixels, or ``0`` when the box hugs the text."""
    try:
        width = float(spec.get("width", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(MAX_WIDTH, width))


def wrapping(spec):
    """True when lines wider than the box should soft-wrap."""
    return bool(spec.get("wrap", False))


def _length(font, text):
    """Advance width of ``text``, tolerating fonts without ``getlength``."""
    try:
        return float(font.getlength(text))
    except AttributeError:
        try:
            return float(font.getbbox(text)[2])
        except Exception:
            return 0.0


# Per-glyph advance widths are measured constantly by caret, selection and
# layout maths, and ``font.getlength`` is comparatively costly, so keep a small
# cache keyed by ``(id(font), char)``. Cached ``ImageFont`` instances live in
# ``fonts._font_cache`` for the process lifetime, so their ids are stable.
_CHAR_WIDTH_CACHE = {}
_CHAR_WIDTH_CACHE_CAP = 40000


def _char_width(font, char):
    """Advance width of a single glyph, memoised across calls."""
    key = (id(font), char)
    cached = _CHAR_WIDTH_CACHE.get(key)
    if cached is not None:
        return cached
    width = _length(font, char)
    if len(_CHAR_WIDTH_CACHE) < _CHAR_WIDTH_CACHE_CAP:
        _CHAR_WIDTH_CACHE[key] = width
    return width


def _is_combining(char):
    """Whether ``char`` is a combining mark that joins the previous glyph.

    Letter-spacing is applied per glyph, so the renderer draws character by
    character; keeping combining marks, joiners and variation selectors grouped
    with their base character stops accents from being split off and rendered
    alone."""
    code = ord(char)
    return (
        unicodedata.combining(char) != 0
        or char == "\u200d"
        or 0xFE00 <= code <= 0xFE0F
    )


def _metrics(font, spec):
    """``(ascent, descent)`` in pixels, with a size-based fallback."""
    try:
        ascent, descent = font.getmetrics()
        return float(ascent), float(descent)
    except Exception:
        size = int(spec.get("size", DEFAULT_SIZE))
        return size * 0.8, size * 0.2


def _prefix_width(font, prefix, track):
    """Width of a partial line including its inter-glyph tracking.

    Tracking is added between glyph clusters, so a combining mark that follows
    its base character draws tight against it (no tracking inserted).
    """
    if not prefix:
        return 0.0
    total = 0.0
    boundaries = 0
    for char in prefix:
        total += _char_width(font, char)
        if not _is_combining(char):
            boundaries += 1
    return total + track * (boundaries - 1 if boundaries else 0)


def _cluster_width(font, cluster):
    """Advance width of one glyph cluster (no internal tracking)."""
    return sum(_char_width(font, char) for char in cluster)


def _wrap_segment(font, line, max_width, track):
    """Split ``line`` into rows that fit ``max_width``.

    Returns ``(offset, text)`` pairs whose offsets index into ``line`` so the
    caret can map a column back to the full string. Words break at spaces and a
    word longer than the box breaks at the glyph that overflows.
    """
    if not line or max_width <= 0:
        return [(0, line)]
    prefix = [0.0]
    for char in line:
        step = _char_width(font, char)
        if not _is_combining(char):
            step += track
        prefix.append(prefix[-1] + step)

    def width(start, end):
        if end <= start:
            return 0.0
        return prefix[end] - prefix[start] - track

    if width(0, len(line)) <= max_width:
        return [(0, line)]
    rows = []
    start = 0
    length = len(line)
    while start < length:
        end = start
        space = -1
        while end < length and width(start, end + 1) <= max_width:
            if line[end] == " ":
                space = end
            end += 1
        if end == start:
            end = start + 1
        elif end < length and space > start:
            end = space + 1
        rows.append((start, line[start:end]))
        start = end
    return rows


def visual_lines(spec):
    """The laid-out rows, as ``{"text", "start", "para_end"}``.

    ``start`` is a caret index into the spec's full ``text`` so a caret offset
    can be located in any row. ``para_end`` marks the last visual row of a
    logical line, which is where paragraph spacing goes and where justification
    stops stretching.
    """
    text = display_text(spec)
    font = load_font(spec)
    track = letter_spacing(spec)
    width = box_width(spec)
    wrap = wrapping(spec) and width > 0
    rows = []
    cursor = 0
    for logical in text.split("\n"):
        pieces = _wrap_segment(font, logical, width, track) if wrap else [(0, logical)]
        for offset, chunk in pieces:
            rows.append({
                "text": chunk,
                "start": cursor + offset,
                "para_end": offset + len(chunk) >= len(logical),
            })
        cursor += len(logical) + 1
    return rows


def _row_tops(spec, rows):
    """Top offsets of ``rows`` plus the block height and base line advance.

    Paragraph spacing is inserted after the last visual row of each logical
    line, so wrapped rows stay tight while paragraphs breathe. The returned
    height is measured from the first row's top to the last row's bottom.
    """
    font = load_font(spec)
    ascent, descent = _metrics(font, spec)
    advance = ascent + descent + line_spacing(spec)
    gap = paragraph_spacing(spec)
    tops = []
    y = 0.0
    for index, row in enumerate(rows):
        tops.append(y)
        y += advance
        if row.get("para_end") and index < len(rows) - 1:
            y += gap
    height = tops[-1] + ascent + descent if tops else 0.0
    return tops, height, advance


# --------------------------------------------------------------- geometry
def rotate_point(theta, x, y):
    """Rotate ``(x, y)`` by ``theta`` radians about the origin."""
    cos = math.cos(theta)
    sin = math.sin(theta)
    return (cos * x - sin * y, sin * x + cos * y)


def _theta(spec):
    return math.radians(float(spec.get("rotation", 0.0) or 0.0))


def to_doc(spec, lx, ly):
    """Map a local, unrotated point to document pixels."""
    rx, ry = rotate_point(_theta(spec), lx, ly)
    return (spec.get("x", 0.0) + rx, spec.get("y", 0.0) + ry)


def to_local(spec, x, y):
    """Map a document point into the unrotated local space of ``spec``."""
    rx, ry = rotate_point(-_theta(spec), x - spec.get("x", 0.0), y - spec.get("y", 0.0))
    return (rx, ry)


def transform(spec, matrix):
    """Map a text object through an affine ``matrix`` (in place).

    The anchor moves as a point, the block's rotation picks up the matrix's
    rotation, and the font size (with the pixel-valued tracking, paragraph
    spacing and wrap width) scales by :func:`matrix_scale`-style
    ``sqrt(|det|)``. This is exact for a similarity (translate/rotate/uniform-
    scale) transform; a skew or non-uniform scale is approximated by that single
    scalar, since a text block has one size.
    """
    a, b, c, d, e, f = matrix
    x = float(spec.get("x", 0.0) or 0.0)
    y = float(spec.get("y", 0.0) or 0.0)
    spec["x"] = a * x + c * y + e
    spec["y"] = b * x + d * y + f
    det = abs(a * d - b * c)
    scale = math.sqrt(det) if det > 0.0 else 1.0
    if abs(scale - 1.0) > 1e-9:
        try:
            spec["size"] = max(
                1, int(round(int(spec.get("size", DEFAULT_SIZE)) * scale)))
        except (TypeError, ValueError):
            pass
        for key in ("letter_spacing", "paragraph_spacing", "width"):
            try:
                value = float(spec.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if value:
                spec[key] = value * scale
    spec["rotation"] = (float(spec.get("rotation", 0.0) or 0.0)
                        + math.degrees(math.atan2(b, a)))
    return spec


def local_rect(spec):
    """The text block's rectangle in local space, as ``(x0, y0, x1, y1)``."""
    font = load_font(spec)
    rows = visual_lines(spec)
    track = letter_spacing(spec)
    widths = [_prefix_width(font, row["text"], track) for row in rows]
    box = box_width(spec)
    width = box if box > 0 else (max(widths) if widths else 0.0)
    if width < 1:
        width = max(3.0, _length(font, " "))
    ascent, descent = _metrics(font, spec)
    _tops, height, _advance = _row_tops(spec, rows)
    if not rows:
        height = ascent + descent
    return (0.0, 0.0, float(width), float(height))


def corners(spec):
    """The four rotated corners in document pixels, in :data:`CORNER_ORDER`."""
    x0, y0, x1, y1 = local_rect(spec)
    return [
        to_doc(spec, x0, y0), to_doc(spec, x1, y0),
        to_doc(spec, x1, y1), to_doc(spec, x0, y1),
    ]


def corner_local(spec, name):
    """The local-space point of a named corner."""
    x0, y0, x1, y1 = local_rect(spec)
    return {
        "nw": (x0, y0), "ne": (x1, y0), "se": (x1, y1), "sw": (x0, y1),
    }[name]


def rotate_handle(spec):
    """Document point of the rotation handle above the block's top edge."""
    x0, y0, x1, _ = local_rect(spec)
    offset = max(16.0, 0.35 * int(spec.get("size", DEFAULT_SIZE)))
    return to_doc(spec, (x0 + x1) / 2.0, y0 - offset)


def default_pivot_local(spec):
    """The block centre in local space, the default rotation origin."""
    x0, y0, x1, y1 = local_rect(spec)
    return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def anchor_locals(spec):
    """The block's snap anchors in local space: corners, edges and centre."""
    x0, y0, x1, y1 = local_rect(spec)
    mx = (x0 + x1) / 2.0
    my = (y0 + y1) / 2.0
    return [
        (x0, y0), (mx, y0), (x1, y0),
        (x0, my), (mx, my), (x1, my),
        (x0, y1), (mx, y1), (x1, y1),
    ]


def pivot_point(spec, pivot=None):
    """Document point of the rotation origin (``None`` means the centre)."""
    if pivot is None:
        pivot = default_pivot_local(spec)
    return to_doc(spec, *pivot)


def handle_points(spec, pivot=None):
    """Named transform-handle document points, including ``"rot"`` and edges."""
    x0, y0, x1, y1 = local_rect(spec)
    mid_y = (y0 + y1) / 2.0
    points = dict(zip(CORNER_ORDER, corners(spec)))
    points["e"] = to_doc(spec, x1, mid_y)
    points["w"] = to_doc(spec, x0, mid_y)
    points["rot"] = rotate_handle(spec)
    points["pivot"] = pivot_point(spec, pivot)
    return points


def hit(spec, x, y):
    """True when document ``(x, y)`` falls inside ``spec``'s block."""
    lx, ly = to_local(spec, x, y)
    x0, y0, x1, y1 = local_rect(spec)
    return x0 <= lx <= x1 and y0 <= ly <= y1


def hit_handle(spec, x, y, tolerance, pivot=None, pivot_enabled=True):
    """A named handle under ``(x, y)``, ``"move"`` inside the block, or ``None``.

    ``pivot_enabled`` gates the rotation-origin marker: the text tool only
    exposes it while Alt is held, so a plain click at the block centre still
    places the caret.
    """
    points = handle_points(spec, pivot)
    keys = (["pivot"] if pivot_enabled else []) + ["rot"] \
        + list(CORNER_ORDER) + list(EDGE_ORDER)
    for key in keys:
        hx, hy = points[key]
        if abs(hx - x) <= tolerance and abs(hy - y) <= tolerance:
            return key
    if hit(spec, x, y):
        return "move"
    return None


# ------------------------------------------------------------------ caret
def _align_offset(align, box_width, line_width):
    """Horizontal shift of a line within the box for the given alignment."""
    if align == "center":
        return (box_width - line_width) / 2.0
    if align == "right":
        return box_width - line_width
    return 0.0


def _justify_space(spec, row, line_width, max_width):
    """Extra pixels added after each space to justify ``row``.

    Justification needs a layout box and skips the last row of a paragraph, so
    paragraph endings stay ragged. A row without spaces cannot stretch and is
    left as-is.
    """
    if spec.get("align") != "justify" or box_width(spec) <= 0:
        return 0.0
    if row.get("para_end"):
        return 0.0
    spaces = row.get("text", "").count(" ")
    if spaces <= 0:
        return 0.0
    return max(0.0, max_width - line_width) / spaces


def _row_offset(spec, line_width, max_width):
    """Left edge of a row; justified rows always start at the box edge."""
    if spec.get("align") == "justify":
        return 0.0
    return _align_offset(spec.get("align", "left"), max_width, line_width)


def _justified_prefix_width(font, prefix, track, per_space):
    """Advance width of ``prefix`` including its share of justification."""
    return _prefix_width(font, prefix, track) + per_space * prefix.count(" ")


def _caret_row(rows, caret):
    """Index of the laid-out row holding ``caret`` (the later row on a break)."""
    index = 0
    for position, row in enumerate(rows):
        if row["start"] <= caret:
            index = position
        else:
            break
    return index


def caret_point(spec, caret):
    """``(dx, dy, height)`` of the caret for ``caret``, in local space."""
    text = display_text(spec)
    try:
        caret = int(caret)
    except (TypeError, ValueError):
        caret = len(text)
    caret = max(0, min(len(text), caret))
    font = load_font(spec)
    track = letter_spacing(spec)
    rows = visual_lines(spec)
    widths = [_prefix_width(font, row["text"], track) for row in rows]
    box = box_width(spec)
    max_width = box if box > 0 else (max(widths) if widths else 0.0)
    tops, _height, _advance = _row_tops(spec, rows)
    index = _caret_row(rows, caret)
    row = rows[index] if rows else {"text": "", "start": 0}
    line_width = widths[index] if index < len(widths) else 0.0
    per_space = _justify_space(spec, row, line_width, max_width)
    col = max(0, min(caret - row["start"], len(row["text"])))
    dx = _justified_prefix_width(font, row["text"][:col], track, per_space)
    dx += _row_offset(spec, line_width, max_width)
    ascent, descent = _metrics(font, spec)
    dy = tops[index] if index < len(tops) else 0.0
    return dx, dy, ascent + descent


def caret_from_point(spec, x, y):
    """The caret index nearest document ``(x, y)`` (used when clicking the text)."""
    font = load_font(spec)
    track = letter_spacing(spec)
    lx, ly = to_local(spec, x, y)
    rows = visual_lines(spec)
    if not rows:
        return 0
    widths = [_prefix_width(font, item["text"], track) for item in rows]
    box = box_width(spec)
    max_width = box if box > 0 else (max(widths) if widths else 0.0)
    tops, _height, _advance = _row_tops(spec, rows)
    index = 0
    for position, top in enumerate(tops):
        if ly >= top:
            index = position
        else:
            break
    index = max(0, min(len(rows) - 1, index))
    row = rows[index]
    line = row["text"]
    line_width = widths[index]
    per_space = _justify_space(spec, row, line_width, max_width)
    rel_x = lx - _row_offset(spec, line_width, max_width)
    best = None
    col = 0
    for position in range(len(line) + 1):
        distance = abs(
            _justified_prefix_width(font, line[:position], track, per_space) - rel_x
        )
        if best is None or distance < best:
            best = distance
            col = position
    return row["start"] + col


# --------------------------------------------------------------- navigation
def _is_word_char(char):
    """Whether ``char`` counts as part of a word for word-wise navigation."""
    return char.isalnum() or char == "_"


def word_left(text, caret):
    """Caret index at the start of the word before ``caret`` (Ctrl+Left)."""
    index = max(0, min(len(text), int(caret)))
    while index > 0 and not _is_word_char(text[index - 1]):
        index -= 1
    while index > 0 and _is_word_char(text[index - 1]):
        index -= 1
    return index


def word_right(text, caret):
    """Caret index at the end of the word after ``caret`` (Ctrl+Right)."""
    length = len(text)
    index = max(0, min(length, int(caret)))
    while index < length and not _is_word_char(text[index]):
        index += 1
    while index < length and _is_word_char(text[index]):
        index += 1
    return index


def word_bounds(text, position):
    """The ``(start, end)`` caret range of the word at ``position``.

    Used by double-click selection. When ``position`` sits on a word edge the
    word to the left is preferred; outside a word the run of non-word characters
    (whitespace or punctuation) up to the line break is returned.
    """
    text = text or ""
    length = len(text)
    position = max(0, min(length, int(position)))
    if length == 0:
        return (0, 0)
    if position >= length or not _is_word_char(text[position]):
        if position > 0 and _is_word_char(text[position - 1]):
            position -= 1
        elif position >= length:
            position = length - 1
    if _is_word_char(text[position]):
        start = position
        while start > 0 and _is_word_char(text[start - 1]):
            start -= 1
        end = position
        while end < length and _is_word_char(text[end]):
            end += 1
        return (start, end)
    start = position
    while start > 0 and text[start - 1] != "\n" and not _is_word_char(text[start - 1]):
        start -= 1
    end = position
    while end < length and text[end] != "\n" and not _is_word_char(text[end]):
        end += 1
    return (start, end)


def line_bounds(text, position):
    """The ``(start, end)`` caret range of the logical line at ``position``."""
    text = text or ""
    length = len(text)
    position = max(0, min(length, int(position)))
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end == -1:
        end = length
    return (start, end)


def caret_vertical(spec, caret, direction):
    """Move ``caret`` one visual row ``direction`` (``-1`` up, ``1`` down).

    The caret keeps its horizontal position by landing on the column nearest the
    current one in the target row, or stays put at the first/last row. Works
    across soft-wrapped rows because it walks the laid-out rows, not logical
    lines.
    """
    text = display_text(spec)
    caret = max(0, min(len(text), int(caret)))
    rows = visual_lines(spec)
    if not rows:
        return caret
    font = load_font(spec)
    track = letter_spacing(spec)
    widths = [_prefix_width(font, row["text"], track) for row in rows]
    box = box_width(spec)
    max_width = box if box > 0 else (max(widths) if widths else 0.0)
    index = _caret_row(rows, caret)
    target = index + (1 if direction > 0 else -1)
    if target < 0 or target >= len(rows):
        return caret
    current_x, _dy, _height = caret_point(spec, caret)
    row = rows[target]
    line_width = widths[target]
    per_space = _justify_space(spec, row, line_width, max_width)
    offset = _row_offset(spec, line_width, max_width)
    best = None
    col = 0
    for position in range(len(row["text"]) + 1):
        distance = abs(
            offset + _justified_prefix_width(
                font, row["text"][:position], track, per_space) - current_x
        )
        if best is None or distance < best:
            best = distance
            col = position
    return row["start"] + col


def selection_rects(spec, start, end):
    """Quads (document pixels) covering the selected caret range ``[start, end)``.

    One quad per laid-out row the range touches, so a selection that spans
    several visual rows (including soft-wrapped ones) highlights each row on its
    own. A selected line break adds a small sliver past the end of the row so
    the newline is visible.
    """
    text = display_text(spec)
    start, end = sorted((int(start), int(end)))
    start = max(0, min(len(text), start))
    end = max(0, min(len(text), end))
    if end <= start:
        return []
    font = load_font(spec)
    track = letter_spacing(spec)
    rows = visual_lines(spec)
    widths = [_prefix_width(font, row["text"], track) for row in rows]
    box = box_width(spec)
    max_width = box if box > 0 else (max(widths) if widths else 0.0)
    ascent, descent = _metrics(font, spec)
    tops, _block_height, _advance = _row_tops(spec, rows)
    height = ascent + descent
    break_width = max(_length(font, " "), 2.0)
    quads = []
    for index, row in enumerate(rows):
        row_start = row["start"]
        row_end = row_start + len(row["text"])
        line = row["text"]
        lo = max(start, row_start)
        hi = min(end, row_end)
        covers_break = (
            row_end < len(text) and text[row_end] == "\n"
            and start <= row_end < end
        )
        if lo >= hi and not covers_break:
            continue
        line_width = widths[index]
        per_space = _justify_space(spec, row, line_width, max_width)
        offset = _row_offset(spec, line_width, max_width)
        x0 = offset + _justified_prefix_width(
            font, line[:lo - row_start], track, per_space)
        x1 = offset + _justified_prefix_width(
            font, line[:hi - row_start], track, per_space)
        if covers_break:
            x1 += break_width
        y0 = tops[index] if index < len(tops) else 0.0
        y1 = y0 + height
        quads.append([
            to_doc(spec, x0, y0), to_doc(spec, x1, y0),
            to_doc(spec, x1, y1), to_doc(spec, x0, y1),
        ])
    return quads


def session_geometry(spec, caret, anchor=None, pivot=None):
    """Overlay geometry (document pixels) for an in-progress text edit.

    Returns the rotated frame quad, its corner and edge handles, the rotation
    stem, the rotation-origin marker, the caret segment and any selected-range
    quads so the canvas can draw an interactive box with a text highlight.
    """
    quad = corners(spec)
    points = handle_points(spec, pivot)
    rot = points["rot"]
    x0, y0, x1, _ = local_rect(spec)
    top_mid = to_doc(spec, (x0 + x1) / 2.0, y0)
    dx, dy, height = caret_point(spec, caret)
    selection = selection_rects(spec, anchor, caret) if anchor is not None else []
    return {
        "frame": quad,
        "handles": [points[name] for name in CORNER_ORDER + EDGE_ORDER],
        "rotate": rot,
        "stem": (top_mid, rot),
        "pivot": points["pivot"],
        "caret": (to_doc(spec, dx, dy), to_doc(spec, dx, dy + height)),
        "caret_on": True,
        "selection": selection,
    }


# ----------------------------------------------------------------- render
def _draw_lines(draw, spec, origin, fill, stroke_fill=None, stroke_width=0):
    """Draw ``spec``'s laid-out rows with tracking and stroke, at ``origin``."""
    text = display_text(spec)
    if not text:
        return
    font = load_font(spec)
    track = letter_spacing(spec)
    rows = visual_lines(spec)
    ascent, _descent = _metrics(font, spec)
    widths = [_prefix_width(font, row["text"], track) for row in rows]
    box = box_width(spec)
    max_width = box if box > 0 else (max(widths) if widths else 0.0)
    tops, _height, _advance = _row_tops(spec, rows)
    underline = bool(spec.get("underline"))
    strike = bool(spec.get("strikethrough"))
    size = int(spec.get("size", DEFAULT_SIZE))
    thickness = max(1, int(round(size / 14.0)))
    underline_offset = max(1.0, size * 0.08)
    strike_offset = ascent * 0.55
    ox, oy = origin
    for index, row in enumerate(rows):
        line = row["text"]
        line_width = widths[index]
        per_space = _justify_space(spec, row, line_width, max_width)
        x = ox + _row_offset(spec, line_width, max_width)
        y = oy + (tops[index] if index < len(tops) else 0.0)
        if track == 0 and per_space == 0:
            draw.text(
                (x, y), line, font=font, fill=fill,
                stroke_width=stroke_width, stroke_fill=stroke_fill,
            )
        else:
            # Draw glyph by glyph for tracking, but keep each base character with
            # its combining marks so accents are never rendered on their own.
            cursor = x
            cluster = ""
            for char in line:
                if cluster and _is_combining(char):
                    cluster += char
                    continue
                if cluster:
                    draw.text(
                        (cursor, y), cluster, font=font, fill=fill,
                        stroke_width=stroke_width, stroke_fill=stroke_fill,
                    )
                    cursor += _cluster_width(font, cluster) + track
                    if cluster == " ":
                        cursor += per_space
                cluster = char
            if cluster:
                draw.text(
                    (cursor, y), cluster, font=font, fill=fill,
                    stroke_width=stroke_width, stroke_fill=stroke_fill,
                )
        drawn_width = line_width + per_space * line.count(" ")
        if drawn_width <= 0:
            continue
        if underline:
            uy = y + ascent + underline_offset
            draw.line([(x, uy), (x + drawn_width, uy)],
                      fill=fill, width=thickness)
        if strike:
            sy = y + strike_offset
            draw.line([(x, sy), (x + drawn_width, sy)],
                      fill=fill, width=thickness)


def _compose_sprite(spec):
    """Render ``spec`` (text, stroke and shadow) to one RGBA sprite.

    Returns ``(sprite, anchor_px)`` where ``anchor_px`` is the sprite pixel that
    corresponds to the text block's local origin.
    """
    x0, y0, x1, y1 = local_rect(spec)
    stroke = spec.get("stroke")
    stroke_width = int(stroke["width"]) if stroke else 0
    pad = 4 + stroke_width
    block_w = max(1, int(math.ceil(x1 - x0)) + pad * 2)
    block_h = max(1, int(math.ceil(y1 - y0)) + pad * 2)
    origin = (pad - x0, pad - y0)

    block = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    alpha = int(round(255 * opacity(spec)))
    _draw_lines(
        ImageDraw.Draw(block), spec, origin,
        rgba(spec.get("color", "#000000"), alpha),
        stroke_fill=rgba(stroke["color"], alpha) if stroke else None,
        stroke_width=stroke_width,
    )
    anchor_px = (pad - x0, pad - y0)

    shadow = spec.get("shadow")
    if not shadow:
        return block, anchor_px
    dx, dy, blur = shadow["dx"], shadow["dy"], shadow["blur"]
    margin = int(math.ceil(max(abs(dx), abs(dy)) + blur * 2)) + 2
    sprite = Image.new("RGBA", (block_w + 2 * margin, block_h + 2 * margin), (0, 0, 0, 0))
    shadow_img = Image.new("RGBA", (block_w, block_h), (0, 0, 0, 0))
    _draw_lines(
        ImageDraw.Draw(shadow_img), spec, origin,
        rgba(shadow["color"], alpha),
    )
    if blur > 0:
        shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(blur))
    sprite.alpha_composite(shadow_img, dest=(margin + dx, margin + dy))
    sprite.alpha_composite(block, dest=(margin, margin))
    return sprite, (anchor_px[0] + margin, anchor_px[1] + margin)


def paste(image, spec, offset=(0, 0)):
    """Composite ``spec``'s text onto ``image``, minus the layer ``offset``.

    Returns the resulting image (mutated in place). ``offset`` is the layer's
    document offset, so an anchor in document space lands correctly in the layer
    buffer.
    """
    if not spec.get("text", "") or spec.get("hidden"):
        return image
    sprite, anchor_px = _compose_sprite(spec)
    ox, oy = offset
    ax = spec.get("x", 0.0) - ox
    ay = spec.get("y", 0.0) - oy
    theta = _theta(spec)
    if abs(theta) < 1e-9:
        image.alpha_composite(
            sprite, dest=(int(round(ax - anchor_px[0])), int(round(ay - anchor_px[1]))),
        )
        return image
    rotated = sprite.rotate(-math.degrees(theta), resample=Image.BICUBIC, expand=True)
    cx, cy = (local_rect(spec)[0] + local_rect(spec)[2]) / 2.0, \
        (local_rect(spec)[1] + local_rect(spec)[3]) / 2.0
    rx, ry = rotate_point(theta, cx, cy)
    dest = (
        int(round(ax + rx - rotated.width / 2.0)),
        int(round(ay + ry - rotated.height / 2.0)),
    )
    image.alpha_composite(rotated, dest=dest)
    return image
