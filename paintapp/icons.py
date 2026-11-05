"""Tabler icon rendering for the small status badges in the UI.

The SVG sources live in ``paintapp/assets/icons/tabler`` and are vendored from
Tabler Icons (MIT; see the ``LICENSE`` beside them). They are rasterised on
demand with ``resvg`` (the ``resvg-py`` package), tinted by swapping the source
``currentColor`` for the requested theme colour, and cached as ``PhotoImage``s
keyed on ``(name, size, color)`` so rebuilding a list reuses the renders.

Adding an icon later is just a matter of dropping its Tabler outline SVG into
the folder and referencing its file stem; :func:`photo` renders whatever is
there. ``resvg-py`` is the only extra dependency and it is a nicety, not a
requirement: when it or the assets are missing :func:`available` reports
``False`` and callers fall back to text glyphs.
"""

import io
import os

from PIL import Image, ImageOps, ImageTk

# The vendored Tabler outline SVGs. A badge names an icon by its file stem, so
# adding one is: drop ``<name>.svg`` beside the rest and reference it.
ICON_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "icons", "tabler"
)

try:
    import resvg_py
except Exception:  # pragma: no cover - the text-glyph fallback is what matters
    resvg_py = None

# (name, size, color) -> PhotoImage, so two rows asking for the same badge
# share one render. Non-icon text badges never touch this cache.
_CACHE = {}
# Icons we already failed to render, so a broken file is not re-read every row.
_MISSING = set()


def available():
    """Whether the Tabler assets and the SVG rasteriser can be used."""
    return resvg_py is not None and os.path.isdir(ICON_DIR)


def clear():
    """Drop the cached photos, e.g. before a new Tk root replaces them."""
    _CACHE.clear()


def photo(name, size, color, padding=0):
    """A ``PhotoImage`` of Tabler ``name``, ``size`` px square, tinted ``color``.

    ``padding`` adds that many transparent pixels around the glyph. Tk ignores
    a button's own ``padx``/``pady`` when it shows an image, so an icon that
    needs breathing room inside its button gets it baked in here instead.

    Returns ``None`` when the icon cannot be rendered, so the caller can fall
    back to a text glyph. ``name`` of ``None`` (a text-only badge) also returns
    ``None``.
    """
    if not name:
        return None
    key = (name, size, color, padding)
    cached = _CACHE.get(key)
    if cached is not None:
        return cached
    image = _render(name, size, color)
    if image is None:
        return None
    if padding:
        image = ImageOps.expand(image, border=padding, fill=(0, 0, 0, 0))
    try:
        rendered = ImageTk.PhotoImage(image)
    except RuntimeError:
        # No Tk root yet; try again once one exists.
        return None
    _CACHE[key] = rendered
    return rendered


def _render(name, size, color):
    """Rasterise ``name`` at ``size`` px, tinted ``color``, or ``None``."""
    if name in _MISSING or not available():
        return None
    path = os.path.join(ICON_DIR, name + ".svg")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            svg = handle.read()
    except OSError:
        _MISSING.add(name)
        return None
    # Tabler outline icons stroke with ``currentColor``; swapping it for the
    # theme colour is the whole tint, since every path inherits the root stroke.
    if "currentColor" in svg:
        svg = svg.replace("currentColor", color)
    try:
        png = resvg_py.svg_to_bytes(svg_string=svg, width=size, height=size)
        image = Image.open(io.BytesIO(bytes(png)))
        image.load()
    except Exception:
        _MISSING.add(name)
        return None
    return image.convert("RGBA")
