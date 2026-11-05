"""Gradient fills behind the Gradient tool.

A gradient is rasterised to an RGB image plus an 8-bit alpha ramp covering the
layer's own pixel space, then composited through a coverage mask with
:func:`brush.composite_image`, so a fill obeys the tool's opacity, blend mode,
per-stop alpha and (optionally) the active selection exactly like a brush
stroke.

``linear`` interpolates along the drag vector, ``radial`` by distance from the
drag start, ``square`` by Chebyshev distance, ``diamond`` by Manhattan distance,
``conical`` by the swept angle and ``reflected`` mirrors the linear ramp. A
``spread`` mode (clamp/repeat/mirror) decides how values outside ``0..1`` fold.

Colours come from a sorted list of stops ``[position, "#rrggbb", alpha]`` so a
gradient can carry any number of colours and fade to transparency. The maths
runs in numpy, which is already a hard dependency of the app's engine.
"""

import math

import numpy as np
from PIL import Image, ImageChops

from . import brush
from .constants import rgba

MODES = [
    ("linear", "Linear"),
    ("reflected", "Reflected"),
    ("radial", "Radial"),
    ("square", "Square"),
    ("diamond", "Diamond"),
    ("conical", "Conical"),
]

SPREADS = [("clamp", "Clamp"), ("repeat", "Repeat"), ("mirror", "Mirror")]

# Fixed presets, expressed as stop lists. Presets that follow the tool's own
# From/To colours are resolved by :func:`preset_stops` instead.
PRESET_STOPS = {
    "black_white": [[0.0, "#000000", 255], [1.0, "#ffffff", 255]],
    "white_black": [[0.0, "#ffffff", 255], [1.0, "#000000", 255]],
    "black_transparent": [[0.0, "#000000", 255], [1.0, "#000000", 0]],
    "white_transparent": [[0.0, "#ffffff", 255], [1.0, "#ffffff", 0]],
    "rainbow": [
        [0.0, "#ff0000", 255], [0.2, "#ffff00", 255], [0.4, "#00ff00", 255],
        [0.6, "#00ffff", 255], [0.8, "#0000ff", 255], [1.0, "#ff00ff", 255],
    ],
    "sunset": [
        [0.0, "#2b1055", 255], [0.5, "#ff6f61", 255], [1.0, "#ffe29a", 255],
    ],
    "ocean": [
        [0.0, "#001f3f", 255], [0.5, "#0074d9", 255], [1.0, "#7fdbff", 255],
    ],
}

PRESETS = [
    ("custom", "Custom"),
    ("black_white", "Black / White"),
    ("white_black", "White / Black"),
    ("from_to", "From / To"),
    ("from_transparent", "From / Transparent"),
    ("to_transparent", "To / Transparent"),
    ("black_transparent", "Black / Transparent"),
    ("white_transparent", "White / Transparent"),
    ("rainbow", "Rainbow"),
    ("sunset", "Sunset"),
    ("ocean", "Ocean"),
]


def _hex(rgb):
    """Format an ``(r, g, b)`` triple as ``#rrggbb``."""
    return "#%02x%02x%02x" % (int(rgb[0]), int(rgb[1]), int(rgb[2]))


def preset_stops(key, from_color, to_color):
    """Return a fresh stop list for ``key``, or ``None`` for "custom".

    ``from_color``/``to_color`` feed the presets that follow the tool's own
    From/To colours; the fixed presets ignore them.
    """
    if key == "custom":
        return None
    if key == "from_to":
        return [[0.0, from_color, 255], [1.0, to_color, 255]]
    if key == "from_transparent":
        return [[0.0, from_color, 255], [1.0, from_color, 0]]
    if key == "to_transparent":
        return [[0.0, to_color, 255], [1.0, to_color, 0]]
    stops = PRESET_STOPS.get(key)
    return [list(stop) for stop in stops] if stops else None


def normalize_stops(stops):
    """Sanitize persisted stops into sorted ``[pos, "#rrggbb", alpha]`` rows."""
    clean = []
    for stop in stops or ():
        try:
            pos, color, alpha = stop[0], stop[1], stop[2]
        except (TypeError, IndexError):
            continue
        try:
            pos = max(0.0, min(1.0, float(pos)))
            alpha = max(0, min(255, int(alpha)))
        except (TypeError, ValueError):
            continue
        clean.append([pos, str(color), alpha])
    clean.sort(key=lambda row: row[0])
    return clean


def _spread(t, mode):
    """Fold ``t`` back into ``0..1`` according to a spread mode."""
    if mode == "repeat":
        return t - np.floor(t)
    if mode == "mirror":
        folded = np.abs(np.mod(t, 2.0))
        return np.where(folded > 1.0, 2.0 - folded, folded)
    return np.clip(t, 0.0, 1.0)


def _field(size, start, end, mode):
    """The raw 0..1 (or beyond) gradient parameter for every pixel."""
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    dx, dy = bx - ax, by - ay
    span2 = dx * dx + dy * dy
    radius = math.sqrt(span2)

    if mode == "conical":
        angle = np.arctan2(-(grid_y - ay), grid_x - ax)
        base = math.atan2(-dy, dx)
        return ((angle - base) % (2.0 * math.pi)) / (2.0 * math.pi)
    if mode in ("radial", "square", "diamond"):
        if radius <= 1e-6:
            return np.zeros_like(grid_x)
        off_x, off_y = grid_x - ax, grid_y - ay
        if mode == "square":
            return np.maximum(np.abs(off_x), np.abs(off_y)) / radius
        if mode == "diamond":
            return (np.abs(off_x) + np.abs(off_y)) / radius
        return np.hypot(off_x, off_y) / radius

    # linear / reflected share the projected distance along the drag vector.
    if span2 <= 1e-6:
        return np.zeros_like(grid_x)
    u = ((grid_x - ax) * dx + (grid_y - ay) * dy) / span2
    if mode == "reflected":
        folded = np.abs(np.mod(u, 2.0))
        return np.where(folded > 1.0, 2.0 - folded, folded)
    return u


def _sample(t, stops):
    """Sample ``stops`` at parameter array ``t``; returns (rgb, alpha) uint8."""
    positions = np.array([float(s[0]) for s in stops], dtype=np.float32)
    colors = np.array([rgba(s[1])[:3] for s in stops], dtype=np.float32)
    alphas = np.array([float(s[2]) for s in stops], dtype=np.float32)
    shape = t.shape
    flat = t.ravel()
    channels = [np.interp(flat, positions, colors[:, index]) for index in range(3)]
    alpha = np.interp(flat, positions, alphas)
    rgb = np.stack(channels, axis=-1).reshape(shape[0], shape[1], 3)
    alpha = alpha.reshape(shape)
    return (
        np.clip(rgb, 0, 255).astype(np.uint8),
        np.clip(alpha, 0, 255).astype(np.uint8),
    )


def gradient_layers(size, start, end, mode="linear", stops=None,
                    color_a=(0, 0, 0), color_b=(255, 255, 255),
                    spread="clamp"):
    """Rasterise a gradient of ``size`` into ``(rgb_image, alpha_image)``.

    ``stops`` is an optional sorted list of ``[pos, "#rrggbb", alpha]``; when
    omitted the two ``color_a``/``color_b`` endpoints are used instead.
    """
    if not stops:
        stops = [[0.0, _hex(color_a), 255], [1.0, _hex(color_b), 255]]
    if len(stops) < 2:
        stops = stops * 2 if stops else [[0.0, "#000000", 255],
                                         [1.0, "#ffffff", 255]]
    t = _spread(_field(size, start, end, mode), spread)
    rgb, alpha = _sample(t, stops)
    return Image.fromarray(rgb, "RGB"), Image.fromarray(alpha, "L")


def gradient_rgb(size, start, end, mode="linear",
                 color_a=(0, 0, 0), color_b=(255, 255, 255),
                 stops=None, spread="clamp"):
    """Rasterise a gradient of ``size`` from ``start`` to ``end`` (RGB image)."""
    rgb, _alpha = gradient_layers(
        size, start, end, mode, stops=stops, color_a=color_a,
        color_b=color_b, spread=spread,
    )
    return rgb


def strip_image(stops, width, height):
    """Render a horizontal RGBA preview strip of ``stops`` for the editor."""
    width = max(2, int(width))
    height = max(1, int(height))
    positions = np.linspace(0.0, 1.0, width, dtype=np.float32)
    t = np.tile(positions, (height, 1))
    rgb, alpha = _sample(t, stops)
    out = Image.fromarray(rgb, "RGB").convert("RGBA")
    out.putalpha(Image.fromarray(alpha, "L"))
    return out


def apply(layer, gradient, coverage, opacity=100, blend="normal", alpha=None):
    """Composite ``gradient`` (RGB, layer-sized) into ``layer`` via ``coverage``.

    ``alpha`` is an optional per-pixel transparency ramp (``L`` image) folded
    into the coverage mask so a gradient can fade out along its axis.
    """
    if alpha is not None:
        coverage = ImageChops.multiply(coverage, alpha)
    if opacity < 100:
        factor = max(1, min(100, int(round(opacity)))) / 100.0
        coverage = coverage.point(lambda value: int(value * factor))
    layer.set_image(
        brush.composite_image(layer.image, gradient, coverage, blend)
    )
