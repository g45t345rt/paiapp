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

import base64
import io
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

# Fields larger than this are sampled through a lookup table of this many
# entries (well above the 256 output levels, so banding is not introduced).
_LUT_SIZE = 4096

# Downscale factor for draft (live-edit) gradient renders.
DRAFT_FACTOR = 2

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
    # Broadcast a row and a column instead of materialising two full meshgrids;
    # on large canvases (A4 at 300 dpi is ~8.7M px) this halves the work.
    grid_x = np.arange(width, dtype=np.float32)[None, :]
    grid_y = np.arange(height, dtype=np.float32)[:, None]
    dx, dy = bx - ax, by - ay
    span2 = dx * dx + dy * dy
    radius = math.sqrt(span2)

    def _zeros():
        return np.zeros((height, width), dtype=np.float32)

    if mode == "conical":
        angle = np.arctan2(-(grid_y - ay), grid_x - ax)
        base = math.atan2(-dy, dx)
        return ((angle - base) % (2.0 * math.pi)) / (2.0 * math.pi)
    if mode in ("radial", "square", "diamond"):
        if radius <= 1e-6:
            return _zeros()
        off_x, off_y = grid_x - ax, grid_y - ay
        if mode == "square":
            return np.maximum(np.abs(off_x), np.abs(off_y)) / radius
        if mode == "diamond":
            return (np.abs(off_x) + np.abs(off_y)) / radius
        return np.hypot(off_x, off_y) / radius

    # linear / reflected share the projected distance along the drag vector.
    if span2 <= 1e-6:
        return _zeros()
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
    if t.size <= _LUT_SIZE:
        flat = t.ravel()
        channels = [
            np.interp(flat, positions, colors[:, index]) for index in range(3)
        ]
        alpha = np.interp(flat, positions, alphas)
        rgb = np.stack(channels, axis=-1).reshape(shape[0], shape[1], 3)
        alpha = alpha.reshape(shape)
        return (
            np.clip(rgb, 0, 255).astype(np.uint8),
            np.clip(alpha, 0, 255).astype(np.uint8),
        )
    # Large fields: build a lookup table once and index it, which is far
    # cheaper than four full-size np.interp passes plus float->uint8 copies.
    rgb_image, alpha_image = _sample_images(t, stops)
    return np.asarray(rgb_image), np.asarray(alpha_image)


def _packed_lut(stops):
    """A ``_LUT_SIZE`` table of stops packed as little-endian RGBA uint32."""
    positions = np.array([float(s[0]) for s in stops], dtype=np.float32)
    colors = np.array([rgba(s[1])[:3] for s in stops], dtype=np.float32)
    alphas = np.array([float(s[2]) for s in stops], dtype=np.float32)
    ramp = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float32)
    lut = np.empty((_LUT_SIZE, 4), dtype=np.uint8)
    for index in range(3):
        lut[:, index] = np.clip(
            np.rint(np.interp(ramp, positions, colors[:, index])), 0, 255
        )
    lut[:, 3] = np.clip(np.rint(np.interp(ramp, positions, alphas)), 0, 255)
    return lut.view("<u4").ravel()


def _sample_images(t, stops):
    """Sample ``stops`` at ``t`` straight into ``(RGB, L)`` PIL images.

    One uint32 gather fills every RGBA pixel at once; PIL then splits the
    channels natively, avoiding numpy strided copies of the full canvas.
    """
    height, width = t.shape
    idx = np.clip(t, 0.0, 1.0, out=np.empty(t.shape, dtype=np.float32))
    idx *= _LUT_SIZE - 1
    idx += 0.5
    packed = _packed_lut(stops)[idx.astype(np.uint16)]
    image = Image.frombuffer("RGBA", (width, height), packed, "raw", "RGBA", 0, 1)
    return image.convert("RGB"), image.getchannel("A")


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
    if t.size > _LUT_SIZE:
        return _sample_images(t, stops)
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


def _point(value, default=(0.0, 0.0)):
    """Coerce ``value`` to a finite ``(x, y)`` pair."""
    try:
        x, y = value[0], value[1]
        return (float(x), float(y))
    except (TypeError, IndexError, ValueError):
        return (float(default[0]), float(default[1]))


def _blend_key(value):
    """Normalise a blend mode label/key to the key used by ``brush``."""
    if not value:
        return "normal"
    key = str(value).strip().lower()
    if key in brush.BLEND_MODES:
        return key
    for mode, label in brush.BLEND_MODES.items():
        if label.lower() == key:
            return mode
    return "normal"


def _decode_mask(data):
    """Decode a stored base64 PNG mask into an ``L`` image (or ``None``)."""
    if not isinstance(data, str) or not data:
        return None
    try:
        raw = base64.b64decode(data.encode("ascii"))
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image.convert("L")
    except Exception:
        return None


def _encode_mask(mask):
    """Encode an ``L`` mask as base64 PNG for the project manifest."""
    if not isinstance(mask, Image.Image):
        return None
    buf = io.BytesIO()
    mask.convert("L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _mask_image(mask):
    """Return a usable ``L`` mask from an image or a stored base64 string."""
    if isinstance(mask, Image.Image):
        return mask.convert("L")
    return _decode_mask(mask)


def normalize(spec):
    """Return a valid gradient-object spec built from ``spec``.

    A gradient object keeps its axis in document space and the full Gradient
    tool settings (stops, type, spread, opacity and blend). Specs are kept on
    the layer so the gradient stays editable instead of being baked into the
    layer pixels. An optional selection mask is carried with it, so a live
    gradient still respects the selection it was drawn through.
    """
    if not isinstance(spec, dict):
        return None
    stops = normalize_stops(spec.get("stops"))
    if len(stops) < 2:
        stops = [
            [0.0, str(spec.get("from") or "#000000"), 255],
            [1.0, str(spec.get("to") or "#ffffff"), 255],
        ]
    mode = str(spec.get("mode") or "linear").lower()
    if mode not in dict(MODES):
        mode = "linear"
    spread = str(spec.get("spread") or "clamp").lower()
    if spread not in dict(SPREADS):
        spread = "clamp"
    try:
        opacity = max(1, min(100, int(spec.get("opacity", 100))))
    except (TypeError, ValueError):
        opacity = 100
    mask = _mask_image(spec.get("mask"))
    return {
        "start": _point(spec.get("start")),
        "end": _point(spec.get("end")),
        "mode": mode,
        "stops": stops,
        "spread": spread,
        "opacity": opacity,
        "blend": _blend_key(spec.get("blend")),
        "hidden": bool(spec.get("hidden", False)),
        "mask": mask,
    }


def clone(spec):
    """Return an independent copy of a gradient-object spec."""
    normalized = normalize(spec)
    if normalized is None:
        return None
    if normalized["mask"] is not None:
        normalized["mask"] = normalized["mask"].copy()
    return normalized


def to_json(spec):
    """A JSON-safe copy of ``spec`` (the mask is stored as base64 PNG)."""
    normalized = normalize(spec)
    if normalized is None:
        return None
    normalized["mask"] = _encode_mask(normalized["mask"])
    return normalized


def from_json(entry):
    """Rebuild a spec from :func:`to_json` output."""
    return normalize(entry)


def describe(spec):
    """A short human-readable label for the layer panel row."""
    normalized = normalize(spec)
    if normalized is None:
        return "Gradient"
    label = dict(MODES).get(normalized["mode"], "Linear")
    return f"{label} gradient"


def mask_for_size(spec, size):
    """The spec's optional coverage mask, cropped/padded to ``size``."""
    mask = _mask_image(spec.get("mask"))
    if mask is None:
        return None
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    if mask.size == (width, height):
        return mask
    fitted = Image.new("L", (width, height), 0)
    fitted.paste(mask, (0, 0))
    return fitted


def paste(image, spec, offset=(0, 0), draft=False):
    """Draw one gradient object over ``image`` (RGBA) at ``offset``.

    The gradient covers the full layer buffer, like the Gradient tool's fill,
    and is composited through the spec's stops' alpha, opacity, blend and
    optional selection mask. ``draft`` rasterises the ramp at a reduced
    resolution and upscales it, for fast feedback while settings are being
    dragged; gradients are smooth, so the difference is barely visible.
    """
    if spec.get("hidden"):
        return image
    normalized = normalize(spec)
    if normalized is None:
        return image
    width, height = image.size
    ox, oy = int(offset[0]), int(offset[1])
    start = (normalized["start"][0] - ox, normalized["start"][1] - oy)
    end = (normalized["end"][0] - ox, normalized["end"][1] - oy)
    factor = DRAFT_FACTOR if draft and width * height > 1_000_000 else 1
    if factor > 1:
        small = (max(1, -(-width // factor)), max(1, -(-height // factor)))
        rgb, alpha = gradient_layers(
            small,
            (start[0] / factor, start[1] / factor),
            (end[0] / factor, end[1] / factor),
            normalized["mode"],
            stops=normalized["stops"], spread=normalized["spread"],
        )
        full = (small[0] * factor, small[1] * factor)
        rgb = rgb.resize(full, Image.NEAREST)
        alpha = alpha.resize(full, Image.NEAREST)
        if full != (width, height):
            rgb = rgb.crop((0, 0, width, height))
            alpha = alpha.crop((0, 0, width, height))
    else:
        rgb, alpha = gradient_layers(
            (width, height), start, end, normalized["mode"],
            stops=normalized["stops"], spread=normalized["spread"],
        )
    coverage = alpha
    mask = mask_for_size(normalized, (width, height))
    if mask is not None:
        coverage = ImageChops.multiply(coverage, mask)
    if normalized["opacity"] < 100:
        factor = normalized["opacity"] / 100.0
        coverage = coverage.point(lambda value: int(value * factor))
    return brush.composite_image(image, rgb, coverage, normalized["blend"])


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
