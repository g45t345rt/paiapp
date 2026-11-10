"""The stamping engine behind the brush, eraser and mask tools.

A stroke is built from soft "dabs" rather than a single hard line so the tools
can offer hardness, opacity, flow, spacing, anti-aliasing and tip shape. An
effect-line pattern (see :data:`EFFECT_PATTERNS`) swaps that tip for a
directional stamp whose orientation follows the stroke, so a drag builds manga
action lines. The painted result is accumulated into a per-stroke coverage
buffer (one ``"L"`` image) and the layer is rebuilt from a snapshot taken when
the stroke began, so overlapping dabs blend predictably and the overall opacity
stays a ceiling no matter how slowly the pointer moves.
"""

import colorsys
import math
import random

import numpy as np

from PIL import Image, ImageChops, ImageDraw, ImageFilter


def _union_rect(a, b):
    """The bounding ``(x0, y0, x1, y1)`` rectangle covering ``a`` and ``b``."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


# RGB blend operators available for the brush and the colour eraser. ``normal``
# is handled separately because it needs no channel maths.
BLEND_MODES = {
    "normal": "Normal",
    "darken": "Darken",
    "multiply": "Multiply",
    "color_burn": "Color Burn",
    "linear_burn": "Linear Burn",
    "darker_color": "Darker Color",
    "lighten": "Lighten",
    "screen": "Screen",
    "color_dodge": "Color Dodge",
    "add": "Add",
    "lighter_color": "Lighter Color",
    "overlay": "Overlay",
    "soft_light": "Soft Light",
    "hard_light": "Hard Light",
    "vivid_light": "Vivid Light",
    "linear_light": "Linear Light",
    "pin_light": "Pin Light",
    "hard_mix": "Hard Mix",
    "difference": "Difference",
    "exclusion": "Exclusion",
    "subtract": "Subtract",
    "divide": "Divide",
    "hue": "Hue",
    "saturation": "Saturation",
    "color": "Color",
    "luminosity": "Luminosity",
}

# Modes PIL implements directly; the rest are computed in :func:`_blend_x`.
_BLEND_OPS = {
    "multiply": ImageChops.multiply,
    "screen": ImageChops.screen,
    "overlay": ImageChops.overlay,
    "lighten": ImageChops.lighter,
    "darken": ImageChops.darker,
    "add": ImageChops.add,
    "subtract": ImageChops.subtract,
    "hard_light": ImageChops.hard_light,
    "soft_light": ImageChops.soft_light,
    "difference": ImageChops.difference,
}

TIP_SHAPES = {"round": "Round", "square": "Square"}

# Dry-media textures stamped into a dab's alpha (see :func:`grain_tile`).
# ``"none"`` leaves the dab solid. The routines are procedural, so no image
# assets ship with the app.
GRAIN_TYPES = {
    "none": "None",
    "noise": "Noise",
    "paper": "Paper",
    "canvas": "Canvas",
    "halftone": "Halftone",
}

# Where the grain sits: ``canvas`` samples it at document position, so the
# texture stays put and the brush reveals its "tooth" as it passes; ``stamp``
# anchors it to each dab, so the pattern travels with the brush.
GRAIN_MODES = {"canvas": "Canvas", "stamp": "Stamp"}

#: Size of the generated grain tiles, in pixels.
_GRAIN_SIZE = 256
_GRAIN_CACHE = {}
_GRAIN_CACHE_LIMIT = 48


def _build_grain(grain):
    """Rasterise the base ``"L"`` texture for a grain type (before strength)."""
    n = _GRAIN_SIZE
    if grain == "canvas":
        img = Image.new("L", (n, n), 255)
        px = img.load()
        period = 16.0
        for y in range(n):
            b = 0.5 + 0.5 * math.cos(2.0 * math.pi * y / period)
            for x in range(n):
                a = 0.5 + 0.5 * math.cos(2.0 * math.pi * x / period)
                px[x, y] = int(max(0.0, min(255.0, 255.0 * (0.45 + 0.4 * a * b))))
        return img
    if grain == "halftone":
        img = Image.new("L", (n, n), 255)
        draw = ImageDraw.Draw(img)
        step, radius = 16, 4
        for yy in range(0, n, step):
            for xx in range(0, n, step):
                cx, cy = xx + step // 2, yy + step // 2
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius),
                             fill=8)
        return img
    # noise / paper: white-noise, softened into blotches for paper.
    rng = random.Random(0x5EED)
    img = Image.new("L", (n, n))
    img.putdata([rng.randint(0, 255) for _ in range(n * n)])
    if grain == "paper":
        img = img.resize((32, 32), Image.BILINEAR).resize((n, n), Image.BILINEAR)
    return img


def grain_tile(grain, strength, scale=1.0):
    """Return a cached grain tile for ``grain`` already faded by ``strength``.

    ``strength`` is ``0..1`` (``0`` yields a flat white tile, i.e. no effect);
    ``scale`` (>1 coarser, <1 finer) resizes the tile so the pattern reads at
    different sizes.
    """
    if grain not in GRAIN_TYPES or grain == "none":
        return None
    strength = _clamp01(strength)
    scale = max(0.25, min(4.0, float(scale)))
    key = (grain, int(round(strength * 20.0)), int(round(scale * 10.0)))
    cached = _GRAIN_CACHE.get(key)
    if cached is not None:
        return cached
    tile = _build_grain(grain)
    if abs(scale - 1.0) > 0.01:
        size = max(8, int(round(_GRAIN_SIZE * scale)))
        tile = tile.resize((size, size), Image.BILINEAR)
    white = Image.new("L", tile.size, 255)
    tile = Image.blend(white, tile, strength)
    if len(_GRAIN_CACHE) >= _GRAIN_CACHE_LIMIT:
        _GRAIN_CACHE.clear()
    _GRAIN_CACHE[key] = tile
    return tile


def _grain_region(tile, ox, oy, width, height):
    """A ``width``-by-``height`` grain patch tiled from ``tile`` at ``(ox, oy)``."""
    tw, th = tile.size
    region = Image.new("L", (width, height))
    start_x = ox % tw
    start_y = oy % th
    for yy in range(start_y - th, height + th, th):
        for xx in range(start_x - tw, width + tw, tw):
            region.paste(tile, (xx, yy))
    return region

# Effect-line patterns for the brush and eraser. Unlike a plain tip, these draw
# a directional ink shape so a dragged stroke builds manga action lines rather
# than a solid band; ``"none"`` keeps the normal round/square tip.
EFFECT_PATTERNS = {
    "none": "None",
    "speed": "Speed Lines",
    "focus": "Focus Lines",
    "flash": "Flash",
    "sparkle": "Sparkle",
}

_DAB_CACHE = {}
_DAB_CACHE_LIMIT = 64

# Dabs at least this wide are rasterised on the GPU (see ``gpuops.brush_dab``);
# smaller dabs stay on the PIL loop, which is faster than the upload overhead.
_GPU_DAB_MIN = 64

# Canvas-grain rects at least this many pixels are grained on the GPU (see
# ``gpuops.grain_overlay``); smaller ones use the PIL tiling, cheaper than the
# round-trip.
_GPU_GRAIN_MIN = 64 * 64
_EFFECT_CACHE = {}
_EFFECT_CACHE_LIMIT = 128


def _build_dab(size, hardness, antialias, tip):
    """Rasterise one dab as an ``"L"`` alpha stamp of ``size`` pixels.

    Large dabs are rendered on the GPU when a GL context is available (the PIL
    loop below is per-pixel Python, which is what makes big brushes slow); small
    dabs stay on the CPU, where they are cheaper than a GPU round-trip.
    """
    d = max(1, int(round(size)))
    if d == 1:
        return Image.new("L", (1, 1), 255)
    if d >= _GPU_DAB_MIN:
        from . import gpuops
        gpu_dab = gpuops.brush_dab(d, hardness, antialias, tip)
        if gpu_dab is not None:
            return gpu_dab
    # Anti-aliasing is done by supersampling then downscaling, which keeps the
    # softened rim of a low-hardness dab smooth even at small sizes.
    ss = 3 if antialias else 1
    n = d * ss
    img = Image.new("L", (n, n), 0)
    px = img.load()
    r = (n - 1) / 2.0
    core = r * max(0.0, min(100.0, hardness)) / 100.0
    falloff = max(1e-6, r - core)
    for y in range(n):
        dy = y - r
        for x in range(n):
            dx = x - r
            if tip == "square":
                dist = max(abs(dx), abs(dy))
            else:
                dist = math.hypot(dx, dy)
            if dist >= r:
                value = 0
            elif dist <= core:
                value = 255
            else:
                value = int(round(255.0 * (1.0 - (dist - core) / falloff)))
                if value < 0:
                    value = 0
            px[x, y] = value
    if ss > 1:
        img = img.resize((d, d), Image.LANCZOS)
    return img


def make_dab(size, hardness, antialias, tip):
    """Return a cached ``"L"`` dab stamp for the given brush parameters."""
    key = (max(1, int(round(size))), int(round(hardness)),
           bool(antialias), tip if tip in TIP_SHAPES else "round")
    cached = _DAB_CACHE.get(key)
    if cached is not None:
        return cached
    dab = _build_dab(*key)
    if len(_DAB_CACHE) >= _DAB_CACHE_LIMIT:
        _DAB_CACHE.clear()
    _DAB_CACHE[key] = dab
    return dab


def _build_effect_dab(size, pattern, angle, hardness, antialias):
    """Rasterise one directional effect dab as an ``"L"`` stamp.

    ``angle`` is in degrees clockwise. The pattern decides the ink shape:

    ``"speed"``/``"focus"``  parallel chords along ``angle``, so stamped dabs
        join into unbroken flow lines (``focus`` is used where each dab's angle
        points at a focal centre, making the lines converge on it).
    ``"flash"``              rays radiating from the dab centre (a sunburst).
    ``"sparkle"``            a four-point star with soft diagonals (a glint).
    """
    d = max(2, int(round(size)))
    ss = 3 if antialias else 1
    n = d * ss
    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    r = (n - 1) / 2.0
    # Softer brushes get thinner strokes: the stroke weight is a fraction of the
    # dab, shrinking a little as hardness rises.
    softness = min(100.0, max(0.0, hardness)) / 100.0
    weight = max(1, int(round(n * (0.12 - 0.06 * softness))))
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    if pattern in ("speed", "focus"):
        px, py = -sin, cos
        gap = max(2.0, n * 0.22)
        count = int(r / gap) + 1
        for j in range(-count, count + 1):
            ox, oy = px * j * gap, py * j * gap
            draw.line(
                (r + ox - cos * r, r + oy - sin * r,
                 r + ox + cos * r, r + oy + sin * r),
                fill=255, width=weight,
            )
    elif pattern == "flash":
        rays = 12
        inner = r * 0.15
        for i in range(rays):
            a = theta + math.pi * 2.0 * i / rays
            ca, sa = math.cos(a), math.sin(a)
            draw.line(
                (r + ca * inner, r + sa * inner, r + ca * r, r + sa * r),
                fill=255, width=weight,
            )
    else:  # sparkle
        diag = r * 0.55
        draw.line((r - r, r, r + r, r), fill=255, width=weight)
        draw.line((r, r - r, r, r + r), fill=255, width=weight)
        for ca, sa in ((0.7071, 0.7071), (0.7071, -0.7071),
                       (-0.7071, 0.7071), (-0.7071, -0.7071)):
            draw.line((r, r, r + ca * diag, r + sa * diag),
                      fill=255, width=weight)
    if ss > 1:
        img = img.resize((d, d), Image.LANCZOS)
    return img


def make_effect_dab(size, pattern, angle, hardness, antialias):
    """Return a cached directional dab for ``pattern`` at ``angle`` degrees.

    The angle is quantised to 5-degree steps so the cache stays small while the
    orientation still follows the stroke smoothly.
    """
    if pattern not in EFFECT_PATTERNS or pattern == "none":
        pattern = "speed"
    d = max(2, int(round(size)))
    step = int(round(float(angle) / 5.0)) % 72
    key = (d, pattern, step, int(round(hardness)), bool(antialias))
    cached = _EFFECT_CACHE.get(key)
    if cached is not None:
        return cached
    dab = _build_effect_dab(d, pattern, step * 5.0, hardness, antialias)
    if len(_EFFECT_CACHE) >= _EFFECT_CACHE_LIMIT:
        _EFFECT_CACHE.clear()
    _EFFECT_CACHE[key] = dab
    return dab


def _clamp01(value):
    """Clamp ``value`` into the ``0.0..1.0`` range."""
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else float(value))


class Dynamics:
    """Velocity-driven per-dab modulation plus start/end taper.

    Every strength is ``0.0..1.0`` and defaults to ``0``, so a default
    ``Dynamics`` leaves a stroke exactly as it was before this class existed.
    A non-zero strength means "let speed change this attribute": as the pointer
    moves faster, ``size``/``opacity``/``flow``/``hardness`` shrink and
    ``spacing`` widens, which is the mouse-usable stand-in for pen pressure.

    ``taper_in``/``taper_out`` are fade lengths in document pixels applied to the
    start and end of the stroke (short strokes simply fade proportionally);
    ``taper`` sets both. Speed arrives either as an
    explicit normalised ``0..1`` value on :meth:`Stroke.move_to` (the controller
    measures it from real event timing) or, when omitted, is estimated from the
    distance between consecutive pointer samples normalised by ``speed_ref``.
    Either way it is smoothed with an exponential average so it does not jitter
    at the mouse's polling rate.
    """

    def __init__(self, size=0.0, opacity=0.0, flow=0.0, hardness=0.0,
                 spacing=0.0, taper=0.0, taper_in=None, taper_out=None,
                 min_scale=0.15, smoothing=0.55, speed_ref=15.0):
        self.size = _clamp01(size)
        self.opacity = _clamp01(opacity)
        self.flow = _clamp01(flow)
        self.hardness = _clamp01(hardness)
        self.spacing = _clamp01(spacing)
        # ``taper`` sets both ends at once; ``taper_in``/``taper_out`` override
        # each end independently (a start wedge vs. a stroked-out finish).
        both = max(0.0, float(taper))
        self.taper_in = (max(0.0, float(taper_in)) if taper_in is not None
                         else both)
        self.taper_out = (max(0.0, float(taper_out)) if taper_out is not None
                          else both)
        # Floor the size a fully-fast dab may shrink to (fraction of base size).
        self.min_scale = _clamp01(min_scale)
        self.smoothing = _clamp01(smoothing)
        self.speed_ref = max(1e-3, float(speed_ref))

    @property
    def taper(self):
        """The longest of the two tapers (length of the taper window)."""
        return max(self.taper_in, self.taper_out)

    def active(self):
        """True when any strength would change a stroke from its base form."""
        return bool(self.size or self.opacity or self.flow or self.hardness
                    or self.spacing or self.taper_in or self.taper_out)


def _accumulate(coverage, dab, cx, cy, amount, cap):
    """Add ``dab`` (scaled by ``amount``) at ``(cx, cy)`` into ``coverage``.

    Coverage builds up additively so repeated dabs darken a pixel, but never
    past ``cap`` (the tool's opacity), which keeps that opacity a true ceiling.

    Returns the ``(left, top, right, bottom)`` rectangle actually touched, or
    ``None`` when the dab fell entirely outside the buffer. The caller unions
    these into the stroke's dirty box so :meth:`Stroke.commit` can rebuild only
    the changed region instead of the whole layer.
    """
    dw, dh = dab.size
    left = cx - dw // 2
    top = cy - dh // 2
    sx0 = max(0, -left)
    sy0 = max(0, -top)
    dx0 = max(0, left)
    dy0 = max(0, top)
    w = min(dw - sx0, coverage.width - dx0)
    h = min(dh - sy0, coverage.height - dy0)
    if w <= 0 or h <= 0:
        return None
    src = dab.crop((sx0, sy0, sx0 + w, sy0 + h))
    if amount < 1.0:
        src = src.point(lambda v: int(v * amount))
    patch = coverage.crop((dx0, dy0, dx0 + w, dy0 + h))
    combined = ImageChops.add(patch, src, scale=1.0)
    if cap < 255:
        combined = combined.point(lambda v: v if v <= cap else cap)
    coverage.paste(combined, (dx0, dy0))
    return (dx0, dy0, dx0 + w, dy0 + h)


def _rgb_array(image):
    return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _array_image(values):
    return Image.fromarray(
        np.clip(values * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8), "RGB")


def _rgb_to_hsl(arr):
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    top = np.maximum(np.maximum(r, g), b)
    bottom = np.minimum(np.minimum(r, g), b)
    light = (top + bottom) * 0.5
    delta = top - bottom
    safe = np.where(delta < 1e-6, 1.0, delta)
    sat = np.where(delta < 1e-6, 0.0,
                   delta / (1.0 - np.abs(2.0 * light - 1.0) + 1e-6))
    hue = np.where(
        top == r, ((g - b) / safe) % 6.0,
        np.where(top == g, (b - r) / safe + 2.0, (r - g) / safe + 4.0))
    hue = np.where(delta < 1e-6, 0.0, (hue / 6.0) % 1.0)
    return hue, sat, light


def _hsl_to_rgb(hue, sat, light):
    chroma = (1.0 - np.abs(2.0 * light - 1.0)) * sat
    sector = hue * 6.0
    x = chroma * (1.0 - np.abs(sector % 2.0 - 1.0))
    offset = light - chroma * 0.5
    index = np.floor(sector).astype(np.int64) % 6
    zero = np.zeros_like(chroma)
    r = np.select([index == 0, index == 1, index == 2, index == 3, index == 4,
                   index == 5], [chroma, x, zero, zero, x, chroma])
    g = np.select([index == 0, index == 1, index == 2, index == 3, index == 4,
                   index == 5], [x, chroma, chroma, x, zero, zero])
    b = np.select([index == 0, index == 1, index == 2, index == 3, index == 4,
                   index == 5], [zero, zero, x, chroma, chroma, x])
    return np.stack([r + offset, g + offset, b + offset], axis=-1)


def _blend_hsl(mode, base, src):
    """Hue/saturation/colour/luminosity: take channels from each side."""
    base_h, base_s, base_l = _rgb_to_hsl(base)
    src_h, src_s, src_l = _rgb_to_hsl(src)
    if mode == "hue":
        return _hsl_to_rgb(src_h, base_s, base_l)
    if mode == "saturation":
        return _hsl_to_rgb(base_h, src_s, base_l)
    if mode == "color":
        return _hsl_to_rgb(src_h, src_s, base_l)
    return _hsl_to_rgb(base_h, base_s, src_l)  # luminosity


def _luminance(arr):
    """Per-pixel Rec.709 luminance of an ``HxWx3`` array in ``0..1``."""
    return (0.2126 * arr[..., 0] + 0.7152 * arr[..., 1]
            + 0.0722 * arr[..., 2])


def _blend_x(mode, base, src):
    """The numpy blends (values in ``0..1``) not covered by ``_BLEND_OPS``."""
    if mode == "color_dodge":
        return base / np.maximum(1e-6, 1.0 - src)
    if mode == "color_burn":
        return 1.0 - (1.0 - base) / np.maximum(1e-6, src)
    if mode == "linear_burn":
        return np.clip(base + src - 1.0, 0.0, 1.0)
    if mode == "linear_light":
        return np.clip(base + 2.0 * src - 1.0, 0.0, 1.0)
    if mode == "vivid_light":
        burn = 1.0 - (1.0 - base) / np.maximum(1e-6, 2.0 * src)
        dodge = base / np.maximum(1e-6, 2.0 - 2.0 * src)
        return np.clip(np.where(src <= 0.5, burn, dodge), 0.0, 1.0)
    if mode == "pin_light":
        dark = np.minimum(base, 2.0 * src)
        light = np.maximum(base, 2.0 * (src - 0.5))
        return np.clip(np.where(src <= 0.5, dark, light), 0.0, 1.0)
    if mode == "hard_mix":
        return np.where(base + src >= 1.0, 1.0, 0.0)
    if mode == "divide":
        return np.clip(base / np.maximum(1e-6, src), 0.0, 1.0)
    if mode in ("darker_color", "lighter_color"):
        # Whole-pixel decision: take all three channels from whichever pixel is
        # darker/lighter by luminance (Photoshop's Darker/Lighter Color).
        base_lum = _luminance(base)
        src_lum = _luminance(src)
        if mode == "darker_color":
            choose_base = base_lum <= src_lum
        else:
            choose_base = base_lum >= src_lum
        return np.where(choose_base[..., None], base, src)
    if mode == "exclusion":
        return base + src - 2.0 * base * src
    return _blend_hsl(mode, base, src)


def _blend(mode, base_rgb, src_rgb):
    """Blend two RGB images with ``mode``; ``normal`` returns the source."""
    if mode == "normal":
        return src_rgb
    op = _BLEND_OPS.get(mode)
    if op is not None:
        return op(base_rgb, src_rgb)
    return _array_image(
        _blend_x(mode, _rgb_array(base_rgb), _rgb_array(src_rgb)))


def composite_color(base, color, coverage, mode="normal"):
    """Paint ``color`` over ``base`` (RGBA) using ``coverage`` and a blend mode."""
    solid = Image.new("RGB", base.size, tuple(color))
    if mode == "normal":
        blended = solid
    else:
        blended = _blend(mode, base.convert("RGB"), solid)
        # Bare pixels have no colour to blend against, so blend modes fall back
        # to the raw paint colour wherever the layer is transparent.
        blended = Image.composite(blended, solid, base.getchannel("A"))
    result = base.copy()
    result.paste(blended, (0, 0), coverage)
    return result


def erase_alpha(base, coverage):
    """Return ``base`` with its alpha reduced by ``coverage``."""
    result = base.copy()
    inverse = ImageChops.invert(coverage)
    result.putalpha(ImageChops.multiply(base.getchannel("A"), inverse))
    return result


def paint_mask(base_mask, value, coverage):
    """Interpolate ``base_mask`` toward the flat ``value`` by ``coverage``."""
    solid = Image.new("L", base_mask.size, max(0, min(255, int(value))))
    result = base_mask.copy()
    result.paste(solid, (0, 0), coverage)
    return result


def _rejoin_rgba(rgb, alpha):
    """Combine an RGB result with an alpha channel into an RGBA image."""
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


def composite_image(base, overlay_rgb, coverage, mode="normal"):
    """Paint a full-colour ``overlay`` over ``base`` (RGBA) via ``coverage``.

    Like :func:`composite_color` but the painted pixels come from an image
    (used by the gradient and clone tools) instead of a single flat colour.
    ``overlay`` may be RGB (stamped fully opaque) or RGBA, in which case its
    own alpha is blended too, so a transparent part of the overlay clears the
    base rather than painting over it.
    """
    if mode == "normal":
        blended = overlay_rgb
    else:
        overlay = overlay_rgb.convert("RGB")
        blended = _blend(mode, base.convert("RGB"), overlay)
        blended = Image.composite(blended, overlay, base.getchannel("A"))
    result = base.copy()
    result.paste(blended, (0, 0), coverage)
    return result


# Tonal ranges for the dodge and burn brushes. Each weights the effect by how
# dark or light a pixel already is, so a range targets one band of the tonal
# scale instead of lightening or darkening every pixel evenly.
TONAL_RANGES = {
    "shadows": "Shadows",
    "midtones": "Midtones",
    "highlights": "Highlights",
}

# Tonal protection eases the effect off over the top quarter of the tonal scale
# as a pixel approaches the extreme it is pushed toward, so dodged highlights
# and burned shadows keep detail instead of clipping to flat white or black.
_PROTECT_SPAN = 0.25
_PROTECT_STRENGTH = 0.75


def _range_weight(tone, range_):
    """Fraction of the effect applied to a pixel whose tone is ``tone`` (0..1)."""
    if range_ == "shadows":
        return 1.0 - tone
    if range_ == "highlights":
        return tone
    # Midtones peak at half tone and taper toward pure black and white.
    return 4.0 * tone * (1.0 - tone)


def _tone_lut(range_, protect, lighten):
    """Build a 256-entry LUT weighting a tone by range and tone protection."""
    lut = []
    for level in range(256):
        tone = level / 255.0
        weight = _range_weight(tone, range_)
        if protect:
            if lighten:
                edge = max(0.0, (tone - (1.0 - _PROTECT_SPAN)) / _PROTECT_SPAN)
            else:
                edge = max(0.0, (_PROTECT_SPAN - tone) / _PROTECT_SPAN)
            weight *= 1.0 - _PROTECT_STRENGTH * min(1.0, edge)
        lut.append(int(round(255.0 * max(0.0, min(1.0, weight)))))
    return lut


def _tone_mask(rgb, coverage, range_, exposure, protect, lighten):
    """Scale ``coverage`` by the tone range, tone protection and exposure."""
    weight = rgb.convert("L").point(_tone_lut(range_, protect, lighten))
    mask = ImageChops.multiply(coverage, weight)
    if exposure < 100:
        mask = mask.point(lambda value: int(value * exposure / 100.0))
    return mask.convert("RGB")


def dodge(base, coverage, *, range_="midtones", exposure=50, protect=True):
    """Lighten ``base`` (RGBA) toward white by ``coverage``.

    ``range_`` picks the tonal band to lighten, ``exposure`` scales the effect
    and ``protect`` eases it off near white so highlights do not clip.
    """
    rgb = base.convert("RGB")
    mask = _tone_mask(rgb, coverage, range_, exposure, protect, True)
    lifted = ImageChops.add(rgb, ImageChops.multiply(ImageChops.invert(rgb),
                                                     mask), scale=1.0)
    return _rejoin_rgba(lifted, base.getchannel("A"))


def burn(base, coverage, *, range_="midtones", exposure=50, protect=True):
    """Darken ``base`` (RGBA) toward black by ``coverage``.

    Mirrors :func:`dodge`: ``range_`` picks the tonal band to darken,
    ``exposure`` scales the effect and ``protect`` eases it off near black so
    shadows keep detail.
    """
    rgb = base.convert("RGB")
    mask = _tone_mask(rgb, coverage, range_, exposure, protect, False)
    darkened = ImageChops.multiply(rgb, ImageChops.invert(mask))
    return _rejoin_rgba(darkened, base.getchannel("A"))


class Stroke:
    """One continuous dab track, rebuilt onto the layer from a fixed snapshot.

    ``kind`` selects the target and how coverage is applied:

    ``"brush"``            paint ``color`` over the layer pixels.
    ``"eraser"``           reduce layer alpha to transparency.
    ``"eraser-color"``     paint ``color`` over the layer pixels (fill eraser).
    ``"mask"``             push the layer mask toward ``paint_value``.
    ``"clone"``            paint pixels sampled from ``source`` (see below).
    ``"dodge"``            lighten the layer pixels toward white.
    ``"burn"``             darken the layer pixels toward black.
    ``"smudge"``           drag existing colours along the stroke.

    ``clone`` samples ``source`` (an RGBA image in the layer's own pixel space)
    offset by ``source_offset`` from the dab position. ``smudge`` keeps its own
    running buffer so each dab carries the colour it just picked up. ``dodge``
    and ``burn`` read ``tone_range``, ``tone_exposure`` and ``tone_protect`` to
    confine the effect to a tonal band. ``pattern`` (see :data:`EFFECT_PATTERNS`)
    swaps the round/square tip for a directional effect-line stamp. An optional
    ``clip`` mask (layer-sized ``"L"``) confines the whole stroke to the active
    selection. A ``dynamics`` (see :class:`Dynamics`) modulates size, opacity,
    flow, hardness and spacing per dab by pointer speed and tapers the ends;
    ``None`` keeps every dab identical.
    """

    def __init__(self, layer, kind, *, size, hardness, flow, opacity,
                 spacing, antialias, tip, pattern="none", color=(0, 0, 0),
                 blend="normal", paint_value=255, source=None,
                 source_offset=(0, 0), tone_range="midtones",
                 tone_exposure=50, tone_protect=True, clip=None,
                 dynamics=None, grain="none", grain_scale=1.0,
                 grain_strength=0.0, grain_mode="canvas", jitter_hue=0.0,
                 jitter_sat=0.0, jitter_val=0.0, pickup=0.0, jitter_seed=None,
                 wet_edge=0.0):
        self.layer = layer
        self.kind = kind
        # Optional layer-sized ``"L"`` mask confining the stroke to the active
        # selection, so a brush (or eraser, clone, ...) cannot paint outside it.
        self.clip = clip
        self.size = max(1, int(round(size)))
        self.color = tuple(color)
        self.blend = blend
        self.paint_value = paint_value
        self.opacity = max(1, min(100, int(round(opacity))))
        self.cap = int(round(255 * self.opacity / 100.0))
        self._tip = tip if tip in TIP_SHAPES else "round"
        self.dab = make_dab(self.size, hardness, antialias, self._tip)
        # Velocity-driven per-dab modulation and stroke taper (see ``Dynamics``).
        # A default ``Dynamics`` is inactive, so the stroke behaves exactly as
        # before and the fast path keeps using the single cached dab.
        self.dynamics = dynamics if dynamics is not None else Dynamics()
        self._fixed = not self.dynamics.active()
        self._speed = 0.0
        self._arc = 0.0
        # Distance travelled since the last dab, carried between move_to calls
        # so dabs stay one ``step`` apart regardless of event granularity.
        self._pending = 0.0
        self._trail = []
        self._base_spacing = max(1, int(spacing))
        # Dry-media grain: a texture multiplied into each dab's alpha. ``none``
        # or zero strength leaves the stroke exactly as before.
        self.grain = grain if grain in GRAIN_TYPES else "none"
        self.grain_scale = max(0.25, min(4.0, float(grain_scale)))
        self.grain_strength = _clamp01(grain_strength)
        self.grain_mode = grain_mode if grain_mode in GRAIN_MODES else "canvas"
        self._grain_on = self.grain != "none" and self.grain_strength > 0.0
        self._grain_tile = (grain_tile(self.grain, self.grain_strength,
                                       self.grain_scale) if self._grain_on
                            else None)
        self._grain_stamp_cache = {}
        # Colour dynamics: per-dab hue/saturation/value jitter and paint pickup
        # (mixing the colour under the brush into what is deposited). Active only
        # for the tools that paint colour; zero strengths leave the stroke as a
        # single flat colour via the cheaper coverage path.
        self.jitter_hue = _clamp01(jitter_hue)
        self.jitter_sat = _clamp01(jitter_sat)
        self.jitter_val = _clamp01(jitter_val)
        self.pickup = _clamp01(pickup)
        self._jitter_any = bool(self.jitter_hue or self.jitter_sat
                                or self.jitter_val)
        self._color_active = (self._jitter_any or self.pickup > 0.0) and \
            kind in ("brush", "eraser-color")
        self._color_map = None
        self._rng = random.Random(0xC0FFEE if jitter_seed is None
                                  else jitter_seed)
        # Wet edge: boost a dab's rim so pigment piles up at the boundary of a
        # stroke, the way a watercolour wash dries with a darker edge.
        self.wet_edge = _clamp01(wet_edge)
        self._wet_cache = {}
        # Effect-line patterns replace the tip with a directional stamp whose
        # orientation follows the stroke (see ``make_effect_dab``).
        self.pattern = pattern if pattern in EFFECT_PATTERNS else "none"
        self._hardness = hardness
        self._antialias = bool(antialias)
        self._origin = None
        self._angle = 0.0
        self.step = max(1.0, self.size * max(1, spacing) / 100.0)
        self.amount = max(1, min(100, int(round(flow)))) / 100.0
        self.coverage = Image.new("L", layer.size, 0)
        self.source = source
        self.source_offset = (int(source_offset[0]), int(source_offset[1]))
        self.tone_range = tone_range if tone_range in TONAL_RANGES else "midtones"
        self.tone_exposure = max(1, min(100, int(round(tone_exposure))))
        self.tone_protect = bool(tone_protect)
        if kind == "mask":
            if layer.mask is None:
                layer.mask = Image.new("L", layer.size, 255)
            self.base = layer.mask.copy()
        else:
            self.base = layer.image.copy()
        # Smudge pulls colour from its own buffer, which starts as a copy of the
        # layer and is updated in place as the stroke drags colour along.
        self._smudge_buf = layer.image.copy() if kind == "smudge" else None
        self.last = None
        # Union of the rectangles stamped since the last commit, so commit can
        # rebuild just that region. ``None`` means nothing changed.
        self._dirty = None
        # The rectangles the stroke has touched since the live preview last read
        # them, in the layer's own pixel space. The view uses it to recomposite
        # only the new dabs instead of the whole viewport (see
        # ``CanvasView._live_composite``); reading clears it so a long stroke does
        # not grow the patch back to its whole bounding box.
        self._render_dirty = None

    def begin(self, x, y):
        self.last = (x, y)
        self._origin = (x, y)
        self._arc = 0.0
        self._pending = 0.0
        if self.dynamics.taper_out > 0:
            self._trail = [(x, y, 0.0)]
        if self.kind == "smudge":
            self._smudge_dab(x, y)
            return
        self._stamp(x, y)
        self.commit()

    def _step_for(self, distance, speed=None):
        """Dab spacing for this segment, widened as the pointer speeds up."""
        if self._fixed:
            return self.step
        self._update_speed(distance, speed)
        size, _hardness, _amount, _cap = self._dab_params(self._arc)
        spacing = self._base_spacing * (1.0 + self.dynamics.spacing * 2.0
                                        * self._speed)
        return max(1.0, size * max(1.0, spacing) / 100.0)

    def move_to(self, x, y, speed=None):
        if self.last is None:
            self.begin(x, y)
            return
        x0, y0 = self.last
        dx = x - x0
        dy = y - y0
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return
        step = self._step_for(distance, speed)
        if self.kind == "smudge":
            # Each dab picks colour up one sub-step behind the cursor, so the
            # colour trails and is dragged forward along the stroke instead of
            # being re-stamped in place (which would leave the pixels as-is).
            steps = max(1, int(distance / step))
            lag_x = dx / steps
            lag_y = dy / steps
            for i in range(1, steps + 1):
                t = i / steps
                x_i = x0 + dx * t
                y_i = y0 + dy * t
                self._smudge_dab(x_i, y_i, (x_i - lag_x, y_i - lag_y))
            self.last = (x, y)
            return
        # Stamp a dab every ``step`` pixels of travel, carrying the distance
        # since the previous dab across segments. Without the carry a pointer
        # that moves less than one step between events (the usual case) stamps
        # a dab per event, so the spacing collapses to the event distance - a
        # fixed number of pixels - and a large soft brush packs into a hard,
        # solid band. Tying dabs to distance instead keeps the spacing a fixed
        # fraction of the brush size however the pointer is sampled.
        total = self._pending + distance
        dabs = int(total / step)
        if dabs <= 0:
            self._pending = total
            self._arc += distance
            self.last = (x, y)
            return
        self._pending = total - dabs * step
        start_arc = self._arc
        taper = self.dynamics.taper_out
        # The first dab lands one ``step`` after the previous one, i.e.
        # ``step - before`` into this segment; the rest follow at ``step``.
        before = total - distance
        first = step - before
        for i in range(dabs):
            travelled = first + i * step
            t = travelled / distance
            px = x0 + dx * t
            py = y0 + dy * t
            self._arc = start_arc + travelled
            self._stamp(px, py, self._stamp_angle(px, py, dx, dy))
            if taper > 0:
                self._trail.append((px, py, self._arc))
        self._arc = start_arc + distance
        if taper > 0:
            self._trim_trail()
        self.last = (x, y)
        self.commit()

    def finish(self):
        """Fade the trailing end of the stroke out, then leave it committed.

        The end of a stroke is only known when it ends, so the taper is applied
        here: the coverage of the final ``taper`` pixels is multiplied down along
        the recorded trail and that region is recomposited from the snapshot, so
        the tip fades instead of stopping flat. Smudge writes straight to the
        layer and keeps no coverage, so it is left untouched.
        """
        taper = self.dynamics.taper_out
        if taper <= 0 or self.kind == "smudge" or self._arc <= 0:
            return
        trailing = [point for point in self._trail
                    if point[2] > self._arc - taper]
        if len(trailing) < 2:
            return
        pad = self.size / 2.0 + 2.0
        xs = [point[0] for point in trailing]
        ys = [point[1] for point in trailing]
        x0 = max(0, int(math.floor(min(xs) - pad)))
        y0 = max(0, int(math.floor(min(ys) - pad)))
        x1 = min(self.coverage.width, int(math.ceil(max(xs) + pad)))
        y1 = min(self.coverage.height, int(math.ceil(max(ys) + pad)))
        if x1 <= x0 or y1 <= y0:
            return
        size = (x1 - x0, y1 - y0)
        radius = max(1.0, self.size / 2.0)
        # The mask starts fully opaque (no change) and each trailing dab stamps
        # its own fade value; overlapping dabs take the darkest (most tapered).
        mask = Image.new("L", size, 255)
        for px, py, arc in trailing:
            # Distance from the tip: 0 at the end (fully faded), ``taper`` where
            # the fade begins (untouched).
            fade = (self._arc - arc) / taper
            value = int(round(255 * _clamp01(fade)))
            disc = Image.new("L", size, 255)
            ImageDraw.Draw(disc).ellipse(
                (px - x0 - radius, py - y0 - radius,
                 px - x0 + radius, py - y0 + radius),
                fill=value,
            )
            mask = ImageChops.darker(mask, disc)
        region = ImageChops.multiply(
            self.coverage.crop((x0, y0, x1, y1)), mask)
        self.coverage.paste(region, (x0, y0))
        self._commit_rect((x0, y0, x1, y1))

    def _trim_trail(self):
        """Drop trail points older than the taper window (they cannot taper)."""
        cutoff = self._arc - self.dynamics.taper_out
        if cutoff <= 0:
            return
        self._trail = [point for point in self._trail if point[2] >= cutoff]

    def _update_speed(self, distance, speed=None):
        """Fold this segment's speed into the smoothed ``0..1`` pointer speed."""
        n = distance / self.dynamics.speed_ref if speed is None else speed
        n = _clamp01(n)
        self._speed += (n - self._speed) * self.dynamics.smoothing

    def _dab_params(self, arc):
        """The ``(size, hardness, amount, cap)`` for a dab at arc length ``arc``."""
        d = self.dynamics
        n = self._speed
        size = self.size * (1.0 - d.size * (1.0 - d.min_scale) * n)
        hardness = self._hardness * (1.0 - d.hardness * n)
        amount = self.amount * (1.0 - d.flow * n)
        cap = self.cap * (1.0 - d.opacity * n)
        if d.taper_in > 0:
            fade = arc / d.taper_in
            if fade < 0.15:
                fade = 0.15
            elif fade > 1.0:
                fade = 1.0
            size *= fade
            amount *= fade
            cap *= fade
        cap = int(round(max(1.0, min(255.0, cap))))
        return max(1.0, size), hardness, amount, cap

    def _dab_for(self, size, hardness, angle):
        """Rasterise (or fetch cached) the dab for the current per-dab params."""
        if self.pattern != "none":
            return make_effect_dab(size, self.pattern, angle, hardness,
                                   self._antialias)
        return make_dab(size, hardness, self._antialias, self._tip)

    def _textured(self, dab, cx, cy):
        """Multiply ``dab`` by its **stamp** grain, or return it unchanged.

        Stamp grain is anchored to the dab, so its textured stamp is cached per
        dab size. Canvas grain is not applied here: it is a property of the
        surface rather than the dab, so it is multiplied into the stroke's
        coverage once per commit instead (see :meth:`_grain_coverage`), which
        removes a tiling pass from every dab.
        """
        if not self._grain_on or self.grain_mode != "stamp":
            return dab
        dw, dh = dab.size
        patch = self._grain_stamp_cache.get((dw, dh))
        if patch is None:
            patch = _grain_region(self._grain_tile, 0, 0, dw, dh)
            self._grain_stamp_cache[(dw, dh)] = patch
        return ImageChops.multiply(dab, patch)

    def _wet(self, dab, hardness):
        """Return ``dab`` with its rim boosted for the wet-edge look.

        The rim is the band between the dab and a slightly eroded copy of it;
        adding it back at the wet-edge strength makes the stroke's boundary
        darker than its middle, as wet media dries. Cached per dab size and
        hardness, since that is what the stamp depends on.
        """
        if self.wet_edge <= 0.0:
            return dab
        key = (dab.size, int(round(hardness)))
        cached = self._wet_cache.get(key)
        if cached is None:
            radius = max(1, min(3, dab.size[0] // 24))
            inner = dab.filter(ImageFilter.MinFilter(radius * 2 + 1))
            ring = ImageChops.subtract(dab, inner)
            ring = ring.point(lambda v: int(v * self.wet_edge))
            cached = ImageChops.add(dab, ring)
            self._wet_cache[key] = cached
        return cached

    def _grain_coverage(self, cov, rect):
        """Lay the canvas grain over a coverage rectangle at document position."""
        width = rect[2] - rect[0]
        height = rect[3] - rect[1]
        if width * height >= _GPU_GRAIN_MIN:
            from . import gpuops
            overlaid = gpuops.grain_overlay(cov, self._grain_tile,
                                            rect[0], rect[1])
            if overlaid is not None:
                return overlaid
        return ImageChops.multiply(
            cov, _grain_region(self._grain_tile, rect[0], rect[1],
                               width, height))

    def _sample_color(self, cx, cy):
        """The layer's colour at ``(cx, cy)``, or ``None`` where it is empty."""
        image = self.layer.image
        if not (0 <= cx < image.width and 0 <= cy < image.height):
            return None
        pixel = image.getpixel((cx, cy))
        if pixel[3] < 8:
            return None
        return (pixel[0], pixel[1], pixel[2])

    def _dab_color(self, cx, cy):
        """The colour for the dab at ``(cx, cy)``: base, mixed and jittered."""
        r, g, b = self.color
        if self.pickup > 0.0:
            sampled = self._sample_color(cx, cy)
            if sampled is not None:
                mix = self.pickup
                r += (sampled[0] - r) * mix
                g += (sampled[1] - g) * mix
                b += (sampled[2] - b) * mix
        if self._jitter_any:
            h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
            h = (h + self._rng.uniform(-1.0, 1.0) * self.jitter_hue * 0.5) % 1.0
            s = _clamp01(s + self._rng.uniform(-1.0, 1.0) * self.jitter_sat)
            v = _clamp01(v + self._rng.uniform(-1.0, 1.0) * self.jitter_val)
            r, g, b = colorsys.hsv_to_rgb(h, s, v)
            r, g, b = r * 255.0, g * 255.0, b * 255.0
        return (int(round(r)), int(round(g)), int(round(b)))

    def _paint_color(self, dab, cx, cy, amount, color):
        """Blend ``color`` into the colour map through ``dab`` (scaled by amount).

        Each dab is weighted by its own coverage *relative to the coverage
        already laid down*, so overlapping dabs average their colours instead of
        the last dab overwriting the ones below (which banded a jittered or
        picked-up stroke). The stroke's ``coverage`` at this point holds the
        earlier dabs, so it supplies the running weight.
        """
        dw, dh = dab.size
        left = cx - dw // 2
        top = cy - dh // 2
        sx0 = max(0, -left)
        sy0 = max(0, -top)
        dx0 = max(0, left)
        dy0 = max(0, top)
        w = min(dw - sx0, self._color_map.width - dx0)
        h = min(dh - sy0, self._color_map.height - dy0)
        if w <= 0 or h <= 0:
            return
        mask = np.asarray(dab.crop((sx0, sy0, sx0 + w, sy0 + h)),
                          dtype=np.float32) / 255.0
        if amount < 1.0:
            mask = mask * amount
        coverage = np.asarray(
            self.coverage.crop((dx0, dy0, dx0 + w, dy0 + h)),
            dtype=np.float32) / 255.0
        denom = coverage + mask
        fraction = np.divide(mask, denom, out=np.zeros_like(mask),
                             where=denom > 1e-6)[..., None]
        old = np.asarray(self._color_map.crop((dx0, dy0, dx0 + w, dy0 + h)),
                         dtype=np.float32)
        target = np.asarray(color, dtype=np.float32)[None, None, :]
        blended = old * (1.0 - fraction) + target * fraction
        self._color_map.paste(
            Image.fromarray(
                np.clip(blended + 0.5, 0.0, 255.0).astype(np.uint8), "RGB"),
            (dx0, dy0),
        )

    def _stamp_angle(self, x, y, dx, dy):
        """Orientation (degrees) for the effect dab stamped at ``(x, y)``.

        ``focus`` points each dab at the stroke's origin so the lines converge
        there; the other patterns follow the stroke direction. ``flash`` uses a
        fixed angle because its rays are already radial.
        """
        if self.pattern == "focus" and self._origin is not None:
            vx = self._origin[0] - x
            vy = self._origin[1] - y
            if math.hypot(vx, vy) > 1e-6:
                return math.degrees(math.atan2(vy, vx))
            return self._angle
        if self.pattern == "flash":
            return self._angle
        if math.hypot(dx, dy) > 1e-6:
            self._angle = math.degrees(math.atan2(dy, dx))
        return self._angle

    def _smudge_dab(self, x, y, sample=None):
        """Drag the running buffer's colour to ``(x, y)`` and carry it on.

        The colour deposited at ``(x, y)`` is picked up from ``sample`` (in
        layer pixels), which defaults to the dab itself. Sampling a point
        behind the dab is what moves colour along the stroke; the buffer keeps
        the picked-up pixels, so the next dab drags them further forward.
        """
        size = self.size
        half = size // 2
        cx = int(round(x)) - half
        cy = int(round(y)) - half
        if sample is None:
            sx, sy = cx, cy
        else:
            sx = int(round(sample[0])) - half
            sy = int(round(sample[1])) - half
        buf = self._smudge_buf
        patch = buf.crop((sx, sy, sx + size, sy + size))
        alpha = ImageChops.multiply(patch.getchannel("A"), self.dab)
        if self.amount < 1.0:
            alpha = alpha.point(lambda v: int(v * self.amount))
        if self.clip is not None:
            alpha = ImageChops.multiply(
                alpha, self.clip.crop((cx, cy, cx + size, cy + size))
            )
        patch = patch.copy()
        patch.putalpha(alpha)
        self.layer.begin_write().alpha_composite(patch, (cx, cy))
        # The picked-up colour is what the next dab drags forward.
        buf.alpha_composite(patch, (cx, cy))
        self._note_dirty((cx, cy, cx + size, cy + size))
        self.layer.invalidate_filters()

    def _stamp(self, x, y, angle=0.0):
        cx = int(round(x))
        cy = int(round(y))
        if self._fixed:
            amount = self.amount
            cap = self.cap
            hardness = self._hardness
            # A fixed stroke still needs its directional stamp rebuilt per dab
            # (the angle changes); a plain tip reuses the cached round/square dab.
            if self.pattern != "none":
                dab = make_effect_dab(self.size, self.pattern, angle,
                                      self._hardness, self._antialias)
            else:
                dab = self.dab
        else:
            size, hardness, amount, cap = self._dab_params(self._arc)
            dab = self._dab_for(size, hardness, angle)
        if self._grain_on and self.grain_mode == "stamp":
            dab = self._textured(dab, cx, cy)
        if self.wet_edge > 0.0 and self.pattern == "none":
            dab = self._wet(dab, hardness)
        if self._color_active:
            if self._color_map is None:
                self._color_map = Image.new("RGB", self.coverage.size,
                                            tuple(self.color))
            self._paint_color(dab, cx, cy, amount, self._dab_color(cx, cy))
        rect = _accumulate(self.coverage, dab, cx, cy, amount, cap)
        if rect is None:
            return
        if self._dirty is None:
            self._dirty = rect
        else:
            dx0, dy0, dx1, dy1 = self._dirty
            rx0, ry0, rx1, ry1 = rect
            self._dirty = (min(dx0, rx0), min(dy0, ry0),
                           max(dx1, rx1), max(dy1, ry1))

    def commit(self):
        """Rebuild the changed region of the layer from the stroke snapshot.

        Only the rectangle touched by the dabs since the previous commit is
        recomposited, so a stroke on a large layer costs the size of the dabs
        rather than the size of the layer. Coverage outside the dirty box is
        unchanged, so the rest of the target already holds the correct pixels.
        """
        rect = self._dirty
        self._dirty = None
        if rect is None:
            return
        self._commit_rect(rect)

    def _note_dirty(self, rect):
        """Union ``rect`` (layer pixels) into the pending render dirty box."""
        self._render_dirty = (rect if self._render_dirty is None
                              else _union_rect(self._render_dirty, rect))

    def take_render_dirty(self):
        """The rectangle touched since the last read, then clear it."""
        rect = self._render_dirty
        self._render_dirty = None
        return rect

    def _commit_rect(self, rect):
        """Recomposite ``rect`` from the stroke snapshot through the coverage."""
        self._note_dirty(rect)
        cov = self.coverage.crop(rect)
        if self._grain_on and self.grain_mode == "canvas":
            cov = self._grain_coverage(cov, rect)
        if self.clip is not None:
            # Intersect the stroke's coverage with the selection so nothing
            # outside it is touched, whatever the stroke kind.
            cov = ImageChops.multiply(cov, self.clip.crop(rect))
        if self.kind == "mask":
            region = paint_mask(
                self.base.crop(rect), self.paint_value, cov
            )
            self.layer.begin_mask_write().paste(region, (rect[0], rect[1]))
        elif self.kind == "eraser":
            region = erase_alpha(self.base.crop(rect), cov)
            self.layer.begin_write().paste(region, (rect[0], rect[1]))
        elif self.kind == "clone":
            width = rect[2] - rect[0]
            height = rect[3] - rect[1]
            sx = rect[0] - self.source_offset[0]
            sy = rect[1] - self.source_offset[1]
            # Keep the source's alpha: compositing an RGBA overlay through the
            # coverage blends all four channels, so cloning a transparent
            # region clears the destination instead of stamping opaque black.
            sampled = self.source.crop((sx, sy, sx + width, sy + height))
            region = composite_image(
                self.base.crop(rect), sampled, cov, "normal"
            )
            self.layer.begin_write().paste(region, (rect[0], rect[1]))
        elif self.kind in ("dodge", "burn"):
            tone = dodge if self.kind == "dodge" else burn
            region = tone(
                self.base.crop(rect), cov,
                range_=self.tone_range, exposure=self.tone_exposure,
                protect=self.tone_protect,
            )
            self.layer.begin_write().paste(region, (rect[0], rect[1]))
        elif self._color_map is not None:
            # Per-dab colours (jitter / pickup) were accumulated into a colour
            # map; the coverage still supplies the alpha mask.
            region = composite_image(
                self.base.crop(rect), self._color_map.crop(rect), cov,
                self.blend,
            )
            self.layer.begin_write().paste(region, (rect[0], rect[1]))
        else:
            region = composite_color(
                self.base.crop(rect), self.color, cov,
                self.blend,
            )
            self.layer.begin_write().paste(region, (rect[0], rect[1]))
        self.layer.invalidate_filters()
