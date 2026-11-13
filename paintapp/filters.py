"""Non-destructive image filters applied to layers at composite time.

A filter is a plain, serializable spec stored on a layer::

    {"type": "gaussian_blur", "enabled": True, "params": {"radius": 4.0}}

The layer's base pixels are never modified. ``document.Layer.filtered_image``
runs the enabled filters, in order, over a copy of the base image and the
compositor uses that result, so filters stay editable and reversible.

Most filters here are PIL-only. A few operations (vibrance, colour balance,
selective colour, gradient map, the manga halftone and screentone screens) use
numpy for the per-pixel maths, which is already a hard dependency of the app's
engine. Each filter keeps the layer's alpha channel, except the blurs, which
blur alpha too so softened edges stay soft.
"""

import copy
import math

import numpy as np

from PIL import (
    Image,
    ImageChops,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)

from .constants import rgba
from . import gpuimage

# id -> {"label", "category", "params", "summary", "apply"}
#
# A param is ``{"key", "label", "min", "max", "default", "step", "kind"}`` where
# ``kind`` is one of ``"float"`` (default), ``"int"``, ``"bool"``, ``"curve"``,
# ``"color"`` or ``"choice"`` (a fixed set of labelled options in ``"options"``).
FILTERS = {}


def _register(fid, category, label, params, apply_fn, summary):
    FILTERS[fid] = {
        "label": label,
        "category": category,
        "params": params,
        "apply": apply_fn,
        "summary": summary,
    }


def _p(key, label, low, high, default, step=1, kind="float"):
    return {
        "key": key, "label": label, "min": low, "max": high,
        "default": default, "step": step, "kind": kind,
    }


def _c(key, label, default="#000000"):
    """A param descriptor for a colour picker (a ``#rrggbb`` string)."""
    return {"key": key, "label": label, "default": default, "kind": "color"}


def _choice(key, label, options, default):
    """A param descriptor for a fixed set of labelled options.

    ``options`` is a list of ``(value, label)`` pairs: ``value`` is the string
    stored in the spec and ``label`` what the editor shows. The default must be
    one of the values.
    """
    return {
        "key": key, "label": label, "default": default,
        "kind": "choice", "options": list(options),
    }


def _rgb_parts(image):
    """Split ``image`` into an RGB image and its alpha (or ``None``)."""
    if image.mode == "RGBA":
        return image.convert("RGB"), image.getchannel("A")
    return image.convert("RGB"), None


def _rejoin(rgb, alpha):
    """Combine an RGB result with the original alpha channel."""
    if alpha is None:
        return rgb
    out = rgb.convert("RGBA")
    out.putalpha(alpha)
    return out


# ----------------------------------------------------------------- adjustments
def _brightness(image, params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(ImageEnhance.Brightness(rgb).enhance(params["factor"]), alpha)


def _contrast(image, params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(ImageEnhance.Contrast(rgb).enhance(params["factor"]), alpha)


def _saturation(image, params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(ImageEnhance.Color(rgb).enhance(params["factor"]), alpha)


def _hue(image, params):
    rgb, alpha = _rgb_parts(image)
    shift = int(round(params["degrees"])) % 360
    if shift:
        hsv = rgb.convert("HSV")
        h, s, v = hsv.split()
        offset = int(round(shift / 360.0 * 256)) % 256
        h = h.point(lambda value, off=offset: (value + off) % 256)
        rgb = Image.merge("HSV", (h, s, v)).convert("RGB")
    return _rejoin(rgb, alpha)


def _grayscale(image, params):
    amount = params["amount"] / 100.0
    if amount <= 0:
        return image
    rgb, alpha = _rgb_parts(image)
    gray = ImageOps.grayscale(rgb).convert("RGB")
    return _rejoin(Image.blend(rgb, gray, min(1.0, amount)), alpha)


def _invert(image, params):
    amount = params["amount"] / 100.0
    if amount <= 0:
        return image
    rgb, alpha = _rgb_parts(image)
    inverted = ImageOps.invert(rgb)
    return _rejoin(Image.blend(rgb, inverted, min(1.0, amount)), alpha)


def _posterize(image, params):
    bits = max(1, min(8, int(params["bits"])))
    rgb, alpha = _rgb_parts(image)
    return _rejoin(ImageOps.posterize(rgb, bits), alpha)


def _threshold(image, params):
    level = int(params["level"])
    rgb, alpha = _rgb_parts(image)
    gray = ImageOps.grayscale(rgb)
    out = gray.point(lambda value: 255 if value > level else 0).convert("RGB")
    return _rejoin(out, alpha)


def _clamp01(value):
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def _levels_lut(params):
    """A 256-entry tone table mapping input to output for a Levels adjustment.

    Built from the classic five controls: the input black/white points that
    stretch the range, a midtone gamma, and the output black/white points that
    compress the result. Values are clamped so a degenerate range (white <=
    black) still produces a usable table.
    """
    in_black = float(params["in_black"])
    in_white = float(params["in_white"])
    gamma = max(0.01, float(params["gamma"]))
    out_black = float(params["out_black"])
    out_white = float(params["out_white"])
    span = in_white - in_black
    if span <= 0:
        span = 1.0
    inverse = 1.0 / gamma
    lut = []
    for value in range(256):
        t = _clamp01((value - in_black) / span)
        if inverse != 1.0:
            t = t ** inverse
        out = out_black + t * (out_white - out_black)
        lut.append(int(round(max(0.0, min(255.0, out)))))
    return lut


def _spline_lut(points):
    """A 256-entry tone table through monotone-x control points in ``0..1``.

    ``points`` is a non-decreasing-``x`` list of ``(x, y)`` floats spanning
    ``0..1``. The curve is a cubic Hermite spline whose tangents follow
    Fritsch-Carlson, so it never overshoots between points; a segment that
    reverses direction gets a flat tangent, which keeps free-form curves
    (where a user may drag one point below another) well behaved too.
    """
    if len(points) < 2:
        level = int(round(_clamp01(points[0][1]) * 255)) if points else 0
        return [level] * 256
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    count = len(points)
    slopes = [
        (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i]) for i in range(count - 1)
    ]
    tangents = [0.0] * count
    tangents[0] = slopes[0]
    tangents[-1] = slopes[-1]
    for i in range(1, count - 1):
        if slopes[i - 1] * slopes[i] <= 0:
            tangents[i] = 0.0
        else:
            tangents[i] = (slopes[i - 1] + slopes[i]) / 2.0
    for i in range(count - 1):
        if slopes[i] == 0.0:
            tangents[i] = tangents[i + 1] = 0.0
        else:
            a = tangents[i] / slopes[i]
            b = tangents[i + 1] / slopes[i]
            magnitude = a * a + b * b
            if magnitude > 9.0:
                scale = 3.0 / (magnitude ** 0.5)
                tangents[i] = scale * a * slopes[i]
                tangents[i + 1] = scale * b * slopes[i]
    lut = []
    for value in range(256):
        x = value / 255.0
        segment = 0
        while segment < count - 2 and x > xs[segment + 1]:
            segment += 1
        width = xs[segment + 1] - xs[segment]
        t = (x - xs[segment]) / width if width > 0 else 0.0
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        y = (
            h00 * ys[segment]
            + h10 * width * tangents[segment]
            + h01 * ys[segment + 1]
            + h11 * width * tangents[segment + 1]
        )
        lut.append(int(round(_clamp01(y) * 255)))
    return lut


def _curve_p(key, label, default):
    """A param descriptor for an interactive curve (list of ``[x, y]`` points)."""
    return {"key": key, "label": label, "default": default, "kind": "curve"}


def default_curve():
    """The identity curve as control points in ``0..255``."""
    return [[0, 0], [255, 255]]


def coerce_curve(value):
    """Clean a curve point list: int pairs, x clamped and sorted, endpoints fixed.

    The first and last points are pinned to ``x=0`` and ``x=255`` so the tone
    table always spans the whole input range, and duplicate x values are
    dropped. Returns the identity curve when fewer than two points remain.
    """
    points = []
    if isinstance(value, (list, tuple)):
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            try:
                x = int(round(float(item[0])))
                y = int(round(float(item[1])))
            except (TypeError, ValueError):
                continue
            points.append([max(0, min(255, x)), max(0, min(255, y))])
    points.sort(key=lambda point: point[0])
    cleaned = []
    for point in points:
        if cleaned and point[0] == cleaned[-1][0]:
            continue
        cleaned.append(point)
    if len(cleaned) < 2:
        return default_curve()
    cleaned[0][0] = 0
    cleaned[-1][0] = 255
    return cleaned


def curve_lut(points):
    """A 256-entry tone table for curve ``points`` (0..255).

    Shared by the Curves filter and its editor dialog so the drawn curve and the
    applied tone match exactly.
    """
    cleaned = coerce_curve(points)
    return _spline_lut([(x / 255.0, y / 255.0) for x, y in cleaned])


def _points_lut(params):
    return curve_lut(params.get("points"))


def _points(image, params):
    rgb, alpha = _rgb_parts(image)
    table = _points_lut(params)
    return _rejoin(rgb.point(lambda value, _t=table: _t[value]), alpha)


def _levels(image, params):
    rgb, alpha = _rgb_parts(image)
    table = _levels_lut(params)
    return _rejoin(rgb.point(lambda value, _t=table: _t[value]), alpha)


# ---------------------------------------------------------------------- blur
def _gaussian_blur(image, params):
    return image.filter(ImageFilter.GaussianBlur(float(params["radius"])))


def _box_blur(image, params):
    return image.filter(ImageFilter.BoxBlur(float(params["radius"])))


def _median(image, params):
    size = max(1, int(round(params["size"])))
    return image.filter(ImageFilter.MedianFilter(size))


def _unsharp(image, params):
    rgb, alpha = _rgb_parts(image)
    out = rgb.filter(ImageFilter.UnsharpMask(
        radius=float(params["radius"]),
        percent=int(params["percent"]),
        threshold=int(params["threshold"]),
    ))
    return _rejoin(out, alpha)


def _sharpen(image, params):
    rgb, alpha = _rgb_parts(image)
    out = ImageEnhance.Sharpness(rgb).enhance(params["factor"])
    return _rejoin(out, alpha)


# -------------------------------------------------------------------- stylize
def _edge_enhance(image, params):
    rgb, alpha = _rgb_parts(image)
    filt = ImageFilter.EDGE_ENHANCE_MORE if params["more"] else ImageFilter.EDGE_ENHANCE
    return _rejoin(rgb.filter(filt), alpha)


def _emboss(image, _params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(rgb.filter(ImageFilter.EMBOSS), alpha)


def _contour(image, _params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(rgb.filter(ImageFilter.CONTOUR), alpha)


def _smooth(image, params):
    rgb, alpha = _rgb_parts(image)
    filt = ImageFilter.SMOOTH_MORE if params["more"] else ImageFilter.SMOOTH
    return _rejoin(rgb.filter(filt), alpha)


def _detail(image, _params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(rgb.filter(ImageFilter.DETAIL), alpha)


def _find_edges(image, _params):
    rgb, alpha = _rgb_parts(image)
    return _rejoin(rgb.filter(ImageFilter.FIND_EDGES), alpha)


# --------------------------------------------------------------- pixel/color
def _pixelate(image, params):
    size = max(1, int(round(params["size"])))
    width, height = image.size
    small = image.resize(
        (max(1, width // size), max(1, height // size)), Image.BILINEAR
    )
    return small.resize((width, height), Image.NEAREST)


def _quantize(image, params):
    colors = max(2, min(256, int(params["colors"])))
    rgb, alpha = _rgb_parts(image)
    out = rgb.convert("P", palette=Image.ADAPTIVE, colors=colors).convert("RGB")
    return _rejoin(out, alpha)


def _noise(image, params):
    sigma = float(params["amount"])
    if sigma <= 0:
        return image
    rgb, alpha = _rgb_parts(image)
    noise = Image.effect_noise(rgb.size, sigma).convert("RGB")
    out = ImageChops.add(rgb, noise, scale=1.0, offset=-128)
    return _rejoin(out, alpha)


def _vignette(image, params):
    strength = params["strength"] / 100.0
    if strength <= 0:
        return image
    rgb, alpha = _rgb_parts(image)
    small = 64
    mask = Image.new("L", (small, small), 0)
    draw = ImageDraw.Draw(mask)
    steps = 24
    for i in range(steps):
        frac = (i + 1) / steps
        inset = (1 - frac) / 2 * small
        draw.ellipse(
            (inset, inset, small - inset - 1, small - inset - 1),
            fill=int(255 * frac),
        )
    mask = mask.resize(rgb.size, Image.BILINEAR)
    mask = mask.point(lambda value: 255 - int((255 - value) * strength))
    out = Image.composite(rgb, Image.new("RGB", rgb.size, (0, 0, 0)), mask)
    return _rejoin(out, alpha)


# ------------------------------------------------------------- more adjustments
def _exposure(image, params):
    """Scale brightness by an exposure amount measured in stops."""
    gain = 2.0 ** float(params["stops"])
    table = [max(0, min(255, int(round(v * gain)))) for v in range(256)]
    rgb, alpha = _rgb_parts(image)
    return _rejoin(rgb.point(lambda value, _t=table: _t[value]), alpha)


def _vibrance(image, params):
    """Boost saturation most in the least-saturated pixels, sparing skin."""
    amount = float(params["amount"]) / 100.0
    if amount == 0.0:
        return image
    rgb, alpha = _rgb_parts(image)
    arr = np.asarray(rgb).astype(np.float32)
    high = arr.max(axis=2, keepdims=True)
    low = arr.min(axis=2, keepdims=True)
    sat = (high - low) / np.maximum(high, 1.0)
    weight = (1.0 - sat) * amount
    gray = arr.mean(axis=2, keepdims=True)
    out = gray + (arr - gray) * (1.0 + weight)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _rejoin(Image.fromarray(out, "RGB"), alpha)


def _color_balance(image, params):
    """Shift each channel, weighting the change toward what is missing."""
    shifts = (
        float(params["red"]) / 100.0,
        float(params["green"]) / 100.0,
        float(params["blue"]) / 100.0,
    )
    rgb, alpha = _rgb_parts(image)
    arr = np.asarray(rgb).astype(np.float32)
    for index, amount in enumerate(shifts):
        if amount == 0.0:
            continue
        channel = arr[:, :, index]
        if amount >= 0:
            arr[:, :, index] = channel + (255.0 - channel) * amount
        else:
            arr[:, :, index] = channel + channel * amount
    out = np.clip(arr, 0, 255).astype(np.uint8)
    return _rejoin(Image.fromarray(out, "RGB"), alpha)


# Hue centres (0-255) for the six selective-colour ranges.
_SELECTIVE_CENTERS = (0, 42, 85, 128, 170, 213)
_SELECTIVE_NAMES = (
    ("reds", "Reds"), ("yellows", "Yellows"), ("greens", "Greens"),
    ("cyans", "Cyans"), ("blues", "Blues"), ("magentas", "Magentas"),
)


def _selective(image, params):
    """Shift cyan/red, magenta/green and yellow/blue within one hue range."""
    rgb, alpha = _rgb_parts(image)
    hsv = np.asarray(rgb.convert("HSV")).astype(np.float32)
    hue = hsv[:, :, 0]
    center = _SELECTIVE_CENTERS[max(0, min(5, int(params["target"])))]
    distance = np.abs(((hue - center + 128.0) % 256.0) - 128.0)
    weight = np.clip(1.0 - distance / 48.0, 0.0, 1.0)[:, :, None]
    shifts = np.array([
        float(params["cyan_red"]) / 100.0,
        float(params["magenta_green"]) / 100.0,
        float(params["yellow_blue"]) / 100.0,
    ], dtype=np.float32)
    arr = np.asarray(rgb).astype(np.float32)
    adjusted = arr.copy()
    for index in range(3):
        amount = shifts[index]
        if amount == 0.0:
            continue
        channel = adjusted[:, :, index]
        if amount >= 0:
            adjusted[:, :, index] = channel + (255.0 - channel) * amount
        else:
            adjusted[:, :, index] = channel + channel * amount
    out = arr * (1.0 - weight) + adjusted * weight
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _rejoin(Image.fromarray(out, "RGB"), alpha)


def _gradient_map(image, params):
    """Remap luminance through a two-colour gradient."""
    rgb, alpha = _rgb_parts(image)
    gray = np.asarray(ImageOps.grayscale(rgb)).astype(np.float32) / 255.0
    start = np.array(rgba(params["start"])[:3], dtype=np.float32)
    end = np.array(rgba(params["end"])[:3], dtype=np.float32)
    t = gray[:, :, None]
    out = start[None, None, :] * (1.0 - t) + end[None, None, :] * t
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _rejoin(Image.fromarray(out, "RGB"), alpha)


def _photo_filter(image, params):
    """Tint the image with a colour at a given density (a warm/cool filter)."""
    density = float(params["density"]) / 100.0
    if density <= 0.0:
        return image
    rgb, alpha = _rgb_parts(image)
    tint = Image.new("RGB", rgb.size, tuple(rgba(params["color"])[:3]))
    out = Image.blend(rgb, tint, min(1.0, density))
    return _rejoin(out, alpha)


# ------------------------------------------------------------- more blur/stylize
def _motion_blur(image, params):
    """A straight-line blur at ``angle`` degrees over ``distance`` pixels."""
    distance = max(1, int(round(params["distance"])))
    angle = float(params["angle"])
    rgb, alpha = _rgb_parts(image)
    rotated = rgb.rotate(angle, resample=Image.BILINEAR, expand=False)
    blurred = rotated.filter(ImageFilter.BoxBlur((distance, 0)))
    back = blurred.rotate(-angle, resample=Image.BILINEAR, expand=False)
    return _rejoin(back, alpha)


def _radial_blur(image, params):
    """A zoom blur: blend the image with progressively smaller copies."""
    strength = max(0.0, min(1.0, params["strength"] / 100.0))
    if strength <= 0.0:
        return image
    width, height = image.size
    out = image.convert("RGBA")
    steps = 8
    for step in range(1, steps + 1):
        factor = 1.0 - strength * (step / steps) * 0.25
        small = image.resize(
            (max(1, int(width * factor)), max(1, int(height * factor))),
            Image.BILINEAR,
        )
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        layer.paste(small, ((width - small.width) // 2,
                            (height - small.height) // 2))
        out = Image.blend(out, layer, 0.5)
    return out


def _lens_blur(image, params):
    """A soft defocus with a highlight bloom, approximating a lens blur."""
    radius = max(0.0, float(params["radius"]))
    rgb, alpha = _rgb_parts(image)
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius))
    highlights = rgb.point(lambda v: 255 if v > 170 else 0)
    bloom = highlights.filter(ImageFilter.GaussianBlur(radius * 1.5))
    bloom = bloom.point(lambda v: int(v * 0.6))
    return _rejoin(ImageChops.screen(blurred, bloom), alpha)


def _glow(image, params):
    """Screen a blurred copy over the original for a soft bloom."""
    strength = max(0.0, min(1.0, params["strength"] / 100.0))
    if strength <= 0.0:
        return image
    radius = max(0.0, float(params["radius"]))
    rgb, alpha = _rgb_parts(image)
    blurred = rgb.filter(ImageFilter.GaussianBlur(radius))
    screened = ImageChops.screen(rgb, blurred)
    return _rejoin(Image.blend(rgb, screened, strength), alpha)


def _oil_paint(image, params):
    """Flatten the image into painterly patches of colour."""
    size = max(3, min(9, int(round(params["size"])) | 1))
    rgb, alpha = _rgb_parts(image)
    out = rgb.filter(ImageFilter.MedianFilter(size))
    out = ImageOps.posterize(out, 5)
    out = out.filter(ImageFilter.ModeFilter(size))
    return _rejoin(out, alpha)


def _watercolor(image, params):
    """Smooth washes of colour with darkened edges."""
    size = max(3, min(9, int(round(params["size"])) | 1))
    rgb, alpha = _rgb_parts(image)
    smooth = rgb.filter(ImageFilter.MedianFilter(size))
    smooth = ImageOps.posterize(smooth, 6)
    edges = rgb.convert("L").filter(ImageFilter.FIND_EDGES)
    lines = edges.point(lambda v: 255 - min(255, v * 2))
    outline = Image.merge("RGB", (lines, lines, lines))
    return _rejoin(ImageChops.multiply(smooth, outline), alpha)


# ---------------------------------------------------------------------- manga
# The two manga screens are pure functions of the pixel coordinate and the
# grayscale value, so they run as fragment shaders: one pass replaces the
# grid allocation and the several full-canvas float planes the numpy versions
# build. The shaders mirror the numpy maths exactly (the same luma weights,
# the same rotated cell coordinate, the same strict comparisons), so the two
# agree.
# ``_halftone`` outputs pure black/white so its values quantise exactly; the
# screentone blend is read back as floats and truncated to bytes, matching
# ``np.clip(...).astype(np.uint8)``. ``_HALFTONE_IDS``/``_SCREENTONE_IDS`` map the
# stored pattern names to the shader's integer selectors.
_HALFTONE_FRAGMENT = """#version 330
uniform sampler2D u_tex0;
uniform vec2 u_tex_size;
uniform float u_cell;
uniform float u_angle;
uniform int u_shape;  // 0 dot, 1 line, 2 cross
out vec4 f_color;
void main() {
    vec2 pix = floor(gl_FragCoord.xy);
    vec4 c = texture(u_tex0, (pix + 0.5) / u_tex_size);
    float gray = floor(255.0 * (c.r * 299.0 + c.g * 587.0 + c.b * 114.0)
                       / 1000.0 + 0.5);
    float theta = radians(u_angle);
    float co = cos(theta);
    float si = sin(theta);
    float rx = pix.x * co + pix.y * si;
    float ry = -pix.x * si + pix.y * co;
    float u = mod(rx, u_cell);
    float v = mod(ry, u_cell);
    float half = u_cell * 0.5;
    float dx = abs(u - half) / half;
    float dy = abs(v - half) / half;
    float radius = (255.0 - gray) / 255.0 * 1.0001;
    bool ink;
    if (u_shape == 1) {
        ink = dy < radius;
    } else if (u_shape == 2) {
        ink = (dx < radius) || (dy < radius);
    } else {
        ink = sqrt(dx * dx + dy * dy) / sqrt(2.0) < radius;
    }
    float value = ink ? 0.0 : 1.0;
    f_color = vec4(value, value, value, c.a);
}
"""

_SCREENTONE_FRAGMENT = """#version 330
uniform sampler2D u_tex0;
uniform vec2 u_tex_size;
uniform float u_cell;
uniform float u_angle;
uniform int u_pattern;  // 0 dot, 1 line, 2 cross, 3 grid
uniform float u_strength;
uniform vec3 u_color;
out vec4 f_color;
void main() {
    vec2 pix = floor(gl_FragCoord.xy);
    vec4 c = texture(u_tex0, (pix + 0.5) / u_tex_size);
    float gray = floor(255.0 * (c.r * 299.0 + c.g * 587.0 + c.b * 114.0)
                       / 1000.0 + 0.5);
    float theta = radians(u_angle);
    float co = cos(theta);
    float si = sin(theta);
    float rx = pix.x * co + pix.y * si;
    float ry = -pix.x * si + pix.y * co;
    float u = mod(rx, u_cell);
    float v = mod(ry, u_cell);
    float half = u_cell * 0.5;
    float band = u_cell * 0.12;
    bool mask;
    if (u_pattern == 1) {
        mask = abs(v - half) < band;
    } else if (u_pattern == 2) {
        mask = (abs(v - half) < band) || (abs(u - half) < band);
    } else if (u_pattern == 3) {
        mask = (u < band) || (v < band);
    } else {
        mask = (u - half) * (u - half) + (v - half) * (v - half)
             < (u_cell * 0.30) * (u_cell * 0.30);
    }
    float darkness = 1.0 - gray / 255.0;
    float weight = (mask ? 1.0 : 0.0) * darkness * u_strength;
    f_color = vec4(c.rgb * (1.0 - weight) + u_color * weight, c.a);
}
"""

_HALFTONE_IDS = {"dot": 0, "line": 1, "cross": 2}
_SCREENTONE_IDS = {"dot": 0, "line": 1, "cross": 2, "grid": 3}


def _screen_coords(height, width, size, angle):
    """Rotated cell coordinates for a ``height``-by-``width`` image.

    Returns per-pixel positions inside a ``size``-by-``size`` cell rotated by
    ``angle`` degrees, as ``(u, v, cx, cy)``: ``u``/``v`` are the wrapped cell
    coordinates in pixels and ``cx``/``cy`` the ``size/2`` centre, so callers
    compare ``u``/``v`` against the centre directly. Rotating the coordinate
    grid rather than the image keeps the screen crisp with no resampling blur.
    """
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    theta = math.radians(angle)
    cos, sin = math.cos(theta), math.sin(theta)
    rx = xx * cos + yy * sin
    ry = -xx * sin + yy * cos
    u = np.mod(rx, size)
    v = np.mod(ry, size)
    return u, v, size / 2.0, size / 2.0


def _halftone(image, params):
    """Screen a grayscale copy into ink dots, lines or a crosshatch.

    The printed image is pure black/white: darker tones grow the ink shape
    (a dot, a bar, or both axes crossed) and lighter tones shrink it, which is
    the amplitude-modulated screen a manga tone layer is built from.
    """
    rgb, alpha = _rgb_parts(image)
    gray = np.asarray(ImageOps.grayscale(rgb)).astype(np.float32)
    height, width = gray.shape
    cell = max(2, int(round(params["cell"])))
    angle = float(params["angle"])
    shape = str(params["shape"])
    if gpuimage.available():
        screened = gpuimage.render(
            _HALFTONE_FRAGMENT, (width, height), textures=[image],
            uniforms={
                "u_tex_size": (float(width), float(height)),
                "u_cell": float(cell),
                "u_angle": angle,
                "u_shape": _HALFTONE_IDS.get(shape, 0),
            },
        )
        if screened is not None:
            out = Image.fromarray(screened, "RGBA")
            return out if alpha is not None else out.convert("RGB")
    u, v, cx, cy = _screen_coords(height, width, cell, angle)
    dx = np.abs(u - cx) / cx
    dy = np.abs(v - cy) / cy
    # The cell corner sits exactly at distance 1.0, so nudge the coverage just
    # past it: a fully black tone then inks every pixel, while pure white inks
    # none (its radius stays 0 and the comparison below is strict).
    radius = (255.0 - gray) / 255.0 * 1.0001
    if shape == "line":
        ink = dy < radius
    elif shape == "cross":
        ink = (dx < radius) | (dy < radius)
    else:
        distance = np.sqrt(dx * dx + dy * dy) / math.sqrt(2.0)
        ink = distance < radius
    out = np.where(ink, 0, 255).astype(np.uint8)
    return _rejoin(Image.fromarray(out, "L").convert("RGB"), alpha)


_SCREENTONE_PATTERNS = ("dot", "line", "cross", "grid")


def _line_noise(index):
    """Deterministic 0..1 noise for each ray or line, as a float32 array.

    The effects need per-line irregularity but must stay pure functions of the
    spec, so the variation comes from a cheap integer hash of the line index
    rather than a random module. The same index always yields the same offset.
    """
    value = np.asarray(index, dtype=np.int64)
    value = (value * 1103515245 + 12345) & 0x7FFFFFFF
    return ((value >> 8) % 65536).astype(np.float32) / 65535.0


def _focus_lines(image, params):
    """Radial concentration lines converging on a focal point.

    The lines are painted over the existing pixels and grow wider toward the
    image edge, leaving a clear disc around the focus so the subject stays
    readable. ``jitter`` varies each ray's angle and width; the alpha channel is
    untouched, so the filter inks only the areas the layer already shows.
    """
    rgb, alpha = _rgb_parts(image)
    base = np.asarray(rgb).astype(np.float32)
    height, width = base.shape[:2]
    opacity = _clamp01(float(params["opacity"]) / 100.0)
    if opacity <= 0.0:
        return image

    cx = float(params["center_x"]) / 100.0 * width
    cy = float(params["center_y"]) / 100.0 * height
    rays = max(2, int(round(float(params["rays"]))))
    line_width = max(1.0, float(params["width"]))
    jitter = _clamp01(float(params["jitter"]) / 100.0)
    max_distance = max(1.0, math.hypot(width * 0.5, height * 0.5))
    clear = _clamp01(float(params["inner"]) / 100.0) * max_distance

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx + 0.5 - cx
    dy = yy + 0.5 - cy
    distance = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx) - math.radians(float(params["angle"]))
    step = 2.0 * math.pi / rays
    slot = angle / step
    noise = _line_noise(np.floor(slot))
    phase = np.mod(slot + (noise - 0.5) * jitter * 0.5, 1.0)
    # Perpendicular distance in pixels to the nearest ray centre.
    offset = np.abs(phase - 0.5) * step * np.maximum(distance, 1.0)

    edge = np.clip(
        (distance - clear) / max(1.0, max_distance - clear), 0.0, 1.0
    )
    taper = 0.35 + 0.65 * edge
    width_px = line_width * taper * (1.0 + (noise - 0.5) * jitter)
    fade = np.clip(
        (distance - clear) / max(1.0, line_width * 2.0), 0.0, 1.0
    )
    mask = (offset < np.maximum(0.5, width_px * 0.5)) * fade * opacity

    color = np.array(rgba(params["color"])[:3], dtype=np.float32)
    out = base * (1.0 - mask[..., None]) + color * mask[..., None]
    return _rejoin(
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), alpha
    )


def _speed_lines(image, params):
    """Parallel action lines with irregular spacing and dash lengths.

    Lines run along ``angle``; ``spacing`` and ``thickness`` control the grid,
    ``length`` the ink-to-gap ratio along each line, and ``jitter`` the
    per-line offset and width. The alpha channel is left untouched.
    """
    rgb, alpha = _rgb_parts(image)
    base = np.asarray(rgb).astype(np.float32)
    height, width = base.shape[:2]
    opacity = _clamp01(float(params["opacity"]) / 100.0)
    if opacity <= 0.0:
        return image

    theta = math.radians(float(params["angle"]))
    cos, sin = math.cos(theta), math.sin(theta)
    spacing = max(2.0, float(params["spacing"]))
    thickness = max(1.0, float(params["thickness"]))
    length = _clamp01(float(params["length"]) / 100.0)
    jitter = _clamp01(float(params["jitter"]) / 100.0)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    along = (xx + 0.5) * cos + (yy + 0.5) * sin
    across = -(xx + 0.5) * sin + (yy + 0.5) * cos
    index = np.floor(across / spacing)
    noise = _line_noise(index)
    shifted = np.mod(across + (noise - 0.5) * spacing * jitter, spacing)
    width_px = thickness * (1.0 + (noise - 0.5) * 0.8 * jitter)
    line = shifted < np.maximum(0.5, width_px)

    # A dash period tied to the image size keeps the streaks readable at any
    # resolution; the per-line offset stops the gaps from lining up.
    period = max(spacing * 2.0, (width + height) * 0.18)
    dash = np.mod(along + noise * period, period) < period * length
    mask = (line & dash) * opacity

    color = np.array(rgba(params["color"])[:3], dtype=np.float32)
    out = base * (1.0 - mask[..., None]) + color * mask[..., None]
    return _rejoin(
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), alpha
    )


def _screentone(image, params):
    """Overlay a repeating tone pattern, gated by the image's darkness.

    Shadows and midtones take the tone colour where the pattern is set; light
    areas stay untouched, so the tone reads as a screen on the artwork rather
    than a flat wash over it. ``strength`` scales how much tone is laid down.
    """
    rgb, alpha = _rgb_parts(image)
    gray = np.asarray(ImageOps.grayscale(rgb)).astype(np.float32)
    height, width = gray.shape
    size = max(2, int(round(params["size"])))
    angle = float(params["angle"])
    pattern = str(params["pattern"])
    if pattern not in _SCREENTONE_PATTERNS:
        pattern = "dot"
    strength = max(0, min(100, int(params["strength"]))) / 100.0
    if gpuimage.available():
        color = rgba(params["color"])[:3]
        screen = gpuimage.render(
            _SCREENTONE_FRAGMENT, (width, height), textures=[image],
            uniforms={
                "u_tex_size": (float(width), float(height)),
                "u_cell": float(size),
                "u_angle": angle,
                "u_pattern": _SCREENTONE_IDS.get(pattern, 0),
                "u_strength": strength,
                "u_color": tuple(channel / 255.0 for channel in color),
            },
            dtype="f4",
        )
        if screen is not None:
            blended = np.clip(screen[..., :3] * 255.0, 0.0, 255.0).astype(np.uint8)
            return _rejoin(Image.fromarray(blended, "RGB"), alpha)
    u, v, cx, cy = _screen_coords(height, width, size, angle)
    band = size * 0.12
    if pattern == "line":
        mask = np.abs(v - cy) < band
    elif pattern == "cross":
        mask = (np.abs(v - cy) < band) | (np.abs(u - cx) < band)
    elif pattern == "grid":
        mask = (u < band) | (v < band)
    else:
        mask = (u - cx) ** 2 + (v - cy) ** 2 < (size * 0.30) ** 2
    darkness = 1.0 - gray / 255.0
    weight = mask.astype(np.float32) * darkness * strength
    color = np.array(rgba(params["color"])[:3], dtype=np.float32)
    base = np.asarray(rgb).astype(np.float32)
    out = base * (1.0 - weight[:, :, None]) + color[None, None, :] * weight[:, :, None]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return _rejoin(Image.fromarray(out, "RGB"), alpha)


_register(
    "brightness", "adjust", "Brightness",
    [_p("factor", "Amount", 0.0, 2.0, 1.0, 0.01)],
    _brightness, lambda p: f"Brightness {int(p['factor'] * 100)}%",
)
_register(
    "contrast", "adjust", "Contrast",
    [_p("factor", "Amount", 0.0, 2.0, 1.0, 0.01)],
    _contrast, lambda p: f"Contrast {int(p['factor'] * 100)}%",
)
_register(
    "saturation", "adjust", "Saturation",
    [_p("factor", "Amount", 0.0, 2.0, 1.0, 0.01)],
    _saturation, lambda p: f"Saturation {int(p['factor'] * 100)}%",
)
_register(
    "hue", "adjust", "Hue Shift",
    [_p("degrees", "Degrees", -180, 180, 0, 1, "int")],
    _hue, lambda p: f"Hue {int(p['degrees'])}\u00b0",
)
_register(
    "grayscale", "adjust", "Grayscale",
    [_p("amount", "Amount", 0, 100, 100, 1, "int")],
    _grayscale, lambda p: f"Grayscale {int(p['amount'])}%",
)
_register(
    "invert", "adjust", "Invert",
    [_p("amount", "Amount", 0, 100, 100, 1, "int")],
    _invert, lambda p: f"Invert {int(p['amount'])}%",
)
_register(
    "posterize", "adjust", "Posterize",
    [_p("bits", "Bits", 1, 8, 4, 1, "int")],
    _posterize, lambda p: f"Posterize {int(p['bits'])} bits",
)
_register(
    "threshold", "adjust", "Threshold",
    [_p("level", "Level", 0, 255, 128, 1, "int")],
    _threshold, lambda p: f"Threshold {int(p['level'])}",
)

_register(
    "levels", "adjust", "Levels",
    [
        _p("in_black", "Input black", 0, 255, 0, 1, "int"),
        _p("in_white", "Input white", 0, 255, 255, 1, "int"),
        _p("gamma", "Gamma", 0.1, 9.99, 1.0, 0.01),
        _p("out_black", "Output black", 0, 255, 0, 1, "int"),
        _p("out_white", "Output white", 0, 255, 255, 1, "int"),
    ],
    _levels,
    lambda p: (
        f"Levels {int(p['in_black'])}\u2013{int(p['in_white'])} "
        f"\u03b3{p['gamma']:g}"
    ),
)
_register(
    "curves", "adjust", "Curves",
    [_curve_p("points", "Curve", default_curve())],
    _points,
    lambda p: f"Curves ({len(coerce_curve(p.get('points')))} pts)",
)

_register(
    "gaussian_blur", "blur", "Gaussian Blur",
    [_p("radius", "Radius", 0.0, 50.0, 4.0, 0.5)],
    _gaussian_blur, lambda p: f"Gaussian Blur {p['radius']:g}px",
)
_register(
    "box_blur", "blur", "Box Blur",
    [_p("radius", "Radius", 0.0, 50.0, 4.0, 0.5)],
    _box_blur, lambda p: f"Box Blur {p['radius']:g}px",
)
_register(
    "median", "blur", "Median",
    [_p("size", "Size", 1, 15, 3, 2, "int")],
    _median, lambda p: f"Median {int(p['size'])}",
)
_register(
    "unsharp", "blur", "Unsharp Mask",
    [
        _p("radius", "Radius", 0.0, 50.0, 2.0, 0.5),
        _p("percent", "Amount", 0, 400, 150, 5, "int"),
        _p("threshold", "Threshold", 0, 255, 3, 1, "int"),
    ],
    _unsharp, lambda p: f"Unsharp {int(p['percent'])}%",
)
_register(
    "sharpen", "blur", "Sharpen",
    [_p("factor", "Amount", 0.0, 4.0, 1.5, 0.05)],
    _sharpen, lambda p: f"Sharpen {p['factor']:g}",
)

_register(
    "edge_enhance", "stylize", "Edge Enhance",
    [_p("more", "Stronger", 0, 1, 0, 1, "bool")],
    _edge_enhance, lambda p: "Edge Enhance (strong)" if p["more"] else "Edge Enhance",
)
_register(
    "emboss", "stylize", "Emboss", [], _emboss, lambda _p: "Emboss",
)
_register(
    "contour", "stylize", "Contour", [], _contour, lambda _p: "Contour",
)
_register(
    "smooth", "stylize", "Smooth",
    [_p("more", "Stronger", 0, 1, 0, 1, "bool")],
    _smooth, lambda p: "Smooth (strong)" if p["more"] else "Smooth",
)
_register(
    "detail", "stylize", "Detail", [], _detail, lambda _p: "Detail",
)
_register(
    "find_edges", "stylize", "Find Edges", [], _find_edges, lambda _p: "Find Edges",
)

_register(
    "pixelate", "pixel", "Pixelate",
    [_p("size", "Block", 1, 64, 8, 1, "int")],
    _pixelate, lambda p: f"Pixelate {int(p['size'])}px",
)
_register(
    "quantize", "pixel", "Quantize",
    [_p("colors", "Colors", 2, 256, 16, 1, "int")],
    _quantize, lambda p: f"Quantize {int(p['colors'])} colors",
)
_register(
    "noise", "pixel", "Noise",
    [_p("amount", "Amount", 0.0, 128.0, 24.0, 1.0)],
    _noise, lambda p: f"Noise {int(p['amount'])}",
)
_register(
    "vignette", "pixel", "Vignette",
    [_p("strength", "Strength", 0, 100, 60, 1, "int")],
    _vignette, lambda p: f"Vignette {int(p['strength'])}%",
)

_register(
    "exposure", "adjust", "Exposure",
    [_p("stops", "Stops", -4.0, 4.0, 0.0, 0.1)],
    _exposure, lambda p: f"Exposure {p['stops']:+.1f} EV",
)
_register(
    "vibrance", "adjust", "Vibrance",
    [_p("amount", "Amount", -100, 100, 20, 1, "int")],
    _vibrance, lambda p: f"Vibrance {int(p['amount'])}%",
)
_register(
    "color_balance", "adjust", "Color Balance",
    [
        _p("red", "Red", -100, 100, 0, 1, "int"),
        _p("green", "Green", -100, 100, 0, 1, "int"),
        _p("blue", "Blue", -100, 100, 0, 1, "int"),
    ],
    _color_balance,
    lambda p: (f"Balance R{int(p['red'])} G{int(p['green'])} "
               f"B{int(p['blue'])}"),
)

_register(
    "selective", "color", "Selective Color",
    [
        _p("target", "Range", 0, 5, 0, 1, "int"),
        _p("cyan_red", "Cyan \u2013 Red", -100, 100, 0, 1, "int"),
        _p("magenta_green", "Magenta \u2013 Green", -100, 100, 0, 1, "int"),
        _p("yellow_blue", "Yellow \u2013 Blue", -100, 100, 0, 1, "int"),
    ],
    _selective,
    lambda p: ("Selective "
               + _SELECTIVE_NAMES[max(0, min(5, int(p['target'])))][1]),
)
_register(
    "gradient_map", "color", "Gradient Map",
    [_c("start", "Shadow color", "#000000"), _c("end", "Highlight color", "#ffffff")],
    _gradient_map, lambda _p: "Gradient Map",
)
_register(
    "photo_filter", "color", "Photo Filter",
    [_c("color", "Color", "#ec8a00"),
     _p("density", "Density", 0, 100, 25, 1, "int")],
    _photo_filter, lambda p: f"Photo Filter {int(p['density'])}%",
)

_register(
    "motion_blur", "blur", "Motion Blur",
    [_p("angle", "Angle", 0, 360, 0, 1, "int"),
     _p("distance", "Distance", 1, 100, 12, 1, "int")],
    _motion_blur,
    lambda p: f"Motion Blur {int(p['distance'])}px @ {int(p['angle'])}\u00b0",
)
_register(
    "radial_blur", "blur", "Radial Blur",
    [_p("strength", "Strength", 0, 100, 30, 1, "int")],
    _radial_blur, lambda p: f"Radial Blur {int(p['strength'])}%",
)
_register(
    "lens_blur", "blur", "Lens Blur",
    [_p("radius", "Radius", 0.0, 50.0, 8.0, 0.5)],
    _lens_blur, lambda p: f"Lens Blur {p['radius']:g}px",
)

_register(
    "glow", "stylize", "Glow",
    [_p("radius", "Radius", 0.0, 50.0, 6.0, 0.5),
     _p("strength", "Strength", 0, 100, 60, 1, "int")],
    _glow, lambda p: f"Glow {int(p['strength'])}%",
)
_register(
    "oil_paint", "stylize", "Oil Paint",
    [_p("size", "Brush size", 3, 9, 5, 2, "int")],
    _oil_paint, lambda p: f"Oil Paint {int(p['size'])}",
)
_register(
    "watercolor", "stylize", "Watercolor",
    [_p("size", "Detail", 3, 9, 5, 2, "int")],
    _watercolor, lambda p: f"Watercolor {int(p['size'])}",
)

_register(
    "halftone", "manga", "Halftone",
    [
        _choice(
            "shape", "Screen",
            [("dot", "Dot"), ("line", "Line"), ("cross", "Crosshatch")],
            "dot",
        ),
        _p("cell", "Cell size", 2, 64, 6, 1, "int"),
        _p("angle", "Angle", 0, 180, 45, 1, "int"),
    ],
    _halftone,
    lambda p: f"Halftone {int(p['cell'])}px @ {int(p['angle'])}\u00b0",
)
_register(
    "screentone", "manga", "Screentone",
    [
        _choice(
            "pattern", "Pattern",
            [("dot", "Dots"), ("line", "Lines"), ("cross", "Crosshatch"),
             ("grid", "Grid")],
            "dot",
        ),
        _p("size", "Size", 2, 64, 8, 1, "int"),
        _p("angle", "Angle", 0, 180, 45, 1, "int"),
        _c("color", "Tone color", "#000000"),
        _p("strength", "Strength", 0, 100, 100, 1, "int"),
    ],
    _screentone,
    lambda p: f"Screentone {int(p['size'])}px @ {int(p['angle'])}\u00b0",
)
_register(
    "focus_lines", "manga", "Focus Lines",
    [
        _p("center_x", "Center X %", 0, 100, 50, 1, "int"),
        _p("center_y", "Center Y %", 0, 100, 50, 1, "int"),
        _p("rays", "Lines", 4, 240, 48, 1, "int"),
        _p("width", "Line width", 1, 16, 2, 1, "int"),
        _p("inner", "Clear centre", 0, 100, 18, 1, "int"),
        _p("angle", "Angle", 0, 180, 0, 1, "int"),
        _p("jitter", "Roughness", 0, 100, 25, 1, "int"),
        _p("opacity", "Opacity", 0, 100, 100, 1, "int"),
        _c("color", "Ink color", "#000000"),
    ],
    _focus_lines,
    lambda p: f"Focus Lines {int(p['rays'])} @ {int(p['angle'])}\u00b0",
)
_register(
    "speed_lines", "manga", "Speed Lines",
    [
        _p("angle", "Angle", 0, 180, 0, 1, "int"),
        _p("spacing", "Spacing", 2, 80, 14, 1, "int"),
        _p("thickness", "Line width", 1, 16, 2, 1, "int"),
        _p("length", "Dash length", 5, 100, 70, 1, "int"),
        _p("jitter", "Roughness", 0, 100, 30, 1, "int"),
        _p("opacity", "Opacity", 0, 100, 100, 1, "int"),
        _c("color", "Ink color", "#000000"),
    ],
    _speed_lines,
    lambda p: f"Speed Lines {int(p['spacing'])}px @ {int(p['angle'])}\u00b0",
)

# Category key -> display label, in menu order.
CATEGORIES = [
    ("adjust", "Adjustments"),
    ("color", "Color"),
    ("blur", "Blur & Sharpen"),
    ("stylize", "Stylize"),
    ("pixel", "Pixel & Color"),
    ("manga", "Manga"),
]


# ------------------------------------------------------------------ helpers
def label(filter_id):
    entry = FILTERS.get(filter_id)
    return entry["label"] if entry else filter_id


def default_params(filter_id):
    entry = FILTERS.get(filter_id)
    if entry is None:
        return {}
    # Deep-copy defaults: a curve's default is a list, which must not be shared
    # between every spec built from the same filter.
    return {param["key"]: copy.deepcopy(param["default"]) for param in entry["params"]}


def new_spec(filter_id):
    """A fresh, enabled spec for ``filter_id`` with default parameters."""
    return {
        "type": filter_id,
        "enabled": True,
        "params": default_params(filter_id),
    }


def clone(spec):
    """A deep copy of ``spec`` safe to store in a history snapshot."""
    return {
        "type": spec.get("type"),
        "enabled": bool(spec.get("enabled", True)),
        "params": copy.deepcopy(spec.get("params", {})),
    }


def normalize_spec(spec):
    """Coerce a loaded spec to a known filter, or return ``None``.

    Unknown types are dropped and missing params fall back to their defaults, so
    projects saved by a newer build still load.
    """
    if not isinstance(spec, dict):
        return None
    filter_id = spec.get("type")
    entry = FILTERS.get(filter_id)
    if entry is None:
        return None
    params = default_params(filter_id)
    loaded = spec.get("params")
    if isinstance(loaded, dict):
        for param in entry["params"]:
            key = param["key"]
            if key in loaded:
                value = loaded[key]
                if param["kind"] == "bool":
                    params[key] = bool(value)
                elif param["kind"] == "int":
                    params[key] = int(round(float(value)))
                elif param["kind"] == "curve":
                    params[key] = coerce_curve(value)
                elif param["kind"] in ("color", "choice"):
                    params[key] = str(value)
                else:
                    params[key] = float(value)
    return {
        "type": filter_id,
        "enabled": bool(spec.get("enabled", True)),
        "params": params,
    }


def describe(spec):
    """A short human label for ``spec`` (used by the layer list)."""
    entry = FILTERS.get(spec.get("type"))
    if entry is None:
        return str(spec.get("type", "Filter"))
    try:
        return entry["summary"](spec.get("params", {}))
    except Exception:
        return entry["label"]


def _freeze(value):
    """Make ``value`` hashable for cache keys (lists/dicts become tuples)."""
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((key, _freeze(item)) for key, item in value.items()))
    return value


def spec_key(spec):
    """A hashable, order-independent signature of ``spec`` for caching."""
    params = spec.get("params", {})
    return (
        spec.get("type"),
        bool(spec.get("enabled", True)),
        tuple(sorted((k, _freeze(params[k])) for k in params)),
    )


def chain_key(specs):
    """Cache signature for an ordered chain of specs."""
    return tuple(spec_key(spec) for spec in specs if spec.get("enabled", True))


def filters_in_category(category):
    """Ids in ``category``, in registration order."""
    return [fid for fid, entry in FILTERS.items() if entry["category"] == category]


def apply_chain(image, specs):
    """Apply every enabled spec, in order, returning a new image."""
    result = image
    for spec in specs:
        if not spec.get("enabled", True):
            continue
        entry = FILTERS.get(spec.get("type"))
        if entry is None:
            continue
        result = entry["apply"](result, spec.get("params", {}))
    return result
