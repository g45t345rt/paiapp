"""Reusable GPU pixel operations for the raster tools (optional).

:mod:`paintapp.gpuimage` provides the context and the shader plumbing; this
module keeps the operations more than one tool needs. Right now that is the
square morphology behind the paint bucket's gap seal (:mod:`paintapp.fill`) and
the selection's grow/shrink (:mod:`paintapp.selection`), which would otherwise
be PIL's ``MinFilter``/``MaxFilter`` - a full-canvas rank filter whose cost grows
with the square of the radius.

Each entry point returns ``None`` when no GL context is usable, so the callers
keep their PIL path and nothing changes on a machine without GL.
"""

from PIL import Image

from . import gpuimage

# Separable square morphology: the window is min/max over ``+/-`` ``u_radius``
# steps along one axis, so a horizontal pass followed by a vertical one equals
# PIL's square ``MinFilter``/``MaxFilter`` (the structuring element is the same
# square). Sampling clamps to the edge, matching PIL's border handling. ``u_mode``
# picks erosion (0, the minimum) or dilation (1, the maximum).
_MORPH_FRAGMENT = """#version 330
uniform sampler2D u_tex0;
uniform vec2 u_tex_size;
uniform vec2 u_step;
uniform int u_radius;
uniform int u_mode;
out vec4 f_color;
void main() {
    vec2 base = floor(gl_FragCoord.xy);
    vec4 acc = texture(u_tex0, (base + 0.5) / u_tex_size);
    for (int i = 1; i <= 64; ++i) {
        if (i > u_radius) break;
        vec2 forward = base + u_step * float(i);
        vec2 backward = base - u_step * float(i);
        vec2 uv_forward = (clamp(forward, vec2(0.0), u_tex_size - 1.0) + 0.5) / u_tex_size;
        vec2 uv_backward = (clamp(backward, vec2(0.0), u_tex_size - 1.0) + 0.5) / u_tex_size;
        vec4 a = texture(u_tex0, uv_forward);
        vec4 b = texture(u_tex0, uv_backward);
        if (u_mode == 0) {
            acc = min(acc, min(a, b));
        } else {
            acc = max(acc, max(a, b));
        }
    }
    f_color = acc;
}
"""

# The largest radius the shader loop is unrolled for; PIL's callers cap at 20.
MAX_RADIUS = 64


# A soft round/square brush dab, matching ``brush._build_dab``: each pixel is
# full inside the "core", falls off linearly to zero at the rim, and is drawn at
# the supersampled size so the caller can downscale to the same anti-aliased
# edge as the PIL path. ``floor(gl_FragCoord)`` samples the same integer grid as
# the PIL loop, so a rim pixel reads as empty exactly when the CPU's does.
_DAB_FRAGMENT = """#version 330
uniform float u_radius;
uniform float u_core;
uniform float u_falloff;
uniform int u_square;
out vec4 f_color;
void main() {
    vec2 d = floor(gl_FragCoord.xy) - vec2(u_radius);
    float dist = (u_square == 1) ? max(abs(d.x), abs(d.y)) : length(d);
    float value;
    if (dist >= u_radius) {
        value = 0.0;
    } else if (dist <= u_core) {
        value = 1.0;
    } else {
        value = 1.0 - (dist - u_core) / u_falloff;
    }
    value = clamp(value, 0.0, 1.0);
    f_color = vec4(value, value, value, 1.0);
}
"""


def brush_dab(size, hardness, antialias, tip, supersample=3):
    """Rasterise a brush dab as an ``"L"`` image on the GPU, or ``None``.

    Mirrors :func:`paintapp.brush._build_dab` (same core/falloff maths and the
    same supersample-then-downscale anti-aliasing) so swapping the PIL loop for
    this pass leaves the dab essentially unchanged. Returns ``None`` when no GL
    context is available, so the caller keeps its CPU path.
    """
    d = max(2, int(round(size)))
    if not gpuimage.available():
        return None
    ss = supersample if antialias else 1
    n = d * ss
    radius = (n - 1) / 2.0
    core = radius * max(0.0, min(100.0, float(hardness))) / 100.0
    falloff = max(1e-6, radius - core)
    pixels = gpuimage.render(
        _DAB_FRAGMENT, (n, n),
        uniforms={
            "u_radius": float(radius),
            "u_core": float(core),
            "u_falloff": float(falloff),
            "u_square": 1 if tip == "square" else 0,
        },
    )
    if pixels is None:
        return None
    dab = Image.fromarray(pixels[..., 0], "L")
    if ss > 1:
        dab = dab.resize((d, d), Image.LANCZOS)
    return dab


def morphology(mask, radius, mode):
    """Erode (``"min"``) or dilate (``"max"``) an ``"L"`` mask by a square window.

    ``radius`` is in pixels and the window is ``2 * radius + 1`` square, exactly
    like ``ImageFilter.MinFilter``/``MaxFilter``. Returns an ``"L"`` image, or
    ``None`` when no GL context is available or the pass fails.
    """
    radius = int(radius)
    if radius < 1 or radius > MAX_RADIUS or not gpuimage.available():
        return None
    width, height = mask.size
    if width < 1 or height < 1:
        return None
    uniforms = {
        "u_tex_size": (float(width), float(height)),
        "u_radius": radius,
        "u_mode": 0 if mode == "min" else 1,
    }
    horizontal = gpuimage.render(
        _MORPH_FRAGMENT, (width, height), textures=[mask],
        uniforms=dict(uniforms, u_step=(1.0, 0.0)),
    )
    if horizontal is None:
        return None
    column = Image.fromarray(horizontal[..., 0], "L")
    vertical = gpuimage.render(
        _MORPH_FRAGMENT, (width, height), textures=[column],
        uniforms=dict(uniforms, u_step=(0.0, 1.0)),
    )
    if vertical is None:
        return None
    return Image.fromarray(vertical[..., 0], "L")


# Multiplies one image (the stroke coverage) by a tiled grain texture sampled at
# a document offset: ``grain`` tiles with wraparound, so the same document pixel
# always reads the same value no matter where the stroke's dirty rectangle
# starts. ``u_size`` (the render target size) is set by :mod:`gpuimage`.
_GRAIN_FRAGMENT = """#version 330
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform vec2 u_size;
uniform vec2 u_tile_size;
uniform vec2 u_offset;
out vec4 f_color;
void main() {
    vec2 pixel = floor(gl_FragCoord.xy);
    float coverage = texture(u_tex0, (pixel + 0.5) / u_size).x;
    vec2 grain_at = mod(pixel - u_offset, u_tile_size);
    float grain = texture(u_tex1, (grain_at + 0.5) / u_tile_size).x;
    float value = coverage * grain;
    f_color = vec4(value, value, value, 1.0);
}
"""


def grain_overlay(coverage, tile, offset_x, offset_y):
    """Multiply ``coverage`` by the tiled ``tile`` at a document offset.

    ``tile`` wraps, so ``(offset_x, offset_y)`` places it in document space and a
    stroke's grain stays put as it paints. Returns an ``"L"`` image, or ``None``
    when no GL context is available.
    """
    if not gpuimage.available():
        return None
    width, height = coverage.size
    tile_width, tile_height = tile.size
    if width < 1 or height < 1 or tile_width < 1 or tile_height < 1:
        return None
    pixels = gpuimage.render(
        _GRAIN_FRAGMENT, (width, height), textures=[coverage, tile],
        uniforms={
            "u_tile_size": (float(tile_width), float(tile_height)),
            "u_offset": (float(offset_x), float(offset_y)),
        },
    )
    if pixels is None:
        return None
    return Image.fromarray(pixels[..., 0], "L")


def dilate(mask, radius):
    """Dilate ``mask`` by ``radius`` pixels, or ``None`` without a GL context."""
    return morphology(mask, radius, "max")


def erode(mask, radius):
    """Erode ``mask`` by ``radius`` pixels, or ``None`` without a GL context."""
    return morphology(mask, radius, "min")
