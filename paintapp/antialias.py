"""Supersampled anti-aliasing shared by the raster tools.

Pillow's ``ImageDraw`` rasterises primitives with hard edges, so the Shape and
Path tools draw their geometry into a canvas scaled up by :data:`SCALE` and
average it back down, the same trick :func:`brush._build_dab` uses for its
anti-aliased stamp.

The tools render only the geometry's bounding box rather than the whole layer,
so :func:`scale_for` shrinks the scale when that box is large enough that a
supersampled canvas would cost too much memory.
"""

import math

import numpy as np
from PIL import Image, ImageChops, ImageFilter

# Supersampling factor for a canvas that comfortably fits the memory cap.
SCALE = 3

# Ceiling on a supersampled canvas, in pixels (~144 MB as RGBA). Larger
# geometry falls back to a smaller scale, then to no anti-aliasing.
MAX_PIXELS = 36_000_000


def scale_for(width, height, preferred=SCALE):
    """The largest scale up to ``preferred`` that fits :data:`MAX_PIXELS`."""
    width = max(1, int(width))
    height = max(1, int(height))
    for scale in range(max(1, int(preferred)), 1, -1):
        if width * height * scale * scale <= MAX_PIXELS:
            return scale
    return 1


def downscale(image, scale):
    """Average an RGBA ``image`` down by an integer ``scale``.

    Colours are premultiplied by alpha before the box filter and divided back
    out afterwards, so the transparent black around a shape cannot bleed into
    the colour of an edge pixel and darken it. The divide runs on the already
    downscaled image, keeping the float buffers small.
    """
    if scale <= 1:
        return image
    width, height = image.size
    out_size = (max(1, width // scale), max(1, height // scale))
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    premultiplied = ImageChops.multiply(rgba.convert("RGB"), alpha.convert("RGB"))
    small_alpha = alpha.resize(out_size, Image.BOX)
    small_color = premultiplied.resize(out_size, Image.BOX)
    a = np.asarray(small_alpha, dtype=np.float32) / 255.0
    rgb = np.asarray(small_color, dtype=np.float32)
    rgb = np.clip(rgb / np.maximum(a, 1e-6)[..., None], 0.0, 255.0)
    out = np.empty((out_size[1], out_size[0], 4), dtype=np.uint8)
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = np.asarray(small_alpha)
    return Image.fromarray(out, "RGBA")


def affine_resample(image, size, data, resample=Image.BICUBIC):
    """Resample ``image`` through Pillow's ``AFFINE`` ``data`` without aliasing.

    ``Image.transform`` samples a *fixed-size* kernel and, unlike
    :meth:`Image.resize`, never widens it when the mapping shrinks the image.
    A 2x reduction then reads a 4x4 neighbourhood sized for 1:1, so most source
    pixels are skipped and thin strokes break up - which is why a squashed layer
    looks worse than the same image scaled by an image editor that uses a
    scale-adaptive filter (GIMP's NoHalo/LoHalo, Photoshop's bicubic-sharper).

    ``data`` is the six-tuple ``Image.transform`` expects, mapping output pixel
    centres into source coordinates. A pure scale/translation is handed to
    :meth:`Image.resize` (whose kernels *are* scale-adaptive when shrinking, and
    which covers the whole source) and the off-canvas part is composited
    transparent, exactly as the affine's fill would. A rotated or sheared shrink
    is prefilted with a resize down to the output scale first, then resampled
    with the affine, so the fixed kernel only ever enlarges. Upscales are
    untouched and stay byte-identical to ``Image.transform``; ``NEAREST`` keeps
    its hard-edged lookup and is never prefilted.
    """
    a, b, c, d, e, f = data
    if resample == Image.NEAREST:
        # "Nearest" is a deliberate hard-edged lookup: no prefilter.
        return image.transform(size, Image.AFFINE, data, resample=resample)
    if abs(b) < 1e-9 and abs(d) < 1e-9 and a > 0 and e > 0:
        if a <= 1.0 + 1e-9 and e <= 1.0 + 1e-9:
            return image.transform(size, Image.AFFINE, data, resample=resample)
        return _resize_box(image, size, a, c, e, f, resample)
    # A rotation/shear: prefilters the reduction, then rotates at scale.
    ratio = max(math.hypot(a, d), math.hypot(b, e))
    if ratio > 1.02:
        width = max(1, int(round(image.width / ratio)))
        height = max(1, int(round(image.height / ratio)))
        if (width, height) != image.size:
            sx = width / image.width
            sy = height / image.height
            data = (a * sx, b * sx, c * sx, d * sy, e * sy, f * sy)
            image = image.resize((width, height), resample)
    return image.transform(size, Image.AFFINE, data, resample=resample)


def _resize_box(image, size, a, c, e, f, resample):
    """Scale ``image`` into ``size`` through the axis-aligned mapping.

    ``Image.resize(box=...)`` refuses a box that leaves the source, so the
    overlap is resampled on its own and pasted into an empty output at the
    matching offset; outside the source stays the mode's zero fill, as the
    affine path would leave it.
    """
    width, height = image.size
    x0, y0 = c, f
    x1, y1 = c + a * size[0], f + e * size[1]
    left, top = max(0.0, x0), max(0.0, y0)
    right, bottom = min(float(width), x1), min(float(height), y1)
    if right <= left or bottom <= top:
        return Image.new(image.mode, size)
    # The output rectangle covered by the on-canvas part of the mapping.
    ox0 = max(0, int(math.floor((left - x0) / a)))
    oy0 = max(0, int(math.floor((top - y0) / e)))
    ox1 = min(size[0], int(math.ceil((right - x0) / a)))
    oy1 = min(size[1], int(math.ceil((bottom - y0) / e)))
    if ox1 <= ox0 or oy1 <= oy0:
        return Image.new(image.mode, size)
    box = (
        max(0.0, x0 + a * ox0), max(0.0, y0 + e * oy0),
        min(float(width), x0 + a * ox1), min(float(height), y0 + e * oy1),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return Image.new(image.mode, size)
    part = image.resize((ox1 - ox0, oy1 - oy0), resample, box=box)
    out = Image.new(image.mode, size)
    out.paste(part, (ox0, oy0))
    return out


def blur_premultiplied(image, radius):
    """Soften an RGBA ``image``'s alpha by ``radius`` without a dark halo.

    Feathering is a Gaussian blur of the coverage, but the colour must stay the
    paint's own. Blurring a sprite whose transparent pixels are black would drag
    that black into the softened edge and leave a dark ring, so the colour is
    premultiplied by alpha before the blur and divided back out afterwards - the
    same trick :func:`downscale` uses.
    """
    if radius <= 0:
        return image
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    premultiplied = ImageChops.multiply(rgba.convert("RGB"), alpha.convert("RGB"))
    blurred_alpha = alpha.filter(ImageFilter.GaussianBlur(radius))
    blurred_color = premultiplied.filter(ImageFilter.GaussianBlur(radius))
    a = np.asarray(blurred_alpha, dtype=np.float32) / 255.0
    rgb = np.asarray(blurred_color, dtype=np.float32)
    rgb = np.clip(rgb / np.maximum(a, 1e-6)[..., None], 0.0, 255.0)
    out = np.empty((rgba.height, rgba.width, 4), dtype=np.uint8)
    out[..., :3] = rgb.astype(np.uint8)
    out[..., 3] = np.asarray(blurred_alpha)
    return Image.fromarray(out, "RGBA")
