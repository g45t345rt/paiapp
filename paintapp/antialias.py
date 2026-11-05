"""Supersampled anti-aliasing shared by the raster tools.

Pillow's ``ImageDraw`` rasterises primitives with hard edges, so the Shape and
Path tools draw their geometry into a canvas scaled up by :data:`SCALE` and
average it back down, the same trick :func:`brush._build_dab` uses for its
anti-aliased stamp.

The tools render only the geometry's bounding box rather than the whole layer,
so :func:`scale_for` shrinks the scale when that box is large enough that a
supersampled canvas would cost too much memory.
"""

import numpy as np
from PIL import Image, ImageChops

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
