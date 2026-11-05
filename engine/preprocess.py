"""Control-image preprocessors for ControlNet.

A ControlNet does not read a photograph directly: it is trained on a "control
hint" -- a canny edge map, a line drawing, and so on -- and consumes it through
its own conditioning embedding. These helpers turn the app's pixels (a canvas,
a layer or an asset) into such a hint.

Everything here is a pure ``numpy`` / ``Pillow`` image operation, so no model
download or extra dependency is needed. A hint that already exists -- a depth or
pose map produced elsewhere -- is passed through unchanged with the ``none``
preprocessor, so those control types work too as long as the user supplies the
map. :data:`PREPROCESSORS` lists the names in display order and :func:`prepare`
is the single entry point the pipeline calls.
"""

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# The available preprocessors, in display order. ``none`` is a plain resize, so
# it doubles as "the control image is already a hint".
PREPROCESSORS = ("none", "canny", "lineart")

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = _SOBEL_X.T.copy()

# Canny's double threshold on the 0..255 gradient magnitude and the blur applied
# before the gradients, matching the usual ControlNet canny recipe.
_CANNY_LOW = 100.0
_CANNY_HIGH = 200.0
_CANNY_SIGMA = 1.4


def prepare(image, name, size):
    """Return ``image`` as a ControlNet hint at ``size`` (a ``(width, height)``).

    The source is resized first, so an edge map is computed at the resolution
    the ControlNet actually sees. ``name`` selects the preprocessor; an unknown
    name falls back to ``none``. Returns ``None`` for a ``None`` image.
    """
    if image is None:
        return None
    width, height = int(size[0]), int(size[1])
    rgb = image.convert("RGB").resize((width, height), Image.LANCZOS)
    if name == "canny":
        return _canny(rgb)
    if name == "lineart":
        return _lineart(rgb)
    return rgb


def _canny(image):
    """A Canny edge map: white edges on black, as the ControlNet expects."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    blurred = np.asarray(
        Image.fromarray(gray.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(_CANNY_SIGMA)
        ),
        dtype=np.float32,
    )
    gx = _convolve(blurred, _SOBEL_X)
    gy = _convolve(blurred, _SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    direction = np.arctan2(gy, gx)
    suppressed = _non_max_suppression(magnitude, direction)
    edges = _hysteresis(suppressed, _CANNY_LOW, _CANNY_HIGH)
    return Image.fromarray((edges * 255).astype(np.uint8), "L").convert("RGB")


def _lineart(image):
    """A soft line drawing: dark lines on white, from a blurred edge filter."""
    gray = image.convert("L").filter(ImageFilter.GaussianBlur(1.0))
    return ImageOps.invert(gray.filter(ImageFilter.FIND_EDGES)).convert("RGB")


def _convolve(array, kernel):
    """Correlate ``array`` with a small ``kernel`` (edge-padded, same size)."""
    kh, kw = kernel.shape
    padded = np.pad(
        array, ((kh // 2, kh // 2), (kw // 2, kw // 2)), mode="edge",
    )
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.shape)
    return (windows * kernel).sum(axis=(-1, -2))


def _non_max_suppression(magnitude, direction):
    """Thin the gradient ridges to one-pixel-wide edges."""
    angle = (np.degrees(direction) + 180.0) % 180.0
    left = np.roll(magnitude, 1, axis=1)
    right = np.roll(magnitude, -1, axis=1)
    up = np.roll(magnitude, 1, axis=0)
    down = np.roll(magnitude, -1, axis=0)
    up_left = np.roll(up, 1, axis=1)
    up_right = np.roll(up, -1, axis=1)
    down_left = np.roll(down, 1, axis=1)
    down_right = np.roll(down, -1, axis=1)
    horizontal = (angle < 22.5) | (angle >= 157.5)
    diagonal_up = (angle >= 22.5) & (angle < 67.5)
    vertical = (angle >= 67.5) & (angle < 112.5)
    diagonal_down = (angle >= 112.5) & (angle < 157.5)
    keep = (
        (horizontal & (magnitude >= left) & (magnitude >= right))
        | (diagonal_up & (magnitude >= up_right) & (magnitude >= down_left))
        | (vertical & (magnitude >= up) & (magnitude >= down))
        | (diagonal_down & (magnitude >= up_left) & (magnitude >= down_right))
    )
    return np.where(keep, magnitude, 0.0)


def _hysteresis(magnitude, low, high):
    """Keep weak edges only where they connect to a strong one (8-connected)."""
    strong = magnitude >= high
    weak = (magnitude >= low) & ~strong
    edges = strong.copy()
    height, width = magnitude.shape
    stack = list(zip(*np.nonzero(strong)))
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width and weak[ny, nx]:
                    weak[ny, nx] = False
                    edges[ny, nx] = True
                    stack.append((ny, nx))
    return edges
