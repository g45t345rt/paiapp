"""Control-image preprocessors for ControlNet.

A ControlNet does not read a photograph directly: it is trained on a "control
hint" -- a canny edge map, a line drawing, and so on -- and consumes it through
its own conditioning embedding. These helpers turn the app's pixels (a canvas,
a layer or an asset) into such a hint.

Everything here is a pure ``numpy`` / ``Pillow`` image operation, so no model
download or extra dependency is needed. A hint that already exists -- a depth,
pose or segmentation map produced elsewhere -- is passed through unchanged with
the ``none`` preprocessor, so those control types work too as long as the user
supplies the map. Model-backed preprocessors (depth, openpose, segmentation) are
deliberately not included: they need weights the paint engine does not ship.
:data:`PREPROCESSORS` lists the names in display order and :func:`prepare` is
the single entry point the pipeline calls.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

# The available preprocessors, in display order. ``none`` is a plain resize, so
# it doubles as "the control image is already a hint". The rest are pure image
# operations that approximate the named annotator; ``tile`` is the identity
# resize a Tile ControlNet expects.
PREPROCESSORS = (
    "none", "canny", "lineart", "softedge", "sketch", "scribble", "mlsd",
    "normal", "threshold", "invert", "tile",
)

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = _SOBEL_X.T.copy()

# Canny's double threshold on the 0..255 gradient magnitude and the blur applied
# before the gradients, matching the usual ControlNet canny recipe.
_CANNY_LOW = 100.0
_CANNY_HIGH = 200.0
_CANNY_SIGMA = 1.4

# The normalize map scales the Sobel gradient into a tangent-space normal. The
# value is tuned so flat areas stay near (128, 128, 255) while edges tilt.
_NORMAL_STRENGTH = 2.0

# MLSD detects straight lines on a working copy capped at this long edge, so the
# Hough accumulator stays small on a 2048px canvas. The lines are mapped back to
# the requested output size.
_MLSD_WORKING = 512
_MLSD_MAX_LINES = 48
_MLSD_MIN_VOTES = 12
_MLSD_PEAK_WINDOW = 4


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
    if name == "softedge":
        return _softedge(rgb)
    if name == "sketch":
        return _sketch(rgb)
    if name == "scribble":
        return _scribble(rgb)
    if name == "mlsd":
        return _mlsd(rgb)
    if name == "normal":
        return _normal(rgb)
    if name == "threshold":
        return _threshold(rgb)
    if name == "invert":
        return ImageOps.invert(rgb)
    return rgb


def _canny(image):
    """A Canny edge map: white edges on black, as the ControlNet expects."""
    return _binary(_canny_mask(image))


def _canny_mask(image):
    """The boolean Canny edge map ``image`` produces at its own size."""
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
    return _hysteresis(suppressed, _CANNY_LOW, _CANNY_HIGH)


def _binary(mask):
    """A boolean mask as the white-on-black RGB image ControlNets expect."""
    return Image.fromarray((mask * 255).astype(np.uint8), "L").convert("RGB")


def _lineart(image):
    """A soft line drawing: dark lines on white, from a blurred edge filter."""
    gray = image.convert("L").filter(ImageFilter.GaussianBlur(1.0))
    return ImageOps.invert(gray.filter(ImageFilter.FIND_EDGES)).convert("RGB")


def _softedge(image):
    """A soft, non-thresholded edge map: white ridges on black.

    Unlike ``canny`` the gradient magnitude is kept continuous and gamma-boosted,
    which is the softer hint a SoftEdge ControlNet is usually paired with.
    """
    gray = np.asarray(
        image.convert("L").filter(ImageFilter.GaussianBlur(1.2)), dtype=np.float32,
    )
    gx = _convolve(gray, _SOBEL_X)
    gy = _convolve(gray, _SOBEL_Y)
    magnitude = np.hypot(gx, gy)
    peak = float(magnitude.max())
    if peak <= 0.0:
        return Image.new("RGB", image.size, (0, 0, 0))
    soft = np.clip(magnitude / peak * 2.5, 0.0, 1.0) ** 0.7
    return Image.fromarray((soft * 255.0).astype(np.uint8), "L").convert("RGB")


def _sketch(image):
    """A pencil sketch: dark strokes on white, from a colour-dodge blend."""
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    inverted = 255.0 - gray
    blurred = np.asarray(
        Image.fromarray(inverted.astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(5.0)
        ),
        dtype=np.float32,
    )
    # Colour dodge of the grey base over the blurred negative keeps strokes dark
    # and the paper white; the epsilon avoids a divide by zero on pure white.
    dodge = gray * 255.0 / np.maximum(255.0 - blurred, 1.0)
    dodge = np.clip(dodge, 0.0, 255.0)
    return Image.fromarray(dodge.astype(np.uint8), "L").convert("RGB")


def _scribble(image):
    """A sparse ink scribble: black strokes on white.

    A difference-of-Gaussians keeps the strong, hand-drawn-looking contours and
    drops the low-frequency tone a full line drawing would keep. Only pixels
    darker than the local mean survive, so the result stays sparse.
    """
    gray = image.convert("L")
    narrow = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32,
    )
    wide = np.asarray(
        gray.filter(ImageFilter.GaussianBlur(2.5)), dtype=np.float32,
    )
    dog = narrow - wide
    cutoff = dog.mean() - 0.35 * (dog.std() + 1e-6)
    ink = dog < cutoff
    # A one-pixel dilation keeps thin strokes visible after the ControlNet's
    # own downsampling. MaxFilter pads with the image edge, so no stroke wraps
    # around to the opposite border.
    dilated = np.asarray(
        Image.fromarray((ink * 255).astype(np.uint8), "L").filter(
            ImageFilter.MaxFilter(3)
        ),
        dtype=np.uint8,
    ) > 0
    return _binary(~dilated)


def _mlsd(image):
    """Straight-line detection: white lines on black, like an MLSD hint.

    A Hough transform finds the dominant lines on a capped working copy; the
    endpoints are then mapped back to the requested size before drawing, so the
    output stays one pixel wide at any resolution.
    """
    width, height = image.size
    scale = min(1.0, _MLSD_WORKING / float(max(width, height)))
    work_w = max(1, int(round(width * scale)))
    work_h = max(1, int(round(height * scale)))
    working = image.convert("RGB").resize((work_w, work_h), Image.LANCZOS)
    edges = _canny_mask(working)

    canvas = Image.new("L", (width, height), 0)
    picks = _hough_peaks(edges)
    if not picks:
        return canvas.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    sx = width / float(work_w)
    sy = height / float(work_h)
    reach = float(max(work_w, work_h)) * 2.0
    for rho, theta in picks:
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        x0 = rho * cos_t
        y0 = rho * sin_t
        dx = -reach * sin_t
        dy = reach * cos_t
        draw.line(
            ((x0 + dx) * sx, (y0 + dy) * sy, (x0 - dx) * sx, (y0 - dy) * sy),
            fill=255, width=1,
        )
    return canvas.convert("RGB")


def _hough_peaks(edges):
    """The dominant ``(rho, theta)`` lines in a boolean edge map."""
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return []
    height, width = edges.shape
    diagonal = float(np.hypot(width, height))
    offset = int(np.ceil(diagonal)) + 1
    rho_bins = 2 * offset + 1
    thetas = np.deg2rad(np.arange(180.0))
    accum = np.zeros((rho_bins, 180), dtype=np.int32)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    for index in range(180):
        rho = np.rint(xs * cos_t[index] + ys * sin_t[index]).astype(np.int64)
        accum[:, index] = np.bincount(
            rho + offset, minlength=rho_bins,
        )[:rho_bins]

    flat = accum.ravel()
    peak = int(flat.max())
    if peak < _MLSD_MIN_VOTES:
        return []
    threshold = max(_MLSD_MIN_VOTES, int(peak * 0.25))
    used = np.zeros_like(accum, dtype=bool)
    picks = []
    for index in np.argsort(flat)[::-1]:
        if len(picks) >= _MLSD_MAX_LINES or int(flat[index]) < threshold:
            break
        rho_bin, theta_bin = divmod(int(index), 180)
        if used[rho_bin, theta_bin]:
            continue
        picks.append((rho_bin - offset, float(thetas[theta_bin])))
        lo_rho = max(0, rho_bin - _MLSD_PEAK_WINDOW)
        hi_rho = min(rho_bins, rho_bin + _MLSD_PEAK_WINDOW + 1)
        lo_theta = max(0, theta_bin - _MLSD_PEAK_WINDOW)
        hi_theta = min(180, theta_bin + _MLSD_PEAK_WINDOW + 1)
        used[lo_rho:hi_rho, lo_theta:hi_theta] = True
    return picks


def _normal(image):
    """A tangent-space normal map derived from the image's intensity gradient."""
    gray = np.asarray(
        image.convert("L").filter(ImageFilter.GaussianBlur(1.0)), dtype=np.float32,
    ) / 255.0
    gx = _convolve(gray, _SOBEL_X) * _NORMAL_STRENGTH
    gy = _convolve(gray, _SOBEL_Y) * _NORMAL_STRENGTH
    ones = np.ones_like(gray)
    length = np.sqrt(gx * gx + gy * gy + ones)
    nx = -gx / length
    ny = -gy / length
    nz = ones / length
    rgb = np.stack((nx, ny, nz), axis=-1) * 0.5 + 0.5
    return Image.fromarray(
        np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8), "RGB",
    )


def _threshold(image):
    """A binary black/white map split at Otsu's threshold."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    cutoff = _otsu(gray)
    return _binary(gray > cutoff)


def _otsu(gray):
    """Otsu's threshold for an 8-bit grayscale array."""
    histogram = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = float(gray.size)
    if total <= 0.0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    weight_bg = np.cumsum(histogram)
    weight_fg = total - weight_bg
    sum_total = float((histogram * levels).sum())
    sum_bg = np.cumsum(histogram * levels)
    mean_bg = np.divide(
        sum_bg, weight_bg, out=np.zeros_like(sum_bg), where=weight_bg > 0,
    )
    mean_fg = np.divide(
        sum_total - sum_bg, weight_fg,
        out=np.zeros_like(sum_bg), where=weight_fg > 0,
    )
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    return int(np.argmax(between))


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
