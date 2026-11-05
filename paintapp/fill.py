"""Flood-fill engine behind the paint bucket tool.

The bucket reasons about a layer's own pixels. From the clicked pixel it grows a
coverage mask across the neighbouring pixels whose colour is within
``tolerance`` of the seed, or - when ``contiguous`` is off - across every such
pixel in the image. The mask is then handed to :func:`brush.composite_color`, so
a fill obeys the tool's opacity and blend mode exactly like a brush stroke.

Colour matching uses numpy because it is already a hard dependency of the app's
engine; the contiguous pass is a scanline fill over boolean masks, so a region
costs a few visits per pixel rather than a Python object per pixel.
"""

import numpy as np
from PIL import Image, ImageChops, ImageFilter

from . import brush, gpuops

# Tolerance that compares all four channels of an RGBA layer, so a fill of a
# bare transparent area (alpha 0) does not bleed into painted pixels that only
# differ in alpha.
_MATCH_CHANNELS = 4

# Gaussian radius used to anti-alias a region's hard 0/255 coverage. One pixel
# is enough to ramp the edge without visibly feathering the fill.
_ANTIALIAS_RADIUS = 1.0


def _match_mask(image, seed, tolerance):
    """Boolean mask of pixels whose colour is within ``tolerance`` of ``seed``."""
    data = np.asarray(image)
    x, y = seed
    target = data[y, x].astype(np.int16)
    height, width = data.shape[:2]
    match = np.ones((height, width), dtype=bool)
    for band in range(min(_MATCH_CHANNELS, data.shape[2])):
        channel = data[:, :, band].astype(np.int16)
        match &= np.abs(channel - int(target[band])) <= tolerance
    return match


def _contiguous(match, seed):
    """Scanline flood fill: ``match`` pixels connected to ``seed`` (4-way)."""
    height, width = match.shape
    filled = np.zeros((height, width), dtype=bool)
    sx, sy = int(seed[0]), int(seed[1])
    if not (0 <= sx < width and 0 <= sy < height) or not match[sy, sx]:
        return filled
    stack = [(sx, sy)]
    while stack:
        x, y = stack.pop()
        if filled[y, x] or not match[y, x]:
            continue
        row = match[y]
        marked = filled[y]
        left = x
        while left > 0 and row[left - 1] and not marked[left - 1]:
            left -= 1
        right = x
        while right < width - 1 and row[right + 1] and not marked[right + 1]:
            right += 1
        marked[left:right + 1] = True
        # Seed one run per unvisited span on the rows above and below.
        for ny in (y - 1, y + 1):
            if ny < 0 or ny >= height:
                continue
            span = match[ny, left:right + 1] & ~filled[ny, left:right + 1]
            runs = np.flatnonzero(span)
            if runs.size == 0:
                continue
            starts = runs[np.concatenate(([True], np.diff(runs) > 1))]
            for run in starts:
                stack.append((left + int(run), ny))
    return filled


def _antialias_mask(mask):
    """Soften a hard ``"L"`` coverage mask to a one-pixel edge ramp.

    The mask is blurred and then clipped back to the region, so an edge pixel
    picks up the fractional coverage of the surrounding area while the fill
    still cannot bleed past the matched region.
    """
    softened = mask.filter(ImageFilter.GaussianBlur(_ANTIALIAS_RADIUS))
    return ImageChops.multiply(softened, mask)


def _nearest_open(opened, seed, radius):
    """Nearest surviving core pixel to ``seed``, or ``None`` if none is near.

    The search is bounded to ``radius`` so it only reaches the core of the same
    region; a region thinner than the seal has no core and reports ``None``.
    """
    sx, sy = seed
    height, width = opened.shape
    x0, x1 = max(0, sx - radius), min(width, sx + radius + 1)
    y0, y1 = max(0, sy - radius), min(height, sy + radius + 1)
    window = opened[y0:y1, x0:x1]
    if not window.any():
        return None
    ys, xs = np.nonzero(window)
    distance = (ys + y0 - sy) ** 2 + (xs + x0 - sx) ** 2
    nearest = int(np.argmin(distance))
    return int(xs[nearest] + x0), int(ys[nearest] + y0)


def _close_gap(match, seed, radius):
    """Flood fill ``match`` with breaks up to ``2 * radius`` pixels sealed.

    The barrier (non-matching pixels) is dilated so a small break in an outline
    no longer lets the region leak outside. The fill is then grown back to the
    true edge and clipped to the original match, so it reaches the line without
    crossing the sealed gap.
    """
    size = 2 * int(radius) + 1
    source = Image.fromarray(np.where(match, 255, 0).astype(np.uint8))
    eroded = gpuops.erode(source, int(radius))
    if eroded is None:
        eroded = source.filter(ImageFilter.MinFilter(size))
    opened = np.asarray(eroded) > 0
    sx, sy = int(seed[0]), int(seed[1])
    height, width = match.shape
    if not (0 <= sx < width and 0 <= sy < height) or not match[sy, sx]:
        return np.zeros_like(match)
    if not opened[sy, sx]:
        # The click sits within ``radius`` of an edge and was sealed away; start
        # from the nearest core pixel so the bucket fills the same region.
        near = _nearest_open(opened, (sx, sy), int(radius))
        if near is None:
            return _contiguous(match, seed)
        seed = near
    region = _contiguous(opened, seed)
    region_image = Image.fromarray(np.where(region, 255, 0).astype(np.uint8))
    dilated = gpuops.dilate(region_image, int(radius))
    if dilated is None:
        dilated = region_image.filter(ImageFilter.MaxFilter(size))
    grown = np.asarray(dilated) > 0
    return grown & match


def region_mask(image, seed, tolerance=0, contiguous=True, bounds=None,
                antialias=False, gap=0):
    """Coverage mask (``"L"``) of the pixels a bucket click at ``seed`` fills.

    ``bounds`` optionally restricts the result, e.g. to the active selection.
    With ``antialias`` the region's edge is softened to partial coverage. With
    ``gap`` (contiguous fills only) breaks in the outline up to ``2 * gap``
    pixels wide are sealed so the fill stays inside them.
    """
    match = _match_mask(image, seed, tolerance)
    gap = max(0, int(gap))
    if contiguous:
        region = _close_gap(match, seed, gap) if gap else _contiguous(match, seed)
    else:
        region = match
    mask = Image.fromarray(np.where(region, 255, 0).astype(np.uint8))
    if antialias:
        mask = _antialias_mask(mask)
    if bounds is not None:
        mask = ImageChops.multiply(mask, bounds)
    return mask


def apply_fill(layer, color, coverage, opacity=100, blend="normal"):
    """Composite a flat ``color`` into ``layer`` through ``coverage``.

    ``layer``'s pixels are replaced with a new image, so a history snapshot that
    still references the old object stays intact (copy-on-write).
    """
    if opacity < 100:
        factor = max(1, min(100, int(round(opacity)))) / 100.0
        coverage = coverage.point(lambda value: int(value * factor))
    layer.set_image(brush.composite_color(layer.image, color, coverage, blend))
