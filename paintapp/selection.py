"""Selection regions for the canvas.

A selection is a hard-edged 8-bit mask plus an outline used to draw the
on-screen marching ants. The rectangular marquee, the freeform lasso and the
magic wand all build the same object, so every selection-aware action (move,
cut, copy, paste, delete, invert, grow, feather) works identically for any of
them.

Boxes use ``(left, top, right, bottom)`` with the right/bottom edges exclusive,
matching :meth:`PIL.Image.Image.crop`.

A selection made from a mask (the wand, or a grow/shrink/invert/feather result)
has no meaningful polygon, so it carries ``edges`` - the boundary line segments
in document pixels - and the view draws the ants from those instead.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import gpuops

# Above this many boundary segments the outline is drawn from the bounding box
# instead, so a noisy wand selection on a photo cannot flood Tk with line items.
MAX_EDGES = 8000

# Arc segments per rounded corner when approximating the marquee outline.
CORNER_STEPS = 6


def _apply(matrix, x, y):
    """Map a point through an affine matrix ``(a, b, c, d, e, f)``."""
    a, b, c, d, e, f = matrix
    return (a * x + c * y + e, b * x + d * y + f)


def _invert(matrix):
    """Return the inverse of an affine matrix."""
    a, b, c, d, e, f = matrix
    det = a * d - b * c
    if abs(det) < 1e-12:
        det = 1e-12
    return (
        d / det,
        -b / det,
        -c / det,
        a / det,
        (c * f - d * e) / det,
        (b * e - a * f) / det,
    )


def corner_radius(box, radius):
    """Clamp ``radius`` to what ``box`` ``(left, top, right, bottom)`` allows."""
    span = min(abs(box[2] - box[0]), abs(box[3] - box[1])) / 2.0
    return max(0.0, min(float(radius), span))


def rounded_polygon(box, radius, steps=CORNER_STEPS):
    """Outline points of a rounded rectangle, clockwise from the top-left arc."""
    left, top, right, bottom = box
    r = radius
    corners = (
        (left + r, top + r, math.pi, 1.5 * math.pi),
        (right - r, top + r, 1.5 * math.pi, 2.0 * math.pi),
        (right - r, bottom - r, 0.0, 0.5 * math.pi),
        (left + r, bottom - r, 0.5 * math.pi, math.pi),
    )
    points = []
    for cx, cy, start, end in corners:
        for step in range(steps + 1):
            angle = start + (end - start) * step / steps
            points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    return points


def mask_edges(mask):
    """Boundary segments of a binary mask, as ``((x1, y1), (x2, y2))`` pairs.

    Returns ``None`` when the mask is empty or has too many segments to draw
    cheaply (the caller then falls back to the bounding box).
    """
    arr = np.asarray(mask) > 127
    if not arr.any():
        return None
    height, width = arr.shape
    segments = []
    left = np.zeros((height, width + 1), dtype=bool)
    right = np.zeros((height, width + 1), dtype=bool)
    left[:, 1:] = arr
    right[:, :width] = arr
    ys, xs = np.nonzero(left != right)
    for y, x in zip(ys.tolist(), xs.tolist()):
        segments.append(((x, y), (x, y + 1)))
        if len(segments) > MAX_EDGES:
            return None
    top = np.zeros((height + 1, width), dtype=bool)
    bottom = np.zeros((height + 1, width), dtype=bool)
    top[1:, :] = arr
    bottom[:height, :] = arr
    ys, xs = np.nonzero(top != bottom)
    for y, x in zip(ys.tolist(), xs.tolist()):
        segments.append(((x, y), (x + 1, y)))
        if len(segments) > MAX_EDGES:
            return None
    return segments


class Selection:
    def __init__(self, size, mask, polygon, edges=None, rect=False):
        self.size = size
        self.mask = mask
        self.polygon = polygon
        self.edges = edges
        # True for the rectangular marquee, so its bounding-box corners can be
        # dragged to resize it directly (see the Select tool's rect mode).
        self.rect = rect
        self._boundary = None

    @property
    def bbox(self):
        """``(left, top, right, bottom)`` of the mask, or ``None`` when empty."""
        return self.mask.getbbox()

    def is_empty(self):
        return self.mask.getbbox() is None

    def contains(self, x, y):
        width, height = self.size
        if not (0 <= x < width and 0 <= y < height):
            return False
        return self.mask.getpixel((x, y)) > 0

    def outline(self):
        """Outline points in document pixels, for the marching-ants overlay."""
        return list(self.polygon)

    def outline_segments(self):
        """Boundary segments when the selection came from a mask, else ``None``."""
        return self.edges

    def boundary_image(self):
        """A doc-sized RGBA outline (white line, black halo) for mask selections.

        Cached on the selection so a repaint does not re-rasterise thousands of
        segments; the view scales it to the viewport like the mask preview.
        """
        if self._boundary is None:
            mask = Image.new("L", self.size, 0)
            if self.edges:
                pen = ImageDraw.Draw(mask)
                for (x1, y1), (x2, y2) in self.edges:
                    pen.line((x1, y1, x2, y2), fill=255)
            halo = gpuops.dilate(mask, 1)
            if halo is None:
                halo = mask.filter(ImageFilter.MaxFilter(3))
            image = Image.new("RGBA", self.size, (0, 0, 0, 0))
            image.paste(Image.new("RGBA", self.size, (0, 0, 0, 255)), (0, 0), halo)
            image.paste(Image.new("RGBA", self.size, (255, 255, 255, 255)),
                        (0, 0), mask)
            self._boundary = image
        return self._boundary

    def moved(self, dx, dy):
        """Return a copy translated by ``(dx, dy)``."""
        mask = Image.new("L", self.size, 0)
        mask.paste(self.mask, (dx, dy))
        if self.edges is not None:
            return Selection.from_mask(self.size, mask)
        polygon = [(x + dx, y + dy) for x, y in self.polygon]
        return Selection(self.size, mask, polygon, rect=self.rect)

    def transformed(self, matrix):
        """Return a copy mapped through the affine ``matrix``.

        ``matrix`` maps coordinates in the selection's own bounding box (its
        top-left is the local origin) into document pixels, matching the
        transform gizmo. Only the selection's bounding box is resampled, so a
        large canvas stays cheap to drag.
        """
        box = self.bbox
        if box is None:
            return self
        source = self.mask.crop(box)
        width, height = source.size
        mapped = [
            _apply(matrix, u, v)
            for u, v in (
                (0.0, 0.0), (float(width), 0.0),
                (float(width), float(height)), (0.0, float(height)),
            )
        ]
        doc_w, doc_h = self.size
        x0 = max(0, math.floor(min(p[0] for p in mapped)))
        y0 = max(0, math.floor(min(p[1] for p in mapped)))
        x1 = min(doc_w, math.ceil(max(p[0] for p in mapped)))
        y1 = min(doc_h, math.ceil(max(p[1] for p in mapped)))
        out = Image.new("L", self.size, 0)
        if x1 > x0 and y1 > y0:
            inv = _invert(matrix)
            # PIL data maps output pixels back to source pixels; shift the
            # inverse to the tile's document origin ``(x0, y0)``.
            data = (
                inv[0], inv[2], inv[0] * x0 + inv[2] * y0 + inv[4],
                inv[1], inv[3], inv[1] * x0 + inv[3] * y0 + inv[5],
            )
            tile = source.transform(
                (x1 - x0, y1 - y0), Image.AFFINE, data,
                resample=Image.BILINEAR,
            )
            out.paste(tile, (x0, y0))
        if self.edges is None:
            # Keep the marching-ants polygon so the marquee keeps its dashed
            # style; a mask-boundary selection would draw as a solid outline.
            polygon = [
                _apply(matrix, x - box[0], y - box[1])
                for x, y in self.polygon
            ]
            return Selection(self.size, out, polygon, rect=self.rect)
        return Selection.from_mask(self.size, out)

    @classmethod
    def rectangle(cls, size, box, radius=0):
        left, top, right, bottom = box
        mask = Image.new("L", size, 0)
        polygon = [
            (left, top), (right, top), (right, bottom), (left, bottom),
        ]
        if right > left and bottom > top:
            corner = corner_radius(box, radius)
            pen = ImageDraw.Draw(mask)
            if corner > 0:
                pen.rounded_rectangle(
                    (left, top, right - 1, bottom - 1), radius=corner, fill=255
                )
                polygon = rounded_polygon(box, corner)
            else:
                pen.rectangle((left, top, right - 1, bottom - 1), fill=255)
        return cls(size, mask, polygon, rect=True)

    @classmethod
    def lasso(cls, size, points):
        points = list(points)
        mask = Image.new("L", size, 0)
        if len(points) >= 3:
            ImageDraw.Draw(mask).polygon(points, fill=255)
        return cls(size, mask, points)

    @classmethod
    def from_mask(cls, size, mask):
        """Build a selection from an arbitrary 8-bit mask (wand and ops)."""
        box = mask.getbbox()
        if box is None:
            return cls(size, mask, [], None)
        left, top, right, bottom = box
        polygon = [
            (left, top), (right, top), (right, bottom), (left, bottom),
        ]
        return cls(size, mask, polygon, mask_edges(mask))

    # ------------------------------------------------------------ operations
    def inverted(self):
        """Everything outside the current selection, within the canvas."""
        return Selection.from_mask(self.size, self.mask.point(
            lambda value: 255 - value))

    def grown(self, radius):
        """Dilate the selection by ``radius`` pixels."""
        radius = max(1, min(20, int(round(radius))))
        grown = gpuops.dilate(self.mask, radius)
        if grown is None:
            grown = self.mask.filter(ImageFilter.MaxFilter(radius * 2 + 1))
        return Selection.from_mask(self.size, grown)

    def shrunk(self, radius):
        """Erode the selection by ``radius`` pixels."""
        radius = max(1, min(20, int(round(radius))))
        shrunk = gpuops.erode(self.mask, radius)
        if shrunk is None:
            shrunk = self.mask.filter(ImageFilter.MinFilter(radius * 2 + 1))
        return Selection.from_mask(self.size, shrunk)

    def feathered(self, radius):
        """Soften the selection's edge by ``radius`` pixels."""
        radius = max(0.1, float(radius))
        soft = self.mask.filter(ImageFilter.GaussianBlur(radius))
        return Selection.from_mask(self.size, soft)
