"""Exploded-grid view layout for a canvas's layer stack.

The exploded view shows a document's layers side by side in a grid instead of
composited on top of one another, so every layer occupies its own cell and can
be edited on its own. This module only computes *where* each layer sits in that
grid: a :class:`Layout` holds, per layer (in stack order):

* ``cells`` - the ``(left, top, right, bottom)`` rectangle the layer's cell
  occupies in layout space, and
* ``offsets`` - the position of the layer image's top-left corner in layout
  space (the cell's top-left), so ``layer.offset`` and the view's coordinate
  shift can be derived.

Each cell is as large as the largest layer *image* in the stack, and every layer
image is placed at its cell's top-left. Because a layer can only ever hold
pixels inside its own image, a cell that is at least that big can never clip a
layer's content, which keeps painting free of surprises; the layout therefore
depends only on the layers' buffer sizes, not on their pixels, so it never
changes while a stroke is in progress. The document itself is never changed - the
layout is a view-time arrangement.

The grid is uniform so the stack reads as a tidy board: cells run left-to-right
then top-to-bottom in stack order (bottom layer first).
"""

import math

# Gap between cells, in document pixels.
GAP = 40


class Layout:
    """The grid arrangement of a canvas's layers, in layout space."""

    def __init__(self, cells, offsets, columns, size):
        self.cells = cells
        self.offsets = offsets
        self.columns = columns
        self.origin = (0, 0)
        self.size = size

    def cell_at(self, x, y):
        """The layer index whose cell contains layout point ``(x, y)``, or ``None``."""
        for index, (left, top, right, bottom) in enumerate(self.cells):
            if left <= x < right and top <= y < bottom:
                return index
        return None


def compute(canvas, gap=GAP, columns=None):
    """Build the exploded-grid :class:`Layout` for ``canvas``.

    ``columns`` forces the number of columns; ``None`` picks a roughly square
    grid (``ceil(sqrt(n))``).
    """
    layers = list(canvas.layers)
    count = len(layers)
    widths = [max(1, int(layer.image.width)) for layer in layers]
    heights = [max(1, int(layer.image.height)) for layer in layers]
    cell_w = max(widths, default=1)
    cell_h = max(heights, default=1)
    if columns is None:
        columns = max(1, int(math.ceil(math.sqrt(count)))) if count else 1
    columns = max(1, int(columns))
    if count:
        columns = min(columns, count)
    rows = (count + columns - 1) // columns if count else 0

    cells = []
    offsets = []
    for index in range(count):
        column = index % columns
        row = index // columns
        x0 = column * (cell_w + gap)
        y0 = row * (cell_h + gap)
        cells.append((x0, y0, x0 + cell_w, y0 + cell_h))
        offsets.append((x0, y0))

    width = columns * (cell_w + gap) - gap if count else 0
    height = rows * (cell_h + gap) - gap if rows else 0
    return Layout(cells, offsets, columns, (max(0, width), max(0, height)))