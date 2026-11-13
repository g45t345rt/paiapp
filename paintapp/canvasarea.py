"""The canvas tab strip and the rendered view of each document."""

import math
import tkinter as tk
from tkinter import font as tkfont, ttk

from PIL import Image, ImageDraw, ImageTk

from . import colormanagement, shape, symmetry, theme
from .constants import CANVAS_W, CANVAS_H, MAX_CANVAS_SIZE
from .dialogs import ask_color, center_window
from .sizeinput import SizeInput
from .tabstrip import TabStrip

CHECKER_SQUARE = 8
CHECKER_LIGHT = (255, 255, 255, 255)
CHECKER_DARK = (204, 204, 204, 255)
_CHECKER_TILE = None

DEFAULT_OFFCANVAS_BG = (128, 128, 128)

# Ruler strip geometry and the "nice" labelled spacings (document pixels) the
# top/left rulers snap to so labels stay readable at every zoom level.
RULER_SIZE = 20
RULER_MIN_SPACING = 56
RULER_STEPS = (1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500,
               1000, 2000, 2500, 5000, 10000)

# Colour fallback for ruler guides when the theme provides none.
DEFAULT_GUIDE_COLOR = "#00b0f0"

# Guide handles drawn on the ruler: their length along the ruler, how far they
# reach in from the canvas-facing edge, and the grab tolerance in pixels.
GUIDE_MARKER_LEN = 9
GUIDE_MARKER_DEPTH = 7
GUIDE_MARKER_HIT = 6

# Colour fallback for the grid overlay when the theme provides none.
DEFAULT_GRID_COLOR = "#7f7f7f"

# Smallest on-screen grid spacing (device pixels) at which lines stay readable.
# Below it the overlay is skipped so it does not turn the canvas into a tint.
GRID_MIN_PIXELS = 4

# Crop-tool cross-hatch: diagonal line spacing and opacity (device pixels/alpha)
# for the shading composited over the area outside the crop box.
CROP_HATCH_SPACING = 7
CROP_HATCH_ALPHA = 120

# Default grid cell size in document pixels.
DEFAULT_GRID_SIZE = 16

# How long a window-resize drag must settle before the viewport is re-rasterized.
# Dragging a window edge fires a <Configure> per mouse move, and each render
# flattens and resamples the viewport, so repainting per event saturates the UI
# thread - the laggy resize. The frame already on screen is drawn in document
# coordinates and so stays correctly positioned, so it is left untouched while
# the drag runs and the expensive render is deferred to the trailing edge of the
# burst (see :meth:`CanvasView._schedule_render`).
RESIZE_RENDER_DELAY_MS = 140


def rgb_to_hex(rgb):
    """Convert an ``(r, g, b)`` tuple to a Tk ``#rrggbb`` color string."""
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)


def checker_tile():
    """Return a cached RGBA checkerboard tile of ``CHECKER_SQUARE`` squares."""
    global _CHECKER_TILE
    if _CHECKER_TILE is None:
        square = CHECKER_SQUARE
        tile = Image.new("RGBA", (square * 2, square * 2), CHECKER_LIGHT)
        dark = Image.new("RGBA", (square, square), CHECKER_DARK)
        tile.paste(dark, (square, 0))
        tile.paste(dark, (0, square))
        _CHECKER_TILE = tile
    return _CHECKER_TILE


def checkerboard(size):
    """Build an RGBA checkerboard pattern used to show transparent areas."""
    square = CHECKER_SQUARE
    tile = Image.new("RGBA", (square * 2, square * 2), CHECKER_LIGHT)
    dark = Image.new("RGBA", (square, square), CHECKER_DARK)
    tile.paste(dark, (square, 0))
    tile.paste(dark, (0, square))
    width, height = size
    board = Image.new("RGBA", size)
    for y in range(0, height, square * 2):
        for x in range(0, width, square * 2):
            board.paste(tile, (x, y))
    return board


def checker_region(size, phase):
    """An opaque RGBA checkerboard of ``size``, aligned so a light square's
    top-left corner sits at ``(-phase[0], -phase[1])``.

    The cached tile is repeated with byte multiplication instead of pasting one
    tile at a time, so filling a whole viewport costs a few C-level copies
    rather than thousands of Python ``alpha_composite`` calls (which made
    window resizing lag).
    """
    width, height = max(1, int(size[0])), max(1, int(size[1]))
    period = CHECKER_SQUARE * 2
    off_x = int(phase[0]) % period
    off_y = int(phase[1]) % period
    cols = (width + off_x + period - 1) // period + 1
    rows = (height + off_y + period - 1) // period + 1
    raw = checker_tile().tobytes()
    stride = period * 4
    # One band of ``period`` rows, each tiled ``cols`` times across the width;
    # repeating the band downwards then completes the 2D tiling.
    band = b"".join(
        raw[r * stride:(r + 1) * stride] * cols for r in range(period)
    )
    full = Image.frombytes("RGBA", (cols * period, rows * period), band * rows)
    return full.crop((off_x, off_y, off_x + width, off_y + height))


def fade_canvas_edges(board, canvas, bg):
    """Anti-alias the fractional canvas border of an RGB preview.

    The checkerboard and document rectangles are rounded outwards to whole
    pixels, so a sub-pixel sliver of their content bleeds past the canvas edge
    onto the off-canvas background. Fading every border pixel that straddles the
    edge toward ``bg`` by its exact canvas coverage removes that sliver and
    anti-aliases the edge in one pass. Only pixels on the four canvas edges are
    touched, so the cost is ``O(width + height)`` rather than a full-buffer mask.
    """
    cx0, cy0, cx1, cy1 = canvas
    vw, vh = board.size
    if cx0 >= cx1 or cy0 >= cy1 or cx1 <= 0 or cy1 <= 0 or cx0 >= vw or cy0 >= vh:
        return
    bg_rgb = tuple(bg)

    def fade(box, factor):
        if not 0.0 < factor < 1.0 or box[2] <= box[0] or box[3] <= box[1]:
            return
        strip = board.crop(box)
        board.paste(
            Image.blend(Image.new("RGB", strip.size, bg_rgb), strip, factor),
            (box[0], box[1]),
        )

    # Span of pixels fully inside the canvas on each axis; the fractional
    # columns/rows outside it straddle the edge.
    ix0, ix1 = max(0, math.ceil(cx0)), min(vw, math.floor(cx1))
    iy0, iy1 = max(0, math.ceil(cy0)), min(vh, math.floor(cy1))
    cols = [(x, min(x + 1, cx1) - max(x, cx0))
            for x in (math.floor(cx0), math.floor(cx1))
            if 0 <= x < vw and 0.0 < min(x + 1, cx1) - max(x, cx0) < 1.0]
    rows = [(y, min(y + 1, cy1) - max(y, cy0))
            for y in (math.floor(cy0), math.floor(cy1))
            if 0 <= y < vh and 0.0 < min(y + 1, cy1) - max(y, cy0) < 1.0]
    for x, cover in cols:
        fade((x, iy0, x + 1, iy1), cover)
    for y, cover in rows:
        fade((ix0, y, ix1, y + 1), cover)
    # The edge columns and rows were faded over the interior only, so their
    # intersections are still untouched and need the product of both coverages.
    for x, fx in cols:
        for y, fy in rows:
            fade((x, y, x + 1, y + 1), fx * fy)


def visible_crop(image, origin, zoom, top_left, view_size, source_scale=1.0):
    """Crop ``image`` to the region visible in a viewport of ``view_size``.

    ``top_left`` is the document coordinate shown at the viewport's top-left;
    ``origin`` is the document coordinate of ``image``'s top-left, and
    ``source_scale`` is ``image``'s pixels per document pixel (a view that
    composites at display resolution passes its zoom here). Returns the scaled
    crop and the canvas coordinate to place it at, or ``(None, item)`` when the
    image lies entirely outside the viewport.
    """
    ox, oy = origin
    tlx, tly = top_left
    vw, vh = view_size
    ss = source_scale if source_scale > 0 else 1.0
    sx0 = max(0, math.floor((tlx - ox) * ss))
    sy0 = max(0, math.floor((tly - oy) * ss))
    sx1 = min(image.width, math.ceil((tlx - ox + vw / zoom) * ss))
    sy1 = min(image.height, math.ceil((tly - oy + vh / zoom) * ss))
    item = ((ox + sx0 / ss) * zoom, (oy + sy0 / ss) * zoom)
    if sx1 <= sx0 or sy1 <= sy0:
        return None, item
    if sx0 == 0 and sy0 == 0 and sx1 == image.width and sy1 == image.height:
        # The whole image is visible. Copying it through ``crop`` allocates a
        # duplicate and re-runs Pillow's decompression-bomb check, which a large
        # in-memory composite can trip even though it was never decoded.
        crop = image
    else:
        crop = image.crop((sx0, sy0, sx1, sy1))
    target = zoom / ss
    if abs(target - 1.0) > 1e-9:
        width = max(1, int(round((sx1 - sx0) * target)))
        height = max(1, int(round((sy1 - sy0) * target)))
        resample = Image.NEAREST if target > 1.0 else Image.LANCZOS
        crop = crop.resize((width, height), resample)
    return crop, item


def compose_region(composite, doc_size, origin, zoom, top_left, view_size,
                   bg=DEFAULT_OFFCANVAS_BG, source_scale=1.0, underlay=None,
                   cells=None):
    """Build the RGB preview for the visible viewport only.

    Rendering just the viewport keeps zooming independent of the document size:
    the full-document buffer would otherwise be rebuilt at ``doc_size * zoom``
    on every zoom step. ``source_scale`` is ``composite``'s pixels per document
    pixel, so a display-resolution composite is placed without a second resize.
    The transparency checkerboard is confined to the canvas rectangle, and the
    margin around it is filled with the off-canvas background color.

    ``cells`` switches to the exploded-grid presentation: it is a list of
    layout-space ``(left, top, right, bottom)`` cell rectangles, and the
    checkerboard is drawn inside each cell (so the gaps between tiles show the
    off-canvas background) and the canvas edge fade is skipped.
    """
    vw = max(1, int(view_size[0]))
    vh = max(1, int(view_size[1]))
    ox, oy = origin
    tlx, tly = top_left
    crop, item = visible_crop(
        composite, origin, zoom, top_left, (vw, vh), source_scale
    )
    if composite.mode != "RGBA":
        board = Image.new("RGB", (vw, vh), tuple(bg))
        if crop is not None:
            board.paste(crop, (int(round(item[0] - tlx * zoom)),
                               int(round(item[1] - tly * zoom))))
        fade_canvas_edges(
            board, (-tlx * zoom, -tly * zoom,
                    (doc_size[0] - tlx) * zoom, (doc_size[1] - tly) * zoom),
            bg,
        )
        return board

    if cells is not None:
        board = Image.new("RGBA", (vw, vh), tuple(bg) + (255,))
        period = CHECKER_SQUARE * 2
        for (left, top, right, bottom) in cells:
            cx0 = (left - tlx) * zoom
            cy0 = (top - tly) * zoom
            cx1 = (right - tlx) * zoom
            cy1 = (bottom - tly) * zoom
            ix0 = max(0, math.floor(cx0))
            iy0 = max(0, math.floor(cy0))
            ix1 = min(vw, math.ceil(cx1))
            iy1 = min(vh, math.ceil(cy1))
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            phase_x = int(round((ix0 - cx0) % period))
            phase_y = int(round((iy0 - cy0) % period))
            board.paste(
                checker_region((ix1 - ix0, iy1 - iy0), (phase_x, phase_y)),
                (ix0, iy0),
            )
        if crop is not None:
            board.alpha_composite(crop, (int(round(item[0] - tlx * zoom)),
                                         int(round(item[1] - tly * zoom))))
        return board.convert("RGB")

    board = Image.new("RGBA", (vw, vh), tuple(bg) + (255,))
    cx0 = (0 - tlx) * zoom
    cy0 = (0 - tly) * zoom
    cx1 = cx0 + doc_size[0] * zoom
    cy1 = cy0 + doc_size[1] * zoom
    ix0 = max(0, math.floor(cx0))
    iy0 = max(0, math.floor(cy0))
    ix1 = min(vw, math.ceil(cx1))
    iy1 = min(vh, math.ceil(cy1))
    # When the composite is opaque and covers the whole on-screen canvas, the
    # transparency checkerboard underneath it can never show. Skip building and
    # pasting it (and alpha-composite with a plain paste) - the checkerboard and
    # the alpha blend are two full-viewport passes on every zoom step.
    crop_xy = None
    opaque = False
    if crop is not None:
        crop_xy = (int(round(item[0] - tlx * zoom)),
                   int(round(item[1] - tly * zoom)))
        if (crop_xy[0] <= ix0 and crop_xy[1] <= iy0
                and crop_xy[0] + crop.width >= ix1
                and crop_xy[1] + crop.height >= iy1):
            opaque = (crop.mode != "RGBA"
                      or crop.getextrema()[3][0] == 255)
    if ix1 > ix0 and iy1 > iy0 and not opaque:
        iw, ih = ix1 - ix0, iy1 - iy0
        period = CHECKER_SQUARE * 2
        phase_x = int(round((ix0 - cx0) % period))
        phase_y = int(round((iy0 - cy0) % period))
        board.paste(checker_region((iw, ih), (phase_x, phase_y)), (ix0, iy0))
    if underlay is not None:
        # Drawn over the checkerboard and under the layers, so it reads as being
        # behind the figure (the pose's ground grid).
        board.alpha_composite(underlay, (0, 0))
    if crop is not None:
        if opaque:
            # Opaque: a plain paste replaces the pixels without the blend, and
            # no checkerboard was laid down beneath it.
            board.paste(crop, crop_xy)
        else:
            board.alpha_composite(crop, crop_xy)
    board = board.convert("RGB")
    fade_canvas_edges(board, (cx0, cy0, cx1, cy1), bg)
    return board


# Where the old canvas sits inside a resized one. The fraction is the part of
# the added space that goes before the old content: ``(0, 0)`` pins the content
# to the top left and grows right/down, ``(0.5, 0.5)`` grows evenly around it.
# The same fractions position content when the canvas is shrunk.
CANVAS_ANCHORS = (
    ("nw", 0.0, 0.0, "\u2196"),
    ("n", 0.5, 0.0, "\u2191"),
    ("ne", 1.0, 0.0, "\u2197"),
    ("w", 0.0, 0.5, "\u2190"),
    ("c", 0.5, 0.5, "\u25cf"),
    ("e", 1.0, 0.5, "\u2192"),
    ("sw", 0.0, 1.0, "\u2199"),
    ("s", 0.5, 1.0, "\u2193"),
    ("se", 1.0, 1.0, "\u2198"),
)

CANVAS_ANCHOR_FRACTIONS = {
    name: (fx, fy) for name, fx, fy, _label in CANVAS_ANCHORS
}


def prompt_resize_canvas(root, initial=(CANVAS_W, CANVAS_H),
                         title="Resize / Expand Canvas"):
    """Ask for a canvas size and where the existing content sits.

    Returns ``(width, height, anchor)`` or ``None``. ``anchor`` is one of
    :data:`CANVAS_ANCHORS` and decides where the old canvas sits inside the
    new one. The size may grow or shrink the canvas; no generation is implied.
    """
    dialog = tk.Toplevel(root)
    dialog.title(title)
    dialog.transient(root)
    dialog.resizable(False, False)
    dialog.withdraw()

    frame = tk.Frame(dialog, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    size_input = SizeInput(frame, initial[0], initial[1], field_width=10)
    size_input.grid(row=0, column=0, sticky="w", pady=2)

    anchor_frame = tk.LabelFrame(frame, text="Keep content", padx=4, pady=4)
    anchor_frame.grid(row=1, column=0, sticky="w", pady=(8, 0))
    anchor_var = tk.StringVar(value="c")
    for index, (value, _fx, _fy, label) in enumerate(CANVAS_ANCHORS):
        row, column = divmod(index, 3)
        tk.Radiobutton(
            anchor_frame, text=label, variable=anchor_var, value=value,
            indicatoron=False, width=3,
        ).grid(row=row, column=column, padx=1, pady=1)

    tk.Label(
        frame, text="Shrinking crops content outside the new canvas.",
        anchor="w",
    ).grid(row=2, column=0, sticky="w", pady=(2, 0))

    error_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=error_var, fg="red", anchor="w").grid(
        row=3, column=0, sticky="w", pady=(2, 0)
    )

    result = {"value": None}

    def on_ok(_event=None):
        width, height = size_input.values()
        if width is None or height is None:
            error_var.set("Enter whole numbers.")
            return
        if width < 1 or height < 1:
            error_var.set("Width and height must be at least 1.")
            return
        if width > MAX_CANVAS_SIZE or height > MAX_CANVAS_SIZE:
            error_var.set(f"Maximum size is {MAX_CANVAS_SIZE}.")
            return
        result["value"] = (width, height, anchor_var.get())
        dialog.destroy()

    def on_cancel(_event=None):
        dialog.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(row=4, column=0, sticky="e", pady=(8, 0))
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=on_cancel).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    dialog.bind("<Return>", on_ok)
    dialog.bind("<Escape>", on_cancel)
    dialog.protocol("WM_DELETE_WINDOW", on_cancel)

    theme.restyle(dialog)
    center_window(root, dialog)
    size_input.width_field.focus_set()
    size_input.width_field.selection_range(0, tk.END)

    dialog.grab_set()
    root.wait_window(dialog)
    return result["value"]


# Preset canvas sizes offered by the New Canvas dialog, as ``(label, (w, h))``.
# "Custom" is always available; the caller's current default is inserted as a
# labelled preset at build time when it is not already in the list.
CANVAS_PRESETS = (
    ("800 x 640", (800, 640)),
    ("1024 x 768", (1024, 768)),
    ("768 x 768", (768, 768)),
    ("1024 x 1024", (1024, 1024)),
    ("896 x 1152", (896, 1152)),
    ("1152 x 896", (1152, 896)),
    ("1280 x 720", (1280, 720)),
    ("1920 x 1080", (1920, 1080)),
    ("1080 x 1080", (1080, 1080)),
    ("1080 x 1920", (1080, 1920)),
    ("2048 x 2048", (2048, 2048)),
    # Print pages at 300 dpi (a millimetre size multiplied by 300 / 25.4), so a
    # canvas lines up with the matching export page preset. Portrait first.
    ("1748 x 2480 (A5 300dpi)", (1748, 2480)),
    ("2480 x 1748 (A5 300dpi landscape)", (2480, 1748)),
    ("2150 x 3035 (B5 300dpi)", (2150, 3035)),
    ("3035 x 2150 (B5 300dpi landscape)", (3035, 2150)),
    ("2480 x 3508 (A4 300dpi)", (2480, 3508)),
    ("3508 x 2480 (A4 300dpi landscape)", (3508, 2480)),
    ("3035 x 4299 (B4 300dpi)", (3035, 4299)),
    ("4299 x 3035 (B4 300dpi landscape)", (4299, 3035)),
    ("2550 x 3300 (Letter 300dpi)", (2550, 3300)),
    ("3300 x 2550 (Letter 300dpi landscape)", (3300, 2550)),
)

# Background fill choices for a new canvas: ``label -> rgba``. ``None`` is the
# Custom entry, whose color comes from the dialog's color picker.
CANVAS_FILLS = (
    ("Transparent", (0, 0, 0, 0)),
    ("White", (255, 255, 255, 255)),
    ("Black", (0, 0, 0, 255)),
    ("Custom\u2026", None),
)

# Color modes: ``rgba`` keeps transparency (shown as the checkerboard), ``rgb``
# forces an opaque background.
CANVAS_COLOR_MODES = (
    ("RGBA (transparent)", "rgba"),
    ("RGB (opaque)", "rgb"),
)

# Print-size units, as ``(label, factor to inches)``.
_PRINT_UNITS = (
    ("in", 1.0),
    ("cm", 1.0 / 2.54),
    ("mm", 1.0 / 25.4),
)

# Starting-layer bounds for the New Canvas dialog.
_MIN_START_LAYERS = 1
_MAX_START_LAYERS = 100


def hex_to_rgb(color, default=(255, 255, 255)):
    """Parse ``#rgb``/``#rrggbb`` into an ``(r, g, b)`` tuple, else ``default``."""
    try:
        text = color.lstrip("#")
        if len(text) == 3:
            text = "".join(c * 2 for c in text)
        return tuple(int(text[i:i + 2], 16) for i in (0, 2, 4))
    except (ValueError, AttributeError, IndexError, TypeError):
        return default


def prompt_new_canvas(root, initial=(CANVAS_W, CANVAS_H), default_name="Canvas",
                      default_layers=_MIN_START_LAYERS):
    """Ask for a new canvas and the settings it should start with.

    Returns a dict ``{"name", "size", "fill", "layers", "make_default"}`` where
    ``size`` is ``(width, height)``, ``fill`` is the ``(r, g, b, a)`` background
    applied to the first layer, ``layers`` is how many empty layers to create,
    and ``make_default`` asks the caller to remember the size. Returns ``None``
    when cancelled.
    """
    dialog = tk.Toplevel(root)
    dialog.title("New Canvas")
    dialog.transient(root)
    dialog.resizable(False, False)
    dialog.withdraw()

    frame = tk.Frame(dialog, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)
    frame.columnconfigure(1, weight=1)

    error_var = tk.StringVar(value="")
    result = {"value": None}
    # Guards the programmatic size writes (preset/orientation/print) so the
    # size's own on_change does not bounce the preset dropdown back to Custom.
    syncing = {"size": False}

    initial_size = (int(initial[0]), int(initial[1]))
    preset_map = {"Custom": None}
    for label, size in CANVAS_PRESETS:
        preset_map.setdefault(label, size)
    default_label = next(
        (label for label, size in CANVAS_PRESETS if size == initial_size), None
    )
    if default_label is None:
        default_label = f"{initial_size[0]} x {initial_size[1]}"
        preset_map[default_label] = initial_size

    row = 0

    # ------------------------------------------------------------------ name
    tk.Label(frame, text="Name", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    name_var = tk.StringVar(value=default_name)
    name_entry = tk.Entry(frame, textvariable=name_var, width=28)
    name_entry.grid(row=row, column=1, sticky="we", pady=2)
    row += 1

    # --------------------------------------------------------------- preset
    tk.Label(frame, text="Preset", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    preset_var = tk.StringVar(value=default_label)
    preset_combo = ttk.Combobox(
        frame, textvariable=preset_var, state="readonly", width=26,
        values=list(preset_map.keys()),
    )
    preset_combo.grid(row=row, column=1, sticky="we", pady=2)
    row += 1

    # ----------------------------------------------------------------- size
    tk.Label(frame, text="Size", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    size_input = SizeInput(
        frame, initial_size[0], initial_size[1], field_width=8,
        on_change=lambda *_a: _on_size_change(),
    )
    size_input.grid(row=row, column=1, sticky="w", pady=2)
    row += 1

    # ---------------------------------------------------------- orientation
    tk.Label(frame, text="Orientation", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    orient_var = tk.StringVar(
        value="landscape" if initial_size[0] > initial_size[1] else "portrait"
    )
    orient_holder = tk.Frame(frame)
    orient_holder.grid(row=row, column=1, sticky="w", pady=2)
    for text, kind in (("Portrait", "portrait"), ("Landscape", "landscape")):
        tk.Radiobutton(
            orient_holder, text=text, value=kind, variable=orient_var,
            command=lambda: _set_orientation(),
        ).pack(side=tk.LEFT, padx=(0, 12))
    row += 1

    # ----------------------------------------------------------- background
    tk.Label(frame, text="Background", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    bg_holder = tk.Frame(frame)
    bg_holder.grid(row=row, column=1, sticky="w", pady=2)
    custom_fill = {"rgb": (255, 255, 255)}
    fill_var = tk.StringVar(value="Transparent")
    bg_combo = ttk.Combobox(
        bg_holder, textvariable=fill_var, state="readonly", width=14,
        values=[label for label, _value in CANVAS_FILLS],
    )
    bg_combo.pack(side=tk.LEFT)
    swatch = tk.Button(bg_holder, width=4, command=lambda: _choose_custom_fill())
    theme.skip(swatch)
    row += 1

    # ----------------------------------------------------------- color mode
    tk.Label(frame, text="Color mode", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    mode_var = tk.StringVar(value=CANVAS_COLOR_MODES[0][0])
    mode_combo = ttk.Combobox(
        frame, textvariable=mode_var, state="readonly", width=22,
        values=[label for label, _value in CANVAS_COLOR_MODES],
    )
    mode_combo.grid(row=row, column=1, sticky="we", pady=2)
    mode_var.trace_add("write", lambda *_a: _on_mode_change())
    row += 1

    # --------------------------------------------------------------- layers
    tk.Label(frame, text="Starting layers", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    layers_var = tk.StringVar(value=str(max(_MIN_START_LAYERS, int(default_layers))))
    tk.Spinbox(
        frame, from_=_MIN_START_LAYERS, to=_MAX_START_LAYERS, width=6,
        textvariable=layers_var,
    ).grid(row=row, column=1, sticky="w", pady=2)
    row += 1

    # ----------------------------------------------------------- print size
    tk.Label(frame, text="Print size", anchor="w").grid(
        row=row, column=0, sticky="w", pady=2,
    )
    print_holder = tk.Frame(frame)
    print_holder.grid(row=row, column=1, sticky="w", pady=2)
    print_w_var = tk.StringVar(value="")
    print_h_var = tk.StringVar(value="")
    unit_var = tk.StringVar(value=_PRINT_UNITS[0][0])
    dpi_var = tk.StringVar(value="300")
    tk.Entry(print_holder, textvariable=print_w_var, width=5).pack(side=tk.LEFT)
    tk.Label(print_holder, text="x").pack(side=tk.LEFT, padx=2)
    tk.Entry(print_holder, textvariable=print_h_var, width=5).pack(side=tk.LEFT)
    ttk.Combobox(
        print_holder, textvariable=unit_var, state="readonly", width=4,
        values=[label for label, _factor in _PRINT_UNITS],
    ).pack(side=tk.LEFT, padx=(4, 0))
    tk.Label(print_holder, text="DPI").pack(side=tk.LEFT, padx=(8, 2))
    tk.Spinbox(
        print_holder, from_=1, to=1200, width=5, textvariable=dpi_var,
    ).pack(side=tk.LEFT)
    tk.Button(print_holder, text="Apply", command=lambda: _apply_print_size()).pack(
        side=tk.LEFT, padx=(6, 0)
    )
    row += 1

    # ------------------------------------------------------ remember default
    default_var = tk.BooleanVar(value=False)
    tk.Checkbutton(
        frame, text="Use as default for new canvases", variable=default_var,
    ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
    row += 1

    tk.Label(frame, textvariable=error_var, fg="red", anchor="w").grid(
        row=row, column=0, columnspan=2, sticky="w", pady=(4, 0)
    )
    row += 1

    # --------------------------------------------------------------- actions
    def _refresh_orientation():
        """Mirror the current dimensions in the orientation radios."""
        width, height = size_input.values()
        if width is None or height is None or width == height:
            return
        orient_var.set("landscape" if width > height else "portrait")

    def _on_size_change():
        if not syncing["size"]:
            preset_var.set("Custom")
        _refresh_orientation()

    def _apply_preset():
        size = preset_map.get(preset_var.get())
        if size is None:
            return
        syncing["size"] = True
        size_input.set_values(*size)
        syncing["size"] = False
        error_var.set("")

    def _set_orientation():
        """Portrait/Landscape: swap the dimensions when the orientation differs."""
        width, height = size_input.values()
        if width is None or height is None:
            width, height = initial_size
        if orient_var.get() == "portrait" and width > height:
            width, height = height, width
        elif orient_var.get() == "landscape" and height > width:
            width, height = height, width
        syncing["size"] = True
        size_input.set_values(width, height)
        syncing["size"] = False
        preset_var.set("Custom")

    def _apply_swatch():
        try:
            swatch.configure(bg=rgb_to_hex(custom_fill["rgb"]))
        except tk.TclError:
            pass

    def _update_swatch_visibility():
        """The swatch button only shows for the Custom background entry."""
        if fill_var.get().startswith("Custom"):
            swatch.pack(side=tk.LEFT, padx=(6, 0))
        else:
            swatch.pack_forget()

    def _choose_custom_fill():
        chosen = ask_color(
            dialog, rgb_to_hex(custom_fill["rgb"]),
            title="Background Color", persist_recent=False,
        )
        if chosen:
            custom_fill["rgb"] = hex_to_rgb(chosen)
            fill_var.set("Custom\u2026")
            _apply_swatch()
            _update_swatch_visibility()

    def _on_mode_change():
        opaque = dict(CANVAS_COLOR_MODES).get(mode_var.get()) == "rgb"
        if opaque and fill_var.get() == "Transparent":
            fill_var.set("White")

    def _apply_print_size():
        try:
            width_in = float(print_w_var.get())
            height_in = float(print_h_var.get())
            dpi = max(1, int(float(dpi_var.get())))
        except (TypeError, ValueError, tk.TclError):
            error_var.set("Enter a print width, height and DPI.")
            return
        factor = dict(_PRINT_UNITS).get(unit_var.get(), 1.0)
        width = max(1, min(MAX_CANVAS_SIZE, int(round(width_in * factor * dpi))))
        height = max(1, min(MAX_CANVAS_SIZE, int(round(height_in * factor * dpi))))
        syncing["size"] = True
        size_input.set_values(width, height)
        syncing["size"] = False
        preset_var.set("Custom")
        error_var.set("")

    def _current_fill():
        value = None
        for label, fill in CANVAS_FILLS:
            if label == fill_var.get():
                value = fill
                break
        if value is None:
            value = tuple(custom_fill["rgb"]) + (255,)
        opaque = dict(CANVAS_COLOR_MODES).get(mode_var.get()) == "rgb"
        if opaque and value[3] == 0:
            value = (255, 255, 255, 255)
        return value

    def on_ok(_event=None):
        width, height = size_input.values()
        if width is None or height is None:
            error_var.set("Enter whole numbers.")
            return
        if width < 1 or height < 1:
            error_var.set("Width and height must be at least 1.")
            return
        if width > MAX_CANVAS_SIZE or height > MAX_CANVAS_SIZE:
            error_var.set(f"Maximum size is {MAX_CANVAS_SIZE}.")
            return
        try:
            layers = max(_MIN_START_LAYERS, min(_MAX_START_LAYERS, int(float(layers_var.get()))))
        except (TypeError, ValueError, tk.TclError):
            layers = _MIN_START_LAYERS
        result["value"] = {
            "name": name_var.get().strip() or default_name,
            "size": (width, height),
            "fill": _current_fill(),
            "layers": layers,
            "make_default": bool(default_var.get()),
        }
        dialog.destroy()

    def on_cancel(_event=None):
        dialog.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(row=row, column=0, columnspan=2, sticky="e", pady=(8, 0))
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=on_cancel).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    preset_combo.bind("<<ComboboxSelected>>", lambda _e: _apply_preset())
    bg_combo.bind("<<ComboboxSelected>>", lambda _e: _update_swatch_visibility())
    _apply_swatch()
    _update_swatch_visibility()

    dialog.bind("<Return>", on_ok)
    dialog.bind("<Escape>", on_cancel)
    dialog.protocol("WM_DELETE_WINDOW", on_cancel)

    theme.restyle(dialog)
    center_window(root, dialog)
    name_entry.focus_set()
    name_entry.selection_range(0, tk.END)

    dialog.grab_set()
    root.wait_window(dialog)
    return result["value"]


def ruler_step(zoom):
    """Return a "nice" labelled tick spacing in document pixels for ``zoom``."""
    for step in RULER_STEPS:
        if step * zoom >= RULER_MIN_SPACING:
            return step
    return RULER_STEPS[-1]


class RulerCanvas(tk.Canvas):
    """A ruler strip drawn along the top or left edge of a :class:`CanvasView`.

    Ticks are labelled with document pixel coordinates and track the view's zoom
    and scroll position, so they line up with the pixels they annotate. Colours
    come from :mod:`paintapp.theme`; a canvas is skipped by the theme walker, so
    this widget re-themes itself through :func:`theme.subscribe`.
    """

    def __init__(self, parent, view, orient=tk.HORIZONTAL):
        self._horizontal = orient == tk.HORIZONTAL
        if self._horizontal:
            super().__init__(parent, height=RULER_SIZE, bd=0,
                             highlightthickness=0, takefocus=0)
        else:
            super().__init__(parent, width=RULER_SIZE, bd=0,
                             highlightthickness=0, takefocus=0)
        self.view = view
        self._font = tkfont.nametofont("TkDefaultFont").copy()
        self._font.configure(size=7)
        self._tick_color = "#888888"
        self._text_color = "#666666"
        # Dragging off a ruler pulls out a guide: the horizontal top ruler
        # creates a vertical line (fixed x), the vertical left ruler a
        # horizontal one (fixed y). Each guide also shows a handle on its ruler
        # that can be dragged to reposition it, or double/right-clicked to
        # remove it, without switching tools.
        self._guide_orient = "v" if self._horizontal else "h"
        self._drag_mode = None
        self._drag_index = None
        self._drag_pushed = False
        # Last (zoom, size, start, guides) the ticks were drawn for, so the
        # several redraw requests a single resize produces collapse into one.
        self._last_state = None
        # Guides move along the ruler's own axis, so the cursor matches that
        # direction: the top ruler slides left/right, the left ruler up/down.
        self.configure(
            cursor="sb_h_double_arrow" if self._horizontal else "sb_v_double_arrow"
        )
        self._apply_theme()
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._guide_press)
        self.bind("<B1-Motion>", self._guide_motion)
        self.bind("<ButtonRelease-1>", self._guide_release)
        self.bind("<Double-Button-1>", self._guide_double)
        self.bind("<Button-3>", self._guide_remove_at)
        theme.subscribe(self._apply_theme)

    def _apply_theme(self):
        if not self.winfo_exists():
            return
        self._tick_color = theme.color("ruler_tick") or self._tick_color
        self._text_color = theme.color("ruler_text") or self._text_color
        background = theme.color("ruler_bg")
        if background:
            try:
                self.configure(bg=background)
            except tk.TclError:
                return
        # Colours changed, so the cached tick state must not suppress the repaint.
        self._last_state = None
        self.redraw()
        # The view is skipped by the theme walker, so repaint its guides here
        # (their colour also comes from the theme).
        view = getattr(self, "view", None)
        if view is not None:
            view._draw_grid()
            view._draw_guides()

    def _axis_transform(self):
        """``(scale, offset)`` mapping a document coordinate to a ruler pixel.

        ``doc * zoom - start``. The ruler always annotates the document axes in
        their own (ascending) direction, so the origin stays at the canvas
        corner. A view flip mirrors the artwork in place rather than the
        coordinate frame, so the ruler is unaffected by it; only a rotation,
        which has no axis-aligned document frame left to label, blanks it.
        """
        view = self.view
        if self._horizontal:
            return view.zoom, -view.canvasx(0)
        return view.zoom, -view.canvasy(0)

    def redraw(self):
        """Repaint the ticks for the view's current zoom and scroll offset."""
        if not self.winfo_exists():
            return
        view = self.view
        if view.rotated:
            # A rotated view has no axis-aligned document axes to annotate, so
            # the ruler is blanked rather than shown against the wrong frame.
            self._last_state = None
            self.delete("all")
            return
        zoom = view.zoom
        if self._horizontal:
            size = self.winfo_width()
            start = view.canvasx(0)
        else:
            size = view.winfo_height()
            start = view.canvasy(0)
        if size <= 1:
            return
        guides = tuple(
            (orient, pos) for orient, pos in view.guides
            if orient == self._guide_orient
        )
        scale, offset = self._axis_transform()
        state = (zoom, size, start, scale, offset, view.view_flip,
                 view.view_flip_v, guides)
        # A single resize fires several redraw requests (the view's re-render,
        # then the scroll callbacks and this ruler's own <Configure>), so skip
        # the work when nothing about the ticks has actually changed.
        if state == self._last_state:
            return
        self._last_state = state
        self.delete("all")
        step = ruler_step(zoom)
        minor = step / 5.0 if step % 5 == 0 else step / 2.0
        # The visible document interval runs between the ruler's two edges; a
        # flipped axis reverses which edge is which, so take the min/max.
        edge_a = (0.0 - offset) / scale
        edge_b = (size - offset) / scale
        doc_start, doc_end = min(edge_a, edge_b), max(edge_a, edge_b)
        # Minor ticks first; the labelled majors are drawn over the shared
        # positions, so no de-duplication is needed.
        self._draw_ticks(minor, doc_start, doc_end, scale, offset, size, False)
        self._draw_ticks(step, doc_start, doc_end, scale, offset, size, True)
        # Guide handles sit on top of the ticks so they stay easy to grab.
        self._draw_guide_markers()

    def _draw_ticks(self, step, doc_start, doc_end, scale, offset, size,
                    labelled):
        if step <= 0:
            return
        thickness = (
            self.winfo_height() if self._horizontal else self.winfo_width()
        )
        if thickness <= 1:
            thickness = RULER_SIZE
        doc = math.floor(doc_start / step) * step
        limit = doc_end + step * 0.5
        while doc <= limit:
            pos = scale * doc + offset
            if -1.0 <= pos <= size + 1.0:
                if labelled:
                    self._draw_major(pos, thickness, int(round(doc)))
                else:
                    self._draw_minor(pos, thickness)
            doc += step

    def _draw_minor(self, pos, thickness):
        length = 4
        if self._horizontal:
            self.create_line(pos, thickness - length, pos, thickness,
                             fill=self._tick_color)
        else:
            self.create_line(thickness - length, pos, thickness, pos,
                             fill=self._tick_color)

    def _draw_major(self, pos, thickness, value):
        length = 8
        if self._horizontal:
            self.create_line(pos, thickness - length, pos, thickness,
                             fill=self._tick_color)
            self.create_text(pos + 2, 1, text=str(value), anchor="nw",
                             fill=self._text_color, font=self._font)
        else:
            self.create_line(thickness - length, pos, thickness, pos,
                             fill=self._tick_color)
            self.create_text(thickness / 2.0, pos + 12, text=str(value),
                             angle=90, anchor="center",
                             fill=self._text_color, font=self._font)

    # ------------------------------------------------------------------ guides
    def _guide_pos(self, event):
        """The document coordinate the pointer sits at along the ruler's axis."""
        view = self.view
        if self._horizontal:
            offset = self.winfo_rootx() + event.x - view.winfo_rootx()
            x = view.canvasx(offset)
            if view.view_flip:
                # Undo the horizontal mirror so the guide follows the pointer.
                x = 2.0 * view._pivot_canvas()[0] - x
            return int(round(x / view.zoom))
        offset = self.winfo_rooty() + event.y - view.winfo_rooty()
        y = view.canvasy(offset)
        if view.view_flip_v:
            # Undo the vertical mirror so the guide follows the pointer.
            y = 2.0 * view._pivot_canvas()[1] - y
        return int(round(y / view.zoom))

    def _over_view(self, event):
        """True when the pointer is currently over the canvas view."""
        view = self.view
        x = self.winfo_rootx() + event.x
        y = self.winfo_rooty() + event.y
        return (
            view.winfo_rootx() <= x < view.winfo_rootx() + view.winfo_width()
            and view.winfo_rooty() <= y < view.winfo_rooty() + view.winfo_height()
        )

    def _guide_marker_at(self, event):
        """Index of the guide handle under ``event``, or ``None``.

        Only guides of this ruler's axis have handles here: the top ruler owns
        the vertical guides, the left ruler the horizontal ones.
        """
        axis = event.x if self._horizontal else event.y
        best = None
        best_distance = float(GUIDE_MARKER_HIT)
        for index, (orient, pos) in enumerate(self.view.guides):
            if orient != self._guide_orient:
                continue
            distance = abs(self._guide_axis_pixel(pos) - axis)
            if distance <= best_distance:
                best_distance = distance
                best = index
        return best

    def _guide_axis_pixel(self, pos):
        """Where a guide at document ``pos`` sits along this ruler, in pixels.

        A mirrored view draws the guide with the artwork, so its handle must
        follow the same view transform rather than the plain document mapping
        the ticks use.
        """
        view = self.view
        if self._horizontal:
            x, _ = view._doc_to_canvas(pos, 0.0)
            return x - view.canvasx(0)
        _, y = view._doc_to_canvas(0.0, pos)
        return y - view.canvasy(0)

    def _draw_guide_markers(self):
        """Draw a draggable handle for each guide of this ruler's axis."""
        if not self.view.guides_visible:
            return
        color = theme.color("guide_line") or DEFAULT_GUIDE_COLOR
        thickness = self.winfo_height() if self._horizontal else self.winfo_width()
        if thickness <= 1:
            thickness = RULER_SIZE
        half = GUIDE_MARKER_LEN / 2.0
        for orient, pos in self.view.guides:
            if orient != self._guide_orient:
                continue
            p = self._guide_axis_pixel(pos)
            if self._horizontal:
                self.create_rectangle(
                    p - half, thickness - GUIDE_MARKER_DEPTH, p + half,
                    thickness, fill=color, outline="", tags="guidemarker",
                )
            else:
                self.create_rectangle(
                    thickness - GUIDE_MARKER_DEPTH, p - half, thickness,
                    p + half, fill=color, outline="", tags="guidemarker",
                )

    def _guide_press(self, event):
        if self.view.rotated:
            # Guides are drawn along document axes, which a rotated view does
            # not show; disable ruler-driven guide edits while rotated. A
            # flip (horizontal or vertical) keeps the axes aligned (mirrored),
            # so editing stays available there.
            return
        self._drag_mode = None
        self._drag_index = None
        self._drag_pushed = False
        index = self._guide_marker_at(event)
        if index is None:
            self._drag_mode = "create"
        else:
            self._drag_mode = "move"
            self._drag_index = index

    def _guide_motion(self, event):
        if self._drag_mode == "create":
            self.view.set_pending_guide(self._guide_orient, self._guide_pos(event))
            return
        if self._drag_mode != "move" or self._drag_index is None:
            return
        if not (0 <= self._drag_index < len(self.view.guides)):
            return
        if not self._drag_pushed:
            self.view.push_guide_undo()
            self._drag_pushed = True
        self.view.set_guide_pos(self._drag_index, self._guide_pos(event))

    def _guide_release(self, event):
        mode = self._drag_mode
        self._drag_mode = None
        self._drag_index = None
        self._drag_pushed = False
        if mode != "create":
            # A moved handle already updated the guide live while dragging.
            return
        self.view.clear_pending_guide()
        if self._over_view(event):
            self.view.push_guide_undo()
            self.view.add_guide(self._guide_orient, self._guide_pos(event))

    def _guide_double(self, event):
        """A double-click removes the guide whose handle was clicked."""
        index = self._guide_marker_at(event)
        self._drag_mode = None
        self._drag_index = None
        self._drag_pushed = False
        if index is None:
            return
        self.view.push_guide_undo()
        self.view.pop_guide(index)
        return "break"

    def _guide_remove_at(self, event):
        """A right-click removes the guide whose handle was clicked."""
        index = self._guide_marker_at(event)
        if index is None:
            return
        self.view.push_guide_undo()
        self.view.pop_guide(index)
        return "break"


def opacity_stipple(opacity):
    """A Tk stipple bitmap that dithers an item to roughly ``opacity`` percent.

    Tk canvas items have no alpha channel, but a stipple leaves its "off" pixels
    unpainted, so whatever is behind (the document, the transparency
    checkerboard) shows through. That makes it a faithful, if coarse,
    transparency preview for the Path tool's opacity setting.
    """
    if opacity >= 100:
        return None
    if opacity >= 75:
        return "gray75"
    if opacity >= 50:
        return "gray50"
    if opacity >= 25:
        return "gray25"
    return "gray12"


class CanvasView(tk.Canvas):
    """The rendered pixels of one :class:`document.Canvas` plus overlay items.

    The view keeps a ``zoom`` factor. All public coordinates are document
    pixels; :meth:`view_to_doc` and :meth:`event_to_doc` convert from view and
    widget event coordinates respectively.
    """

    MIN_ZOOM = 0.1
    MAX_ZOOM = 16.0
    ZOOM_STEP = 1.25

    def __init__(self, parent, doc, bg=DEFAULT_OFFCANVAS_BG):
        super().__init__(
            parent, bg=rgb_to_hex(bg), cursor="crosshair",
            width=doc.width, height=doc.height,
            highlightthickness=0,
            xscrollincrement=1, yscrollincrement=1,
        )
        self.doc = doc
        self.bg_color = tuple(bg)
        self.zoom = 1.0
        # Non-destructive view transform: the artwork can be rotated by an
        # arbitrary angle and mirrored without touching document pixels. The
        # rotation is screen-clockwise-positive and happens about
        # ``_view_pivot`` (a document point); ``None`` means "use the point
        # currently under the viewport centre". Horizontal and vertical flips
        # are independent toggles (both at once read as a 180 degree turn), as
        # in GIMP. Coordinates are mapped through :meth:`_xform_pt` on the way
        # to the canvas, and the rendered raster is warped to match (see
        # :meth:`render`).
        self.view_angle = 0.0
        self.view_flip = False
        self.view_flip_v = False
        self._view_pivot = None
        # Cached canonical board and warp coefficients, kept so the crop dim can
        # be re-composited without re-flattening the viewport (see
        # :meth:`_refresh_crop_image`).
        self._warp = None
        self._warp_resample = Image.NEAREST
        # Exploded-grid view: when ``_layout`` is set the view renders each
        # layer in its own cell (see ``paintapp.exploded``) instead of stacking
        # them. ``_shift`` maps document coordinates to layout coordinates for
        # the active layer, so tools and overlays keep working against the cell
        # being edited; ``_frame_size`` is the layout's extent (the document size
        # when not exploded).
        self._layout = None
        self._shift = (0, 0)
        self._frame_size = (doc.width, doc.height)
        # Camera saved on the way into the exploded view, restored on the way
        # out (see :meth:`enter_exploded`). ``_exploded_camera_framed`` marks a
        # view that already has an exploded camera, so a tab switch does not
        # reframe it (see :meth:`paintapp.app.PaintApp._activate_canvas`).
        self._stacked_view_state = None
        self._exploded_camera_framed = False
        # Document pixel that should sit at the viewport's top-left on the next
        # successful render. Set when a view is recreated (see
        # :meth:`set_view_state`) so its camera survives an undo/redo rebuild.
        self._pending_anchor = None
        self._photo = None
        # The canvas item holding ``_photo``. It is kept across renders and
        # re-pointed with ``itemconfig`` instead of being deleted and recreated
        # with every overlay item (see :meth:`render`), which cut Tk work on each
        # zoom/pan frame.
        self._base_item = None
        self._cursor = None
        self._clone_source = None
        self.selection = None
        self.layout_boxes = []
        self._gizmo = None
        self._path_data = None
        self._vector_overlay = None
        self._poses_overlay = None
        self._warp_grid = None
        self._puppet_overlay = None
        self._symmetry = None
        self._text_session = None
        self._shape_preview = None
        self._gradient_preview = None
        self._crop_preview = None
        # The undimmed base board of the last render, kept so the crop dim can
        # be re-composited on every drag step (and dropped again) without
        # re-flattening the viewport (see :meth:`_dim_crop`).
        self._base_board = None
        self._selection_image = None
        self._selection_label = None
        self._mask_preview = None
        # Document position of the mask preview's top-left: the mask lives in a
        # layer's own buffer frame, which sits at that layer's offset (a link is
        # not baked to the canvas, so its mask is not document-aligned).
        self._mask_origin = (0, 0)
        self._mask_overlay = (255, 0, 0)
        self._mask_image = None
        # A finished generation shown for review before it is applied, as an
        # ``(image, origin)`` pair in document pixels (see
        # :meth:`set_generation_preview`). It is composited into the base image
        # on each render rather than drawn as its own canvas item. The live
        # denoising frames reuse it, so a run's preview and its result match.
        self._gen_preview = None
        self._content_w = doc.width
        self._content_h = doc.height
        # Document coordinate of the preview buffer's top-left corner. It is
        # ``(0, 0)`` normally and negative when off-canvas content extends past
        # the canvas top/left. The canvas itself is always drawn with its own
        # origin at canvas pixel ``(0, 0)``, so toggling the preview never pans
        # the view; the buffer is simply placed at ``origin * zoom``.
        self._origin = (0, 0)
        self._source = None
        self._source_origin = (0, 0)
        self._scroll = (0, 0, doc.width, doc.height)
        # Document rectangle currently rasterized into the base image, plus the
        # render/refresh bookkeeping that keeps panning covered (see
        # :meth:`_ensure_covered`).
        self._covered = None
        self._rendering = False
        self._rerender_pending = False
        # Set while a resize-triggered render is queued, so a burst of
        # <Configure> events collapses to a single repaint (see
        # :meth:`_schedule_render`). ``_configure_job`` is that pending
        # ``after`` call, kept so a newer event can reset the debounce timer.
        self._configure_render_pending = False
        self._configure_job = None
        self._rendered_view_size = None
        self._render_window = None
        self._render_size = None
        # Ruler strips attached around this view (see :meth:`attach_rulers`).
        self._rulers = []
        self._ruler_widgets = ()
        # Ruler guides live on the document (see ``Canvas.guides``); the pending
        # guide is the line being dragged out of a ruler but not yet dropped.
        # ``_shared_guides`` overrides the document's own list when the app is
        # in "same guides on all canvases" mode (see :meth:`set_shared_guides`).
        self.guides_visible = True
        self._pending_guide = None
        self._shared_guides = None
        # Grid overlay and snapping (see :meth:`set_grid` and :meth:`snap_doc`).
        self.grid_visible = False
        self.grid_size = DEFAULT_GRID_SIZE
        self.grid_snap = False
        # Outline the active layer's bounds on the stacked canvas (see
        # :meth:`_draw_layer_outline`); the exploded grid already marks the
        # active cell, so this applies to the normal view only.
        self.layer_outline_visible = True
        # Viewport-only rendering: the app installs a provider that flattens just
        # the requested document region (see :meth:`set_source_providers`), so a
        # huge canvas is not composited whole on every repaint. ``_content_info``
        # supplies the full content bounds for the scroll region.
        self._content_provider = None
        self._region_provider = None
        # The frame cached for a live stroke: a display composite of the viewport
        # that a moving brush patches in place (see :meth:`_live_composite`).
        self._live_draft = None
        self._content_origin = (0, 0)
        self._content_size = (doc.width, doc.height)
        self._render_deferred = False
        # Set by :class:`CanvasArea`; called just before a guide edit so the app
        # can snapshot the document for undo.
        self.on_guides_changed = None
        # Set by :class:`CanvasArea`; called after a guide edit so the other tabs
        # repaint when the guides are shared across canvases.
        self.on_guides_edited = None
        # With no scrollbars attached, these callbacks exist purely to keep the
        # rulers in sync whenever Tk scrolls or re-clamps the view.
        self.config(xscrollcommand=self._on_scroll,
                    yscrollcommand=self._on_scroll)
        self._update_scrollregion()
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        """Re-render on resize so padding and the camera anchor stay correct.

        A resize drag fires a <Configure> for every intermediate size, and
        rasterizing each one synchronously saturates the UI thread, which is what
        made resizing feel laggy. The work is deferred to the end of the burst
        (see :meth:`_schedule_render`), so only the final size in a burst is
        drawn. The size guard still skips the duplicate events Tk emits for a
        size that was already rendered.
        """
        if event.widget is not self:
            return
        if self._source is None and self._region_provider is None:
            return
        if (event.width, event.height) == self._rendered_view_size:
            return
        self._schedule_render()

    def _schedule_render(self):
        """Debounce a burst of resize events into a single trailing render.

        A drag reaches Tk's idle state between mouse moves, so the old
        ``after_idle`` still repainted on every step and each of those renders
        flattens and resamples the viewport - the laggy resize. Instead the
        render is deferred by :data:`RESIZE_RENDER_DELAY_MS` and the timer is
        reset on every event, so a drag that keeps arriving never starts the
        expensive work; only a pause settles into one render of the final size.
        The frame already on screen is drawn in document coordinates, so it
        stays put and needs no work while the drag runs.
        """
        if not self.winfo_exists():
            return
        # A view with nothing rendered yet (a freshly shown tab) must paint at
        # once rather than stay blank for the debounce window.
        if self._source is None:
            self._render_after_resize()
            return
        if self._configure_job is not None:
            try:
                self.after_cancel(self._configure_job)
            except tk.TclError:
                pass
            self._configure_job = None
        self._configure_render_pending = True
        self._configure_job = self.after(
            RESIZE_RENDER_DELAY_MS, self._render_after_resize
        )

    def _render_after_resize(self):
        self._configure_render_pending = False
        self._configure_job = None
        if not self.winfo_exists():
            return
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            return
        if (self.winfo_width(), self.winfo_height()) == self._rendered_view_size:
            return
        self.render()


    # ------------------------------------------------------------------ rulers
    def attach_rulers(self, widgets, rulers):
        """Register the ruler strips drawn around this view."""
        self._ruler_widgets = tuple(widgets)
        self._rulers = list(rulers)
        self._redraw_rulers()

    def set_rulers_visible(self, visible):
        """Show or hide the ruler strips (and their shared corner)."""
        for widget in self._ruler_widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()
        if visible:
            self._redraw_rulers()

    def _redraw_rulers(self):
        for ruler in self._rulers:
            ruler.redraw()

    def _on_scroll(self, *_args):
        if self._rulers:
            self._redraw_rulers()
        # Guides and grid lines span the visible viewport, so they must be
        # re-anchored when the view scrolls. The grid is drawn first so the
        # guides stay stacked on top of it. The generation preview is baked
        # into the base image, so it scrolls with it and needs no redraw.
        self._draw_grid()
        self._draw_guides()
        self._ensure_covered()

    def _ensure_covered(self):
        """Re-render when scrolling moves the viewport off the rendered region.

        Only the visible viewport is rasterized, so panning past it would show a
        blank margin. Coalesce a deferred re-render when that happens.
        """
        if self._rendering or self._source is None or self._covered is None:
            return
        if self._configure_render_pending:
            # A resize drag is mid-flight: Tk re-clamps the view as the viewport
            # grows, which lands here, but the trailing resize render will cover
            # the new viewport. Do not stack a second full render behind it.
            return
        if self._rerender_pending:
            return
        if not self.winfo_exists():
            return
        # Box the document region the viewport now covers (a rotated rectangle
        # when the view is transformed) and test the covered box against it.
        vw = self.winfo_width()
        vh = self.winfo_height()
        corners = ((0, 0), (vw, 0), (vw, vh), (0, vh))
        pts = [self.view_to_doc(self.canvasx(cx), self.canvasy(cy))
               for cx, cy in corners]
        tlx = min(p[0] for p in pts)
        tly = min(p[1] for p in pts)
        vx1 = max(p[0] for p in pts)
        vy1 = max(p[1] for p in pts)
        rx0, ry0, rx1, ry1 = self._covered
        if (tlx >= rx0 - 0.5 and tly >= ry0 - 0.5
                and vx1 <= rx1 + 0.5 and vy1 <= ry1 + 0.5):
            return
        self._rerender_pending = True
        self.after_idle(self._rerender_viewport)

    def _rerender_viewport(self):
        self._rerender_pending = False
        if self._source is not None and self.winfo_exists():
            pad = max(self.winfo_width(), self.winfo_height()) // 2
            self.render(pad=pad)

    def _defer_render(self):
        """Render once the widget has a real size (see :meth:`render`)."""
        if self._render_deferred:
            return
        self._render_deferred = True
        if self.winfo_exists():
            self.after_idle(self._deferred_render)

    def _deferred_render(self):
        self._render_deferred = False
        if not self.winfo_exists():
            return
        if self.winfo_width() <= 1 or self.winfo_height() <= 1:
            return
        self.render()

    def set_source_providers(self, content_provider, region_provider):
        """Install the callbacks that flatten the document for this view.

        ``content_provider()`` returns ``(origin, size)`` in document pixels for
        the full content bounds (used for the scroll region), and
        ``region_provider(region, scale)`` returns ``(image, origin)`` for a
        ``(left, top, right, bottom)`` document rectangle, composited at
        ``scale`` pixels per document pixel. Compositing only the requested
        region keeps repaints proportional to the viewport instead of the canvas
        size.
        """
        self._content_provider = content_provider
        self._region_provider = region_provider

    # ----------------------------------------------------------------- layout
    @property
    def exploded(self):
        """True while the view is showing the exploded-grid layout."""
        return self._layout is not None

    def set_layout(self, layout):
        """Show ``layout``'s exploded grid, or the normal stack when ``None``.

        The view's coordinate shift is recomputed for the active layer so
        painting and the overlays address that layer's cell. Callers render
        afterwards (see :meth:`paintapp.app.PaintApp.render`).
        """
        self._layout = layout
        self.refresh_layout_shift()

    def clear_layout(self):
        """Return the view to the normal stacked composite."""
        self._layout = None
        self._shift = (0, 0)
        self._frame_size = (self.doc.width, self.doc.height)

    def enter_exploded(self, layout):
        """Show ``layout``'s grid while keeping the current camera.

        The stacked view's zoom is kept, and its top-left anchor is shifted
        into the active layer's cell, so the artwork under the viewport stays
        put when the grid opens instead of the whole board being reframed. The
        stacked camera is saved for :meth:`leave_exploded`, which restores it
        when the grid is closed.
        """
        if layout is None or self._layout is not None:
            return
        self._stacked_view_state = self.view_state()
        self.set_layout(layout)
        state = dict(self._stacked_view_state)
        ax, ay = state.get("anchor", (0.0, 0.0))
        sx, sy = self._active_shift()
        state["anchor"] = (ax - sx, ay - sy)
        self.set_view_state(state)
        self._exploded_camera_framed = True

    def leave_exploded(self):
        """Return to the stacked view and restore the camera saved on entry."""
        if self._layout is None:
            return
        self.clear_layout()
        state = self._stacked_view_state
        self._stacked_view_state = None
        self._exploded_camera_framed = False
        if state is not None:
            self.set_view_state(state)

    def refresh_layout_shift(self):
        """Recompute the active-layer shift for the current exploded layout."""
        layout = self._layout
        if layout is None:
            self._shift = (0, 0)
            self._frame_size = (self.doc.width, self.doc.height)
            return
        self._frame_size = layout.size
        self._shift = self._active_shift()

    def _active_shift(self):
        """The document-to-layout shift for the currently active layer.

        Computed live (not cached) so a layer switch from anywhere - a tile
        click, the layer panel - is reflected by the next coordinate conversion
        without needing a repaint first.
        """
        layout = self._layout
        if layout is None:
            return (0, 0)
        index = getattr(self.doc, "active", 0)
        if 0 <= index < len(layout.offsets) and 0 <= index < len(self.doc.layers):
            dx, dy = layout.offsets[index]
            ox, oy = self.doc.layers[index].offset
            return (ox - dx, oy - dy)
        return (0, 0)

    def layer_at_event(self, event):
        """The layer index whose exploded cell is under ``event``, or ``None``."""
        layout = self._layout
        if layout is None:
            return None
        x, y = self.view_to_doc(self.canvasx(event.x), self.canvasy(event.y))
        return layout.cell_at(x, y)

    def doc_frame_rect(self):
        """The document-space rectangle a whole-canvas action should cover.

        Normally the canvas itself. In exploded view the editable frame is the
        active layer's image (its cell), so actions like Select All target that
        instead of the document rectangle the grid does not show.
        """
        if self._layout is not None:
            index = getattr(self.doc, "active", 0)
            if 0 <= index < len(self.doc.layers):
                layer = self.doc.layers[index]
                ox, oy = layer.offset
                return (ox, oy, ox + layer.image.width, oy + layer.image.height)
        return (0, 0, self.doc.width, self.doc.height)

    # ------------------------------------------------------------------ render
    def render(self, composite=None, origin=None, top_left=None, pad=0,
               live=False, dirty=None, zoom_preview=False):
        self._rendering = True
        if not zoom_preview:
            self._cancel_zoom_refine()
        # A live stroke (``live``) hands over the document rectangle the brush
        # just changed (``dirty``); the frame already on screen is patched over
        # that rectangle instead of being re-flattened whole, so painting a large
        # canvas costs the dab rather than the document (see :meth:`_live_composite`).
        if not live or self.transformed or self._layout is not None \
                or composite is not None:
            self._live_draft = None
        # Render the visible viewport (plus ``pad`` device pixels of margin, so
        # a pan within the margin needs no re-raster). ``top_left`` is the
        # document pixel that should sit at the viewport's top-left; it defaults
        # to whichever document pixel is there now so the camera stays put.
        if top_left is None:
            top_left = self._pending_anchor
        if top_left is None:
            top_left = self._view_to_doc_float(self.canvasx(0), self.canvasy(0))
        content_origin, content_size = self._content_bounds()
        self._content_origin = content_origin
        self._content_size = content_size
        self._content_w = max(1, int(round(content_size[0] * self.zoom)))
        self._content_h = max(1, int(round(content_size[1] * self.zoom)))
        self._update_scrollregion(content_origin)
        self._restore_anchor(top_left)
        view_w = self.winfo_width()
        view_h = self.winfo_height()
        if view_w <= 1 or view_h <= 1:
            # The widget has not been laid out yet (e.g. a tab that was just
            # packed). Rendering now would have to guess a viewport size, and the
            # old fallback guessed the whole content size, flattening an entire
            # large canvas on every tab switch. Wait for Tk to report a real size
            # (its <Configure> re-renders us) instead.
            self._rendering = False
            # Force the next <Configure> to render even at a previously seen
            # size, so a tab shown again is not left blank.
            self._rendered_view_size = None
            self._defer_render()
            return
        actual = (self.canvasx(0) / self.zoom, self.canvasy(0) / self.zoom)
        pad = max(0, int(pad))
        if self.transformed:
            self._render_transformed(view_w, view_h, pad, composite, origin)
        else:
            window = (actual[0] - pad / self.zoom, actual[1] - pad / self.zoom)
            render_size = (view_w + pad * 2, view_h + pad * 2)
            self._render_window = window
            self._render_size = render_size
            # Composite at display resolution: a zoomed-out view flattens the
            # viewport's pixels instead of the (much larger) document rectangle it
            # covers, which would otherwise grow as 1/zoom^2. ``source_scale`` is the
            # composite's pixels per document pixel.
            source_scale = min(1.0, self.zoom) if composite is None else 1.0
            if (zoom_preview and composite is None and not live
                    and self._layout is None and self.zoom < 1.0):
                preview_source = self._zoom_preview_source()
                if preview_source is not None:
                    composite, origin, source_scale = preview_source
            if composite is None:
                region = (
                    math.floor(window[0]), math.floor(window[1]),
                    math.ceil(window[0] + render_size[0] / self.zoom),
                    math.ceil(window[1] + render_size[1] / self.zoom),
                )
                if live:
                    composite, origin = self._live_composite(
                        region, source_scale, dirty, render_size
                    )
                else:
                    composite, origin = self._flatten(region, source_scale)
            if origin is None:
                origin = (0, 0)
            self._source = composite
            self._source_origin = origin
            self._origin = origin
            underlay = None
            grid = None if self._layout is not None else (self._poses_overlay or {}).get("grid")
            if grid and composite.mode == "RGBA":
                underlay = self._pose_grid_image(grid, window, render_size)
            preview = compose_region(
                composite, self._frame_size, origin, self.zoom, window,
                render_size, self.bg_color, source_scale, underlay,
                self._layout.cells if self._layout is not None else None,
            )
            if self._layout is None:
                preview = self._overlay_generation_preview(preview, window, render_size)
            else:
                # The exploded grid maps the preview onto the active layer's cell
                # rather than the document, so it stays visible while the grid shows.
                preview = self._overlay_generation_preview(
                    preview, window, render_size, self._preview_layout_origin()
                )
            preview = self._color_managed(preview)
            # Keep the undimmed board so the crop dim is composited into the single
            # base image rather than painted as stippled canvas items, which repaint
            # slowly and flicker while the window is moved or the view scrolls.
            self._base_board = preview
            if self._layout is None:
                preview = self._dim_crop(preview)
            self._photo = ImageTk.PhotoImage(preview)
            self._covered = (
                window[0], window[1],
                window[0] + preview.width / self.zoom,
                window[1] + preview.height / self.zoom,
            )
            self._place_base(window[0] * self.zoom, window[1] * self.zoom)
        # Keep the base image item across renders and re-point it, rather than
        # deleting every item and rebuilding the whole canvas each frame. Every
        # overlay draw method deletes and redraws its own tagged items, so they
        # clean themselves up; the base item is at the bottom and is created
        # first (below them) the one time it is made.
        self._draw_doc_border()
        self._draw_grid()
        self._draw_symmetry()
        self._draw_mask_preview()
        self._draw_guides()
        self._draw_cursor()
        self._draw_clone_source()
        self._draw_selection()
        self._draw_selection_label()
        self._draw_layout_boxes()
        self._draw_gizmo()
        self._draw_path()
        self._draw_vector_overlay()
        self._draw_poses_overlay()
        self._draw_warp_grid()
        self._draw_puppet_overlay()
        self._draw_shape_preview()
        self._draw_gradient_preview()
        self._draw_crop_preview()
        self._draw_text_session()
        self._redraw_rulers()
        self._rendering = False
        self._pending_anchor = None
        self._rendered_view_size = (self.winfo_width(), self.winfo_height())

    def _live_composite(self, region, scale, dirty, render_size):
        """The composite for a live stroke, patched over its dirty rectangle.

        The first live frame flattens the viewport once (the same cost as a
        normal repaint) and is kept. Every later frame recomposites just the
        document rectangle the brush touched since and pastes it into a copy of
        that frame, so a fast stroke on a large canvas costs the dab's pixels
        instead of the whole viewport. At 100% zoom the patch is pixel-identical
        to a full re-flatten; when zoomed out the per-layer resample makes the
        patch a hair off at its edge, so it stands in as a draft and the release
        repaints the frame whole.

        A frame is only reused while the camera, zoom and provider are unchanged;
        anything else (a pan, a zoom, a layer change) rebuilds it. When the stack
        has a group, a clip or a filtered active layer - none of which the
        per-pixel patch models exactly - the frame is re-flattened each time,
        which is still correct (just the old cost).
        """
        frame = self._live_draft
        key = (scale, region, tuple(self._frame_size), tuple(render_size))
        provider = self._region_provider
        # Compare providers by value, not identity: ``PaintApp.render`` installs
        # a fresh ``self._region_composite`` bound method on every frame, and a
        # bound method is a new object on each attribute access (``a.m is a.m``
        # is False), so identity would discard the cached frame on every paint
        # event and re-flatten the whole viewport - the large-canvas paint lag.
        # Two bound methods of the same instance and function compare equal, while
        # a switch to the exploded view's provider does not, so equality reuses
        # the frame only while the same compositor is in effect.
        if (frame is None or frame["key"] != key
                or frame["provider"] != provider or frame.get("stale")):
            composite, origin = self._flatten(region, scale)
            frame = {
                "key": key,
                "provider": provider,
                "source": composite,
                "origin": origin,
                "anchor": origin,
                "stale": not self._patch_supported(),
            }
            self._live_draft = frame
            return composite, origin
        if dirty is None or frame["stale"]:
            return frame["source"], frame["origin"]
        source = frame["source"]
        origin = frame["origin"]
        if source.width <= 0 or source.height <= 0:
            return source, origin
        dl, dt, dr, db = (int(dirty[0]), int(dirty[1]),
                          int(dirty[2]), int(dirty[3]))
        # Clamp the dirty rectangle to the frame's document extent.
        fl, ft = origin
        fr = fl + source.width / scale
        fb = ft + source.height / scale
        dl = max(dl, fl)
        dt = max(dt, ft)
        dr = min(dr, fr)
        db = min(db, fb)
        if dl >= dr or dt >= db:
            return source, origin
        sub, sub_origin = self._flatten(
            (dl, dt, dr, db), scale, draft=True, anchor=frame["anchor"]
        )
        if sub.width <= 0 or sub.height <= 0:
            return source, origin
        dx = int(round((sub_origin[0] - origin[0]) * scale))
        dy = int(round((sub_origin[1] - origin[1]) * scale))
        patched = source.copy()
        patched.paste(sub, (dx, dy))
        frame["source"] = patched
        self._source = patched
        return patched, origin

    def _patch_supported(self):
        """True when the live stroke may patch the frame instead of re-flattening.

        Groups (a group filter reads neighbouring pixels) and clipped layers are
        handled by :meth:`Canvas._flatten`'s specialised paths, and a filtered
        active layer bleeds past the dab, so none of them can be patched from a
        small rectangle. A zoomed-out frame also flattens each layer with its own
        resample of a cropped region, so a patched rectangle can differ from a
        full re-flatten by a hair at its edge; the live patch is therefore a
        draft at ``scale < 1`` and the release repaints the frame whole.
        """
        doc = self.doc
        if doc is None or self._layout is not None:
            return False
        layers = getattr(doc, "layers", None)
        if not layers:
            return False
        if any(getattr(layer, "group", None) is not None for layer in layers):
            return False
        if any(getattr(layer, "clip", False) for layer in layers):
            return False
        active = getattr(doc, "active", 0)
        if 0 <= active < len(layers) and layers[active].filters:
            return False
        return True

    def _place_base(self, cx, cy):
        """Create or re-point the base image item at canvas ``(cx, cy)`` (NW)."""
        try:
            exists = (self._base_item is not None
                      and self.type(self._base_item) == "image")
        except tk.TclError:
            exists = False
        if not exists:
            self._base_item = tk.Canvas.create_image(
                self, cx, cy, anchor=tk.NW, image=self._photo, tags="base",
            )
            # Created anew only if the item was somehow dropped; keep it under
            # the overlays so it can never cover them.
            self.tag_lower(self._base_item)
        else:
            self.coords(self._base_item, cx, cy)
            self.itemconfig(self._base_item, image=self._photo)

    def _render_transformed(self, view_w, view_h, pad, composite, origin):
        """Rasterize the viewport for a rotated/flipped camera.

        The document region the padded viewport covers is inverse-mapped (a
        rotated rectangle), flattened axis-aligned, then warped onto the viewport
        with the view matrix. ``pad`` device pixels of margin on each side let a
        small pan reuse the raster instead of re-flattening.
        """
        zoom = self.zoom
        _m00, _m01, _m10, _m11, i00, i01, i10, i11 = self._matrix()
        px, py = self._pivot_canvas()
        pad = max(0, int(pad))
        # Canvas coordinate of the padded board's top-left corner.
        bx = self.canvasx(0) - pad
        by = self.canvasy(0) - pad
        out_w = max(1, int(view_w + pad * 2))
        out_h = max(1, int(view_h + pad * 2))
        corners = (
            (bx, by), (bx + out_w, by), (bx + out_w, by + out_h), (bx, by + out_h),
        )
        mapped = [self._unxform_pt(cx, cy) for cx, cy in corners]
        cbx0 = min(p[0] for p in mapped)
        cby0 = min(p[1] for p in mapped)
        cbx1 = max(p[0] for p in mapped)
        cby1 = max(p[1] for p in mapped)
        window = (cbx0 / zoom, cby0 / zoom)
        board_w = max(1, int(math.ceil(cbx1 - cbx0)))
        board_h = max(1, int(math.ceil(cby1 - cby0)))
        source_scale = min(1.0, zoom) if composite is None else 1.0
        if composite is None:
            region = (
                math.floor(window[0]), math.floor(window[1]),
                math.ceil(window[0] + board_w / zoom),
                math.ceil(window[1] + board_h / zoom),
            )
            composite, origin = self._flatten(region, source_scale)
        if origin is None:
            origin = (0, 0)
        self._source = composite
        self._source_origin = origin
        self._origin = origin
        self._render_window = window
        self._render_size = (board_w, board_h)
        grid = (self._poses_overlay or {}).get("grid")
        underlay = None
        if grid and composite.mode == "RGBA":
            underlay = self._pose_grid_image(grid, window, (board_w, board_h))
        board = compose_region(
            composite, self._frame_size, origin, zoom, window,
            (board_w, board_h), self.bg_color, source_scale, underlay,
        )
        board = self._overlay_generation_preview(board, window, (board_w, board_h))
        board = self._overlay_mask_veil(board, window, (board_w, board_h))
        board = self._color_managed(board)
        # Keep the undimmed board so the crop dim can be re-composited and
        # re-warped without flattening again (see :meth:`_refresh_crop_image`).
        self._base_board = board
        board = self._dim_crop(board)
        # Output pixel -> board pixel affine: board pixel 0 is canonical ``cbx0``.
        a02 = i00 * (bx - px) + i01 * (by - py) + px - cbx0
        a12 = i10 * (bx - px) + i11 * (by - py) + py - cby0
        coeffs = (i00, i01, a02, i10, i11, a12)
        resample = Image.BICUBIC if source_scale >= 1.0 else Image.BILINEAR
        warped = board.convert("RGB").transform(
            (out_w, out_h), Image.AFFINE, coeffs, resample=resample
        )
        self._warp = (board, coeffs, (out_w, out_h))
        self._warp_resample = resample
        self._photo = ImageTk.PhotoImage(warped)
        # The board covers canonical [cbx0, cbx1] x [cby0, cby1] -> a document box.
        self._covered = (cbx0 / zoom, cby0 / zoom, cbx1 / zoom, cby1 / zoom)
        self._place_base(bx, by)

    def _overlay_mask_veil(self, board, window, size):
        """Composite the mask-edit veil into a canonical board (rotated view).

        The unrotated view draws the veil as its own canvas item; a transformed
        view cannot place an axis-aligned image, so it is baked into the board
        before the warp, where it rotates with the artwork.
        """
        if self._mask_preview is None or self._layout is not None:
            return board
        crop, item = visible_crop(
            self._mask_preview, self._mask_origin, self.zoom, window, size
        )
        if crop is None:
            return board
        red, green, blue = self._mask_overlay
        veil = Image.new("RGBA", crop.size, (red, green, blue, 0))
        veil.putalpha(crop.point(lambda value: int((255 - value) * 0.5)))
        position = (
            int(round(item[0] - window[0] * self.zoom)),
            int(round(item[1] - window[1] * self.zoom)),
        )
        board = board.convert("RGBA")
        board.alpha_composite(veil, position)
        return board.convert("RGB")

    def _content_bounds(self):
        """Full content bounds as ``(origin, size)`` in document pixels."""
        if self._content_provider is not None:
            return self._content_provider()
        return (0, 0), (self.doc.width, self.doc.height)

    def _flatten(self, region, scale=1.0, draft=False, anchor=None):
        """Flatten ``region`` (document pixels) via the app's provider.

        ``scale`` is the requested pixels per document pixel. ``draft``/``anchor``
        are forwarded to the provider for the live stroke patch (see
        :meth:`_live_composite`). Falls back to a full-document composite when no
        provider is installed, so a bare view (e.g. in tests) still renders.
        """
        if self._region_provider is not None:
            return self._region_provider(region, scale, draft=draft, anchor=anchor)
        return self.doc.composite_transparent(), (0, 0)

    def _color_managed(self, image):
        """Convert a composited preview from the working space to the monitor.

        ICC-transform the preview through the canvas's working profile (into the
        assumed sRGB monitor space), soft-proofing against its proof profile when
        one is set. A no-op - and no cost - in the default sRGB configuration.
        """
        canvas = self.doc
        if canvas is None:
            return image
        working = getattr(canvas, "working_profile", None)
        proof = getattr(canvas, "proof_profile", None)
        proof_on = bool(proof) and not colormanagement.is_srgb(proof)
        if colormanagement.is_srgb(working) and not proof_on:
            return image
        return colormanagement.to_display(
            image, working, colormanagement.SRGB, proof if proof_on else None,
            proof_intent=getattr(
                canvas, "proof_intent", colormanagement.DEFAULT_PROOF_INTENT
            ),
        )

    def set_bg_color(self, bg):
        """Change the off-canvas background color and re-render the view."""
        self.bg_color = tuple(bg)
        self.config(bg=rgb_to_hex(bg))
        if self._source is not None:
            self.render()

    def _update_scrollregion(self, origin=(0, 0)):
        z = self.zoom
        ox, oy = origin
        frame_w, frame_h = self._frame_size
        cw = self._content_size[0]
        ch = self._content_size[1]
        rects = [
            (ox, oy, ox + cw, oy + ch),
            (0, 0, frame_w, frame_h),
        ]
        # A generation preview may sit outside the canvas, so let the view
        # scroll to it while it is shown.
        preview = self._preview_bounds()
        if preview is not None:
            rects.append(preview)
        left = top = float("inf")
        right = bottom = float("-inf")
        for (l, t, r, b) in rects:
            bx0, by0, bx1, by1 = self._doc_rect_to_canvas_bbox(l, t, r, b)
            left = min(left, bx0)
            top = min(top, by0)
            right = max(right, bx1)
            bottom = max(bottom, by1)
        # A viewport of pan room on every side keeps Tk from clamping the
        # anchor and lets the Drag/Pan tool scroll in any direction, even when
        # the canvas fits the viewport.
        view_w = max(1, self.winfo_width())
        view_h = max(1, self.winfo_height())
        left -= view_w
        top -= view_h
        right += view_w
        bottom += view_h
        self._scroll = (left, top, right, bottom)
        self.config(scrollregion=self._scroll)

    def _restore_anchor(self, anchor):
        """Scroll so document pixel ``anchor`` returns to the viewport top-left.

        The scroll region is grown to contain the anchored viewport when the
        anchor falls outside it. After a zoom or a view rotation the anchor can
        land farther from the content box than the single-viewport pan margin
        :meth:`_update_scrollregion` adds - for example when zooming about a
        cursor over empty off-canvas space, or turning a view whose pivot sits
        off the content. Tk clamps ``*view_moveto`` to the scroll region, so
        without this the camera is jerked to the region edge on every step and
        the canvas appears to jump away ("lose the canvas").
        """
        qx, qy = self._doc_to_canvas(anchor[0], anchor[1])
        left, top, right, bottom = self._scroll
        view_w = max(1, self.winfo_width())
        view_h = max(1, self.winfo_height())
        nleft = min(left, qx)
        ntop = min(top, qy)
        nright = max(right, qx + view_w)
        nbottom = max(bottom, qy + view_h)
        if (nleft, ntop, nright, nbottom) != (left, top, right, bottom):
            self._scroll = (nleft, ntop, nright, nbottom)
            self.config(scrollregion=self._scroll)
            left, top, right, bottom = self._scroll
        span_x = right - left
        span_y = bottom - top
        if span_x > 0:
            self.xview_moveto((qx - left) / span_x)
        if span_y > 0:
            self.yview_moveto((qy - top) / span_y)

    # ------------------------------------------------------- view transform
    @property
    def transformed(self):
        """Whether the view is currently rotated or flipped.

        The exploded grid lays layers out in fixed cells and does not share the
        document's coordinate space, so the transform is suspended while it is
        shown.
        """
        if self._layout is not None:
            return False
        return self.view_flip or self.view_flip_v or abs(self.view_angle) > 1e-9

    @property
    def rotated(self):
        """Whether the view is rotated (a flip alone keeps the axes aligned).

        The rulers annotate document axes. A mirror keeps vertical guides
        vertical and horizontal ones horizontal, so the ruler mapping still
        makes sense (mirrored); only a rotation has no axis-aligned document
        frame to label.
        """
        if self._layout is not None:
            return False
        return abs(self.view_angle) > 1e-9

    def _pivot_doc(self):
        """The document point rotation and flips happen about."""
        if self._view_pivot is not None:
            return self._view_pivot
        return (self.doc.width / 2.0, self.doc.height / 2.0)

    def _pivot_canvas(self):
        """The pivot in canonical canvas coordinates (``doc * zoom``)."""
        px, py = self._pivot_doc()
        return px * self.zoom, py * self.zoom

    def _matrix(self):
        """The rigid view matrix and its transpose (inverse).

        ``M = R(angle) @ F`` where ``F = diag(sx, sy)`` mirrors each axis
        independently (``sx``/``sy`` are ``-1`` for a horizontal/vertical flip,
        ``+1`` otherwise) and ``R`` is a rotation. Both are orthogonal, so the
        inverse is simply the transpose. Coordinates are in canonical canvas
        space (y points down), so a positive angle reads as a clockwise turn on
        screen. The two flips together give ``F = -I``, a 180 degree turn, so
        the determinant is ``sx * sy`` (``-1`` for a single flip).
        """
        c = math.cos(self.view_angle)
        s = math.sin(self.view_angle)
        sx = -1.0 if self.view_flip else 1.0
        sy = -1.0 if self.view_flip_v else 1.0
        m00, m01, m10, m11 = c * sx, -s * sy, s * sx, c * sy
        # Transpose.
        return m00, m01, m10, m11, m00, m10, m01, m11

    def _xform_pt(self, x, y):
        """Map a canonical canvas point through the view transform."""
        if not self.transformed:
            return x, y
        px, py = self._pivot_canvas()
        m00, m01, m10, m11, _i00, _i01, _i10, _i11 = self._matrix()
        dx, dy = x - px, y - py
        return px + m00 * dx + m01 * dy, py + m10 * dx + m11 * dy

    def _unxform_pt(self, x, y):
        """Map a canvas point back to canonical canvas space."""
        if not self.transformed:
            return x, y
        px, py = self._pivot_canvas()
        _m00, _m01, _m10, _m11, i00, i01, i10, i11 = self._matrix()
        dx, dy = x - px, y - py
        return px + i00 * dx + i01 * dy, py + i10 * dx + i11 * dy

    def _view_canvas_bounds(self, pad=0):
        """The canonical canvas rect the viewport covers, as ``(l, t, r, b)``.

        Without a transform this is just the scrolled viewport; with one the
        viewport's four corners are inverse-mapped and boxed, which is the region
        a grid or guide line must span so it crosses the whole screen.
        """
        vw = self.winfo_width()
        vh = self.winfo_height()
        x0, y0 = self.canvasx(0), self.canvasy(0)
        if not self.transformed:
            return (x0 - pad, y0 - pad, x0 + vw + pad, y0 + vh + pad)
        corners = ((x0, y0), (x0 + vw, y0), (x0 + vw, y0 + vh), (x0, y0 + vh))
        mapped = [self._unxform_pt(cx, cy) for cx, cy in corners]
        left = min(p[0] for p in mapped) - pad
        top = min(p[1] for p in mapped) - pad
        right = max(p[0] for p in mapped) + pad
        bottom = max(p[1] for p in mapped) + pad
        return (left, top, right, bottom)

    def _doc_rect_to_canvas_bbox(self, left, top, right, bottom):
        """Box the four transformed corners of a document rect."""
        corners = (
            self._doc_to_canvas(left, top), self._doc_to_canvas(right, top),
            self._doc_to_canvas(right, bottom), self._doc_to_canvas(left, bottom),
        )
        return (
            min(p[0] for p in corners), min(p[1] for p in corners),
            max(p[0] for p in corners), max(p[1] for p in corners),
        )

    # ------------------------------------------------- transform-aware draw
    # Overlays are drawn in canonical canvas coordinates (what ``_doc_to_view``
    # returns) and go through these wrappers, which apply the rigid view
    # transform to every primitive. With no transform they delegate straight to
    # Tk, so the common path is unchanged. A rectangle becomes a polygon because
    # a rotated rectangle is not axis-aligned; an ellipse is sampled for the
    # same reason (a circle stays a circle).
    def _flat_coords(self, args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return list(args[0])
        return list(args)

    def _transform_points(self, coords):
        out = []
        for i in range(0, len(coords) - 1, 2):
            vx, vy = self._xform_pt(coords[i], coords[i + 1])
            out.append(vx)
            out.append(vy)
        return out

    def _transparent_default(self, kwargs):
        """Ensure a converted shape keeps Tk's transparent default interior.

        A Tk polygon reports an opaque default ``-fill`` when the option is
        omitted, whereas the rectangle/oval it replaces had no interior. Copy the
        options and force ``fill=""`` unless the caller chose one.
        """
        kwargs = dict(kwargs)
        kwargs.setdefault("fill", "")
        return kwargs

    def create_line(self, *args, **kwargs):
        if self.transformed:
            points = self._transform_points(self._flat_coords(args))
            return tk.Canvas.create_line(self, *points, **kwargs)
        return tk.Canvas.create_line(self, *args, **kwargs)

    def create_polygon(self, *args, **kwargs):
        if self.transformed:
            points = self._transform_points(self._flat_coords(args))
            kwargs = self._transparent_default(kwargs)
            return tk.Canvas.create_polygon(self, *points, **kwargs)
        return tk.Canvas.create_polygon(self, *args, **kwargs)

    def create_rectangle(self, *args, **kwargs):
        if not self.transformed:
            return tk.Canvas.create_rectangle(self, *args, **kwargs)
        x0, y0, x1, y1 = args[0], args[1], args[2], args[3]
        points = self._transform_points(
            (x0, y0, x1, y0, x1, y1, x0, y1)
        )
        kwargs = self._transparent_default(kwargs)
        return tk.Canvas.create_polygon(self, *points, **kwargs)

    def create_oval(self, *args, **kwargs):
        if not self.transformed:
            return tk.Canvas.create_oval(self, *args, **kwargs)
        x0, y0, x1, y1 = args[0], args[1], args[2], args[3]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        rx, ry = abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0
        vx, vy = self._xform_pt(cx, cy)
        kwargs = self._transparent_default(kwargs)
        if abs(rx - ry) < 1e-9:
            return tk.Canvas.create_oval(
                self, vx - rx, vy - ry, vx + rx, vy + ry, **kwargs
            )
        points = []
        steps = 48
        for i in range(steps):
            a = 2.0 * math.pi * i / steps
            px, py = self._xform_pt(cx + rx * math.cos(a), cy + ry * math.sin(a))
            points.append(px)
            points.append(py)
        return tk.Canvas.create_polygon(self, *points, **kwargs)

    def create_text(self, x, y, **kwargs):
        if self.transformed:
            vx, vy = self._xform_pt(x, y)
            kwargs = dict(kwargs)
            # Tk's angle is counter-clockwise; the view angle is clockwise.
            kwargs["angle"] = kwargs.get("angle", 0) - math.degrees(self.view_angle)
            return tk.Canvas.create_text(self, vx, vy, **kwargs)
        return tk.Canvas.create_text(self, x, y, **kwargs)

    def _doc_to_view(self, x, y):
        """Convert a document coordinate to a canonical view pixel.

        In exploded view the active layer's cell is the editing frame, so the
        document coordinate is shifted into that cell before scaling; the
        overlays (cursor, gizmo, selection) then land in the right cell. The
        result is *before* the rigid view transform (rotation/flip): the drawing
        wrappers apply that, so an overlay call site keeps calling this and gets
        a correctly rotated item. Use :meth:`_doc_to_canvas` where the final
        transformed position is needed directly (scroll anchors, coverage).
        """
        if self._layout is not None:
            sx, sy = self._active_shift()
            x -= sx
            y -= sy
        return x * self.zoom, y * self.zoom

    def _doc_to_canvas(self, x, y):
        """Workspace coordinate to the canvas position it is actually drawn at.

        Workspace coordinates equal document coordinates in the normal view; in
        exploded view they are already *layout* coordinates. The active layer's
        document-to-cell shift belongs only to :meth:`_doc_to_view`, where a
        document coordinate (an overlay position or the pointer) must land in
        its cell. Applying it here as well would offset the scroll anchor, the
        scroll region and :meth:`fit_document_rect` by that layer's cell origin,
        leaving the exploded board displaced from the viewport. This is the
        exact inverse of :meth:`_view_to_doc_float`.
        """
        bx, by = x * self.zoom, y * self.zoom
        if self.transformed:
            return self._xform_pt(bx, by)
        return bx, by

    def _draw_doc_border(self):
        self.delete("docborder")
        if self._layout is not None:
            # One border per cell so each layer reads as its own tile, with the
            # cell being edited picked out in the accent color.
            active = getattr(self.doc, "active", -1)
            accent = theme.color("accent") or "#3b82f6"
            for index, (left, top, right, bottom) in enumerate(self._layout.cells):
                chosen = index == active
                self.create_rectangle(
                    left * self.zoom, top * self.zoom,
                    right * self.zoom - 1, bottom * self.zoom - 1,
                    outline=accent if chosen else "#4a4a4a",
                    width=2 if chosen else 1, tags="docborder",
                )
            return
        left, top = self._doc_to_view(0, 0)
        right, bottom = self._doc_to_view(self.doc.width, self.doc.height)
        self.create_rectangle(
            left, top, right - 1, bottom - 1,
            outline="#4a4a4a", width=1, tags="docborder",
        )
        self._draw_layer_outline()

    def _draw_layer_outline(self):
        """Outline the active layer's bounds on the stacked canvas.

        Drawn as a dashed rectangle in the same accent color the exploded view
        uses for its selected cell, so the active layer is identifiable in the
        normal composited view. It shares the ``docborder`` tag and is skipped
        in exploded view (which already marks the active cell).
        """
        if not self.layer_outline_visible or self._layout is not None:
            return
        layers = getattr(self.doc, "layers", None)
        index = getattr(self.doc, "active", -1)
        if not layers:
            return
        group = getattr(self, "outline_group", None)
        targets = [l for l in layers if group is not None and l.group is group]
        if not targets:
            if not (0 <= index < len(layers)):
                return
            targets = [layers[index]]
        box = None
        for layer in targets:
            width, height = layer.size
            if width <= 0 or height <= 0:
                continue
            bbox = self._layer_alpha_bbox(layer)
            if bbox is None:
                continue
            ox, oy = layer.offset
            rect = (ox + bbox[0], oy + bbox[1], ox + bbox[2], oy + bbox[3])
            box = rect if box is None else (
                min(box[0], rect[0]), min(box[1], rect[1]),
                max(box[2], rect[2]), max(box[3], rect[3]),
            )
        if box is None:
            return
        x0, y0 = self._doc_to_view(box[0], box[1])
        x1, y1 = self._doc_to_view(box[2], box[3])
        accent = theme.color("accent") or "#3b82f6"
        self.create_rectangle(
            x0, y0, x1, y1, outline=accent, width=2, dash=(6, 4),
            tags="docborder",
        )

    def _layer_alpha_bbox(self, layer):
        """Bounding box of the layer's visible (alpha > 0) content, or ``None``.

        Measured on :meth:`Layer.composited_image` so texts, vector strokes,
        fills, paths, poses, filters and the mask all count, not only the raw
        pixels. ``composited_image`` returns the same cached object while the
        content is unchanged, so the bbox is cached against that identity and
        the alpha scan only reruns after an edit.
        """
        try:
            image = layer.composited_image()
        except Exception:
            return (0, 0) + tuple(layer.size)
        cache = getattr(self, "_outline_bbox_cache", None)
        if not isinstance(cache, dict):
            cache = self._outline_bbox_cache = {}
        key = getattr(layer, "uid", id(layer))
        cached = cache.get(key)
        if cached is not None and cached[0] is image:
            return cached[1]
        if "A" in image.getbands():
            bbox = image.getchannel("A").getbbox()
        else:
            bbox = (0, 0) + image.size
        if len(cache) > 256:
            cache.clear()
        cache[key] = (image, bbox)
        return bbox

    def set_layer_outline(self, visible):
        """Show or hide the active-layer outline for this view."""
        self.layer_outline_visible = bool(visible)
        self._draw_doc_border()

    # -------------------------------------------------------------------- grid
    def set_grid(self, visible=None, size=None):
        """Show or hide the grid overlay and set its cell size.

        ``size`` is the grid spacing in document pixels; both arguments are
        optional so a caller can change just one. The overlay is clipped to the
        canvas rectangle and the visible viewport.
        """
        if visible is not None:
            self.grid_visible = bool(visible)
        if size is not None:
            try:
                size = int(size)
            except (TypeError, ValueError):
                size = self.grid_size
            self.grid_size = max(1, size)
        self._draw_grid()

    def set_grid_snap(self, enabled):
        """Allow (``True``) or forbid (``False``) rounding coordinates to the grid."""
        self.grid_snap = bool(enabled)

    def snap_doc(self, x, y):
        """Round a document point to the nearest grid intersection.

        Returns the point unchanged when grid snapping is off, so callers can
        always route their coordinates through here.
        """
        if not self.grid_snap:
            return x, y
        if self._layout is not None:
            # The grid is drawn in document space and is hidden in exploded
            # view, so snapping there would be meaningless.
            return x, y
        step = max(1, self.grid_size)
        return int(round(x / step)) * step, int(round(y / step)) * step

    def _draw_grid(self):
        """Draw the grid overlay over the canvas for the visible viewport."""
        if not self.winfo_exists():
            return
        self.delete("grid")
        if not self.grid_visible or self._layout is not None:
            return
        zoom = self.zoom
        spacing = max(1, self.grid_size) * zoom
        if spacing < GRID_MIN_PIXELS:
            # Too dense to read at this zoom; skipping beats tinting the canvas.
            return
        # The grid lives in document space; clip it to both the canvas and the
        # canonical region the (possibly rotated) viewport covers, then let the
        # draw wrappers rotate each line.
        left, top, right, bottom = self._view_canvas_bounds()
        left = max(left, 0.0)
        top = max(top, 0.0)
        right = min(right, self.doc.width * zoom)
        bottom = min(bottom, self.doc.height * zoom)
        if left >= right or top >= bottom:
            return
        color = theme.color("grid_line") or DEFAULT_GRID_COLOR
        # First grid line at or past the visible left/top edge.
        x = math.ceil(left / spacing - 1e-9) * spacing
        while x <= right + 1e-9:
            self.create_line(x, top, x, bottom, fill=color, tags="grid")
            x += spacing
        y = math.ceil(top / spacing - 1e-9) * spacing
        while y <= bottom + 1e-9:
            self.create_line(left, y, right, y, fill=color, tags="grid")
            y += spacing

    # ------------------------------------------------------------------ guides
    @property
    def guides(self):
        """The guide list this view draws and edits.

        Normally the document's own list; when the app shares guides across
        canvases, the shared list the app installed instead (see
        :meth:`set_shared_guides`).
        """
        if self._shared_guides is not None:
            return self._shared_guides
        return self.doc.guides

    def set_shared_guides(self, guides):
        """Use ``guides`` (a shared list) instead of the document's own, or ``None``."""
        self._shared_guides = guides
        self._draw_guides()
        self._redraw_rulers()

    def set_guides_visible(self, visible):
        """Show or hide the guides (and their ruler handles) for this view."""
        self.guides_visible = bool(visible)
        self._draw_guides()
        self._redraw_rulers()

    def guide_hit(self, x, y, tolerance):
        """Index of the guide within ``tolerance`` of document ``(x, y)``."""
        for index, (orient, pos) in enumerate(self.guides):
            near = abs(x - pos) if orient == "v" else abs(y - pos)
            if near <= tolerance:
                return index
        return None

    def push_guide_undo(self):
        """Ask the app to snapshot the document before a guide edit."""
        if callable(self.on_guides_changed):
            self.on_guides_changed()

    def add_guide(self, orient, pos):
        """Add a guide at document ``pos`` and redraw overlay and handles."""
        self.guides.append((orient, int(round(pos))))
        self._guides_edited()

    def set_guide_pos(self, index, pos):
        """Move an existing guide along its axis and redraw overlay and handles."""
        orient = self.guides[index][0]
        self.guides[index] = (orient, int(round(pos)))
        self._guides_edited()

    def pop_guide(self, index):
        """Remove an existing guide and redraw overlay and handles."""
        del self.guides[index]
        self._guides_edited()

    def _guides_edited(self):
        """Repaint this view's guides and let the area refresh the other tabs."""
        self._draw_guides()
        self._redraw_rulers()
        if callable(self.on_guides_edited):
            self.on_guides_edited(self)

    def set_pending_guide(self, orient, pos):
        """Show the guide being dragged out of a ruler, before it is dropped."""
        self._pending_guide = (orient, int(round(pos)))
        self._draw_guides()

    def clear_pending_guide(self):
        wanted, self._pending_guide = self._pending_guide, None
        if wanted is not None:
            self._draw_guides()

    def _draw_guides(self):
        if not self.winfo_exists():
            return
        self.delete("guide")
        if not self.guides_visible or self._layout is not None:
            return
        color = theme.color("guide_line") or DEFAULT_GUIDE_COLOR
        for orient, pos in self.guides:
            self._draw_guide_line(orient, pos, color, None)
        if self._pending_guide is not None:
            orient, pos = self._pending_guide
            self._draw_guide_line(orient, pos, color, (4, 3))

    def _draw_guide_line(self, orient, pos, color, dash):
        """Draw one guide across the visible viewport, in canvas coordinates."""
        left, top, right, bottom = self._view_canvas_bounds()
        if orient == "v":
            x = pos * self.zoom
            self.create_line(
                x, top, x, bottom, fill=color, dash=dash, tags="guide",
            )
        else:
            y = pos * self.zoom
            self.create_line(
                left, y, right, y, fill=color, dash=dash, tags="guide",
            )

    # ------------------------------------------------------------------ mask
    def set_mask_preview(self, mask, origin=(0, 0)):
        """Show ``mask`` (an ``"L"`` image) as a translucent veil, or hide it.

        ``origin`` is the document position of the mask's top-left, so a mask in
        a layer's own buffer frame (a link's) lands over the pixels it shapes.
        """
        self._mask_preview = mask
        self._mask_origin = tuple(origin)
        self._draw_mask_preview()

    def set_mask_overlay_color(self, rgb):
        """Tint the mask preview veil with ``rgb`` (an ``(r, g, b)`` tuple)."""
        rgb = tuple(rgb)
        if rgb != self._mask_overlay:
            self._mask_overlay = rgb
            self._draw_mask_preview()

    def _draw_mask_preview(self):
        self.delete("maskpreview")
        if self._mask_preview is None:
            return
        if self._layout is not None:
            # The mask belongs to a layer's own space; a single document-space
            # veil does not map onto the exploded grid.
            return
        if self.transformed:
            # A rotated view cannot place an axis-aligned veil item; it is
            # composited into the warped board instead (see
            # :meth:`_overlay_mask_veil`).
            return
        if self._render_window is not None:
            top_left = self._render_window
            view_size = self._render_size
        else:
            view_w = self.winfo_width()
            view_h = self.winfo_height()
            if view_w <= 1 or view_h <= 1:
                view_w, view_h = self._content_w, self._content_h
            top_left = (self.canvasx(0) / self.zoom, self.canvasy(0) / self.zoom)
            view_size = (view_w, view_h)
        mask, item = visible_crop(
            self._mask_preview, self._mask_origin, self.zoom, top_left, view_size
        )
        if mask is None:
            return
        red, green, blue = self._mask_overlay
        veil = Image.new("RGBA", mask.size, (red, green, blue, 0))
        veil.putalpha(mask.point(lambda value: int((255 - value) * 0.5)))
        self._mask_image = ImageTk.PhotoImage(veil)
        self.create_image(
            item[0], item[1], anchor=tk.NW, image=self._mask_image,
            tags="maskpreview",
        )

    # ------------------------------------------------------------------ preview
    def set_generation_preview(self, image, origin=(0, 0), fit=True, scale=1.0):
        """Show a finished generation in the viewport before it is applied.

        ``image`` is drawn at document ``origin`` over the dimmed document, so
        the view's own zoom and pan inspect it like any other content; with
        ``fit`` the view zooms and scrolls to show the whole canvas-and-preview
        rectangle. Passing ``None`` clears the preview. The same call renders the
        live denoising frames, so what forms on screen is what the result will
        look like.

        ``scale`` is ``image``'s pixels per document pixel. It is ``1.0`` for a
        document-sized result, and greater than one for an upscaled inpaint crop
        (see ``_plan_inpaint_crop``), so the crop is placed back at the document
        region it stands in for instead of at its raw pixel size.
        """
        if image is None:
            self.clear_generation_preview()
            return
        scale = float(scale) if scale and scale > 0 else 1.0
        self._gen_preview = (
            image, (int(origin[0]), int(origin[1])), scale,
        )
        if fit:
            # Zoom and scroll so the whole canvas-and-preview rectangle is on
            # screen straight away; the view's zoom controls inspect it after.
            self.fit_document_rect(self.preview_fit_bounds())
        else:
            self.render()

    def clear_generation_preview(self):
        """Hide the generation preview and restore the scroll region."""
        had = self._gen_preview is not None
        self._gen_preview = None
        if had and self._source is not None:
            self.render()

    def has_generation_preview(self):
        return self._gen_preview is not None

    def preview_fit_bounds(self):
        """The rectangle a preview should be framed with, for the current view.

        The document rectangle of the canvas unioned with the preview in the
        stacked view, or the active cell unioned with the preview in the
        exploded grid (where the preview maps onto the active layer's tile).
        """
        if self._layout is not None:
            bounds = self._preview_layout_bounds()
            if bounds is not None:
                return bounds
        return self._preview_bounds()

    def _preview_bounds(self):
        """Document rectangle of the canvas unioned with the preview."""
        if self._gen_preview is None:
            return None
        image, (ox, oy), scale = self._gen_preview
        width = image.width / scale
        height = image.height / scale
        return (
            min(0, ox), min(0, oy),
            max(self.doc.width, ox + width),
            max(self.doc.height, oy + height),
        )

    def _preview_layout_bounds(self):
        """Layout rectangle of the active cell unioned with the preview, or None.

        The exploded-grid counterpart of :meth:`_preview_bounds`: the preview is
        shown over the active layer's tile, so framing that cell is what puts it
        on screen after a generation.
        """
        origin = self._preview_layout_origin()
        if origin is None:
            return None
        image, _doc_origin, scale = self._gen_preview
        width = image.width / scale
        height = image.height / scale
        index = getattr(self.doc, "active", 0)
        if not (0 <= index < len(self._layout.cells)):
            return None
        left, top, right, bottom = self._layout.cells[index]
        return (
            min(left, origin[0]), min(top, origin[1]),
            max(right, origin[0] + width),
            max(bottom, origin[1] + height),
        )

    def _preview_layout_origin(self):
        """The preview's top-left in layout space, centred on the active cell.

        The preview is recorded in document space (see
        :meth:`set_generation_preview`), which an exploded grid does not share.
        This maps it onto the active layer's tile - the cell that would receive
        it - so it stays visible while the grid is shown. Returns ``None`` when
        the grid or the preview is not currently shown.
        """
        if self._layout is None or self._gen_preview is None:
            return None
        image, _doc_origin, scale = self._gen_preview
        width = image.width / scale
        height = image.height / scale
        index = getattr(self.doc, "active", 0)
        if not (0 <= index < len(self._layout.cells)):
            return None
        left, top, right, bottom = self._layout.cells[index]
        return (
            int(left + (right - left - width) // 2),
            int(top + (bottom - top - height) // 2),
        )

    def _preview_fit_zoom(self, rect, pad=24):
        """The zoom that fits document ``rect`` in the current viewport."""
        view_w = self.winfo_width()
        view_h = self.winfo_height()
        if rect is None or view_w <= 1 or view_h <= 1:
            return self.zoom
        width = max(1.0, rect[2] - rect[0])
        height = max(1.0, rect[3] - rect[1])
        zoom = min((view_w - pad) / width, (view_h - pad) / height)
        return max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))

    def fit_document_rect(self, rect, pad=24):
        """Zoom and scroll so document ``rect`` is centred in the viewport."""
        if rect is None or not self.winfo_exists():
            return
        view_w = self.winfo_width()
        view_h = self.winfo_height()
        if view_w <= 1 or view_h <= 1:
            return
        x0, y0, x1, y1 = rect
        zoom = self._preview_fit_zoom(rect, pad)
        self.zoom = zoom
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        qx, qy = self._doc_to_canvas(center_x, center_y)
        self.render(top_left=self._view_to_doc_float(
            qx - view_w / 2.0, qy - view_h / 2.0,
        ))

    def _overlay_generation_preview(self, board, window, render_size, origin=None):
        """Dim ``board`` and lay the un-dimmed preview over it, in one image.

        The preview used to be drawn as two separate canvas items: a
        full-viewport stippled veil and the preview image on top. Both had to
        be deleted and rebuilt on every scroll, and the stippled rectangle is
        slow to repaint, which made the preview flicker while zooming or
        panning. Compositing the veil and the preview into the single base
        image removes that redraw: the one item that remains scrolls and
        rescales like the rest of the canvas.

        ``origin`` overrides the document-space position recorded with the
        preview; the exploded grid passes the layout-space origin of the active
        cell so the preview lands on the tile instead of the document.
        """
        if self._gen_preview is None:
            return board
        board = Image.blend(board, Image.new("RGB", board.size, (0, 0, 0)), 0.5)
        image, doc_origin, scale = self._gen_preview
        if origin is None:
            origin = doc_origin
        crop, item = visible_crop(
            image, origin, self.zoom, window, render_size, source_scale=scale,
        )
        if crop is None:
            return board
        position = (
            int(round(item[0] - window[0] * self.zoom)),
            int(round(item[1] - window[1] * self.zoom)),
        )
        if crop.mode == "RGBA":
            board.paste(crop, position, crop)
        else:
            board.paste(crop, position)
        return board

    # ------------------------------------------------------------------ zoom
    # Delay after the last wheel zoom before the exact repaint replaces the
    # gesture preview.
    ZOOM_REFINE_MS = 180

    def _zoom_preview_source(self):
        """A whole-content composite at a power-of-two scale, for zoom gestures.

        Zoomed out, every zoom step otherwise re-flattens each layer of the
        viewport at the new scale. During a wheel gesture the view instead
        resamples one cached flatten of the whole content (at the pyramid level
        just above the zoom), which costs a single crop and resize per step
        however many layers the document has; once the wheel stops,
        :meth:`_refine_zoom` repaints exactly. The cache is keyed on every
        layer's signature, so any edit rebuilds it. Returns ``(image, origin,
        scale)`` or ``None`` when there is nothing to use.
        """
        doc = self.doc
        if doc is None or self._region_provider is None:
            return None
        level = 1
        while level < 5 and self.zoom <= 0.5 ** (level + 1):
            level += 1
        scale = 0.5 ** level
        (ox, oy), (cw, ch) = self._content_origin, self._content_size
        signature = getattr(doc, "_layer_signature", None)
        layers = getattr(doc, "layers", None) or []
        try:
            stack = tuple(signature(layer) for layer in layers) if signature else None
        except Exception:
            return None
        if stack is None:
            return None
        key = (id(doc), self._region_provider, (ox, oy, cw, ch),
               tuple(self._frame_size), stack)
        cache = getattr(self, "_zoom_preview_cache", None)
        if cache is None or cache["key"] != key:
            cache = {"key": key, "levels": {}}
            self._zoom_preview_cache = cache
        levels = cache["levels"]
        entry = levels.get(level)
        if entry is None and levels:
            finer = [n for n in levels if n < level]
            if finer:
                # Halve the nearest finer level instead of flattening again.
                have = max(finer)
                image, origin = levels[have]
                for n in range(have + 1, level + 1):
                    image = image.reduce(2)
                    levels[n] = (image, origin)
                entry = levels[level]
            else:
                # Only coarser levels exist (zooming in from far out): upsample
                # the nearest one for the gesture; the refine repaints sharp.
                have = min(levels)
                entry = levels[have]
                level = have
        if entry is None:
            region = (int(math.floor(ox)), int(math.floor(oy)),
                      int(math.ceil(ox + cw)), int(math.ceil(oy + ch)))
            result = self._flatten(region, scale, draft=True)
            if result is None:
                return None
            image, origin = result
            if image.width <= 0 or image.height <= 0:
                return None
            entry = (image, origin)
            levels[level] = entry
        image, origin = entry
        return image, origin, 0.5 ** level

    def _cancel_zoom_refine(self):
        job = getattr(self, "_zoom_refine_job", None)
        self._zoom_refine_job = None
        if job is not None:
            try:
                self.after_cancel(job)
            except tk.TclError:
                pass

    def _refine_zoom(self):
        self._zoom_refine_job = None
        try:
            if self.winfo_exists():
                self.render()
        except tk.TclError:
            pass

    def set_zoom(self, factor, focus=None, preview=False):
        """Set the zoom factor, optionally keeping ``focus`` (an event) fixed.

        ``preview`` (a wheel gesture) paints a fast approximation when zoomed
        out and schedules the exact repaint for when the gesture pauses.
        """
        factor = max(self.MIN_ZOOM, min(self.MAX_ZOOM, factor))
        if abs(factor - self.zoom) < 1e-9:
            return
        if focus is not None:
            focus_doc = self._view_to_doc_float(
                self.canvasx(focus.x), self.canvasy(focus.y)
            )
            self.zoom = factor
            # Keep the document point under the cursor pinned: put its new
            # canvas position back at the same widget pixel.
            qx, qy = self._doc_to_canvas(*focus_doc)
            top_left = self._view_to_doc_float(qx - focus.x, qy - focus.y)
        else:
            top_left = self._view_to_doc_float(self.canvasx(0), self.canvasy(0))
            self.zoom = factor
        use_preview = preview and factor < 1.0 and not self.transformed
        self.render(top_left=top_left, zoom_preview=use_preview)
        if use_preview:
            self._cancel_zoom_refine()
            self._zoom_refine_job = self.after(self.ZOOM_REFINE_MS, self._refine_zoom)

    def zoom_in(self, focus=None):
        self.set_zoom(self.zoom * self.ZOOM_STEP, focus)

    def zoom_out(self, focus=None):
        self.set_zoom(self.zoom / self.ZOOM_STEP, focus)

    def reset_zoom(self):
        self.set_zoom(1.0)

    # --------------------------------------------------------- view rotation
    def _set_pivot_to_view_center(self):
        """Pivot the view transform about the point under the viewport centre."""
        vw = self.winfo_width()
        vh = self.winfo_height()
        if vw <= 1 or vh <= 1:
            self._view_pivot = None
            return
        self._view_pivot = self._view_to_doc_float(
            self.canvasx(vw / 2.0), self.canvasy(vh / 2.0)
        )

    def begin_view_rotate(self):
        """Pick the rotation pivot (the viewport centre) for a drag-rotate.

        Returns the pivot in view (widget) coordinates so the caller can measure
        the pointer's angle about it.
        """
        self._set_pivot_to_view_center()
        return self.winfo_width() / 2.0, self.winfo_height() / 2.0

    def _apply_view_change(self, mutate, pivot=None):
        """Apply a camera change while keeping the pivot fixed on screen.

        Records where the pivot currently sits in the viewport, applies
        ``mutate`` (which edits ``view_angle``/``view_flip``/``view_flip_v``),
        then renders anchored so that same document point returns to the same
        screen position. That is what makes the camera turn/flip on itself:
        otherwise the transform is anchored to a fixed *document* point, and
        once the user pans away from it the artwork swings around that
        off-centre point (the "canvas" rotating) instead of the camera turning
        in place. ``pivot`` overrides the pivot (a document point) for this
        change; the default is the point under the viewport centre.
        """
        vw = self.winfo_width()
        vh = self.winfo_height()
        if vw <= 1 or vh <= 1:
            return
        if pivot is not None:
            self._view_pivot = pivot
        if self._view_pivot is None:
            self._set_pivot_to_view_center()
        # Where the pivot sits in the viewport before the change.
        px, py = self._pivot_doc()
        qx, qy = self._doc_to_canvas(px, py)
        view_x, view_y = qx - self.canvasx(0), qy - self.canvasy(0)
        mutate()
        # Re-anchor so the pivot returns to that same screen position.
        rx, ry = self._doc_to_canvas(px, py)
        top_left = self._view_to_doc_float(rx - view_x, ry - view_y)
        self.render(top_left=top_left)

    def rotate_view(self, degrees):
        """Rotate the camera clockwise by ``degrees`` about the viewport centre."""
        if self._layout is not None:
            return

        def mutate():
            self.view_angle = (
                self.view_angle + math.radians(degrees)
            ) % (2.0 * math.pi)
            # Back to an unrotated view: a live flip must mirror about the
            # document centre so the ascending ruler keeps its origin at the
            # canvas corner (see :meth:`_apply_view_change`).
            if abs(self.view_angle) <= 1e-9 and (self.view_flip or self.view_flip_v):
                self._view_pivot = (self.doc.width / 2.0, self.doc.height / 2.0)

        # A rotation turns about the viewport centre, so drop any pivot a flip
        # left at the document centre.
        self._set_pivot_to_view_center()
        self._apply_view_change(mutate)

    def set_view_angle(self, radians_angle):
        """Set the view rotation directly (used while drag-rotating)."""
        if self._layout is not None:
            return
        self.view_angle = radians_angle % (2.0 * math.pi)
        if abs(self.view_angle) <= 1e-9 and (self.view_flip or self.view_flip_v):
            self._view_pivot = (self.doc.width / 2.0, self.doc.height / 2.0)
        self.render()

    def flip_view(self):
        """Mirror the artwork horizontally about the document centre.

        The canvas origin stays put and the ruler keeps counting up from the
        top-left corner; only the artwork swaps sides.
        """
        if self._layout is not None:
            return

        def mutate():
            self.view_flip = not self.view_flip

        self._apply_view_change(
            mutate, pivot=(self.doc.width / 2.0, self.doc.height / 2.0)
        )

    def flip_view_vertical(self):
        """Mirror the artwork vertically about the document centre."""
        if self._layout is not None:
            return

        def mutate():
            self.view_flip_v = not self.view_flip_v

        self._apply_view_change(
            mutate, pivot=(self.doc.width / 2.0, self.doc.height / 2.0)
        )

    def reset_view(self):
        """Clear the view rotation and both flips, keeping zoom and pan."""
        if not (self.view_flip or self.view_flip_v
                or abs(self.view_angle) >= 1e-9):
            return
        self.view_angle = 0.0
        self.view_flip = False
        self.view_flip_v = False
        self._view_pivot = None
        self.render()

    def view_state(self):
        """The camera (zoom, anchor, transform) for :meth:`set_view_state`."""
        return {
            "zoom": self.zoom,
            "anchor": self._view_to_doc_float(self.canvasx(0), self.canvasy(0)),
            "angle": self.view_angle,
            "flip": self.view_flip,
            "flip_v": self.view_flip_v,
            "pivot": self._view_pivot,
        }

    def set_view_state(self, state):
        """Restore a camera saved by :meth:`view_state`.

        Used to keep the camera put across an undo/redo rebuild: the anchor is
        applied on the next successful render, so a view recreated before it has
        been laid out still lands on the right document pixels once it is shown.
        Also restores the view rotation/flip. Accepts the legacy ``(zoom,
        anchor)`` tuple for state saved by older code.
        """
        if isinstance(state, dict):
            self.view_angle = float(state.get("angle", 0.0))
            self.view_flip = bool(state.get("flip", False))
            self.view_flip_v = bool(state.get("flip_v", False))
            self._view_pivot = state.get("pivot")
            zoom = state.get("zoom", self.zoom)
            anchor = state.get("anchor", (0.0, 0.0))
        else:
            zoom, anchor = state
        self.zoom = max(self.MIN_ZOOM, min(self.MAX_ZOOM, zoom))
        self._pending_anchor = (float(anchor[0]), float(anchor[1]))

    # ------------------------------------------------------------------ coords
    def _view_to_doc_float(self, x, y):
        """Like :meth:`view_to_doc` but without rounding to whole pixels."""
        if self.transformed:
            x, y = self._unxform_pt(x, y)
        return x / self.zoom, y / self.zoom

    def view_to_doc(self, x, y):
        """Convert scaled view coordinates to workspace pixel coordinates.

        Workspace coordinates equal document coordinates in the normal view; in
        exploded view they are *layout* coordinates (see :meth:`event_to_doc` for
        the active layer's document coordinates). A rotated/flipped view is
        undone first, so this is the exact inverse of :meth:`_doc_to_view`.
        """
        if self.transformed:
            x, y = self._unxform_pt(x, y)
        return int(round(x / self.zoom)), int(round(y / self.zoom))

    def event_to_doc(self, event):
        """Convert a widget event position to document pixel coordinates.

        In exploded view the pointer is mapped from its cell back to the active
        layer's document coordinates (via the layout shift) so tools address the
        layer being edited.
        """
        x, y = self.view_to_doc(self.canvasx(event.x), self.canvasy(event.y))
        if self._layout is not None:
            sx, sy = self._active_shift()
            x += sx
            y += sy
        return x, y

    # ------------------------------------------------------------------ cursor
    def set_cursor_overlay(self, x, y=None, radius=0, tip="round"):
        """Set the brush overlay; coordinates are in document pixels."""
        self._cursor = None if x is None else (x, y, radius, tip)
        self._draw_cursor()

    def _draw_cursor(self):
        self.delete("brush")
        if self._cursor is None:
            return
        x, y, r, tip = self._cursor
        cx, cy = self._doc_to_view(x, y)
        cr = r * self.zoom
        if tip == "square":
            self.create_rectangle(
                cx - cr, cy - cr, cx + cr, cy + cr,
                outline="#555555", dash=(3, 2), width=1, tags="brush",
            )
        else:
            self.create_oval(
                cx - cr, cy - cr, cx + cr, cy + cr,
                outline="#555555", dash=(3, 2), width=1, tags="brush",
            )

    # -------------------------------------------------------- clone source
    def set_clone_source_overlay(self, pos):
        """Show a persistent crosshair at the clone source ``pos``, or hide it.

        ``pos`` is a document-pixel ``(x, y)`` pair; ``None`` clears the marker.
        """
        self._clone_source = None if pos is None else (pos[0], pos[1])
        self._draw_clone_source()

    def _draw_clone_source(self):
        self.delete("clonesource")
        if self._clone_source is None:
            return
        x, y = self._clone_source
        cx, cy = self._doc_to_view(x, y)
        # Two-tone crosshair: a dark halo under a bright arm keeps the source
        # marker readable on both light and dark canvas regions.
        arm = 9
        for color, width in (("#000000", 3), ("#ffffff", 1)):
            self.create_line(
                cx - arm, cy, cx + arm, cy,
                fill=color, width=width, tags="clonesource",
            )
            self.create_line(
                cx, cy - arm, cx, cy + arm,
                fill=color, width=width, tags="clonesource",
            )
        self.create_oval(
            cx - 3, cy - 3, cx + 3, cy + 3,
            outline="#000000", fill="#ffffff", width=1, tags="clonesource",
        )

    # ------------------------------------------------------------------ gizmo
    def set_gizmo(self, geometry):
        """Store transform-gizmo geometry (document pixels) and redraw it."""
        self._gizmo = geometry
        self._draw_gizmo()

    def _draw_gizmo(self):
        self.delete("gizmo")
        if self._gizmo is None:
            return
        frame = [self._doc_to_view(x, y) for x, y in self._gizmo["frame"]]
        self.create_polygon(
            *[c for point in frame for c in point],
            fill="", outline="#0078d7", width=1, tags="gizmo",
        )
        (sx, sy), (rx, ry) = self._gizmo["stem"]
        v_sx, v_sy = self._doc_to_view(sx, sy)
        v_rx, v_ry = self._doc_to_view(rx, ry)
        self.create_line(
            v_sx, v_sy, v_rx, v_ry,
            fill="#0078d7", width=1, tags="gizmo",
        )
        handle = 4
        for hx, hy in self._gizmo["handles"]:
            cx, cy = self._doc_to_view(hx, hy)
            self.create_rectangle(
                cx - handle, cy - handle, cx + handle, cy + handle,
                fill="#ffffff", outline="#0078d7", width=1, tags="gizmo",
            )
        hx, hy = self._gizmo["rotate"]
        cx, cy = self._doc_to_view(hx, hy)
        self.create_oval(
            cx - handle, cy - handle, cx + handle, cy + handle,
            fill="#ffffff", outline="#0078d7", width=1, tags="gizmo",
        )
        pivot = self._gizmo.get("pivot")
        if pivot is not None:
            px, py = self._doc_to_view(*pivot)
            arm = handle + 2
            self.create_oval(
                px - handle, py - handle, px + handle, py + handle,
                fill="#ffffff", outline="#e06c00", width=1, tags="gizmo",
            )
            self.create_line(
                px - arm, py, px + arm, py,
                fill="#e06c00", width=1, tags="gizmo",
            )
            self.create_line(
                px, py - arm, px, py + arm,
                fill="#e06c00", width=1, tags="gizmo",
            )

    # ------------------------------------------------------------------ path
    def set_path(self, data):
        """Store path-tool overlay data (see :meth:`path.Path.preview`) and redraw."""
        self._path_data = data
        self._draw_path()

    def _draw_path(self):
        self.delete("path")
        data = self._path_data
        if data is None:
            return
        anchors = data.get("anchors") or []
        handles = data.get("handles") or []
        show_handles = data.get("show_handles", True)
        mode = data.get("mode", "fill")
        fill_color = data.get("fill_color") or "#0078d7"
        stroke_color = data.get("stroke_color") or "#0078d7"
        width = max(1.0, float(data.get("width", 1)) * self.zoom)
        outline = max(0.0, float(data.get("outline", 0)) * self.zoom)
        # Tk spells the square cap "projecting".
        cap = {"butt": "butt", "square": "projecting"}.get(
            data.get("cap", "round"), "round")
        join = data.get("join", "round")
        opacity = max(1, min(100, int(data.get("opacity", 100))))
        stipple = opacity_stipple(opacity)

        def to_view(points):
            coords = []
            for x, y in points:
                vx, vy = self._doc_to_view(x, y)
                coords.extend((vx, vy))
            return coords

        def stroke_options(color, thickness):
            options = {
                "fill": color, "width": thickness,
                "capstyle": cap, "joinstyle": join, "tags": "path",
            }
            if stipple is not None:
                options["stipple"] = stipple
            return options

        # Preview the fill/stroke the current Output settings would produce, so
        # the overlay shows the result rather than only the bare path line.
        if mode in ("fill", "both"):
            shape = data.get("shape") or []
            if len(shape) >= 2:
                coords = to_view(shape)
                polygon = {"fill": fill_color, "outline": "", "tags": "path"}
                if stipple is not None:
                    polygon["stipple"] = stipple
                self.create_polygon(*coords, **polygon)
                if mode == "fill" and outline > 0:
                    self.create_line(
                        *coords, coords[0], coords[1],
                        **stroke_options(fill_color, outline))
        if mode in ("stroke", "both"):
            line = data.get("line") or []
            if len(line) >= 2:
                coords = to_view(line)
                if data.get("closed"):
                    coords = coords + coords[:2]
                self.create_line(*coords, **stroke_options(stroke_color, width))

        # The pending segment from the last anchor to the cursor, as a guide.
        rubber = data.get("rubber")
        if rubber is not None and anchors:
            ax, ay = self._doc_to_view(*anchors[-1])
            rx, ry = self._doc_to_view(*rubber)
            self.create_line(
                ax, ay, rx, ry, fill="#0078d7", width=1, dash=(4, 3),
                tags="path",
            )
        for x, y in anchors:
            vx, vy = self._doc_to_view(x, y)
            self.create_rectangle(
                vx - 3, vy - 3, vx + 3, vy + 3,
                fill="#ffffff", outline="#0078d7", width=1, tags="path",
            )
        # Handles are drawn after the anchors so a short handle (whose grip
        # overlaps its anchor) stays visible on top, matching the path tool's
        # grab priority, which tests handles before anchors.
        for index, handle in enumerate(handles):
            if not show_handles or index >= len(anchors):
                break
            h_in, h_out = handle
            ax, ay = self._doc_to_view(*anchors[index])
            for hx, hy in (h_in, h_out):
                vx, vy = self._doc_to_view(hx, hy)
                if abs(vx - ax) < 0.5 and abs(vy - ay) < 0.5:
                    continue
                self.create_line(
                    ax, ay, vx, vy, fill="#7fb3e0", width=1, tags="path",
                )
                self.create_oval(
                    vx - 3, vy - 3, vx + 3, vy + 3,
                    fill="#ffffff", outline="#0078d7", width=1, tags="path",
                )

    # ---------------------------------------------------------------- vector
    def set_vector_overlay(self, data):
        """Store vector-tool overlay data (document pixels) and redraw it.

        ``data`` may carry an in-progress ``preview`` polyline, the ``stroke``
        outline of the selected line, its ``anchors``, its bezier ``handles``
        (``[anchor_x, anchor_y, handle_x, handle_y]``) and the ``active_anchor``
        index. Passing ``None`` clears the overlay.
        """
        self._vector_overlay = data
        self._draw_vector_overlay()

    def _draw_vector_overlay(self):
        self.delete("vector")
        data = self._vector_overlay
        if data is None:
            return
        accent = theme.color("accent") or "#0078d7"
        for points in data.get("selected") or []:
            if len(points) >= 2:
                coords = []
                for x, y in points:
                    vx, vy = self._doc_to_view(x, y)
                    coords.extend((vx, vy))
                self.create_line(
                    *coords, fill=accent, width=1, tags="vector",
                )
        stroke = data.get("stroke")
        if stroke:
            for points in stroke:
                if len(points) < 2:
                    continue
                coords = []
                for x, y in points:
                    vx, vy = self._doc_to_view(x, y)
                    coords.extend((vx, vy))
                self.create_line(
                    *coords, fill=accent, width=1, dash=(4, 3), tags="vector",
                )
        preview = data.get("preview")
        if preview and len(preview) >= 2:
            coords = []
            for x, y in preview:
                vx, vy = self._doc_to_view(x, y)
                coords.extend((vx, vy))
            self.create_line(
                *coords, fill=accent, width=2,
                capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="vector",
            )
        box = data.get("box")
        if box:
            vx0, vy0 = self._doc_to_view(box[0], box[1])
            vx1, vy1 = self._doc_to_view(box[2], box[3])
            self.create_rectangle(
                vx0, vy0, vx1, vy1, outline=accent, dash=(4, 3),
                width=1, tags="vector",
            )
            top_x = (vx0 + vx1) / 2.0
            self.create_line(
                top_x, vy0 - 20, top_x, vy0, fill=accent, width=1,
                tags="vector",
            )
            self.create_rectangle(
                top_x - 4, vy0 - 24, top_x + 4, vy0 - 16,
                fill="#ffffff", outline=accent, width=1, tags="vector",
            )
            for hx, hy in ((vx0, vy0), (vx1, vy0), (vx0, vy1), (vx1, vy1)):
                self.create_rectangle(
                    hx - 4, hy - 4, hx + 4, hy + 4,
                    fill=accent, outline="#ffffff", width=1, tags="vector",
                )
        marquee = data.get("marquee")
        if marquee:
            mx0, my0 = self._doc_to_view(marquee[0], marquee[1])
            mx1, my1 = self._doc_to_view(marquee[2], marquee[3])
            self.create_rectangle(
                min(mx0, mx1), min(my0, my1), max(mx0, mx1), max(my0, my1),
                outline=accent, dash=(3, 2), width=1, tags="vector",
            )
        for ax, ay, hx, hy in data.get("handles") or []:
            vx, vy = self._doc_to_view(ax, ay)
            hvx, hvy = self._doc_to_view(hx, hy)
            self.create_line(
                vx, vy, hvx, hvy, fill=accent, width=1, dash=(2, 2),
                tags="vector",
            )
            self.create_oval(
                hvx - 3, hvy - 3, hvx + 3, hvy + 3,
                fill="#ffffff", outline=accent, width=1, tags="vector",
            )
        half = max(2, int(data.get("anchor_size", 4)))
        active = data.get("active_anchor", -1)
        for index, (x, y) in enumerate(data.get("anchors") or []):
            vx, vy = self._doc_to_view(x, y)
            if index == active:
                self.create_rectangle(
                    vx - half - 1, vy - half - 1, vx + half + 1, vy + half + 1,
                    fill=accent, outline="#ffffff", width=1, tags="vector",
                )
            else:
                self.create_rectangle(
                    vx - half, vy - half, vx + half, vy + half,
                    fill="#ffffff", outline=accent, width=1, tags="vector",
                )

    # ------------------------------------------------------------- poses
    def set_poses_overlay(self, data):
        """Store poses-tool overlay data (document pixels) and redraw it.

        ``data`` carries the projected ``joints``, the limb ``bones`` and the
        ``root`` anchor of the selected figure, plus the ``active_joint`` being
        dragged. Passing ``None`` clears the overlay.
        """
        self._poses_overlay = data
        self._draw_poses_overlay()

    def _draw_poses_overlay(self):
        self.delete("poses")
        data = self._poses_overlay
        if not data:
            return
        accent = theme.color("accent") or "#0078d7"
        for x1, y1, x2, y2 in data.get("bones") or []:
            vx1, vy1 = self._doc_to_view(x1, y1)
            vx2, vy2 = self._doc_to_view(x2, y2)
            self.create_line(
                vx1, vy1, vx2, vy2, fill=accent, width=1, dash=(3, 2),
                tags="poses",
            )
        root = data.get("root")
        if root is not None:
            vx, vy = self._doc_to_view(root[0], root[1])
            self.create_oval(
                vx - 6, vy - 6, vx + 6, vy + 6, outline=accent, width=2,
                tags="poses",
            )
        outline = data.get("outline")
        if data.get("kind") == "primitive" and outline is not None:
            # The selected primitive's projected bounds, as a dashed box.
            left, top, right, bottom = outline
            vl, vt = self._doc_to_view(left, top)
            vr, vb = self._doc_to_view(right, bottom)
            self.create_rectangle(
                vl, vt, vr, vb, outline="#d8b13a", width=1, dash=(4, 3),
                tags="poses",
            )
        hover = data.get("hover")
        figure = data.get("figure_gizmo")
        if figure:
            # Whole-figure transform gizmo, modal: only the active mode's handle
            # (move arrows, scale square or rotate ring) is drawn, around the
            # pelvis at a fixed screen size. The handle under the cursor gets a
            # white halo and a thicker line.
            mode = data.get("figure_mode", "move")
            ox, oy = figure["origin"]
            vx, vy = self._doc_to_view(ox, oy)
            if mode == "rotate":
                ring = figure.get("ring") or []
                coords = []
                for px, py in ring:
                    pvx, pvy = self._doc_to_view(px, py)
                    coords.extend((pvx, pvy))
                if coords:
                    coords.extend(coords[:2])
                    hot = hover == ("figure", "rotate")
                    if hot:
                        self.create_line(*coords, fill="#ffffff", width=5,
                                         tags="poses")
                    self.create_line(*coords, fill="#d8b13a",
                                     width=4 if hot else 2, tags="poses")
            elif mode == "scale":
                sx, sy = self._doc_to_view(*figure["scale"])
                hot = hover == ("figure", "scale")
                self.create_line(vx, vy, sx, sy, fill="#d8b13a",
                                 width=3 if hot else 1, dash=() if hot else (3, 2),
                                 tags="poses")
                self.create_rectangle(sx - 5, sy - 5, sx + 5, sy + 5,
                                      outline="#ffffff", fill="#d8b13a",
                                      width=3 if hot else 1, tags="poses")
            else:
                for name, colour in (("move_x", "#e05a4f"), ("move_y", "#54b054")):
                    (ax, ay), (bx, by) = figure[name]
                    vax, vay = self._doc_to_view(ax, ay)
                    vbx, vby = self._doc_to_view(bx, by)
                    hot = hover == ("figure", name)
                    if hot:
                        self.create_line(vax, vay, vbx, vby, fill="#ffffff",
                                         width=6, arrow=tk.LAST, tags="poses")
                    self.create_line(vax, vay, vbx, vby, fill=colour,
                                     width=4 if hot else 2, arrow=tk.LAST,
                                     tags="poses")
            self.create_oval(vx - 3, vy - 3, vx + 3, vy + 3, fill="#d8b13a",
                             outline="", tags="poses")
        primitive = data.get("primitive_gizmo")
        if primitive:
            # The selected primitive's modal axis gizmo: arrows translate, axis
            # handles scale, and world-axis rings rotate. Only the active mode's
            # control is drawn, matching the whole-figure gizmo's modal behaviour.
            mode = data.get("figure_mode", "move")
            vx, vy = self._doc_to_view(*primitive["origin"])
            colours = (("x", "#e05a4f"), ("y", "#54b054"), ("z", "#4f86d6"))
            if mode == "rotate":
                for name, colour in colours:
                    points = (primitive.get("rings") or {}).get(name)
                    if not points:
                        continue
                    coords = []
                    for px, py in points:
                        pvx, pvy = self._doc_to_view(px, py)
                        coords.extend((pvx, pvy))
                    coords.extend(coords[:2])
                    hot = hover == ("axis", name)
                    if hot:
                        self.create_line(*coords, fill="#ffffff", width=5,
                                         tags="poses")
                    self.create_line(*coords, fill=colour, width=4 if hot else 2,
                                     tags="poses")
            else:
                for name, colour in colours:
                    handle = (primitive.get("handles") or {}).get(name)
                    if handle is None:
                        continue
                    hx, hy = self._doc_to_view(handle[0], handle[1])
                    hot = hover == ("axis", name)
                    if hot:
                        self.create_line(vx, vy, hx, hy, fill="#ffffff", width=6,
                                         arrow=tk.LAST, tags="poses")
                    self.create_line(vx, vy, hx, hy, fill=colour,
                                     width=4 if hot else 2, arrow=tk.LAST,
                                     tags="poses")
                    if mode == "scale":
                        self.create_rectangle(
                            hx - 5, hy - 5, hx + 5, hy + 5, fill=colour,
                            outline="#ffffff", width=2, tags="poses",
                        )
            self.create_oval(vx - 3, vy - 3, vx + 3, vy + 3, fill="#d8b13a",
                             outline="", tags="poses")
        gizmo = data.get("gizmo")
        if gizmo:
            # Rotation rings about the world axes, drawn as projected 3D circles.
            for name, colour in (("x", "#e05a4f"), ("y", "#54b054"), ("z", "#4f86d6")):
                points = (gizmo.get("rings") or {}).get(name)
                if not points:
                    continue
                coords = []
                for px, py in points:
                    vx, vy = self._doc_to_view(px, py)
                    coords.extend((vx, vy))
                coords.extend(coords[:2])
                hot = hover == ("ring", name)
                if hot:
                    self.create_line(*coords, fill="#ffffff", width=5, tags="poses")
                self.create_line(*coords, fill=colour, width=4 if hot else 2,
                                 tags="poses")
        active = data.get("active_joint")
        selected = data.get("selected_bone")
        ik_joints = set(data.get("ik_joints") or ())
        for name, (x, y) in (data.get("joints") or {}).items():
            vx, vy = self._doc_to_view(x, y)
            if name == active or name == selected:
                # The dragged or selected bone stays a solid square so it reads
                # as the joint currently under control.
                self.create_rectangle(
                    vx - 5, vy - 5, vx + 5, vy + 5,
                    fill=accent, outline="#ffffff", width=1, tags="poses",
                )
            elif name in ik_joints:
                # Filled squares mark the handles that solve a whole limb with IK.
                self.create_rectangle(
                    vx - 4, vy - 4, vx + 4, vy + 4,
                    fill="#ffffff", outline=accent, width=2, tags="poses",
                )
            else:
                self.create_oval(
                    vx - 4, vy - 4, vx + 4, vy + 4,
                    fill="#ffffff", outline=accent, width=1, tags="poses",
                )

    def _pose_grid_image(self, segments, window, size):
        """Rasterize the ground grid as a viewport-sized RGBA underlay.

        The grid is composited under the layers (see :func:`compose_region`), so
        the figure and its strokes cover the lines instead of the other way
        round.
        """
        image = Image.new("RGBA", (int(size[0]), int(size[1])), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        wx, wy = window
        zoom = self.zoom
        limit = 1.0e5
        for x0, y0, x1, y1 in segments:
            draw.line(
                (max(-limit, min(limit, (x0 - wx) * zoom)),
                 max(-limit, min(limit, (y0 - wy) * zoom)),
                 max(-limit, min(limit, (x1 - wx) * zoom)),
                 max(-limit, min(limit, (y1 - wy) * zoom))),
                fill=(143, 143, 143, 255), width=1,
            )
        return image

    # -------------------------------------------------------------- mesh warp
    def set_warp_grid(self, geometry):
        """Store the mesh-warp grid (document pixels) and redraw it."""
        self._warp_grid = geometry
        self._draw_warp_grid()

    def _draw_warp_grid(self):
        self.delete("warp")
        geometry = self._warp_grid
        if geometry is None:
            return
        accent = theme.color("accent") or "#0078d7"
        line_rows = geometry.get("horizontal", [])
        for polyline in line_rows + geometry.get("vertical", []):
            if len(polyline) < 2:
                continue
            coords = []
            for x, y in polyline:
                vx, vy = self._doc_to_view(x, y)
                coords.extend((vx, vy))
            self.create_line(*coords, fill=accent, width=1, tags="warp")
        for polyline in line_rows:
            for x, y in polyline:
                vx, vy = self._doc_to_view(x, y)
                self.create_rectangle(
                    vx - 3, vy - 3, vx + 3, vy + 3,
                    fill="#ffffff", outline=accent, width=1, tags="warp",
                )

    # ------------------------------------------------------------- puppet warp
    def set_puppet_overlay(self, data):
        """Store the puppet-warp overlay (pins/mesh/band) and redraw it."""
        self._puppet_overlay = data
        self._draw_puppet_overlay()

    def _draw_puppet_overlay(self):
        self.delete("puppet")
        data = self._puppet_overlay
        if not data:
            return
        accent = theme.color("accent") or "#0078d7"
        mesh = data.get("mesh")
        if data.get("show_mesh") and mesh:
            for polyline in mesh:
                self._puppet_polyline(polyline, accent)
        handle_radius = data.get("handle_radius", 20.0)
        for pin in data.get("pins", []):
            rx, ry = pin["rest"]
            px, py = pin["pos"]
            angle = pin["angle"]
            vrx, vry = self._doc_to_view(rx, ry)
            vpx, vpy = self._doc_to_view(px, py)
            if abs(vpx - vrx) > 0.5 or abs(vpy - vry) > 0.5:
                # The dashed stem and hollow ring mark where the pin started.
                self.create_line(
                    vrx, vry, vpx, vpy, fill=accent, width=1, dash=(3, 2),
                    tags="puppet",
                )
                self.create_oval(
                    vrx - 3, vry - 3, vrx + 3, vry + 3,
                    outline=accent, width=1, tags="puppet",
                )
            selected = pin.get("selected")
            locked = not pin.get("correct", True)
            if selected:
                # The outer ring is the rotation handle: drag anywhere on it (or
                # its knob) to spin the pin, the way a puppet-warp handle works.
                ring = handle_radius * max(self.zoom, 1e-6)
                self.create_oval(
                    vpx - ring, vpy - ring, vpx + ring, vpy + ring,
                    outline=accent, width=1, dash=(2, 3), tags="puppet",
                )
                hx = px + math.sin(angle) * handle_radius
                hy = py - math.cos(angle) * handle_radius
                vhx, vhy = self._doc_to_view(hx, hy)
                self.create_line(
                    vpx, vpy, vhx, vhy, fill=accent, width=1, dash=(2, 2),
                    tags="puppet",
                )
                self.create_oval(
                    vhx - 3, vhy - 3, vhx + 3, vhy + 3,
                    fill="#ffffff", outline=accent, width=1, tags="puppet",
                )
            radius = 5 if selected else 4
            fill = "#ff4040" if selected else "#ffffff"
            if locked:
                # A locked pin (excluded from rotation correction) is a square.
                self.create_rectangle(
                    vpx - radius, vpy - radius, vpx + radius, vpy + radius,
                    fill=fill, outline=accent, width=2, tags="puppet",
                )
            else:
                self.create_oval(
                    vpx - radius, vpy - radius, vpx + radius, vpy + radius,
                    fill=fill, outline="#ffffff" if selected else accent,
                    width=2 if selected else 1, tags="puppet",
                )
        band = data.get("band")
        if band:
            vx0, vy0 = self._doc_to_view(min(band[0], band[2]),
                                         min(band[1], band[3]))
            vx1, vy1 = self._doc_to_view(max(band[0], band[2]),
                                         max(band[1], band[3]))
            self.create_rectangle(
                vx0, vy0, vx1, vy1, outline=accent, dash=(4, 3), width=1,
                tags="puppet",
            )

    def _puppet_polyline(self, polyline, color):
        if len(polyline) < 2:
            return
        coords = []
        for x, y in polyline:
            vx, vy = self._doc_to_view(x, y)
            coords.extend((vx, vy))
        self.create_line(*coords, fill=color, width=1, tags="puppet")

    # -------------------------------------------------------------- symmetry
    def set_symmetry(self, guide):
        """Store the symmetry guide (``mode``/``count``/``mirror``) for drawing."""
        self._symmetry = guide

    def _draw_symmetry(self):
        self.delete("symmetry")
        if not self._symmetry or self._layout is not None:
            return
        lines = symmetry.axes(self._symmetry, self.doc.size)
        if not lines:
            return
        accent = theme.color("accent") or "#0078d7"
        for (x0, y0), (x1, y1) in lines:
            vx0, vy0 = self._doc_to_view(x0, y0)
            vx1, vy1 = self._doc_to_view(x1, y1)
            self.create_line(
                vx0, vy0, vx1, vy1, fill=accent, width=1, dash=(4, 4),
                tags="symmetry",
            )

    # ------------------------------------------------------------------ text
    def set_text_session(self, geometry):
        """Store in-progress text-edit geometry (document pixels) and redraw it."""
        self._text_session = geometry
        self._draw_text_session()

    def set_text_caret_on(self, visible):
        """Toggle just the caret (used by the blinking loop)."""
        if self._text_session is None:
            return
        self._text_session["caret_on"] = bool(visible)
        self._draw_text_session()

    def _draw_text_session(self):
        self.delete("textsession")
        geometry = self._text_session
        if geometry is None:
            return
        accent = theme.color("accent") or "#0078d7"
        for quad in geometry.get("selection", []):
            coords = []
            for x, y in quad:
                vx, vy = self._doc_to_view(x, y)
                coords.extend((vx, vy))
            self.create_polygon(
                *coords, fill=accent, outline="", stipple="gray50",
                tags="textsession",
            )
        frame = [self._doc_to_view(x, y) for x, y in geometry["frame"]]
        self.create_polygon(
            *[coord for point in frame for coord in point],
            fill="", outline="#0078d7", dash=(4, 3), width=1, tags="textsession",
        )
        (sx, sy), (rx, ry) = geometry["stem"]
        v_sx, v_sy = self._doc_to_view(sx, sy)
        v_rx, v_ry = self._doc_to_view(rx, ry)
        self.create_line(
            v_sx, v_sy, v_rx, v_ry, fill="#0078d7", width=1, tags="textsession",
        )
        handle = 4
        for hx, hy in geometry["handles"]:
            cx, cy = self._doc_to_view(hx, hy)
            self.create_rectangle(
                cx - handle, cy - handle, cx + handle, cy + handle,
                fill="#ffffff", outline="#0078d7", width=1, tags="textsession",
            )
        hx, hy = geometry["rotate"]
        cx, cy = self._doc_to_view(hx, hy)
        self.create_oval(
            cx - handle, cy - handle, cx + handle, cy + handle,
            fill="#ffffff", outline="#0078d7", width=1, tags="textsession",
        )
        pivot = geometry.get("pivot")
        if pivot is not None:
            px, py = self._doc_to_view(*pivot)
            arm = handle + 2
            self.create_oval(
                px - handle, py - handle, px + handle, py + handle,
                fill="#ffffff", outline="#e06c00", width=1, tags="textsession",
            )
            self.create_line(
                px - arm, py, px + arm, py,
                fill="#e06c00", width=1, tags="textsession",
            )
            self.create_line(
                px, py - arm, px, py + arm,
                fill="#e06c00", width=1, tags="textsession",
            )
        caret = geometry.get("caret")
        if caret is None or not geometry.get("caret_on", True):
            return
        (x1, y1), (x2, y2) = caret
        vx1, vy1 = self._doc_to_view(x1, y1)
        vx2, vy2 = self._doc_to_view(x2, y2)
        self.create_line(
            vx1, vy1, vx2, vy2, fill=accent, width=2, tags="textsession",
        )

    # ------------------------------------------------------------------ select
    def set_selection(self, selection):
        """Store a :class:`selection.Selection` (or ``None``) and redraw it."""
        self.selection = selection
        self._draw_selection()

    def _draw_selection(self):
        self.delete("selection")
        self._selection_image = None
        if self.selection is None:
            return
        if ((self.selection.masked
             or self.selection.outline_segments() is not None)
                and not self.transformed):
            self._draw_selection_image()
            return
        points = self.selection.outline()
        if len(points) < 2:
            return
        coords = []
        for x, y in points:
            vx, vy = self._doc_to_view(x, y)
            coords.extend((vx, vy))
        self.create_polygon(
            *coords, fill="", outline="#ffffff", width=1, tags="selection",
        )
        self.create_polygon(
            *coords, fill="", outline="#000000", dash=(5, 3), width=1,
            tags="selection",
        )

    def set_selection_label(self, box):
        """Show a live ``x, y, w, h`` readout beside a rectangle marquee.

        ``box`` is a document-pixel ``(left, top, right, bottom)`` rectangle,
        or ``None`` to hide the readout.
        """
        self._selection_label = box
        self._draw_selection_label()

    def _draw_selection_label(self):
        self.delete("sellabel")
        box = self._selection_label
        if box is None:
            return
        left, top, right, bottom = box
        if right <= left and bottom <= top:
            return
        text = f"x {left}  y {top}  w {right - left}  h {bottom - top}"
        vx, vy = self._doc_to_view(left, top)
        text_id = self.create_text(
            vx + 6, vy + 6, text=text, anchor="nw", fill="#ffffff",
            font="TkDefaultFont", tags="sellabel",
        )
        bounds = self.bbox(text_id)
        if bounds is not None:
            x0, y0, x1, y1 = bounds
            # ``bbox`` is a screen rectangle, so this chip must bypass the view
            # transform wrappers (it already sits where it belongs).
            tk.Canvas.create_rectangle(
                self, x0 - 3, y0 - 2, x1 + 3, y1 + 2, fill="#000000",
                outline="", tags="sellabel",
            )
            self.tag_raise(text_id)

    def _draw_selection_image(self):
        """Draw a mask selection's boundary image, scaled to the viewport."""
        if self.transformed:
            # An axis-aligned boundary bitmap cannot be placed on a rotated
            # view; the caller falls back to the polygon outline instead.
            return
        if self._render_window is not None:
            top_left = self._render_window
            view_size = self._render_size
        else:
            view_w = self.winfo_width()
            view_h = self.winfo_height()
            if view_w <= 1 or view_h <= 1:
                view_w, view_h = self._content_w, self._content_h
            top_left = (self.canvasx(0) / self.zoom, self.canvasy(0) / self.zoom)
            view_size = (view_w, view_h)
        boundary = self.selection.boundary_image()
        shift = self._active_shift() if self._layout is not None else (0, 0)
        crop, item = visible_crop(
            boundary, (0, 0), self.zoom,
            (top_left[0] + shift[0], top_left[1] + shift[1]), view_size,
        )
        if crop is None:
            return
        if shift != (0, 0):
            # ``visible_crop`` placed the crop at its document position; shift it
            # back into the view so it lands in the active layer's cell.
            item = (item[0] - shift[0] * self.zoom, item[1] - shift[1] * self.zoom)
        self._selection_image = ImageTk.PhotoImage(crop)
        self.create_image(
            item[0], item[1], anchor=tk.NW, image=self._selection_image,
            tags="selection",
        )

    # ------------------------------------------------------------------ layout
    def set_layout_boxes(self, boxes):
        """Store the MOSS layout items to overlay and redraw them.

        ``boxes`` is a list of dicts with a document-pixel ``bbox`` plus the
        ``color``, ``title``, ``text`` and ``caption`` to show beside it.
        """
        self.layout_boxes = list(boxes or [])
        self._draw_layout_boxes()

    def _draw_layout_boxes(self):
        self.delete("layoutbox")
        if self._layout is not None:
            return
        for item in self.layout_boxes or ():
            box = item.get("bbox")
            if not box:
                continue
            left, top, right, bottom = box
            x0, y0 = self._doc_to_view(left, top)
            x1, y1 = self._doc_to_view(right, bottom)
            if x1 <= x0 or y1 <= y0:
                continue
            color = item.get("color") or "#3b82f6"
            self.create_rectangle(
                x0, y0, x1, y1, outline=color, width=2, tags="layoutbox",
            )
            pad = 4
            title = (item.get("title") or item.get("name") or "").strip()
            label_bottom = y0
            if title:
                text_id = self.create_text(
                    x0 + pad, y0 + pad, text=title, anchor="nw", fill="#ffffff",
                    font="TkDefaultFont", tags="layoutbox",
                )
                bounds = self.bbox(text_id)
                if bounds is not None:
                    # A screen-space chip around the text; bypass the transform.
                    chip = tk.Canvas.create_rectangle(
                        self, bounds[0] - pad, bounds[1] - 2, bounds[2] + pad,
                        bounds[3] + 2, fill=color, outline="", tags="layoutbox",
                    )
                    self.tag_raise(text_id, chip)
                    label_bottom = bounds[3]
            lines = []
            if item.get("text"):
                lines.append(item["text"])
            if item.get("caption"):
                lines.append(item["caption"])
            if lines and (x1 - x0) > 70 and (y1 - y0) > 40:
                self.create_text(
                    x0 + pad, label_bottom + 4, text="\n".join(lines), anchor="nw",
                    fill=color, font="TkDefaultFont",
                    width=max(40, int(x1 - x0 - pad * 2)), tags="layoutbox",
                )

    # --------------------------------------------------- tool previews
    def set_shape_preview(self, data):
        """Show the shape being dragged (see the Shape tool)."""
        self._shape_preview = data
        self._draw_shape_preview()

    def _draw_shape_preview(self):
        self.delete("shapepreview")
        data = self._shape_preview
        if data is None:
            return
        box = data["box"]
        x0, y0 = self._doc_to_view(box[0], box[1])
        x1, y1 = self._doc_to_view(box[2], box[3])
        color = data.get("color") or "#0078d7"
        left, top = min(x0, x1), min(y0, y1)
        right, bottom = max(x0, x1), max(y0, y1)
        if data["kind"] == "line":
            self.create_line(
                x0, y0, x1, y1, fill=color, width=1, dash=(4, 3),
                tags="shapepreview",
            )
        elif data["kind"] == "ellipse":
            self.create_oval(
                left, top, right, bottom, outline=color, width=1, dash=(4, 3),
                tags="shapepreview",
            )
        elif data["kind"] in shape.POLYGONS:
            points = []
            for px, py in shape.polygon_points(
                data["kind"], (left, top, right, bottom)
            ):
                points.extend((px, py))
            self.create_polygon(
                points, fill="", outline=color, width=1, dash=(4, 3),
                tags="shapepreview",
            )
        else:
            radius = float(data.get("radius") or 0) * self.zoom
            if radius > 0:
                self._rounded_rect(
                    left, top, right, bottom, radius, fill="", outline=color,
                    width=1, dash=(4, 3), tags="shapepreview",
                )
            else:
                self.create_rectangle(
                    left, top, right, bottom, outline=color, width=1, dash=(4, 3),
                    tags="shapepreview",
                )

    def _rounded_rect(self, x0, y0, x1, y1, radius, **kwargs):
        radius = max(0, min(radius, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
        points = (
            x0 + radius, y0, x1 - radius, y0, x1, y0,
            x1, y0 + radius, x1, y1 - radius, x1, y1,
            x1 - radius, y1, x0 + radius, y1, x0, y1,
            x0, y1 - radius, x0, y0 + radius, x0, y0,
        )
        self.create_polygon(points, smooth=True, **kwargs)

    def set_gradient_preview(self, data):
        """Show the gradient drag axis (see the Gradient tool)."""
        self._gradient_preview = data
        self._draw_gradient_preview()

    def _draw_gradient_preview(self):
        self.delete("gradientpreview")
        data = self._gradient_preview
        if data is None:
            return
        color = data.get("color") or "#0078d7"
        x0, y0 = self._doc_to_view(*data["start"])
        x1, y1 = self._doc_to_view(*data["end"])
        self.create_line(x0, y0, x1, y1, fill=color, width=1, tags="gradientpreview")
        for px, py in ((x0, y0), (x1, y1)):
            self.create_oval(
                px - 4, py - 4, px + 4, py + 4, outline=color, width=1,
                tags="gradientpreview",
            )

    def set_crop_preview(self, box):
        """Show the crop rectangle with the area outside it dimmed."""
        self._crop_preview = box
        # The dimming is part of the base image, so refresh it in place instead
        # of redrawing overlay items; only the fast border and handles are
        # drawn as canvas items (see :meth:`_draw_crop_preview`).
        self._refresh_crop_image()
        self._draw_crop_preview()

    def _dim_crop(self, board):
        """Return ``board`` with a cross-hatch over everything outside the crop.

        The crop shading used to be four stippled canvas rectangles. A stippled
        fill repaints slowly and leaves its "off" pixels showing whatever was
        under it, so moving or resizing the window (which forces repaints of the
        viewport) made the crop area flicker through the dither colors.
        Compositing the cross-hatch into the single base image removes that
        separate item entirely: one image scrolls and repaints like the rest of
        the canvas, with no flicker.
        """
        box = self._crop_preview
        if board is None or box is None or self._layout is not None:
            return board
        window = self._render_window
        if window is None:
            return board
        zoom = self.zoom
        wx, wy = window
        left, top, right, bottom = box
        left, right = min(left, right), max(left, right)
        top, bottom = min(top, bottom), max(top, bottom)
        # The hatch is confined to the canvas itself (as the rectangle shades
        # were), so the off-canvas margin keeps the plain background color.
        bx0 = max(0.0, (0 - wx) * zoom)
        by0 = max(0.0, (0 - wy) * zoom)
        bx1 = min(float(board.width), (self.doc.width - wx) * zoom)
        by1 = min(float(board.height), (self.doc.height - wy) * zoom)
        if bx1 <= bx0 or by1 <= by0:
            return board
        cl = min(max((left - wx) * zoom, bx0), bx1)
        ct = min(max((top - wy) * zoom, by0), by1)
        cr = min(max((right - wx) * zoom, bx0), bx1)
        cb = min(max((bottom - wy) * zoom, by0), by1)
        # The region that may be shaded: the canvas rectangle minus the crop box.
        region = Image.new("L", board.size, 0)
        rd = ImageDraw.Draw(region)
        rx0 = max(0, int(bx0))
        ry0 = max(0, int(by0))
        rx1 = min(board.width - 1, int(math.ceil(bx1)) - 1)
        ry1 = min(board.height - 1, int(math.ceil(by1)) - 1)
        if rx1 >= rx0 and ry1 >= ry0:
            rd.rectangle((rx0, ry0, rx1, ry1), fill=255)
        if cr > cl and cb > ct:
            # Only punch the crop box out when it actually overlaps the board; a
            # rotated view can push it entirely off the visible region.
            rd.rectangle(
                (int(cl), int(ct), int(math.ceil(cr)) - 1, int(math.ceil(cb)) - 1),
                fill=0,
            )
        # Diagonal lines both ways, in device pixels so the hatch keeps a
        # constant on-screen density at every zoom level.
        hatch = Image.new("L", board.size, 0)
        hd = ImageDraw.Draw(hatch)
        spacing = CROP_HATCH_SPACING
        alpha = CROP_HATCH_ALPHA
        width, height = board.size
        for x in range(-height, width, spacing):
            hd.line((x, 0, x + height, height), fill=alpha, width=1)
        for x in range(0, width + height, spacing):
            hd.line((x, 0, x - height, height), fill=alpha, width=1)
        hatch = Image.composite(hatch, Image.new("L", board.size, 0), region)
        dimmed = board.convert("RGB").copy()
        dimmed.paste(Image.new("RGB", board.size, (0, 0, 0)), (0, 0), hatch)
        return dimmed

    def _refresh_crop_image(self):
        """Re-blend the base board with the current crop dim, if one is shown."""
        board = self._base_board
        if board is None:
            return
        image = self._dim_crop(board)
        if self.transformed and self._warp is not None:
            # The stored board is the canonical (unwarped) one, so re-apply the
            # view warp after re-dimming.
            _board, coeffs, out_size = self._warp
            image = image.convert("RGB").transform(
                out_size, Image.AFFINE, coeffs, resample=self._warp_resample,
            )
        self._photo = ImageTk.PhotoImage(image)
        try:
            self.itemconfigure("base", image=self._photo)
        except tk.TclError:
            pass

    def _draw_crop_preview(self):
        self.delete("croppreview")
        box = self._crop_preview
        if box is None or self._layout is not None:
            return
        left, top = self._doc_to_view(box[0], box[1])
        right, bottom = self._doc_to_view(box[2], box[3])
        left, right = min(left, right), max(left, right)
        top, bottom = min(top, bottom), max(top, bottom)
        self.create_rectangle(
            left, top, right, bottom, outline="#ffffff", width=1,
            tags="croppreview",
        )
        self.create_rectangle(
            left, top, right, bottom, outline="#000000", dash=(5, 3), width=1,
            tags="croppreview",
        )
        handles = (
            (left, top), (right, top), (right, bottom), (left, bottom),
            ((left + right) / 2.0, top), (right, (top + bottom) / 2.0),
            ((left + right) / 2.0, bottom), (left, (top + bottom) / 2.0),
        )
        for hx, hy in handles:
            self.create_rectangle(
                hx - 3, hy - 3, hx + 3, hy + 3, fill="#ffffff", outline="#000000",
                tags="croppreview",
            )

    def clear_tool_overlays(self):
        """Drop controller-owned tool overlays from this view.

        Overlay snapshots live on the view, but the drawing controller is
        global and its session moves to whichever canvas is active. A hidden
        tab must not keep drawing a session that no longer belongs to it, so
        the view being left is cleared when switching canvases (and every view
        is cleared when an undo re-points them at fresh documents). Per-canvas
        state - the selection, layout boxes and generation preview - is
        intentionally kept.
        """
        self.set_cursor_overlay(None)
        self.set_selection_label(None)
        self.set_gizmo(None)
        self.set_path(None)
        self.set_vector_overlay(None)
        self.set_poses_overlay(None)
        self.set_warp_grid(None)
        self.set_puppet_overlay(None)
        self.set_text_session(None)
        self.set_shape_preview(None)
        self.set_gradient_preview(None)
        self.set_crop_preview(None)


def _shifted_index(index, from_index, to_index):
    """Where ``index`` lands once the item at ``from_index`` moves to ``to_index``."""
    if index is None:
        return None
    if index == from_index:
        return to_index
    if from_index < index <= to_index:
        return index - 1
    if to_index <= index < from_index:
        return index + 1
    return index


class CanvasArea(tk.Frame):
    """The tab strip and the rendered view of each document.

    ``on_activate(index)`` fires when the user switches tabs, ``on_rename(index)``
    on a tab double-click, ``on_context(index, x_root, y_root)`` on a tab
    right-click, ``on_view_created(view)`` whenever a new view is
    built so the drawing controller can attach its bindings,
    ``on_add_canvas()`` when the trailing "+" tab is chosen, and
    ``on_guides_changed()`` just before a ruler guide is added, moved or removed
    so the app can snapshot the document for undo.

    Only the active page is packed into ``stack``, so switching canvases is a
    pack/forget rather than a full notebook re-layout.
    """

    def __init__(self, parent, on_activate, on_rename, on_view_created,
                 on_reorder=None, on_gallery_close=None, on_gallery_zoom=None,
                 on_add_canvas=None, on_guides_changed=None, on_context=None):
        super().__init__(parent)
        self._on_activate = on_activate
        self._on_rename = on_rename
        self._on_view_created = on_view_created
        self._on_reorder = on_reorder
        self._on_gallery_close = on_gallery_close
        self._on_gallery_zoom = on_gallery_zoom
        self._on_add_canvas = on_add_canvas
        self._on_guides_changed = on_guides_changed
        self._on_context = on_context
        self._ready = False
        self._pages = []
        self._views = []
        self._packed = None
        self._selected = 0
        self.gallery = None
        self._gallery_shown = False
        # When set to ``"left"``/``"right"`` the gallery is docked beside the
        # editor instead of replacing it (see :meth:`show_gallery_side`).
        self._gallery_dock = None
        self._gallery_sidebar = None
        self._gallery_side = None
        # Floating action bar overlaid on the active page while a generation
        # preview is shown (see :meth:`show_preview_bar`).
        self._preview_bar = None
        self.bg_color = DEFAULT_OFFCANVAS_BG
        self.rulers_visible = True
        self.guides_visible = True
        # When set, every tab draws and edits this one shared guide list instead
        # of its own document's guides (see :meth:`set_shared_guides`).
        self._shared_guides = None
        # Grid overlay and snapping applied to every canvas tab.
        self.grid_visible = False
        self.grid_size = DEFAULT_GRID_SIZE
        self.grid_snap = False

        # The strip and the pages share an ``editor`` frame so the gallery can
        # replace both at once, and the strip stays pinned above the canvas.
        self.editor = tk.Frame(self)
        self.editor.pack(fill=tk.BOTH, expand=True)
        self.tabs = TabStrip(
            self.editor,
            on_select=self._tab_selected,
            on_rename=self._tab_renamed,
            on_add=self._trigger_add,
            on_move=self.reorder,
            on_move_commit=self._tab_move_committed,
            on_context=self._tab_context,
        )
        self.tabs.pack(side=tk.TOP, fill=tk.X)
        self.stack = tk.Frame(self.editor)
        self.stack.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._ready = True

    def rebuild(self, documents, preserve_view=False):
        """Recreate every tab from scratch (used after load/new document).

        With ``preserve_view`` set, each recreated tab inherits the zoom and
        camera of the tab that was at the same index, so an undo/redo does not
        reset the user's view of the document.
        """
        states = []
        if preserve_view:
            states = [view.view_state() for view in self._views]
        self._ready = False
        self.hide_preview_bar()
        for page in list(self._pages):
            page.pack_forget()
            page.destroy()
        self._pages = []
        self._views = []
        self._packed = None
        for doc in documents:
            self._add(doc)
        if preserve_view:
            for view, state in zip(self._views, states):
                view.set_view_state(state)
        self._ready = True
        self.select(0)

    def retarget(self, documents):
        """Point the existing tabs at ``documents`` in place.

        Used by undo/redo when the number of canvases is unchanged: keeping the
        widgets and their zoom/scroll avoids the blank frame a full rebuild
        shows. Falls back to :meth:`rebuild` when the count differs (e.g. undoing
        an add/delete canvas).
        """
        if len(documents) != len(self._views):
            self.rebuild(documents, preserve_view=True)
            return
        for view, doc in zip(self._views, documents):
            view.doc = doc
            doc.widget = view
            # A selection belongs to the document it was drawn on; drop it so a
            # restored document never inherits the previous one's mask.
            view.set_selection(None)
            # Tool overlays describe a controller session for the document it
            # was started on; the restored document must not redraw them.
            view.clear_tool_overlays()
        self.refresh_titles(documents)

    def _show_page(self, index):
        """Pack the page at ``index`` and hide the one shown before it."""
        if not (0 <= index < len(self._pages)):
            return
        if self._packed == index:
            return
        if self._packed is not None and 0 <= self._packed < len(self._pages):
            self._pages[self._packed].pack_forget()
            # Tool sessions are controller-global; a hidden page must not keep
            # drawing the overlay of a session that now belongs to the page we
            # are switching to (see :meth:`CanvasView.clear_tool_overlays`).
            if 0 <= self._packed < len(self._views):
                self._views[self._packed].clear_tool_overlays()
        self._packed = index
        self._pages[index].pack(fill=tk.BOTH, expand=True)

    def add(self, doc):
        """Append a single tab for ``doc`` and select it."""
        self._ready = False
        self._add(doc)
        self._ready = True
        self.select(len(self._views) - 1)

    def remove(self, index):
        """Drop the page at ``index``; the caller re-selects afterwards."""
        self._ready = False
        page = self._pages.pop(index)
        self._views.pop(index)
        if self._packed == index:
            self._packed = None
        elif self._packed is not None and self._packed > index:
            self._packed -= 1
        page.pack_forget()
        page.destroy()
        self._ready = True

    def refresh_titles(self, documents):
        """Re-label every tab from ``documents``, keeping the selection."""
        self.tabs.set_titles([doc.name for doc in documents], self._selected)

    def select(self, index):
        """Show the page and tab for ``index`` without reporting activation."""
        self._selected = index
        self._show_page(index)
        self.refresh_titles([view.doc for view in self._views])

    def reorder(self, from_index, to_index):
        """Move the tab at ``from_index`` to ``to_index``.

        Keeps ``_pages`` and ``_views`` in sync with the strip, and follows the
        shown page and current selection as they shift.
        """
        if from_index == to_index:
            return
        if not (0 <= from_index < len(self._pages)):
            return
        if not (0 <= to_index < len(self._pages)):
            return
        page = self._pages.pop(from_index)
        view = self._views.pop(from_index)
        self._pages.insert(to_index, page)
        self._views.insert(to_index, view)
        self._packed = _shifted_index(self._packed, from_index, to_index)
        self._selected = _shifted_index(self._selected, from_index, to_index)
        self.refresh_titles([item.doc for item in self._views])

    def active_index(self):
        if not self._views:
            return None
        return max(0, min(self._selected, len(self._views) - 1))

    def current_view(self):
        index = self.active_index()
        if index is None or index >= len(self._views):
            return None
        return self._views[index]

    def set_bg_color(self, bg):
        """Apply an off-canvas background color to every canvas tab."""
        self.bg_color = tuple(bg)
        color = rgb_to_hex(bg)
        for page in self._pages:
            try:
                page.config(bg=color)
            except tk.TclError:
                pass
        for view in self._views:
            view.set_bg_color(bg)

    # ------------------------------------------------------------- preview bar
    def show_preview_bar(self, buttons):
        """Overlay a floating action bar on the active page.

        ``buttons`` is a sequence of ``(label, callback)`` pairs, optionally
        ``(label, callback, enabled)``. Used by the generation preview to offer
        rejecting the run, or landing it in a layer or the canvas assets.
        """
        self.hide_preview_bar()
        index = self.active_index()
        if index is None or not (0 <= index < len(self._pages)):
            return
        bar = tk.Frame(
            self._pages[index], bd=0, highlightthickness=1,
            highlightbackground=theme.color("border"),
        )
        theme.register(bar, bg="panel_bg", highlightbackground="border")
        for spec in buttons:
            label, callback = spec[0], spec[1]
            enabled = spec[2] if len(spec) > 2 else True
            button = tk.Button(
                bar, text=label, command=callback,
                state=tk.NORMAL if enabled else tk.DISABLED,
            )
            button.pack(side=tk.LEFT, padx=(0, 6), pady=6)
        # The bar is built after the last theme pass, so theme it here: the
        # class defaults give the buttons their dark/light colors.
        theme.restyle(bar)
        # ``place`` keeps the bar floating over the canvas without taking space
        # from it, so the preview is shown at its full size underneath.
        bar.place(relx=0.5, rely=1.0, anchor="s", y=-16)
        bar.lift()
        self._preview_bar = bar

    def hide_preview_bar(self):
        """Remove the preview action bar, if one is shown."""
        bar = self._preview_bar
        self._preview_bar = None
        if bar is None:
            return
        try:
            bar.destroy()
        except tk.TclError:
            pass

    # ------------------------------------------------------------- gallery
    def _ensure_gallery(self):
        """Build the full-screen thumbnail board the first time it is shown."""
        from .gallery import GalleryView
        if self.gallery is None:
            self.gallery = GalleryView(
                self,
                on_activate=self._gallery_open,
                on_close=self._gallery_close,
                on_zoom=self._on_gallery_zoom,
                on_context=self._tab_context,
            )
        return self.gallery

    def _ensure_gallery_sidebar(self):
        """Build the docked thumbnail strip the first time it is shown."""
        from .gallery import GallerySidebar
        if self._gallery_sidebar is None:
            self._gallery_sidebar = GallerySidebar(
                self,
                on_activate=self._gallery_sidebar_activate,
                on_context=self._tab_context,
            )
        return self._gallery_sidebar

    def _gallery_sidebar_activate(self, index):
        """Switch to the canvas clicked in the docked strip, keeping it open."""
        if index is None or index == self.active_index():
            return
        self.select(index)
        if self._on_activate is not None:
            self._on_activate(index)

    def _rebuild_layout(self):
        """Pack the editor, the full-screen board or the docked strip.

        A docked strip keeps the editor visible beside it; the full-screen
        board replaces the editor.
        """
        for widget in (self.editor, self._gallery_dock):
            if widget is not None:
                widget.pack_forget()
        # The full-screen board is a child of the canvas area and absent from
        # the container tree, so it is forgotten and re-packed by hand.
        if self.gallery is not None:
            self.gallery.pack_forget()
        # Full-screen wins over the docked strip while it is open, but the
        # strip's state is kept so closing it brings the dock back.
        if self._gallery_shown and self.gallery is not None:
            self.gallery.pack(fill=tk.BOTH, expand=True)
        elif self._gallery_side is not None and self._gallery_dock is not None:
            self._gallery_dock.pack(side=self._gallery_side, fill=tk.Y)
            self.editor.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            # The strip's real parent is the canvas area, so its sibling dock
            # frame would otherwise paint over (and hide) it on Windows. Raise
            # the strip above the dock, as the sidebar does for grouped panels.
            self._gallery_sidebar.lift()
        else:
            self.editor.pack(fill=tk.BOTH, expand=True)

    def show_gallery(self):
        """Replace the tabbed editor with the thumbnail board.

        A docked strip, if any, is left configured but hidden behind the board;
        :meth:`hide_gallery` restores it.
        """
        gallery = self._ensure_gallery()
        selected = self.active_index() or 0
        self._gallery_shown = True
        gallery.set_documents([view.doc for view in self._views], selected)
        self._rebuild_layout()
        gallery.focus_canvas()

    def hide_gallery(self):
        """Return to the tabbed editor, leaving the current selection alone."""
        self._gallery_shown = False
        self._rebuild_layout()

    def show_gallery_side(self, side="left"):
        """Dock the thumbnail strip beside the canvas instead of replacing it.

        The editor stays live, so the user can paint and watch a thumbnail
        update; the strip is a plain vertical list that fits each canvas to the
        rail's width.
        """
        from .gallery import GalleryDock
        self._gallery_shown = False
        sidebar = self._ensure_gallery_sidebar()
        side = "right" if side == "right" else "left"
        if self._gallery_dock is None or self._gallery_side != side:
            if self._gallery_dock is not None:
                self._gallery_dock.destroy()
            self._gallery_side = side
            self._gallery_dock = GalleryDock(self, sidebar, side=side)
        selected = self.active_index() or 0
        self._rebuild_layout()
        sidebar.set_documents([view.doc for view in self._views], selected)
        # Re-measure once the rail has its real width, so every thumbnail fits
        # it rather than the placeholder width used before the dock settled.
        self.after_idle(self._refresh_side_layout)
        sidebar.focus_canvas()

    def _refresh_side_layout(self):
        """Re-fit the docked strip once its rail has a real width."""
        if self._gallery_side is None or self._gallery_sidebar is None:
            return
        self._gallery_sidebar.lift()
        self._gallery_sidebar.set_documents(
            [view.doc for view in self._views], self.active_index() or 0
        )

    def hide_gallery_side(self):
        """Remove the docked strip and let the editor take the freed space."""
        if self._gallery_dock is None:
            self._gallery_side = None
            return
        self._gallery_side = None
        self._rebuild_layout()

    def refresh_gallery_side(self):
        """Re-sync a docked strip's thumbnails without resetting its scroll."""
        if self._gallery_side is None or self._gallery_sidebar is None:
            return
        selected = self.active_index() or 0
        self._gallery_sidebar.set_documents(
            [view.doc for view in self._views], selected
        )

    def gallery_visible(self):
        return self._gallery_shown

    def gallery_side_visible(self):
        return self._gallery_side is not None

    def gallery_side(self):
        return self._gallery_side

    def gallery_zoom(self):
        return self.gallery.zoom if self.gallery is not None else None

    def _gallery_open(self, index):
        """Open the thumbnail the user chose and leave the full-screen board."""
        self.select(index)
        self.hide_gallery()
        self._on_activate(index)
        if self._on_gallery_close is not None:
            self._on_gallery_close()

    def _gallery_close(self):
        """Leave the full-screen board (a docked strip, if any, returns)."""
        self.hide_gallery()
        index = self.active_index()
        if index is not None:
            self._on_activate(index)
        if self._on_gallery_close is not None:
            self._on_gallery_close()

    def _add(self, doc):
        page = tk.Frame(self.stack, bg=rgb_to_hex(self.bg_color))
        # The page backdrop is document-view chrome, not app chrome: keep the
        # off-canvas color in both themes.
        theme.skip_bg(page)
        # An inner frame holds the rulers and the view so they stay flush and
        # line up with the pixels they annotate, with no gap to the tab edges.
        inner = tk.Frame(page, bg=rgb_to_hex(self.bg_color))
        theme.skip_bg(inner)
        inner.grid(row=0, column=0, sticky="nsew")
        page.rowconfigure(0, weight=1)
        page.columnconfigure(0, weight=1)

        view = CanvasView(inner, doc, bg=self.bg_color)
        corner = tk.Frame(inner, bg=rgb_to_hex(self.bg_color))
        theme.register(corner, bg="ruler_bg")
        top_ruler = RulerCanvas(inner, view, orient=tk.HORIZONTAL)
        left_ruler = RulerCanvas(inner, view, orient=tk.VERTICAL)

        view.grid(row=1, column=1, sticky="nsew")
        top_ruler.grid(row=0, column=1, sticky="ew")
        left_ruler.grid(row=1, column=0, sticky="ns")
        corner.grid(row=0, column=0, sticky="nsew")
        inner.rowconfigure(1, weight=1)
        inner.columnconfigure(1, weight=1)

        view.attach_rulers((corner, top_ruler, left_ruler),
                           (top_ruler, left_ruler))
        view.set_rulers_visible(self.rulers_visible)

        view.on_guides_changed = self._on_guides_changed
        view.on_guides_edited = self._guides_edited
        view.set_guides_visible(self.guides_visible)
        view.set_shared_guides(self._shared_guides)
        view.set_grid(self.grid_visible, self.grid_size)
        view.set_grid_snap(self.grid_snap)

        doc.widget = view
        self._pages.append(page)
        self._views.append(view)
        self._on_view_created(view)

    def set_rulers_visible(self, visible):
        """Show or hide the ruler strips around every canvas tab."""
        self.rulers_visible = bool(visible)
        for view in self._views:
            view.set_rulers_visible(self.rulers_visible)

    def set_guides_visible(self, visible):
        """Show or hide the ruler guides on every canvas tab."""
        self.guides_visible = bool(visible)
        for view in self._views:
            view.set_guides_visible(self.guides_visible)

    def set_grid_visible(self, visible):
        """Show or hide the grid overlay on every canvas tab."""
        self.grid_visible = bool(visible)
        for view in self._views:
            view.set_grid(self.grid_visible, None)

    def set_layer_outline_visible(self, visible):
        """Show or hide the active-layer outline on every canvas tab."""
        for view in self._views:
            view.set_layer_outline(visible)

    def set_outline_group(self, group):
        """Outline ``group``'s members instead of the active layer (``None`` clears)."""
        for view in self._views:
            view.outline_group = group
            view._draw_doc_border()

    def set_grid_size(self, size):
        """Set the grid cell size (document pixels) on every canvas tab."""
        try:
            size = max(1, int(size))
        except (TypeError, ValueError):
            return
        self.grid_size = size
        for view in self._views:
            view.set_grid(None, self.grid_size)

    def set_grid_snap(self, enabled):
        """Enable or disable grid snapping on every canvas tab."""
        self.grid_snap = bool(enabled)
        for view in self._views:
            view.set_grid_snap(self.grid_snap)

    def set_shared_guides(self, guides):
        """Point every tab at ``guides`` (a shared list) or at its own (``None``)."""
        self._shared_guides = guides
        for view in self._views:
            view.set_shared_guides(guides)

    def _guides_edited(self, source):
        """Repaint the guides of every tab except ``source`` after a shared edit."""
        if self._shared_guides is None:
            return
        for view in self._views:
            if view is not source:
                view.set_shared_guides(self._shared_guides)

    def _tab_selected(self, index):
        if not self._ready:
            return
        if not (0 <= index < len(self._views)):
            return
        self._selected = index
        self._show_page(index)
        self._on_activate(index)

    def _tab_renamed(self, index):
        if not (0 <= index < len(self._views)):
            return
        self._on_rename(index)

    def _tab_context(self, index, x_root, y_root):
        if not (0 <= index < len(self._views)):
            return
        if self._on_context is not None:
            self._on_context(index, x_root, y_root)

    def _tab_move_committed(self):
        """Report a finished tab drag so the app can snapshot it for undo."""
        active_doc = None
        if self._views:
            index = self.active_index()
            active_doc = self._views[index].doc
        if self._on_reorder is not None:
            self._on_reorder([view.doc for view in self._views], active_doc)

    def _trigger_add(self):
        if self._on_add_canvas is not None:
            self._on_add_canvas()
