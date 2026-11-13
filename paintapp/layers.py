"""The layer list panel."""

import tkinter as tk
import weakref
from tkinter import font as tkfont, ttk

from PIL import Image, ImageTk

from . import document, filters, gradient, icons, path, textrender, theme, vector
from .canvasarea import checkerboard
from .scrollbar import ArrowScrollbar
from .ui import PopupMenu, Tooltip

# Longest edge of a layer row's preview thumbnail, in pixels. Every row is the
# same height (thumbnail plus padding) so the list stays a tidy column even when
# neighbouring rows have no preview to show.
THUMB = 22
ROW_PAD = 1
ROW_H = THUMB + 2 * ROW_PAD

# Gap between a layer's preview and its mask preview, in pixels, so the two
# thumbnails read as separate cells instead of one wide image.
MASK_GAP = 2

# Hairline rule drawn along the bottom of every row, in pixels, so neighbouring
# layers, groups and their sub-rows read as separate entries. It is a seam
# rather than a full border: ``DIVIDER_MIX`` is how far the rule leans from the
# row's own fill toward the theme border.
ROW_DIVIDER = 1
DIVIDER_MIX = 0.6

# Padding inside each row, above and below its content, in pixels, so an item
# is inset on both sides and the divider lands in the middle of the space
# between one row's content and the next's. A row is its content height plus a
# pad on each side, and the rows are laid out on that same pitch, which the
# scroll maths follows.
ROW_GAP = 4
ROW_PITCH = ROW_H + 2 * ROW_GAP


def _uid(target):
    """Cache key of a layer or group: its stable ``uid``.

    Unlike ``id()`` it survives an undo/redo restore, which rebuilds the layer
    objects, so unchanged rows keep their cached thumbnails. Objects without a
    ``uid`` (test doubles) fall back to ``id``.
    """
    uid = getattr(target, "uid", None)
    return ("uid", uid) if uid is not None else ("id", id(target))


def thumbnail(image):
    """A tiny ``RGB`` preview of ``image`` over a checkerboard.

    ``Image.BOX`` averages whole source boxes, so at this reduction it is
    both cheaper than a windowed filter and a truer thumbnail; the
    checkerboard stands in for transparency, as in the Assets panel. Pure PIL
    work, so a project load can build it off the Tk thread.
    """
    source = image if image.mode == "RGBA" else image.convert("RGBA")
    width, height = source.size
    scale = min(THUMB / max(1, width), THUMB / max(1, height), 1.0)
    target = (max(1, round(width * scale)), max(1, round(height * scale)))
    board = checkerboard(target)
    board.alpha_composite(source.resize(target, Image.BOX))
    return board.convert("RGB")


def prepare_thumbnail(layer):
    """Precompute ``layer``'s row preview, stamped with its pixel token.

    :meth:`LayerPanel._build_photos` uses it while the token still matches,
    so only the cheap ``PhotoImage`` conversion is left for the Tk thread.
    """
    layer._prepared_thumb = (
        layer.pixels_token, thumbnail(layer.composited_image(include_mask=False))
    )
# Tint a mask preview paints the kept region with, over the checkerboard.
# Matches the Region Prompts panel so every mask thumbnail reads the same way.
MASK_TINT = (90, 170, 255)

# How long a layer's pixels must sit still before its preview is rebuilt, in
# milliseconds. Downscaling a full-resolution layer costs milliseconds per
# megapixel, so a preview is not rebuilt while the user is still painting: every
# new edit pushes the rebuild back until the stroke has settled.
PREVIEW_DELAY = 400

# Left padding for each nesting level, in pixels: a group's members sit one
# level in, and their filter/text rows two, so the indentation reads as
# ownership. A group's own filter rows are one level in.
INDENT = 16

# How far the pointer must move before a press on a layer row turns into a
# reorder drag, in pixels. Matches the Assets panel's slop, so a click that
# wobbles a little still just selects.
DRAG_SLOP = 6

# Longest edge of the reorder drag ghost, in pixels: a little larger than a row's
# thumbnail so the carried layer reads as something held, not as a second row.
GHOST = 44

# Leading icons for a layer's indented object rows, in place of the old "T "/"P "
# text prefixes. Each names a vendored Tabler outline icon (see
# :mod:`paintapp.icons`), a text glyph fallback for when the rasteriser is
# missing, a theme color token and a tooltip, and is drawn in the row's leading
# preview cell. ``glyph`` keeps the object's initial so a text-only fallback
# still reads.
FILTER_LEAD = ("color-filter", "\u2022", "muted", "Filter")
TEXT_LEAD = ("typography", "T", "muted", "Text")
PATH_LEAD = ("vector-bezier", "P", "muted", "Path")
GRADIENT_LEAD = ("color-filter", "G", "muted", "Gradient")
VECTOR_LINE_LEAD = ("vector-bezier", "V", "muted", "Vector line")
VECTOR_FILL_LEAD = ("vector-polygon", "F", "muted", "Vector fill")

# A group header's disclosure chevron, drawn as a small icon before the name in
# place of the "\u25be"/"\u25b8" text glyph it used to lead with; the glyph stays
# as the fallback. It sits beside the name rather than in the preview cell, which
# the group's composited thumbnail occupies.
GROUP_EXPANDED = ("chevron-down", "\u25be", "muted", "Collapse group")
GROUP_COLLAPSED = ("chevron-right", "\u25b8", "muted", "Expand group")

# A layer header's disclosure chevron; like a group it sits before the name and
# collapses the layer's indented object rows without affecting its pixels.
LAYER_EXPANDED = ("chevron-down", "\u25be", "muted", "Collapse layer")
LAYER_COLLAPSED = ("chevron-right", "\u25b8", "muted", "Expand layer")

# A group header's type icon, a folder, drawn beside the chevron so a group reads
# as a container the way the object rows read as a filter, text or path.
GROUP_ICON = ("folder", "\u229e", "muted", "Group")

# Compact glyphs for the primary action row. The full action set also lives in
# the "more" menu and in the list's right-click menu, so these stay icon-only
# and narrow enough that the whole row fits the rail at its minimum width.
#
# The eye button is the one-click shortcut for the "more" menu's Show/Hide
# action: a filled dot while the target is visible, a hollow one while it is
# hidden, so its state reads without opening a menu.
ICON_EYE_ON = "\u25c9"
ICON_EYE_OFF = "\u25cb"
ICON_ADD = "+"
ICON_GROUP = "\u229e"
ICON_DUPLICATE = "\u29c9"
ICON_DELETE = "\u2715"
ICON_MOVE_UP = "\u25b2"
ICON_MOVE_DOWN = "\u25bc"
ICON_MORE = "\u22ef"

# Status icons for a layer row, shown at the right edge of the row instead of
# the parenthesised marks the list used to spell out after the name. Each badge
# names a vendored Tabler outline icon (see :mod:`paintapp.icons`), a theme
# color token and a tooltip, so the meaning reads at a glance without the row
# growing a text suffix. The glyphs below are the fallback used only when the
# icon rasteriser is unavailable. The blend mode keeps its name (there is no
# single icon for the full mode list) and rides the same strip as a text badge.
ICON_SIZE = 15
# Space between a row's status badges, in pixels, so adjacent state icons read
# as separate marks rather than one crowded cluster.
BADGE_GAP = 5
# The action-strip buttons draw one Tabler icon each, a little larger than the
# row badges so they stay tappable, with a couple of transparent pixels of
# padding baked into the icon (Tk ignores a button's own padx/pady for images).
BUTTON_ICON = 16
BUTTON_PAD = 2
ICON_MASTER = "\u2605"
ICON_LINK = "\u2192"
ICON_LOCKED = "\U0001f512"
ICON_MASK = "\u25d0"
ICON_CLIPPED = "\u2934"
ICON_VECTOR = "\u25c7"

# Tail appended to a row name that cannot fit its label, so a long layer,
# group, filter or text name ends in an ellipsis instead of being clipped or
# pushing the row's status icons off the right edge.
ELLIPSIS = "\u2026"

# Horizontal chrome a Tk label spends on border and padding, in pixels, kept
# out of the width the name is measured against so the last glyph is not clipped.
LABEL_CHROME = 4


def _fit_text(font, text, width):
    """Trim ``text`` to ``width`` pixels, ending with an ellipsis if needed.

    ``font`` measures the string in the same face the label draws it in. The
    longest prefix that still leaves room for :data:`ELLIPSIS` is found by
    binary search so the caller can run this on every resize cheaply.
    """
    if width <= 0 or font.measure(text) <= width:
        return text
    tail = font.measure(ELLIPSIS)
    if width <= tail:
        return ELLIPSIS
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if font.measure(text[:mid]) + tail <= width:
            low = mid
        else:
            high = mid - 1
    return text[:low].rstrip() + ELLIPSIS



def _gray(hex_color, factor=1.0):
    """Return ``hex_color`` as a gray hex string, scaled by ``factor``.

    A hidden layer is drawn in grayscale rather than carrying a "(hidden)"
    mark. The tone is the luminance of the color the theme would otherwise
    use, so it still sits correctly on a light or dark panel, and ``factor``
    darkens it a little to mark a hidden row that is also selected.
    """
    if not isinstance(hex_color, str) or not hex_color.startswith("#") \
            or len(hex_color) != 7:
        return hex_color
    try:
        red = int(hex_color[1:3], 16)
        green = int(hex_color[3:5], 16)
        blue = int(hex_color[5:7], 16)
    except ValueError:
        return hex_color
    level = round(0.299 * red + 0.587 * green + 0.114 * blue)
    level = max(0, min(255, round(level * factor)))
    return f"#{level:02x}{level:02x}{level:02x}"


def _divider_color(bg):
    """A faint rule colour for a row filled with ``bg``.

    The rule under each row blends the row's own fill toward the theme border,
    so it reads as a subtle seam on the normal, selected and hidden fills alike
    rather than a hard line that fights whatever colour the row currently shows.
    A non-hex ``bg`` (a test double) falls back to the border colour.
    """
    border = theme.color("border")
    if not isinstance(bg, str) or not bg.startswith("#") or len(bg) != 7:
        return border
    try:
        red = int(bg[1:3], 16)
        green = int(bg[3:5], 16)
        blue = int(bg[5:7], 16)
        bred = int(border[1:3], 16)
        bgreen = int(border[3:5], 16)
        bblue = int(border[5:7], 16)
    except (ValueError, TypeError, IndexError):
        return border
    return "#%02x%02x%02x" % (
        round(red + (bred - red) * DIVIDER_MIX),
        round(green + (bgreen - green) * DIVIDER_MIX),
        round(blue + (bblue - blue) * DIVIDER_MIX),
    )


class LayerPanel(tk.Frame):
    """Presentational layer list; all mutations are delegated to callbacks.

    The list is drawn top-most-first, matching how layers stack visually, as a
    scrolling column of row widgets rather than a listbox: each layer row leads
    with a tiny preview of that layer's image over a checkerboard, so a layer is
    recognizable at a glance without selecting it. Each layer's non-destructive
    filters and its editable text, path and vector objects are listed on
    indented rows directly below it, and a layer header can be collapsed just
    like a group header to hide those rows. Rows are tracked in
    :attr:`_rows` so a selected row maps
    back to its layer (``("layer", index)``), filter
    (``("filter", layer, index)``), text (``("text", layer, index)``) or path
    (``("path", layer, index)``). A layer row or a group header can also be
    dragged to a new position in the stack: a washed-out ghost of the row follows
    the cursor, an insertion marker shows where it will land, and
    ``on_reorder(source, target, above)`` is fired once on release so the
    coordinator can apply the move. ``source`` and ``target`` are row
    descriptors (``("layer", index)`` or ``("group", group)``), so dropping a
    layer on a group's header or between its members adds it to that group, and
    a dragged group moves with all of its members.

    Previewing a layer costs a full-resolution downscale (milliseconds per
    megapixel), so each row's preview is cached against the layer's pixel
    version and only rebuilt after :meth:`sync_previews` sees an edit and the
    pixels have been still for a moment; see :data:`PREVIEW_DELAY`.

    The primary actions are compact icon buttons (Show/Hide/Add/Add Vector/
    Group/Duplicate/Delete/Move up/Move down) that act on whichever row is
    selected - the active
    layer for a layer row, the selected filter or text object for one of their
    rows. The Show/Hide button is a one-click shortcut for the layer or group
    visibility toggle, its dot reflecting the current state, so it never needs
    the "more" menu. The Group button adds a new empty group, whose header is
    listed at the top
    until it has members. The remaining actions (Rename, Show/Hide, Lock, Clear,
    Merge, Add Group, Group Selected Layers) live behind the "more" menu, and the
    same full set is available from the list's right-click menu, so none of them
    is clipped when the rail is narrow. Layer-only actions are hidden or disabled
    while a filter or text row is selected, and selecting a group header row
    swaps the "more"/right-click menus for the group actions
    (Rename, Show/Hide, Expand/Collapse, Ungroup,
    Flatten, Delete). A group is only a way to organize layers, so its header is
    not an edit target: selecting one leaves the active layer untouched and
    disables the layer-only appearance controls (opacity, blend and masks).
    A group's own non-destructive filters are listed on indented rows under its
    header (matching ``("group_filter", group, index)``); they edit the group's
    members composited together, so one filter can affect several layers at once.
    """

    def __init__(self, parent, on_add, on_delete, on_move, on_toggle,
                 on_merge, on_select, on_opacity, on_rename, on_lock,
                 on_add_mask, on_delete_mask, on_apply_mask, on_mask_edit,
                 on_outline=None,
                 on_filter_select=None, on_filter_edit=None, on_duplicate=None,
                 on_clear=None, on_filter_toggle=None, on_filter_remove=None,
                 on_filter_move=None, on_filter_clear=None,
                 on_text_select=None, on_text_edit=None,
                 on_text_remove=None, on_text_move=None,
                 on_path_select=None, on_path_edit=None,
                 on_path_remove=None, on_path_move=None,
                 on_apply_filters=None, on_rasterize_texts=None,
                 on_rasterize_paths=None, on_rasterize_path=None,
                 on_apply_filter=None, on_rasterize_text=None,
                 on_group_select=None, on_group_rename=None,
                 on_group_toggle=None, on_group_collapse=None,
                 on_group_ungroup=None, on_group_flatten=None,
                 on_group_delete=None,
                 on_group_move=None, on_group_create=None, on_group_add=None,
                 on_group_assign=None, on_clip=None, on_blend=None,
                 on_group_filter_select=None, on_group_filter_edit=None,
                 on_group_filter_toggle=None, on_group_filter_remove=None,
                 on_group_filter_move=None, on_group_filter_clear=None,
                 on_group_filter_apply=None, on_isolate=None, on_show_all=None,
                 on_unisolate=None, is_isolated=None,
                 on_reorder=None, on_add_to_assets=None, on_master_convert=None,
                 on_master_remove=None,
                 on_link_create=None, on_unlink=None, on_select_master=None,
                 on_link_reset_transform=None,
                 get_master_choices=None, on_move_to_canvas=None,
                 get_canvas_choices=None, on_group_duplicate=None,
                 on_text_duplicate=None, on_text_toggle=None,
                 on_path_duplicate=None, on_path_toggle=None,
                 on_gradient_select=None, on_gradient_edit=None,
                 on_gradient_remove=None, on_gradient_move=None,
                 on_gradient_duplicate=None, on_gradient_toggle=None,
                 on_rasterize_gradients=None, on_rasterize_gradient=None,
                 on_vector_select=None, on_vector_edit=None,
                 on_vector_remove=None, on_vector_duplicate=None,
                 on_vector_move=None, on_layer_collapse=None,
                 on_add_vector=None):
        super().__init__(parent)
        self._on_group_duplicate = on_group_duplicate
        self._on_text_duplicate = on_text_duplicate
        self._on_text_toggle = on_text_toggle
        self._on_path_duplicate = on_path_duplicate
        self._on_path_toggle = on_path_toggle
        self._on_gradient_select = on_gradient_select
        self._on_gradient_edit = on_gradient_edit
        self._on_gradient_remove = on_gradient_remove
        self._on_gradient_move = on_gradient_move
        self._on_gradient_duplicate = on_gradient_duplicate
        self._on_gradient_toggle = on_gradient_toggle
        self._on_rasterize_gradients = on_rasterize_gradients
        self._on_rasterize_gradient = on_rasterize_gradient
        self._on_vector_select = on_vector_select
        self._on_vector_edit = on_vector_edit
        self._on_vector_remove = on_vector_remove
        self._on_vector_duplicate = on_vector_duplicate
        self._on_vector_move = on_vector_move
        self._on_layer_collapse = on_layer_collapse
        self._on_add_vector = on_add_vector
        self._on_add = on_add
        self._on_delete = on_delete
        self._on_duplicate = on_duplicate
        self._on_clear = on_clear
        self._on_move = on_move
        self._on_toggle = on_toggle
        self._on_merge = on_merge
        self._on_select = on_select
        self._on_opacity = on_opacity
        self._on_rename = on_rename
        self._on_lock = on_lock
        self._on_add_mask = on_add_mask
        self._on_delete_mask = on_delete_mask
        self._on_apply_mask = on_apply_mask
        self._on_mask_edit = on_mask_edit
        self._on_outline = on_outline
        self._on_filter_select = on_filter_select
        self._on_filter_edit = on_filter_edit
        self._on_filter_toggle = on_filter_toggle
        self._on_filter_remove = on_filter_remove
        self._on_filter_move = on_filter_move
        self._on_filter_clear = on_filter_clear
        self._on_text_select = on_text_select
        self._on_text_edit = on_text_edit
        self._on_text_remove = on_text_remove
        self._on_text_move = on_text_move
        self._on_path_select = on_path_select
        self._on_path_edit = on_path_edit
        self._on_path_remove = on_path_remove
        self._on_path_move = on_path_move
        self._on_apply_filters = on_apply_filters
        self._on_rasterize_texts = on_rasterize_texts
        self._on_rasterize_paths = on_rasterize_paths
        self._on_rasterize_path = on_rasterize_path
        self._on_apply_filter = on_apply_filter
        self._on_rasterize_text = on_rasterize_text
        self._on_group_select = on_group_select
        self._on_group_rename = on_group_rename
        self._on_group_toggle = on_group_toggle
        self._on_group_collapse = on_group_collapse
        self._on_group_ungroup = on_group_ungroup
        self._on_group_flatten = on_group_flatten
        self._on_group_delete = on_group_delete
        self._on_group_move = on_group_move
        self._on_group_create = on_group_create
        self._on_group_add = on_group_add
        self._on_group_assign = on_group_assign
        self._on_clip = on_clip
        self._on_blend = on_blend
        self._on_group_filter_select = on_group_filter_select
        self._on_group_filter_edit = on_group_filter_edit
        self._on_group_filter_toggle = on_group_filter_toggle
        self._on_group_filter_remove = on_group_filter_remove
        self._on_group_filter_move = on_group_filter_move
        self._on_group_filter_clear = on_group_filter_clear
        self._on_group_filter_apply = on_group_filter_apply
        self._on_isolate = on_isolate
        self._on_show_all = on_show_all
        self._on_unisolate = on_unisolate
        self._is_isolated = is_isolated
        self._on_reorder = on_reorder
        self._on_add_to_assets = on_add_to_assets
        self._on_master_convert = on_master_convert
        self._on_master_remove = on_master_remove
        self._on_link_create = on_link_create
        self._on_unlink = on_unlink
        self._on_select_master = on_select_master
        self._on_link_reset_transform = on_link_reset_transform
        self._get_master_choices = get_master_choices
        self._on_move_to_canvas = on_move_to_canvas
        self._get_canvas_choices = get_canvas_choices
        self._rows = []
        self._row_widgets = []
        self._row_index = {}
        self._reuse_rows = None
        self._preview_cache = {}
        self._mask_preview_cache = {}
        self._preview_job = None
        self._layers = []
        self._groups = []
        self._active = 0
        self._selection_kind = "layer"
        self._selected = []
        self._anchor = None
        self._drop_layer = None
        self._selected_filter = None
        self._selected_text = None
        self._selected_path = None
        self._selected_gradient = None
        self._selected_vector = None
        self._selected_group = None
        self._selected_group_filter = None
        self._mask_layer = None
        # Reorder-drag state: the press that may become a drag, the row being
        # dragged once it has, its ghost window and the insertion marker.
        self._press = None
        self._drag_row = None
        self._ghost = None
        self._drop_line = None
        self._build()
        self.bind("<Destroy>", self._on_destroy)
        theme.subscribe(self._on_theme)

    def _build(self):
        # A frame rule above and below the scrolling list, so the column of rows
        # reads as a bounded region rather than bleeding into the panel.
        list_border_top = tk.Frame(self, height=1, bd=0, highlightthickness=0)
        list_border_top.pack(fill=tk.X)
        theme.register(list_border_top, bg="border")
        list_frame = tk.Frame(self)
        list_frame.pack(fill=tk.X)
        scrollbar = ArrowScrollbar(list_frame)
        self.canvas = tk.Canvas(
            list_frame, height=ROW_H * 6, highlightthickness=0,
            bg=theme.color("listbox_bg"), takefocus=True,
        )
        scrollbar.set_command(self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Arrow keys walk the list only while it holds keyboard focus, so a
        # focused combobox or text field keeps its own arrow behaviour. Clicking
        # a row (see :meth:`_on_press`) or the empty canvas focuses the list.
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<Up>", lambda _e: self._on_arrow(-1))
        self.canvas.bind("<Down>", lambda _e: self._on_arrow(1))
        self.inner = tk.Frame(self.canvas, bg=theme.color("listbox_bg"))
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # A plain wheel over a rail list scrolls the rail (the sidebar owns that
        # policy), so Ctrl+wheel is what scrolls this list itself; it must reach
        # the row widgets embedded in the canvas, hence the recursive bind.
        self._bind_wheel(self.canvas)

        list_border_bottom = tk.Frame(self, height=1, bd=0, highlightthickness=0)
        list_border_bottom.pack(fill=tk.X)
        theme.register(list_border_bottom, bg="border")

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(4, 0))

        def icon_button(name, glyph, command, tooltip):
            button = tk.Button(
                buttons, padx=3, pady=2,
                borderwidth=1, highlightthickness=0, command=command,
            )
            button.pack(side=tk.LEFT, padx=1)
            self._set_button_icon(button, name, glyph)
            return button, Tooltip(button, tooltip)

        self.eye_btn, self.eye_tip = icon_button(
            "eye", ICON_EYE_ON, self._handle_toggle, "Show/hide layer"
        )
        self.add_btn, self.add_tip = icon_button(
            "plus", ICON_ADD, self._on_add, "Add layer"
        )
        self.add_vector_btn, self.add_vector_tip = icon_button(
            "vector", ICON_VECTOR, self._on_add_vector, "Add vector layer"
        )
        self.group_btn, self.group_tip = icon_button(
            "folder-plus", ICON_GROUP, self._on_group_add, "Add group"
        )
        self.duplicate_btn, self.duplicate_tip = icon_button(
            "copy", ICON_DUPLICATE, self._handle_duplicate, "Duplicate layer"
        )
        self.delete_btn, self.delete_tip = icon_button(
            "trash", ICON_DELETE, self.delete_current, "Delete layer"
        )
        self.up_btn, self.up_tip = icon_button(
            "arrow-up", ICON_MOVE_UP, lambda: self._handle_move(1), "Move up"
        )
        self.down_btn, self.down_tip = icon_button(
            "arrow-down", ICON_MOVE_DOWN, lambda: self._handle_move(-1),
            "Move down",
        )
        self.more_btn, self.more_tip = icon_button(
            "dots", ICON_MORE, self._post_overflow, "More layer actions"
        )

        blend_row = tk.Frame(self)
        blend_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(blend_row, text="Blend:").pack(side=tk.LEFT)
        self.blend_var = tk.StringVar(value="Normal")
        self.blend_menu = ttk.Combobox(
            blend_row, textvariable=self.blend_var,
            values=document.blend_mode_labels(),
            state="readonly", width=1,
        )
        self.blend_menu.bind("<<ComboboxSelected>>", self._handle_blend)
        self.blend_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # ttk builds the combobox drop-down lazily and rebuilds its rows on
        # every post, so the group headers are recolored from its <Map> handler
        # (mirrors GenerationPanel's model picker) rather than once.
        self._blend_popdown_path = None
        self._blend_headers = document.blend_headers()
        self.after_idle(self._blend_popdown)

        opacity_row = tk.Frame(self)
        opacity_row.pack(fill=tk.X, pady=(4, 2))
        tk.Label(opacity_row, text="Opacity:").pack(side=tk.LEFT)
        self.opacity_var = tk.IntVar(value=100)
        self.opacity_scale = tk.Scale(
            opacity_row, from_=0, to=100, orient=tk.HORIZONTAL,
            variable=self.opacity_var, command=self._handle_opacity, length=140,
        )
        self.opacity_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        mask_row = tk.Frame(self)
        mask_row.pack(fill=tk.X, pady=(2, 4))
        self.mask_edit_var = tk.BooleanVar(value=False)
        self.mask_edit_btn = tk.Checkbutton(
            mask_row, text="Edit mask", variable=self.mask_edit_var,
            command=self._handle_mask_edit,
        )
        self.mask_edit_btn.pack(side=tk.LEFT)
        self.mask_add_btn = tk.Button(
            mask_row, text="Set \u25be", width=6, padx=0, command=self._post_mask_menu
        )
        self.mask_add_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.mask_add_tip = Tooltip(
            self.mask_add_btn,
            "Add a mask that reveals or hides the selection (or the whole layer)",
        )
        self.apply_mask_btn = tk.Button(mask_row, text="Apply", width=5, command=self._on_apply_mask)
        self.apply_mask_btn.pack(side=tk.LEFT, padx=2)
        self.delete_mask_btn = tk.Button(mask_row, text="Del", width=4, command=self._on_delete_mask)
        self.delete_mask_btn.pack(side=tk.LEFT, padx=2)

        outline_row = tk.Frame(self)
        outline_row.pack(fill=tk.X, pady=(0, 8))
        self.outline_var = tk.BooleanVar(value=True)
        self.outline_btn = tk.Checkbutton(
            outline_row, text="Show layer outline", variable=self.outline_var,
            command=self._handle_outline,
        )
        self.outline_btn.pack(side=tk.LEFT)
        self.outline_tip = Tooltip(
            self.outline_btn,
            "Outline the active layer's bounds on the canvas",
        )
        self._sync_button_state()

    # ------------------------------------------------------------------ public
    def refresh(self, layers, active, selected=None, filter_selection=None,
                text_selection=None, path_selection=None, group_selection=None,
                groups=None, group_filter_selection=None,
                vector_selection=None, gradient_selection=None):
        """Rebuild the list from ``layers`` and select the ``active`` layer.

        Group members are shown indented beneath a group header row that can be
        collapsed. ``groups`` is the canvas's full group list; any group with no
        members (an empty group added on its own) is listed by its header at the
        top. When ``selected`` (document-order indices) is given, those
        layer rows are selected instead of just the active one.
        ``filter_selection`` is an optional ``(layer_index, filter_index)`` whose
        row is also selected, ``text_selection`` an optional
        ``(layer_index, text_index)`` and ``path_selection`` an optional
        ``(layer_index, path_index)`` doing the same for a text or path row.
        ``group_selection`` is an optional group whose header row is selected
        instead, and ``group_filter_selection`` an optional
        ``(group, filter_index)`` selecting one of the group's filter rows.
        ``vector_selection`` is an optional ``(layer_index, item_index)`` for
        one of a vector layer's editable strokes/fills. ``gradient_selection``
        is an optional ``(layer_index, gradient_index)`` for one of the layer's
        editable gradient fills.
        """
        # Keep the existing rows alive so unchanged entries can be reused in
        # place. Collapsing, deleting or toggling one row then only touches the
        # rows whose content really changed, instead of tearing down and
        # recreating every widget in the list.
        old_widgets = self._row_widgets
        self._reuse_rows = {}
        for widgets in old_widgets:
            key = widgets.get("key")
            if key is not None:
                self._reuse_rows[key] = widgets
        self._rows = []
        self._row_widgets = []
        self._row_index = {}
        self._drop_layer = None
        # A rebuild can drop an in-flight reorder drag's origin row, so drop the
        # press/ghost/marker state rather than leave it dangling.
        self._press = None
        self._drag_row = None
        self._end_ghost()
        self._drop_line = None
        self._layers = layers
        self._groups = list(groups) if groups else []
        # Preview photos outlive a rebuild (that is what makes re-listing cheap,
        # including a return to a canvas), so drop only the ones whose layer has
        # been deleted since.
        self._drop_dead_previews()
        self._active = active
        layer_row = {}
        group_row = {}
        # Empty groups have no member rows to hang off, so their headers go
        # first, newest on top, above every layer row.
        member_groups = {getattr(layer, "group", None) for layer in layers}
        if groups:
            for group in reversed(groups):
                if group in member_groups:
                    continue
                self._add_row(
                    self._group_row_text(group), ("group", group),
                    hidden=not group.visible,
                    badges=self._group_badges(group, 0),
                    group=group,
                    name_icons=[
                        GROUP_COLLAPSED if group.collapsed else GROUP_EXPANDED,
                        GROUP_ICON,
                    ],
                    key=("group", _uid(group)),
                )
                group_row[id(group)] = len(self._rows) - 1
                self._insert_group_filter_rows(group)
        index = len(layers) - 1
        while index >= 0:
            layer = layers[index]
            group = getattr(layer, "group", None)
            if group is None:
                self._insert_layer_row(layers, index, layer_row, 0)
                index -= 1
                continue
            start = index
            while (start - 1 >= 0
                   and getattr(layers[start - 1], "group", None) is group):
                start -= 1
            self._add_row(
                self._group_row_text(group), ("group", group),
                hidden=not group.visible,
                badges=self._group_badges(group, index - start + 1),
                group=group,
                name_icons=[
                    GROUP_COLLAPSED if group.collapsed else GROUP_EXPANDED,
                    GROUP_ICON,
                ],
                key=("group", _uid(group)),
            )
            group_row[id(group)] = len(self._rows) - 1
            self._insert_group_filter_rows(group)
            if not group.collapsed:
                for member in range(index, start - 1, -1):
                    self._insert_layer_row(layers, member, layer_row, 1)
            index = start - 1

        # Drop the widgets whose rows are gone (a collapsed child, a deleted
        # layer) and re-pack the survivors in the new order. Re-pack is cheap and
        # keeps reused rows in the right place without recreating them; when the
        # order is unchanged the rows are left untouched entirely.
        used = {id(widgets["row"]) for widgets in self._row_widgets}
        for widgets in old_widgets:
            if id(widgets["row"]) not in used:
                try:
                    widgets["row"].destroy()
                except tk.TclError:
                    pass
        previous = [widgets["row"] for widgets in old_widgets]
        current = [widgets["row"] for widgets in self._row_widgets]
        if current != previous:
            # Re-packing an already packed widget only reconfigures it and
            # leaves it where it was, so a re-inserted row (an expanded layer's
            # object rows) would stay at the bottom. Unpack the survivors first,
            # then pack them in the new order; Tk redraws once afterwards.
            for widgets in self._row_widgets:
                widgets["row"].pack_forget()
            for widgets in self._row_widgets:
                widgets["row"].pack(fill=tk.X)
        self._reuse_rows = None

        self._layout_dividers()
        self._selection_kind = "layer"
        self._selected = []
        self._anchor = None
        self._selected_filter = None
        self._selected_text = None
        self._selected_path = None
        self._selected_gradient = None
        self._selected_group = None
        self._selected_group_filter = None
        self._selected_vector = None
        # A specific group filter row takes precedence over the bare group
        # header, so adding a filter leaves the new row selected (as it does for
        # a layer's filters).
        if group_filter_selection is not None:
            group, filter_index = group_filter_selection
            for row, entry in enumerate(self._rows):
                if entry == ("group_filter", group, filter_index):
                    self._selected = [row]
                    self._anchor = row
                    self._selection_kind = "group_filter"
                    self._selected_group = group
                    self._selected_group_filter = (group, filter_index)
                    break
        elif group_selection is not None and id(group_selection) in group_row:
            self._selected = [group_row[id(group_selection)]]
            self._anchor = self._selected[0]
            self._selection_kind = "group"
            self._selected_group = group_selection
        elif filter_selection is not None:
            layer_index, filter_index = filter_selection
            for row, entry in enumerate(self._rows):
                if entry == ("filter", layer_index, filter_index):
                    self._selected = [row]
                    self._anchor = row
                    self._selection_kind = "filter"
                    self._selected_filter = (layer_index, filter_index)
                    break
        elif text_selection is not None:
            layer_index, text_index = text_selection
            for row, entry in enumerate(self._rows):
                if entry == ("text", layer_index, text_index):
                    self._selected = [row]
                    self._anchor = row
                    self._selection_kind = "text"
                    self._selected_text = (layer_index, text_index)
                    break
        elif vector_selection is not None:
            layer_index, item_index = vector_selection
            for row, entry in enumerate(self._rows):
                if entry == ("vector", layer_index, item_index):
                    owner = layer_row.get(layer_index)
                    # Keep the owning layer highlighted alongside its object,
                    # so adding a vector stroke does not make the layer row look
                    # deselected.
                    self._selected = sorted(
                        {row, owner} - {None}
                    )
                    self._anchor = row
                    self._selection_kind = "vector"
                    self._selected_vector = (layer_index, item_index)
                    break
            else:
                # A collapsed layer hides the object row; keep its header
                # selected rather than leaving the list with no selection.
                owner = layer_row.get(layer_index)
                if owner is not None:
                    self._selected = [owner]
                    self._anchor = owner
        elif path_selection is not None:
            layer_index, path_index = path_selection
            for row, entry in enumerate(self._rows):
                if entry == ("path", layer_index, path_index):
                    self._selected = [row]
                    self._anchor = row
                    self._selection_kind = "path"
                    self._selected_path = (layer_index, path_index)
                    break
        elif gradient_selection is not None:
            layer_index, gradient_index = gradient_selection
            for row, entry in enumerate(self._rows):
                if entry == ("gradient", layer_index, gradient_index):
                    self._selected = [row]
                    self._anchor = row
                    self._selection_kind = "gradient"
                    self._selected_gradient = (layer_index, gradient_index)
                    break
        else:
            for index in selected if selected is not None else [active]:
                row = layer_row.get(index)
                if row is not None and row not in self._selected:
                    self._selected.append(row)
            self._selected.sort()
            if self._selected:
                self._anchor = self._selected[0]
        if self._anchor is None:
            self._anchor = layer_row.get(active, 0)
        self._apply_selection_colors()
        if layers:
            self.opacity_var.set(round(layers[active].opacity * 100))
            self.blend_var.set(
                document.BLEND_LABELS.get(getattr(layers[active], "blend", "normal"),
                                          "Normal")
            )
            self.update_mask_state(layers[active])
        else:
            self.opacity_var.set(100)
            self.blend_var.set("Normal")
            self.update_mask_state(None)
        self._sync_button_state()

    def _add_row(self, text, entry, indent=0, layer=None, hidden=False,
                 badges=None, group=None, leading=None, name_icons=None,
                 key=None):
        """Append one row widget and record it against its ``entry``.

        Every row is a fixed height (thumbnail plus padding) and reserves the
        same leading cell, so rows line up whether or not they carry a preview.
        ``indent`` is a nesting level, not a pixel count; ``layer`` is the layer
        row's document object, whose pixels lead the row as a small preview, and
        ``group`` is a group row's :class:`~paintapp.document.Group`, whose
        members are composited into the same preview cell. ``hidden`` marks a
        layer or group row whose target is not visible, so its preview and label
        are drawn in grayscale rather than a "(hidden)" mark. ``leading`` is an
        optional ``(icon, glyph, color token, tooltip)`` type icon drawn in the
        preview cell of a row that has no preview (a filter, text or path row),
        replacing the text prefix such a row used to spell out. ``name_icons`` is
        an optional list of such descriptors drawn in order just before the name,
        for markers that belong beside the text rather than in the preview cell
        (a group's disclosure chevron and folder type icon). ``badges`` is a
        list of ``(icon, glyph, color token, tooltip)`` status icons pinned to
        the right edge of the row, replacing the parenthesised marks that used
        to trail the name. ``icon`` names a vendored Tabler SVG (``None`` for a
        text-only badge) and ``glyph`` is the fallback when the icon rasteriser
        is missing. ``key`` is a stable identity used to reuse the row's widgets
        across refreshes; rows with the same key and structure are updated in
        place instead of being rebuilt. Returns the row index.
        """
        bg = theme.color("listbox_bg")
        token, photos = None, None
        mask_token, mask_photos = None, None
        if layer is not None:
            token, photos = self._cached_preview(layer)
            if layer.mask is not None:
                mask_token, mask_photos = self._cached_mask_preview(layer)
        elif group is not None:
            token, photos = self._cached_group_preview(group)
        photo = photos[0] if photos else None
        if leading is not None and photo is not None:
            leading = None
        has_mask = mask_photos is not None
        badges = list(badges or ())
        name_icons = list(name_icons or ())
        # Everything that changes the widget tree rather than just its colors,
        # text or images. A matching signature means the old row can be updated
        # in place; a mismatch means it must be rebuilt.
        signature = (
            indent, layer is not None, group is not None, has_mask,
            leading, len(name_icons),
        )
        old = None
        if key is not None and self._reuse_rows:
            old = self._reuse_rows.pop(key, None)
            if old is not None:
                try:
                    if not old["row"].winfo_exists():
                        old = None
                except tk.TclError:
                    old = None
        if old is not None and old.get("signature") == signature:
            self._reuse_row(
                old, entry, text, hidden, badges, name_icons,
                token, photos, mask_token, mask_photos,
            )
            return len(self._rows) - 1
        # The row is its content height plus a pad above and below, so the
        # content sits inset and the rule below it lands squarely in the gap to
        # the next row rather than hugging the thumbnail.
        row = tk.Frame(self.inner, bg=bg, height=ROW_H + 2 * ROW_GAP)
        row.pack(fill=tk.X)
        row.pack_propagate(False)
        widgets = [row]
        if indent:
            spacer = tk.Frame(row, width=indent * INDENT, bg=bg)
            spacer.pack(side=tk.LEFT, fill=tk.Y)
            widgets.append(spacer)
        cell = tk.Frame(row, width=THUMB, height=ROW_H, bg=bg)
        cell.pack(side=tk.LEFT, fill=tk.Y)
        cell.pack_propagate(False)
        widgets.append(cell)
        # The label is made either way, so a preview that could not be built has
        # somewhere to appear once it can.
        thumb = tk.Label(cell, bg=bg, bd=0)
        if photo is not None:
            thumb.config(image=photo)
        thumb.place(relx=0.5, rely=0.5, anchor="center")
        widgets.append(thumb)
        # A row without a preview (a filter, text or path row) shows its type
        # icon in the same leading cell, so it lines up under the layer's
        # thumbnail instead of pushing its name rightward with a text prefix.
        if leading is not None and photo is None:
            self._paint_leading(thumb, leading, bg, theme.color(leading[2]))
            Tooltip(thumb, leading[3])
        else:
            leading = None
        # A mask rides next to the layer's own preview, so the mask and the
        # pixels it shapes are read together without opening the mask tools.
        mask_cell = None
        mask_thumb = None
        mask_photo = None
        mask_gap = None
        if mask_photos is not None:
            mask_gap = tk.Frame(row, width=MASK_GAP, bg=bg)
            mask_gap.pack(side=tk.LEFT, fill=tk.Y)
            widgets.append(mask_gap)
            mask_cell = tk.Frame(row, width=THUMB, height=ROW_H, bg=bg)
            mask_cell.pack(side=tk.LEFT, fill=tk.Y)
            mask_cell.pack_propagate(False)
            widgets.append(mask_cell)
            mask_thumb = tk.Label(mask_cell, bg=bg, bd=0)
            mask_photo = mask_photos[0] if mask_photos else None
            if mask_photo is not None:
                mask_thumb.config(image=mask_photo)
            mask_thumb.place(relx=0.5, rely=0.5, anchor="center")
            Tooltip(mask_thumb, "Layer mask")
            widgets.append(mask_thumb)
        # Status icons claim the right edge before the name, so a long layer
        # name can shrink the label but never clip the icons off the row.
        badge_widgets = []
        badge_strip = None
        if badges:
            strip = tk.Frame(row, bg=bg)
            strip.pack(side=tk.RIGHT, fill=tk.Y)
            widgets.append(strip)
            badge_strip = strip
            for icon_name, glyph, token_name, tooltip in badges:
                icon = tk.Label(strip, bg=bg, bd=0)
                self._paint_badge(
                    icon, icon_name, glyph, bg, theme.color(token_name)
                )
                icon.pack(side=tk.LEFT, padx=(0, BADGE_GAP))
                icon_tip = Tooltip(icon, tooltip)
                widgets.append(icon)
                badge_widgets.append(
                    (icon, icon_name, glyph, token_name, icon_tip)
                )
        # Name icons sit between the preview cell and the name, packed before
        # the label so they take their width from the left like the cells do.
        name_icon_widgets = []
        for position, name_icon in enumerate(name_icons):
            icon_label = tk.Label(row, bg=bg, bd=0)
            self._paint_leading(icon_label, name_icon, bg,
                                theme.color(name_icon[2]))
            icon_label.pack(side=tk.LEFT, padx=((4 if position == 0 else 1), 0))
            icon_tip = Tooltip(icon_label, name_icon[3])
            widgets.append(icon_label)
            name_icon_widgets.append((icon_label, icon_tip, name_icon))
        label = tk.Label(row, text=text, anchor="w", bg=bg, fg=theme.color("text"))
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        widgets.append(label)
        # A hairline seam along the row's bottom edge. Created last so it stacks
        # above the full-height indent and preview cells, and placed (not
        # packed) so adding it never disturbs the row's fixed-height layout.
        # The left gutter is filled in later by :meth:`_layout_dividers`, once
        # every row is known, so the rule can tell an in-group separator from a
        # group boundary; here it just spans the full width.
        divider = tk.Frame(row, height=ROW_DIVIDER, bd=0, highlightthickness=0,
                           bg=_divider_color(bg))
        divider.place(relx=0.0, x=0, rely=1.0, relwidth=1.0, anchor="sw")
        widgets.append(divider)
        self._rows.append(entry)
        index = len(self._rows) - 1
        label.bind("<Configure>", self._on_row_configure)
        # The name's tooltip stays empty until the label actually has to trim
        # the name, so hovering only explains a name the row cannot show whole.
        name_tip = Tooltip(label, "")
        for widget in widgets:
            self._row_index[widget] = index
            widget.bind("<Button-1>", self._on_row_press)
            widget.bind("<B1-Motion>", self._on_motion)
            widget.bind("<ButtonRelease-1>", self._on_release)
            widget.bind("<Double-Button-1>", self._on_row_double)
            widget.bind("<Button-3>", self._on_row_context)
        self._row_widgets.append({
            "row": row, "cell": cell, "label": label,
            "thumb": thumb, "photos": photos, "photo": photo, "token": token,
            "leading": leading,
            "name_icons": name_icon_widgets,
            "mask_cell": mask_cell, "mask_thumb": mask_thumb,
            "mask_gap": mask_gap,
            "mask_photos": mask_photos, "mask_photo": mask_photo,
            "mask_token": mask_token,
            "hidden": bool(hidden), "badges": badge_widgets,
            "badge_strip": badge_strip,
            "divider": divider,
            "indent": indent,
            "full_text": text, "font": None,
            "tooltip": name_tip, "truncated": None,
            "key": key, "signature": signature, "_widgets": widgets,
        })
        return index

    def _reuse_row(self, widgets, entry, text, hidden, badges, name_icons,
                   token, photos, mask_token, mask_photos):
        """Update a row's mutable parts in place and record it for this build.

        Keeps the frame, thumbnail cells and layout, so dropping an object or
        collapsing a group does not flash the whole list. Only the status badges
        are added, removed or retargeted in place; indent, masks and name-icon
        count changes never reach here, they get a new row in :meth:`_add_row`.
        """
        index = len(self._rows)
        self._rows.append(entry)
        widgets["hidden"] = bool(hidden)
        widgets["full_text"] = text
        widgets["truncated"] = None
        label = widgets["label"]
        if label.cget("text") != text:
            label.config(text=text)
        widgets["tooltip"].set_text("")
        try:
            width = label.winfo_width()
        except tk.TclError:
            width = 0
        if width > 1:
            self._fit_label(widgets, width)
        widgets["photos"] = photos
        widgets["photo"] = None
        widgets["token"] = token
        widgets["thumb"].config(image="")
        if widgets.get("mask_thumb") is not None:
            widgets["mask_photos"] = mask_photos
            widgets["mask_photo"] = None
            widgets["mask_token"] = mask_token
            widgets["mask_thumb"].config(image="")
        for slot, name_icon in enumerate(name_icons):
            icon_tip = widgets["name_icons"][slot][1]
            icon_tip.set_text(name_icon[3])
        widgets["name_icons"] = [
            (icon_label, icon_tip, name_icon)
            for (icon_label, icon_tip, _old), name_icon
            in zip(widgets["name_icons"], name_icons)
        ]
        self._sync_badges(widgets, badges)
        self._row_widgets.append(widgets)
        for widget in widgets["_widgets"]:
            self._row_index[widget] = index

    def _sync_badges(self, widgets, badges):
        """Match a reused row's status badges to ``badges``.

        A row can gain or lose badges without changing its widget structure (a
        collapsed group shows its member count, a layer shows or hides its
        state icons), so the strip and its labels are reconciled in place rather
        than forcing the whole row to be rebuilt.
        """
        bg = theme.color("listbox_bg")
        row = widgets["row"]
        label = widgets["label"]
        strip = widgets.get("badge_strip")
        if badges and strip is None:
            strip = tk.Frame(row, bg=bg)
            # Pin the strip to the right edge; it must precede the expanding
            # label in pack order, so insert it before rather than appending.
            strip.pack(side=tk.RIGHT, fill=tk.Y, before=label)
            widgets["badge_strip"] = strip
            widgets["_widgets"].append(strip)
        elif not badges and strip is not None:
            for icon, _name, _glyph, _token, _tip in widgets["badges"]:
                try:
                    widgets["_widgets"].remove(icon)
                except ValueError:
                    pass
            try:
                strip.destroy()
            except tk.TclError:
                pass
            widgets["badge_strip"] = None
            widgets["badges"] = []
            widgets["_widgets"].remove(strip)
            return
        current = widgets["badges"]
        for slot, badge in enumerate(badges):
            icon_name, glyph, token_name, tooltip = badge
            if slot < len(current):
                icon, _old_name, _old_glyph, _old_token, icon_tip = current[slot]
                icon_tip.set_text(tooltip)
                current[slot] = (icon, icon_name, glyph, token_name, icon_tip)
            else:
                icon = tk.Label(strip, bg=bg, bd=0)
                self._paint_badge(icon, icon_name, glyph, bg,
                                  theme.color(token_name))
                icon.pack(side=tk.LEFT, padx=(0, BADGE_GAP))
                icon_tip = Tooltip(icon, tooltip)
                widgets["_widgets"].append(icon)
                current.append((icon, icon_name, glyph, token_name, icon_tip))
        for icon, _name, _glyph, _token, _tip in current[len(badges):]:
            try:
                icon.destroy()
            except tk.TclError:
                pass
            widgets["_widgets"].remove(icon)
        del current[len(badges):]

    def _layout_dividers(self):
        """Place each row's bottom rule, trimmed to its indent, past the last row.

        A rule between two nested rows starts at the shallower of their two
        indents, so a group's members and their sub-rows read as one bracket
        under the header. The rule below a group's last member is the boundary
        to the next top-level entry, so it spans the full width: taking the
        minimum of an indent and a top-level ``0`` gives zero there by itself.
        The final row is left without a rule, since the panel's bottom border
        already closes the list. Called once per rebuild, after every row exists.
        """
        last = len(self._row_widgets) - 1
        for index, widgets in enumerate(self._row_widgets):
            divider = widgets.get("divider")
            if divider is None:
                continue
            if index >= last:
                try:
                    divider.place_forget()
                except tk.TclError:
                    pass
                continue
            next_indent = self._row_widgets[index + 1].get("indent", 0)
            gutter = min(widgets.get("indent", 0), next_indent) * INDENT
            try:
                divider.place_configure(x=gutter, width=-gutter)
            except tk.TclError:
                pass

    def _fit_row_label(self, event, index):
        """Trim row ``index``'s name to its label width with an ellipsis.

        Bound to the name label's ``<Configure>``, so a long name ends in a
        trailing ellipsis whenever the row is narrowed, and comes back in full
        when it widens. The label is packed to fill the row and expand, so the
        shrinking text never changes the row's layout.
        """
        if index >= len(self._row_widgets):
            return
        self._fit_label(self._row_widgets[index], event.width)

    def _fit_label(self, widgets, width):
        """Trim ``widgets``' name to ``width`` pixels with an ellipsis.

        Used both by the label's ``<Configure>`` handler and when a reused row's
        name changes, since the label width does not change in that case and no
        ``<Configure>`` event fires.
        """
        full_text = widgets.get("full_text")
        if not full_text:
            return
        label = widgets["label"]
        font = widgets.get("font")
        if font is None:
            try:
                font = tkfont.Font(root=self, font=label.cget("font"))
            except tk.TclError:
                font = tkfont.nametofont("TkDefaultFont")
            widgets["font"] = font
        text = _fit_text(font, full_text, width - LABEL_CHROME)
        if label.cget("text") != text:
            label.config(text=text)
        # Only a trimmed name needs the tooltip, and it is retargeted only when
        # the trimmed/whole state flips so a resize does not churn the popup.
        truncated = text != full_text
        if widgets.get("truncated") != truncated:
            widgets["truncated"] = truncated
            widgets["tooltip"].set_text(full_text if truncated else "")

    def _paint_badge(self, icon, icon_name, glyph, bg, color):
        """Draw one status badge: its Tabler icon, or ``glyph`` as a fallback.

        ``icon_name`` is a file stem in the vendored icon set (or ``None`` for
        a text-only badge such as the blend mode or a group's member count).
        When the rasteriser is missing the badge falls back to the plain text
        ``glyph``, so the row still reads without the icon assets.
        """
        image = icons.photo(icon_name, ICON_SIZE, color)
        if image is not None:
            icon.config(image=image, text="", fg=color, bg=bg)
        else:
            icon.config(image="", text=glyph, fg=color, bg=bg)

    def _paint_leading(self, label, leading, bg, color):
        """Draw a row's leading type icon, or its glyph fallback.

        Mirrors :meth:`_paint_badge`: the vendored Tabler ``icon`` tinted
        ``color`` when the rasteriser is available, else the plain text
        ``glyph``. The label is centred in the preview cell, so either form
        lands where a layer's thumbnail would.
        """
        name, glyph, _token, _tip = leading
        image = icons.photo(name, ICON_SIZE, color)
        if image is not None:
            label.config(image=image, text="", fg=color, bg=bg)
        else:
            label.config(image="", text=glyph, fg=color, bg=bg)

    def _set_button_icon(self, button, name, glyph, enabled=True):
        """Give an action-strip button its Tabler icon, or its ``glyph``.

        A disabled button cannot dim an image the way Tk dims text, so it takes
        a muted-icon render instead. When the rasteriser is missing the button
        falls back to the plain text ``glyph`` and Tk's own disabled colouring.
        A button showing an image reads ``width`` in pixels rather than text
        units, so the icon path clears it (otherwise the button would clamp to a
        couple of pixels) and the text fallback restores the fixed slot. Tk also
        ignores ``padx``/``pady`` on an image button, so the icon is rendered
        with :data:`BUTTON_PAD` pixels of transparent margin for breathing room.
        """
        image = None
        if name:
            color = theme.color("text" if enabled else "muted")
            image = icons.photo(name, BUTTON_ICON, color, padding=BUTTON_PAD)
        if image is not None:
            button.config(image=image, text="", width=0, height=0)
        else:
            button.config(image="", text=glyph, width=2, height=0)

    # ---------------------------------------------------------------- previews
    def _cached_preview(self, layer):
        """The layer's ``(pixel token, preview photos)``, rebuilding if stale.

        A layer's preview is cached across list rebuilds keyed on the layer's
        pixel version, so re-listing layers (for a rename, a visibility change,
        a group collapse, or a return to a canvas) costs nothing while an actual
        edit rebuilds the one layer that changed. The cache holds the layer
        weakly, so it never keeps a deleted layer's pixels alive.
        """
        token = layer.pixels_token
        entry = self._preview_cache.get(_uid(layer))
        if entry is not None \
                and entry["token"] == token:
            return token, entry["photos"]
        photos = self._build_photos(layer)
        if photos[0] is None:
            return token, None
        self._preview_cache[_uid(layer)] = {
            "layer": weakref.ref(layer), "token": token, "photos": photos,
        }
        return token, photos

    def _build_photos(self, layer):
        """The ``(color, gray, dark)`` preview photos of ``layer``.

        The preview mirrors the layer's own content - its filtered pixels with
        any editable text or vector strokes drawn over them - so a vector or
        text layer is not blank in the list. The mask is left out because the
        row shows it as a separate preview beside this one. The grayscale pair
        is what a hidden layer's row shows, ``dark`` standing in for a hidden
        row that is also selected. Returns ``(None, None, None)`` when no
        preview can be made.
        """
        try:
            prepared = getattr(layer, "_prepared_thumb", None)
            if prepared is not None and prepared[0] == layer.pixels_token:
                color = prepared[1]
            else:
                composited = getattr(layer, "composited_image", None)
                source = (composited(include_mask=False)
                          if composited is not None else layer.image)
                color = self._thumb(source)
            # Used once; a later edit rebuilds the preview the normal way.
            if prepared is not None:
                layer._prepared_thumb = None
            gray = color.convert("L").convert("RGB")
            dark = Image.blend(gray, Image.new("RGB", gray.size, (0, 0, 0)), 0.25)
            return (
                ImageTk.PhotoImage(color),
                ImageTk.PhotoImage(gray),
                ImageTk.PhotoImage(dark),
            )
        except (tk.TclError, ValueError):
            return None, None, None

    def _cached_mask_preview(self, layer):
        """The layer mask's ``(pixel token, preview photos)``, rebuilt if stale.

        Mirrors :meth:`_cached_preview`; the token is the layer's pixel token,
        which already folds in ``_mask_token``, so a mask edit matches the
        layer preview's own staleness check. The cache holds the layer weakly.
        """
        token = layer.pixels_token
        entry = self._mask_preview_cache.get(_uid(layer))
        if entry is not None \
                and entry["token"] == token:
            return token, entry["photos"]
        photos = self._build_mask_photos(layer)
        if photos[0] is None:
            return token, None
        self._mask_preview_cache[_uid(layer)] = {
            "layer": weakref.ref(layer), "token": token, "photos": photos,
        }
        return token, photos

    def _build_mask_photos(self, layer):
        """The ``(color, gray, dark)`` preview photos of ``layer``'s mask.

        Kept regions are tinted over a checkerboard, matching the Region
        Prompts panel's mask thumbnails, so a mask's shape reads at a glance.
        The grayscale pair is what a hidden layer's row shows. Returns
        ``(None, None, None)`` when the layer has no mask or no preview can be
        made.
        """
        try:
            mask = layer.mask
            if mask is None:
                return None, None, None
            color = self._mask_thumb(mask)
            gray = color.convert("L").convert("RGB")
            dark = Image.blend(gray, Image.new("RGB", gray.size, (0, 0, 0)), 0.25)
            return (
                ImageTk.PhotoImage(color),
                ImageTk.PhotoImage(gray),
                ImageTk.PhotoImage(dark),
            )
        except (tk.TclError, ValueError):
            return None, None, None

    def _mask_thumb(self, mask):
        """A tiny ``RGB`` preview of a mask: its kept region tinted over a checker."""
        source = mask.convert("L")
        width, height = source.size
        scale = min(THUMB / max(1, width), THUMB / max(1, height), 1.0)
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        board = checkerboard(target)
        tint = Image.new("RGBA", target, MASK_TINT + (255,))
        tint.putalpha(source.resize(target, Image.BOX))
        board.alpha_composite(tint)
        return board.convert("RGB")

    def _cached_group_preview(self, group):
        """The group's ``(token, preview photos)``, rebuilding if stale.

        Mirrors :meth:`_cached_preview` for a layer, but the token folds in every
        visible member's pixel version, offset and opacity plus the group's own
        filter chain and opacity, so the row's thumbnail tracks any change to the
        group as a whole. The cache holds the group weakly.
        """
        token = self._group_token(group)
        entry = self._preview_cache.get(_uid(group))
        if entry is not None \
                and entry["token"] == token:
            return token, entry["photos"]
        photos = self._build_group_photos(group)
        if photos[0] is None:
            return token, None
        self._preview_cache[_uid(group)] = {
            "layer": weakref.ref(group), "token": token, "photos": photos,
        }
        return token, photos

    def _group_token(self, group):
        """A value that changes whenever the group's composite would.

        Combines each member's :attr:`~paintapp.document.Layer.pixels_token`
        with the inputs that place it in the group's composite (offset, opacity,
        visibility and clip flag), the group's own filter chain and its opacity.
        """
        members = group.members(self._layers)
        return (
            tuple(
                (
                    _uid(member), member.pixels_token, tuple(member.offset),
                    member.visible, member.opacity,
                    getattr(member, "clip", False),
                )
                for member in members
            ),
            filters.chain_key(group.filters),
            group.opacity,
        )

    def _build_group_photos(self, group):
        """The ``(color, gray, dark)`` preview photos of ``group``'s members.

        The composite mirrors what the group contributes to the canvas - its
        visible members flattened together, with the group's filters and opacity
        applied - so an organized or collapsed group still reads at a glance.
        Returns ``(None, None, None)`` when no preview can be made (for example a
        group with no visible member).
        """
        try:
            image = group.preview(self._layers)
            if image is None:
                return None, None, None
            color = self._thumb(image)
            gray = color.convert("L").convert("RGB")
            dark = Image.blend(gray, Image.new("RGB", gray.size, (0, 0, 0)), 0.25)
            return (
                ImageTk.PhotoImage(color),
                ImageTk.PhotoImage(gray),
                ImageTk.PhotoImage(dark),
            )
        except (tk.TclError, ValueError):
            return None, None, None

    def _thumb(self, image):
        """A tiny ``RGB`` preview of ``image`` over a checkerboard (see :func:`thumbnail`)."""
        return thumbnail(image)

    def sync_previews(self):
        """Rebuild previews whose layer pixels changed, once painting settles.

        Called as the document is redrawn, so a stroke marks its layer's preview
        as stale immediately but only pays for the downscale after the pixels
        have been still for :data:`PREVIEW_DELAY`; another edit in the meantime
        simply pushes the rebuild back. Nothing is done (and no timer is armed)
        while every preview is current, which is the common case.
        """
        if not self._stale_previews():
            return
        self._cancel_preview_job()
        try:
            self._preview_job = self.after(PREVIEW_DELAY, self._rebuild_previews)
        except tk.TclError:
            self._preview_job = None

    def _stale_previews(self):
        """True when some layer or group row's preview no longer matches."""
        for row, entry in enumerate(self._rows):
            if row >= len(self._row_widgets):
                continue
            if entry[0] == "layer":
                target = self._layer_at(entry[1])
                if target is None:
                    continue
                token = target.pixels_token
            elif entry[0] == "group":
                token = self._group_token(entry[1])
            else:
                continue
            if self._row_widgets[row]["token"] != token:
                return True
        return False

    def _rebuild_previews(self):
        """Repaint the previews of the layers and groups whose pixels changed."""
        self._preview_job = None
        if not self.winfo_exists():
            return
        for row, entry in enumerate(self._rows):
            if row >= len(self._row_widgets):
                continue
            if entry[0] == "layer":
                target = self._layer_at(entry[1])
                if target is None:
                    continue
                token = target.pixels_token
                build = self._build_photos
            elif entry[0] == "group":
                target = entry[1]
                token = self._group_token(target)
                build = self._build_group_photos
            else:
                continue
            widgets = self._row_widgets[row]
            if widgets["token"] == token:
                continue
            photos = build(target)
            widgets["token"] = token
            if photos[0] is None:
                # The target no longer has a preview (for example a group whose
                # every member is now hidden); clear the row's thumbnail rather
                # than leave a stale one showing.
                widgets["photos"] = None
                widgets["photo"] = None
                widgets["thumb"].config(image="")
                if widgets.get("mask_thumb") is not None:
                    widgets["mask_photos"] = None
                    widgets["mask_photo"] = None
                    widgets["mask_token"] = token
                    widgets["mask_thumb"].config(image="")
                continue
            widgets["photos"] = photos
            widgets["token"] = token
            self._preview_cache[_uid(target)] = {
                "layer": weakref.ref(target), "token": token, "photos": photos,
            }
            if entry[0] == "layer" and widgets.get("mask_thumb") is not None:
                mask_photos = self._build_mask_photos(target)
                widgets["mask_photos"] = mask_photos
                widgets["mask_photo"] = None
                widgets["mask_token"] = token
                if mask_photos[0] is None:
                    widgets["mask_thumb"].config(image="")
                else:
                    self._mask_preview_cache[_uid(target)] = {
                        "layer": weakref.ref(target), "token": token,
                        "photos": mask_photos,
                    }
            self._show_preview(widgets, row in self._selected)

    def _drop_dead_previews(self):
        """Forget cached previews whose layer has been deleted.

        An undo/redo restore rebuilds the layer objects but keeps each layer's
        ``uid``, so entries are first re-pointed at the live objects of the
        shown canvas; only previews whose layer is truly gone are dropped.
        """
        live = list(self._layers or []) + list(self._groups or [])
        for target in live:
            ref = weakref.ref(target)
            for cache in (self._preview_cache, self._mask_preview_cache):
                entry = cache.get(_uid(target))
                if entry is not None and entry["layer"]() is not target:
                    entry["layer"] = ref
        self._preview_cache = {
            key: entry for key, entry in self._preview_cache.items()
            if entry["layer"]() is not None
        }
        self._mask_preview_cache = {
            key: entry for key, entry in self._mask_preview_cache.items()
            if entry["layer"]() is not None
        }

    def _layer_at(self, index):
        """The layer with document index ``index`` on the shown canvas."""
        if 0 <= index < len(self._layers):
            return self._layers[index]
        return None

    def _cancel_preview_job(self):
        job = self._preview_job
        self._preview_job = None
        if job is None:
            return
        try:
            self.after_cancel(job)
        except tk.TclError:
            pass

    def _group_row_text(self, group):
        """The label for a group header row: its name, without the arrow.

        The disclosure chevron is drawn as an icon before the name (see
        :data:`GROUP_EXPANDED`/:data:`GROUP_COLLAPSED`), so the label itself is
        just the name. The collapsed member count rides the row as a right-edge
        badge (see :meth:`_group_badges`) and a hidden group carries no mark at
        all: its row is drawn in grayscale instead.
        """
        return group.name

    def _group_badges(self, group, member_count):
        """The right-edge ``(icon, glyph, color token, tooltip)`` badges of a group.

        A collapsed group shows how many layers it holds, matching the count
        that used to sit in parentheses after the name. The count is a text
        badge, so its icon name is ``None``.
        """
        if not group.collapsed:
            return []
        noun = "layer" if member_count == 1 else "layers"
        return [(None, str(member_count), "muted",
                 f"{member_count} {noun} hidden by the collapse")]

    def _layer_badges(self, layer):
        """The right-edge ``(icon, glyph, color token, tooltip)`` badges of a layer.

        ``icon`` names a vendored Tabler outline SVG and ``glyph`` is the text
        fallback used when the rasteriser is missing. These compact icons
        replace the parenthesised marks the row used to spell out after the
        name: hidden, master, link, locked, mask, clipped, vector and the blend
        mode (kept as a short text badge, since the full mode list has no single
        icon). A hidden layer leads with the crossed eye even though its row is
        also drawn in grayscale, so the state reads without relying on the
        dimming alone. A trailing text badge counts the layer's object rows
        (filters, texts, paths, vector strokes and fills), so a filter, a text
        or a vector stroke can be read at a glance.
        """
        badges = []
        if not layer.visible:
            badges.append(("eye-off", ICON_EYE_OFF, "muted", "Hidden"))
        if getattr(layer, "is_master", False):
            badges.append(("crown", ICON_MASTER, "accent", "Master layer"))
        if getattr(layer, "is_link", False):
            source = layer.content_source()
            if source is not layer:
                badges.append(("link", ICON_LINK, "info",
                               f"Linked to '{source.name}'"))
            else:
                badges.append(("link", ICON_LINK, "info", "Linked layer"))
        if layer.locked:
            badges.append(("lock", ICON_LOCKED, "warn", "Locked"))
        if getattr(layer, "mask", None) is not None:
            badges.append(("mask", ICON_MASK, "success", "Has a mask"))
        if getattr(layer, "clip", False):
            badges.append(("arrow-bar-to-down", ICON_CLIPPED, "accent",
                           "Clipped to the layer below"))
        if getattr(layer, "vector", False):
            badges.append(("vector", ICON_VECTOR, "accent", "Vector layer"))
        if getattr(layer, "blend", "normal") != "normal":
            label = document.BLEND_LABELS.get(layer.blend, layer.blend)
            badges.append((None, label, "muted", f"Blend mode: {label}"))
        count = self._layer_object_count(layer)
        noun = "object" if count == 1 else "objects"
        badges.append((None, str(count), "muted", f"{count} {noun}"))
        return badges

    def _layer_object_count(self, layer):
        """How many object rows ``layer`` lists beneath it.

        Filters, texts, paths, gradient fills, vector strokes and vector fills
        are all kept as editable specs and each gets an indented row, so the
        badge counts every one of them. The same total applies to vector and
        ordinary raster layers, which both may carry any of these kinds.
        """
        return sum(len(getattr(layer, attr, None) or ())
                   for attr in ("filters", "texts", "paths", "gradients",
                                "strokes", "fills"))

    def _insert_layer_row(self, layers, index, layer_row, indent):
        """Insert one layer row (and its indented object rows) into the list.

        A collapsed layer keeps its header but omits every object row beneath
        it, mirroring how a collapsed group hides its members.
        """
        layer = layers[index]
        collapsed = bool(getattr(layer, "collapsed", False))
        # Only a layer with object rows can be expanded, so a plain raster
        # layer gets no disclosure chevron to suggest hidden children.
        has_objects = self._layer_object_count(layer) > 0
        name_icons = []
        if has_objects:
            name_icons.append(
                LAYER_COLLAPSED if collapsed else LAYER_EXPANDED
            )
        layer_row[index] = self._add_row(
            layer.name, ("layer", index), indent, layer,
            hidden=not layer.visible, badges=self._layer_badges(layer),
            name_icons=name_icons,
            key=("layer", _uid(layer)),
        )
        if collapsed:
            return
        for filter_index, spec in enumerate(layer.filters):
            text = filters.describe(spec)
            if not spec.get("enabled", True):
                text += "  (off)"
            self._add_row(text, ("filter", index, filter_index), indent + 1,
                          leading=FILTER_LEAD,
                          key=("filter", _uid(layer), filter_index))
        for text_index, spec in enumerate(layer.texts):
            self._add_row(textrender.describe(spec), ("text", index, text_index),
                          indent + 1, leading=TEXT_LEAD,
                          hidden=bool(spec.get("hidden")),
                          key=("text", _uid(layer), text_index))
        for path_index, spec in enumerate(layer.paths):
            self._add_row(path.describe(spec), ("path", index, path_index),
                          indent + 1, leading=PATH_LEAD,
                          hidden=bool(spec.get("hidden")),
                          key=("path", _uid(layer), path_index))
        for gradient_index, spec in enumerate(
                getattr(layer, "gradients", None) or []):
            self._add_row(gradient.describe(spec),
                          ("gradient", index, gradient_index), indent + 1,
                          leading=GRADIENT_LEAD,
                          hidden=bool(spec.get("hidden")),
                          key=("gradient", _uid(layer), gradient_index))
        item_index = 0
        for spec in getattr(layer, "strokes", None) or []:
            self._add_row(vector.describe(spec), ("vector", index, item_index),
                          indent + 1, leading=VECTOR_LINE_LEAD,
                          key=("vector", _uid(layer), item_index))
            item_index += 1
        for spec in getattr(layer, "fills", None) or []:
            self._add_row(vector.describe_fill(spec),
                          ("vector", index, item_index), indent + 1,
                          leading=VECTOR_FILL_LEAD,
                          key=("vector", _uid(layer), item_index))
            item_index += 1

    def _insert_group_filter_rows(self, group):
        """Insert indented rows for a group's non-destructive filters.

        They sit directly under the group header (outside the member indent, so
        they read as belonging to the group rather than to one member) and stay
        visible while the group is collapsed, since the filter still applies.
        """
        for filter_index, spec in enumerate(group.filters):
            text = filters.describe(spec)
            if not spec.get("enabled", True):
                text += "  (off)"
            self._add_row(text, ("group_filter", group, filter_index), 1,
                          leading=FILTER_LEAD,
                          key=("group_filter", _uid(group), filter_index))

    def selected_group(self):
        """The group behind the selected row, or ``None``.

        A group header row selects its group; a group filter row also counts,
        since adding or baking a filter still targets that group.
        """
        if self._selection_kind in ("group", "group_filter"):
            return self._selected_group
        return None

    def update_mask_state(self, layer):
        """Enable only the mask controls that make sense for ``layer``.

        A group header is not an editable target, so while one is selected the
        mask controls stay disabled even though the active layer is unchanged.
        A link layer keeps its own mask editable, but there is nothing to bake a
        link's mask into (its pixels belong to the master), so Apply stays off.
        """
        self._mask_layer = layer
        has_layer = layer is not None
        has_mask = has_layer and getattr(layer, "mask", None) is not None
        is_link = bool(has_layer and getattr(layer, "is_link", False))
        is_group = (
            self._selection_kind in ("group", "group_filter")
            and self._selected_group is not None
        )
        self.mask_add_btn.config(
            state=tk.DISABLED if (not has_layer or has_mask or is_group) else tk.NORMAL
        )
        editable = has_mask and not is_group
        for button in (self.mask_edit_btn, self.delete_mask_btn):
            button.config(state=tk.NORMAL if editable else tk.DISABLED)
        self.apply_mask_btn.config(
            state=tk.NORMAL if (editable and not is_link) else tk.DISABLED
        )

    def set_mask_edit(self, enabled):
        """Reflect the app's mask-edit state without re-firing the callback."""
        self.mask_edit_var.set(bool(enabled))

    def set_layer_outline(self, enabled):
        """Reflect the app's layer-outline state without re-firing the callback."""
        self.outline_var.set(bool(enabled))

    def set_opacity(self, value):
        self.opacity_var.set(round(value * 100))

    def select_layer(self, index):
        """Highlight the row for layer ``index`` without rebuilding or dispatching.

        Used when the active layer is changed from outside the panel (for
        example by clicking a tile in the exploded view), so the list's
        selection follows the active layer without a full rebuild and without
        re-firing the select callback that would bounce back to the caller.
        """
        row = None
        for candidate, entry in enumerate(self._rows):
            if entry == ("layer", index):
                row = candidate
                break
        if row is None:
            return
        self._selected = [row]
        self._anchor = row
        self._selection_kind = "layer"
        self._selected_filter = None
        self._selected_text = None
        self._selected_path = None
        self._selected_gradient = None
        self._selected_group = None
        self._selected_group_filter = None
        self._apply_selection_colors()
        self._sync_button_state()

    def step_layer(self, delta):
        """The layer index ``delta`` visible rows from the selected layer.

        Arrow-key navigation steps between the layer rows that are actually on
        screen, skipping group headers and filter, text and path rows, so
        members of a collapsed group are not reachable and the walk matches the
        column the user sees. ``delta`` counts rows down the list, matching the
        Up (``-1``) and Down (``1``) arrow keys. Returns ``None`` at an end of
        the list or when no layer rows are listed.
        """
        rows = [i for i, entry in enumerate(self._rows) if entry[0] == "layer"]
        if not rows:
            return None
        current = None
        for i in rows:
            if self._rows[i] == ("layer", self._active):
                current = i
                break
        if current is None:
            target = rows[0] if delta > 0 else rows[-1]
            return self._rows[target][1]
        position = rows.index(current) + delta
        if position < 0 or position >= len(rows):
            return None
        return self._rows[rows[position]][1]

    def reveal_layer(self, index):
        """Scroll the row for layer ``index`` fully into view, if it is listed."""
        for row, entry in enumerate(self._rows):
            if entry == ("layer", index):
                self.reveal_row(row)
                return

    def reveal_row(self, row):
        """Scroll the list the least amount needed to show row ``row`` whole.

        Used after keyboard navigation so the newly selected layer is brought
        on screen without recentring the whole list under the user's cursor.
        """
        try:
            view_height = self.canvas.winfo_height()
        except tk.TclError:
            return
        if view_height <= 1:
            return
        total = self.inner.winfo_height() or len(self._rows) * ROW_PITCH
        if total <= view_height:
            return
        top = row * ROW_PITCH
        bottom = top + ROW_PITCH
        first = self.canvas.canvasy(0)
        if top < first:
            self.canvas.yview_moveto(top / total)
        elif bottom > first + view_height:
            self.canvas.yview_moveto((bottom - view_height) / total)

    def selected_indices(self, count):
        """Document-order indices of the selected layer rows (filters excluded)."""
        indices = set()
        for row in self._selected:
            if not (0 <= row < len(self._rows)):
                continue
            entry = self._rows[row]
            if entry[0] == "layer":
                indices.add(entry[1])
        return sorted(indices)

    def selected_groups(self):
        """The groups whose header rows are selected, in row order."""
        groups = []
        for row in sorted(self._selected):
            if not (0 <= row < len(self._rows)):
                continue
            entry = self._rows[row]
            if entry[0] == "group" and entry[1] not in groups:
                groups.append(entry[1])
        return groups

    def layer_row_at(self, x_root, y_root):
        """The layer index of the row under ``(x_root, y_root)``, or ``None``.

        Used to accept an asset dragged in from the Assets panel: a layer row,
        or an indented filter or text row (which belong to their layer), is a
        valid target; the pointer must also be over a row widget, so a drop
        elsewhere in the rail is ignored rather than snapping to a row.
        """
        try:
            widget = self.canvas.winfo_containing(x_root, y_root)
        except KeyError:
            return None
        row = self._row_under(widget)
        if row is None:
            return None
        entry = self._rows[row]
        if entry[0] in ("layer", "filter", "text", "path"):
            return entry[1]
        return None

    def _row_under(self, widget):
        """The row index a widget (or one of its ancestors) belongs to."""
        node = widget
        while node is not None:
            row = self._row_index.get(node)
            if row is not None:
                return row
            node = getattr(node, "master", None)
        return None

    def set_drop_target(self, layer_index):
        """Tint the layer row an incoming asset is hovering over (or clear it).

        The highlight is painted onto the row widgets rather than by changing
        the selection, so it neither disturbs the active layer nor trains the
        coordinator's selection state. Passing ``None`` clears it.
        """
        self._drop_layer = layer_index
        self._apply_selection_colors()

    def _apply_selection_colors(self):
        """Paint every row for the current selection and drop highlight."""
        sel_bg = theme.color("sel_bg")
        sel_fg = theme.color("sel_fg")
        accent = theme.color("accent")
        bg = theme.color("listbox_bg")
        fg = theme.color("text")
        muted = theme.color("muted")
        for row, entry in enumerate(self._rows):
            if row >= len(self._row_widgets):
                break
            selected = row in self._selected
            widgets = self._row_widgets[row]
            if (self._drop_layer is not None and entry[0] == "layer"
                    and entry[1] == self._drop_layer):
                row_bg, row_fg = accent, sel_fg
            elif widgets["hidden"]:
                # A hidden layer reads as grayscale instead of a "(hidden)"
                # mark, so the row keeps the panel's background and takes the
                # theme's muted grey; a hidden row that is also selected is
                # greyed a little darker rather than getting the accent fill,
                # so it stays grayscale.
                if selected:
                    row_bg, row_fg = _gray(bg, 0.85), _gray(muted, 0.7)
                else:
                    row_bg, row_fg = bg, muted
            elif selected:
                row_bg, row_fg = sel_bg, sel_fg
            else:
                row_bg, row_fg = bg, fg
            widgets["row"].config(bg=row_bg)
            widgets["cell"].config(bg=row_bg)
            widgets["thumb"].config(bg=row_bg)
            if widgets.get("leading") is not None:
                # A type icon keeps its own muted token on a plain row and takes
                # the row's foreground when the row is selected or hidden, so it
                # stays legible against the fill like the status badges do.
                self._paint_leading(
                    widgets["thumb"], widgets["leading"], row_bg,
                    row_fg if (selected or widgets["hidden"])
                    else theme.color(widgets["leading"][2]),
                )
            if widgets.get("name_icons"):
                # The name icons (a group's chevron and folder) follow the same
                # rule as the type icon: their own muted token on a plain row,
                # the row foreground when the row is selected or grayscale.
                for icon_label, _tip, name_icon in widgets["name_icons"]:
                    self._paint_leading(
                        icon_label, name_icon, row_bg,
                        row_fg if (selected or widgets["hidden"])
                        else theme.color(name_icon[2]),
                    )
            if widgets.get("mask_cell") is not None:
                widgets["mask_gap"].config(bg=row_bg)
                widgets["mask_cell"].config(bg=row_bg)
                widgets["mask_thumb"].config(bg=row_bg)
            widgets["label"].config(bg=row_bg, fg=row_fg)
            if widgets.get("divider") is not None:
                # The seam follows whatever fill the row currently shows, so a
                # selected or highlighted row keeps a subtle rule rather than a
                # stray line in the unselected background colour.
                widgets["divider"].config(bg=_divider_color(row_bg))
            if widgets["badge_strip"] is not None:
                widgets["badge_strip"].config(bg=row_bg)
            if widgets["badges"]:
                # The status icons keep their own theme color on a plain row;
                # on a selected, dropped or hidden row they take the row's
                # foreground so they stay legible against the fill.
                tinted = selected or widgets["hidden"] or (
                    self._drop_layer is not None and entry[0] == "layer"
                    and entry[1] == self._drop_layer)
                for icon, icon_name, glyph, token_name, _tip in widgets["badges"]:
                    self._paint_badge(
                        icon, icon_name, glyph, row_bg,
                        row_fg if tinted else theme.color(token_name),
                    )
            self._show_preview(widgets, selected)

    def _show_preview(self, widgets, selected):
        """Show a row's color, grayscale or dark-grayscale preview.

        A normal row shows the layer's color preview; a hidden row shows the
        grayscale one, or the darker grayscale while it is selected. A layer
        with a mask shows the same variant of its mask preview beside it. Does
        nothing when the row already shows the right photo or has none.
        """
        photos = widgets["photos"]
        if photos and photos[0] is not None:
            if widgets["hidden"]:
                photo = photos[2] if selected else photos[1]
            else:
                photo = photos[0]
            if widgets["photo"] is not photo:
                widgets["photo"] = photo
                widgets["thumb"].config(image=photo)
        mask_photos = widgets.get("mask_photos")
        if mask_photos and mask_photos[0] is not None:
            if widgets["hidden"]:
                mask_photo = mask_photos[2] if selected else mask_photos[1]
            else:
                mask_photo = mask_photos[0]
            if widgets["mask_photo"] is not mask_photo:
                widgets["mask_photo"] = mask_photo
                widgets["mask_thumb"].config(image=mask_photo)

    def _select_rows(self, rows, target):
        """Repaint the list for ``rows`` and route the ``target`` row.

        The target is the row the user acted on (which is not always the last
        row of a Shift range) so the coordinator's active layer follows the
        click rather than the range's end.
        """
        self._selected = sorted({row for row in rows if 0 <= row < len(self._rows)})
        self._apply_selection_colors()
        self._dispatch(target)

    def focus_list(self):
        """Give the list keyboard focus so the arrow keys walk it."""
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass

    def _on_canvas_press(self, _event):
        """A press on the empty canvas hands the list keyboard focus."""
        self.focus_list()

    def _on_arrow(self, delta):
        """Move the active layer with an arrow key while the list has focus.

        Bound on the canvas, so it only fires when the list itself is focused and
        never competes with a focused combobox or text field. ``delta`` follows
        the on-screen list (``-1`` up, ``1`` down); the moved-to row is selected
        and dispatched like a click, and scrolled into view.
        """
        target = self.step_layer(delta)
        if target is None or target == self._active:
            return "break"
        for row, entry in enumerate(self._rows):
            if entry == ("layer", target):
                self._active = target
                self._select_rows([row], row)
                self.reveal_row(row)
                break
        return "break"

    # ------------------------------------------------------------------ events
    def _row_for(self, event):
        """The current row index of the widget that raised ``event``.

        Bindings look the row up at event time rather than capturing an index at
        creation, so a row that survives a refresh can move to a new position
        without rebinding every widget.
        """
        return self._row_index.get(event.widget)

    def _on_row_press(self, event):
        row = self._row_for(event)
        if row is not None:
            self._on_press(event, row)

    def _on_row_double(self, event):
        row = self._row_for(event)
        if row is not None:
            self._on_double(event, row)

    def _on_row_context(self, event):
        row = self._row_for(event)
        if row is not None:
            self._on_context(event, row)

    def _on_row_configure(self, event):
        row = self._row_for(event)
        if row is not None:
            self._fit_row_label(event, row)

    def _on_press(self, event, row):
        """A click on a row: plain selects it, Ctrl toggles, Shift extends."""
        if not (0 <= row < len(self._rows)):
            return
        # The click also hands the list keyboard focus so the arrow keys walk it
        # (see :meth:`_on_arrow`).
        self.focus_list()
        # Remember the press so a following drag on the row can become a reorder
        # (see :meth:`_on_motion`); cleared by the drag, a double click or a
        # right click so neither starts one.
        self._press = {
            "row": row, "x_root": event.x_root, "y_root": event.y_root,
        }
        self._drag_row = None
        ctrl = bool(event.state & 0x0004)
        shift = bool(event.state & 0x0001)
        if ctrl:
            if row in self._selected and len(self._selected) > 1:
                self._selected.remove(row)
                self._selected.sort()
                self._anchor = self._selected[0]
                self._apply_selection_colors()
                self._dispatch(self._selected[0])
                return
            if row not in self._selected:
                self._selected.append(row)
            self._anchor = row
            self._select_rows(self._selected, row)
            return
        if shift and self._anchor is not None:
            low, high = sorted((self._anchor, row))
            self._select_rows(range(low, high + 1), row)
            return
        self._anchor = row
        self._select_rows([row], row)

    def _dispatch(self, row):
        """Route a row to the layer, filter or group callback."""
        if row < 0 or row >= len(self._rows):
            return
        entry = self._rows[row]
        if entry[0] == "group":
            self._selection_kind = "group"
            self._selected_group = entry[1]
            self._selected_filter = None
            self._selected_text = None
            self._selected_path = None
            self._selected_gradient = None
            self._selected_group_filter = None
            if self._on_group_select is not None:
                self._on_group_select(entry[1])
        elif entry[0] == "group_filter":
            self._selection_kind = "group_filter"
            self._selected_group = entry[1]
            self._selected_group_filter = (entry[1], entry[2])
            self._selected_filter = None
            self._selected_text = None
            self._selected_path = None
            self._selected_gradient = None
            if self._on_group_filter_select is not None:
                self._on_group_filter_select(entry[1], entry[2])
        elif entry[0] == "layer":
            self._selection_kind = "layer"
            self._selected_filter = None
            self._selected_text = None
            self._selected_path = None
            self._selected_gradient = None
            self._selected_group = None
            self._selected_group_filter = None
            self._on_select(entry[1])
        elif entry[0] == "text":
            self._selection_kind = "text"
            self._selected_text = (entry[1], entry[2])
            self._selected_path = None
            self._selected_gradient = None
            self._selected_filter = None
            self._selected_group = None
            self._selected_group_filter = None
            if self._on_text_select is not None:
                self._on_text_select(entry[1], entry[2])
        elif entry[0] == "vector":
            self._selection_kind = "vector"
            self._selected_vector = (entry[1], entry[2])
            self._selected_path = None
            self._selected_gradient = None
            self._selected_text = None
            self._selected_filter = None
            self._selected_group = None
            self._selected_group_filter = None
            if self._on_vector_select is not None:
                self._on_vector_select(entry[1], entry[2])
        elif entry[0] == "path":
            self._selection_kind = "path"
            self._selected_path = (entry[1], entry[2])
            self._selected_gradient = None
            self._selected_text = None
            self._selected_filter = None
            self._selected_group = None
            self._selected_group_filter = None
            if self._on_path_select is not None:
                self._on_path_select(entry[1], entry[2])
        elif entry[0] == "gradient":
            self._selection_kind = "gradient"
            self._selected_gradient = (entry[1], entry[2])
            self._selected_path = None
            self._selected_text = None
            self._selected_filter = None
            self._selected_group = None
            self._selected_group_filter = None
            if self._on_gradient_select is not None:
                self._on_gradient_select(entry[1], entry[2])
        else:
            self._selection_kind = "filter"
            self._selected_filter = (entry[1], entry[2])
            self._selected_text = None
            self._selected_gradient = None
            self._selected_group = None
            self._selected_group_filter = None
            if self._on_filter_select is not None:
                self._on_filter_select(entry[1], entry[2])
        self._sync_button_state()

    def _on_double(self, _event, row):
        """A double click edits an object row (or collapses a layer/group)."""
        if not (0 <= row < len(self._rows)):
            return
        # A double click is not a drag; forget the press that started it.
        self._press = None
        self._drag_row = None
        self._anchor = row
        self._select_rows([row], row)
        entry = self._rows[row]
        if entry[0] == "layer":
            if self._on_layer_collapse is not None:
                self._on_layer_collapse()
            else:
                self._on_rename()
        elif entry[0] == "group":
            if self._on_group_collapse is not None:
                self._on_group_collapse()
        elif entry[0] == "group_filter":
            if self._on_group_filter_edit is not None:
                self._on_group_filter_edit(entry[1], entry[2])
        elif entry[0] == "text":
            if self._on_text_edit is not None:
                self._on_text_edit(entry[1], entry[2])
        elif entry[0] == "path":
            if self._on_path_edit is not None:
                self._on_path_edit(entry[1], entry[2])
        elif entry[0] == "gradient":
            if self._on_gradient_edit is not None:
                self._on_gradient_edit(entry[1], entry[2])
        elif entry[0] == "vector":
            if self._on_vector_edit is not None:
                self._on_vector_edit(entry[1], entry[2])
        elif self._on_filter_edit is not None:
            self._on_filter_edit(entry[1], entry[2])

    # -------------------------------------------------------------- actions
    def _filter_active(self):
        """True when a filter row, not a layer row, is the current target."""
        return self._selection_kind == "filter" and self._selected_filter is not None

    def _group_filter_active(self):
        """True when a group's filter row is the current target."""
        return (
            self._selection_kind == "group_filter"
            and self._selected_group_filter is not None
        )

    def _text_active(self):
        """True when a text row, not a layer row, is the current target."""
        return self._selection_kind == "text" and self._selected_text is not None

    def _path_active(self):
        """True when a path row, not a layer row, is the current target."""
        return self._selection_kind == "path" and self._selected_path is not None

    def _gradient_active(self):
        """True when a gradient row, not a layer row, is the current target."""
        return (
            self._selection_kind == "gradient"
            and self._selected_gradient is not None
        )

    def _vector_active(self):
        """True when an editable vector row is the current target."""
        return (
            self._selection_kind == "vector"
            and self._selected_vector is not None
        )

    def select_vector_row(self, layer_index, item_index):
        """Highlight a vector row without rebuilding the list.

        Rebuilding from inside a row's click callback would destroy the widget
        between the two halves of a double-click, so selection-only updates use
        this method. Returns ``True`` when the row was found.
        """
        for row, entry in enumerate(self._rows):
            if entry == ("vector", layer_index, item_index):
                self._selection_kind = "vector"
                self._selected_vector = (layer_index, item_index)
                self._selected_path = None
                self._selected_gradient = None
                self._selected_text = None
                self._selected_filter = None
                self._selected_group = None
                self._selected_group_filter = None
                owner = next(
                    (i for i, candidate in enumerate(self._rows)
                     if candidate == ("layer", layer_index)),
                    None,
                )
                self._selected = sorted({row, owner} - {None})
                self._anchor = row
                # Repaint only: routing the row again would call back into the
                # app, which would re-enter this method through a canvas sync.
                self._apply_selection_colors()
                self._sync_button_state()
                return True
        return False

    def _visibility_state(self):
        """Whether the current target is visible, or ``None`` when it has none.

        Only a layer row (forwarded to the app's layer toggle) and a group
        header (its group toggle) carry a visibility the eye button can flip;
        a filter, text or path row returns ``None`` so the button is disabled
        there. A multi-layer selection counts as visible only when every layer
        is, matching how its other aggregate states are shown.
        """
        if self._vector_active():
            return None
        if self._text_active() or self._path_active() or self._gradient_active():
            spec = self._selected_object_spec()
            if spec is None:
                return None
            return not spec.get("hidden", False)
        if self._group_active():
            return bool(self._selected_group.visible)
        if self._filter_active() or self._group_filter_active():
            return None
        indices = self.selected_indices(len(self._layers))
        if not indices:
            return None
        return all(self._layers[index].visible for index in indices)

    def _sync_button_state(self):
        """Relabel and enable the shared buttons for the selected row type."""
        has_layers = bool(self._layers)
        is_filter = self._filter_active()
        is_text = self._text_active()
        is_path = self._path_active()
        is_gradient = self._gradient_active()
        is_vector = self._vector_active()
        is_item = is_text or is_path or is_gradient or is_vector
        is_group = self._selection_kind == "group" and self._selected_group is not None
        is_group_filter = self._group_filter_active()
        is_group_like = is_group or is_group_filter
        duplicate_enabled = bool(has_layers or is_group) and not (
            is_filter or is_group_filter
        )
        self.duplicate_btn.config(
            state=tk.NORMAL if duplicate_enabled else tk.DISABLED
        )
        self._set_button_icon(self.duplicate_btn, "copy", ICON_DUPLICATE,
                              enabled=duplicate_enabled)
        self.blend_menu.config(
            state="disabled"
            if (not has_layers or is_filter or is_item or is_group_like)
            else "readonly"
        )
        self.opacity_scale.config(
            state=tk.DISABLED if (not has_layers or is_group_like) else tk.NORMAL
        )
        group_enabled = not (is_filter or is_item or is_group_like)
        self.group_btn.config(state=tk.NORMAL if group_enabled else tk.DISABLED)
        self._set_button_icon(self.group_btn, "folder-plus", ICON_GROUP,
                              enabled=group_enabled)
        row_enabled = bool(has_layers or is_group_like)
        for button, name, glyph in (
            (self.delete_btn, "trash", ICON_DELETE),
            (self.up_btn, "arrow-up", ICON_MOVE_UP),
            (self.down_btn, "arrow-down", ICON_MOVE_DOWN),
            (self.more_btn, "dots", ICON_MORE),
        ):
            button.config(state=tk.NORMAL if row_enabled else tk.DISABLED)
            self._set_button_icon(button, name, glyph, enabled=row_enabled)
        visible = self._visibility_state()
        if visible is None:
            self.eye_btn.config(state=tk.DISABLED)
            self._set_button_icon(self.eye_btn, "eye", ICON_EYE_ON,
                                  enabled=False)
            self.eye_tip.set_text("Show/hide layer")
        else:
            if self._group_active():
                target = "group"
            elif is_text:
                target = "text"
            elif is_path:
                target = "path"
            elif is_gradient:
                target = "gradient"
            elif is_vector:
                target = "vector object"
            else:
                target = "layer"
            self.eye_btn.config(state=tk.NORMAL)
            self._set_button_icon(
                self.eye_btn, "eye" if visible else "eye-off",
                ICON_EYE_ON if visible else ICON_EYE_OFF, enabled=True,
            )
            self.eye_tip.set_text(
                f"Hide {target}" if visible else f"Show {target}"
            )
        if is_group:
            self.duplicate_tip.set_text("Duplicate group")
        elif is_text:
            self.duplicate_tip.set_text("Duplicate text")
        elif is_path:
            self.duplicate_tip.set_text("Duplicate path")
        elif is_gradient:
            self.duplicate_tip.set_text("Duplicate gradient")
        elif is_vector:
            self.duplicate_tip.set_text("Duplicate vector object")
        else:
            self.duplicate_tip.set_text("Duplicate layer")
        if is_group_filter:
            self.delete_tip.set_text("Remove filter")
            self.up_tip.set_text("Move filter up")
            self.down_tip.set_text("Move filter down")
            self.more_tip.set_text("More group filter actions")
        elif is_group:
            self.delete_tip.set_text("Delete group")
            self.up_tip.set_text("Move group up")
            self.down_tip.set_text("Move group down")
            self.more_tip.set_text("More group actions")
        elif is_text:
            self.delete_tip.set_text("Remove text")
            self.up_tip.set_text("Move text up")
            self.down_tip.set_text("Move text down")
            self.more_tip.set_text("More text actions")
        elif is_path:
            self.delete_tip.set_text("Remove path")
            self.up_tip.set_text("Move path up")
            self.down_tip.set_text("Move path down")
            self.more_tip.set_text("More path actions")
        elif is_gradient:
            self.delete_tip.set_text("Remove gradient")
            self.up_tip.set_text("Move gradient up")
            self.down_tip.set_text("Move gradient down")
            self.more_tip.set_text("More gradient actions")
        elif is_vector:
            self.delete_tip.set_text("Remove vector object")
            self.up_tip.set_text("Move vector object up")
            self.down_tip.set_text("Move vector object down")
            self.more_tip.set_text("More vector object actions")
        elif is_filter:
            self.delete_tip.set_text("Remove filter")
            self.up_tip.set_text("Move filter up")
            self.down_tip.set_text("Move filter down")
            self.more_tip.set_text("More filter actions")
        else:
            self.delete_tip.set_text("Delete layer")
            self.up_tip.set_text("Move layer up")
            self.down_tip.set_text("Move layer down")
            self.more_tip.set_text("More layer actions")
        if self._mask_layer is not None:
            self.update_mask_state(self._mask_layer)

    def _group_active(self):
        """True when a group header row is the current target (not a filter row)."""
        return self._selection_kind == "group" and self._selected_group is not None

    def _handle_rename(self):
        if self._group_filter_active():
            if self._on_group_filter_edit is not None:
                self._on_group_filter_edit(*self._selected_group_filter)
        elif self._group_active():
            if self._on_group_rename is not None:
                self._on_group_rename()
        elif self._text_active() and self._on_text_edit is not None:
            self._on_text_edit(*self._selected_text)
        elif self._path_active() and self._on_path_edit is not None:
            self._on_path_edit(*self._selected_path)
        elif self._gradient_active() and self._on_gradient_edit is not None:
            self._on_gradient_edit(*self._selected_gradient)
        elif self._vector_active() and self._on_vector_edit is not None:
            self._on_vector_edit(*self._selected_vector)
        elif self._filter_active() and self._on_filter_edit is not None:
            self._on_filter_edit(*self._selected_filter)
        else:
            self._on_rename()

    def delete_current(self):
        """Remove whatever row is currently selected in the list.

        A layer row deletes its layer(s), a text or path row removes that
        object, a filter row (layer or group) removes that filter, and a group
        header deletes the group. Shared by the Delete button and the Delete key.
        """
        if self._group_filter_active():
            if self._on_group_filter_remove is not None:
                self._on_group_filter_remove()
        elif self._group_active():
            if self._on_group_delete is not None:
                self._on_group_delete()
        elif self._text_active() and self._on_text_remove is not None:
            self._on_text_remove()
        elif self._path_active() and self._on_path_remove is not None:
            self._on_path_remove()
        elif self._gradient_active() and self._on_gradient_remove is not None:
            self._on_gradient_remove()
        elif self._vector_active() and self._on_vector_remove is not None:
            self._on_vector_remove()
        elif self._filter_active() and self._on_filter_remove is not None:
            self._on_filter_remove()
        else:
            self._on_delete()

    def _handle_clear(self):
        if (self._text_active() or self._path_active()
                or self._gradient_active() or self._vector_active()):
            return
        if self._group_filter_active() and self._on_group_filter_clear is not None:
            self._on_group_filter_clear()
        elif self._filter_active() and self._on_filter_clear is not None:
            self._on_filter_clear()
        else:
            self._on_clear()

    def _handle_move(self, delta):
        if self._group_filter_active():
            if self._on_group_filter_move is not None:
                self._on_group_filter_move(delta)
        elif self._group_active():
            if self._on_group_move is not None:
                self._on_group_move(delta)
        elif self._text_active() and self._on_text_move is not None:
            self._on_text_move(delta)
        elif self._path_active() and self._on_path_move is not None:
            self._on_path_move(delta)
        elif self._gradient_active() and self._on_gradient_move is not None:
            self._on_gradient_move(delta)
        elif self._vector_active() and self._on_vector_move is not None:
            self._on_vector_move(delta)
        elif self._filter_active() and self._on_filter_move is not None:
            self._on_filter_move(delta)
        else:
            self._on_move(delta)

    def _handle_move_to_canvas(self, target_index):
        if self._on_move_to_canvas is not None:
            self._on_move_to_canvas(target_index)

    def _handle_toggle(self):
        if self._group_filter_active():
            if self._on_group_filter_toggle is not None:
                self._on_group_filter_toggle()
        elif self._group_active():
            if self._on_group_toggle is not None:
                self._on_group_toggle()
        elif self._text_active():
            if self._on_text_toggle is not None:
                self._on_text_toggle()
        elif self._path_active():
            if self._on_path_toggle is not None:
                self._on_path_toggle()
        elif self._gradient_active():
            if self._on_gradient_toggle is not None:
                self._on_gradient_toggle()
        elif self._vector_active():
            return
        elif self._filter_active() and self._on_filter_toggle is not None:
            self._on_filter_toggle()
        else:
            self._on_toggle()

    def _selected_object_spec(self):
        """The spec of the selected text or path row, or ``None``."""
        if self._text_active():
            layer_index, index = self._selected_text
            attr = "texts"
        elif self._path_active():
            layer_index, index = self._selected_path
            attr = "paths"
        elif self._gradient_active():
            layer_index, index = self._selected_gradient
            attr = "gradients"
        elif self._vector_active():
            layer_index, index = self._selected_vector
            if not (0 <= layer_index < len(self._layers)):
                return None
            layer = self._layers[layer_index]
            items = (list(getattr(layer, "strokes", None) or [])
                     + list(getattr(layer, "fills", None) or []))
            if 0 <= index < len(items):
                return items[index]
            return None
        else:
            return None
        if not (0 <= layer_index < len(self._layers)):
            return None
        items = getattr(self._layers[layer_index], attr, None) or []
        if not (0 <= index < len(items)):
            return None
        return items[index]

    def _handle_duplicate(self):
        """Duplicate whatever row is selected: group, text, path or layer(s)."""
        if self._group_filter_active() or self._filter_active():
            return
        if self._group_active():
            if self._on_group_duplicate is not None:
                self._on_group_duplicate()
        elif self._text_active():
            if self._on_text_duplicate is not None:
                self._on_text_duplicate()
        elif self._path_active():
            if self._on_path_duplicate is not None:
                self._on_path_duplicate()
        elif self._gradient_active():
            if self._on_gradient_duplicate is not None:
                self._on_gradient_duplicate()
        elif self._vector_active():
            if self._on_vector_duplicate is not None:
                self._on_vector_duplicate()
        elif self._on_duplicate is not None:
            self._on_duplicate()

    def _handle_lock(self):
        self._on_lock()

    def _handle_isolate(self):
        if self._on_isolate is not None:
            self._on_isolate()

    def _handle_toggle_isolate(self):
        if self._is_isolated():
            self._handle_unisolate()
        else:
            self._handle_isolate()

    def _handle_show_all(self):
        if self._on_show_all is not None:
            self._on_show_all()

    def _handle_unisolate(self):
        if self._on_unisolate is not None:
            self._on_unisolate()

    def _handle_add_to_assets(self):
        if self._on_add_to_assets is not None:
            self._on_add_to_assets()

    def _handle_merge(self):
        self._on_merge()

    def _selected_layer_clipped(self):
        """True when every selected layer row is currently clipped."""
        indices = self.selected_indices(len(self._layers))
        if not indices:
            return False
        return all(getattr(self._layers[i], "clip", False) for i in indices)

    def _context_layer(self):
        """The layer behind the current layer, filter, text or path selection."""
        if self._filter_active():
            index = self._selected_filter[0]
        elif self._text_active():
            index = self._selected_text[0]
        elif self._path_active():
            index = self._selected_path[0]
        elif self._gradient_active():
            index = self._selected_gradient[0]
        else:
            indices = self.selected_indices(len(self._layers))
            index = indices[0] if indices else None
        if index is None or not (0 <= index < len(self._layers)):
            return None
        return self._layers[index]

    def _context_layers(self):
        """Every layer the layer actions target: the whole layer selection,
        or just the owning layer for a filter, text or path row."""
        if (self._filter_active() or self._text_active()
                or self._path_active() or self._gradient_active()):
            layer = self._context_layer()
            return [layer] if layer is not None else []
        return [
            self._layers[i] for i in self.selected_indices(len(self._layers))
            if 0 <= i < len(self._layers)
        ]

    def _layer_has_filters(self):
        """True when any target layer has filters and can be modified."""
        return any(
            l.filters and not l.is_locked() for l in self._context_layers()
        )

    def _group_has_filters(self):
        """True when the target group exists and has any filters."""
        group = self._context_group()
        return bool(group is not None and group.filters)

    def _context_group(self):
        """The group behind a group header or group filter row selection."""
        if self._group_active() or self._group_filter_active():
            return self._selected_group
        return None

    def _layer_has_texts(self):
        """True when any target layer has text objects and can be modified."""
        return any(
            l.texts and not l.is_locked() for l in self._context_layers()
        )

    def _layer_has_paths(self):
        """True when any target layer has path objects and can be modified."""
        return any(
            l.paths and not l.is_locked() for l in self._context_layers()
        )

    def _layer_has_gradients(self):
        """True when any target layer has gradient objects and can be modified."""
        return any(
            (getattr(l, "gradients", None) or []) and not l.is_locked()
            for l in self._context_layers()
        )

    def _clip_selected(self):
        if self._on_clip is not None:
            self._on_clip()

    # ------------------------------------------------------------------ menus
    def _action_menu(self, include_primary):
        """Build a menu of the actions for the current selection.

        ``include_primary`` also lists the actions already offered as buttons
        (Add/Duplicate); the overflow menu omits them, the right-click menu
        keeps them so it is a complete action list on its own.
        """
        menu = PopupMenu(self)
        if self._group_active():
            menu.add_command("Rename Group...", command=self._handle_rename)
            menu.add_command("Duplicate Group", command=self._handle_duplicate)
            menu.add_command("Show/Hide Group", command=self._handle_toggle)
            collapse_label = (
                "Expand Group" if self._selected_group.collapsed
                else "Collapse Group"
            )
            menu.add_command(collapse_label, command=lambda: self._toggle_collapse())
            menu.add_separator()
            menu.add_command(
                "Clear Group Filters", command=self._on_group_filter_clear,
                enabled=self._group_has_filters,
            )
            menu.add_separator()
            menu.add_command("Ungroup", command=self._on_group_ungroup)
            menu.add_command("Flatten Group", command=self._on_group_flatten)
            menu.add_separator()
            menu.add_command("Move Group Up", command=lambda: self._handle_move(1))
            menu.add_command("Move Group Down",
                             command=lambda: self._handle_move(-1))
            menu.add_separator()
            menu.add_command("Delete Group", command=self.delete_current)
            return menu
        if self._group_filter_active():
            menu.add_command("Edit Filter...", command=self._handle_rename)
            menu.add_command("Toggle Filter", command=self._handle_toggle)
            menu.add_command("Clear Group Filters", command=self._handle_clear)
            menu.add_separator()
            menu.add_command(
                "Apply Filters (Flatten Group)",
                command=self._on_group_filter_apply,
                enabled=self._group_has_filters,
            )
            menu.add_separator()
            menu.add_command("Remove Filter", command=self.delete_current)
            return menu
        if self._text_active():
            menu.add_command("Edit Text...", command=self._handle_rename)
            menu.add_command("Duplicate Text", command=self._handle_duplicate)
            menu.add_command("Show/Hide Text", command=self._handle_toggle)
            menu.add_command(
                "Rasterize Text", command=self._on_rasterize_text,
                enabled=self._layer_has_texts,
            )
            menu.add_separator()
            menu.add_command("Move Text Up", command=lambda: self._handle_move(1))
            menu.add_command("Move Text Down",
                             command=lambda: self._handle_move(-1))
            menu.add_separator()
            menu.add_command("Remove Text", command=self.delete_current)
            return menu
        if self._path_active():
            menu.add_command("Edit Path", command=self._handle_rename)
            menu.add_command("Duplicate Path", command=self._handle_duplicate)
            menu.add_command("Show/Hide Path", command=self._handle_toggle)
            menu.add_command(
                "Rasterize Path", command=self._on_rasterize_path,
                enabled=self._layer_has_paths,
            )
            menu.add_separator()
            menu.add_command("Move Path Up", command=lambda: self._handle_move(1))
            menu.add_command("Move Path Down",
                             command=lambda: self._handle_move(-1))
            menu.add_separator()
            menu.add_command("Remove Path", command=self.delete_current)
            return menu
        if self._gradient_active():
            menu.add_command("Edit Gradient", command=self._handle_rename)
            menu.add_command("Duplicate Gradient",
                             command=self._handle_duplicate)
            menu.add_command("Show/Hide Gradient",
                             command=self._handle_toggle)
            menu.add_command(
                "Rasterize Gradient", command=self._on_rasterize_gradient,
                enabled=self._layer_has_gradients,
            )
            menu.add_separator()
            menu.add_command("Move Gradient Up",
                             command=lambda: self._handle_move(1))
            menu.add_command("Move Gradient Down",
                             command=lambda: self._handle_move(-1))
            menu.add_separator()
            menu.add_command("Remove Gradient", command=self.delete_current)
            return menu
        if self._vector_active():
            menu.add_command("Edit Vector Object", command=self._handle_rename)
            menu.add_command("Duplicate Vector Object",
                             command=self._handle_duplicate)
            menu.add_separator()
            menu.add_command("Move Vector Object Up",
                             command=lambda: self._handle_move(1))
            menu.add_command("Move Vector Object Down",
                             command=lambda: self._handle_move(-1))
            menu.add_separator()
            menu.add_command("Remove Vector Object", command=self.delete_current)
            return menu
        if self._filter_active():
            menu.add_command("Edit Filter...", command=self._handle_rename)
            menu.add_command("Toggle Filter", command=self._handle_toggle)
            menu.add_command("Clear Filters", command=self._handle_clear)
            menu.add_separator()
            menu.add_command(
                "Apply Filter", command=self._on_apply_filter,
                enabled=self._layer_has_filters,
            )
            menu.add_separator()
            menu.add_command("Remove Filter", command=self.delete_current)
            return menu
        # Rename and Select Master target one layer, so they are muted while
        # several layers are selected; every other layer action applies to
        # the whole selection.
        single = len(self.selected_indices(len(self._layers))) <= 1
        # Layer actions
        menu.add_command(
            "Rename Layer...", command=self._handle_rename, enabled=single
        )
        if include_primary:
            menu.add_command("Add Layer", command=self._on_add)
            menu.add_command("Add Vector Layer", command=self._on_add_vector)
            menu.add_command("Duplicate Layer", command=self._on_duplicate)
            menu.add_command(
                "Merge Layers", command=self._handle_merge, enabled=not single
            )
            menu.add_command("Delete Layer", command=self.delete_current)
        menu.add_separator()
        # Visibility / state actions
        menu.add_command("Show/Hide Layer", command=self._handle_toggle)
        collapsed = all(
            getattr(layer, "collapsed", False) for layer in self._context_layers()
        )
        menu.add_command(
            "Expand Layer" if collapsed else "Collapse Layer",
            command=self._toggle_layer_collapse,
        )
        menu.add_command("Lock/Unlock Layer", command=self._handle_lock)
        menu.add_command(
            "Isolate/Unisolate Layer", command=self._handle_toggle_isolate,
            enabled=bool(self._layers),
        )
        menu.add_command("Show All Layers", command=self._handle_show_all)
        menu.add_separator()
        # Content actions
        clip_label = (
            "Unclip Layer" if self._selected_layer_clipped()
            else "Clip to Layer Below"
        )
        menu.add_command(clip_label, command=self._clip_selected)
        menu.add_command("Clear Layer", command=self._handle_clear)
        menu.add_command("Add to Assets", command=self._handle_add_to_assets)
        menu.add_separator()
        # Link Layer actions
        self._add_link_menu(menu, single)
        # Effects / rasterize actions
        menu.add_command(
            "Apply Filters", command=self._on_apply_filters,
            enabled=self._layer_has_filters,
        )
        menu.add_command(
            "Rasterize Texts", command=self._on_rasterize_texts,
            enabled=self._layer_has_texts,
        )
        menu.add_command(
            "Rasterize Paths", command=self._on_rasterize_paths,
            enabled=self._layer_has_paths,
        )
        menu.add_command(
            "Rasterize Gradients", command=self._on_rasterize_gradients,
            enabled=self._layer_has_gradients,
        )
        menu.add_separator()
        # Arrange actions
        menu.add_command("Move Up", command=lambda: self._handle_move(1))
        menu.add_command("Move Down", command=lambda: self._handle_move(-1))
        choices = self._get_canvas_choices() if self._get_canvas_choices else []
        move_canvas_menu = PopupMenu(menu)
        for label, index in choices:
            move_canvas_menu.add_command(
                label,
                command=lambda i=index: self._handle_move_to_canvas(i),
            )
        menu.add_cascade("Move to Canvas", move_canvas_menu, enabled=bool(choices))
        menu.add_separator()
        # Group actions
        menu.add_command("Add Group", command=self._on_group_add)
        menu.add_command("Group Selected Layers", command=self._on_group_create)
        if self._groups:
            move_menu = PopupMenu(menu)
            for group in self._groups:
                move_menu.add_command(
                    group.name,
                    command=lambda g=group: self._on_group_assign(g),
                )
            menu.add_cascade("Move to Group", move_menu)
        return menu

    def _add_link_menu(self, menu, single=True):
        """Add the Link Layer actions for the current layer selection.

        ``single`` is False while several layers are selected: Convert, Remove
        Master, Unlink and Reset Transform then apply to every eligible
        selected layer, and Select Master (one destination) is muted.
        """
        layer = self._context_layer()
        targets = [layer] if single else self._context_layers()
        any_link = any(getattr(l, "is_link", False) for l in targets)
        any_master = any(getattr(l, "is_master", False) for l in targets)
        any_plain = any(
            not getattr(l, "is_link", False) and not getattr(l, "is_master", False)
            for l in targets if l is not None
        )
        any_transform = any(
            getattr(l, "is_link", False) and getattr(l, "transform", None) is not None
            for l in targets
        )
        is_link = bool(
            single and layer is not None and getattr(layer, "is_link", False)
        )
        menu.add_command(
            "Convert to Master Layer", command=self._handle_master_convert,
            enabled=any_plain,
        )
        menu.add_command(
            "Remove Master Layer", command=self._handle_master_remove,
            enabled=any_master,
        )
        choices = self._get_master_choices() if self._get_master_choices else []
        if choices:
            link_menu = PopupMenu(menu)
            for label, master in choices:
                link_menu.add_command(
                    label, command=lambda m=master: self._handle_link_create(m)
                )
            menu.add_cascade("Create Link Layer", link_menu, enabled=True)
        else:
            menu.add_command(
                "Create Link Layer", command=lambda: None, enabled=False
            )
        menu.add_command(
            "Unlink Layer", command=self._handle_unlink, enabled=any_link
        )
        menu.add_command(
            "Select Master", command=self._handle_select_master, enabled=is_link
        )
        menu.add_command(
            "Reset Link Transform", command=self._handle_link_reset_transform,
            enabled=any_transform,
        )
        menu.add_separator()

    def _handle_master_convert(self):
        if self._on_master_convert is not None:
            self._on_master_convert()

    def _handle_master_remove(self):
        if self._on_master_remove is not None:
            self._on_master_remove()

    def _handle_link_create(self, master):
        if self._on_link_create is not None:
            self._on_link_create(master)

    def _handle_unlink(self):
        if self._on_unlink is not None:
            self._on_unlink()

    def _handle_select_master(self):
        if self._on_select_master is not None:
            self._on_select_master()

    def _handle_link_reset_transform(self):
        if self._on_link_reset_transform is not None:
            self._on_link_reset_transform()

    def _post_overflow(self):
        """Drop the secondary layer/filter actions under the "more" button."""
        self._action_menu(include_primary=False).post(
            self.more_btn.winfo_rootx(),
            self.more_btn.winfo_rooty() + self.more_btn.winfo_height(),
        )

    def _post_mask_menu(self):
        """Drop the two mask-creation actions under the "Set" button."""
        menu = PopupMenu(self)
        menu.add_command(
            "Reveal Mask", command=lambda: self._on_add_mask(False)
        )
        menu.add_command(
            "Hide Mask", command=lambda: self._on_add_mask(True)
        )
        menu.post(
            self.mask_add_btn.winfo_rootx(),
            self.mask_add_btn.winfo_rooty() + self.mask_add_btn.winfo_height(),
        )

    def _on_context(self, event, row):
        """Right-click: target the row under the pointer, then list actions."""
        self._press = None
        self._drag_row = None
        if 0 <= row < len(self._rows):
            if row not in self._selected:
                self._anchor = row
                self._selected = [row]
                self._apply_selection_colors()
            self._dispatch(row)
        self._action_menu(include_primary=True).post(event.x_root, event.y_root)
        return "break"

    # ----------------------------------------------------------- reorder drag
    def _on_motion(self, event):
        """Turn a press into a reorder drag once the pointer has moved, then drag.

        A press that never moves stays a plain click; only a layer row or a
        group header can be dragged (a filter, text or group-filter row clears
        the press), and the ghost is only shown once the pointer has passed
        :data:`DRAG_SLOP`.
        """
        press = self._press
        if press is None:
            return
        if self._drag_row is None:
            dx = event.x_root - press["x_root"]
            dy = event.y_root - press["y_root"]
            if dx * dx + dy * dy < DRAG_SLOP * DRAG_SLOP:
                return
            row = press["row"]
            if not (0 <= row < len(self._rows)) \
                    or self._rows[row][0] not in ("layer", "group"):
                self._press = None
                return
            if not self._begin_ghost(row, event.x_root, event.y_root):
                self._press = None
                return
            self._drag_row = row
        self._move_ghost(event.x_root, event.y_root)
        self._show_drop(event.x_root, event.y_root)

    def _on_release(self, event):
        """Finish a reorder drag: drop the ghost and offer the move to the app."""
        press = self._press
        drag_row = self._drag_row
        self._press = None
        self._drag_row = None
        self._end_ghost()
        self._clear_drop_line()
        if drag_row is None or press is None or self._on_reorder is None:
            return
        drop = self._drop_at(event.x_root, event.y_root)
        if drop is None:
            return
        source = self._rows[press["row"]]
        target, above, _row = drop
        if source != target:
            self._on_reorder(source, target, above)

    def _show_drop(self, x_root, y_root):
        """Place the insertion marker at the boundary a drop would use."""
        if self._drag_row is None:
            self._clear_drop_line()
            return
        drop = self._drop_at(x_root, y_root)
        source = self._rows[self._drag_row]
        if drop is None or drop[0] == source:
            self._clear_drop_line()
            return
        _target, above, row = drop
        row_widget = self._row_widgets[row]["row"]
        try:
            top = row_widget.winfo_y()
            height = row_widget.winfo_height()
        except tk.TclError:
            self._clear_drop_line()
            return
        self._show_drop_line(top if above else top + height)

    def _drop_at(self, x_root, y_root):
        """The ``(target, above, row)`` a reorder drop at the pointer would use.

        ``None`` unless the pointer is in the list's column: only layer rows and
        group headers are reorder targets, so a filter, text or group-filter row
        yields nothing. The pointer need not sit exactly over a layer row,
        though. ``target`` is the row's entry - ``("layer", index)`` or
        ``("group", group)`` - and ``above`` is whether the drop lands above or
        below the target row: the pointer's position within a row decides it,
        while the room above the first row (a group header on top) or below the
        last (empty space under the list) snaps to the start or the end of the
        stack, so the marker still shows there. Rows are matched by their screen
        box rather than by ``winfo_containing`` so the drag ghost under the
        cursor never hides the row being pointed at.
        """
        try:
            left = self.canvas.winfo_rootx()
            top = self.canvas.winfo_rooty()
            right = left + self.canvas.winfo_width()
            bottom = top + self.canvas.winfo_height()
        except tk.TclError:
            return None
        if not (left <= x_root <= right and top <= y_root <= bottom):
            return None
        first_top = None
        last_bottom = None
        for row, entry in enumerate(self._rows):
            if entry[0] not in ("layer", "group") \
                    or row >= len(self._row_widgets):
                continue
            row_widget = self._row_widgets[row]["row"]
            try:
                row_top = row_widget.winfo_rooty()
                height = row_widget.winfo_height()
            except tk.TclError:
                continue
            if first_top is None:
                first_top = row_top
            last_bottom = row_top + height
            if row_top <= y_root < row_top + height:
                above = (y_root - row_top) < height / 2.0
                return entry, above, row
        if first_top is None:
            return None
        if y_root < first_top:
            return self._edge_drop(start=True)
        if y_root >= last_bottom:
            return self._edge_drop(start=False)
        return None

    def _edge_drop(self, start):
        """The ``(target, above, row)`` for a drop at the start or end of the stack.

        ``start`` targets the first draggable row (dropped above it), otherwise
        the last (dropped below it); ``None`` when there is no such row.
        """
        rows = [
            row for row, entry in enumerate(self._rows)
            if entry[0] in ("layer", "group") and row < len(self._row_widgets)
        ]
        if not rows:
            return None
        row = rows[0] if start else rows[-1]
        return self._rows[row], start, row

    def _show_drop_line(self, y):
        """Place the accent insertion marker at ``y`` in the row column.

        The marker is a child of the row column, which is only as tall as its
        rows, so the boundary below the last row lands exactly on the frame's
        edge; clamping keeps the marker inside it, where it is not clipped away
        and actually paints at the start or end of the stack.
        """
        line = self._drop_line
        if line is None or not self._line_alive(line):
            line = tk.Frame(self.inner, height=2, bg=theme.color("accent"))
            self._drop_line = line
        try:
            height = int(line.winfo_reqheight()) or 2
            limit = max(0, self.inner.winfo_height() - height)
            line.place(x=0, y=max(0, min(int(y), limit)), relwidth=1)
            line.lift()
        except tk.TclError:
            pass

    def _clear_drop_line(self):
        line = self._drop_line
        if line is None:
            return
        try:
            line.place_forget()
        except tk.TclError:
            pass

    @staticmethod
    def _line_alive(line):
        """True while a previously placed marker widget still exists."""
        try:
            return bool(line.winfo_exists())
        except tk.TclError:
            return False

    def _begin_ghost(self, row, x_root, y_root):
        """Show a washed-out snapshot of the dragged row following the cursor.

        Tkinter cannot give an individual widget an alpha, so (as the Assets
        panel does) the layer's thumbnail is blended toward the panel background
        as a stand-in for opacity. It lives in a borderless top-level window so
        the OS moves it atomically without smearing the rows beneath it. A group
        header is represented by its topmost member's thumbnail, or by its label
        when the group has no members yet. Returns ``True`` when it is shown.
        """
        self._end_ghost()
        entry = self._rows[row]
        if entry[0] == "group":
            layer = self._group_preview_layer(entry[1])
            label = self._group_row_text(entry[1])
        else:
            layer = self._layer_at(entry[1])
            label = layer.name if layer is not None else "Layer"
        try:
            if layer is not None:
                composited = getattr(layer, "composited_image", None)
                source = composited() if composited is not None else layer.image
                source = source.convert("RGBA")
                source.thumbnail((GHOST, GHOST), Image.LANCZOS)
                board = checkerboard(source.size)
                board.alpha_composite(source)
                washed = Image.blend(
                    board.convert("RGB"),
                    Image.new("RGB", board.size, theme.color("ghost_blend")),
                    0.4,
                )
                photo = ImageTk.PhotoImage(washed)
                window = self._ghost_window()
                overlay = tk.Label(
                    window, image=photo, bd=0, highlightthickness=1,
                    highlightbackground=theme.color("float_border"),
                )
                overlay.pack()
                width, height = washed.width, washed.height
            else:
                # A group with no members has no pixels to show, so carry its
                # header text instead of an image.
                window = self._ghost_window()
                overlay = tk.Label(
                    window, text=label, bd=0, padx=6, pady=3,
                    bg=theme.color("listbox_bg"), fg=theme.color("text"),
                    highlightthickness=1,
                    highlightbackground=theme.color("float_border"),
                )
                overlay.pack()
                window.update_idletasks()
                photo = None
                width = overlay.winfo_reqwidth()
                height = overlay.winfo_reqheight()
        except (tk.TclError, ValueError):
            return False
        self._ghost = {
            "widget": window, "photo": photo,
            "width": width, "height": height,
        }
        self._move_ghost(x_root, y_root)
        return True

    def _ghost_window(self):
        """A borderless, top-most toplevel for a drag ghost."""
        window = tk.Toplevel(self)
        window.overrideredirect(True)
        try:
            window.attributes("-topmost", True)
        except tk.TclError:
            pass
        return window

    def _group_preview_layer(self, group):
        """The topmost member of ``group``, or ``None`` when it has none."""
        for index in range(len(self._layers) - 1, -1, -1):
            if getattr(self._layers[index], "group", None) is group:
                return self._layers[index]
        return None

    def _move_ghost(self, x_root, y_root):
        ghost = self._ghost
        if ghost is None:
            return
        x = x_root - ghost["width"] // 2
        y = y_root - ghost["height"] // 2
        try:
            ghost["widget"].geometry(f"+{int(x)}+{int(y)}")
        except tk.TclError:
            pass

    def _end_ghost(self):
        ghost = self._ghost
        if ghost is not None:
            try:
                ghost["widget"].destroy()
            except tk.TclError:
                pass
            self._ghost = None

    def _toggle_collapse(self):
        if self._on_group_collapse is not None:
            self._on_group_collapse()

    def _toggle_layer_collapse(self):
        if self._on_layer_collapse is not None:
            self._on_layer_collapse()

    def _handle_opacity(self, _value=None):
        self._on_opacity(self.opacity_var.get())

    def _handle_blend(self, _event=None):
        if self._on_blend is None:
            return
        label = self.blend_var.get()
        if document.is_blend_divider(label):
            # Divider rows are display-only; selecting one must not change the
            # layer. Restore the label of the layer's current blend mode.
            active = (self._layers[self._active]
                      if self._layers and self._active < len(self._layers)
                      else None)
            blend = getattr(active, "blend", "normal") if active else "normal"
            self.blend_var.set(document.BLEND_LABELS.get(blend, "Normal"))
            return
        for mode, mode_label in document.BLEND_MODES:
            if mode_label == label:
                self._on_blend(mode)
                return

    def _blend_popdown(self):
        """The blend combobox's drop-down path, created on first use.

        ``ttk`` builds the pop-down itself, so it is not a tkinter widget; it
        is addressed by its Tcl path. The pop-down is hidden until the picker
        is first posted and its rows are rebuilt every time it opens, so the
        recoloring is driven by a ``<Map>`` binding set up here. This mirrors
        the model picker in ``generation``.
        """
        exists = self._blend_popdown_path is not None and self.blend_menu.tk.call(
            "winfo", "exists", self._blend_popdown_path
        )
        if not exists:
            try:
                self._blend_popdown_path = self.blend_menu.tk.call(
                    "ttk::combobox::PopdownWindow", self.blend_menu
                )
            except tk.TclError:
                self._blend_popdown_path = None
                return None
            self.blend_menu.tk.call(
                "bind", self._blend_popdown_path, "<Map>", "+"
                + self.blend_menu.register(self._recolor_blend_rows)
            )
        return self._blend_popdown_path

    def _blend_listbox(self):
        """The listbox Tcl path inside the pop-down, or ``None``."""
        popdown = self._blend_popdown()
        return self._find_listbox(popdown) if popdown else None

    def _find_listbox(self, path):
        tkcmd = self.blend_menu.tk
        children = tkcmd.splitlist(tkcmd.call("winfo", "children", path))
        for child in children:
            if tkcmd.call("winfo", "class", child) == "Listbox":
                return child
            found = self._find_listbox(child)
            if found is not None:
                return found
        return None

    def _recolor_blend_rows(self, _event=None):
        """Tint the group header rows so they stand apart from the modes.

        The dashes stay; a distinct foreground and background make the headers
        read as section labels even though a Tk listbox cannot bold a single
        row. Mirrors the model picker in ``generation``.
        """
        listbox = self._blend_listbox()
        if listbox is None:
            return
        tkcmd = self.blend_menu.tk
        header = (theme.color("accent"), theme.color("stripe_bg"))
        row = (theme.color("text"), theme.color("listbox_bg"))
        count = int(tkcmd.call(listbox, "size"))
        for index, value in enumerate(self.blend_menu.cget("values")):
            if index >= count:
                break
            fg, bg = header if value in self._blend_headers else row
            tkcmd.call(
                listbox, "itemconfigure", index,
                "-foreground", fg, "-background", bg,
            )

    def _handle_mask_edit(self):
        self._on_mask_edit(self.mask_edit_var.get())

    def _handle_outline(self):
        if self._on_outline is not None:
            self._on_outline(self.outline_var.get())

    # ---------------------------------------------------------------- scrolling
    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_ctrl_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    def _bind_wheel(self, widget):
        """Let Ctrl+wheel scroll the list from anywhere over it.

        A plain wheel is left alone, so it scrolls the sidebar rail as usual
        (the rail owns that policy for its lists). A canvas only receives a
        binding over itself, not over the row widgets embedded in it, so each
        row (and every child) gets the Ctrl binding too.
        """
        widget.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)

    def _on_destroy(self, event):
        """Stop a pending preview rebuild when the panel itself goes away.

        ``Destroy`` also reaches the parent from each row torn down by a
        rebuild, so only the panel's own destruction matters here.
        """
        if event.widget is self:
            self._cancel_preview_job()
            self._end_ghost()
            self._clear_drop_line()

    def _on_theme(self):
        """Rebuild the rows so their colours follow a theme change."""
        bg = theme.color("listbox_bg")
        try:
            self.canvas.config(bg=bg)
            self.inner.config(bg=bg)
        except tk.TclError:
            return
        self.refresh(
            self._layers,
            self._active,
            self.selected_indices(len(self._layers)),
            filter_selection=self._selected_filter,
            text_selection=self._selected_text,
            path_selection=self._selected_path,
            gradient_selection=self._selected_gradient,
            group_selection=(
                self._selected_group if self._selection_kind == "group" else None
            ),
            groups=self._groups,
            group_filter_selection=self._selected_group_filter,
        )
