"""Tool selector with per-tool settings and the controller that drives events."""

import math
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageTk

from . import (
    antialias, brush, dialogs, fill, fontpicker, fonts, gradient,
    gradienteditor, icons, poses, shape, symmetry, textrender, theme, vector,
    warp,
)
from .constants import (
    ALT_STATE_MASK, BRUSH_SLIDER_MAX, MAX_BRUSH_SIZE, PATH_OUTLINE_MAX, rgba,
    TOOL_FULL_MIN_W, TOOL_ICON2_MIN_W,
)
from .gizmo import SelectionGizmo, TransformGizmo
from .layout_to_image import (
    LayoutEditor, LayoutModel, SUPPORTED_FAMILIES as LAYOUT_FAMILIES,
)
from . import path as pathspec
from .path import Path
from .scrollbar import ArrowScrollbar
from .selection import Selection
from .ui import PopupMenu, Tooltip, build_menus


#: Longest edge of a pose list preview thumbnail, in pixels.
POSES_PREVIEW_H = 44
#: Tallest the scrollable pose list grows before it scrolls, in pixels.
POSES_LIST_MAX_H = 150
#: The Poses scene zoom slider's range, as a percentage of the default zoom.
POSE_SCENE_ZOOM_MIN = 10
POSE_SCENE_ZOOM_MAX = 1000

#: Pointer speed (document pixels per millisecond) that maps to full "Size
#: speed" dynamics; faster than this clamps to the strongest modulation. Used
#: together with the real motion-event timing so the effect does not depend on
#: how often events are delivered.
PAINT_SPEED_FULL = 2.5

#: Stabiliser styles. ``simple`` is the original exponential smoothing;
#: ``weighted`` averages the last few samples (a longer, steadier line);
#: ``velocity`` eases off as the pointer speeds up so it never lags a fast flick.
STABILIZE_MODES = {
    "simple": "Simple",
    "weighted": "Weighted",
    "velocity": "Velocity",
}

#: How many recent pointer samples the weighted stabiliser keeps.
STABILIZE_HISTORY = 40

#: The brush settings preview strip at its base size, in pixels, and the widest
#: stroke shown on it (wider brushes are clipped to the strip's height). The
#: strip scales with its container: :func:`brush_preview_image` keeps this
#: aspect ratio and scales the stroke, margins and arc to match.
BRUSH_PREVIEW_W = 224
BRUSH_PREVIEW_H = 56
BRUSH_PREVIEW_MAX = 46


def brush_preview_image(size, hardness, flow, opacity, spacing, tip, pattern,
                        dynamics, *, kind="brush", color=(40, 40, 40),
                        fill=(150, 170, 200), background=(240, 240, 240),
                        grain="none", grain_scale=1.0, grain_strength=0.0,
                        grain_mode="canvas", jitter_hue=0.0, jitter_sat=0.0,
                        jitter_val=0.0, pickup=0.0, wet_edge=0.0,
                        width=None, height=None):
    """Render a short sample stroke for the settings panel's preview strip.

    The stroke sweeps left-to-right along a shallow arc, starting slow, speeding
    up mid-way and slowing again, so size/opacity dynamics, hardness, tip,
    pattern and both tapers all show in one glance. An ``"eraser"`` preview
    fills the strip first and erases through it; otherwise the stroke is painted
    in ``color`` over ``background``.

    ``width`` sizes the strip (``height`` defaults to the base aspect ratio).
    Everything drawn - the stroke, its margins, the arc and the background -
    scales with it so the sample keeps the same proportions as the panel widens.
    """
    from .document import Layer
    if width is None:
        width = BRUSH_PREVIEW_W
    width = max(1, int(width))
    if height is None:
        height = max(1, round(width * BRUSH_PREVIEW_H / BRUSH_PREVIEW_W))
    height = max(1, int(height))
    scale = width / float(BRUSH_PREVIEW_W)
    layer = Layer("preview", (width, height), (0, 0, 0, 0))
    if kind in ("eraser", "eraser-color"):
        layer.image.paste((fill[0], fill[1], fill[2], 255), (0, 0, width, height))
    stroke = brush.Stroke(
        layer, kind, size=max(1.0, min(size, BRUSH_PREVIEW_MAX) * scale),
        hardness=hardness, flow=flow, opacity=opacity, spacing=spacing,
        antialias=True, tip=tip, pattern=pattern, dynamics=dynamics,
        color=color, grain=grain, grain_scale=grain_scale,
        grain_strength=grain_strength, grain_mode=grain_mode,
        jitter_hue=jitter_hue, jitter_sat=jitter_sat, jitter_val=jitter_val,
        pickup=pickup, wet_edge=wet_edge,
    )
    margin = 12 * scale
    x0, x1 = margin, width - margin
    cy = height / 2.0
    amp = min(9.0 * scale, height / 4.0)
    steps = 24
    stroke.begin(x0, cy)
    for i in range(1, steps + 1):
        t = i / steps
        stroke.move_to(x0 + (x1 - x0) * t, cy + amp * math.sin(t * math.pi * 1.3),
                       speed=math.sin(t * math.pi))
    stroke.finish()
    board = Image.new("RGBA", (width, height), (*background, 255))
    board.alpha_composite(layer.image)
    return board


class BrushPreview(tk.Canvas):
    """A strip drawing a sample stroke for the active tool's settings.

    The strip scales with its container: on resize ``render(width, height)`` is
    asked for a fresh image at the new size (keeping the base aspect ratio) so
    the sample stays crisp instead of being stretched.
    """

    def __init__(self, parent, render=None):
        super().__init__(parent, width=BRUSH_PREVIEW_W, height=BRUSH_PREVIEW_H,
                         highlightthickness=1, bd=0)
        self._photo = None
        self._render = render
        self._size = (BRUSH_PREVIEW_W, BRUSH_PREVIEW_H)
        self.bind("<Configure>", self._on_configure)

    def refresh(self):
        """Re-render the sample at the strip's current width."""
        if self._render is None:
            return
        width = self.winfo_width()
        if width <= 1:
            width = BRUSH_PREVIEW_W
        height = max(1, round(width * BRUSH_PREVIEW_H / BRUSH_PREVIEW_W))
        self._size = (width, height)
        if self.cget("height") != height:
            self.configure(height=height)
        image = self._render(width, height)
        self._photo = ImageTk.PhotoImage(image)
        self.delete("all")
        self.create_image(0, 0, anchor="nw", image=self._photo)

    def _on_configure(self, event):
        if event.width != self._size[0]:
            self.refresh()


def scene_zoom_percent(scene):
    """A pose scene's zoom as an integer percentage of the default."""
    return int(round(float(scene.get("zoom", poses.DEFAULT_ZOOM))
                     / poses.DEFAULT_ZOOM * 100.0))




class ToolsPanel(tk.Frame):
    """The tool selector plus a settings panel for the active tool.

    Exactly one tool is always active - Brush is an explicit tool rather than
    the implicit default. ``on_mode_change`` is called whenever the active tool
    changes so the coordinator can update the cursor and status text, and the
    settings panel is rebuilt to show the options for the newly selected tool.

    The settings live in ``settings_parent``, the content frame of its own
    sidebar container, so the options get a real panel header (collapse, float,
    dock, drag) instead of a titled border inside the tool selector. When no
    parent is given a :class:`tk.LabelFrame` is built in place, keeping the
    panel usable on its own.
    """

    # Active-tool key -> boolean flag on this panel.
    TOOL_FLAGS = {
        "arrow": "arrow_tool",
        "brush": "brush_tool",
        "pick": "eyedropper",
        "eraser": "eraser",
        "fill": "fill_tool",
        "gradient": "gradient_tool",
        "shape": "shape_tool",
        "clone": "clone_tool",
        "smudge": "smudge_tool",
        "dodge": "dodge_tool",
        "burn": "burn_tool",
        "liquify": "liquify_tool",
        "warp": "warp_tool",
        "puppet": "puppet_tool",
        "select": "select_tool",
        "layout": "layout_tool",
        "paste": "paste_tool",
        "crop": "crop_tool",
        "path": "path_tool",
        "vector": "vector_tool",
        "poses": "poses_tool",
        "move": "move_tool",
        "drag": "drag_tool",
        "transform": "transform_tool",
        "text": "text_tool",
        "mask": "mask_edit",
        "inpaint_mask": "inpaint_mask_edit",
    }

    # Tool buttons in display order: (key, label). Ordered by expected use,
    # most-used first, with Arrow pinned to the top slot.
    TOOL_BUTTONS = [
        ("arrow", "Arrow"),
        ("brush", "Brush"),
        ("eraser", "Erase"),
        ("fill", "Fill"),
        ("pick", "Pick"),
        ("select", "Select"),
        ("move", "Move"),
        ("text", "Text"),
        ("gradient", "Gradient"),
        ("shape", "Shape"),
        ("clone", "Clone"),
        ("crop", "Crop"),
        ("transform", "Transform"),
        ("smudge", "Smudge"),
        ("dodge", "Dodge"),
        ("burn", "Burn"),
        ("liquify", "Liquify"),
        ("warp", "Warp"),
        ("puppet", "Puppet Warp"),
        ("path", "Path"),
        ("vector", "Vector"),
        ("poses", "3D"),
        ("mask", "Mask"),
        ("paste", "Paste"),
        ("drag", "Drag"),
    ]

    # Buttons per row in the tool grid. Two, not four, because each button now
    # shows its icon *and* its label, and four label-width columns overflow the
    # sidebar rail.
    TOOL_COLUMNS = 2

    # The Tabler icon drawn on each tool button (see paintapp.icons). The label
    # from :data:`TOOL_BUTTONS` becomes the tooltip and the fallback text when
    # the icon rasteriser is unavailable.
    TOOL_ICONS = {
        "arrow": "pointer",
        "brush": "brush",
        "eraser": "eraser",
        "fill": "bucket",
        "pick": "color-picker",
        "select": "marquee",
        "move": "arrows-move",
        "text": "typography",
        "gradient": "color-filter",
        "shape": "shape",
        "clone": "copy",
        "crop": "crop",
        "transform": "transform",
        "smudge": "blur",
        "dodge": "sun-high",
        "burn": "flame",
        "liquify": "ripple",
        "warp": "wave-sine",
        "puppet": "bone",
        "path": "vector-bezier",
        "vector": "vector",
        "poses": "cube",
        "mask": "mask",
        "paste": "clipboard",
        "drag": "drag-drop",
    }

    # Size and transparent margin of the tool icons. The margin is baked into
    # the icon because Tk ignores a button's padx/pady when it shows an image.
    TOOL_ICON = 18
    TOOL_PAD = 3

    # The Select tool's modes, chosen from its settings panel.
    SELECT_MODES = [
        ("rect", "Rectangle"),
        ("lasso", "Lasso"),
        ("wand", "Magic Wand"),
        ("segment", "Segment AI"),
    ]

    def __init__(self, parent, on_mode_change, on_brush_size,
                 on_mask_edit, get_text_style=None, on_text_style=None,
                 on_mask_action=None, on_mask_overlay=None,
                 on_path_action=None,                  on_crop_action=None, on_crop_reset=None,
                 on_clone_clear=None, settings_parent=None, layout_parent=None,
                 on_path_settings=None,
                 on_paste=None, on_pick_color=None, on_select_radius=None,
                 get_color=None, on_warp_action=None, on_puppet_action=None,
                 on_poses_action=None, on_vector_action=None,
                 on_vector_settings=None, on_layout_change=None,
                 get_layout_family=None, on_layout_capture=None):
        super().__init__(parent)
        self._on_mode_change = on_mode_change
        self._on_brush_size = on_brush_size
        self._on_mask_edit = on_mask_edit
        self._get_text_style = get_text_style
        self._on_text_style = on_text_style
        self._on_mask_action = on_mask_action
        self._on_mask_overlay = on_mask_overlay
        self._on_path_action = on_path_action
        self._on_path_settings = on_path_settings
        self._on_crop_action = on_crop_action
        self._on_crop_reset = on_crop_reset
        self._on_warp_action = on_warp_action
        self._on_puppet_action = on_puppet_action
        self._on_poses_action = on_poses_action
        self._on_vector_action = on_vector_action
        self._on_vector_settings = on_vector_settings
        self._on_layout_change = on_layout_change
        self._get_layout_family = get_layout_family
        self._on_layout_capture = on_layout_capture
        # The MOSS layout rows live here so they survive the Tool Settings panel
        # being rebuilt on every tool change.
        self.layout_model = LayoutModel()
        self._on_clone_clear = on_clone_clear
        self._on_paste = on_paste
        self._on_pick_color = on_pick_color
        self._on_select_radius = on_select_radius
        self._get_color = get_color
        self._settings_parent = settings_parent
        # The Layout Control editor lives in its own sidebar container rather
        # than the Tool Settings panel, so it is built and refreshed there.
        self._layout_parent = layout_parent
        self._layout_editor = None
        self._layout_family = None
        self.brush_size = 40
        # Brush/eraser/mask settings (see ``brush.Stroke`` for how they map to
        # pixels). They live on the panel so the controller can read them and
        # project save/load can round-trip them through export_settings().
        self.brush_opacity = 100
        self.brush_hardness = 100
        self.brush_flow = 100
        self.brush_blend = "Normal"
        self.brush_spacing = 25
        self.brush_antialias = True
        self.brush_stabilize = 0
        self.brush_stabilize_mode = "simple"
        self.brush_tip = "Round"
        self.brush_pattern = "none"
        # Velocity dynamics (see ``brush.Dynamics``): how strongly faster pointer
        # movement changes each attribute (0-100).
        self.brush_velocity_size = 25
        self.brush_velocity_opacity = 0
        self.brush_velocity_flow = 0
        self.brush_velocity_hardness = 0
        self.brush_velocity_spacing = 0
        # Taper lengths as a percentage of the brush size (0-100), for the start
        # and end of a stroke.
        self.brush_taper_in = 0
        self.brush_taper_out = 0
        # Dry-media grain (see ``brush.GRAIN_TYPES``): type name, texture scale,
        # strength 0-100, and whether it is anchored to the canvas or the dab.
        self.brush_grain = "none"
        self.brush_grain_scale = 100
        self.brush_grain_strength = 0
        self.brush_grain_mode = "canvas"
        # Colour dynamics: per-dab hue/sat/value variation (0-100) and how much
        # the colour under the brush is mixed into what is deposited (0-100).
        self.brush_jitter_hue = 0
        self.brush_jitter_sat = 0
        self.brush_jitter_val = 0
        self.brush_pickup = 0
        # Wet edge: how strongly each dab's rim is darkened (0-100).
        self.brush_wet_edge = 0
        # Whether the "Advanced" settings block is expanded (session-only UI
        # state, deliberately not part of the saved settings or presets).
        self.brush_advanced = False
        self.eraser_advanced = False
        self.eraser_opacity = 100
        self.eraser_hardness = 100
        self.eraser_blend = "Normal"
        self.eraser_spacing = 25
        self.eraser_antialias = True
        self.eraser_stabilize = 0
        self.eraser_stabilize_mode = "simple"
        self.eraser_tip = "Round"
        self.eraser_pattern = "none"
        self.eraser_velocity_size = 25
        self.eraser_velocity_opacity = 0
        self.eraser_velocity_flow = 0
        self.eraser_velocity_hardness = 0
        self.eraser_velocity_spacing = 0
        self.eraser_taper_in = 0
        self.eraser_taper_out = 0
        self.eraser_grain = "none"
        self.eraser_grain_scale = 100
        self.eraser_grain_strength = 0
        self.eraser_grain_mode = "canvas"
        # Colour dynamics for the colour eraser (see the brush equivalents).
        self.eraser_jitter_hue = 0
        self.eraser_jitter_sat = 0
        self.eraser_jitter_val = 0
        self.eraser_pickup = 0
        self.eraser_target = "transparent"
        self.eraser_color = "#ffffff"
        self.mask_value = 0
        self.mask_erase = False
        self.mask_overlay = "#ff0000"
        # Paint bucket settings (see ``fill`` for how they map to pixels).
        self.fill_tolerance = 32
        self.fill_contiguous = True
        self.fill_antialias = True
        self.fill_gap = 0
        self.fill_opacity = 100
        self.fill_blend = "Normal"
        # Gradient tool settings.
        self.gradient_mode = "linear"
        self.gradient_from = "#000000"
        self.gradient_to = "#ffffff"
        self.gradient_stops = []
        self.gradient_spread = "clamp"
        self.gradient_opacity = 100
        self.gradient_blend = "Normal"
        # Shape tool settings.
        self.shape_kind = "rectangle"
        self.shape_mode = "fill"
        self.shape_opacity = 100
        self.shape_blend = "Normal"
        self.shape_radius = 0
        self.shape_antialias = True
        self.shape_new_layer = False
        # Fill/stroke colours. ``None`` follows the main colour picker, so the
        # defaults match the old single-colour behaviour.
        self.shape_fill_color = None
        self.shape_stroke_color = None
        # Where a pasted image lands: a fresh layer ("new_layer"), the active
        # layer ("selected_layer"), or the canvas assets ("asset").
        self.paste_dest = "new_layer"
        # Scale a pasted image to fit the canvas; when off (the default) its
        # native pixels are kept and the layer is centered by offset so
        # off-canvas pixels survive (see the paste options in the selection tools).
        self.paste_fit_canvas = False
        # When set, Paste first asks where the clipboard should land.
        self.paste_ask = True
        # Clone/smudge/dodge/burn share the brush model but keep their own
        # strength and spacing so switching tools does not clobber the brush.
        self.clone_opacity = 100
        self.clone_hardness = 100
        self.clone_spacing = 25
        self.smudge_strength = 60
        self.smudge_hardness = 50
        self.smudge_spacing = 25
        self.dodge_strength = 50
        self.dodge_hardness = 50
        self.dodge_spacing = 25
        self.dodge_range = "midtones"
        self.dodge_exposure = 50
        self.dodge_protect = True
        self.burn_strength = 50
        self.burn_hardness = 50
        self.burn_spacing = 25
        self.burn_range = "midtones"
        self.burn_exposure = 50
        self.burn_protect = True
        # Liquify tool: ``liquify_mode`` is a ``warp.LIQUIFY_MODES`` value,
        # ``liquify_strength`` is 1-100, and the brush size comes from the shared
        # Size row. ``liquify_twirl_cw`` reuses the twirl mode's direction.
        self.liquify_mode = "push"
        self.liquify_strength = 50
        # Mesh Warp tool: the control grid resolution (cells across/down).
        self.warp_cols = 4
        self.warp_rows = 4
        # Puppet Warp tool: how far (document pixels) each pin reaches, extra
        # margin around the warped region, whether the mesh is shown, and whether
        # pin rotations auto-correct. Pins live in the live session.
        self.puppet_influence = 200
        self.puppet_expansion = 20
        # The full triangulation is noisy over a detailed drawing; show only the
        # pins unless the user opts into the mesh from the tool settings.
        self.puppet_show_mesh = False
        self.puppet_correct_rotation = True
        # Select tool mode: "rect" (marquee), "lasso", "wand", or "segment".
        self.select_mode = "rect"
        # Corner rounding for the rectangular marquee (0 = square corners).
        self.select_radius = 0
        # When on, the Select tool transforms the whole active layer with the
        # transform gizmo instead of dragging a selection marquee.
        self.select_transform = False
        # Magic-wand settings (used when the Select tool is in "wand" mode).
        self.wand_tolerance = 32
        self.wand_contiguous = True
        self.wand_sample = "layer"
        # Segment-AI settings (used when the Select tool is in "segment" mode):
        # the SAM ONNX model label and what the click samples ("layer" for the
        # active layer, "image" for the composited canvas).
        self.segment_model = None
        self.segment_sample = "image"
        self.arrow_tool = True
        self.brush_tool = False
        self.eyedropper = False
        self.eraser = False
        self.fill_tool = False
        self.select_tool = False
        self.layout_tool = False
        self.path_tool = False
        self.vector_tool = False
        self.poses_tool = False
        self.move_tool = False
        self.drag_tool = False
        self.transform_tool = False
        self.mask_edit = False
        self.inpaint_mask_edit = False
        self.gradient_tool = False
        self.shape_tool = False
        self.clone_tool = False
        self.smudge_tool = False
        self.dodge_tool = False
        self.burn_tool = False
        self.liquify_tool = False
        self.warp_tool = False
        self.puppet_tool = False
        self.paste_tool = False
        self.crop_tool = False
        self.text_tool = False
        self.path_mode = "fill"
        self.path_cap = "round"
        self.path_join = "round"
        self.path_outline = 0
        self.path_feather = 0
        self.path_opacity = 100
        self.path_antialias = True
        self.path_show_handles = True
        self.path_new_layer = False
        # Fill/stroke colours for the Path tool; ``None`` follows the picker.
        self.path_fill_color = None
        self.path_stroke_color = None
        self.text_new_layer = False
        # Vector tool. ``vector_mode`` picks the sub-tool: "draw" records a new
        # stroke, "edit" moves/adds/deletes anchors, "width" corrects the width
        # of strokes it is dragged over, and "erase" cuts across strokes (splitting
        # them). ``vector_show_anchors`` draws the anchor handles of the selected
        # stroke.
        self.vector_mode = "draw"
        self.vector_show_anchors = True
        # Draw options. ``vector_color`` follows the main picker while it is
        # ``None``; the rest shape every new stroke (cap/join/antialias reach the
        # stroke spec, smooth/detail shape the recorded polyline).
        self.vector_color = None
        self.vector_opacity = 100
        self.vector_cap = "round"
        self.vector_join = "round"
        self.vector_antialias = True
        self.vector_smooth = 0
        self.vector_detail = 15
        # Edit options. ``vector_snap`` pulls a dragged anchor onto a nearby one;
        # ``vector_anchor_size`` is the on-canvas handle size in pixels.
        self.vector_snap = True
        self.vector_anchor_size = 4
        # Correction options. ``vector_width_mode`` is how the width tool adjusts
        # a line ("set" to the brush size, or "thicken"/"narrow"); ``vector_erase_mode``
        # is what the eraser removes ("touch" the covered anchors, or the "whole"
        # line when it is touched).
        self.vector_width_mode = "set"
        self.vector_erase_mode = "touch"
        # When set, scaling a selection with the corner handles scales each
        # line's width too (CSP's "adjust line thickness when scaling").
        self.vector_scale_width = False
        # Poses tool. The size, colour and opacity apply to newly added figures;
        # dragging off the figure turns it in 3D (there is no separate mode).
        self.poses_mode = "pose"
        self.poses_body = poses.DEFAULT_BODY
        self.poses_pose = "relaxed"
        self.poses_hand = "open"
        self.poses_hand_side = "both"
        self.poses_drag = poses.DEFAULT_DRAG_QUALITY
        self._pose_thumbs = {}
        self.poses_scale = poses.DEFAULT_SCALE
        self.poses_color = poses.DEFAULT_COLOR
        self.poses_opacity = 100
        self.poses_grid = True
        # Symmetry guide (see ``symmetry``): mirrors every dab of a painting
        # tool across the canvas centre. ``symmetry_mode`` is the guide mode;
        # ``symmetry_count`` and ``symmetry_mirror`` apply to the radial mode.
        self.symmetry_mode = "off"
        self.symmetry_count = symmetry.DEFAULT_COUNT
        self.symmetry_mirror = True
        self._build()
        theme.subscribe(self._on_theme)
        self._layout_family = self._layout_family_value()
        self._build_layout_editor()

    def _build(self):
        self.size_var = tk.IntVar(value=self.brush_size)
        # The slider gets its own variable so its 1..BRUSH_SLIDER_MAX range
        # cannot clamp a larger size typed into the spinbox, which shares
        # ``size_var`` and reaches up to MAX_BRUSH_SIZE.
        self._size_slider_var = tk.IntVar(
            value=min(self.brush_size, BRUSH_SLIDER_MAX))
        self.size_var.trace_add("write", self._on_size_var_changed)

        # The tool grid is re-laid-out on every rail resize: a narrow rail shows
        # icons alone, a wide one shows each tool's label beside its icon (see
        # :meth:`set_width`). ``_columns`` and ``_compact`` hold the current
        # layout so a resize only rebuilds when the arrangement actually changes.
        self._grid = tk.Frame(self)
        self._grid.pack(fill=tk.X)
        self._compact = False
        self._columns = self.TOOL_COLUMNS
        self._tool_buttons = {}
        self._tool_labels = {}
        for key, label in self.TOOL_BUTTONS:
            command = (
                self.mask_edit_clicked if key == "mask"
                else (lambda k=key: self.select(k))
            )
            button = tk.Button(self._grid, command=command)
            Tooltip(button, label)
            self._tool_buttons[key] = button
            self._tool_labels[key] = label

        if self._settings_parent is not None:
            self.settings = self._settings_parent
        else:
            self.settings = tk.LabelFrame(self, text="Tool settings")
            self.settings.pack(fill=tk.X, pady=(6, 8))

        # Lays out the buttons and syncs their icons; ``self.settings`` must
        # exist first because the sync rebuilds the settings panel.
        self._relayout_tools()

    # ------------------------------------------------------------- tool layout
    def _relayout_tools(self):
        """Arrange the tool buttons for the current compact/full state.

        Drops every column's weight, re-applies it to the active column count,
        then re-grids each button. The final :meth:`_sync_buttons` re-styles the
        icons for the (possibly changed) compact state.
        """
        grid = self._grid
        for column in range(grid.grid_size()[0]):
            grid.columnconfigure(column, weight=0, uniform="")
        for column in range(self._columns):
            grid.columnconfigure(column, weight=1, uniform="tool")
        for index, (key, _label) in enumerate(self.TOOL_BUTTONS):
            row, column = divmod(index, self._columns)
            self._tool_buttons[key].grid(
                row=row, column=column, sticky="ew", padx=1, pady=1)
        self._sync_buttons()

    def set_width(self, width):
        """Adapt the button layout to the tool rail's available ``width``.

        The palette steps through three layouts as the rail widens: a single
        column of icon-only buttons, two columns of icon-only buttons once
        :data:`TOOL_ICON2_MIN_W` is reached, and finally a single column of
        icon+label buttons once there is room for a label at
        :data:`TOOL_FULL_MIN_W`. In the two icon-only states each button keeps
        its label as a tooltip.
        """
        width = max(1, int(width))
        if width >= TOOL_FULL_MIN_W:
            compact, columns = False, 1
        elif width >= TOOL_ICON2_MIN_W:
            compact, columns = True, 2
        else:
            compact, columns = True, 1
        if compact == self._compact and columns == self._columns:
            return
        self._compact = compact
        self._columns = columns
        self._relayout_tools()

    # ------------------------------------------------------------------ tools
    def _active_tool(self):
        for key, flag in self.TOOL_FLAGS.items():
            if getattr(self, flag):
                return key
        return "brush"

    def _any_tool_active(self):
        return any(getattr(self, flag) for flag in self.TOOL_FLAGS.values())

    def select(self, key):
        """Make ``key`` the active tool; deactivating a tool selects brush."""
        if key == "mask":
            if not self.mask_edit:
                self._on_mask_edit()
            return
        if getattr(self, self.TOOL_FLAGS[key]):
            return
        for other, flag in self.TOOL_FLAGS.items():
            setattr(self, flag, other == key)
        if hasattr(self, "_layout_last_pick"):
            self._layout_last_pick = None
        self._sync_buttons()
        self._on_mode_change()

    def _toggle(self, key):
        if getattr(self, self.TOOL_FLAGS[key]):
            self.select("brush")
        else:
            self.select(key)

    def toggle_arrow(self):
        self._toggle("arrow")

    def toggle_brush(self):
        self.select("brush")

    def toggle_eyedropper(self):
        self._toggle("pick")

    def toggle_eraser(self):
        self._toggle("eraser")

    def toggle_fill(self):
        self._toggle("fill")

    def toggle_select(self):
        self._toggle("select")

    def toggle_layout(self):
        self._toggle("layout")

    def toggle_lasso(self):
        self.set_select_mode("lasso")

    def set_select_mode(self, mode):
        """Activate the Select tool in ``mode`` ("rect", "lasso", "wand" or
        "segment")."""
        self.select_mode = mode
        if not self.select_tool:
            self.select("select")
        else:
            self._sync_buttons()
            self._on_mode_change()

    def toggle_path(self):
        self._toggle("path")

    def toggle_vector(self):
        self._toggle("vector")

    def toggle_liquify(self):
        self._toggle("liquify")

    def toggle_warp(self):
        self._toggle("warp")

    def toggle_puppet(self):
        self._toggle("puppet")

    def set_vector_mode(self, mode):
        """Activate the Vector tool in ``mode`` ("draw"/"edit"/"width"/"erase")."""
        if mode in ("draw", "edit", "width", "erase"):
            self.vector_mode = mode
        if not self.vector_tool:
            self.select("vector")
        else:
            self._sync_buttons()
            self._on_mode_change()

    def toggle_poses(self):
        self._toggle("poses")

    def set_poses_mode(self, mode):
        """Activate the Poses tool in ``mode`` ("pose"/"orbit")."""
        if mode in ("pose", "orbit"):
            self.poses_mode = mode
        if not self.poses_tool:
            self.select("poses")
        else:
            self._sync_buttons()
            self._on_mode_change()

    def toggle_move(self):
        self._toggle("move")

    def toggle_drag(self):
        self._toggle("drag")

    def toggle_transform(self):
        self._toggle("transform")

    def toggle_text(self):
        self._toggle("text")

    def toggle_gradient(self):
        self._toggle("gradient")

    def toggle_shape(self):
        self._toggle("shape")

    def toggle_clone(self):
        self._toggle("clone")

    def toggle_smudge(self):
        self._toggle("smudge")

    def toggle_dodge(self):
        self._toggle("dodge")

    def toggle_burn(self):
        self._toggle("burn")

    def toggle_wand(self):
        self.set_select_mode("wand")

    def toggle_paste(self):
        self._toggle("paste")

    def toggle_crop(self):
        self._toggle("crop")

    def mask_edit_clicked(self):
        """Delegates to the coordinator, which owns the mask-edit state."""
        self._on_mask_edit()

    def cancel_modes(self):
        if self.arrow_tool:
            return
        self.select("arrow")

    def _sync_buttons(self):
        active = self._active_tool()
        for key, button in self._tool_buttons.items():
            button.config(relief=tk.SUNKEN if key == active else tk.RAISED)
            self._set_tool_icon(button, key, key == active)
        self._render_settings(active)

    def _on_theme(self):
        """Re-render the tool-button icons with the new theme's colors."""
        active = self._active_tool()
        for key, button in self._tool_buttons.items():
            self._set_tool_icon(button, key, key == active)

    def _set_tool_icon(self, button, key, active):
        """Draw a tool button's icon beside its label (accent when active).

        The active tool is tinted with the accent color in addition to its
        sunken relief, so the current tool reads at a glance. The icon and label
        are laid out left-aligned (``compound="left"``) with a leading space, as
        Tk puts no gap between them. A button showing an image reads ``width``
        in pixels rather than text units, so it is cleared to let the button
        size itself.

        In the rail's compact (narrow) state the label is dropped and the icon
        is centered, so the button fits an icon-only column; the label stays
        reachable as the button's tooltip.
        """
        name = self.TOOL_ICONS.get(key)
        label = self._tool_labels.get(key, key)
        image = None
        if name:
            color = theme.color("accent" if active else "text")
            image = icons.photo(name, self.TOOL_ICON, color,
                                padding=self.TOOL_PAD)
        if image is None:
            # No rasterised icon: always fall back to the text label, even in the
            # compact state, so the button never goes blank.
            button.config(image="", text=label, anchor="w", width=0, height=0,
                          padx=0, pady=0)
        elif self._compact:
            button.config(image=image, text="", compound="center",
                          anchor="center", width=0, height=0, padx=2, pady=2)
        else:
            button.config(image=image, text=f" {label}", compound="left",
                          anchor="w", width=0, height=0, padx=0, pady=0)

    # --------------------------------------------------------------- settings
    def _render_settings(self, active):
        """Rebuild the settings panel for the active tool."""
        for child in self.settings.winfo_children():
            child.destroy()
        self._brush_preview = None
        # The advanced block's toggle button, if the tool has one; the symmetry
        # block packs before it so its guide controls stay above "Advanced".
        self._advanced_anchor = None
        builder = getattr(self, f"_settings_for_{active}", None)
        if builder is None:
            tk.Label(
                self.settings, text="No options for this tool.",
                anchor="w", fg="#888888",
            ).pack(fill=tk.X, pady=2)
        else:
            builder()
        if active in self.SYMMETRY_TOOLS:
            self._symmetry_rows()
        theme.restyle(self.settings)

    def _size_row(self, parent):
        row = tk.Frame(parent)
        row.pack(fill=tk.X)
        tk.Scale(
            row, from_=1, to=BRUSH_SLIDER_MAX, orient=tk.HORIZONTAL,
            variable=self._size_slider_var, command=self._on_size_slider,
            length=100,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Spinbox(
            row, from_=1, to=MAX_BRUSH_SIZE, width=4,
            textvariable=self.size_var, command=self.on_size_change,
        ).pack(side=tk.LEFT, padx=(2, 2))

    # -------------------------------------------------------- setting widgets
    def _slider_row(self, parent, label, var, low, high, command):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(3, 0))
        tk.Label(row, text=label, width=9, anchor="w").pack(side=tk.LEFT)
        tk.Scale(
            row, from_=low, to=high, orient=tk.HORIZONTAL, variable=var,
            command=command, length=88, showvalue=False,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Spinbox(
            row, from_=low, to=high, width=4, textvariable=var,
            command=command,
        ).pack(side=tk.LEFT, padx=(2, 0))

    def _check_row(self, parent, label, var, command):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(3, 0))
        tk.Checkbutton(
            row, text=label, variable=var, command=command, anchor="w",
        ).pack(fill=tk.X)

    def _advanced_toggle(self, parent, attr_name):
        """A header that shows/hides a collapsible "Advanced" block.

        Returns the frame to build the advanced rows into. The open state is
        kept per tool in the plain ``attr_name`` attribute, so it survives the
        panel being rebuilt but is not saved into projects or brush presets.
        """
        frame = tk.Frame(parent)
        var = tk.BooleanVar(value=bool(getattr(self, attr_name, False)))
        button = tk.Button(parent, anchor="w", relief=tk.FLAT, bd=0)

        def apply():
            setattr(self, attr_name, bool(var.get()))
            button.config(text="\u25be Advanced" if var.get() else "\u25b8 Advanced")
            if var.get():
                frame.pack(fill=tk.X, pady=(2, 0))
            else:
                frame.pack_forget()

        button.config(command=lambda: (var.set(not var.get()), apply()))
        button.pack(fill=tk.X, pady=(6, 0))
        self._advanced_anchor = button
        apply()
        return frame

    def _menu_row(self, parent, label, var, options, command):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(3, 0))
        tk.Label(row, text=label, width=9, anchor="w").pack(side=tk.LEFT)
        labels = [text for _value, text in options]
        by_label = {text: value for value, text in options}
        display_var = tk.StringVar(value=self._option_label(var.get(), options))
        combo = ttk.Combobox(
            row, textvariable=display_var, values=labels,
            state="readonly", width=1,
        )

        def on_select(_event=None):
            value = by_label.get(display_var.get())
            if value is None:
                return
            var.set(value)
            if command is not None:
                command()

        combo.bind("<<ComboboxSelected>>", on_select)
        combo._display_var = display_var
        combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        return combo

    @staticmethod
    def _option_label(value, options):
        for key, text in options:
            if key == value:
                return text
        return str(value)

    def _int_var(self, var, default=0):
        try:
            return int(round(float(var.get())))
        except (tk.TclError, ValueError):
            return default

    def _current_color(self):
        """The main colour picker's colour, used as the shape/path default."""
        if self._get_color is None:
            return "#000000"
        try:
            return self._get_color() or "#000000"
        except tk.TclError:
            return "#000000"

    def _color_row(self, parent, label, command):
        """A label above a full-width swatch that spans the settings panel.

        The button is created against ``row`` so it lays out inside the row;
        building it against ``parent`` would let ``pack`` place it across the
        whole panel and squeeze the other rows into a right-hand column.
        """
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(3, 0))
        tk.Label(row, text=label, anchor="w").pack(fill=tk.X)
        button = tk.Button(row, command=command)
        theme.skip(button)
        button.pack(fill=tk.X, pady=(1, 0))
        return button

    def _swatch_row(self, parent):
        """A frame whose children lay out side by side (the shape/path colours).

        Return the frame and create the swatches with it as their parent;
        packing a widget always targets its own master, so a button built
        against ``parent`` would escape the row.
        """
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=(3, 0))
        return row

    def _settings_for_brush(self):
        self._brush_opacity_var = tk.IntVar(value=self.brush_opacity)
        self._brush_hardness_var = tk.IntVar(value=self.brush_hardness)
        self._brush_flow_var = tk.IntVar(value=self.brush_flow)
        self._brush_spacing_var = tk.IntVar(value=self.brush_spacing)
        self._brush_stabilize_var = tk.IntVar(value=self.brush_stabilize)
        self._brush_stabilize_mode_var = tk.StringVar(
            value=self.brush_stabilize_mode)
        self._brush_aa_var = tk.BooleanVar(value=self.brush_antialias)
        self._brush_blend_var = tk.StringVar(value=self.brush_blend)
        self._brush_tip_var = tk.StringVar(value=self.brush_tip)
        self._brush_pattern_var = tk.StringVar(value=self.brush_pattern)
        self._brush_vel_size_var = tk.IntVar(value=self.brush_velocity_size)
        self._brush_vel_opacity_var = tk.IntVar(value=self.brush_velocity_opacity)
        self._brush_vel_flow_var = tk.IntVar(value=self.brush_velocity_flow)
        self._brush_vel_hardness_var = tk.IntVar(
            value=self.brush_velocity_hardness)
        self._brush_vel_spacing_var = tk.IntVar(value=self.brush_velocity_spacing)
        self._brush_taper_in_var = tk.IntVar(value=self.brush_taper_in)
        self._brush_taper_out_var = tk.IntVar(value=self.brush_taper_out)
        self._brush_grain_var = tk.StringVar(value=self.brush_grain)
        self._brush_grain_scale_var = tk.IntVar(value=self.brush_grain_scale)
        self._brush_grain_strength_var = tk.IntVar(
            value=self.brush_grain_strength)
        self._brush_grain_mode_var = tk.StringVar(value=self.brush_grain_mode)
        self._brush_jitter_hue_var = tk.IntVar(value=self.brush_jitter_hue)
        self._brush_jitter_sat_var = tk.IntVar(value=self.brush_jitter_sat)
        self._brush_jitter_val_var = tk.IntVar(value=self.brush_jitter_val)
        self._brush_pickup_var = tk.IntVar(value=self.brush_pickup)
        self._brush_wet_edge_var = tk.IntVar(value=self.brush_wet_edge)

        self._brush_preview = BrushPreview(self.settings,
                                           render=self._render_brush_preview)
        self._brush_preview.pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._slider_row(self.settings, "Opacity", self._brush_opacity_var,
                         1, 100, self._sync_brush)
        self._slider_row(self.settings, "Hardness", self._brush_hardness_var,
                         0, 100, self._sync_brush)
        self._slider_row(self.settings, "Flow", self._brush_flow_var,
                         1, 100, self._sync_brush)
        self._slider_row(self.settings, "Spacing", self._brush_spacing_var,
                         1, 200, self._sync_brush)
        self._slider_row(self.settings, "Smooth", self._brush_stabilize_var,
                         0, 100, self._sync_brush)
        self._menu_row(self.settings, "Stabilizer", self._brush_stabilize_mode_var,
                       STABILIZE_MODES.items(), self._sync_brush)
        self._menu_row(self.settings, "Blend", self._brush_blend_var,
                       brush.BLEND_MODES.items(), self._sync_brush)
        # Everything below is tucked away so the common controls stay readable.
        advanced = self._advanced_toggle(self.settings, "brush_advanced")
        self._slider_row(advanced, "Size speed", self._brush_vel_size_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Opacity spd",
                         self._brush_vel_opacity_var, 0, 100, self._sync_brush)
        self._slider_row(advanced, "Flow spd", self._brush_vel_flow_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Hardness spd",
                         self._brush_vel_hardness_var, 0, 100, self._sync_brush)
        self._slider_row(advanced, "Spacing spd",
                         self._brush_vel_spacing_var, 0, 100, self._sync_brush)
        self._slider_row(advanced, "Taper in", self._brush_taper_in_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Taper out", self._brush_taper_out_var,
                         0, 100, self._sync_brush)
        self._menu_row(advanced, "Grain", self._brush_grain_var,
                       brush.GRAIN_TYPES.items(), self._sync_brush)
        self._slider_row(advanced, "Grain size",
                         self._brush_grain_scale_var, 25, 400, self._sync_brush)
        self._slider_row(advanced, "Grain amt",
                         self._brush_grain_strength_var, 0, 100, self._sync_brush)
        self._menu_row(advanced, "Grain on", self._brush_grain_mode_var,
                       brush.GRAIN_MODES.items(), self._sync_brush)
        self._slider_row(advanced, "Hue var", self._brush_jitter_hue_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Sat var", self._brush_jitter_sat_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Val var", self._brush_jitter_val_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Pickup", self._brush_pickup_var,
                         0, 100, self._sync_brush)
        self._slider_row(advanced, "Wet edge", self._brush_wet_edge_var,
                         0, 100, self._sync_brush)
        self._menu_row(advanced, "Tip", self._brush_tip_var,
                       brush.TIP_SHAPES.items(), self._sync_brush)
        self._menu_row(advanced, "Pattern", self._brush_pattern_var,
                       brush.EFFECT_PATTERNS.items(), self._sync_brush)
        self._check_row(advanced, "Anti-alias", self._brush_aa_var,
                        self._sync_brush)
        self._refresh_brush_preview()

    def _sync_brush(self, *_):
        self.brush_opacity = max(1, self._int_var(self._brush_opacity_var, 100))
        self.brush_hardness = max(0, self._int_var(self._brush_hardness_var, 100))
        self.brush_flow = max(1, self._int_var(self._brush_flow_var, 100))
        self.brush_spacing = max(1, self._int_var(self._brush_spacing_var, 25))
        self.brush_stabilize = max(0, self._int_var(self._brush_stabilize_var, 0))
        self.brush_stabilize_mode = self._brush_stabilize_mode_var.get()
        self.brush_antialias = bool(self._brush_aa_var.get())
        self.brush_blend = self._brush_blend_var.get()
        self.brush_tip = self._brush_tip_var.get()
        self.brush_pattern = self._brush_pattern_var.get()
        self.brush_velocity_size = max(
            0, min(100, self._int_var(self._brush_vel_size_var, 25)))
        self.brush_velocity_opacity = max(
            0, min(100, self._int_var(self._brush_vel_opacity_var, 0)))
        self.brush_velocity_flow = max(
            0, min(100, self._int_var(self._brush_vel_flow_var, 0)))
        self.brush_velocity_hardness = max(
            0, min(100, self._int_var(self._brush_vel_hardness_var, 0)))
        self.brush_velocity_spacing = max(
            0, min(100, self._int_var(self._brush_vel_spacing_var, 0)))
        self.brush_taper_in = max(
            0, min(100, self._int_var(self._brush_taper_in_var, 0)))
        self.brush_taper_out = max(
            0, min(100, self._int_var(self._brush_taper_out_var, 0)))
        self.brush_grain = self._brush_grain_var.get()
        self.brush_grain_scale = max(
            25, min(400, self._int_var(self._brush_grain_scale_var, 100)))
        self.brush_grain_strength = max(
            0, min(100, self._int_var(self._brush_grain_strength_var, 0)))
        self.brush_grain_mode = self._brush_grain_mode_var.get()
        self.brush_jitter_hue = self._brush_jitter_hue_var.get()
        self.brush_jitter_sat = self._brush_jitter_sat_var.get()
        self.brush_jitter_val = self._brush_jitter_val_var.get()
        self.brush_pickup = self._brush_pickup_var.get()
        self.brush_wet_edge = max(
            0, min(100, self._int_var(self._brush_wet_edge_var, 0)))
        self._refresh_brush_preview()
        self._on_brush_size()

    def _render_brush_preview(self, width, height):
        return brush_preview_image(
            self.brush_size, self.brush_hardness, self.brush_flow,
            self.brush_opacity, self.brush_spacing, self.brush_tip.lower(),
            self.brush_pattern,
            brush.Dynamics(
                size=self.brush_velocity_size / 100.0,
                opacity=self.brush_velocity_opacity / 100.0,
                flow=self.brush_velocity_flow / 100.0,
                hardness=self.brush_velocity_hardness / 100.0,
                spacing=self.brush_velocity_spacing / 100.0,
                taper_in=self.brush_size * self.brush_taper_in / 100.0,
                taper_out=self.brush_size * self.brush_taper_out / 100.0,
            ),
            kind="brush",
            color=rgba(self._current_color())[:3],
            grain=self.brush_grain,
            grain_scale=self.brush_grain_scale / 100.0,
            grain_strength=self.brush_grain_strength / 100.0,
            grain_mode=self.brush_grain_mode,
            jitter_hue=self.brush_jitter_hue / 100.0,
            jitter_sat=self.brush_jitter_sat / 100.0,
            jitter_val=self.brush_jitter_val / 100.0,
            pickup=self.brush_pickup / 100.0,
            wet_edge=self.brush_wet_edge / 100.0,
            width=width, height=height,
        )

    def _refresh_brush_preview(self):
        preview = getattr(self, "_brush_preview", None)
        if preview is not None and preview.winfo_exists():
            preview.refresh()

    def refresh_color_previews(self):
        """Repaint the sample strips painted in the main colour.

        The brush sample takes its colour from the colour picker, so a colour
        change repaints it even though no brush setting changed.
        """
        self._refresh_brush_preview()

    def _settings_for_eraser(self):
        self._eraser_opacity_var = tk.IntVar(value=self.eraser_opacity)
        self._eraser_hardness_var = tk.IntVar(value=self.eraser_hardness)
        self._eraser_spacing_var = tk.IntVar(value=self.eraser_spacing)
        self._eraser_stabilize_var = tk.IntVar(value=self.eraser_stabilize)
        self._eraser_stabilize_mode_var = tk.StringVar(
            value=self.eraser_stabilize_mode)
        self._eraser_aa_var = tk.BooleanVar(value=self.eraser_antialias)
        self._eraser_blend_var = tk.StringVar(value=self.eraser_blend)
        self._eraser_tip_var = tk.StringVar(value=self.eraser_tip)
        self._eraser_pattern_var = tk.StringVar(value=self.eraser_pattern)
        self._eraser_target_var = tk.StringVar(value=self.eraser_target)
        self._eraser_vel_size_var = tk.IntVar(value=self.eraser_velocity_size)
        self._eraser_vel_opacity_var = tk.IntVar(
            value=self.eraser_velocity_opacity)
        self._eraser_vel_flow_var = tk.IntVar(value=self.eraser_velocity_flow)
        self._eraser_vel_hardness_var = tk.IntVar(
            value=self.eraser_velocity_hardness)
        self._eraser_vel_spacing_var = tk.IntVar(
            value=self.eraser_velocity_spacing)
        self._eraser_taper_in_var = tk.IntVar(value=self.eraser_taper_in)
        self._eraser_taper_out_var = tk.IntVar(value=self.eraser_taper_out)
        self._eraser_grain_var = tk.StringVar(value=self.eraser_grain)
        self._eraser_grain_scale_var = tk.IntVar(value=self.eraser_grain_scale)
        self._eraser_grain_strength_var = tk.IntVar(
            value=self.eraser_grain_strength)
        self._eraser_grain_mode_var = tk.StringVar(value=self.eraser_grain_mode)
        self._eraser_jitter_hue_var = tk.IntVar(value=self.eraser_jitter_hue)
        self._eraser_jitter_sat_var = tk.IntVar(value=self.eraser_jitter_sat)
        self._eraser_jitter_val_var = tk.IntVar(value=self.eraser_jitter_val)
        self._eraser_pickup_var = tk.IntVar(value=self.eraser_pickup)

        self._eraser_preview = BrushPreview(self.settings,
                                            render=self._render_eraser_preview)
        self._eraser_preview.pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._menu_row(self.settings, "Erase to", self._eraser_target_var,
                       [("transparent", "Transparent"),
                        ("color", "Fill color")], self._sync_eraser)
        self._eraser_color_btn = self._color_row(
            self.settings, "Color", self._choose_eraser_color)
        self._apply_eraser_swatch()
        self._slider_row(self.settings, "Opacity", self._eraser_opacity_var,
                         1, 100, self._sync_eraser)
        self._slider_row(self.settings, "Hardness", self._eraser_hardness_var,
                         0, 100, self._sync_eraser)
        self._slider_row(self.settings, "Spacing", self._eraser_spacing_var,
                         1, 200, self._sync_eraser)
        self._slider_row(self.settings, "Smooth", self._eraser_stabilize_var,
                         0, 100, self._sync_eraser)
        self._menu_row(self.settings, "Stabilizer",
                       self._eraser_stabilize_mode_var,
                       STABILIZE_MODES.items(), self._sync_eraser)
        self._menu_row(self.settings, "Blend", self._eraser_blend_var,
                       brush.BLEND_MODES.items(), self._sync_eraser)
        advanced = self._advanced_toggle(self.settings, "eraser_advanced")
        self._slider_row(advanced, "Size speed", self._eraser_vel_size_var,
                         0, 100, self._sync_eraser)
        self._slider_row(advanced, "Opacity spd",
                         self._eraser_vel_opacity_var, 0, 100, self._sync_eraser)
        self._slider_row(advanced, "Flow spd", self._eraser_vel_flow_var,
                         0, 100, self._sync_eraser)
        self._slider_row(advanced, "Hardness spd",
                         self._eraser_vel_hardness_var, 0, 100, self._sync_eraser)
        self._slider_row(advanced, "Spacing spd",
                         self._eraser_vel_spacing_var, 0, 100, self._sync_eraser)
        self._slider_row(advanced, "Taper in", self._eraser_taper_in_var,
                         0, 100, self._sync_eraser)
        self._slider_row(advanced, "Taper out", self._eraser_taper_out_var,
                         0, 100, self._sync_eraser)
        self._menu_row(advanced, "Grain", self._eraser_grain_var,
                       brush.GRAIN_TYPES.items(), self._sync_eraser)
        self._slider_row(advanced, "Grain size",
                         self._eraser_grain_scale_var, 25, 400, self._sync_eraser)
        self._slider_row(advanced, "Grain amt",
                         self._eraser_grain_strength_var, 0, 100, self._sync_eraser)
        self._menu_row(advanced, "Grain on", self._eraser_grain_mode_var,
                       brush.GRAIN_MODES.items(), self._sync_eraser)
        self._slider_row(advanced, "Hue var", self._eraser_jitter_hue_var,
                         0, 100, self._sync_eraser)
        self._slider_row(advanced, "Sat var", self._eraser_jitter_sat_var,
                         0, 100, self._sync_eraser)
        self._slider_row(advanced, "Val var", self._eraser_jitter_val_var,
                         0, 100, self._sync_eraser)
        self._slider_row(advanced, "Pickup", self._eraser_pickup_var,
                         0, 100, self._sync_eraser)
        self._menu_row(advanced, "Tip", self._eraser_tip_var,
                       brush.TIP_SHAPES.items(), self._sync_eraser)
        self._menu_row(advanced, "Pattern", self._eraser_pattern_var,
                       brush.EFFECT_PATTERNS.items(), self._sync_eraser)
        self._check_row(advanced, "Anti-alias", self._eraser_aa_var,
                        self._sync_eraser)
        self._refresh_eraser_preview()

    def _apply_eraser_swatch(self):
        button = getattr(self, "_eraser_color_btn", None)
        if button is not None:
            button.config(bg=self.eraser_color, activebackground=self.eraser_color)

    def _choose_eraser_color(self):
        def preview(color):
            self.eraser_color = color
            self._apply_eraser_swatch()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), self.eraser_color,
            title="Eraser Color", on_preview=preview,
        )
        if chosen is not None:
            preview(chosen)

    def _sync_eraser(self, *_):
        self.eraser_opacity = max(1, self._int_var(self._eraser_opacity_var, 100))
        self.eraser_hardness = max(0, self._int_var(self._eraser_hardness_var, 100))
        self.eraser_spacing = max(1, self._int_var(self._eraser_spacing_var, 25))
        self.eraser_stabilize = max(0, self._int_var(self._eraser_stabilize_var, 0))
        self.eraser_stabilize_mode = self._eraser_stabilize_mode_var.get()
        self.eraser_antialias = bool(self._eraser_aa_var.get())
        self.eraser_blend = self._eraser_blend_var.get()
        self.eraser_tip = self._eraser_tip_var.get()
        self.eraser_pattern = self._eraser_pattern_var.get()
        self.eraser_target = self._eraser_target_var.get()
        self.eraser_velocity_size = max(
            0, min(100, self._int_var(self._eraser_vel_size_var, 25)))
        self.eraser_velocity_opacity = max(
            0, min(100, self._int_var(self._eraser_vel_opacity_var, 0)))
        self.eraser_velocity_flow = max(
            0, min(100, self._int_var(self._eraser_vel_flow_var, 0)))
        self.eraser_velocity_hardness = max(
            0, min(100, self._int_var(self._eraser_vel_hardness_var, 0)))
        self.eraser_velocity_spacing = max(
            0, min(100, self._int_var(self._eraser_vel_spacing_var, 0)))
        self.eraser_taper_in = max(
            0, min(100, self._int_var(self._eraser_taper_in_var, 0)))
        self.eraser_taper_out = max(
            0, min(100, self._int_var(self._eraser_taper_out_var, 0)))
        self.eraser_grain = self._eraser_grain_var.get()
        self.eraser_grain_scale = max(
            25, min(400, self._int_var(self._eraser_grain_scale_var, 100)))
        self.eraser_grain_strength = max(
            0, min(100, self._int_var(self._eraser_grain_strength_var, 0)))
        self.eraser_grain_mode = self._eraser_grain_mode_var.get()
        self.eraser_jitter_hue = self._eraser_jitter_hue_var.get()
        self.eraser_jitter_sat = self._eraser_jitter_sat_var.get()
        self.eraser_jitter_val = self._eraser_jitter_val_var.get()
        self.eraser_pickup = self._eraser_pickup_var.get()
        self._refresh_eraser_preview()
        self._on_brush_size()

    def _render_eraser_preview(self, width, height):
        color = rgba(self.eraser_color)[:3]
        kind = ("eraser-color" if self.eraser_target == "color" else "eraser")
        return brush_preview_image(
            self.brush_size, self.eraser_hardness, 100, self.eraser_opacity,
            self.eraser_spacing, self.eraser_tip.lower(), self.eraser_pattern,
            brush.Dynamics(
                size=self.eraser_velocity_size / 100.0,
                opacity=self.eraser_velocity_opacity / 100.0,
                flow=self.eraser_velocity_flow / 100.0,
                hardness=self.eraser_velocity_hardness / 100.0,
                spacing=self.eraser_velocity_spacing / 100.0,
                taper_in=self.brush_size * self.eraser_taper_in / 100.0,
                taper_out=self.brush_size * self.eraser_taper_out / 100.0,
            ),
            kind=kind, color=color,
            grain=self.eraser_grain,
            grain_scale=self.eraser_grain_scale / 100.0,
            grain_strength=self.eraser_grain_strength / 100.0,
            grain_mode=self.eraser_grain_mode,
            jitter_hue=self.eraser_jitter_hue / 100.0,
            jitter_sat=self.eraser_jitter_sat / 100.0,
            jitter_val=self.eraser_jitter_val / 100.0,
            pickup=self.eraser_pickup / 100.0,
            width=width, height=height,
        )

    def _refresh_eraser_preview(self):
        preview = getattr(self, "_eraser_preview", None)
        if preview is not None:
            preview.refresh()

    def _settings_for_fill(self):
        self._fill_tolerance_var = tk.IntVar(value=self.fill_tolerance)
        self._fill_opacity_var = tk.IntVar(value=self.fill_opacity)
        self._fill_contiguous_var = tk.BooleanVar(value=self.fill_contiguous)
        self._fill_antialias_var = tk.BooleanVar(value=self.fill_antialias)
        self._fill_gap_var = tk.IntVar(value=self.fill_gap)
        self._fill_blend_var = tk.StringVar(value=self.fill_blend)

        tk.Label(self.settings, text="Fill the active layer from the click.",
                 anchor="w", fg="#888888").pack(fill=tk.X)
        self._slider_row(self.settings, "Tolerance", self._fill_tolerance_var,
                         0, 255, self._sync_fill)
        self._slider_row(self.settings, "Opacity", self._fill_opacity_var,
                         1, 100, self._sync_fill)
        self._slider_row(self.settings, "Close gap", self._fill_gap_var,
                         0, 40, self._sync_fill)
        self._menu_row(self.settings, "Blend", self._fill_blend_var,
                       brush.BLEND_MODES.items(), self._sync_fill)
        self._check_row(self.settings, "Contiguous", self._fill_contiguous_var,
                        self._sync_fill)
        self._check_row(self.settings, "Anti-alias", self._fill_antialias_var,
                        self._sync_fill)

    def _sync_fill(self, *_):
        self.fill_tolerance = max(
            0, min(255, self._int_var(self._fill_tolerance_var, 32))
        )
        self.fill_opacity = max(1, self._int_var(self._fill_opacity_var, 100))
        self.fill_contiguous = bool(self._fill_contiguous_var.get())
        self.fill_antialias = bool(self._fill_antialias_var.get())
        self.fill_gap = max(0, self._int_var(self._fill_gap_var, 0))
        self.fill_blend = self._fill_blend_var.get()

    # ------------------------------------------------------ gradient settings
    def _gradient_stop_list(self):
        """The gradient's stops as ``[pos, "#rrggbb", alpha]`` rows.

        Falls back to a two-stop list built from the From/To colours when no
        explicit stops have been created (or they failed to load).
        """
        stops = gradient.normalize_stops(getattr(self, "gradient_stops", None))
        if len(stops) >= 2:
            return stops
        return [
            [0.0, self.gradient_from, 255],
            [1.0, self.gradient_to, 255],
        ]

    def _settings_for_gradient(self):
        self._gradient_mode_var = tk.StringVar(value=self.gradient_mode)
        self._gradient_spread_var = tk.StringVar(value=self.gradient_spread)
        self._gradient_preset_var = tk.StringVar(value="custom")
        self._gradient_opacity_var = tk.IntVar(value=self.gradient_opacity)
        self._gradient_blend_var = tk.StringVar(value=self.gradient_blend)
        tk.Label(self.settings, text="Drag on the canvas to set the axis.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        self._menu_row(self.settings, "Type", self._gradient_mode_var,
                       gradient.MODES, self._sync_gradient)

        self._gradient_editor = gradienteditor.StopsEditor(
            self.settings, self._gradient_stop_list(),
            on_change=self._on_gradient_stops_changed,
            on_pick_color=self._pick_gradient_stop_color,
        )
        self._gradient_editor.pack(fill=tk.X, pady=(4, 2))
        tk.Label(self.settings,
                 text="Click a stop to select, drag to move, double-click to "
                      "recolour.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)

        self._gradient_stop_pos_var = tk.IntVar(value=0)
        self._gradient_stop_alpha_var = tk.IntVar(value=255)
        self._slider_row(self.settings, "Position", self._gradient_stop_pos_var,
                         0, 100, self._on_gradient_stop_pos)
        self._slider_row(self.settings, "Alpha", self._gradient_stop_alpha_var,
                         0, 255, self._on_gradient_stop_alpha)

        stop_buttons = tk.Frame(self.settings)
        stop_buttons.pack(fill=tk.X, pady=(3, 0))
        tk.Button(stop_buttons, text="Add Stop", command=self._add_gradient_stop,
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Button(stop_buttons, text="Delete", command=self._delete_gradient_stop,
                  ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self._menu_row(self.settings, "Spread", self._gradient_spread_var,
                       gradient.SPREADS, self._sync_gradient)
        self._menu_row(self.settings, "Preset", self._gradient_preset_var,
                       gradient.PRESETS, self._apply_gradient_preset)
        self._slider_row(self.settings, "Opacity", self._gradient_opacity_var,
                         1, 100, self._sync_gradient)
        self._menu_row(self.settings, "Blend", self._gradient_blend_var,
                       brush.BLEND_MODES.items(), self._sync_gradient)
        self._refresh_gradient_stop_controls()

    def _sync_gradient(self, *_):
        self.gradient_mode = self._gradient_mode_var.get()
        self.gradient_spread = self._gradient_spread_var.get()
        self.gradient_opacity = max(
            1, self._int_var(self._gradient_opacity_var, 100)
        )
        self.gradient_blend = self._gradient_blend_var.get()

    def _on_gradient_stops_changed(self):
        """Editor edited its stops: mirror them and refresh the stop controls."""
        editor = getattr(self, "_gradient_editor", None)
        if editor is None:
            return
        stops = editor.get_stops()
        self.gradient_stops = stops
        if stops:
            self.gradient_from = stops[0][1]
            self.gradient_to = stops[-1][1]
        if getattr(self, "_gradient_preset_var", None) is not None:
            self._gradient_preset_var.set("custom")
        self._refresh_gradient_stop_controls()

    def _refresh_gradient_stop_controls(self):
        editor = getattr(self, "_gradient_editor", None)
        if editor is None:
            return
        stop = editor.selected_stop()
        if stop is None:
            return
        self._gradient_stop_pos_var.set(int(round(stop[0] * 100)))
        self._gradient_stop_alpha_var.set(int(stop[2]))

    def _pick_gradient_stop_color(self, current):
        chosen = dialogs.ask_color(
            self.winfo_toplevel(), current,
            title="Gradient Stop", on_preview=self._preview_gradient_stop,
        )
        return chosen

    def _preview_gradient_stop(self, color):
        editor = getattr(self, "_gradient_editor", None)
        if editor is not None:
            editor.set_selected_color(color)

    def _on_gradient_stop_pos(self, *_):
        editor = getattr(self, "_gradient_editor", None)
        if editor is None:
            return
        pos = max(0, min(100, self._int_var(self._gradient_stop_pos_var, 0)))
        editor.set_selected_pos(pos / 100.0)

    def _on_gradient_stop_alpha(self, *_):
        editor = getattr(self, "_gradient_editor", None)
        if editor is None:
            return
        alpha = max(0, min(255, self._int_var(self._gradient_stop_alpha_var, 255)))
        editor.set_selected_alpha(alpha)

    def _add_gradient_stop(self):
        editor = getattr(self, "_gradient_editor", None)
        if editor is not None:
            editor.add_stop()

    def _delete_gradient_stop(self):
        editor = getattr(self, "_gradient_editor", None)
        if editor is not None:
            editor.delete_selected()

    def _apply_gradient_preset(self):
        editor = getattr(self, "_gradient_editor", None)
        if editor is None:
            return
        key = self._gradient_preset_var.get()
        stops = gradient.preset_stops(key, self.gradient_from, self.gradient_to)
        if not stops:
            return
        editor.set_stops(stops)
        editor.set_selected(0)
        self.gradient_stops = editor.get_stops()
        self.gradient_from = self.gradient_stops[0][1]
        self.gradient_to = self.gradient_stops[-1][1]
        self._refresh_gradient_stop_controls()

    # --------------------------------------------------------- shape settings
    def _settings_for_shape(self):
        self._shape_kind_var = tk.StringVar(value=self.shape_kind)
        self._shape_mode_var = tk.StringVar(value=self.shape_mode)
        self._shape_opacity_var = tk.IntVar(value=self.shape_opacity)
        self._shape_blend_var = tk.StringVar(value=self.shape_blend)
        self._shape_radius_var = tk.IntVar(value=self.shape_radius)
        self._shape_aa_var = tk.BooleanVar(value=self.shape_antialias)
        # Drop stale swatch handles so a mode change cannot restyle a widget
        # that the rebuild destroyed.
        self._shape_fill_btn = None
        self._shape_stroke_btn = None
        self._menu_row(self.settings, "Shape", self._shape_kind_var,
                       shape.SHAPES, self._sync_shape)
        self._menu_row(self.settings, "Draw", self._shape_mode_var,
                       [("fill", "Fill"), ("stroke", "Stroke"),
                        ("both", "Both")],
                       self._on_shape_mode)
        self._shape_swatches()
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._slider_row(self.settings, "Corners", self._shape_radius_var,
                         0, 200, self._sync_shape)
        self._slider_row(self.settings, "Opacity", self._shape_opacity_var,
                         1, 100, self._sync_shape)
        self._menu_row(self.settings, "Blend", self._shape_blend_var,
                       brush.BLEND_MODES.items(), self._sync_shape)
        self._check_row(self.settings, "Anti-alias", self._shape_aa_var,
                        self._sync_shape)
        self.shape_new_layer_var = tk.BooleanVar(value=self.shape_new_layer)
        tk.Checkbutton(
            self.settings, text="Create as new layer", anchor="w",
            variable=self.shape_new_layer_var, command=self._on_shape_new_layer,
        ).pack(fill=tk.X, pady=(4, 0))

    def _shape_swatches(self):
        """The Shape tool's colour boxes, shown below the Draw control.

        Fill shows for the fill/both modes and Stroke for stroke/both, so the
        ``both`` mode shows the pair side by side.
        """
        mode = self.shape_mode
        show_fill = mode in ("fill", "both")
        show_stroke = mode in ("stroke", "both")
        if show_fill or show_stroke:
            row = self._swatch_row(self.settings)
            if show_fill:
                self._shape_fill_btn = tk.Button(
                    row, width=3,
                    command=lambda: self._choose_shape_paint("fill"),
                )
                theme.skip(self._shape_fill_btn)
                self._shape_fill_btn.pack(side=tk.LEFT, padx=(0, 6))
            if show_stroke:
                self._shape_stroke_btn = tk.Button(
                    row, width=3,
                    command=lambda: self._choose_shape_paint("stroke"),
                )
                theme.skip(self._shape_stroke_btn)
                self._shape_stroke_btn.pack(side=tk.LEFT, padx=(0, 6))
        self._apply_shape_swatches()

    def _sync_shape(self, *_):
        self.shape_kind = self._shape_kind_var.get()
        self.shape_mode = self._shape_mode_var.get()
        self.shape_opacity = max(1, self._int_var(self._shape_opacity_var, 100))
        self.shape_blend = self._shape_blend_var.get()
        self.shape_radius = max(0, self._int_var(self._shape_radius_var, 0))
        self.shape_antialias = bool(self._shape_aa_var.get())

    def _on_shape_mode(self):
        self._sync_shape()
        self._render_settings("shape")

    def _apply_shape_swatches(self):
        fill = self.shape_fill_color or self._current_color()
        stroke = self.shape_stroke_color or self._current_color()
        for name, color in (("_shape_fill_btn", fill),
                            ("_shape_stroke_btn", stroke)):
            button = getattr(self, name, None)
            if button is not None:
                button.config(bg=color, activebackground=color)

    def _choose_shape_paint(self, target):
        current = (self.shape_fill_color if target == "fill"
                   else self.shape_stroke_color) or self._current_color()

        def preview(color):
            if target == "fill":
                self.shape_fill_color = color
            else:
                self.shape_stroke_color = color
            self._apply_shape_swatches()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), current, title="Shape Color",
            on_preview=preview, on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _on_shape_new_layer(self):
        self.shape_new_layer = bool(self.shape_new_layer_var.get())

    # --------------------------------------------------- clone/retouch settings
    def _settings_for_clone(self):
        self._clone_opacity_var = tk.IntVar(value=self.clone_opacity)
        self._clone_hardness_var = tk.IntVar(value=self.clone_hardness)
        self._clone_spacing_var = tk.IntVar(value=self.clone_spacing)
        tk.Label(self.settings, text="Alt-click to set the source, then paint.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._slider_row(self.settings, "Opacity", self._clone_opacity_var,
                         1, 100, self._sync_clone)
        self._slider_row(self.settings, "Hardness", self._clone_hardness_var,
                         0, 100, self._sync_clone)
        self._slider_row(self.settings, "Spacing", self._clone_spacing_var,
                         1, 200, self._sync_clone)
        tk.Button(
            self.settings, text="Clear Source", command=self._clear_clone_source,
        ).pack(fill=tk.X, pady=(4, 0))

    def _sync_clone(self, *_):
        self.clone_opacity = max(1, self._int_var(self._clone_opacity_var, 100))
        self.clone_hardness = max(0, self._int_var(self._clone_hardness_var, 100))
        self.clone_spacing = max(1, self._int_var(self._clone_spacing_var, 25))

    def _clear_clone_source(self):
        if self._on_clone_clear is not None:
            self._on_clone_clear()

    def _settings_for_smudge(self):
        self._smudge_strength_var = tk.IntVar(value=self.smudge_strength)
        self._smudge_hardness_var = tk.IntVar(value=self.smudge_hardness)
        self._smudge_spacing_var = tk.IntVar(value=self.smudge_spacing)
        tk.Label(self.settings, text="Drag to smear colours along the stroke.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._slider_row(self.settings, "Strength", self._smudge_strength_var,
                         1, 100, self._sync_smudge)
        self._slider_row(self.settings, "Hardness", self._smudge_hardness_var,
                         0, 100, self._sync_smudge)
        self._slider_row(self.settings, "Spacing", self._smudge_spacing_var,
                         1, 200, self._sync_smudge)

    def _sync_smudge(self, *_):
        self.smudge_strength = max(1, self._int_var(self._smudge_strength_var, 60))
        self.smudge_hardness = max(0, self._int_var(self._smudge_hardness_var, 50))
        self.smudge_spacing = max(1, self._int_var(self._smudge_spacing_var, 25))

    def _settings_for_dodge(self):
        self._dodge_strength_var = tk.IntVar(value=self.dodge_strength)
        self._dodge_hardness_var = tk.IntVar(value=self.dodge_hardness)
        self._dodge_spacing_var = tk.IntVar(value=self.dodge_spacing)
        self._dodge_range_var = tk.StringVar(value=self.dodge_range)
        self._dodge_exposure_var = tk.IntVar(value=self.dodge_exposure)
        self._dodge_protect_var = tk.BooleanVar(value=self.dodge_protect)
        tk.Label(self.settings, text="Drag to lighten the pixels under the brush.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._menu_row(self.settings, "Range", self._dodge_range_var,
                       brush.TONAL_RANGES.items(), self._sync_dodge)
        self._slider_row(self.settings, "Exposure", self._dodge_exposure_var,
                         1, 100, self._sync_dodge)
        self._slider_row(self.settings, "Strength", self._dodge_strength_var,
                         1, 100, self._sync_dodge)
        self._slider_row(self.settings, "Hardness", self._dodge_hardness_var,
                         0, 100, self._sync_dodge)
        self._slider_row(self.settings, "Spacing", self._dodge_spacing_var,
                         1, 200, self._sync_dodge)
        self._check_row(self.settings, "Protect Tones",
                        self._dodge_protect_var, self._sync_dodge)

    def _sync_dodge(self, *_):
        self.dodge_strength = max(1, self._int_var(self._dodge_strength_var, 50))
        self.dodge_hardness = max(0, self._int_var(self._dodge_hardness_var, 50))
        self.dodge_spacing = max(1, self._int_var(self._dodge_spacing_var, 25))
        self.dodge_range = self._dodge_range_var.get()
        self.dodge_exposure = max(1, self._int_var(self._dodge_exposure_var, 50))
        self.dodge_protect = bool(self._dodge_protect_var.get())

    def _settings_for_burn(self):
        self._burn_strength_var = tk.IntVar(value=self.burn_strength)
        self._burn_hardness_var = tk.IntVar(value=self.burn_hardness)
        self._burn_spacing_var = tk.IntVar(value=self.burn_spacing)
        self._burn_range_var = tk.StringVar(value=self.burn_range)
        self._burn_exposure_var = tk.IntVar(value=self.burn_exposure)
        self._burn_protect_var = tk.BooleanVar(value=self.burn_protect)
        tk.Label(self.settings, text="Drag to darken the pixels under the brush.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        self._menu_row(self.settings, "Range", self._burn_range_var,
                       brush.TONAL_RANGES.items(), self._sync_burn)
        self._slider_row(self.settings, "Exposure", self._burn_exposure_var,
                         1, 100, self._sync_burn)
        self._slider_row(self.settings, "Strength", self._burn_strength_var,
                         1, 100, self._sync_burn)
        self._slider_row(self.settings, "Hardness", self._burn_hardness_var,
                         0, 100, self._sync_burn)
        self._slider_row(self.settings, "Spacing", self._burn_spacing_var,
                         1, 200, self._sync_burn)
        self._check_row(self.settings, "Protect Tones",
                        self._burn_protect_var, self._sync_burn)

    def _sync_burn(self, *_):
        self.burn_strength = max(1, self._int_var(self._burn_strength_var, 50))
        self.burn_hardness = max(0, self._int_var(self._burn_hardness_var, 50))
        self.burn_spacing = max(1, self._int_var(self._burn_spacing_var, 25))
        self.burn_range = self._burn_range_var.get()
        self.burn_exposure = max(1, self._int_var(self._burn_exposure_var, 50))
        self.burn_protect = bool(self._burn_protect_var.get())

    # ----------------------------------------------------- selection settings
    def _settings_for_select(self):
        self._select_mode_var = tk.StringVar(value=self.select_mode)
        hints = {
            "rect": "Drag a rectangle to select.",
            "lasso": "Drag a freeform outline to select.",
            "wand": "Click to select a region by colour.",
            "segment": "Click an object to select it with AI segmentation.",
        }
        tk.Label(self.settings,
                 text=hints.get(self.select_mode, hints["rect"]),
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        self._menu_row(self.settings, "Mode", self._select_mode_var,
                       self.SELECT_MODES, self._sync_select_mode)
        if self.select_mode == "rect":
            self._select_radius_var = tk.IntVar(value=self.select_radius)
            self._slider_row(self.settings, "Corners", self._select_radius_var,
                             0, 200, self._sync_select_radius)
        if self.select_mode == "wand":
            self._wand_tolerance_var = tk.IntVar(value=self.wand_tolerance)
            self._wand_contiguous_var = tk.BooleanVar(value=self.wand_contiguous)
            self._wand_sample_var = tk.StringVar(value=self.wand_sample)
            self._slider_row(self.settings, "Tolerance",
                             self._wand_tolerance_var, 0, 255, self._sync_wand)
            self._menu_row(self.settings, "Sample", self._wand_sample_var,
                           [("layer", "Active layer"), ("image", "All layers")],
                           self._sync_wand)
            self._check_row(self.settings, "Contiguous",
                            self._wand_contiguous_var, self._sync_wand)
        if self.select_mode == "segment":
            models = self._segment_models()
            if self.segment_model not in models:
                self.segment_model = models[0] if models else None
            self._segment_model_var = tk.StringVar(
                value=self.segment_model or "")
            self._segment_sample_var = tk.StringVar(value=self.segment_sample)
            if models:
                self._menu_row(self.settings, "Model", self._segment_model_var,
                               [(name, name) for name in models],
                               self._sync_segment)
            else:
                tk.Label(
                    self.settings,
                    text=("No segmentation model found. Add a SAM ONNX "
                          "'<name>_encoder.onnx' and '<name>_decoder.onnx' pair "
                          "to the segmentation folder."),
                    anchor="w", fg="#888888", wraplength=210, justify="left",
                ).pack(fill=tk.X)
            self._menu_row(self.settings, "Sample", self._segment_sample_var,
                           [("image", "All layers"), ("layer", "Active layer")],
                           self._sync_segment)
        if self.select_mode not in ("wand", "segment"):
            self._select_transform_var = tk.BooleanVar(
                value=self.select_transform)
            tk.Checkbutton(
                self.settings, text="Transform selection", anchor="w",
                variable=self._select_transform_var,
                command=self._sync_select_transform,
            ).pack(fill=tk.X, pady=(4, 0))

    @staticmethod
    def _segment_models():
        """Labels of the SAM ONNX models available to the Segment tool.

        Imported lazily so building the panel does not pull ``onnxruntime`` in
        when the tool is never used; a missing engine reports no models.
        """
        try:
            from engine import segmentation
            return segmentation.list_models()
        except Exception:
            return []

    def _sync_select_mode(self):
        mode = self._select_mode_var.get()
        if mode == self.select_mode:
            return
        self.select_mode = mode
        if mode in ("wand", "segment"):
            self.select_transform = False
        # Rebuilding the panel destroys the mode combobox, so defer the rebuild
        # until its selection event has finished dispatching.
        self.after_idle(self._rebuild_select_settings)

    def _rebuild_select_settings(self):
        self._render_settings("select")
        self._on_mode_change()

    def _sync_select_radius(self, *_):
        self.select_radius = max(0, self._int_var(self._select_radius_var, 0))
        if self._on_select_radius is not None:
            self._on_select_radius()

    def _sync_select_transform(self):
        self.select_transform = bool(self._select_transform_var.get())
        # Rebuilding the panel resets the toggle's own hint and refreshes the
        # gizmo via the tool-change hook, so defer until the click finishes.
        self.after_idle(self._rebuild_select_settings)

    def _sync_wand(self, *_):
        self.wand_tolerance = max(
            0, min(255, self._int_var(self._wand_tolerance_var, 32))
        )
        self.wand_contiguous = bool(self._wand_contiguous_var.get())
        self.wand_sample = self._wand_sample_var.get()

    def _sync_segment(self):
        """Read the Segment tool's model and sample settings from the panel."""
        model = self._segment_model_var.get()
        self.segment_model = model or None
        self.segment_sample = self._segment_sample_var.get()

    # ------------------------------------------------------------- arrow tool
    def _settings_for_arrow(self):
        tk.Label(self.settings,
                 text="No tool selected. Pick a tool to start drawing.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)

    # ----------------------------------------------------------- layout tool
    def _layout_family_value(self):
        return self._get_layout_family() if self._get_layout_family else None

    def layout_supported(self):
        return self._layout_family_value() in LAYOUT_FAMILIES

    def set_layout_family(self, family):
        """Rebuild the Layout Control editor when the model family changes."""
        if family == self._layout_family:
            return
        self._layout_family = family
        self._build_layout_editor()
        # The panel's switch is the only way to arm the canvas tool, so if it is
        # already on when the model family changes, arm the tool now.
        if self.layout_model.enabled and not self.layout_tool:
            self.select("layout")

    def set_layout_parent(self, parent):
        """Attach the Layout Control panel's content and build its editor."""
        self._layout_parent = parent
        self._layout_family = self._layout_family_value()
        self._build_layout_editor()

    def _build_layout_editor(self):
        """(Re)build the Layout Control panel's editor over the model."""
        if self._layout_parent is None:
            return
        for child in self._layout_parent.winfo_children():
            child.destroy()
        self._layout_editor = LayoutEditor(
            self._layout_parent, self.layout_model,
            supported=self.layout_supported(),
            family=self._layout_family,
            on_capture=self._on_layout_capture,
            on_change=self._on_layout_change,
            on_enabled=self._on_layout_enabled,
        )
        self._layout_editor.pack(fill=tk.X)

    def refresh_layout_settings(self):
        """Rebuild the Layout Control editor from the model."""
        if self._layout_parent is not None:
            self._build_layout_editor()
        elif self.layout_tool:
            self._render_settings("layout")

    def _on_layout_enabled(self, enabled):
        """Activate the canvas Layout tool while the panel's switch is on."""
        if enabled:
            if not self.layout_tool:
                self.select("layout")
        elif self.layout_tool:
            self.select("brush")

    def _settings_for_layout(self):
        tk.Label(self.settings,
                 text="Layout boxes are edited in the Layout Control panel; "
                      "draw a box on the canvas while it is enabled.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)

    # ------------------------------------------------------------ paste tool
    def _settings_for_paste(self):
        tk.Label(self.settings,
                 text="Paste the clipboard onto a new layer, the active layer, "
                      "or into the canvas assets.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        self._paste_setting(self.settings)
        tk.Button(
            self.settings, text="Paste", command=self._do_paste,
        ).pack(fill=tk.X, pady=(6, 0))

    def _do_paste(self):
        if self._on_paste is not None:
            self._on_paste()

    def _paste_setting(self, parent):
        """The shared paste options for the paste tool."""
        self.paste_dest_var = tk.StringVar(value=self.paste_dest)
        for value, label in (
            ("new_layer", "Paste as new layer"),
            ("selected_layer", "Paste in selected layer"),
            ("asset", "Paste in assets"),
        ):
            tk.Radiobutton(
                parent, text=label, anchor="w", value=value,
                variable=self.paste_dest_var, command=self._on_paste_dest,
            ).pack(fill=tk.X, pady=(4, 0))
        self.paste_fit_canvas_var = tk.BooleanVar(value=self.paste_fit_canvas)
        tk.Checkbutton(
            parent, text="Fit to canvas", anchor="w",
            variable=self.paste_fit_canvas_var,
            command=self._on_paste_fit_canvas,
        ).pack(fill=tk.X)
        self.paste_ask_var = tk.BooleanVar(value=self.paste_ask)
        tk.Checkbutton(
            parent, text="Ask before pasting", anchor="w",
            variable=self.paste_ask_var,
            command=self._on_paste_ask,
        ).pack(fill=tk.X)

    def _on_paste_dest(self):
        self.paste_dest = self.paste_dest_var.get()

    def _on_paste_fit_canvas(self):
        self.paste_fit_canvas = bool(self.paste_fit_canvas_var.get())

    def _on_paste_ask(self):
        self.paste_ask = bool(self.paste_ask_var.get())

    # ---------------------------------------------------------- crop settings
    def _settings_for_crop(self):
        tk.Label(self.settings,
                 text="Drag a rectangle, then drag its handles or edges to "
                      "adjust (or Enter) to apply.",
                 anchor="w", fg="#888888", wraplength=210,
                 justify="left").pack(fill=tk.X)
        tk.Button(
            self.settings, text="Apply Crop", command=self._apply_crop,
        ).pack(fill=tk.X, pady=(6, 0))
        tk.Button(
            self.settings, text="Reset", command=self._reset_crop,
        ).pack(fill=tk.X, pady=(2, 0))

    def _apply_crop(self):
        if self._on_crop_action is not None:
            self._on_crop_action()

    def _reset_crop(self):
        if self._on_crop_reset is not None:
            self._on_crop_reset()

    def _settings_for_mask(self):
        self._mask_value_var = tk.IntVar(value=self.mask_value)
        self._mask_erase_var = tk.BooleanVar(value=self.mask_erase)
        self._mask_feather_var = tk.IntVar(value=4)

        # The mask paints with the active brush's stamping settings (hardness,
        # flow, opacity, spacing, tip, pattern, grain, dynamics), so only the
        # mask-specific controls live here; the rest are edited in the Brush
        # tool and the Brushes panel.
        tk.Label(
            self.settings,
            text=("Paints with the active brush. Choose a preset in the "
                  "Brushes panel; edit its stamping in the Brush tool."),
            anchor="w", fg="#888888", wraplength=210, justify="left",
        ).pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X)
        self._size_row(self.settings)
        value_row = tk.Frame(self.settings)
        value_row.pack(fill=tk.X, pady=(3, 0))
        tk.Label(value_row, text="Paint", width=9, anchor="w").pack(side=tk.LEFT)
        for value, label in ((255, "Reveal"), (0, "Hide"), (128, "Soft")):
            tk.Radiobutton(
                value_row, text=label, value=value, variable=self._mask_value_var,
                command=self._sync_mask, indicatoron=False, width=6,
            ).pack(side=tk.LEFT, padx=(0, 2))
        self._slider_row(self.settings, "Value", self._mask_value_var,
                         0, 255, self._sync_mask)
        self._check_row(self.settings, "Erase paint", self._mask_erase_var,
                        self._sync_mask)
        self._mask_overlay_btn = tk.Button(
            self.settings, width=6, command=self._choose_mask_overlay,
        )
        self._color_row(self.settings, "Overlay", self._mask_overlay_btn)
        self._apply_mask_swatch()
        action_row = tk.Frame(self.settings)
        action_row.pack(fill=tk.X, pady=(4, 0))
        tk.Button(action_row, text="Swap", command=self._mask_swap).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(action_row, text="Invert", command=self._mask_invert).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        tk.Button(action_row, text="Reset", command=self._mask_reset).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        feather_row = tk.Frame(self.settings)
        feather_row.pack(fill=tk.X, pady=(3, 0))
        tk.Label(feather_row, text="Feather", width=9, anchor="w").pack(side=tk.LEFT)
        tk.Spinbox(
            feather_row, from_=1, to=60, width=4,
            textvariable=self._mask_feather_var,
        ).pack(side=tk.LEFT)
        tk.Button(feather_row, text="Blur", command=self._mask_feather).pack(
            side=tk.LEFT, padx=(4, 0))

    def _apply_mask_swatch(self):
        button = getattr(self, "_mask_overlay_btn", None)
        if button is not None:
            button.config(bg=self.mask_overlay, activebackground=self.mask_overlay)

    def _sync_mask(self, *_):
        self.mask_value = max(0, min(255, self._int_var(self._mask_value_var, 0)))
        self.mask_erase = bool(self._mask_erase_var.get())

    def _choose_mask_overlay(self):
        def preview(color):
            self.mask_overlay = color
            self._apply_mask_swatch()
            if self._on_mask_overlay is not None:
                self._on_mask_overlay(color)

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), self.mask_overlay,
            title="Mask Overlay Color", on_preview=preview,
        )
        if chosen is not None:
            preview(chosen)

    def _mask_swap(self):
        current = self._int_var(self._mask_value_var, 0)
        self._mask_value_var.set(0 if current >= 128 else 255)
        self._sync_mask()

    def _mask_invert(self):
        if self._on_mask_action is not None:
            self._on_mask_action("invert")

    def _mask_reset(self):
        if self._on_mask_action is not None:
            self._on_mask_action("reset")

    def _mask_feather(self):
        if self._on_mask_action is not None:
            radius = max(1, self._int_var(self._mask_feather_var, 4))
            self._on_mask_action("feather", radius)

    def _settings_for_inpaint_mask(self):
        """The inpaint mask uses the same brush controls as a layer mask."""
        self._settings_for_mask()

    def _settings_for_path(self):
        self._path_mode_var = tk.StringVar(value=self.path_mode)
        self._path_cap_var = tk.StringVar(value=self.path_cap)
        self._path_join_var = tk.StringVar(value=self.path_join)
        self._path_outline_var = tk.IntVar(
            value=max(0, min(int(self.path_outline), PATH_OUTLINE_MAX))
        )
        self._path_feather_var = tk.IntVar(value=self.path_feather)
        self._path_opacity_var = tk.IntVar(value=self.path_opacity)
        self._path_handles_var = tk.BooleanVar(value=self.path_show_handles)
        self._path_aa_var = tk.BooleanVar(value=self.path_antialias)
        # Drop stale swatch handles so a mode change cannot restyle a widget
        # that the rebuild destroyed.
        self._path_fill_btn = None
        self._path_stroke_btn = None

        tk.Label(
            self.settings,
            text=("Drag a round handle to reshape; drag a square point to "
                  "move; right-click a point to delete."),
            anchor="w", fg="#888888", wraplength=210, justify="left",
        ).pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.settings, text="Output", anchor="w").pack(fill=tk.X)
        row = tk.Frame(self.settings)
        row.pack(fill=tk.X)
        for value, label in (("fill", "Fill"), ("stroke", "Stroke"),
                             ("both", "Both")):
            tk.Radiobutton(
                row, text=label, value=value, variable=self._path_mode_var,
                command=self._on_path_mode,
            ).pack(side=tk.LEFT, padx=(0, 6) if value != "both" else (0, 0))

        # Every mode-dependent row is built once here and shown or hidden in
        # place by :meth:`_path_apply_rows`, so switching Output never tears the
        # panel down and rebuilds it.
        self._path_swatch_row = self._swatch_row(self.settings)
        self._path_swatches()
        self._path_size_label = tk.Label(self.settings, text="Size", anchor="w")
        self._path_size_row = tk.Frame(self.settings)
        self._size_row(self._path_size_row)
        self._path_outline_row = tk.Frame(self.settings)
        self._slider_row(self._path_outline_row, "Outline",
                         self._path_outline_var, 0, PATH_OUTLINE_MAX,
                         self._sync_path)
        self._path_cap_row = tk.Frame(self.settings)
        self._menu_row(self._path_cap_row, "Cap", self._path_cap_var,
                       [("round", "Round"), ("butt", "Butt"),
                        ("square", "Square")], self._sync_path)
        self._path_join_row = tk.Frame(self.settings)
        self._menu_row(self._path_join_row, "Join", self._path_join_var,
                       [("round", "Round"), ("bevel", "Bevel")],
                       self._sync_path)

        # A zero-height marker the conditional rows are packed in front of, so
        # re-showing them keeps them above the fixed controls below.
        self._path_rows_anchor = tk.Frame(self.settings, height=0)
        self._path_rows_anchor.pack(fill=tk.X)
        self._path_apply_rows()

        self._slider_row(self.settings, "Feather", self._path_feather_var,
                         0, 60, self._sync_path)
        self._slider_row(self.settings, "Opacity", self._path_opacity_var,
                         1, 100, self._sync_path)
        self._check_row(self.settings, "Anti-alias", self._path_aa_var,
                        self._sync_path)
        self._check_row(self.settings, "Show handles", self._path_handles_var,
                        self._on_path_handles)
        self.path_new_layer_var = tk.BooleanVar(value=self.path_new_layer)
        tk.Checkbutton(
            self.settings, text="Create as new layer", anchor="w",
            variable=self.path_new_layer_var, command=self._on_path_new_layer,
        ).pack(fill=tk.X, pady=(4, 0))
        tk.Button(
            self.settings, text="Finish Path", command=self._finish_path,
        ).pack(fill=tk.X, pady=(6, 0))

    # ---------------------------------------------------------- vector settings
    _VECTOR_HINTS = {
        "draw": "Draw on a vector layer. The stroke is kept as editable "
                "geometry; use Edit to reshape it.",
        "edit": "Click a line to select it. Shift/Ctrl-click or drag a marquee "
                "selects several; drag to move, or the corner handles to scale "
                "and the top handle to rotate. Double-click adds a point, "
                "right-click deletes one.",
        "width": "Drag across lines to correct their width (Adjust: Set, "
                 "Thicken or Narrow).",
        "erase": "Drag to erase; Touch removes covered points, Whole line "
                 "removes the line it touches.",
    }

    def _settings_for_vector(self):
        self._vector_mode_var = tk.StringVar(value=self.vector_mode)
        self._vector_anchors_var = tk.BooleanVar(value=self.vector_show_anchors)
        self._vector_opacity_var = tk.IntVar(value=self.vector_opacity)
        self._vector_cap_var = tk.StringVar(value=self.vector_cap)
        self._vector_join_var = tk.StringVar(value=self.vector_join)
        self._vector_aa_var = tk.BooleanVar(value=self.vector_antialias)
        self._vector_smooth_var = tk.IntVar(value=self.vector_smooth)
        self._vector_detail_var = tk.IntVar(value=self.vector_detail)
        self._vector_snap_var = tk.BooleanVar(value=self.vector_snap)
        self._vector_anchor_size_var = tk.IntVar(value=self.vector_anchor_size)
        self._vector_width_mode_var = tk.StringVar(value=self.vector_width_mode)
        self._vector_erase_mode_var = tk.StringVar(value=self.vector_erase_mode)
        self._vector_scale_width_var = tk.BooleanVar(value=self.vector_scale_width)
        self._vector_prop_width_var = tk.IntVar(value=vector.DEFAULT_WIDTH)
        self._vector_prop_opacity_var = tk.IntVar(value=self.vector_opacity)
        self._vector_prop_color = None
        self._vector_prop_fill = None
        self._vector_syncing = False

        self._vector_hint = tk.Label(
            self.settings, text="", anchor="w", fg="#888888",
            wraplength=210, justify="left",
        )
        self._vector_hint.pack(fill=tk.X, pady=(0, 4))
        tk.Label(self.settings, text="Mode", anchor="w").pack(fill=tk.X)
        row = tk.Frame(self.settings)
        row.pack(fill=tk.X)
        for value, label in (("draw", "Draw"), ("edit", "Edit"),
                             ("width", "Width"), ("erase", "Erase")):
            tk.Radiobutton(
                row, text=label, value=value, variable=self._vector_mode_var,
                command=self._on_vector_mode,
            ).pack(side=tk.LEFT, padx=(0, 4))

        # Every mode-dependent row is built once here and shown or hidden in
        # place by :meth:`_vector_apply_rows`, so switching Mode updates the
        # panel without tearing it down and rebuilding it (which flashed and
        # dropped focus).
        self._vector_size = tk.Frame(self.settings)
        tk.Label(self._vector_size, text="Size", anchor="w").pack(
            fill=tk.X, pady=(4, 0))
        self._size_row(self._vector_size)

        self._vector_width_frame = tk.Frame(self.settings)
        self._menu_row(self._vector_width_frame, "Adjust",
                       self._vector_width_mode_var,
                       [("set", "Set"), ("thicken", "Thicken"),
                        ("narrow", "Narrow")], self._on_vector_correct)

        self._vector_erase_frame = tk.Frame(self.settings)
        self._menu_row(self._vector_erase_frame, "Erase",
                       self._vector_erase_mode_var,
                       [("touch", "Touch"), ("whole", "Whole line")],
                       self._on_vector_correct)

        self._vector_draw = tk.Frame(self.settings)
        swatch = self._swatch_row(self._vector_draw)
        tk.Label(swatch, text="Color", width=9, anchor="w").pack(side=tk.LEFT)
        self._vector_color_btn = tk.Button(
            swatch, width=3, command=self._choose_vector_color)
        theme.skip(self._vector_color_btn)
        self._vector_color_btn.pack(side=tk.LEFT)
        self._vector_follow_btn = tk.Button(
            swatch, text="Follow main", command=self._vector_follow_main)
        self._vector_follow_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._slider_row(self._vector_draw, "Opacity",
                         self._vector_opacity_var, 1, 100, self._on_vector_draw)
        self._menu_row(self._vector_draw, "Cap", self._vector_cap_var,
                       [("round", "Round"), ("butt", "Butt"),
                        ("square", "Square")], self._on_vector_draw)
        self._menu_row(self._vector_draw, "Join", self._vector_join_var,
                       [("round", "Round"), ("miter", "Miter")],
                       self._on_vector_draw)
        self._slider_row(self._vector_draw, "Smooth", self._vector_smooth_var,
                         0, 100, self._on_vector_draw)
        self._slider_row(self._vector_draw, "Detail", self._vector_detail_var,
                         1, 60, self._on_vector_draw)
        self._check_row(self._vector_draw, "Anti-alias", self._vector_aa_var,
                        self._on_vector_draw)
        self._vector_color_note = tk.Label(
            self._vector_draw, text="", anchor="w", fg="#888888")
        self._vector_color_note.pack(fill=tk.X, pady=(4, 0))

        self._vector_edit = tk.Frame(self.settings)
        self._check_row(self._vector_edit, "Show anchors",
                        self._vector_anchors_var, self._on_vector_anchors)
        self._check_row(self._vector_edit, "Snap anchors",
                        self._vector_snap_var, self._on_vector_edit)
        self._check_row(self._vector_edit, "Scale width",
                        self._vector_scale_width_var, self._on_vector_correct)
        self._slider_row(self._vector_edit, "Handle",
                         self._vector_anchor_size_var, 2, 12,
                         self._on_vector_edit)
        actions = tk.Frame(self._vector_edit)
        actions.pack(fill=tk.X, pady=(4, 0))
        for label, action in (("All", "select_all"), ("None", "clear"),
                              ("Delete", "delete")):
            tk.Button(
                actions, text=label,
                command=lambda a=action: self._vector_action(a),
            ).pack(side=tk.LEFT, padx=(0, 3))
        shape_actions = tk.Frame(self._vector_edit)
        shape_actions.pack(fill=tk.X, pady=(3, 0))
        for label, action in (("Corner", "corner"), ("Smooth", "smooth"),
                              ("Split", "split"), ("Close", "toggle_closed")):
            tk.Button(
                shape_actions, text=label,
                command=lambda a=action: self._vector_action(a),
            ).pack(side=tk.LEFT, padx=(0, 3))
        shape_actions2 = tk.Frame(self._vector_edit)
        shape_actions2.pack(fill=tk.X, pady=(3, 0))
        for label, action in (("Simplify", "simplify"), ("Connect", "connect")):
            tk.Button(
                shape_actions2, text=label,
                command=lambda a=action: self._vector_action(a),
            ).pack(side=tk.LEFT, padx=(0, 3))
        tk.Label(self._vector_edit, text="Lines", anchor="w").pack(
            fill=tk.X, pady=(6, 0))
        list_row = tk.Frame(self._vector_edit)
        list_row.pack(fill=tk.X)
        self._vector_list = tk.Listbox(
            list_row, height=6, exportselection=False, activestyle="none")
        self._vector_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._vector_list.bind("<<ListboxSelect>>", self._on_vector_list_select)
        scroll = tk.Scrollbar(list_row, command=self._vector_list.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._vector_list.config(yscrollcommand=scroll.set)
        tk.Label(self._vector_edit, text="Selected line", anchor="w").pack(
            fill=tk.X, pady=(6, 0))
        self._slider_row(self._vector_edit, "Width",
                         self._vector_prop_width_var, vector.MIN_WIDTH,
                         min(vector.MAX_WIDTH, BRUSH_SLIDER_MAX),
                         self._on_vector_prop)
        self._slider_row(self._vector_edit, "Opacity",
                         self._vector_prop_opacity_var, 1, 100,
                         self._on_vector_prop)
        prop_row = self._swatch_row(self._vector_edit)
        tk.Label(prop_row, text="Color", width=9, anchor="w").pack(side=tk.LEFT)
        self._vector_prop_btn = tk.Button(
            prop_row, width=3, command=self._choose_vector_prop_color)
        theme.skip(self._vector_prop_btn)
        self._vector_prop_btn.pack(side=tk.LEFT)
        self._vector_prop_fill_btn = tk.Button(
            prop_row, width=3, command=self._choose_vector_fill_color)
        theme.skip(self._vector_prop_fill_btn)
        self._vector_prop_fill_btn.pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(
            prop_row, text="Apply",
            command=lambda: self._vector_action("apply"),
        ).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(
            prop_row, text="No fill",
            command=lambda: self._vector_action("clear_fill"),
        ).pack(side=tk.LEFT, padx=(4, 0))

        # A zero-height marker the conditional rows are packed in front of, so
        # re-showing them keeps them above any controls packed after this.
        self._vector_rows_anchor = tk.Frame(self.settings, height=0)
        self._vector_rows_anchor.pack(fill=tk.X)
        self._vector_apply_rows()
        self._apply_vector_swatches()

    def _vector_apply_rows(self):
        """Show only the rows the current Vector Mode uses, in place."""
        mode = self.vector_mode
        self._vector_hint.config(text=self._VECTOR_HINTS.get(mode, ""))
        self._vector_size.pack_forget()
        self._vector_width_frame.pack_forget()
        self._vector_erase_frame.pack_forget()
        self._vector_draw.pack_forget()
        self._vector_edit.pack_forget()

        def show(widget, **options):
            widget.pack(fill=tk.X, before=self._vector_rows_anchor, **options)

        if mode in ("draw", "width", "erase"):
            show(self._vector_size)
        if mode == "width":
            show(self._vector_width_frame)
        elif mode == "erase":
            show(self._vector_erase_frame)
        if mode == "draw":
            show(self._vector_draw)
        elif mode == "edit":
            show(self._vector_edit)
            self._vector_action("refresh")

    def _on_vector_mode(self):
        self.vector_mode = self._vector_mode_var.get()
        self._vector_apply_rows()
        self._refresh_vector_overlay()
        self._on_mode_change()

    def _on_vector_anchors(self):
        self.vector_show_anchors = bool(self._vector_anchors_var.get())
        self._refresh_vector_overlay()
        self._on_mode_change()

    def _on_vector_draw(self, *_):
        self.vector_opacity = max(
            1, min(100, self._int_var(self._vector_opacity_var, 100)))
        self.vector_cap = self._vector_cap_var.get()
        self.vector_join = self._vector_join_var.get()
        self.vector_smooth = max(
            0, min(100, self._int_var(self._vector_smooth_var, 0)))
        self.vector_detail = max(
            1, min(60, self._int_var(self._vector_detail_var, 15)))
        self.vector_antialias = bool(self._vector_aa_var.get())

    def _on_vector_edit(self, *_):
        self.vector_snap = bool(self._vector_snap_var.get())
        self.vector_anchor_size = max(
            2, min(12, self._int_var(self._vector_anchor_size_var, 4)))
        self._refresh_vector_overlay()

    def _on_vector_correct(self, *_):
        self.vector_width_mode = self._vector_width_mode_var.get()
        self.vector_erase_mode = self._vector_erase_mode_var.get()
        self.vector_scale_width = bool(self._vector_scale_width_var.get())

    def _on_vector_prop(self, *_):
        pass

    def _refresh_vector_overlay(self):
        if self._on_vector_settings is not None:
            self._on_vector_settings()

    def _effective_vector_color(self):
        return self.vector_color or self._current_color()

    def _apply_vector_swatches(self):
        color = self._effective_vector_color()

        def alive(widget):
            if widget is None:
                return False
            try:
                return bool(widget.winfo_exists())
            except tk.TclError:
                return False

        btn = getattr(self, "_vector_color_btn", None)
        if alive(btn):
            btn.config(bg=color, activebackground=color)
        note = getattr(self, "_vector_color_note", None)
        if alive(note):
            note.config(
                text=("New lines follow the main color."
                      if self.vector_color is None
                      else f"New lines use {self.vector_color}."))
        prop = self._vector_prop_color or color
        prop_btn = getattr(self, "_vector_prop_btn", None)
        if alive(prop_btn):
            prop_btn.config(bg=prop, activebackground=prop)
        fill_btn = getattr(self, "_vector_prop_fill_btn", None)
        if alive(fill_btn):
            fill = self._vector_prop_fill
            fill_btn.config(
                bg=fill or "#f0f0f0", activebackground=fill or "#f0f0f0")

    def _choose_vector_color(self):
        def preview(color):
            self.vector_color = color
            self._apply_vector_swatches()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), self._effective_vector_color(),
            title="Line Color", on_preview=preview, on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _vector_follow_main(self):
        self.vector_color = None
        self._apply_vector_swatches()

    def _choose_vector_prop_color(self):
        def preview(color):
            self._vector_prop_color = color
            self._apply_vector_swatches()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(),
            self._vector_prop_color or self._effective_vector_color(),
            title="Line Color", on_preview=preview, on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _choose_vector_fill_color(self):
        def preview(color):
            self._vector_prop_fill = color
            self._apply_vector_swatches()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(),
            self._vector_prop_fill or "#ffffff",
            title="Fill Color", on_preview=preview, on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _on_vector_list_select(self, _event=None):
        if self._vector_syncing:
            return
        self._vector_action(
            "select", {"indices": list(self._vector_list.curselection())})

    def _vector_prop_values(self):
        return {
            "width": self._int_var(self._vector_prop_width_var,
                                   vector.DEFAULT_WIDTH),
            "opacity": self._int_var(self._vector_prop_opacity_var, 100),
            "color": self._vector_prop_color or self._effective_vector_color(),
            "fill": self._vector_prop_fill,
        }

    def _vector_action(self, action, data=None):
        if self._on_vector_action is not None:
            self._on_vector_action(action, data or self._vector_prop_values())

    def refresh_vector_panel(self, strokes=None, selection=None, primary=None):
        """Repopulate the Lines list and inspector from the controller state.

        Called by the controller (which owns the multi-selection) so the panel
        stays in step with on-canvas selection, marquee and transform changes.
        """
        widget = getattr(self, "_vector_list", None)
        if widget is None:
            return
        try:
            if not widget.winfo_exists():
                return
        except tk.TclError:
            return
        self._vector_syncing = True
        try:
            widget.delete(0, "end")
            for label in strokes or []:
                widget.insert("end", label)
            for index in selection or []:
                if 0 <= index < widget.size():
                    widget.selection_set(index)
            if primary is not None:
                self._vector_prop_width_var.set(
                    int(primary.get("width", vector.DEFAULT_WIDTH)))
                self._vector_prop_opacity_var.set(
                    int(round(float(primary.get("opacity", 1.0)) * 100.0)))
                self._vector_prop_color = primary.get("color")
                self._vector_prop_fill = primary.get("fill")
        finally:
            self._vector_syncing = False
        self._apply_vector_swatches()


    # ------------------------------------------------------- poses settings
    def _settings_for_poses(self):
        poses.set_drag_quality(self.poses_drag)
        self._pose_drag_var = tk.StringVar(value=self.poses_drag)
        self._pose_body_var = tk.StringVar(value=self.poses_body)
        self._pose_hand_var = tk.StringVar(value=self.poses_hand)
        self._pose_hand_side_var = tk.StringVar(value=self.poses_hand_side)
        self._pose_scale_var = tk.IntVar(value=int(self.poses_scale))
        self._pose_opacity_var = tk.IntVar(value=int(self.poses_opacity))
        self._pose_grid_var = tk.BooleanVar(value=self.poses_grid)
        tk.Label(
            self.settings,
            text=("Add a mannequin or primitive, then click a joint and drag to "
                  "pose that limb (drag a square hand/foot handle to solve the "
                  "whole limb). Click an object to cycle Move/Scale/Rotate, drag "
                  "off it to turn the scene in 3D (Shift tilts, Alt+wheel zooms, "
                  "Alt+middle-drag pans), and right-click a joint to reset its "
                  "bone."),
            anchor="w", fg="#888888", wraplength=210, justify="left",
        ).pack(fill=tk.X, pady=(0, 4))
        add_row = tk.Frame(self.settings)
        add_row.pack(fill=tk.X, pady=(0, 4))
        tk.Label(add_row, text="Add", anchor="w").pack(fill=tk.X)
        add_grid = tk.Frame(add_row)
        add_grid.pack(fill=tk.X)
        for column in range(3):
            add_grid.columnconfigure(column, weight=1, uniform="addshape")
        add_items = [("Mannequin", "add")] + [
            (poses.PRIMITIVE_LABELS[shape], "add:" + shape)
            for shape in poses.PRIMITIVES
        ]
        for index, (label, action) in enumerate(add_items):
            tk.Button(
                add_grid, text=label,
                command=lambda a=action: self._poses_action(a),
            ).grid(row=index // 3, column=index % 3, sticky="ew", padx=1, pady=1)
        self._menu_row(
            self.settings, "Body", self._pose_body_var,
            [(name, name.title()) for name in poses.BODY_ORDER],
            self._on_poses_body,
        )
        tk.Label(self.settings, text="Pose", anchor="w").pack(fill=tk.X)
        list_row = tk.Frame(self.settings)
        list_row.pack(fill=tk.X, pady=(1, 3))
        canvas = tk.Canvas(
            list_row, height=POSES_LIST_MAX_H, highlightthickness=0, bd=0)
        scroll = ArrowScrollbar(list_row, command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pose_list = tk.Frame(canvas)
        window = canvas.create_window((0, 0), window=pose_list, anchor="nw")

        def fit_list(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfigure(window, width=canvas.winfo_width())

        pose_list.bind("<Configure>", fit_list)
        canvas.bind("<Configure>", fit_list)
        self._bind_pose_wheel(canvas, canvas)
        for name in poses.POSE_ORDER:
            photo = self._pose_thumbnail(name)
            button = tk.Button(
                pose_list, text=name.replace("_", " ").title(),
                compound=tk.LEFT, anchor="w", bd=1, relief=tk.RIDGE,
                command=lambda n=name: self._poses_action("pose:" + n),
            )
            if photo is not None:
                button.config(image=photo)
            button.pack(fill=tk.X, padx=1, pady=1)
            self._bind_pose_wheel(button, canvas)
        self._menu_row(
            self.settings, "Hands", self._pose_hand_var,
            [(name, name.replace("_", " ").title()) for name in poses.HAND_ORDER],
            self._on_poses_hand,
        )
        self._menu_row(
            self.settings, "While posing", self._pose_drag_var,
            [("fast", "Fast (decimated)"), ("best", "Full quality")],
            self._on_poses_drag,
        )
        tk.Checkbutton(
            self.settings, text="3D ground grid", variable=self._pose_grid_var,
            command=self._on_poses_grid,
        ).pack(anchor="w", pady=(2, 0))
        hand_row = tk.Frame(self.settings)
        hand_row.pack(fill=tk.X, pady=(2, 0))
        for value, label in (("both", "Both"), ("left", "Left"), ("right", "Right")):
            tk.Radiobutton(
                hand_row, text=label, value=value,
                variable=self._pose_hand_side_var,
                command=self._on_poses_hand,
            ).pack(side=tk.LEFT, padx=(0, 4))
        self._slider_row(self.settings, "Height", self._pose_scale_var,
                         poses.MIN_SCALE, poses.MAX_SCALE,
                         self._on_poses_scale)
        self._slider_row(self.settings, "Opacity", self._pose_opacity_var,
                         1, 100, self._on_poses_opacity)
        # The scene's own camera zoom (document pixels per model unit), separate
        # from the 2D canvas zoom. Shown as a percentage of the default so it
        # reads like a dolly; Alt+wheel on the canvas does the same about the
        # cursor. Kept in sync with the active layer's scene.
        self._pose_scene_zoom_guard = False
        self._pose_scene_zoom_var = tk.IntVar(value=100)
        self._slider_row(self.settings, "Scene %", self._pose_scene_zoom_var,
                         POSE_SCENE_ZOOM_MIN, POSE_SCENE_ZOOM_MAX,
                         self._on_poses_scene_zoom)
        scene_actions = tk.Frame(self.settings)
        scene_actions.pack(fill=tk.X, pady=(3, 0))
        tk.Button(
            scene_actions, text="Reset Scene View", command=self._on_poses_reset_scene,
        ).pack(fill=tk.X)
        color_row = tk.Frame(self.settings)
        color_row.pack(fill=tk.X, pady=(3, 0))
        tk.Label(color_row, text="Color", width=9, anchor="w").pack(side=tk.LEFT)
        self._pose_color_btn = tk.Button(
            color_row, width=3, command=self._choose_poses_color,
        )
        theme.skip(self._pose_color_btn)
        self._pose_color_btn.pack(side=tk.LEFT)
        self._pose_color_btn.config(
            bg=self.poses_color, activebackground=self.poses_color,
        )
        actions = tk.Frame(self.settings)
        actions.pack(fill=tk.X, pady=(6, 0))
        for label, action in (("Add", "add"), ("Duplicate", "duplicate"),
                              ("Mirror", "mirror"), ("Reset Pose", "reset")):
            tk.Button(
                actions, text=label,
                command=lambda a=action: self._poses_action(a),
            ).pack(side=tk.LEFT, padx=(0, 3))
        tk.Button(
            self.settings, text="Delete",
            command=lambda: self._poses_action("delete"),
        ).pack(fill=tk.X, pady=(3, 0))
        tk.Label(
            self.settings, text=f"Renderer: {poses.renderer_name()}",
            anchor="w", fg="#888888",
        ).pack(fill=tk.X, pady=(4, 0))

    def _on_poses_grid(self, *_):
        self.poses_grid = bool(self._pose_grid_var.get())
        self._poses_action("grid")

    def _on_poses_body(self, *_):
        self.poses_body = self._pose_body_var.get()
        self._poses_action("body:" + self.poses_body)

    @staticmethod
    def _bind_pose_wheel(widget, canvas):
        """Scroll the pose list with Ctrl+wheel, leaving the plain wheel to the rail."""
        def on_wheel(event):
            canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
            return "break"

        widget.bind("<Control-MouseWheel>", on_wheel)

    def _pose_thumbnail(self, name, height=POSES_PREVIEW_H):
        """A small preview image of a pose preset, cached per body and size."""
        key = (name, self.poses_body, height)
        cached = self._pose_thumbs.get(key)
        if cached is not None:
            return cached
        spec = poses.new(0, 0, scale=86, body=self.poses_body, pose=None)
        if spec is None:
            return None
        poses.apply_pose(spec, name)
        rendered = poses.render_sprite(spec)
        if rendered is None:
            return None
        image = rendered[0]
        if image.height > height:
            width = max(1, round(image.width * height / image.height))
            image = image.resize((width, height), Image.LANCZOS)
        try:
            photo = ImageTk.PhotoImage(image)
        except Exception:
            return None
        self._pose_thumbs[key] = photo
        return photo

    def _on_poses_hand(self, *_):
        self.poses_hand = self._pose_hand_var.get()
        self.poses_hand_side = self._pose_hand_side_var.get()
        self._poses_action(f"hand:{self.poses_hand}:{self.poses_hand_side}")

    def _on_poses_drag(self, *_):
        self.poses_drag = self._pose_drag_var.get()
        poses.set_drag_quality(self.poses_drag)

    def _on_poses_scale(self, *_):
        self.poses_scale = max(
            poses.MIN_SCALE,
            min(poses.MAX_SCALE,
                self._int_var(self._pose_scale_var, poses.DEFAULT_SCALE)),
        )

    def _on_poses_opacity(self, *_):
        self.poses_opacity = max(
            1, min(100, self._int_var(self._pose_opacity_var, 100))
        )

    def _refresh_pose_scene_zoom(self, scene):
        """Point the Scene slider at ``scene`` without echoing a command back.

        Called by the drawing controller after the scene zoom changes (slider,
        Alt+wheel or a scene/layer switch), so the panel always shows the active
        layer's zoom.
        """
        var = getattr(self, "_pose_scene_zoom_var", None)
        if var is None or not isinstance(scene, dict):
            return
        percent = max(10, min(1000, scene_zoom_percent(scene)))
        if self._int_var(var, 100) == percent:
            return
        self._pose_scene_zoom_guard = True
        try:
            var.set(percent)
        finally:
            self._pose_scene_zoom_guard = False

    def _on_poses_scene_zoom(self, *_):
        if getattr(self, "_pose_scene_zoom_guard", False):
            return
        percent = max(10, min(1000, self._int_var(self._pose_scene_zoom_var, 100)))
        self._poses_action(f"scene_zoom:{percent}")

    def _on_poses_reset_scene(self):
        self._poses_action("reset_scene")

    def _choose_poses_color(self):
        current = self.poses_color or poses.DEFAULT_COLOR

        def preview(color):
            self.poses_color = color
            if getattr(self, "_pose_color_btn", None) is not None:
                self._pose_color_btn.config(bg=color, activebackground=color)

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), current, title="Poses Color",
            on_preview=preview, on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _poses_action(self, action):
        if self._on_poses_action is not None:
            self._on_poses_action(action)

    # ------------------------------------------------------ symmetry settings
    def _symmetry_rows(self):
        """The symmetry-guide block for painting tools' settings.

        The block is packed above the tool's "Advanced" toggle when it has one
        (see :meth:`_advanced_toggle`), so the guide controls stay visible
        rather than tucking away with the advanced rows.
        """
        self._symmetry_mode_var = tk.StringVar(value=self.symmetry_mode)
        self._symmetry_count_var = tk.IntVar(value=self.symmetry_count)
        self._symmetry_mirror_var = tk.BooleanVar(value=self.symmetry_mirror)
        block = tk.Frame(self.settings)
        tk.Frame(block, height=1, bd=0).pack(fill=tk.X, pady=(6, 0))
        self._menu_row(
            block, "Symmetry", self._symmetry_mode_var,
            [("off", "Off"), ("vertical", "Vertical"),
             ("horizontal", "Horizontal"), ("both", "Both"),
             ("radial", "Radial")],
            self._on_symmetry_change,
        )
        # The hint stays last; the radial-only controls live in their own frame
        # packed before it, so a mode change shows or hides them without
        # rebuilding the panel (which would destroy the combobox mid-selection).
        self._symmetry_hint = tk.Label(
            block, text="Mirrored about the canvas center.",
            anchor="w", fg="#888888",
        )
        self._symmetry_radial = tk.Frame(block)
        self._slider_row(
            self._symmetry_radial, "Spokes", self._symmetry_count_var,
            symmetry.RADIAL_MIN, symmetry.RADIAL_MAX, self._on_symmetry_change,
        )
        self._check_row(
            self._symmetry_radial, "Mirror spokes", self._symmetry_mirror_var,
            self._on_symmetry_change,
        )
        self._symmetry_hint.pack(fill=tk.X, pady=(2, 0))
        if self.symmetry_mode == "radial":
            self._symmetry_radial.pack(fill=tk.X, before=self._symmetry_hint)
        anchor = self._advanced_anchor
        if anchor is not None:
            block.pack(fill=tk.X, before=anchor)
        else:
            block.pack(fill=tk.X)

    def _on_symmetry_change(self, _event=None):
        self.symmetry_mode = self._symmetry_mode_var.get()
        try:
            self.symmetry_count = int(self._symmetry_count_var.get())
        except (tk.TclError, ValueError):
            pass
        self.symmetry_mirror = bool(self._symmetry_mirror_var.get())
        # Show the radial-only rows in place rather than rebuilding the panel, so
        # the combobox the user is interacting with survives the change.
        if self.symmetry_mode == "radial":
            self._symmetry_radial.pack(fill=tk.X, before=self._symmetry_hint)
        else:
            self._symmetry_radial.pack_forget()
        self._on_mode_change()

    # --------------------------------------------------------- liquify settings
    def _settings_for_liquify(self):
        self._liquify_mode_var = tk.StringVar(value=self.liquify_mode)
        self._liquify_strength_var = tk.IntVar(value=self.liquify_strength)
        tk.Label(
            self.settings, text="Drag to distort the active layer; release to "
                                "commit (one undo step).",
            anchor="w", fg="#888888", wraplength=210, justify="left",
        ).pack(fill=tk.X, pady=(0, 4))
        self._menu_row(
            self.settings, "Mode", self._liquify_mode_var,
            [("push", "Push"), ("expand", "Expand"), ("pinch", "Pinch"),
             ("twirl_cw", "Twirl CW"), ("twirl_ccw", "Twirl CCW")],
            self._on_liquify_change,
        )
        tk.Label(self.settings, text="Size", anchor="w").pack(fill=tk.X, pady=(4, 0))
        self._size_row(self.settings)
        self._slider_row(
            self.settings, "Strength", self._liquify_strength_var, 1, 100,
            self._on_liquify_change,
        )

    def _on_liquify_change(self, _event=None):
        self.liquify_mode = self._liquify_mode_var.get()
        self.liquify_strength = max(
            1, min(100, self._int_var(self._liquify_strength_var, 50))
        )
        self._on_mode_change()

    # ------------------------------------------------------------ warp settings
    def _settings_for_warp(self):
        self._warp_grid_var = tk.StringVar(
            value=f"{self.warp_cols}x{self.warp_rows}"
        )
        tk.Label(
            self.settings, text="Drag the grid points to warp the layer; Apply "
                                "keeps the result, Cancel restores it.",
            anchor="w", fg="#888888", wraplength=210, justify="left",
        ).pack(fill=tk.X, pady=(0, 4))
        self._menu_row(
            self.settings, "Grid", self._warp_grid_var,
            [("2x2", "2 x 2"), ("3x3", "3 x 3"), ("4x4", "4 x 4"),
             ("6x6", "6 x 6"), ("2x4", "2 x 4"), ("4x2", "4 x 2")],
            self._on_warp_grid,
        )
        row = tk.Frame(self.settings)
        row.pack(fill=tk.X, pady=(6, 0))
        for label, action in (("Reset", "reset"), ("Apply", "apply"),
                              ("Cancel", "cancel")):
            tk.Button(
                row, text=label, command=lambda a=action: self._warp_action(a),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

    def _on_warp_grid(self, _event=None):
        try:
            cols, rows = self._warp_grid_var.get().lower().split("x")
            self.warp_cols, self.warp_rows = int(cols), int(rows)
        except (ValueError, AttributeError):
            return
        self._warp_action("rebuild")

    def _warp_action(self, action):
        if self._on_warp_action is not None:
            self._on_warp_action(action)

    # ---------------------------------------------------------- puppet settings
    def _settings_for_puppet(self):
        self._puppet_expansion_var = tk.IntVar(value=self.puppet_expansion)
        self._puppet_mesh_var = tk.BooleanVar(value=self.puppet_show_mesh)
        tk.Label(
            self.settings, text="Tap the drawing to add a pin; drag a pin to "
                                "move it, drag its ring to rotate, drag empty "
                                "space to select; right-click deletes. "
                                "Unmoved pins anchor the mesh. A selection "
                                "limits the warp.",
            anchor="w", fg="#888888", wraplength=210, justify="left",
        ).pack(fill=tk.X, pady=(0, 4))
        self._slider_row(
            self.settings, "Expansion", self._puppet_expansion_var, 0, 200,
            self._on_puppet_settings,
        )
        self._check_row(
            self.settings, "Show mesh", self._puppet_mesh_var,
            self._on_puppet_settings,
        )
        order_row = tk.Frame(self.settings)
        order_row.pack(fill=tk.X, pady=(2, 0))
        for label, action in (("Bring Front", "front"), ("Send Back", "back")):
            tk.Button(
                order_row, text=label,
                command=lambda a=action: self._puppet_action(a),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        row = tk.Frame(self.settings)
        row.pack(fill=tk.X, pady=(6, 0))
        for label, action in (("Clear", "clear"), ("Reset", "reset"),
                              ("Apply", "apply"), ("Cancel", "cancel")):
            tk.Button(
                row, text=label, command=lambda a=action: self._puppet_action(a),
            ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1)

    def _on_puppet_settings(self, _event=None):
        self.puppet_expansion = max(
            0, min(200, self._int_var(self._puppet_expansion_var, 20))
        )
        self.puppet_show_mesh = bool(self._puppet_mesh_var.get())
        self._on_mode_change()

    def _puppet_action(self, action):
        if self._on_puppet_action is not None:
            self._on_puppet_action(action)

    def _path_swatches(self):
        """Build the Path tool's colour boxes, shown below the Output radios.

        Both boxes are created once against :attr:`_path_swatch_row`; the fill
        box shows for the fill/both modes and the stroke box for stroke/both, so
        ``both`` shows the pair side by side. :meth:`_path_apply_rows` packs the
        ones the current mode uses, so switching Output never rebuilds them.
        """
        row = self._path_swatch_row
        self._path_fill_btn = tk.Button(
            row, width=3, command=lambda: self._choose_path_paint("fill"),
        )
        theme.skip(self._path_fill_btn)
        self._path_stroke_btn = tk.Button(
            row, width=3, command=lambda: self._choose_path_paint("stroke"),
        )
        theme.skip(self._path_stroke_btn)
        self._apply_path_swatches()

    def _path_apply_rows(self):
        """Show only the rows the current Path Output mode uses.

        The rows are packed in front of :attr:`_path_rows_anchor` so they keep
        their place above the fixed controls, and nothing is destroyed, so
        switching Output updates the panel in place instead of re-rendering it.
        """
        mode = self.path_mode
        stroked = mode in ("stroke", "both")
        filled = mode in ("fill", "both")
        movable = (
            self._path_swatch_row, self._path_size_label,
            self._path_size_row, self._path_outline_row,
            self._path_cap_row, self._path_join_row,
        )
        for widget in movable:
            widget.pack_forget()
        self._path_fill_btn.pack_forget()
        self._path_stroke_btn.pack_forget()

        def show(widget, **options):
            widget.pack(fill=tk.X, before=self._path_rows_anchor, **options)

        if filled:
            self._path_fill_btn.pack(side=tk.LEFT, padx=(0, 6))
        if stroked:
            self._path_stroke_btn.pack(side=tk.LEFT, padx=(0, 6))
        if filled or stroked:
            show(self._path_swatch_row, pady=(3, 0))
        if stroked:
            show(self._path_size_label, pady=(4, 0))
            show(self._path_size_row)
        if filled and mode == "fill":
            show(self._path_outline_row)
        if stroked or mode == "fill":
            show(self._path_cap_row)
            show(self._path_join_row)

    def _on_path_mode(self):
        self.path_mode = self._path_mode_var.get()
        self._path_apply_rows()
        self._on_mode_change()
        self._refresh_path()

    def _apply_path_swatches(self):
        fill = self.path_fill_color or self._current_color()
        stroke = self.path_stroke_color or self._current_color()
        for name, color in (("_path_fill_btn", fill),
                            ("_path_stroke_btn", stroke)):
            button = getattr(self, name, None)
            if button is not None:
                button.config(bg=color, activebackground=color)

    def _choose_path_paint(self, target):
        current = (self.path_fill_color if target == "fill"
                   else self.path_stroke_color) or self._current_color()

        def preview(color):
            if target == "fill":
                self.path_fill_color = color
            else:
                self.path_stroke_color = color
            self._apply_path_swatches()
            self._refresh_path()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), current, title="Path Color",
            on_preview=preview, on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _on_path_handles(self):
        self.path_show_handles = bool(self._path_handles_var.get())
        self._refresh_path()

    def _settings_for_text(self):
        style = self._get_text_style() if self._get_text_style else {}
        self._text_family_var = tk.StringVar(
            value=style.get("family") or fonts.default_family()
        )
        self._text_size_var = tk.StringVar(
            value=str(style.get("size", textrender.DEFAULT_SIZE))
        )
        self._text_bold_var = tk.BooleanVar(value=bool(style.get("bold", False)))
        self._text_italic_var = tk.BooleanVar(value=bool(style.get("italic", False)))
        self._text_align_var = tk.StringVar(value=style.get("align", "left"))
        self._text_color = style.get("color") or "#000000"

        tk.Label(self.settings, text="Font", anchor="w").pack(fill=tk.X)
        self._font_picker = fontpicker.FontPicker(
            self.settings, family=self._text_family_var.get(),
            on_change=self._on_font_picked,
        )
        self._font_picker.pack(fill=tk.X)

        size_row = tk.Frame(self.settings)
        size_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(size_row, text="Size", anchor="w").pack(side=tk.LEFT)
        size_box = tk.Spinbox(
            size_row, from_=textrender.MIN_SIZE, to=textrender.MAX_SIZE, width=5,
            textvariable=self._text_size_var, command=self._text_style_changed,
        )
        size_box.pack(side=tk.LEFT, padx=(4, 0))
        size_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())
        tk.Checkbutton(
            size_row, text="B", variable=self._text_bold_var,
            command=self._text_style_changed,
        ).pack(side=tk.LEFT, padx=(6, 0))
        tk.Checkbutton(
            size_row, text="I", variable=self._text_italic_var,
            command=self._text_style_changed,
        ).pack(side=tk.LEFT)

        align_row = tk.Frame(self.settings)
        align_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(align_row, text="Align", anchor="w").pack(side=tk.LEFT)
        for value, label in (("left", "Left"), ("center", "Center"),
                             ("right", "Right")):
            tk.Radiobutton(
                align_row, text=label, value=value,
                variable=self._text_align_var, command=self._text_style_changed,
            ).pack(side=tk.LEFT, padx=(4, 0))

        # Wrapping only bites once a box width is set, which the side handles do.
        self._text_wrap_var = tk.BooleanVar(value=bool(style.get("wrap", True)))
        tk.Checkbutton(
            self.settings, text="Wrap to box (drag side handles)", anchor="w",
            variable=self._text_wrap_var, command=self._text_style_changed,
        ).pack(fill=tk.X, pady=(4, 0))

        color_row = tk.Frame(self.settings)
        color_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(color_row, text="Color", anchor="w").pack(side=tk.LEFT)
        self._text_color_btn = tk.Button(
            color_row, width=3, command=lambda: self._choose_text_paint("color"),
        )
        # The button doubles as a swatch, so its background is content, not chrome.
        theme.skip(self._text_color_btn)
        self._text_color_btn.pack(side=tk.LEFT, padx=(4, 0))

        self._text_line_var = tk.StringVar(
            value=str(int(round(style.get(
                "line_spacing", textrender.DEFAULT_LINE_SPACING) * 100)))
        )
        self._text_track_var = tk.StringVar(
            value=str(int(round(style.get("letter_spacing", 0.0) or 0.0)))
        )
        line_row = tk.Frame(self.settings)
        line_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(line_row, text="Line %", anchor="w").pack(side=tk.LEFT)
        line_box = tk.Spinbox(
            line_row, from_=-50, to=300, width=4,
            textvariable=self._text_line_var, command=self._text_style_changed,
        )
        line_box.pack(side=tk.LEFT, padx=(4, 0))
        line_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())
        tk.Label(line_row, text="Track", anchor="w").pack(side=tk.LEFT, padx=(8, 0))
        track_box = tk.Spinbox(
            line_row, from_=-5, to=50, width=4,
            textvariable=self._text_track_var, command=self._text_style_changed,
        )
        track_box.pack(side=tk.LEFT, padx=(4, 0))
        track_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())

        stroke = style.get("stroke") if isinstance(style.get("stroke"), dict) else {}
        self._text_stroke_var = tk.BooleanVar(value=bool(stroke))
        self._text_stroke_width_var = tk.StringVar(
            value=str(stroke.get("width", 2))
        )
        self._text_stroke_color = stroke.get("color") or "#000000"
        stroke_row = tk.Frame(self.settings)
        stroke_row.pack(fill=tk.X, pady=(4, 0))
        tk.Checkbutton(
            stroke_row, text="Outline", variable=self._text_stroke_var,
            command=self._text_style_changed,
        ).pack(side=tk.LEFT)
        tk.Label(stroke_row, text="W", anchor="w").pack(side=tk.LEFT, padx=(6, 0))
        stroke_width_box = tk.Spinbox(
            stroke_row, from_=1, to=20, width=3,
            textvariable=self._text_stroke_width_var,
            command=self._text_style_changed,
        )
        stroke_width_box.pack(side=tk.LEFT, padx=(2, 0))
        stroke_width_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())
        self._text_stroke_btn = tk.Button(
            stroke_row, width=3, command=lambda: self._choose_text_paint("stroke"),
        )
        theme.skip(self._text_stroke_btn)
        self._text_stroke_btn.pack(side=tk.LEFT, padx=(4, 0))

        shadow = style.get("shadow") if isinstance(style.get("shadow"), dict) else {}
        self._text_shadow_var = tk.BooleanVar(value=bool(shadow))
        self._text_shadow_blur_var = tk.StringVar(value=str(shadow.get("blur", 3)))
        self._text_shadow_dx_var = tk.StringVar(value=str(shadow.get("dx", 2)))
        self._text_shadow_dy_var = tk.StringVar(value=str(shadow.get("dy", 2)))
        self._text_shadow_color = shadow.get("color") or "#000000"
        shadow_row = tk.Frame(self.settings)
        shadow_row.pack(fill=tk.X, pady=(4, 0))
        tk.Checkbutton(
            shadow_row, text="Shadow", variable=self._text_shadow_var,
            command=self._text_style_changed,
        ).pack(side=tk.LEFT)
        tk.Label(shadow_row, text="Blur", anchor="w").pack(side=tk.LEFT, padx=(6, 0))
        blur_box = tk.Spinbox(
            shadow_row, from_=0, to=50, width=3,
            textvariable=self._text_shadow_blur_var,
            command=self._text_style_changed,
        )
        blur_box.pack(side=tk.LEFT, padx=(2, 0))
        blur_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())
        tk.Label(shadow_row, text="X", anchor="w").pack(side=tk.LEFT, padx=(6, 0))
        shadow_dx_box = tk.Spinbox(
            shadow_row, from_=-50, to=50, width=3,
            textvariable=self._text_shadow_dx_var,
            command=self._text_style_changed,
        )
        shadow_dx_box.pack(side=tk.LEFT, padx=(2, 0))
        shadow_dx_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())
        tk.Label(shadow_row, text="Y", anchor="w").pack(side=tk.LEFT, padx=(6, 0))
        shadow_dy_box = tk.Spinbox(
            shadow_row, from_=-50, to=50, width=3,
            textvariable=self._text_shadow_dy_var,
            command=self._text_style_changed,
        )
        shadow_dy_box.pack(side=tk.LEFT, padx=(2, 0))
        shadow_dy_box.bind("<KeyRelease>", lambda _e: self._text_style_changed())
        self._text_shadow_btn = tk.Button(
            shadow_row, width=3, command=lambda: self._choose_text_paint("shadow"),
        )
        theme.skip(self._text_shadow_btn)
        self._text_shadow_btn.pack(side=tk.LEFT, padx=(4, 0))
        self._apply_text_swatches()
        self.text_new_layer_var = tk.BooleanVar(value=self.text_new_layer)
        tk.Checkbutton(
            self.settings, text="Create as new layer", anchor="w",
            variable=self.text_new_layer_var, command=self._on_text_new_layer,
        ).pack(fill=tk.X, pady=(4, 0))

    def _on_font_picked(self, family):
        self._text_family_var.set(family)
        self._text_style_changed()

    def _text_size_value(self):
        try:
            size = int(self._text_size_var.get())
        except (tk.TclError, ValueError):
            return textrender.DEFAULT_SIZE
        return max(textrender.MIN_SIZE, min(size, textrender.MAX_SIZE))

    def _text_int(self, var, default, low, high):
        try:
            value = int(round(float(var.get())))
        except (tk.TclError, ValueError):
            return default
        return max(low, min(high, value))

    def _apply_text_swatches(self):
        for name in ("_text_color_btn", "_text_stroke_btn", "_text_shadow_btn"):
            button = getattr(self, name, None)
            if button is None:
                continue
            color = {
                "_text_color_btn": self._text_color,
                "_text_stroke_btn": getattr(self, "_text_stroke_color", "#000000"),
                "_text_shadow_btn": getattr(self, "_text_shadow_color", "#000000"),
            }[name]
            button.config(bg=color, activebackground=color)

    def _choose_text_paint(self, target):
        attr = {
            "color": "_text_color",
            "stroke": "_text_stroke_color",
            "shadow": "_text_shadow_color",
        }[target]

        def preview(color):
            setattr(self, attr, color)
            self._apply_text_swatches()
            self._text_style_changed()

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), getattr(self, attr, "#000000"),
            title="Text Color", on_preview=preview,
            on_pick=self._on_pick_color,
        )
        if chosen is not None:
            preview(chosen)

    def _text_style_changed(self):
        if self._on_text_style is None:
            return
        stroke = None
        if self._text_stroke_var.get():
            stroke = {
                "color": getattr(self, "_text_stroke_color", "#000000"),
                "width": self._text_int(self._text_stroke_width_var, 2, 1, 20),
            }
        shadow = None
        if self._text_shadow_var.get():
            shadow = {
                "color": getattr(self, "_text_shadow_color", "#000000"),
                "blur": self._text_int(self._text_shadow_blur_var, 3, 0, 50),
                "dx": self._text_int(self._text_shadow_dx_var, 2, -50, 50),
                "dy": self._text_int(self._text_shadow_dy_var, 2, -50, 50),
            }
        self._on_text_style({
            "family": self._text_family_var.get() or fonts.default_family(),
            "size": self._text_size_value(),
            "bold": self._text_bold_var.get(),
            "italic": self._text_italic_var.get(),
            "align": self._text_align_var.get() or "left",
            "color": self._text_color,
            "line_spacing": self._text_int(self._text_line_var, 15, -50, 300) / 100.0,
            "letter_spacing": float(
                self._text_int(self._text_track_var, 0, -5, 50)
            ),
            "stroke": stroke,
            "shadow": shadow,
            "wrap": self._text_wrap_var.get(),
        })

    def set_text_style(self, style):
        """Point the built Text panel at ``style`` without applying it.

        Used when a text object is opened for editing so the panel shows that
        object's own font, size, colour and effects instead of whatever was used
        last. Unlike :meth:`_text_style_changed` this does not fire the
        on-change callback, so merely opening an object never mutates it.
        """
        if not self.text_tool or not hasattr(self, "_text_family_var"):
            return
        family = style.get("family") or fonts.default_family()
        self._text_family_var.set(family)
        if getattr(self, "_font_picker", None) is not None:
            self._font_picker.set(family)
        try:
            size = int(style.get("size", textrender.DEFAULT_SIZE))
        except (TypeError, ValueError):
            size = textrender.DEFAULT_SIZE
        self._text_size_var.set(str(size))
        self._text_bold_var.set(bool(style.get("bold", False)))
        self._text_italic_var.set(bool(style.get("italic", False)))
        align = style.get("align", "left")
        self._text_align_var.set(align if align in textrender.ALIGNMENTS else "left")
        self._text_color = style.get("color") or "#000000"
        leading = style.get("line_spacing", textrender.DEFAULT_LINE_SPACING)
        try:
            self._text_line_var.set(str(int(round(float(leading) * 100))))
        except (TypeError, ValueError):
            self._text_line_var.set("15")
        try:
            track = float(style.get("letter_spacing", 0.0) or 0.0)
        except (TypeError, ValueError):
            track = 0.0
        self._text_track_var.set(str(int(round(track))))
        stroke = style.get("stroke")
        self._text_stroke_var.set(bool(stroke))
        if stroke:
            self._text_stroke_color = stroke.get("color") or "#000000"
            self._text_stroke_width_var.set(str(int(stroke.get("width", 2))))
        shadow = style.get("shadow")
        self._text_shadow_var.set(bool(shadow))
        if shadow:
            self._text_shadow_color = shadow.get("color") or "#000000"
            self._text_shadow_blur_var.set(str(int(shadow.get("blur", 3))))
            self._text_shadow_dx_var.set(str(int(shadow.get("dx", 2))))
            self._text_shadow_dy_var.set(str(int(shadow.get("dy", 2))))
        self._text_wrap_var.set(bool(style.get("wrap", True)))
        self._apply_text_swatches()

    def set_path_style(self, spec):
        """Point the built Path panel at a committed path's own settings.

        Used when a path object is opened for editing so the Output mode,
        colours and the various stroke controls show that path's values rather
        than whatever the tool last used. This does not fire the settings
        callback, so merely opening a path never mutates it.
        """
        if not self.path_tool or not hasattr(self, "_path_mode_var"):
            return
        mode = spec.get("mode", self.path_mode)
        self._path_mode_var.set(mode if mode in pathspec.MODES else "fill")
        self.path_mode = self._path_mode_var.get()
        self.path_fill_color = spec.get("fill_color") or self._current_color()
        self.path_stroke_color = spec.get("stroke_color") or self._current_color()
        self._apply_path_swatches()
        try:
            size = int(spec.get("width", self.brush_size))
        except (TypeError, ValueError):
            size = self.brush_size
        self.size_var.set(str(max(1, min(size, MAX_BRUSH_SIZE))))
        self.on_size_change()
        outline = max(0, min(PATH_OUTLINE_MAX, int(spec.get("outline", 0) or 0)))
        feather = max(0, int(spec.get("feather", 0) or 0))
        opacity = max(1, min(100, int(spec.get("opacity", 100) or 100)))
        self._path_outline_var.set(outline)
        self._path_feather_var.set(feather)
        self._path_opacity_var.set(opacity)
        self._path_aa_var.set(bool(spec.get("antialias", True)))
        cap = spec.get("cap", "round")
        join = spec.get("join", "round")
        self._path_cap_var.set(cap if cap in pathspec.CAPS else "round")
        self._path_join_var.set(join if join in pathspec.JOINS else "round")
        self.path_cap = self._path_cap_var.get()
        self.path_join = self._path_join_var.get()
        self.path_outline = outline
        self.path_feather = feather
        self.path_opacity = opacity
        self.path_antialias = bool(spec.get("antialias", True))
        self._path_apply_rows()

    def path_style(self):
        """The Path tool's current settings as style keyword arguments."""
        tools = self
        return {
            "mode": tools.path_mode,
            "fill_color": tools.path_fill_color or tools._current_color(),
            "stroke_color": tools.path_stroke_color or tools._current_color(),
            "width": tools.brush_size,
            "outline": max(0, min(PATH_OUTLINE_MAX, int(tools.path_outline))),
            "feather": max(0, int(tools.path_feather)),
            "opacity": max(1, min(100, int(tools.path_opacity))),
            "antialias": bool(tools.path_antialias),
            "cap": tools.path_cap,
            "join": tools.path_join,
        }

    # ------------------------------------------------------------------ size
    def _sync_path(self, *_):
        # The values the on-canvas preview draws with. A change to any of them
        # repaints the pending path; Feather and Anti-alias are pixel effects
        # that the vector overlay cannot show, so they need no repaint.
        before = (self.path_cap, self.path_join, self.path_outline,
                  self.path_opacity)
        self.path_cap = self._path_cap_var.get()
        self.path_join = self._path_join_var.get()
        # The Outline slider tops out at PATH_OUTLINE_MAX, but its entry box
        # accepts any typed number, so clamp here and echo the clamped value
        # back into the box.
        raw_outline = self._int_var(self._path_outline_var, 0)
        self.path_outline = max(0, min(PATH_OUTLINE_MAX, raw_outline))
        if raw_outline != self.path_outline:
            try:
                self._path_outline_var.set(self.path_outline)
            except tk.TclError:
                pass
        self.path_feather = max(0, self._int_var(self._path_feather_var, 0))
        self.path_opacity = max(
            1, min(100, self._int_var(self._path_opacity_var, 100))
        )
        self.path_antialias = bool(self._path_aa_var.get())
        after = (self.path_cap, self.path_join, self.path_outline,
                 self.path_opacity)
        if before != after:
            self._refresh_path()

    def _refresh_path(self):
        if self._on_path_settings is not None:
            self._on_path_settings()

    def _on_path_new_layer(self):
        self.path_new_layer = bool(self.path_new_layer_var.get())

    def _finish_path(self):
        if self._on_path_action is not None:
            self._on_path_action()

    def _on_text_new_layer(self):
        self.text_new_layer = bool(self.text_new_layer_var.get())

    def on_size_change(self, _value=None):
        try:
            size = int(self.size_var.get())
        except (tk.TclError, ValueError):
            return
        self.brush_size = max(1, min(size, MAX_BRUSH_SIZE))
        self._on_brush_size()

    def _on_size_slider(self, _value=None):
        """The slider moved: mirror it into the size box, which applies it."""
        try:
            size = int(self._size_slider_var.get())
        except (tk.TclError, ValueError):
            return
        if self.size_var.get() != size:
            self.size_var.set(size)
        else:
            self.on_size_change()

    def _on_size_var_changed(self, *_args):
        """Apply a size typed into the box and keep the slider in step.

        The slider is clamped to ``BRUSH_SLIDER_MAX`` while the box may hold a
        larger size, so the slider only shows the clamped value.
        """
        try:
            size = int(self.size_var.get())
        except (tk.TclError, ValueError):
            return
        clamped = max(1, min(size, BRUSH_SLIDER_MAX))
        if self._size_slider_var.get() != clamped:
            self._size_slider_var.set(clamped)
        self.on_size_change()

    # ------------------------------------------------------------ persistence
    _EXPORT_ATTRS = [
        "brush_opacity", "brush_hardness", "brush_flow", "brush_blend",
        "brush_spacing", "brush_antialias", "brush_stabilize",
        "brush_stabilize_mode", "brush_tip",
        "brush_pattern", "brush_velocity_size", "brush_velocity_opacity",
        "brush_velocity_flow", "brush_velocity_hardness",
        "brush_velocity_spacing", "brush_taper_in", "brush_taper_out",
        "brush_grain", "brush_grain_scale", "brush_grain_strength",
        "brush_grain_mode", "brush_jitter_hue", "brush_jitter_sat",
        "brush_jitter_val", "brush_pickup", "brush_wet_edge",
        "eraser_opacity", "eraser_hardness", "eraser_blend", "eraser_spacing",
        "eraser_antialias", "eraser_stabilize", "eraser_stabilize_mode",
        "eraser_target", "eraser_color",
        "eraser_tip", "eraser_pattern", "eraser_velocity_size",
        "eraser_velocity_opacity", "eraser_velocity_flow",
        "eraser_velocity_hardness", "eraser_velocity_spacing",
        "eraser_taper_in", "eraser_taper_out",
        "eraser_grain", "eraser_grain_scale", "eraser_grain_strength",
        "eraser_grain_mode", "eraser_jitter_hue", "eraser_jitter_sat",
        "eraser_jitter_val", "eraser_pickup",
        "mask_value", "mask_erase", "mask_overlay",
        "fill_tolerance", "fill_contiguous", "fill_antialias", "fill_gap",
        "fill_opacity", "fill_blend",
        "gradient_mode", "gradient_from", "gradient_to", "gradient_opacity",
        "gradient_blend", "gradient_stops", "gradient_spread",
        "shape_kind", "shape_mode", "shape_opacity", "shape_blend",
        "shape_radius", "shape_antialias", "shape_new_layer",
        "paste_dest", "paste_fit_canvas", "paste_ask",
        "shape_fill_color", "shape_stroke_color",
        "clone_opacity", "clone_hardness", "clone_spacing",
        "smudge_strength", "smudge_hardness", "smudge_spacing",
        "dodge_strength", "dodge_hardness", "dodge_spacing",
        "dodge_range", "dodge_exposure", "dodge_protect",
        "burn_strength", "burn_hardness", "burn_spacing",
        "burn_range", "burn_exposure", "burn_protect",
        "liquify_mode", "liquify_strength", "warp_cols", "warp_rows",
        "puppet_influence", "puppet_expansion", "puppet_show_mesh",
        "puppet_correct_rotation",
        "select_mode",
        "select_radius",
        "select_transform",
        "wand_tolerance", "wand_contiguous", "wand_sample",
        "segment_model", "segment_sample",
        "path_mode", "path_cap", "path_join", "path_outline", "path_feather",
        "path_opacity", "path_antialias", "path_show_handles", "path_new_layer",
        "path_fill_color", "path_stroke_color",
        "vector_mode", "vector_show_anchors", "vector_color", "vector_opacity",
        "vector_cap", "vector_join", "vector_antialias", "vector_smooth",
        "vector_detail", "vector_snap", "vector_anchor_size",
        "poses_mode", "poses_scale", "poses_color",
        "poses_opacity", "poses_body", "poses_pose", "poses_hand",
        "poses_hand_side", "poses_drag", "poses_grid",
        "symmetry_mode", "symmetry_count", "symmetry_mirror",
    ]

    # Tools whose settings gain the symmetry guide block, i.e. the tools that
    # paint reflected dabs (the vector tool reflects its recorded line instead).
    SYMMETRY_TOOLS = {
        "brush", "eraser", "mask", "clone", "smudge", "dodge", "burn", "vector",
    }

    def export_settings(self):
        """Return the tool settings as a plain, serializable dict."""
        data = {name: getattr(self, name) for name in self._EXPORT_ATTRS}
        data["brush_size"] = self.brush_size
        return data

    def apply_settings(self, data):
        """Restore settings previously produced by :meth:`export_settings`."""
        if not isinstance(data, dict):
            return
        for name in self._EXPORT_ATTRS:
            if name in data:
                setattr(self, name, data[name])
        if "paste_dest" not in data and "paste_new_layer" in data:
            self.paste_dest = (
                "new_layer" if data["paste_new_layer"] else "selected_layer")
        if self.paste_dest not in ("new_layer", "selected_layer", "asset"):
            self.paste_dest = "new_layer"
        if "paste_fit_canvas" not in data and "paste_keep_size" in data:
            self.paste_fit_canvas = not bool(data["paste_keep_size"])
        # Migrate the old single pixel taper to the size-relative in/out pair.
        for prefix in ("brush", "eraser"):
            if f"{prefix}_taper" in data \
                    and f"{prefix}_taper_in" not in data \
                    and f"{prefix}_taper_out" not in data:
                try:
                    value = max(0, min(100, int(data[f"{prefix}_taper"])))
                except (TypeError, ValueError):
                    continue
                setattr(self, f"{prefix}_taper_in", value)
                setattr(self, f"{prefix}_taper_out", value)
        if "brush_size" in data:
            try:
                self.brush_size = max(1, min(int(data["brush_size"]),
                                             MAX_BRUSH_SIZE))
            except (TypeError, ValueError):
                pass
        try:
            self.path_outline = max(0, min(int(self.path_outline),
                                           PATH_OUTLINE_MAX))
        except (TypeError, ValueError):
            pass
        if self.select_mode in ("wand", "segment"):
            self.select_transform = False
        self.size_var.set(self.brush_size)
        self._sync_buttons()

    # ---------------------------------------------------------- brush presets
    # The tools that have a preset library, mapped to the settings prefix their
    # attributes share (see ``_EXPORT_ATTRS``). Every one paints with ``brush_size``
    # and the underlying stamping engine, so a preset is just its own settings.
    PRESET_TOOLS = {
        "brush": "brush_", "eraser": "eraser_", "clone": "clone_",
        "smudge": "smudge_", "dodge": "dodge_", "burn": "burn_",
    }

    def tool_params(self, tool):
        """The current settings for ``tool`` as a plain dict (a preset body)."""
        prefix = self.PRESET_TOOLS.get(tool)
        if prefix is None:
            return {}
        params = {name: getattr(self, name)
                  for name in self._EXPORT_ATTRS if name.startswith(prefix)}
        params["brush_size"] = self.brush_size
        return params

    def apply_tool_params(self, tool, params):
        """Apply a preset body to ``tool``'s settings and rebuild the panel."""
        prefix = self.PRESET_TOOLS.get(tool)
        if prefix is None or not isinstance(params, dict):
            return
        for name, value in params.items():
            if name != "brush_size" and name in self._EXPORT_ATTRS \
                    and name.startswith(prefix):
                setattr(self, name, value)
        if "brush_size" in params:
            try:
                self.brush_size = max(1, min(int(params["brush_size"]),
                                             MAX_BRUSH_SIZE))
                self.size_var.set(self.brush_size)
            except (TypeError, ValueError):
                pass
        self._sync_buttons()


class DrawingController:
    """Routes mouse events on a :class:`canvasarea.CanvasView` to layer edits.

    It reads the active tool flags from ``app.tools`` and the current color from
    ``app.colorpicker``, then asks the app to re-render when pixels change.
    """

    def __init__(self, app):
        self.app = app
        self.last_x = None
        self.last_y = None
        self.cursor_x = None
        self.cursor_y = None
        self.gizmo = TransformGizmo()
        self.sel_gizmo = SelectionGizmo()
        self._transform_pushed = False
        # Timestamp of the last transform drag frame, so a drag throttles the
        # expensive resample/overlay repaint instead of doing it on every motion
        # event; the gizmo frame still follows each event, and a release lands
        # the final, full-quality frame (see ``_transform_drag``).
        self._transform_render_at = 0.0
        self._sel_transform_dragged = False
        self._move_layers = []
        self._move_origin = None
        self._move_pushed = False
        self._pan_view = None
        self._pan_origin = None
        # Alt+middle-drag pans the pose scene's 3D camera (shifting the scene
        # anchor) instead of the 2D viewport; ``None`` when no scene pan is live.
        self._pose_pan = None
        # Shift+middle-drag rotates the 2D view; ``None`` when no rotate is live.
        self._view_rotate = None
        self._stroke_open = False
        self._stroke = None
        # The active stroke and its symmetry reflections, as ``(stroke, mapping)``
        # pairs (see ``symmetry``); the first entry is the identity mapping, so a
        # stroke without a guide is just this one-element list.
        self._strokes = []
        self._smooth_x = None
        self._smooth_y = None
        # Stabiliser state: the last raw pointer sample (for the velocity style)
        # and the recent samples (for the weighted style).
        self._raw_x = None
        self._raw_y = None
        self._stable_history = []
        # Last motion-event timestamp/position, for the velocity-driven brush
        # (see ``_paint_speed``).
        self._paint_time = None
        self._paint_pos = None
        # Last symmetry guide pushed to the view, so a change (and only a change)
        # repaints to show the new axes; see ``sync_symmetry``.
        self._symmetry_last = None
        # Liquify: the live session and the last pointer position (for the push
        # step). Mesh warp: the live session, the node being dragged, and the
        # layer's pre-warp image kept so Cancel can restore it. Puppet warp keeps
        # the same shape, with the dragged pin and its grab offset instead.
        self._liquify = None
        self._liquify_last = None
        self._warp = None
        self._warp_original = None
        self._warp_node = None
        self._warp_press_origin = None
        self._warp_start_offset = None
        self._warp_pushed = False
        self._puppet = None
        self._puppet_original = None
        self._puppet_pushed = False
        self._puppet_preview_job = None
        # Live puppet drag state: which gesture is running ("move"/"rotate"/
        # "band"), where it started, the selected pins' start frames, and the
        # rubber-band rectangle.
        self._puppet_mode = None
        self._puppet_press_origin = None
        self._puppet_start_pos = None
        self._puppet_start_angles = None
        self._puppet_rotate_center = None
        self._puppet_rotate_start = 0.0
        self._puppet_band = None
        self._puppet_band_additive = False
        self._puppet_moved = False
        self._puppet_manipulated = set()
        self._sel_mode = None
        self._sel_origin = None
        self._sel_points = []
        # The Layout tool's in-progress box: the document-pixel corner a drag
        # started from, or ``None`` when no box is being drawn. ``_layout_edit``
        # holds ``(row, handle, start_box, start_xy)`` while an existing box is
        # being moved or resized instead.
        self._layout_origin = None
        self._layout_edit = None
        # The last box picked with the Layout tool, as ``(x, y, row)``, so a
        # repeated click on the same spot can cycle to a box underneath.
        self._layout_last_pick = None
        self._float_selection = None
        self._float_start = None
        self._path = None
        self._path_dragging = None
        self._path_view = None
        self._path_move = None
        self._path_handle = None
        self._path_press_origin = None
        self._path_moved = False
        self._path_close_pending = False
        # Set while a committed path object is open for editing: a dict with its
        # ``layer``, ``index`` and whether an undo entry has been pushed.
        self._path_edit = None
        # Whole-path move drag: the press point and the geometry captured there.
        self._path_move_origin = None
        self._path_move_start = None
        # Timestamp of the last committed-path repaint, so an edit drag throttles
        # its repaints while the handles and overlay still follow every event.
        # ``_path_draft`` records that a drag painted a draft (non-antialiased)
        # frame, so the release repaints it at full quality.
        self._path_render_at = 0.0
        self._path_draft = False
        self._guide_index = None
        self._guide_view = None
        self._guide_pushed = False
        self._text = None
        self._text_drag = None
        self._blink_on = True
        self._gradient_start = None
        self._shape_start = None
        self._crop_start = None
        self._crop_box = None
        self._crop_handle = None
        self._crop_move_box = None
        self._clone_source = None
        self._mask_invert = False
        self._pick_cursor = False
        self._pick_cursor_bound = False
        # Vector tool state. ``_vector_points`` collects the polyline of the
        # stroke being drawn; ``_vector_cur`` is the vector layer the current
        # operation targets. ``_vector_sel`` indexes the stroke whose anchors are
        # shown, ``_vector_anchor`` is the anchor being dragged (or ``None``) and
        # ``_vector_dirty`` guards one undo push per drag.
        self._vector_points = None
        self._vector_cur = None
        self._vector_sel = None
        self._vector_anchor = None
        self._vector_handle_drag = None
        self._vector_dirty = False
        # Multi-selection edit state. ``_vector_selection`` is the ordered set of
        # stroke indices moved/scaled/rotated together, ``_vector_marquee`` the
        # corner a rubber-band selection started from, ``_vector_handle`` the
        # transform handle being dragged ("nw"/"ne"/"sw"/"se"/"rot") and
        # ``_vector_gesture`` the captured geometry that keeps a group drag
        # relative to its start.
        self._vector_selection = []
        self._vector_marquee = None
        self._vector_marquee_end = None
        self._vector_handle = None
        self._vector_gesture = None
        # Poses tool state. ``_pose_layer``/``_pose_sel`` identify the figure
        # being posed; ``_pose_drag`` is the in-progress gesture ("joint",
        # "move" or "rotate"), ``_pose_joint`` the joint it targets, and the
        # captured start values make each drag relative to the press so the
        # pose does not drift. ``_pose_dirty`` guards one undo push per gesture.
        self._pose_layer = None
        self._pose_sel = None
        self._pose_drag = None
        self._pose_joint = None
        self._pose_start = None
        self._pose_angles = None
        self._pose_orient = None
        # Values captured at press so a drag stays relative to where it started:
        # the figure's world ``pos``/``size``/``spin`` for a move/scale/rotate,
        # and the scene camera for an orbit.
        self._pose_pos = None
        self._pose_size = 1.0
        self._pose_spin = 0.0
        self._pose_scene_view = None
        self._pose_scene_roll = 0.0
        self._pose_dirty = False
        # Timestamp of the last posed redraw, so drags throttle the expensive
        # mesh render while the handles keep tracking the cursor.
        self._pose_render_at = 0.0
        # A click on a joint handle selects that bone and shows its rotation
        # rings; only once selected does a drag control it, so a limb is never
        # grabbed by accident. A ring drag then turns the bone about a world axis.
        # ``_pose_moved`` tells a click (select) from a drag (pose).
        self._pose_selected_bone = None
        self._pose_ring_axis = None
        self._pose_ring_angle = None
        self._pose_moved = False
        # Whole-figure transform is modal: the body drives one of ``FIGURE_MODES``
        # and a click (no drag) cycles to the next. ``_pose_cycle`` records that the
        # press started on the already-selected body, so release cycles the mode.
        self._pose_figure_mode = "move"
        self._pose_cycle = False
        # A press on empty space turns the figure in 3D (a click, no drag, just
        # drops the selection).
        self._pose_empty = False
        # The gizmo handle currently under the cursor (for hover highlighting) and
        # the last overlay data, so a motion event can hit-test without rebuilding
        # the projected geometry.
        self._pose_hover = None
        self._pose_overlay_data = None
        # The pivot (pelvis) and the values captured at press so a drag stays
        # relative to where it started.
        self._pose_pivot = None
        self._pose_figure_angle = None
        self._pose_figure_distance = None
        # Primitive-object transform state: the world axis handle being dragged,
        # the object's captured ``dim``, and the gizmo geometry captured at press.
        self._pose_axis = None
        self._pose_dim = (1.0, 1.0, 1.0)
        self._pose_primitive_geom = None

    def attach(self, view):
        view.bind("<Button-1>", self.on_press)
        view.bind("<B1-Motion>", self.on_drag)
        view.bind("<ButtonRelease-1>", self.on_release)
        view.bind("<Button-3>", self.on_mask_alt_press)
        view.bind("<B3-Motion>", self.on_mask_alt_drag)
        view.bind("<ButtonRelease-3>", self.on_mask_alt_release)
        view.bind("<Double-Button-1>", self.on_double_click)
        view.bind("<Triple-Button-1>", self.on_triple_click)
        view.bind("<Return>", self.on_path_return)
        view.bind("<Key>", self.on_text_key)
        view.bind("<Motion>", self.on_motion)
        view.bind("<Leave>", self.on_leave)
        view.bind("<MouseWheel>", self.on_wheel)
        view.bind("<Shift-MouseWheel>", self.on_shift_wheel)
        view.bind("<Control-MouseWheel>", self.on_ctrl_wheel)
        view.bind("<Alt-MouseWheel>", self.on_alt_wheel)
        view.bind("<Button-2>", self.on_middle_press)
        view.bind("<B2-Motion>", self.on_middle_drag)
        view.bind("<ButtonRelease-2>", self.on_middle_release)
        view.bind("<Double-Button-2>", self.on_middle_reset)
        self._bind_pick_cursor_keys()

    def _bind_pick_cursor_keys(self):
        """Watch the Control keys so the pick cursor appears without a move.

        The canvas rarely owns keyboard focus, so the chord is watched on the
        whole window; the handler only reacts when the pointer is over the view.
        """
        if self._pick_cursor_bound:
            return
        self._pick_cursor_bound = True
        for seq in ("<KeyPress-Control_L>", "<KeyPress-Control_R>"):
            self.app.root.bind_all(seq, self._on_ctrl_press, add="+")
        for seq in ("<KeyRelease-Control_L>", "<KeyRelease-Control_R>"):
            self.app.root.bind_all(seq, self._on_ctrl_release, add="+")

    def _on_ctrl_press(self, event):
        self._refresh_pick_cursor(True)

    def _on_ctrl_release(self, event):
        self._refresh_pick_cursor(False)

    def _pointer_over(self, view):
        """Whether the mouse pointer currently sits inside ``view``."""
        try:
            return view.winfo_containing(*view.winfo_pointerxy()) is view
        except tk.TclError:
            return False

    def _refresh_pick_cursor(self, ctrl):
        """Show or drop the pick cursor for the current Control state."""
        view = self.app.area.current_view()
        if view is None or not self._pointer_over(view):
            return
        sample = ctrl and self._pick_tool()
        self._apply_cursor(view, sample=sample)

    # ------------------------------------------------------------------ coords
    def _doc_coords(self, event):
        """Document coordinates for ``event``, snapped to the grid if enabled.

        Every tool funnels its pointer coordinates through here, so enabling
        grid snapping rounds brush, shape, selection, gradient, text and guide
        positions to the nearest grid intersection in one place.
        """
        widget = event.widget
        x, y = widget.event_to_doc(event)
        snap = getattr(widget, "snap_doc", None)
        if snap is not None:
            return snap(x, y)
        return x, y

    def _select_exploded_cell(self, event):
        """Make the layer under an exploded-view press the active one.

        Returns ``True`` when the press changed the selection, ``False`` when it
        hit the already-active tile (or a gap between cells). A change only
        selects the tile and mirrors it in the layers panel; the caller then
        leaves the tool for the next press, so the tool is only ever applied to
        a layer that already is the current selection.
        """
        index = event.widget.layer_at_event(event)
        if index is None or index == self.app.active:
            return False
        layer = self.app.current.layers[index] if self.app.current else None
        self.app._on_layer_select(index)
        panel = getattr(self.app, "layer_panel", None)
        if panel is not None:
            panel.select_layer(index)
        # Recompute the view's coordinate shift for the newly active layer and
        # repaint, so the next press addresses the right cell.
        self.app.render()
        if layer is not None:
            suffix = " (hidden in the stacked view)" if not layer.visible else ""
            self.app.set_result(f"Selected layer '{layer.name}'{suffix}.")
        return True

    def _preview_showing(self, event):
        """Whether a pending preview owns the view under ``event``.

        While a generation or asset preview is shown, the tools must not touch
        the document: only the preview bar's buttons accept or reject it. Zoom
        and pan (wheel, middle-drag) stay live so the preview can be inspected.
        """
        check = getattr(event.widget, "has_generation_preview", None)
        return bool(check is not None and check())

    # ------------------------------------------------------------------ cursor
    def _radius(self):
        return max(self.app.tools.brush_size, 1) / 2

    def refresh_cursor(self):
        view = self.app.area.current_view()
        if view is not None:
            self._apply_cursor(view)

    # Tools that do not paint the layer, so a Control chord on them is not a
    # color sample and keeps its own cursor.
    _SAMPLE_EXCLUDED = (
        "move_tool", "drag_tool", "transform_tool", "select_tool", "text_tool",
        "path_tool", "vector_tool", "fill_tool", "gradient_tool", "shape_tool",
        "crop_tool", "paste_tool", "arrow_tool", "eyedropper", "liquify_tool",
        "warp_tool", "puppet_tool", "poses_tool",
    )

    def _paint_tool(self):
        """Whether the active tool paints the layer (brush and friends)."""
        tools = self.app.tools
        return not any(getattr(tools, flag, False) for flag in self._SAMPLE_EXCLUDED)

    # Non-painting tools whose on-canvas mark is sized by the shared Size row,
    # so they show the brush-size ring even though they are excluded from colour
    # sampling. Liquify grows its dabs by ``brush_size``; the Shape and Vector
    # tools draw their strokes at that width. The Path tool sizes only its
    # stroke output, so it is handled separately (see ``_brush_overlay_tool``).
    _BRUSH_SIZE_TOOLS = ("liquify_tool", "shape_tool", "vector_tool")

    def _brush_overlay_tool(self):
        """Whether the active tool outlines its brush size under the pointer.

        The painting tools show the size ring, plus the tools whose on-canvas
        mark is sized by the shared Size row. The Path tool draws with that
        width only in its stroke modes, matching when its Size row is shown.
        """
        tools = self.app.tools
        if self._paint_tool():
            return True
        if any(getattr(tools, flag, False) for flag in self._BRUSH_SIZE_TOOLS):
            return True
        return bool(tools.path_tool and tools.path_mode in ("stroke", "both"))

    def _pick_tool(self):
        """Whether a Control chord on the active tool should sample the color."""
        return self._paint_tool() and not self.app.tools.eyedropper

    def _samples_color(self, event):
        """Whether ``event`` is a Control chord that samples a color."""
        return bool(event.state & 0x4) and self._pick_tool()

    def _set_pick_cursor(self, view, active):
        """Swap the widget cursor to the eyedropper while a sample is held."""
        if active == self._pick_cursor:
            return
        self._pick_cursor = active
        try:
            view.config(cursor="dotbox" if active else "crosshair")
        except tk.TclError:
            pass

    def _apply_cursor(self, view, sample=False):
        if sample and self._pick_tool():
            view.set_cursor_overlay(None)
            self._set_pick_cursor(view, True)
            return
        if self._pick_tool():
            self._set_pick_cursor(view, False)
        if not self._brush_overlay_tool() or self.cursor_x is None:
            view.set_cursor_overlay(None)
        else:
            view.set_cursor_overlay(
                self.cursor_x, self.cursor_y, self._radius(), self._cursor_tip(),
            )

    def _cursor_tip(self):
        """The tip shape of the active painting tool, for the cursor outline."""
        tools = self.app.tools
        if (tools.mask_edit or tools.inpaint_mask_edit or tools.path_tool
                or any(getattr(tools, flag, False)
                       for flag in self._BRUSH_SIZE_TOOLS)):
            return "round"
        if tools.eraser:
            return tools.eraser_tip.lower()
        return tools.brush_tip.lower()

    def on_motion(self, event):
        self.cursor_x, self.cursor_y = self._doc_coords(event)
        self.app.report_cursor(self.cursor_x, self.cursor_y)
        if self._preview_showing(event):
            event.widget.set_cursor_overlay(None)
            return
        if self._samples_color(event):
            self._apply_cursor(event.widget, sample=True)
            return
        if self.app.tools.transform_tool:
            self._hover_transform(event)
            return
        if self._selection_transform_active() and self.sel_gizmo.active:
            self._hover_selection_transform(event)
            return
        if self.app.tools.layout_tool:
            self._hover_layout(event)
            return
        if self.app.tools.crop_tool:
            self._apply_cursor(event.widget)
            self._hover_crop(event)
            return
        if self.app.tools.move_tool:
            self._hover_guide(event)
            return
        if self.app.tools.path_tool:
            self._update_path_preview(cursor=(self.cursor_x, self.cursor_y))
            self._apply_cursor(event.widget)
            return
        if self.app.tools.vector_tool:
            self._apply_cursor(event.widget)
            return
        if self.app.tools.poses_tool:
            self._apply_cursor(event.widget)
            self._hover_poses(event)
            return
        if self.app.tools.text_tool:
            self._apply_cursor(event.widget)
            self._hover_text(event)
            return
        self._apply_cursor(event.widget)

    def on_leave(self, event):
        self.cursor_x = None
        self.cursor_y = None
        self.app.report_cursor(None, None)
        event.widget.set_cursor_overlay(None)
        if self.app.tools.poses_tool and self._pose_hover is not None:
            self._pose_hover = None
            self._update_poses_overlay()

    # ------------------------------------------------------------------ wheel
    WHEEL_PIXELS = 40

    def on_wheel(self, event):
        view = event.widget
        if view._content_h > view.winfo_height():
            view.yview_scroll(self._wheel_steps(event) * self.WHEEL_PIXELS, "units")
        return "break"

    def on_shift_wheel(self, event):
        view = event.widget
        if view._content_w > view.winfo_width():
            view.xview_scroll(self._wheel_steps(event) * self.WHEEL_PIXELS, "units")
        return "break"

    def _wheel_steps(self, event):
        return int(-event.delta / 120) or (-1 if event.delta > 0 else 1)

    def on_ctrl_wheel(self, event):
        view = event.widget
        if event.delta > 0:
            view.zoom_in(event)
        else:
            view.zoom_out(event)
        self.app.report_zoom(view)
        return "break"

    def on_alt_wheel(self, event):
        """Alt+wheel dollies the pose scene's 3D camera, zooming about the cursor.

        This is the scene's own zoom (how large the figures are in the document),
        not the 2D canvas zoom that Ctrl+wheel drives.
        """
        if not self.app.tools.poses_tool:
            return None
        layer = self._pose_layer or self.app.active_layer()
        if layer is None:
            return "break"
        scene = self._pose_scene(layer)
        steps = self._wheel_steps(event)
        if not steps:
            return "break"
        x, y = self._doc_coords(event)
        poses.set_scene_zoom(
            scene, float(scene["zoom"]) * (1.1 ** -steps), pivot=(x, y),
        )
        poses.sync_scene(layer.poses, scene)
        layer.invalidate_poses()
        self.app.tools._refresh_pose_scene_zoom(scene)
        self._update_poses_overlay()
        self.app.render()
        self.app.set_result(
            f"Scene zoom {scene_zoom_percent(scene)}%."
        )
        return "break"

    # ------------------------------------------------------------------ press
    def _layer_locked(self):
        layer = self.app.active_layer()
        if layer is None:
            self.app.set_result("There is no layer. Add a layer first.")
            return True
        # The exploded view always shows every layer, so a hidden layer is
        # visible and editable there; only the explicit lock still blocks it.
        exploded = getattr(self.app, "show_exploded", False)
        blocked = layer.locked or (not layer.visible and not exploded)
        if blocked:
            if not layer.visible and not exploded:
                self.app.set_result(f"Layer '{layer.name}' is hidden (locked).")
            else:
                self.app.set_result(f"Layer '{layer.name}' is locked.")
            return True
        return False

    def on_press(self, event):
        if self._preview_showing(event):
            return
        event.widget.focus_set()
        tools = self.app.tools
        exploded = getattr(event.widget, "exploded", False)
        # Exploded view: a press on a tile other than the active one only makes
        # that layer the current selection, so the tool is never applied to a
        # layer that was not already selected - a second press edits it. Panning
        # (the drag tool) and the tools that rearrange the document rather than
        # paint a layer do not move the selection - the first because a pan is
        # not an edit, the others because they are not available on the grid.
        # Editing the canvas-sized inpaint mask does not map onto a cell, so it
        # is refused there too.
        if exploded:
            if tools.crop_tool or tools.move_tool or tools.transform_tool \
                    or tools.layout_tool or tools.paste_tool \
                    or tools.inpaint_mask_edit:
                self.app.set_result(
                    "Exploded view: pick a paint tool, then click a tile to edit it."
                )
                return
            if not tools.drag_tool and self._select_exploded_cell(event):
                # The press only selected a different tile: stop here so the
                # tool is not applied until the newly selected layer is clicked
                # again. Panning (drag) and a press on the active tile fall
                # through to the normal tool handling below.
                return
        x, y = self._doc_coords(event)
        self.cursor_x, self.cursor_y = x, y
        self.app.report_cursor(x, y)
        if self.app.canvas_pick_pending():
            self.app.pick_color_at(x, y)
            return
        if tools.eyedropper:
            self.app.pick_color_at(x, y)
            return
        if tools.paste_tool:
            return
        if tools.arrow_tool:
            return
        if tools.layout_tool:
            self._layout_press(x, y)
            return
        if tools.path_tool:
            self._path_press(event, x, y)
            return
        if tools.vector_tool:
            self._vector_press(event, x, y)
            return
        if tools.poses_tool:
            self._poses_press(event, x, y)
            return
        if tools.liquify_tool:
            self._liquify_press(event, x, y)
            return
        if tools.warp_tool:
            self._warp_press(event, x, y)
            return
        if tools.puppet_tool:
            self._puppet_press(event, x, y)
            return
        if tools.text_tool:
            self.last_x = self.last_y = None
            self._text_press(event, x, y)
            return
        if tools.fill_tool:
            self.fill_at(x, y)
            return
        if tools.select_tool and not getattr(tools, "select_transform", False) \
                and tools.select_mode == "wand":
            self.wand_at(x, y)
            return
        if tools.select_tool and not getattr(tools, "select_transform", False) \
                and tools.select_mode == "segment":
            self.segment_at(x, y)
            return
        if tools.crop_tool:
            self._crop_press(event, x, y)
            return
        if tools.gradient_tool:
            self._gradient_press(x, y)
            return
        if tools.shape_tool:
            self._shape_press(x, y)
            return
        if tools.clone_tool and (event.state & ALT_STATE_MASK):
            self.set_clone_source(x, y)
            return
        if tools.select_tool:
            if getattr(tools, "select_transform", False):
                self._selection_transform_press(event)
            else:
                self._begin_select(x, y)
            return
        if tools.drag_tool:
            self._begin_pan(event)
            return
        if tools.transform_tool:
            self._transform_press(event)
            return
        if tools.move_tool:
            if self._begin_guide(event, x, y):
                return
            if self._layer_locked():
                return
            self._begin_move(x, y)
            return
        if tools.clone_tool and self._clone_source is None:
            self.app.set_result("Alt-click to set the clone source first.")
            return
        if self._samples_color(event):  # a Control chord samples the color
            self.last_x = self.last_y = None
            self.app.pick_color_at(x, y)
            return
        if self._layer_locked():
            return
        invert = bool(event.state & 0x1) and (
            tools.mask_edit or tools.inpaint_mask_edit)
        self._paint_press(event, x, y, invert=invert)

    def _paint_press(self, event, x, y, invert=False):
        """Start a paint stroke at ``(x, y)``, inverting a mask when asked."""
        if self._layer_locked():
            return
        self.last_x, self.last_y = x, y
        self.cursor_x, self.cursor_y = x, y
        self._stroke_open = False
        self._mask_invert = invert
        self._begin_paint(x, y)
        self._apply_cursor(event.widget)
        self.app.render()

    def on_mask_alt_press(self, event):
        """Right-click: mask inverse paint, path-anchor delete, or context menu."""
        if self._preview_showing(event):
            return
        if self.app.tools.path_tool and self._path is not None:
            return self._path_right_click(event)
        if self.app.tools.vector_tool:
            x, y = self._doc_coords(event)
            if getattr(self.app.tools, "vector_mode", "draw") == "edit" \
                    and self._vector_delete_anchor(x, y, event.widget):
                return "break"
            return self._post_canvas_menu(event)
        if self.app.tools.puppet_tool and self._puppet is not None:
            x, y = self._doc_coords(event)
            if self._puppet_delete_pin(x, y, event.widget):
                return "break"
            return self._post_canvas_menu(event)
        if self.app.tools.poses_tool:
            x, y = self._doc_coords(event)
            if self._poses_reset_joint(x, y, event.widget):
                return "break"
            return self._post_canvas_menu(event)
        if not self.app.tools.mask_edit:
            return self._post_canvas_menu(event)
        x, y = self._doc_coords(event)
        self.cursor_x, self.cursor_y = x, y
        self.app.report_cursor(x, y)
        self._paint_press(event, x, y, invert=True)
        return "break"

    def on_mask_alt_drag(self, event):
        if self._preview_showing(event):
            return
        if not self.app.tools.mask_edit or self.last_x is None:
            return
        x, y = self._doc_coords(event)
        self._continue_paint(x, y)
        self.app.render()
        self.last_x, self.last_y = x, y
        self.cursor_x, self.cursor_y = x, y
        self._apply_cursor(event.widget)
        return "break"

    def on_mask_alt_release(self, event):
        if not self.app.tools.mask_edit:
            return
        self.on_release(event)
        return "break"

    def _post_canvas_menu(self, event):
        """Post the canvas context menu at the pointer.

        The rows are cascades into the same menu-bar categories, so the canvas
        offers the whole application menu in place (GIMP-style) rather than a
        second, smaller action list that could drift out of sync.
        """
        menu = PopupMenu(event.widget)
        for label, category in build_menus(self.app, event.widget):
            menu.add_cascade(label, category)
        menu.post(event.x_root, event.y_root)
        return "break"

    def _begin_stroke(self):
        if self._stroke_open:
            return
        self.app._push_undo("Paint")
        # Grow the active layer's buffer so the stroke can reach the whole
        # canvas, without discarding pixels moved past the canvas edge. The
        # inpaint mask is canvas-sized already, so it needs no growing. In the
        # exploded view the document offsets are not the editing frame, so
        # growing the buffer here would corrupt the document; skip it.
        if (not self.app.tools.inpaint_mask_edit
                and not getattr(self.app, "show_exploded", False)):
            self.app.current.align_for_painting(self.app.active_layer())
        self._stroke_open = True

    # --------------------------------------------------------------- painting
    def _stroke_mode(self):
        """The kind of stroke the active tool paints, as ``brush.Stroke`` sees it."""
        tools = self.app.tools
        if tools.mask_edit or tools.inpaint_mask_edit:
            return "mask"
        if tools.eraser:
            return "eraser-color" if tools.eraser_target == "color" else "eraser"
        if tools.clone_tool:
            return "clone"
        if tools.smudge_tool:
            return "smudge"
        if tools.dodge_tool:
            return "dodge"
        if tools.burn_tool:
            return "burn"
        return "brush"

    def _stable_point(self, x, y, mode):
        """Smooth the pointer toward a stabilised cursor for the active tool.

        Three styles share the "Smooth" strength: ``simple`` eases the cursor
        toward the pointer (the original behaviour); ``weighted`` averages the
        last few samples for a steadier line; ``velocity`` eases off as the
        pointer speeds up so a fast flick is not left lagging behind.
        """
        tools = self.app.tools
        if tools.eraser:
            stabilize = tools.eraser_stabilize
            style = getattr(tools, "eraser_stabilize_mode", "simple")
        else:
            stabilize = tools.brush_stabilize
            style = getattr(tools, "brush_stabilize_mode", "simple")
        if self._smooth_x is None:
            self._smooth_x, self._smooth_y = x, y
            self._raw_x, self._raw_y = x, y
            self._stable_history = [(x, y)]
            return x, y
        if style == "weighted" and stabilize > 0:
            window = 2 + int(round(stabilize / 100.0 * 18))
            self._stable_history.append((x, y))
            if len(self._stable_history) > STABILIZE_HISTORY:
                del self._stable_history[:-window]
            points = self._stable_history[-window:]
            count = len(points)
            self._smooth_x = sum(point[0] for point in points) / count
            self._smooth_y = sum(point[1] for point in points) / count
        else:
            if style == "velocity":
                speed = math.hypot(x - self._raw_x, y - self._raw_y) / 18.0
                strength = (stabilize / 100.0) * max(0.0, 1.0 - min(1.0, speed))
            else:
                strength = stabilize / 100.0
            factor = max(0.05, 1.0 - strength)
            self._smooth_x += (x - self._smooth_x) * factor
            self._smooth_y += (y - self._smooth_y) * factor
        self._raw_x, self._raw_y = x, y
        return self._smooth_x, self._smooth_y

    def _begin_paint(self, x, y):
        """Start a brush/eraser/mask stroke whose first dab lands at ``(x, y)``.

        With a symmetry guide active, one stroke is created per reflection and
        they are all fed the reflected pointer positions, so a single drag paints
        the mirrored copies together. Every stroke is built before any is started
        so each snapshots the same pre-stroke pixels (dodge, burn and smudge read
        their base image).
        """
        tools = self.app.tools
        mode = self._stroke_mode()
        self._begin_stroke()
        self._smooth_x = self._smooth_y = None
        self._raw_x = self._raw_y = None
        self._stable_history = []
        layer = self.app.stroke_surface()
        ox, oy = layer.offset
        clip = None if tools.inpaint_mask_edit else self._selection_mask(layer, ox, oy)
        mappings = symmetry.transforms(
            self.app.symmetry_guide(), self.app.current.size
        )
        clone_source = layer.image.copy() if mode == "clone" else None
        starts = [mapping(x, y) for mapping in mappings]
        self._strokes = []
        for mapping, (px, py) in zip(mappings, starts):
            stroke = self._make_stroke(layer, mode, (px, py), clone_source)
            stroke.clip = clip
            self._strokes.append((stroke, mapping, (px, py)))
        for stroke, _mapping, (px, py) in self._strokes:
            stroke.begin(px - ox, py - oy)
        self._stroke = self._strokes[0][0]
        self._smooth_x, self._smooth_y = x, y
        self._paint_time = None
        self._paint_pos = (x, y)

    def _make_stroke(self, layer, mode, start, clone_source):
        """Build the ``brush.Stroke`` for ``mode`` whose first dab lands near ``start``."""
        tools = self.app.tools
        x, y = start
        size = tools.brush_size
        if mode == "mask":
            value = tools.mask_value
            if tools.mask_erase:
                value = 0 if value >= 128 else 255
            if self._mask_invert:
                value = 255 - value
            # Masks paint with the active brush's stamping settings so a mask
            # takes the same tip, spacing, hardness, grain and dynamics as the
            # brush; only the painted *value* is mask-specific.
            return brush.Stroke(
                layer, "mask", size=size, hardness=tools.brush_hardness,
                flow=tools.brush_flow, opacity=tools.brush_opacity,
                spacing=tools.brush_spacing, antialias=tools.brush_antialias,
                tip=tools.brush_tip.lower(), pattern=tools.brush_pattern,
                paint_value=value, dynamics=self._brush_dynamics(),
                grain=tools.brush_grain,
                grain_scale=tools.brush_grain_scale / 100.0,
                grain_strength=tools.brush_grain_strength / 100.0,
                grain_mode=tools.brush_grain_mode,
                wet_edge=tools.brush_wet_edge / 100.0,
            )
        if mode == "eraser":
            return brush.Stroke(
                layer, "eraser", size=size, hardness=tools.eraser_hardness,
                flow=100, opacity=tools.eraser_opacity,
                spacing=tools.eraser_spacing,
                antialias=tools.eraser_antialias, tip=tools.eraser_tip.lower(),
                pattern=tools.eraser_pattern,
                dynamics=self._eraser_dynamics(),
                grain=tools.eraser_grain,
                grain_scale=tools.eraser_grain_scale / 100.0,
                grain_strength=tools.eraser_grain_strength / 100.0,
                grain_mode=tools.eraser_grain_mode,
                jitter_hue=tools.eraser_jitter_hue / 100.0,
                jitter_sat=tools.eraser_jitter_sat / 100.0,
                jitter_val=tools.eraser_jitter_val / 100.0,
                pickup=tools.eraser_pickup / 100.0,
            )
        if mode == "eraser-color":
            return brush.Stroke(
                layer, "eraser-color", size=size,
                hardness=tools.eraser_hardness, flow=100,
                opacity=tools.eraser_opacity, spacing=tools.eraser_spacing,
                antialias=tools.eraser_antialias, tip=tools.eraser_tip.lower(),
                pattern=tools.eraser_pattern,
                color=rgba(tools.eraser_color)[:3],
                blend=tools.eraser_blend.lower(),
                dynamics=self._eraser_dynamics(),
                grain=tools.eraser_grain,
                grain_scale=tools.eraser_grain_scale / 100.0,
                grain_strength=tools.eraser_grain_strength / 100.0,
                grain_mode=tools.eraser_grain_mode,
                jitter_hue=tools.eraser_jitter_hue / 100.0,
                jitter_sat=tools.eraser_jitter_sat / 100.0,
                jitter_val=tools.eraser_jitter_val / 100.0,
                pickup=tools.eraser_pickup / 100.0,
            )
        if mode == "clone":
            source_offset = (x - self._clone_source[0], y - self._clone_source[1])
            return brush.Stroke(
                layer, "clone", size=size, hardness=tools.clone_hardness,
                flow=100, opacity=tools.clone_opacity,
                spacing=tools.clone_spacing, antialias=True,
                tip="Round", source=clone_source, source_offset=source_offset,
            )
        if mode == "smudge":
            return brush.Stroke(
                layer, "smudge", size=size, hardness=tools.smudge_hardness,
                flow=tools.smudge_strength, opacity=100,
                spacing=tools.smudge_spacing, antialias=True, tip="Round",
            )
        if mode in ("dodge", "burn"):
            if mode == "dodge":
                hardness, strength = tools.dodge_hardness, tools.dodge_strength
                spacing = tools.dodge_spacing
                tone_range, exposure = tools.dodge_range, tools.dodge_exposure
                protect = tools.dodge_protect
            else:
                hardness, strength = tools.burn_hardness, tools.burn_strength
                spacing = tools.burn_spacing
                tone_range, exposure = tools.burn_range, tools.burn_exposure
                protect = tools.burn_protect
            return brush.Stroke(
                layer, mode, size=size, hardness=hardness, flow=strength,
                opacity=100, spacing=spacing, antialias=True, tip="Round",
                tone_range=tone_range, tone_exposure=exposure,
                tone_protect=protect,
            )
        return brush.Stroke(
            layer, "brush", size=size, hardness=tools.brush_hardness,
            flow=tools.brush_flow, opacity=tools.brush_opacity,
            spacing=tools.brush_spacing, antialias=tools.brush_antialias,
            tip=tools.brush_tip.lower(), pattern=tools.brush_pattern,
            color=rgba(self.app.colorpicker.color)[:3],
            blend=tools.brush_blend.lower(),
            dynamics=self._brush_dynamics(),
            grain=tools.brush_grain,
            grain_scale=tools.brush_grain_scale / 100.0,
            grain_strength=tools.brush_grain_strength / 100.0,
            grain_mode=tools.brush_grain_mode,
            jitter_hue=tools.brush_jitter_hue / 100.0,
            jitter_sat=tools.brush_jitter_sat / 100.0,
            jitter_val=tools.brush_jitter_val / 100.0,
            pickup=tools.brush_pickup / 100.0,
            wet_edge=tools.brush_wet_edge / 100.0,
        )

    def _brush_dynamics(self):
        """Velocity dynamics for the brush, from its speed/taper settings."""
        tools = self.app.tools
        size = tools.brush_size
        return brush.Dynamics(
            size=tools.brush_velocity_size / 100.0,
            opacity=tools.brush_velocity_opacity / 100.0,
            flow=tools.brush_velocity_flow / 100.0,
            hardness=tools.brush_velocity_hardness / 100.0,
            spacing=tools.brush_velocity_spacing / 100.0,
            taper_in=size * tools.brush_taper_in / 100.0,
            taper_out=size * tools.brush_taper_out / 100.0,
        )

    def _eraser_dynamics(self):
        """Velocity dynamics for the eraser, from its speed/taper settings."""
        tools = self.app.tools
        size = tools.brush_size
        return brush.Dynamics(
            size=tools.eraser_velocity_size / 100.0,
            opacity=tools.eraser_velocity_opacity / 100.0,
            flow=tools.eraser_velocity_flow / 100.0,
            hardness=tools.eraser_velocity_hardness / 100.0,
            spacing=tools.eraser_velocity_spacing / 100.0,
            taper_in=size * tools.eraser_taper_in / 100.0,
            taper_out=size * tools.eraser_taper_out / 100.0,
        )

    def _end_paint(self):
        """Finish the live stroke (applying the end taper) before releasing it."""
        for stroke, _mapping, _start in self._strokes:
            stroke.finish()

    def _paint_speed(self, x, y, event):
        """Normalised (0..1) pointer speed for the velocity-driven brush.

        Measured from the real event timing (``event.time``, milliseconds) so the
        dynamics do not depend on how often the window manager delivers motion
        events. A missing timestamp or a zero interval returns ``None``, leaving
        the stroke to estimate speed from its own dab spacing instead.
        """
        now = getattr(event, "time", None)
        last = self._paint_time
        prev = self._paint_pos
        self._paint_time = now
        self._paint_pos = (x, y)
        if now is None or last is None or prev is None:
            return None
        dt = now - last
        if dt <= 0:
            return None
        speed = math.hypot(x - prev[0], y - prev[1]) / dt
        return max(0.0, min(1.0, speed / PAINT_SPEED_FULL))

    def _continue_paint(self, x, y, speed=None):
        if self._stroke is None:
            self._begin_paint(x, y)
            return
        layer = self.app.stroke_surface()
        ox, oy = layer.offset
        sx, sy = self._stable_point(x, y, self._stroke_mode())
        for stroke, mapping, _start in self._strokes:
            px, py = mapping(sx, sy)
            stroke.move_to(px - ox, py - oy, speed=speed)

    # ------------------------------------------------------------------- fill
    def fill_at(self, x, y):
        """Flood-fill the active layer with the current color from ``(x, y)``."""
        doc = self.app.current
        width, height = doc.size
        if not (0 <= x < width and 0 <= y < height):
            return
        if self._layer_locked():
            return
        tools = self.app.tools
        layer = self.app.active_layer()
        self.app._push_undo("Fill")
        # Map document pixels onto the layer's own buffer the way a stroke does,
        # growing it first so a fill can reach the whole canvas.
        doc.align_for_painting(layer)
        ox, oy = layer.offset
        image = layer.image
        seed = (int(x - ox), int(y - oy))
        if not (0 <= seed[0] < image.width and 0 <= seed[1] < image.height):
            self.app._drop_undo()
            return
        coverage = fill.region_mask(
            image, seed, tolerance=tools.fill_tolerance,
            contiguous=tools.fill_contiguous,
            bounds=self._fill_bounds(layer, ox, oy),
            antialias=tools.fill_antialias, gap=tools.fill_gap,
        )
        fill.apply_fill(
            layer, rgba(self.app.colorpicker.color)[:3], coverage,
            opacity=tools.fill_opacity, blend=tools.fill_blend.lower(),
        )
        self.app.render()
        self.app.set_result(f"Filled region on '{layer.name}'.")

    def _selection_mask(self, layer, ox, oy):
        """A layer-sized mask limiting painting to the active selection, if any.

        Returns ``None`` when there is no selection, so callers can skip the
        clipping step entirely.
        """
        selection = self._selection()
        if selection is None or selection.is_empty():
            return None
        mask = Image.new("L", layer.image.size, 0)
        mask.paste(selection.mask, (-ox, -oy))
        return mask

    def _fill_bounds(self, layer, ox, oy):
        """A layer-sized mask limiting a fill to the active selection, if any."""
        return self._selection_mask(layer, ox, oy)

    # -------------------------------------------------------------- gradient
    def _gradient_press(self, x, y):
        self._gradient_start = (x, y)
        self._update_gradient_preview((x, y))

    def _gradient_drag(self, x, y):
        if self._gradient_start is None:
            return
        self._update_gradient_preview((x, y))

    def _gradient_release(self, x, y):
        start = self._gradient_start
        self._gradient_start = None
        self._update_gradient_preview(None)
        if start is None:
            return
        if self._layer_locked():
            return
        tools = self.app.tools
        doc = self.app.current
        layer = self.app.active_layer()
        self.app._push_undo("Gradient")
        doc.align_for_painting(layer)
        ox, oy = layer.offset
        size = layer.image.size
        rgb, alpha = gradient.gradient_layers(
            size, (start[0] - ox, start[1] - oy), (x - ox, y - oy),
            tools.gradient_mode,
            stops=tools._gradient_stop_list(),
            spread=tools.gradient_spread,
        )
        coverage = Image.new("L", size, 255)
        selection = self._selection()
        if selection is not None and not selection.is_empty():
            coverage = Image.new("L", size, 0)
            coverage.paste(selection.mask, (-ox, -oy))
        gradient.apply(
            layer, rgb, coverage, tools.gradient_opacity,
            tools.gradient_blend.lower(), alpha=alpha,
        )
        layer.invalidate_filters()
        self.app.render()
        self.app.set_result("Applied gradient.")

    def _update_gradient_preview(self, end):
        view = self.app.area.current_view()
        if view is None:
            return
        if end is None or self._gradient_start is None:
            view.set_gradient_preview(None)
            return
        view.set_gradient_preview({
            "start": self._gradient_start, "end": end,
            "color": self.app.tools.gradient_from,
        })

    # ----------------------------------------------------------------- shapes
    def _shape_press(self, x, y):
        self._shape_start = (x, y)
        self._update_shape_preview((x, y), False)

    def _shape_drag(self, x, y, shift):
        if self._shape_start is None:
            return
        self._update_shape_preview((x, y), shift)

    def _shape_release(self, x, y, shift):
        start = self._shape_start
        self._shape_start = None
        self._update_shape_preview(None, False)
        if start is None:
            return
        tools = self.app.tools
        if not tools.shape_new_layer and self._layer_locked():
            return
        doc = self.app.current
        layer = self.app.active_layer()
        self.app._push_undo("Shape")
        if tools.shape_new_layer:
            layer = doc.add_layer(
                name="Shape",
                index=(
                    doc.layers.index(doc.active_layer()) + 1
                    if doc.active_layer() is not None else len(doc.layers)
                ),
            )
            self.app.active = doc.layers.index(layer)
            self.app._refresh_layer_list()
        width = max(1, tools.brush_size)
        pad = int(math.ceil(width / 2.0)) + 2
        # The dragged primitive can reach past the canvas edge; grow the layer's
        # buffer to cover it first, so the shape is not clipped at that edge and
        # stays whole when the layer is moved later.
        doc_final = shape.constrain(
            (start[0], start[1], x, y), tools.shape_kind, shift
        )
        doc.align_for_painting(layer, (
            min(doc_final[0], doc_final[2]) - pad,
            min(doc_final[1], doc_final[3]) - pad,
            max(doc_final[0], doc_final[2]) + pad,
            max(doc_final[1], doc_final[3]) + pad,
        ))
        ox, oy = layer.offset
        box = (start[0] - ox, start[1] - oy, x - ox, y - oy)
        fill_color, stroke_color = self._shape_colors(tools)
        mode = tools.shape_mode
        fill = rgba(fill_color) if mode in ("fill", "both") else None
        outline = rgba(stroke_color) if mode in ("stroke", "both") else None
        radius = tools.shape_radius
        temp = Image.new("RGBA", layer.image.size, (0, 0, 0, 0))
        # Draw into a supersampled canvas covering just the shape's bounding
        # box (padded for the stroke width), then average it back down for
        # anti-aliased edges. ``shape.constrain`` is pure, so its result is the
        # geometry that actually gets drawn.
        final = shape.constrain(box, tools.shape_kind, shift)
        left = max(0, int(math.floor(min(final[0], final[2]))) - pad)
        top = max(0, int(math.floor(min(final[1], final[3]))) - pad)
        right = min(layer.image.width,
                    int(math.ceil(max(final[0], final[2]))) + pad)
        bottom = min(layer.image.height,
                     int(math.ceil(max(final[1], final[3]))) + pad)
        if right > left and bottom > top:
            scale = (antialias.scale_for(right - left, bottom - top)
                     if tools.shape_antialias else 1)
            sub = Image.new(
                "RGBA", ((right - left) * scale, (bottom - top) * scale),
                (0, 0, 0, 0),
            )
            shape.draw(
                ImageDraw.Draw(sub), tools.shape_kind,
                tuple((value - (left if index % 2 == 0 else top)) * scale
                      for index, value in enumerate(box)),
                fill=fill, outline=outline, width=width * scale,
                shift=shift, radius=radius * scale,
            )
            if scale > 1:
                sub = antialias.downscale(sub, scale)
            temp.paste(sub, (left, top))
        coverage = temp.getchannel("A")
        if tools.shape_opacity < 100:
            factor = tools.shape_opacity / 100.0
            coverage = coverage.point(lambda value: int(value * factor))
        clip = self._selection_mask(layer, ox, oy)
        if clip is not None:
            coverage = ImageChops.multiply(coverage, clip)
        layer.set_image(brush.composite_image(
            layer.image, temp.convert("RGB"), coverage,
            tools.shape_blend.lower(),
        ))
        layer.invalidate_filters()
        self.app.render()
        self.app.set_result(f"Added {tools.shape_kind} on '{layer.name}'.")

    def _shape_colors(self, tools):
        """Resolve the Shape tool's ``(fill, stroke)`` colours.

        Unset colours fall back to the main picker, so the defaults keep the
        old single-colour behaviour.
        """
        fill = tools.shape_fill_color or self.app.color
        stroke = tools.shape_stroke_color or self.app.color
        return fill, stroke

    def _update_shape_preview(self, end, shift):
        view = self.app.area.current_view()
        if view is None:
            return
        if end is None or self._shape_start is None:
            view.set_shape_preview(None)
            return
        start = self._shape_start
        tools = self.app.tools
        kind = tools.shape_kind
        box = shape.constrain((start[0], start[1], end[0], end[1]), kind, shift)
        fill, stroke = self._shape_colors(tools)
        color = stroke if tools.shape_mode in ("stroke", "both") else fill
        view.set_shape_preview({
            "kind": kind, "box": box, "color": color,
            "radius": tools.shape_radius,
        })

    # ------------------------------------------------------------------ wand
    def wand_at(self, x, y):
        """Select the region of similar colour under ``(x, y)``."""
        doc = self.app.current
        tools = self.app.tools
        width, height = doc.size
        if not (0 <= x < width and 0 <= y < height):
            return
        layer = self.app.active_layer()
        if layer is None:
            self.app.set_result("There is no layer to sample.")
            return
        ox, oy = layer.offset
        if tools.wand_sample == "image":
            image = doc.composite_transparent()
            seed = (x, y)
            origin = (0, 0)
        else:
            image = layer.composited_image()
            seed = (x - ox, y - oy)
            origin = (ox, oy)
        if not (0 <= seed[0] < image.width and 0 <= seed[1] < image.height):
            return
        region = fill.region_mask(
            image, seed, tolerance=tools.wand_tolerance,
            contiguous=tools.wand_contiguous,
        )
        mask = Image.new("L", doc.size, 0)
        mask.paste(region, origin)
        selection = Selection.from_mask(doc.size, mask)
        if selection.is_empty():
            self._set_selection(None)
            self.app.set_result("Nothing to select there.")
            return
        self._set_selection(selection)
        box = selection.bbox
        self.app.set_result(f"Selected {box[2] - box[0]} x {box[3] - box[1]}.")

    def segment_at(self, x, y):
        """Ask the coordinator to AI-segment the object under ``(x, y)``.

        The mask comes back asynchronously (it runs on the job queue), so this
        only hands the click to :meth:`PaintApp.start_segment`; the resulting
        selection is installed when the job finishes.
        """
        doc = self.app.current
        if doc is None:
            return
        width, height = doc.size
        if not (0 <= x < width and 0 <= y < height):
            return
        self.app.start_segment(x, y)

    # ------------------------------------------------------------ clone source
    def set_clone_source(self, x, y):
        self._clone_source = (x, y)
        self.sync_clone_source()
        self.app.set_result("Clone source set; paint to copy from it.")

    def clear_clone_source(self):
        self._clone_source = None
        self.sync_clone_source()
        self.app.set_result("Clone source cleared.")

    def sync_clone_source(self):
        """Show the source crosshair on the current view while cloning."""
        view = self.app.area.current_view()
        if view is None:
            return
        source = self._clone_source if self.app.tools.clone_tool else None
        view.set_clone_source_overlay(source)

    # ------------------------------------------------------------------ crop
    CROP_HANDLE_HIT = 8.0  # document-pixel pickup radius at zoom 1
    CROP_CURSORS = {
        "nw": "size_nw_se", "se": "size_nw_se",
        "ne": "size_ne_sw", "sw": "size_ne_sw",
        "n": "size_ns", "s": "size_ns",
        "e": "size_we", "w": "size_we",
        "move": "fleur",
    }

    def _crop_handles(self, box=None):
        """Handle positions ``(x, y)`` in document pixels for the crop box."""
        box = self._crop_box if box is None else box
        if box is None:
            return {}
        left, top, right, bottom = box
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        return {
            "nw": (left, top), "ne": (right, top),
            "se": (right, bottom), "sw": (left, bottom),
            "n": (cx, top), "e": (right, cy),
            "s": (cx, bottom), "w": (left, cy),
        }

    def _crop_hit(self, x, y, tolerance):
        """Return the handle under ``(x, y)``, ``"move"`` inside, or ``None``."""
        box = self._crop_box
        if box is None:
            return None
        for key, (hx, hy) in self._crop_handles(box).items():
            if abs(hx - x) <= tolerance and abs(hy - y) <= tolerance:
                return key
        left, top, right, bottom = box
        if left <= x <= right and top <= y <= bottom:
            return "move"
        return None

    def _crop_press(self, event, x, y):
        self._crop_handle = None
        self._crop_move_box = None
        if self._crop_box is not None:
            tolerance = self.CROP_HANDLE_HIT / max(event.widget.zoom, 1e-6)
            handle = self._crop_hit(x, y, tolerance)
            if handle is not None:
                self._crop_handle = handle
                self._crop_start = (x, y)
                if handle == "move":
                    self._crop_move_box = self._crop_box
                return
        self._crop_start = (x, y)
        self._update_crop_preview((x, y))

    def _crop_drag(self, x, y):
        if self._crop_handle is not None:
            self._update_crop_adjust(x, y)
            return
        if self._crop_start is None:
            return
        self._update_crop_preview((x, y))

    def _crop_release(self, x, y):
        if self._crop_handle is not None:
            self._update_crop_adjust(x, y)
            self._crop_handle = None
            self._crop_move_box = None
            self._crop_start = None
            return
        if self._crop_start is None:
            return
        self._update_crop_preview((x, y))
        self._crop_start = None

    def _crop_adjust_rect(self, x, y):
        """Resize the crop box for the active handle, or move it as a whole."""
        width, height = self.app.current.size
        handle = self._crop_handle
        if handle == "move":
            bx0, by0, bx1, by1 = self._crop_move_box
            sx, sy = self._crop_start
            box_w = bx1 - bx0
            box_h = by1 - by0
            left = min(max(0.0, bx0 + (x - sx)), max(0.0, width - box_w))
            top = min(max(0.0, by0 + (y - sy)), max(0.0, height - box_h))
            return (left, top, left + box_w, top + box_h)
        left, top, right, bottom = self._crop_box
        if "w" in handle:
            left = max(0.0, min(x, right - 1.0))
        if "e" in handle:
            right = min(float(width), max(x, left + 1.0))
        if "n" in handle:
            top = max(0.0, min(y, bottom - 1.0))
        if "s" in handle:
            bottom = min(float(height), max(y, top + 1.0))
        return (left, top, right, bottom)

    def _update_crop_adjust(self, x, y):
        box = self._crop_adjust_rect(x, y)
        self._crop_box = box
        view = self.app.area.current_view()
        if view is not None:
            view.set_crop_preview(box)

    def _crop_rect(self, end):
        start = self._crop_start
        if start is None:
            return self._crop_box or (0, 0, 0, 0)
        width, height = self.app.current.size
        left = max(0, min(start[0], end[0]))
        top = max(0, min(start[1], end[1]))
        right = min(width, max(start[0], end[0]))
        bottom = min(height, max(start[1], end[1]))
        return (left, top, right, bottom)

    def _update_crop_preview(self, end):
        box = self._crop_rect(end)
        self._crop_box = box
        view = self.app.area.current_view()
        if view is not None:
            view.set_crop_preview(box)

    def _hover_crop(self, event):
        """Show the resize/move cursor over the crop box's handles."""
        view = event.widget
        cursor = "crosshair"
        if self._crop_box is not None:
            x, y = self._doc_coords(event)
            tolerance = self.CROP_HANDLE_HIT / max(view.zoom, 1e-6)
            handle = self._crop_hit(x, y, tolerance)
            cursor = self.CROP_CURSORS.get(handle, "crosshair")
        try:
            view.config(cursor=cursor)
        except tk.TclError:
            view.config(cursor="crosshair")

    def apply_crop(self):
        box = self._crop_box
        if box is None or box[2] - box[0] < 1 or box[3] - box[1] < 1:
            self.app.set_result("Drag a crop rectangle first.")
            return
        self.app.crop_canvas(box)
        self.clear_crop()

    def clear_crop(self):
        """Drop the crop rectangle preview; returns whether there was one."""
        had_box = (
            self._crop_start is not None or self._crop_box is not None
            or self._crop_handle is not None or self._crop_move_box is not None
        )
        self._crop_start = None
        self._crop_box = None
        self._crop_handle = None
        self._crop_move_box = None
        view = self.app.area.current_view()
        if view is not None:
            view.set_crop_preview(None)
        return had_box

    def sync_crop(self):
        """Drop a pending crop preview when the Crop tool is no longer active."""
        if not self.app.tools.crop_tool:
            self.clear_crop()

    def on_drag(self, event):
        if self._preview_showing(event):
            return
        view = event.widget
        tools = self.app.tools
        x1, y1 = self._doc_coords(event)
        if tools.text_tool:
            if self._text_drag is not None:
                self._text_drag_move(x1, y1, bool(event.state & 0x1))
            return
        if tools.layout_tool:
            self.cursor_x, self.cursor_y = x1, y1
            self._layout_drag(x1, y1)
            return
        if tools.path_tool:
            self._path_drag(x1, y1)
            self.cursor_x, self.cursor_y = x1, y1
            self._apply_cursor(view)
            return
        if tools.vector_tool:
            self._vector_drag(event, x1, y1)
            self.cursor_x, self.cursor_y = x1, y1
            self._apply_cursor(view)
            return
        if tools.poses_tool:
            self._poses_drag(x1, y1, bool(event.state & 0x1))
            return
        if tools.liquify_tool:
            self._liquify_drag(x1, y1)
            self.cursor_x, self.cursor_y = x1, y1
            self._apply_cursor(view)
            return
        if tools.warp_tool:
            self._warp_drag(x1, y1)
            return
        if tools.puppet_tool:
            self._puppet_drag(x1, y1)
            return
        if tools.select_tool:
            if getattr(tools, "select_transform", False):
                self._selection_transform_drag(event, x1, y1)
            else:
                self._drag_select(x1, y1)
            return
        if tools.drag_tool:
            self._pan_to(event)
            return
        if tools.transform_tool:
            self._transform_drag(event, x1, y1)
            return
        if tools.move_tool:
            if self._guide_index is not None:
                self._drag_guide(x1, y1)
                return
            self._move_to(x1, y1)
            return
        if tools.gradient_tool:
            self._gradient_drag(x1, y1)
            return
        if tools.shape_tool:
            self._shape_drag(x1, y1, bool(event.state & 0x1))
            self.cursor_x, self.cursor_y = x1, y1
            self._apply_cursor(view)
            return
        if tools.crop_tool:
            self._crop_drag(x1, y1)
            return
        if self.last_x is None:
            return
        self._continue_paint(x1, y1, self._paint_speed(x1, y1, event))
        # Render the committed stroke live so the preview matches every brush
        # setting (opacity, hardness, flow, spacing, blend, tip) exactly.
        self.app.render()
        self.last_x, self.last_y = x1, y1
        self.cursor_x, self.cursor_y = x1, y1
        self._apply_cursor(view)

    def on_release(self, event):
        if self._preview_showing(event):
            return
        if self._guide_index is not None:
            self._end_guide(event)
            return
        if self.app.tools.layout_tool:
            self._layout_release()
            return
        if self.app.tools.text_tool:
            if self._text_drag is not None:
                self._text_drag = None
                self._refresh_text_edit()
            return
        if self.app.tools.path_tool:
            self._path_release()
            return
        if self.app.tools.vector_tool:
            self._vector_release()
            return
        if self.app.tools.poses_tool:
            self._poses_release()
            return
        if self.app.tools.liquify_tool:
            self._liquify_release()
            return
        if self.app.tools.warp_tool:
            self._warp_release()
            return
        if self.app.tools.puppet_tool:
            self._puppet_release()
            return
        if self.app.tools.gradient_tool:
            x, y = self._doc_coords(event)
            self._gradient_release(x, y)
            return
        if self.app.tools.shape_tool:
            x, y = self._doc_coords(event)
            self._shape_release(x, y, bool(event.state & 0x1))
            return
        if self.app.tools.crop_tool:
            x, y = self._doc_coords(event)
            self._crop_release(x, y)
            return
        if self.app.tools.transform_tool:
            self._transform_release()
            return
        if self._selection_transform_active():
            self._selection_transform_release()
            return
        if self.app.tools.select_tool:
            self._release_select()
            return
        if self._pan_view is not None:
            self._end_pan()
            return
        self._end_paint()
        self.last_x = None
        self.last_y = None
        self._stroke_open = False
        self._stroke = None
        self._strokes = []
        self._end_move()
        self.app.render()

    # ------------------------------------------------------------------ select
    def _selection(self):
        view = self.app.area.current_view()
        return view.selection if view is not None else None

    def _set_selection(self, selection):
        view = self.app.area.current_view()
        if view is not None:
            view.set_selection(selection)

    def _set_selection_label(self, box):
        view = self.app.area.current_view()
        if view is not None:
            view.set_selection_label(box)

    def reapply_select_radius(self):
        """Re-round the current rectangle selection for the Corners setting.

        The Corner radius applies live so an already-drawn marquee updates as
        the slider moves, instead of only affecting the next drag.
        """
        tools = self.app.tools
        if getattr(tools, "select_mode", "rect") != "rect":
            return
        if getattr(tools, "select_transform", False):
            return
        selection = self._selection()
        if selection is None or getattr(selection, "rect", False) is not True \
                or selection.is_empty():
            return
        box = selection.bbox
        self._set_selection(
            Selection.rectangle(self.app.current.size, box, tools.select_radius))

    def _rect_from(self, a, b):
        width, height = self.app.current.size
        left = max(0, min(a[0], b[0]))
        top = max(0, min(a[1], b[1]))
        right = min(width, max(a[0], b[0]))
        bottom = min(height, max(a[1], b[1]))
        return (left, top, right, bottom)

    def _begin_select(self, x, y):
        self._set_selection_label(None)
        selection = self._selection()
        if selection is not None and not selection.is_empty() \
                and selection.contains(x, y):
            self._begin_float(x, y, selection)
            return
        size = self.app.current.size
        if self.app.tools.select_mode == "lasso":
            self._sel_mode = "lasso"
            self._sel_points = [(x, y)]
            self._set_selection(Selection.lasso(size, [(x, y), (x, y), (x, y)]))
        else:
            self._sel_mode = "rect"
            self._sel_origin = (x, y)
            box = self._rect_from((x, y), (x, y))
            self._set_selection(
                Selection.rectangle(
                    size, box, getattr(self.app.tools, "select_radius", 0)))
            self._set_selection_label(box)

    def _drag_select(self, x, y):
        if self._sel_mode == "move":
            self._move_float(x, y)
            return
        size = self.app.current.size
        if self._sel_mode == "lasso":
            last = self._sel_points[-1]
            if abs(x - last[0]) + abs(y - last[1]) >= 2:
                self._sel_points.append((x, y))
                points = self._sel_points
                if len(points) < 3:
                    points = points + [points[-1]]
                self._set_selection(Selection.lasso(size, points))
        elif self._sel_mode == "rect":
            box = self._rect_from(self._sel_origin, (x, y))
            self._set_selection(
                Selection.rectangle(
                    size, box, getattr(self.app.tools, "select_radius", 0)))
            self._set_selection_label(box)

    # ---------------------------------------------------------------- layout
    LAYOUT_HANDLE_HIT = 8.0  # document-pixel pickup radius at zoom 1

    def _layout_box_handles(self, box):
        """Handle positions ``(x, y)`` in document pixels for a layout box."""
        left, top, right, bottom = box
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        return {
            "nw": (left, top), "ne": (right, top),
            "se": (right, bottom), "sw": (left, bottom),
            "n": (cx, top), "e": (right, cy),
            "s": (cx, bottom), "w": (left, cy),
        }

    def _layout_interior_rows(self, x, y):
        """Rows whose interior contains ``(x, y)``, topmost (last drawn) first."""
        tools = self.app.tools
        if not tools.layout_model.enabled:
            return []
        hits = []
        for row in reversed(tools.layout_model.rows):
            box = row.get("bbox")
            if not box:
                continue
            left, top, right, bottom = box
            if left <= x <= right and top <= y <= bottom:
                hits.append(row)
        return hits

    def _layout_hit(self, x, y, tolerance):
        """The ``(row, handle)`` under ``(x, y)``, or ``(None, None)``.

        Handles win over a box interior, and later rows win over earlier ones.
        Only hit-tested while the overlay is live (tool active and enabled), so
        an invisible box is never grabbed.
        """
        tools = self.app.tools
        if not tools.layout_model.enabled:
            return None, None
        rows = tools.layout_model.rows
        for row in reversed(rows):
            box = row.get("bbox")
            if not box:
                continue
            for key, (hx, hy) in self._layout_box_handles(box).items():
                if abs(hx - x) <= tolerance and abs(hy - y) <= tolerance:
                    return row, key
        hits = self._layout_interior_rows(x, y)
        if hits:
            return hits[0], "move"
        return None, None

    def _layout_cycle_row(self, x, y, row, handle, tolerance):
        """The row to edit, stepping to the box beneath on a repeated click.

        Boxes often overlap, and a plain hit always finds the topmost. Pressing
        the same spot again steps down through the stack (wrapping at the
        bottom), so every box underneath can be reached and moved or resized.
        """
        if handle != "move":
            self._layout_last_pick = None
            return row
        hits = self._layout_interior_rows(x, y)
        last = self._layout_last_pick
        if (last is not None and abs(last[0] - x) <= tolerance
                and abs(last[1] - y) <= tolerance and last[2] in hits):
            index = hits.index(last[2])
            row = hits[(index + 1) % len(hits)]
        self._layout_last_pick = (x, y, row)
        return row

    def _layout_adjust_box(self, start_box, handle, start_xy, end_xy):
        """The new box for a move/resize gesture, clamped to the canvas."""
        width, height = self.app.current.size
        left, top, right, bottom = (float(v) for v in start_box)
        if handle == "move":
            dx = end_xy[0] - start_xy[0]
            dy = end_xy[1] - start_xy[1]
            box_w = right - left
            box_h = bottom - top
            left = min(max(0.0, left + dx), max(0.0, width - box_w))
            top = min(max(0.0, top + dy), max(0.0, height - box_h))
            return (left, top, left + box_w, top + box_h)
        if "w" in handle:
            left = max(0.0, min(end_xy[0], right - 1.0))
        if "e" in handle:
            right = min(float(width), max(end_xy[0], left + 1.0))
        if "n" in handle:
            top = max(0.0, min(end_xy[1], bottom - 1.0))
        if "s" in handle:
            bottom = min(float(height), max(end_xy[1], top + 1.0))
        return (left, top, right, bottom)

    def _layout_press(self, x, y):
        """Grab an existing box (move/resize) or begin drawing a new one."""
        view = self.app.area.current_view()
        if view is None or self.app.current is None:
            return
        tolerance = self.LAYOUT_HANDLE_HIT / max(view.zoom, 1e-6)
        row, handle = self._layout_hit(x, y, tolerance)
        if row is not None:
            row = self._layout_cycle_row(x, y, row, handle, tolerance)
            self._layout_edit = (row, handle, tuple(row["bbox"]), (x, y))
            self._set_selection(None)
            self._set_selection_label(tuple(row["bbox"]))
            return
        self._layout_last_pick = None
        self._layout_origin = (x, y)
        size = self.app.current.size
        box = self._rect_from((x, y), (x, y))
        view.set_selection(Selection.rectangle(size, box, 0))
        view.set_selection_label(box)

    def _layout_drag(self, x, y):
        view = self.app.area.current_view()
        if view is None or self.app.current is None:
            return
        if self._layout_edit is not None:
            row, handle, start_box, start_xy = self._layout_edit
            box = self._layout_adjust_box(start_box, handle, start_xy, (x, y))
            row["bbox"] = [int(round(v)) for v in box]
            self._set_selection_label(tuple(row["bbox"]))
            self.app._sync_layout_overlay()
            return
        if self._layout_origin is None:
            return
        size = self.app.current.size
        box = self._rect_from(self._layout_origin, (x, y))
        view.set_selection(Selection.rectangle(size, box, 0))
        view.set_selection_label(box)

    def _layout_release(self):
        view = self.app.area.current_view()
        self._set_selection_label(None)
        if self._layout_edit is not None:
            self._layout_edit = None
            self.app._on_layout_change()
            self.app.tools.refresh_layout_settings()
            self.app.set_result("Moved layout item.", "success")
            return
        origin = self._layout_origin
        self._layout_origin = None
        if view is not None:
            view.set_selection(None)
        if origin is None or self.cursor_x is None:
            return
        box = self._rect_from(origin, (self.cursor_x, self.cursor_y))
        if box[2] - box[0] < 4 or box[3] - box[1] < 4:
            self.app.set_result("Drag a larger box to add a layout item.")
            return
        self.app.add_layout_box(box)

    def _hover_layout(self, event):
        """Show the move/resize cursor over an existing box's handles."""
        view = event.widget
        cursor = "crosshair"
        x, y = self._doc_coords(event)
        tolerance = self.LAYOUT_HANDLE_HIT / max(view.zoom, 1e-6)
        row, handle = self._layout_hit(x, y, tolerance)
        if handle is not None:
            cursor = self.CROP_CURSORS.get(handle, "crosshair")
        try:
            view.config(cursor=cursor)
        except tk.TclError:
            view.config(cursor="crosshair")

    def _begin_float(self, x, y, selection):
        """Begin dragging an active selection.

        Dragging inside the marquee repositions the selection itself; the layer
        pixels are left untouched (use cut/paste to move pixels).
        """
        self._sel_mode = "move"
        self._float_selection = selection
        self._float_start = (x, y)

    def _move_float(self, x, y):
        if self._float_selection is None:
            return
        dx = x - self._float_start[0]
        dy = y - self._float_start[1]
        self._set_selection(self._float_selection.moved(dx, dy))

    def _release_select(self):
        if self._sel_mode == "move":
            self._end_float()
            self.app.set_result("Moved selection.")
            return
        self._set_selection_label(None)
        if self._sel_mode in ("rect", "lasso"):
            selection = self._selection()
            if selection is None or selection.is_empty():
                self._set_selection(None)
                self.app.set_result("")
            else:
                left, top, right, bottom = selection.bbox
                self.app.set_result(
                    f"Selected {right - left} x {bottom - top}."
                )
        self._sel_mode = None

    def _end_float(self):
        self._float_selection = None
        self._float_start = None
        self._sel_mode = None

    # --------------------------------------------------------------- path
    PATH_CLOSE_HIT = 8
    PATH_HIT = 6
    # Minimum seconds between repaints while dragging a committed path. A drag
    # paints a draft (non-antialiased) frame, whose re-raster is cheap, so this
    # only trims redundant work on layers with many path objects.
    PATH_RENDER_INTERVAL = 0.05

    def _path_hit_anchor(self, path, x, y, hit):
        """Return the index of the anchor nearest ``(x, y)`` within ``hit``."""
        best = None
        best_distance = hit
        for index, (ax, ay) in enumerate(path.anchors):
            distance = math.hypot(x - ax, y - ay)
            if distance <= best_distance:
                best_distance = distance
                best = index
        return best

    def _path_hit_handle(self, path, x, y, hit):
        """Return the ``(side, index)`` of the handle nearest ``(x, y)``."""
        best = None
        best_distance = hit
        for index, (ax, ay) in enumerate(path.anchors):
            for side, point in (("in", path.handles_in[index]),
                                ("out", path.handles_out[index])):
                if abs(point[0] - ax) < 1e-6 and abs(point[1] - ay) < 1e-6:
                    continue
                distance = math.hypot(x - point[0], y - point[1])
                if distance <= best_distance:
                    best_distance = distance
                    best = (side, index)
        return best

    def _path_press(self, event, x, y):
        view = event.widget
        view.focus_set()
        scale = max(view.zoom, 1e-6)
        path = self._path
        if path is not None and len(path) > 0:
            handle = self._path_hit_handle(path, x, y, self.PATH_HIT / scale)
            if handle is not None:
                self._path_handle = handle
                self._path_press_origin = (x, y)
                self._path_moved = False
                return
            index = self._path_hit_anchor(
                path, x, y, self.PATH_CLOSE_HIT / scale
            )
            if index is not None:
                if event.state & ALT_STATE_MASK:
                    self._path_handle = ("out", index)
                    self._path_press_origin = (x, y)
                    self._path_moved = False
                    return
                self._path_move = index
                self._path_press_origin = (x, y)
                self._path_moved = False
                if index == 0 and len(path) >= 2:
                    self._path_close_pending = True
                return
            # Dragging a committed path's fill or ink moves the whole object.
            if self._path_edit is not None:
                spec = self._path_edit_spec()
                if spec is not None and pathspec.hit(
                        spec, x, y, self.PATH_HIT / scale):
                    self._begin_path_move(x, y)
                    return
        # A press that missed the path being edited commits it and starts a new
        # one, matching how the Text tool commits and starts fresh.
        if self._path_edit is not None:
            self.finish_path()
        if self._path is None or len(self._path) == 0:
            # With nothing pending, clicking an existing path re-opens it for
            # editing (as clicking text does), otherwise a new path begins.
            hit = self._pick_path(x, y, self.PATH_HIT / scale)
            if hit is not None:
                layer, index = hit
                self.app._on_path_select(self.app.current.layers.index(layer),
                                         index)
                self._begin_path_edit(layer, index)
                return
            self._path = Path()
            self._path_view = view
        self._path_dragging = self._path.add(x, y)
        self._update_path_preview(cursor=(x, y))

    def _path_drag(self, x, y):
        path = self._path
        if path is None:
            return
        if self._path_press_origin is not None and not self._path_moved:
            ox, oy = self._path_press_origin
            if abs(x - ox) > 2 or abs(y - oy) > 2:
                self._path_moved = True
                self._path_close_pending = False
        if self._path_move_origin is not None:
            self._translate_path(x, y)
        elif self._path_handle is not None:
            side, index = self._path_handle
            ax, ay = path.anchors[index]
            if side == "out":
                path.set_out_handle(index, x, y)
            else:
                path.set_out_handle(index, 2.0 * ax - x, 2.0 * ay - y)
        elif self._path_move is not None:
            path.move(self._path_move, x, y)
        elif self._path_dragging is not None:
            path.set_out_handle(self._path_dragging, x, y)
        else:
            return
        # A committed path is drawn from its spec, so the edit must be written
        # back and repainted; a pending path only needs the overlay refreshed.
        if self._path_edit is not None:
            self._apply_path_live(throttle=True)
        self._update_path_preview(cursor=(x, y))

    def _begin_path_move(self, x, y):
        """Capture a committed path's geometry to drag it as a whole."""
        path = self._path
        self._path_move_origin = (x, y)
        self._path_move_start = (
            [tuple(point) for point in path.anchors],
            [tuple(point) for point in path.handles_in],
            [tuple(point) for point in path.handles_out],
        )

    def _translate_path(self, x, y):
        """Move the dragged path by the delta from its press point."""
        start = self._path_move_start
        if not start:
            return
        ox, oy = self._path_move_origin
        dx = x - ox
        dy = y - oy
        path = self._path
        path.anchors = [(p[0] + dx, p[1] + dy) for p in start[0]]
        path.handles_in = [(p[0] + dx, p[1] + dy) for p in start[1]]
        path.handles_out = [(p[0] + dx, p[1] + dy) for p in start[2]]

    def _path_release(self):
        """End a path drag, closing the path when the first anchor was tapped."""
        if self._path is not None and self._path_close_pending \
                and not self._path_moved:
            self._clear_path_edit()
            self._path.closed = True
            self.finish_path()
            return
        # A drag paints draft (non-antialiased) frames, so land the final,
        # full-quality frame now.
        if self._path_draft:
            self._apply_path_live()
        self._path_dragging = None
        self._clear_path_edit()

    def _clear_path_edit(self):
        self._path_move = None
        self._path_handle = None
        self._path_press_origin = None
        self._path_moved = False
        self._path_close_pending = False
        self._path_move_origin = None
        self._path_move_start = None
        self._path_draft = False

    def _path_right_click(self, event):
        """Delete the path anchor under a right-click."""
        view = event.widget
        x, y = self._doc_coords(event)
        path = self._path
        if path is None:
            return "break"
        hit = self.PATH_HIT / max(view.zoom, 1e-6)
        index = self._path_hit_anchor(path, x, y, hit)
        if index is None:
            return "break"
        # A committed path is a valid object only with two anchors or more, so
        # refuse a removal that would collapse it (a new path may drop freely).
        if self._path_edit is not None and len(path) <= 2:
            self.app.set_result("A path needs at least two points.")
            return "break"
        path.remove(index)
        self._path_dragging = None
        self._clear_path_edit()
        if len(path) == 0:
            self._path = None
        if self._path_edit is not None:
            self._apply_path_live()
        self._update_path_preview()
        self.app.set_result("Removed a path point.")
        return "break"

    def _update_path_preview(self, cursor=None):
        view = self._path_view or self.app.area.current_view()
        if view is None:
            return
        if self._path is None or len(self._path) == 0:
            view.set_path(None)
            return
        tools = self.app.tools
        editing = self._path_edit is not None
        # While editing a committed path its pixels are drawn by the compositor,
        # so the overlay shows only the handles (no second fill/stroke).
        data = self._path.preview(cursor=None if editing else cursor)
        data["show_handles"] = bool(tools.path_show_handles)
        data["mode"] = "handles" if editing else tools.path_mode
        # Carry the current Output settings so the overlay previews the fill,
        # stroke and outline the path would produce, not just its outline.
        data["fill_color"] = tools.path_fill_color or self.app.color
        data["stroke_color"] = tools.path_stroke_color or self.app.color
        data["width"] = tools.brush_size
        data["outline"] = tools.path_outline
        data["cap"] = tools.path_cap
        data["join"] = tools.path_join
        data["opacity"] = tools.path_opacity
        view.set_path(data)

    def refresh_path(self):
        """Redraw the path overlay (and committed spec) after a settings change."""
        if self._path_edit is not None and self._path is not None:
            self._apply_path_live()
        cursor = None
        if self.cursor_x is not None and self.cursor_y is not None:
            cursor = (self.cursor_x, self.cursor_y)
        self._update_path_preview(cursor=cursor)

    # ---------------------------------------------------- committed path edit
    def _path_edit_spec(self):
        """The spec of the path currently open for editing, or ``None``."""
        edit = self._path_edit
        if edit is None:
            return None
        layer = edit["layer"]
        index = edit["index"]
        if 0 <= index < len(layer.paths):
            return layer.paths[index]
        return None

    def _mark_path_dirty(self):
        """Record one undo entry for the first change of a path edit session."""
        edit = self._path_edit
        if edit is None or edit.get("pushed"):
            return
        edit["pushed"] = True
        self.app._push_undo("Edit path")

    def _apply_path_style(self, spec):
        """Mirror the Path tool's current settings onto ``spec`` (in place)."""
        if self._path_edit is None:
            return
        spec.update(self.app.tools.path_style())

    def _apply_path_live(self, throttle=False):
        """Write the edited geometry and style into the spec and repaint.

        During a drag (``throttle=True``) geometry changes are written every
        event, but the repaint is throttled and draws a *draft* frame with
        anti-aliasing off: the antialiased re-raster supersamples the whole path
        bounding box and averages it down, which costs hundreds of milliseconds
        on a large path and would stall the drag. A release (``throttle=False``)
        restores the tool's anti-aliasing and repaints the final, exact frame.
        """
        edit = self._path_edit
        spec = self._path_edit_spec()
        if edit is None or spec is None or self._path is None:
            return
        self._mark_path_dirty()
        pathspec.apply_geometry(spec, self._path)
        self._apply_path_style(spec)
        if throttle:
            spec["antialias"] = False
        edit["layer"].invalidate_paths()
        now = time.monotonic()
        if throttle and now - self._path_render_at < self.PATH_RENDER_INTERVAL:
            return
        self._path_render_at = now
        self._path_draft = throttle
        self.app.render()

    @staticmethod
    def _path_at(layer, x, y, tolerance=0.0):
        """Index of the topmost path object under ``(x, y)``, or ``None``."""
        for index in range(len(layer.paths) - 1, -1, -1):
            if pathspec.hit(layer.paths[index], x, y, tolerance):
                return index
        return None

    def _pick_path(self, x, y, tolerance=0.0):
        """The path object under ``(x, y)`` across visible layers, or ``None``.

        Returns ``(layer, index)``. Layers are searched front to back so the
        object drawn on top wins; hidden layers (or hidden groups) are skipped.
        """
        doc = self.app.current
        if doc is None:
            return None
        for layer in reversed(doc.layers):
            if not self._layer_visible(layer):
                continue
            index = self._path_at(layer, x, y, tolerance)
            if index is not None:
                return layer, index
        return None

    def _begin_path_edit(self, layer, index):
        """Re-open a committed path object for editing on the canvas."""
        spec = layer.paths[index]
        self.finish_path()
        if spec not in layer.paths:
            return
        index = layer.paths.index(spec)
        # Mirror the path's settings into the panel before the edit session
        # opens, so the panel's callbacks (e.g. the size row) cannot write back
        # to the object and record a spurious undo entry.
        self.app.tools.set_path_style(spec)
        self._path = pathspec.to_path(spec)
        self._path_view = self.app.area.current_view()
        self._path_edit = {
            "layer": layer, "index": index, "pushed": False,
            "original": pathspec.clone(spec),
        }
        # A fresh session paints its first drag frame right away rather than
        # inheriting the previous session's throttle timestamp.
        self._path_render_at = 0.0
        self._path_draft = False
        self.app._path_sel = (self.app.current.layers.index(layer), index)
        self._update_path_preview()
        self.app._refresh_layer_list()
        self.app.render()
        self.app.set_result(
            "Editing path; drag to move, anchors/handles to reshape, "
            "double-click to add, right-click to remove, Enter commits, Esc cancels."
        )

    def edit_path_object(self, layer, index):
        """Re-open a committed path object for editing from the layer panel."""
        if not (0 <= index < len(layer.paths)):
            return
        if layer.is_locked():
            self.app._active_layer_locked()
            return
        view = self.app.area.current_view()
        if view is None:
            return
        if not self.app.tools.path_tool:
            self.app.tools.select("path")
        self._begin_path_edit(layer, index)

    def _path_pick_segment(self, path, x, y, tolerance):
        """``(index, t)`` of the bezier segment nearest ``(x, y)``, or ``None``."""
        anchors = path.anchors
        count = len(anchors)
        if count < 2:
            return None
        indices = list(range(count - 1))
        if path.closed:
            indices.append(count - 1)
        best = None
        best_distance = None
        for index in indices:
            target = (index + 1) % count
            samples = pathspec._cubic(
                anchors[index], path.handles_out[index],
                path.handles_in[target], anchors[target], 20,
            )
            for step, (px, py) in enumerate(samples, start=1):
                distance = math.hypot(x - px, y - py)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best = (index, step / 20.0)
        if best is None or best_distance > tolerance:
            return None
        return best

    def _path_insert_anchor(self, x, y, view):
        """Split the nearest segment of the edited path at ``(x, y)``."""
        path = self._path
        if path is None or len(path.anchors) < 2:
            return False
        tolerance = (self.PATH_CLOSE_HIT * 2) / max(view.zoom, 1e-6)
        picked = self._path_pick_segment(path, x, y, tolerance)
        if picked is None:
            return False
        index, t = picked
        count = len(path.anchors)
        target = (index + 1) % count
        p0 = path.anchors[index]
        p1 = path.handles_out[index]
        p2 = path.handles_in[target]
        p3 = path.anchors[target]

        def lerp(a, b, ratio):
            return (a[0] + (b[0] - a[0]) * ratio, a[1] + (b[1] - a[1]) * ratio)

        p01 = lerp(p0, p1, t)
        p12 = lerp(p1, p2, t)
        p23 = lerp(p2, p3, t)
        p012 = lerp(p01, p12, t)
        p123 = lerp(p12, p23, t)
        middle = lerp(p012, p123, t)
        path.handles_out[index] = p01
        path.handles_in[target] = p23
        at = index + 1
        path.anchors.insert(at, middle)
        path.handles_in.insert(at, p012)
        path.handles_out.insert(at, p123)
        if self._path_edit is not None:
            self._apply_path_live()
        self._update_path_preview()
        self.app.set_result("Added a path point.")
        return True

    def _reset_origin_double_click(self, event):
        """Double-click the rotation-origin marker to send it back to centre."""
        view = event.widget
        x, y = self._doc_coords(event)
        if self.app.tools.text_tool and self._text is not None:
            if not (event.state & 0x8):
                return False
            spec = self._text_spec()
            if spec is None:
                return False
            tolerance = self.TEXT_HANDLE_HIT / max(view.zoom, 1e-6)
            handle = textrender.hit_handle(
                spec, x, y, tolerance, self._text.get("rot_pivot"), True)
            if handle != "pivot":
                return False
            self._text["rot_pivot"] = None
            self._text_drag = None
            self._refresh_text_edit()
            self.app.set_result("Reset the rotation origin to the centre.")
            return True
        if self.app.tools.transform_tool and self._transform_ready():
            if self.gizmo.hit(x, y, self.gizmo.HANDLE_HIT / view.zoom) != "pivot":
                return False
            self.gizmo.reset_rotation_pivot()
            self.gizmo.finish()
            view.set_gizmo(self.gizmo.geometry())
            self.app.render()
            self.app.set_result("Reset the rotation origin to the centre.")
            return True
        if self._selection_transform_active() and self.sel_gizmo.active:
            if self.sel_gizmo.hit(
                    x, y, self.sel_gizmo.HANDLE_HIT / view.zoom) != "pivot":
                return False
            self.sel_gizmo.reset_rotation_pivot()
            self.sel_gizmo.finish()
            view.set_gizmo(self.sel_gizmo.geometry())
            self.app.render()
            self.app.set_result("Reset the rotation origin to the centre.")
            return True
        return False

    def on_double_click(self, event):
        if self._preview_showing(event):
            return
        if self._reset_origin_double_click(event):
            return "break"
        if self.app.tools.text_tool and self._text is not None:
            x, y = self._doc_coords(event)
            self._select_text_unit(x, y, word=True)
            return "break"
        if self.app.tools.vector_tool:
            if getattr(self.app.tools, "vector_mode", "draw") == "edit":
                x, y = self._doc_coords(event)
                if self._vector_add_anchor(x, y, event.widget):
                    return "break"
            return
        if not self.app.tools.path_tool:
            return
        if self._path_edit is not None and self._path is not None:
            x, y = self._doc_coords(event)
            if self._path_insert_anchor(x, y, event.widget):
                return "break"
        if self._path_move is None and self._path_handle is None:
            self._drop_duplicate_anchor()
        self.finish_path()
        return "break"

    def on_triple_click(self, event):
        if self._preview_showing(event):
            return
        if self.app.tools.text_tool and self._text is not None:
            x, y = self._doc_coords(event)
            self._select_text_unit(x, y, word=False)
            return "break"

    def on_path_return(self, event):
        if self._preview_showing(event):
            return
        if self._text is not None:
            if event.state & 0x1:
                self._insert_text("\n")
            else:
                self.commit_text()
            return "break"
        if self.app.tools.crop_tool and self._crop_box is not None:
            self.apply_crop()
            return "break"
        if not self.app.tools.path_tool or self._path is None:
            return
        self.finish_path()
        return "break"

    # ----------------------------------------------------------------- vector
    VECTOR_HIT = 7
    CONNECT_TOLERANCE = 6.0
    WARP_HIT = 9
    PUPPET_HIT = 9
    PUPPET_HANDLE = 22
    POSE_HIT = 10

    def _vector_layer(self, create):
        """The vector layer the tool edits.

        Returns the active layer when it is a vector layer; when ``create`` and
        it is not, inserts a fresh vector layer above the active one and makes
        it active, so drawing always lands on a vector layer. Reports and
        returns ``None`` when there is no usable layer.
        """
        if self.app.current is None:
            self.app.set_result("There is no canvas. Add a canvas first.")
            return None
        layer = self.app.active_layer()
        if layer is not None and getattr(layer, "vector", False):
            if layer.is_locked():
                self.app.set_result(f"Vector layer '{layer.name}' is locked.")
                return None
            return layer
        if not create:
            return None
        doc = self.app.current
        index = doc.layers.index(layer) + 1 if layer is not None else len(doc.layers)
        new_layer = doc.add_layer(name="Vector Layer", index=index, vector=True)
        self.app.active = doc.layers.index(new_layer)
        self.app._refresh_layer_list()
        self.app.set_result("Added a vector layer.")
        return new_layer

    def _vector_missing_layer(self):
        self.app.set_result(
            "Select a vector layer to edit lines (Layer > New Vector Layer)."
        )

    def _vector_items(self, layer):
        """Every editable vector item: the strokes, then the compound fills.

        Selection indices address this combined list, so the anchor/transform
        tools work the same on a line and on a filled shape.
        """
        if layer is None:
            return []
        return list(layer.strokes) + list(getattr(layer, "fills", []) or [])

    def _vector_item(self, layer, index):
        items = self._vector_items(layer)
        if index is None or not (0 <= index < len(items)):
            return None
        return items[index]

    def _vector_is_fill(self, layer, index):
        """True when combined index ``index`` refers to a compound fill."""
        return (index is not None and layer is not None
                and index >= len(layer.strokes))

    def _vector_remove_item(self, layer, index):
        """Delete combined index ``index`` from the right backing list."""
        if self._vector_is_fill(layer, index):
            del layer.fills[index - len(layer.strokes)]
        else:
            del layer.strokes[index]

    def _vector_hit_stroke(self, layer, x, y, tolerance):
        """Index of the topmost vector item under ``(x, y)``, or ``None``."""
        items = self._vector_items(layer)
        for index in range(len(items) - 1, -1, -1):
            if vector.item_hit(items[index], x, y, tolerance):
                return index
        return None

    def _vector_press(self, event, x, y):
        mode = getattr(self.app.tools, "vector_mode", "draw")
        if mode == "draw":
            self._vector_begin_draw(x, y)
        elif mode == "edit":
            self._vector_edit_press(event, x, y)
        elif mode == "width":
            self._vector_width_begin(x, y)
        else:
            self._vector_erase_begin(x, y)

    def _vector_drag(self, event, x, y):
        mode = getattr(self.app.tools, "vector_mode", "draw")
        if mode == "draw":
            self._vector_draw_to(x, y)
        elif mode == "edit":
            self._vector_edit_drag(event, x, y)
        elif mode == "width":
            self._vector_width_apply(x, y)
        else:
            self._vector_erase_apply(x, y)

    def _vector_release(self):
        mode = getattr(self.app.tools, "vector_mode", "draw")
        if mode == "draw":
            self._vector_finish_draw()
            return
        if self._vector_marquee is not None:
            self._finish_vector_marquee()
        self._vector_handle = None
        self._vector_handle_drag = None
        self._vector_gesture = None
        self._vector_dirty = False
        self._sync_vector_selection()
        self._update_vector_overlay()

    # --------------------------------------------------------------- draw
    def _vector_begin_draw(self, x, y):
        layer = self._vector_layer(create=True)
        if layer is None:
            return
        self._vector_cur = layer
        self._vector_sel = None
        self._vector_points = [[x, y]]
        self._update_vector_overlay(cursor=(x, y))
        self.app.render()

    def _vector_draw_to(self, x, y):
        points = self._vector_points
        if points is None:
            return
        last = points[-1]
        if abs(x - last[0]) + abs(y - last[1]) < 1.0:
            return
        points.append([x, y])
        self._update_vector_overlay(cursor=(x, y))
        self.app.render()

    def _vector_finish_draw(self):
        points = self._vector_points
        self._vector_points = None
        layer = self._vector_cur
        self._vector_cur = None
        view = self.app.area.current_view()
        if view is not None:
            view.set_vector_overlay(None)
        if not points or layer is None or layer not in self.app.current.layers:
            self._vector_sel = None
            return
        tools = self.app.tools
        opacity = max(
            1, min(100, int(getattr(tools, "vector_opacity", 100)))) / 100.0
        color = getattr(tools, "vector_color", None) or self.app.color
        cap = getattr(tools, "vector_cap", "round")
        join = getattr(tools, "vector_join", "round")
        aa = bool(getattr(tools, "vector_antialias", True))
        smooth = int(getattr(tools, "vector_smooth", 0))
        detail = max(1, int(getattr(tools, "vector_detail", 15))) / 10.0
        smoothed = vector.smooth(points, smooth)
        merged = vector.simplify(smoothed, distance=detail)
        mappings = symmetry.transforms(
            self.app.symmetry_guide(), self.app.current.size
        )
        specs = []
        for mapping in mappings:
            spec = vector.new_stroke(
                [mapping(px, py) for px, py in merged],
                tools.brush_size, color, opacity=opacity, cap=cap, join=join,
                antialias=aa,
            )
            if spec is not None:
                specs.append(spec)
        if not specs:
            self._vector_sel = None
            return
        self.app._push_undo("Vector stroke")
        layer.strokes.extend(specs)
        layer.invalidate_vectors()
        self._vector_sel = len(layer.strokes) - 1
        self._vector_selection = [self._vector_sel]
        self._vector_cur = layer
        self._sync_vector_selection()
        self.app.render()
        if len(specs) > 1:
            self.app.set_result(
                f"Drew {len(specs)} mirrored vector lines on '{layer.name}'."
            )
        else:
            self.app.set_result(f"Drew a vector line on '{layer.name}'.")

    # --------------------------------------------------------------- edit
    def _vector_edit_press(self, event, x, y):
        layer = self._vector_layer(create=False)
        if layer is None:
            self._vector_missing_layer()
            return
        self._vector_cur = layer
        self._vector_dirty = False
        self._vector_marquee = None
        self._vector_handle = None
        self._vector_gesture = None
        self._vector_handle_drag = None
        zoom = max(event.widget.zoom, 1e-6)
        tolerance = self.VECTOR_HIT / zoom
        ctrl = bool(event.state & 0x4)
        shift = bool(event.state & 0x1)
        total = len(self._vector_items(layer))
        self._vector_selection = [
            i for i in self._vector_selection if 0 <= i < total
        ]

        handle = None
        if getattr(self.app.tools, "vector_show_anchors", True):
            handle = self._vector_handle_at(layer, x, y, zoom)
        if handle is not None:
            self._vector_handle = handle
            self._capture_vector_gesture(layer, x, y, handle)
            self._update_vector_overlay()
            return

        selected = self._vector_item(layer, self._vector_sel)
        if (getattr(self.app.tools, "vector_show_anchors", True)
                and selected is not None and not ctrl and not shift):
            grabbed = vector.item_handle_hit(selected, x, y, tolerance)
            if grabbed is not None:
                self._vector_handle_drag = grabbed
                self._update_vector_overlay()
                return

        if (getattr(self.app.tools, "vector_show_anchors", True)
                and selected is not None):
            index = vector.item_anchor_hit(selected, x, y, tolerance)
            if index is not None and not ctrl and not shift:
                self._vector_anchor = index
                self._update_vector_overlay()
                return

        hit = self._vector_hit_stroke(layer, x, y, tolerance)
        if hit is None:
            self._vector_marquee = (x, y)
            self._vector_marquee_end = (x, y)
            self._vector_anchor = None
            if not ctrl and not shift:
                self._vector_selection = []
                self._vector_sel = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            return
        if ctrl or shift:
            if hit in self._vector_selection:
                self._vector_selection = [
                    i for i in self._vector_selection if i != hit
                ]
            else:
                self._vector_selection = sorted(
                    set(self._vector_selection) | {hit})
            self._vector_sel = hit if hit in self._vector_selection else None
            self._vector_anchor = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            return
        if hit in self._vector_selection and len(self._vector_selection) > 1:
            self._vector_sel = hit
            self._vector_anchor = None
            self._capture_vector_gesture(layer, x, y, None)
            self._update_vector_overlay()
            return
        self._vector_sel = hit
        self._vector_selection = [hit]
        self._vector_anchor = None
        self._sync_vector_selection()
        self._update_vector_overlay()

    def _vector_edit_drag(self, event, x, y):
        layer = self._vector_cur
        if layer is None or layer not in self.app.current.layers:
            return
        if self._vector_handle_drag is not None:
            spec = self._vector_item(layer, self._vector_sel)
            if spec is None:
                return
            side, index = self._vector_handle_drag
            if not self._vector_dirty:
                self.app._push_undo("Edit vector handle")
                self._vector_dirty = True
            alt = bool(event.state & 0x8)
            vector.item_set_handle(spec, index, side, x, y, mirror=not alt)
            layer.invalidate_vectors()
            self._update_vector_overlay()
            self.app.render()
            return
        if self._vector_marquee is not None:
            self._vector_marquee_end = (x, y)
            self._update_vector_overlay()
            return
        if self._vector_gesture is not None:
            self._apply_vector_gesture(layer, x, y, event)
            return
        spec = self._vector_item(layer, self._vector_sel)
        if spec is None or self._vector_anchor is None:
            return
        if not self._vector_dirty:
            self.app._push_undo("Move vector point")
            self._vector_dirty = True
        nx, ny = self._snap_vector_anchor(
            layer, self._vector_sel, self._vector_anchor, x, y,
            self.VECTOR_HIT / max(event.widget.zoom, 1e-6))
        vector.item_move_point(spec, self._vector_anchor, nx, ny)
        layer.invalidate_vectors()
        self._update_vector_overlay()
        self.app.render()

    # ------------------------------------------------------ selection/transform
    def _vector_selection_set(self):
        """The selected item indices that still exist in the current layer."""
        layer = self._vector_cur
        if layer is None:
            return []
        total = len(self._vector_items(layer))
        return [i for i in self._vector_selection if 0 <= i < total]

    def _vector_handle_at(self, layer, x, y, zoom):
        """Which group transform handle (if any) ``(x, y)`` grabs."""
        selection = self._vector_selection_set()
        if len(selection) < 2:
            return None
        box = vector.union_bounds(
            [self._vector_item(layer, i) for i in selection])
        if box is None:
            return None
        left, top, right, bottom = box
        hit = max(6.0, float(getattr(self.app.tools, "vector_anchor_size", 4))) / zoom
        centre_x = (left + right) / 2.0
        if math.hypot(x - centre_x, y - (top - 20.0 / zoom)) <= hit:
            return "rot"
        corners = (
            ("nw", left, top), ("ne", right, top),
            ("sw", left, bottom), ("se", right, bottom),
        )
        for name, hx, hy in corners:
            if math.hypot(x - hx, y - hy) <= hit:
                return name
        return None

    def _capture_vector_gesture(self, layer, x, y, handle):
        selection = self._vector_selection_set()
        if not selection:
            return False
        self._vector_gesture = {
            "start": (x, y),
            "handle": handle,
            "origins": [
                (i, {
                    "snapshot": vector.item_snapshot(self._vector_item(layer, i)),
                    "width": vector.item_width(self._vector_item(layer, i)),
                })
                for i in selection
            ],
            "box": vector.union_bounds(
                [self._vector_item(layer, i) for i in selection]),
        }
        return True

    def _restore_vector_origin(self, spec, origin):
        """Put a gesture's captured geometry back before re-applying a transform."""
        vector.item_restore(spec, origin["snapshot"])

    def _apply_vector_gesture(self, layer, x, y, event):
        gesture = self._vector_gesture
        if gesture is None:
            return
        if not self._vector_dirty:
            self.app._push_undo("Transform vector lines")
            self._vector_dirty = True
        handle = gesture.get("handle")
        if handle is None:
            dx = x - gesture["start"][0]
            dy = y - gesture["start"][1]
            for index, origin in gesture["origins"]:
                spec = self._vector_item(layer, index)
                if spec is None:
                    continue
                self._restore_vector_origin(spec, origin)
                vector.item_translate(spec, dx, dy)
        else:
            self._apply_vector_handle(layer, gesture, x, y, event)
        layer.invalidate_vectors()
        self._update_vector_overlay()
        self.app.render()

    def _apply_vector_handle(self, layer, gesture, x, y, event):
        left, top, right, bottom = gesture["box"]
        centre_x = (left + right) / 2.0
        centre_y = (top + bottom) / 2.0
        handle = gesture["handle"]
        if handle == "rot":
            start_angle = math.atan2(
                gesture["start"][1] - centre_y, gesture["start"][0] - centre_x)
            now_angle = math.atan2(y - centre_y, x - centre_x)
            angle = now_angle - start_angle
            for index, origin in gesture["origins"]:
                spec = self._vector_item(layer, index)
                if spec is None:
                    continue
                self._restore_vector_origin(spec, origin)
                vector.item_rotate(spec, angle, centre_x, centre_y)
            return
        if "w" in handle:
            anchor_x = right
            sx = (anchor_x - x) / max(1e-6, anchor_x - left)
        else:
            anchor_x = left
            sx = (x - anchor_x) / max(1e-6, right - anchor_x)
        if "n" in handle:
            anchor_y = bottom
            sy = (anchor_y - y) / max(1e-6, anchor_y - top)
        else:
            anchor_y = top
            sy = (y - anchor_y) / max(1e-6, bottom - anchor_y)
        if event.state & 0x1:  # Shift keeps the aspect ratio
            sx = sy = max(abs(sx), abs(sy))
        scale_width = bool(getattr(self.app.tools, "vector_scale_width", False))
        width_factor = (abs(sx) + abs(sy)) / 2.0
        for index, origin in gesture["origins"]:
            spec = self._vector_item(layer, index)
            if spec is None:
                continue
            self._restore_vector_origin(spec, origin)
            vector.item_scale(spec, sx, sy, anchor_x, anchor_y)
            if scale_width and origin.get("width") is not None:
                vector.correct_width(
                    spec, origin["width"] * width_factor)

    def _finish_vector_marquee(self):
        layer = self._vector_cur
        start = self._vector_marquee
        end = getattr(self, "_vector_marquee_end", None)
        self._vector_marquee = None
        self._vector_marquee_end = None
        if layer is None or start is None or end is None:
            return
        left, right = sorted((start[0], end[0]))
        top, bottom = sorted((start[1], end[1]))
        if right - left < 2 and bottom - top < 2:
            return
        selected = []
        for index, spec in enumerate(self._vector_items(layer)):
            box = vector.item_bounds(spec)
            if box is None:
                continue
            if not (box[2] < left or box[0] > right
                    or box[3] < top or box[1] > bottom):
                selected.append(index)
        self._vector_selection = selected
        self._vector_sel = selected[-1] if selected else None
        self._vector_anchor = None

    def _snap_vector_anchor(self, layer, stroke_index, anchor_index, x, y,
                            tolerance):
        if not getattr(self.app.tools, "vector_snap", True):
            return x, y
        best = None
        best_distance = tolerance * 1.5
        for si, spec in enumerate(self._vector_items(layer)):
            for ai, (ax, ay) in enumerate(vector.item_anchor_points(spec)):
                if si == stroke_index and ai == anchor_index:
                    continue
                distance = math.hypot(x - ax, y - ay)
                if distance <= best_distance:
                    best_distance = distance
                    best = (ax, ay)
        return best if best is not None else (x, y)

    def _sync_vector_selection(self):
        """Push the current selection into the settings panel's line list."""
        tools = getattr(self.app, "tools", None)
        panel = getattr(tools, "refresh_vector_panel", None)
        if panel is None:
            return
        layer = self._vector_cur
        current = self.app.current
        if layer is None or current is None or layer not in current.layers:
            panel([], [])
            return
        items = self._vector_items(layer)
        labels = [
            vector.describe_fill(spec) if vector.is_fill(spec)
            else vector.describe(spec)
            for spec in items
        ]
        primary = self._vector_item(layer, self._vector_sel)
        panel(labels, self._vector_selection, primary)

    def refresh_vector(self):
        """Redraw the vector overlay after a settings change."""
        self._update_vector_overlay()

    def on_vector_action(self, action, data):
        """Handle the Vector settings panel's line-list and inspector controls."""
        if action == "refresh":
            if self.app.current is None:
                return
            layer = self._vector_layer(create=False)
            self._vector_cur = layer
            if layer is not None:
                total = len(self._vector_items(layer))
                self._vector_selection = [
                    i for i in self._vector_selection if i < total
                ]
            else:
                self._vector_selection = []
            self._sync_vector_selection()
            self._update_vector_overlay()
            return
        layer = self._vector_layer(create=False)
        if action == "select_all":
            if layer is None:
                self._vector_missing_layer()
                return
            self._vector_cur = layer
            items = self._vector_items(layer)
            self._vector_selection = list(range(len(items)))
            self._vector_sel = self._vector_selection[-1] if items else None
            self._vector_anchor = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            return
        if action == "clear":
            self._vector_selection = []
            self._vector_sel = None
            self._vector_anchor = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            return
        if action == "select":
            if layer is None:
                return
            self._vector_cur = layer
            total = len(self._vector_items(layer))
            indices = [
                int(i) for i in data.get("indices", [])
                if 0 <= int(i) < total
            ]
            self._vector_selection = indices
            self._vector_sel = indices[-1] if indices else None
            self._vector_anchor = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            return
        if action == "delete":
            targets = self._vector_selection_set()
            if layer is None or not targets:
                self.app.set_result("Select one or more lines first.")
                return
            self.app._push_undo("Delete vector lines")
            for index in sorted(targets, reverse=True):
                self._vector_remove_item(layer, index)
            layer.invalidate_vectors()
            self._vector_selection = []
            self._vector_sel = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Deleted {len(targets)} vector shape"
                f"{'s' if len(targets) != 1 else ''}.")
            return
        if action == "clear_fill":
            if layer is None:
                self._vector_missing_layer()
                return
            targets = self._vector_selection_set()
            if not targets and self._vector_sel is not None \
                    and                 0 <= self._vector_sel < len(self._vector_items(layer)):
                targets = [self._vector_sel]
            if not targets:
                self.app.set_result("Select a line first.")
                return
            self._vector_cur = layer
            self.app._push_undo("Remove vector fill")
            for index in targets:
                spec = self._vector_item(layer, index)
                if spec is not None and not vector.is_fill(spec):
                    spec["fill"] = None
            layer.invalidate_vectors()
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Removed fill from {len(targets)} vector line"
                f"{'s' if len(targets) != 1 else ''}.")
            return
        if action == "apply":
            if layer is None:
                self._vector_missing_layer()
                return
            targets = self._vector_selection_set()
            if not targets and self._vector_sel is not None \
                    and                 0 <= self._vector_sel < len(self._vector_items(layer)):
                targets = [self._vector_sel]
            if not targets:
                self.app.set_result("Select a line to edit its properties.")
                return
            self._vector_cur = layer
            self.app._push_undo("Edit vector line")
            for index in targets:
                spec = self._vector_item(layer, index)
                if spec is None:
                    continue
                if not vector.is_fill(spec):
                    vector.correct_width(spec, data.get("width", spec["width"]))
                    spec["fill"] = data.get("fill")
                spec["opacity"] = max(
                    0.0, min(1.0, float(data.get("opacity", 100)) / 100.0))
                spec["color"] = data.get("color", spec["color"])
            layer.invalidate_vectors()
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Updated {len(targets)} vector line"
                f"{'s' if len(targets) != 1 else ''}.")
            return
        if action in ("corner", "smooth", "toggle_closed"):
            if layer is None:
                self._vector_missing_layer()
                return
            targets = self._vector_selection_set()
            if not targets and self._vector_sel is not None \
                    and                 0 <= self._vector_sel < len(self._vector_items(layer)):
                targets = [self._vector_sel]
            if not targets:
                self.app.set_result("Select a line first.")
                return
            self._vector_cur = layer
            self.app._push_undo("Edit vector line")
            for index in targets:
                spec = self._vector_item(layer, index)
                if spec is None:
                    continue
                if action == "toggle_closed":
                    if not vector.is_fill(spec):
                        vector.toggle_closed(spec)
                    continue
                edit = (vector.item_set_corner if action == "corner"
                        else vector.item_set_smooth)
                for anchor in self._vector_anchor_indices(spec):
                    edit(spec, anchor)
            layer.invalidate_vectors()
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Updated {len(targets)} vector line"
                f"{'s' if len(targets) != 1 else ''}.")
            return
        if action == "split":
            spec = self._vector_item(layer, self._vector_sel)
            if layer is None or spec is None or vector.is_fill(spec):
                self.app.set_result("Select a line to split.")
                return
            anchor = self._vector_anchor
            if anchor is None or not (
                    0 <= anchor < vector.item_anchor_count(spec)):
                self.app.set_result("Click an anchor to split the line there.")
                return
            self._vector_cur = layer
            self.app._push_undo("Split vector line")
            pieces = vector.split_at(spec, anchor)
            layer.strokes[self._vector_sel:self._vector_sel + 1] = pieces
            layer.invalidate_vectors()
            self._vector_selection = list(
                range(self._vector_sel, self._vector_sel + len(pieces)))
            self._vector_sel = self._vector_selection[0]
            self._vector_anchor = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Split the line into {len(pieces)} "
                f"{'lines' if len(pieces) != 1 else 'line'}.")
            return
        if action == "simplify":
            if layer is None:
                self._vector_missing_layer()
                return
            targets = self._vector_selection_set()
            if not targets and self._vector_sel is not None:
                targets = [self._vector_sel]
            targets = [i for i in targets
                       if not self._vector_is_fill(layer, i)]
            if not targets:
                self.app.set_result("Select a line to simplify.")
                return
            self._vector_cur = layer
            self.app._push_undo("Simplify vector lines")
            for index in targets:
                vector.simplify_spec(layer.strokes[index])
            layer.invalidate_vectors()
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Simplified {len(targets)} vector line"
                f"{'s' if len(targets) != 1 else ''}.")
            return
        if action == "connect":
            targets = sorted(
                i for i in self._vector_selection_set()
                if not self._vector_is_fill(layer, i)) if layer else []
            if layer is None or len(targets) < 2:
                self.app.set_result("Select two or more open lines to connect.")
                return
            self._vector_cur = layer
            self.app._push_undo("Connect vector lines")
            chosen = [layer.strokes[i] for i in targets]
            merged = vector.connect(chosen, tolerance=CONNECT_TOLERANCE)
            first = min(targets)
            for index in sorted(targets, reverse=True):
                del layer.strokes[index]
            layer.strokes[first:first] = merged
            layer.invalidate_vectors()
            self._vector_selection = list(range(first, first + len(merged)))
            self._vector_sel = first if merged else None
            self._vector_anchor = None
            self._sync_vector_selection()
            self._update_vector_overlay()
            self.app.render()
            self.app.set_result(
                f"Connected into {len(merged)} line"
                f"{'s' if len(merged) != 1 else ''}.")
            return

    def _vector_anchor_indices(self, spec):
        """The anchors an edit applies to: the active one, else every anchor."""
        count = vector.item_anchor_count(spec)
        if self._vector_anchor is not None and 0 <= self._vector_anchor < count:
            return [self._vector_anchor]
        return list(range(count))

    def _vector_nudge_key(self, event):
        """Arrow-key nudge of the selected anchor (Edit mode)."""
        if getattr(self.app.tools, "vector_mode", "") != "edit":
            return False
        layer = self._vector_cur
        if (layer is None or layer not in self.app.current.layers
                or self._vector_sel is None or self._vector_anchor is None):
            return False
        deltas = {"Left": (-1, 0), "Right": (1, 0),
                  "Up": (0, -1), "Down": (0, 1)}
        if event.keysym not in deltas:
            return False
        spec = self._vector_item(layer, self._vector_sel)
        if spec is None or not (
                0 <= self._vector_anchor < vector.item_anchor_count(spec)):
            return False
        step = 10 if (event.state & 0x1) else 1
        dx, dy = deltas[event.keysym]
        self.app._push_undo("Nudge vector point")
        point = vector.item_anchor_points(spec)[self._vector_anchor]
        vector.item_move_point(
            spec, self._vector_anchor,
            point[0] + dx * step, point[1] + dy * step)
        layer.invalidate_vectors()
        self._update_vector_overlay()
        self.app.render()
        return True

    def _vector_add_anchor(self, x, y, view):
        """Insert an anchor on the selected stroke (double-click in Edit mode)."""
        layer = self._vector_layer(create=False)
        if layer is None:
            return False
        self._vector_cur = layer
        tolerance = self.VECTOR_HIT / max(view.zoom, 1e-6)
        if self._vector_sel is None or self._vector_sel >= len(
                self._vector_items(layer)):
            self._vector_sel = self._vector_hit_stroke(layer, x, y, tolerance)
        item = self._vector_item(layer, self._vector_sel)
        if item is None:
            return False
        near = vector.item_nearest_segment(item, x, y)
        if near is None:
            return False
        contour_index, segment, t = near
        self.app._push_undo("Add vector point")
        vector.item_split(item, contour_index, segment, t)
        layer.invalidate_vectors()
        self._sync_vector_selection()
        self._update_vector_overlay()
        self.app.render()
        return True

    def _vector_delete_anchor(self, x, y, view):
        """Delete the anchor under ``(x, y)`` (right-click in Edit mode)."""
        layer = self._vector_layer(create=False)
        spec = self._vector_item(layer, self._vector_sel) if layer else None
        if spec is None:
            return False
        tolerance = self.VECTOR_HIT / max(view.zoom, 1e-6)
        index = vector.item_anchor_hit(spec, x, y, tolerance)
        if index is None:
            return False
        self.app._push_undo("Delete vector point")
        remaining = vector.item_remove_point(spec, index)
        if not remaining:
            self._vector_remove_item(layer, self._vector_sel)
            self._vector_sel = None
            total = len(self._vector_items(layer))
            self._vector_selection = [
                i for i in self._vector_selection if i < total
            ]
        elif self._vector_sel is not None \
                and self._vector_sel >= len(self._vector_items(layer)):
            self._vector_sel = None
        layer.invalidate_vectors()
        self._sync_vector_selection()
        self._update_vector_overlay()
        self.app.render()
        self.app.set_result("Deleted a vector point.")
        return True

    # -------------------------------------------------------------- width
    def _vector_width_begin(self, x, y):
        layer = self._vector_layer(create=False)
        if layer is None:
            self._vector_missing_layer()
            return
        self._vector_cur = layer
        self._vector_sel = None
        self._vector_dirty = False
        self._vector_width_done = set()
        self._update_vector_overlay()
        self._vector_width_apply(x, y)

    def _vector_width_apply(self, x, y):
        layer = self._vector_cur
        if layer is None or layer not in self.app.current.layers:
            return
        view = self.app.area.current_view()
        tolerance = self.VECTOR_HIT / max(view.zoom, 1e-6) if view else self.VECTOR_HIT
        width = self.app.tools.brush_size
        mode = getattr(self.app.tools, "vector_width_mode", "set")
        done = getattr(self, "_vector_width_done", None)
        if done is None:
            done = self._vector_width_done = set()
        targets = []
        for spec in layer.strokes:
            if not vector.hit_stroke(spec, x, y, tolerance):
                continue
            if mode == "set" and spec["width"] == width:
                continue
            if mode != "set" and id(spec) in done:
                continue
            targets.append(spec)
        if not targets:
            return
        if not self._vector_dirty:
            self.app._push_undo("Correct line width")
            self._vector_dirty = True
        step = max(1, int(round(width / 4.0)))
        for spec in targets:
            if mode == "thicken":
                vector.correct_width(spec, spec["width"] + step)
                done.add(id(spec))
            elif mode == "narrow":
                vector.correct_width(spec, spec["width"] - step)
                done.add(id(spec))
            else:
                vector.correct_width(spec, width)
        layer.invalidate_vectors()
        self.app.render()

    # -------------------------------------------------------------- erase
    def _vector_erase_begin(self, x, y):
        layer = self._vector_layer(create=False)
        if layer is None:
            self._vector_missing_layer()
            return
        self._vector_cur = layer
        self._vector_sel = None
        self._vector_dirty = False
        self._update_vector_overlay()
        self._vector_erase_apply(x, y)

    def _vector_erase_apply(self, x, y):
        layer = self._vector_cur
        if layer is None or layer not in self.app.current.layers:
            return
        radius = max(1.0, self.app.tools.brush_size / 2.0)
        mode = getattr(self.app.tools, "vector_erase_mode", "touch")
        strokes = []
        changed = False
        for spec in layer.strokes:
            pieces = vector.erase(spec, x, y, radius, mode=mode)
            if len(pieces) == 1 and pieces[0] is spec:
                strokes.append(spec)
            else:
                changed = True
                strokes.extend(pieces)
        if not changed:
            return
        if not self._vector_dirty:
            self.app._push_undo("Vector erase")
            self._vector_dirty = True
        layer.strokes = strokes
        if self._vector_sel is not None and self._vector_sel >= len(strokes):
            self._vector_sel = None
        self._vector_selection = []
        layer.invalidate_vectors()
        self.app.render()

    # ------------------------------------------------------------- overlay
    def _update_vector_overlay(self, cursor=None):
        view = self.app.area.current_view()
        if view is None:
            return
        data = {}
        if self._vector_points:
            preview = [list(point) for point in self._vector_points]
            if cursor is not None:
                preview.append([cursor[0], cursor[1]])
            data["preview"] = preview
        layer = self._vector_cur
        if layer is not None and layer not in self.app.current.layers:
            layer = None
        if layer is not None:
            selection = self._vector_selection_set()
            if selection:
                item_paths = []
                for i in selection:
                    item_paths.extend(vector.item_flatten(
                        self._vector_item(layer, i)))
                data["selected"] = item_paths
                if (len(selection) >= 2
                        and getattr(self.app.tools, "vector_mode", "") == "edit"
                        and getattr(self.app.tools, "vector_show_anchors", True)):
                    box = vector.union_bounds(
                        [self._vector_item(layer, i) for i in selection])
                    if box is not None:
                        data["box"] = list(box)
        spec = self._vector_item(layer, self._vector_sel)
        if (getattr(self.app.tools, "vector_show_anchors", True)
                and spec is not None):
            data["stroke"] = vector.item_flatten(spec)
            data["anchors"] = vector.item_anchor_points(spec)
            data["handles"] = vector.item_handle_pairs(spec)
            data["active_anchor"] = (
                self._vector_anchor if self._vector_anchor is not None else -1
            )
        data["anchor_size"] = max(
            2, int(getattr(self.app.tools, "vector_anchor_size", 4)))
        if self._vector_marquee is not None:
            end = getattr(self, "_vector_marquee_end", None) or self._vector_marquee
            data["marquee"] = [
                self._vector_marquee[0], self._vector_marquee[1], end[0], end[1]
            ]
        view.set_vector_overlay(data or None)

    def sync_vector(self):
        """Refresh (or clear) the vector overlay after a tool/mode change."""
        if self.app.tools.vector_tool:
            self._sync_vector_selection()
            self._update_vector_overlay()
        else:
            self._vector_points = None
            self._vector_anchor = None
            self._vector_dirty = False
            self._vector_selection = []
            self._vector_marquee = None
            self._vector_handle = None
            self._vector_handle_drag = None
            self._vector_gesture = None
            view = self.app.area.current_view()
            if view is not None:
                view.set_vector_overlay(None)

    # -------------------------------------------------------------- poses
    def _poses_selected(self):
        """``(layer, spec)`` for the selected figure, or ``(layer, None)``."""
        layer = self._pose_layer
        current = self.app.current
        if layer is None or current is None or layer not in current.layers:
            return None, None
        if self._pose_sel is None or not (0 <= self._pose_sel < len(layer.poses)):
            return layer, None
        return layer, layer.poses[self._pose_sel]

    def _poses_pick(self, layer, x, y, tolerance):
        """Topmost figure under ``(x, y)`` as ``(index, joint_name_or_None)``."""
        for index in range(len(layer.poses) - 1, -1, -1):
            spec = layer.poses[index]
            joint = poses.joint_at(spec, x, y, tolerance)
            if joint is not None:
                return index, joint
            if poses.hit_body(spec, x, y):
                return index, None
        return None, None

    def _poses_target_layer(self):
        """The unlocked layer the Poses tool may edit, or ``None``."""
        layer = self.app.active_layer()
        if layer is None:
            self.app.set_result("There is no layer. Add a layer first.")
            return None
        if layer.is_locked():
            self.app.set_result(f"Layer '{layer.name}' is locked.")
            return None
        return layer

    def _pose_scene(self, layer=None):
        """The shared 3D scene of ``layer`` (or the posed layer), or ``None``."""
        layer = layer if layer is not None else self._pose_layer
        if layer is None:
            layer = self.app.active_layer()
        if layer is None:
            return None
        scene = getattr(layer, "pose_scene", None)
        if scene is None:
            scene = poses.new_scene(layer.size)
            layer.pose_scene = scene
        return scene

    def _poses_add(self, x, y):
        layer = self._poses_target_layer()
        if layer is None:
            return
        tools = self.app.tools
        scene = layer.pose_scene
        scale = float(getattr(tools, "poses_scale", poses.DEFAULT_SCALE))
        # A new figure stands beside the selected one (or at the scene origin),
        # sharing the layer's one scene camera and floor.
        base = [0.0, 0.0, 0.0]
        if (self._pose_sel is not None and 0 <= self._pose_sel < len(layer.poses)):
            selected = poses.position(layer.poses[self._pose_sel])
            base = [selected[0] + 0.9, selected[1], selected[2] + 0.4]
        spec = poses.new(
            scene["x"], scene["y"], scale,
            body=getattr(tools, "poses_body", poses.DEFAULT_BODY),
            color=getattr(tools, "poses_color", poses.DEFAULT_COLOR),
            pose=getattr(tools, "poses_pose", "relaxed"),
        )
        if spec is None:
            self.app.set_result("The pose models are missing from the install.")
            return
        spec["pos"] = base
        # New figures share the scene's world scale: the Height control sets their
        # intrinsic size relative to a default figure, so zooming the scene scales
        # them with the rest instead of pinning them to a pixel height.
        spec["size"] = max(0.01, float(scale) / poses.DEFAULT_SCALE)
        spec["opacity"] = max(
            0.0, min(1.0, getattr(tools, "poses_opacity", 100) / 100.0)
        )
        poses.sync_spec(spec, scene)
        self.app._push_undo("Add pose")
        layer.poses.append(spec)
        layer.invalidate_poses()
        self._pose_layer = layer
        self._pose_sel = len(layer.poses) - 1
        self._pose_dirty = False
        self._update_poses_overlay()
        self.app.render()
        self.app.set_result(f"Added a {spec['body']} pose.")

    def _poses_add_primitive(self, shape):
        """Add a rigid primitive object to the layer's 3D scene."""
        if shape not in poses.PRIMITIVES:
            return
        layer = self._poses_target_layer()
        if layer is None:
            return
        tools = self.app.tools
        scene = layer.pose_scene
        # A new object stands beside the selected one (or at the scene origin),
        # sharing the layer's one scene camera and floor. A primitive of size 1
        # spans about one model unit, so it reads at roughly half a figure's
        # height; the Height control scales a new object's own size.
        base = [0.0, 0.0, 0.0]
        if (self._pose_sel is not None and 0 <= self._pose_sel < len(layer.poses)):
            selected = poses.position(layer.poses[self._pose_sel])
            base = [selected[0] + 0.9, selected[1], selected[2]]
        spec = poses.new_primitive(
            0, 0, scene, shape=shape,
            color=poses.DEFAULT_PRIMITIVE_COLOR,
        )
        if spec is None:
            return
        spec["pos"] = base
        spec["size"] = max(
            0.01, float(getattr(tools, "poses_scale", poses.DEFAULT_SCALE))
            / poses.DEFAULT_SCALE,
        )
        spec["opacity"] = max(
            0.0, min(1.0, getattr(tools, "poses_opacity", 100) / 100.0)
        )
        poses.sync_spec(spec, scene)
        self.app._push_undo("Add primitive")
        layer.poses.append(spec)
        layer.invalidate_poses()
        self._pose_layer = layer
        self._pose_sel = len(layer.poses) - 1
        self._pose_selected_bone = None
        self._pose_dirty = False
        self._update_poses_overlay()
        self.app.render()
        self.app.set_result(f"Added a {poses.PRIMITIVE_LABELS[shape].lower()}.")

    def _poses_press(self, event, x, y):
        layer = self._poses_target_layer()
        if layer is None:
            return
        tolerance = self.POSE_HIT / max(event.widget.zoom, 1e-6)
        self._pose_layer = layer
        self._pose_start = (x, y)
        self._pose_dirty = False
        self._pose_moved = False
        self._pose_cycle = False
        self._pose_empty = False
        self._pose_render_at = 0.0

        selected = None
        if self._pose_sel is not None and 0 <= self._pose_sel < len(layer.poses):
            selected = layer.poses[self._pose_sel]

        index, joint = self._poses_pick(layer, x, y, tolerance)

        if index is None:
            if selected is not None:
                # The selected object's rotation rings and gizmo handles extend
                # past the body/geometry, so a press out on one of them grabs it
                # instead of orbiting.
                if poses.is_primitive(selected):
                    if self._pose_primitive_handle_press(selected, x, y, tolerance):
                        return
                elif self._pose_selected_bone:
                    if self._pose_rings_press(selected, x, y, tolerance):
                        return
                elif self._pose_figure_handle_press(selected, x, y, tolerance):
                    return
            # Anywhere else turns the whole scene (all figures and the floor) in
            # 3D; a click (a press with no drag) only drops the selection, on
            # release.
            if layer.poses:
                scene = self._pose_scene(layer)
                self._pose_drag = "orbit"
                self._pose_joint = None
                self._pose_selected_bone = None
                self._pose_empty = selected is not None
                self._pose_scene_view = [scene["yaw"], scene["pitch"]]
                self._pose_scene_roll = float(scene["roll"])
                poses.set_preview(True)
                self._update_poses_overlay()
                return
            # No figures at all: a click on empty space is a no-op.
            self._pose_sel = None
            self._pose_joint = None
            self._pose_selected_bone = None
            self._pose_drag = None
            self._update_poses_overlay()
            return
        spec = layer.poses[index]

        if poses.is_primitive(spec):
            self._pose_press_primitive(spec, index, x, y, tolerance)
            return

        # A press on a joint handle only selects that bone; a drag controls it
        # once it is selected, so a limb is never grabbed by accident.
        if joint is not None:
            already = index == self._pose_sel and joint == self._pose_selected_bone
            self._pose_sel = index
            self._pose_selected_bone = joint
            self._pose_joint = joint if already else None
            self._pose_drag = "joint" if already else None
            if already:
                poses.set_preview(True)
            self._update_poses_overlay()
            return

        # A ring of the selected bone rotates it about a world axis.
        if self._pose_selected_bone and index == self._pose_sel:
            if self._pose_rings_press(spec, x, y, tolerance):
                return

        # The body drives the whole-figure transform in the current mode; a
        # click (a press with no drag) cycles to the next mode on release.
        self._pose_cycle = (index == self._pose_sel
                            and self._pose_selected_bone is None)
        self._pose_sel = index
        self._pose_joint = None
        self._pose_selected_bone = None
        self._pose_begin_figure(spec, x, y)
        self._update_poses_overlay()

    def _pose_rings_press(self, spec, x, y, tolerance):
        """Start a world-axis ring drag on the selected bone; False when missed."""
        if not self._pose_selected_bone:
            return False
        axis, _distance = poses.gizmo_hit(
            spec, poses.gizmo(spec, self._pose_selected_bone),
            x, y, max(tolerance * 1.5, 8.0),
        )
        if axis is None:
            return False
        self._pose_drag = "ring"
        self._pose_joint = self._pose_selected_bone
        self._pose_ring_axis = axis
        self._pose_ring_angle = poses.gizmo_angle(
            spec, self._pose_selected_bone, axis, x, y)
        poses.set_preview(True)
        self._update_poses_overlay()
        return True

    def _pose_figure_handle_press(self, spec, x, y, tolerance):
        """Start a whole-figure drag when the press is on the drawn handle."""
        geometry = poses.figure_gizmo(spec)
        if not geometry:
            return False
        accepted = {"move": ("move_x", "move_y"), "scale": ("scale",),
                    "rotate": ("rotate",)}.get(self._pose_figure_mode, ())
        handle, _distance = poses.figure_hit(
            spec, geometry, x, y, max(tolerance, 8.0))
        if handle not in accepted:
            return False
        self._pose_joint = None
        self._pose_selected_bone = None
        self._pose_begin_figure(spec, x, y)
        self._update_poses_overlay()
        return True

    def _pose_press_primitive(self, spec, index, x, y, tolerance):
        """Select a primitive and start (or arm) a gizmo transform drag.

        A press on the already-selected object's own axis handle begins the drag;
        a plain press on the object just selects it and arms the mode cycle, so a
        click (press with no drag) advances Move/Scale/Rotate on release.
        """
        already = index == self._pose_sel
        self._pose_sel = index
        self._pose_joint = None
        self._pose_selected_bone = None
        self._pose_cycle = already
        axis = None
        if already:
            axis = poses.primitive_handle_at(
                spec, poses.primitive_gizmo(spec, self._pose_figure_mode),
                x, y, max(tolerance, 8.0),
            )
        if axis is None:
            self._pose_drag = None
        else:
            self._pose_begin_primitive(spec, x, y, axis)
        self._update_poses_overlay()

    def _pose_primitive_handle_press(self, spec, x, y, tolerance):
        """Start a primitive gizmo drag when the press is on the drawn handle."""
        axis = poses.primitive_handle_at(
            spec, poses.primitive_gizmo(spec, self._pose_figure_mode),
            x, y, max(tolerance, 8.0),
        )
        if axis is None:
            return False
        self._pose_joint = None
        self._pose_selected_bone = None
        self._pose_begin_primitive(spec, x, y, axis)
        self._update_poses_overlay()
        return True

    def _pose_begin_primitive(self, spec, x, y, axis):
        """Capture the values a primitive move/scale/rotate drag is relative to."""
        mode = self._pose_figure_mode
        geometry = poses.primitive_gizmo(spec, mode)
        if geometry is None:
            self._pose_drag = None
            return
        self._pose_axis = axis
        self._pose_primitive_geom = geometry
        self._pose_pos = poses.position(spec)
        self._pose_dim = list(spec.get("dim") or (1.0, 1.0, 1.0))
        self._pose_pivot = geometry["origin"]
        self._pose_figure_distance = max(
            math.hypot(x - self._pose_pivot[0], y - self._pose_pivot[1]), 1.0,
        )
        if mode == "rotate":
            self._pose_ring_angle = poses.primitive_ring_angle(spec, axis, x, y)
            self._pose_drag = "primitive_rotate"
        elif mode == "scale":
            self._pose_drag = "primitive_scale"
        else:
            self._pose_drag = "primitive_move"
        poses.set_preview(True)

    def _pose_begin_figure(self, spec, x, y):
        """Capture the values a whole-figure move/scale/rotate drag is relative to."""
        geometry = poses.figure_gizmo(spec)
        if geometry is None:
            self._pose_drag = None
            return
        self._pose_pivot = geometry["origin"]
        self._pose_pos = poses.position(spec)
        self._pose_size = float(spec.get("size", 1.0))
        self._pose_spin = float(spec.get("spin", 0.0))
        self._pose_figure_angle = poses.figure_angle(spec, x, y)
        self._pose_figure_distance = max(
            math.hypot(x - self._pose_pivot[0], y - self._pose_pivot[1]), 1.0,
        )
        self._pose_drag = {
            "move": "move", "scale": "figure_scale", "rotate": "figure_rotate",
        }[self._pose_figure_mode]
        poses.set_preview(True)

    def _poses_drag(self, x, y, roll=False):
        if self._pose_drag is None or self._pose_start is None:
            return
        layer, spec = self._poses_selected()
        if self._pose_drag != "orbit" and spec is None:
            return
        if self._pose_drag == "orbit":
            # An orbit turns the whole scene, so it needs no selected figure.
            if layer is None:
                layer = self._pose_layer or self.app.active_layer()
            if layer is None:
                return
        if not self._pose_dirty:
            self.app._push_undo("Pose figure")
            self._pose_dirty = True
        scene = self._pose_scene(layer)
        sx, sy = self._pose_start
        dx = x - sx
        dy = y - sy
        if abs(dx) > 3 or abs(dy) > 3:
            self._pose_moved = True
        if self._pose_drag == "orbit":
            yaw0, pitch0 = self._pose_scene_view
            if roll:
                scene["roll"] = float(self._pose_scene_roll + dx * 0.6)
            else:
                scene["yaw"] = float(yaw0 + dx * 0.6)
                scene["pitch"] = float(max(-80.0, min(80.0, pitch0 - dy * 0.6)))
            poses.sync_scene(layer.poses, scene)
        elif self._pose_drag == "move":
            poses.move_by_doc(spec, scene, self._pose_pos, dx, dy)
            poses.sync_spec(spec, scene)
        elif self._pose_drag == "figure_rotate":
            angle = poses.figure_angle(spec, x, y)
            if angle is not None and self._pose_figure_angle is not None:
                spec["spin"] = float(self._pose_spin
                                     + math.degrees(angle - self._pose_figure_angle))
                poses.sync_spec(spec, scene)
                poses.keep_pelvis(spec, scene, self._pose_pivot)
        elif self._pose_drag == "figure_scale":
            distance = math.hypot(x - self._pose_pivot[0], y - self._pose_pivot[1])
            factor = distance / max(self._pose_figure_distance, 1.0)
            low, high = poses.size_bounds(scene, spec["body"])
            spec["size"] = max(low, min(high, float(self._pose_size * factor)))
            poses.sync_spec(spec, scene)
            poses.keep_pelvis(spec, scene, self._pose_pivot)
        elif self._pose_drag == "ring":
            angle = poses.gizmo_angle(spec, self._pose_joint, self._pose_ring_axis, x, y)
            if angle is not None and self._pose_ring_angle is not None:
                delta = angle - self._pose_ring_angle
                while delta > math.pi:
                    delta -= 2.0 * math.pi
                while delta < -math.pi:
                    delta += 2.0 * math.pi
                poses.apply_axis_rotation(
                    spec, self._pose_joint,
                    poses.GIZMO_AXES[self._pose_ring_axis], delta,
                )
                self._pose_ring_angle = angle
        elif self._pose_drag == "primitive_move":
            poses.move_primitive(
                spec, self._pose_pos, self._pose_axis,
                self._pose_primitive_geom, dx, dy,
            )
            poses.sync_spec(spec, scene)
        elif self._pose_drag == "primitive_scale":
            distance = math.hypot(x - self._pose_pivot[0], y - self._pose_pivot[1])
            factor = distance / max(self._pose_figure_distance, 1.0)
            poses.scale_primitive(spec, self._pose_dim, self._pose_axis, factor)
            poses.sync_spec(spec, scene)
        elif self._pose_drag == "primitive_rotate":
            angle = poses.primitive_ring_angle(spec, self._pose_axis, x, y)
            if angle is not None and self._pose_ring_angle is not None:
                delta = angle - self._pose_ring_angle
                while delta > math.pi:
                    delta -= 2.0 * math.pi
                while delta < -math.pi:
                    delta += 2.0 * math.pi
                poses.rotate_primitive(spec, self._pose_axis, delta)
                poses.sync_spec(spec, scene)
                self._pose_ring_angle = angle
        else:
            # End-effector handles solve the whole limb (IK); every other handle
            # aims its own bone at the cursor.
            if self._pose_joint in poses.IK_CHAIN:
                poses.ik_at(spec, self._pose_joint, x, y)
            else:
                poses.aim_at(spec, self._pose_joint, x, y)
        layer.invalidate_poses()
        # The handles follow the cursor on every event (that is cheap forward
        # kinematics), but redrawing the figure costs tens of milliseconds, so
        # the mesh is throttled to keep the drag responsive; the release always
        # draws the final, full-quality frame.
        self._update_poses_overlay()
        now = time.monotonic()
        # The GPU draws a full-detail frame in a couple of milliseconds, so only
        # the CPU backend needs the redraw throttled.
        if poses.gpu_active() or now - self._pose_render_at >= 0.09:
            self._pose_render_at = now
            self.app.render()

    def _poses_release(self):
        poses.set_preview(False)
        layer, spec = self._poses_selected()
        # A click (a press with no drag) on empty space drops the selection; a
        # click on the selected body cycles the figure transform between move,
        # scale and rotate; a joint press only selects its bone, and ring/joint
        # drags leave the selection as it is.
        if not self._pose_moved:
            if self._pose_empty:
                self._pose_sel = None
                self._pose_selected_bone = None
            elif self._pose_cycle:
                self._advance_figure_mode()
        if self._pose_dirty:
            if layer is not None:
                layer.invalidate_poses()
            self.app.set_result(
                "Turned the scene." if self._pose_drag == "orbit"
                else ("Transformed the object." if poses.is_primitive(spec)
                      else "Posed the figure.")
            )
            self.app.render()
        self._pose_drag = None
        self._pose_ring_axis = None
        self._pose_joint = None
        self._pose_axis = None
        self._pose_primitive_geom = None
        self._pose_dirty = False
        self._pose_moved = False
        self._pose_cycle = False
        self._pose_empty = False
        self._update_poses_overlay()

    def _advance_figure_mode(self):
        """Cycle the body transform to the next whole-figure mode."""
        order = poses.FIGURE_MODES
        start = order.index(self._pose_figure_mode) if self._pose_figure_mode in order else -1
        self._pose_figure_mode = order[(start + 1) % len(order)]
        _layer, spec = self._poses_selected()
        if spec is not None and poses.is_primitive(spec):
            self.app.set_result(
                f"Object {self._pose_figure_mode}: drag a handle to "
                f"{self._pose_figure_mode} the object."
            )
        else:
            self.app.set_result(
                f"Figure {self._pose_figure_mode}: drag the body to "
                f"{self._pose_figure_mode} the pose."
            )

    def _poses_reset_joint(self, x, y, view):
        """Reset the joint under a right-click to rest. True when it hit one."""
        layer, spec = self._poses_selected()
        if spec is None:
            return False
        tolerance = self.POSE_HIT / max(view.zoom, 1e-6)
        joint = poses.joint_at(spec, x, y, tolerance)
        if joint is None:
            return False
        self.app._push_undo("Reset poses joint")
        spec["joints"].pop(joint, None)
        layer.invalidate_poses()
        self._update_poses_overlay()
        self.app.render()
        self.app.set_result("Reset the joint to rest.")
        return True

    def _pose_scene_layer(self):
        """The layer whose scene the Poses tool should edit."""
        layer = self._pose_layer
        current = self.app.current
        if layer is None or current is None or layer not in current.layers:
            layer = self.app.active_layer()
        return layer

    def _set_pose_scene_zoom(self, percent):
        """Dolly the scene to ``percent`` of the default zoom (from the slider)."""
        layer = self._pose_scene_layer()
        if layer is None:
            self.app.set_result("There is no layer. Add a layer first.")
            return
        scene = self._pose_scene(layer)
        poses.set_scene_zoom(scene, poses.DEFAULT_ZOOM * float(percent) / 100.0)
        poses.sync_scene(layer.poses, scene)
        layer.invalidate_poses()
        self._update_poses_overlay()
        self.app.render()
        self.app.set_result(f"Scene zoom {int(round(percent))}%.")

    def _reset_pose_scene(self):
        """Restore the layer's scene camera (view, roll, zoom and anchor)."""
        layer = self._pose_scene_layer()
        if layer is None:
            self.app.set_result("There is no layer. Add a layer first.")
            return
        self.app._push_undo("Reset scene view")
        layer.pose_scene = poses.new_scene(layer.size)
        poses.sync_scene(layer.poses, layer.pose_scene)
        layer.invalidate_poses()
        self._update_poses_overlay()
        self.app.render()
        self.app.set_result("Reset the scene view.")

    def on_poses_action(self, action):
        """Handle the Poses settings controls (add/duplicate/reset/mirror/…)."""
        if action == "grid":
            self._update_poses_overlay()
            self.app.render()
            return
        if action == "add" or (action.startswith("pose:") and action.endswith(":new")):
            current = self.app.current
            if current is None:
                self.app.set_result("There is no canvas. Add a canvas first.")
                return
            self._poses_add(current.width / 2.0, current.height / 2.0)
            return
        if action.startswith("add:"):
            if self.app.current is None:
                self.app.set_result("There is no canvas. Add a canvas first.")
                return
            self._poses_add_primitive(action.split(":", 1)[1])
            return
        # Scene camera actions act on the whole scene, so they need no selection.
        if action == "reset_scene":
            self._reset_pose_scene()
            return
        if action.startswith("scene_zoom:"):
            try:
                percent = float(action.split(":", 1)[1])
            except ValueError:
                return
            self._set_pose_scene_zoom(percent)
            return
        layer, spec = self._poses_selected()
        if spec is None:
            if action == "duplicate":
                self.on_poses_action("add")
            else:
                self.app.set_result("Select an object first.")
            return
        if poses.is_primitive(spec) and action.split(":", 1)[0] in ("pose", "hand", "body"):
            self.app.set_result("That preset applies to a mannequin, not a primitive.")
            return
        if action == "duplicate":
            duplicate = poses.clone(spec)
            base = poses.position(duplicate)
            duplicate["pos"] = [base[0] + 0.5, base[1], base[2] + 0.5]
            poses.sync_spec(duplicate, self._pose_scene(layer))
            self.app._push_undo("Duplicate pose")
            layer.poses.append(duplicate)
            layer.invalidate_poses()
            self._pose_sel = len(layer.poses) - 1
            self._update_poses_overlay()
            self.app.render()
            self.app.set_result("Duplicated the pose.")
        elif action == "reset":
            self.app._push_undo("Reset pose")
            poses.reset(spec)
            poses.sync_spec(spec, self._pose_scene(layer))
            layer.invalidate_poses()
            self._update_poses_overlay()
            self.app.render()
            self.app.set_result("Reset the pose to its resting stance.")
        elif action == "mirror":
            self.app._push_undo("Mirror pose")
            poses.mirror(spec)
            poses.sync_spec(spec, self._pose_scene(layer))
            layer.invalidate_poses()
            self._update_poses_overlay()
            self.app.render()
            self.app.set_result("Mirrored the pose.")
        elif action.startswith("pose:"):
            name = action.split(":", 1)[1]
            self.app._push_undo("Apply pose")
            if poses.apply_pose(spec, name):
                layer.invalidate_poses()
                self._update_poses_overlay()
                self.app.render()
                self.app.set_result(f"Applied the '{name}' pose.")
        elif action.startswith("hand:"):
            parts = action.split(":", 2)
            if len(parts) == 3:
                self.app._push_undo("Hand pose")
                poses.apply_hand(spec, parts[1], parts[2])
                layer.invalidate_poses()
                self._update_poses_overlay()
                self.app.render()
                self.app.set_result(f"Applied the '{parts[1]}' hand pose.")
        elif action.startswith("body:"):
            body = action.split(":", 1)[1]
            if body in poses.BODIES:
                self.app._push_undo("Change body")
                spec["body"] = body
                poses.sync_spec(spec, self._pose_scene(layer))
                layer.invalidate_poses()
                self._update_poses_overlay()
                self.app.render()
                self.app.set_result(f"Switched to the {body} body.")
        elif action == "delete":
            self.app._push_undo("Delete pose")
            layer.poses.pop(self._pose_sel)
            layer.invalidate_poses()
            self._pose_sel = None
            self._update_poses_overlay()
            self.app.render()
            self.app.set_result("Deleted the pose.")

    def _update_poses_overlay(self):
        view = self.app.area.current_view()
        if view is None:
            return
        if not self.app.tools.poses_tool:
            self._pose_overlay_data = None
            view.set_poses_overlay(None)
            return
        _layer, spec = self._poses_selected()
        layer = self._pose_layer
        current = self.app.current
        if layer is None or current is None or layer not in current.layers:
            layer = self.app.active_layer()
        scene = self._pose_scene(layer) if layer is not None else None
        if layer is not None:
            poses.sync_scene(layer.poses, scene)
        self.app.tools._refresh_pose_scene_zoom(scene)
        data = {}
        if spec is not None:
            data = poses.overlay(spec)
            data["active_joint"] = self._pose_joint
            if poses.is_primitive(spec):
                # A primitive has no skeleton; its one control is the modal
                # axis gizmo (move/scale/rotate) around the object.
                data["selected"] = True
                data["primitive_gizmo"] = poses.primitive_gizmo(
                    spec, self._pose_figure_mode)
                data["figure_mode"] = self._pose_figure_mode
            else:
                data["selected_bone"] = self._pose_selected_bone
                if self._pose_selected_bone:
                    # A selected bone shows its rotation rings and hides the
                    # whole-figure gizmo, so only one control is on the body at once.
                    data["gizmo"] = poses.gizmo(spec, self._pose_selected_bone)
                else:
                    data["figure_gizmo"] = poses.figure_gizmo(spec)
                    data["figure_mode"] = self._pose_figure_mode
            data["hover"] = self._pose_hover
        if (getattr(self.app.tools, "poses_grid", False) and scene is not None
                and layer is not None and layer.poses):
            # One ground grid for the whole scene, drawn once under every figure.
            data["grid"] = poses.scene_grid(scene)
        if not data:
            self._pose_overlay_data = None
            view.set_poses_overlay(None)
            return
        self._pose_overlay_data = data
        view.set_poses_overlay(data)

    def _hover_poses(self, event):
        """Highlight the gizmo handle under the cursor, redrawing only on change.

        Hit-tests the geometry kept from the last overlay build rather than
        reprojecting it, and only rebuilds the overlay when the hovered handle
        actually changes, so a plain mouse move stays cheap.
        """
        data = self._pose_overlay_data
        x, y = self.cursor_x, self.cursor_y
        if not data or x is None or y is None:
            return
        tolerance = self.POSE_HIT / max(event.widget.zoom, 1e-6)
        hover = None
        geometry = data.get("primitive_gizmo")
        if geometry is not None:
            axis, _distance = poses.primitive_hit(
                None, geometry, x, y, max(tolerance * 1.6, 8.0))
            if axis is not None:
                hover = ("axis", axis)
        else:
            geometry = data.get("gizmo")
            if geometry is not None:
                axis, _distance = poses.gizmo_hit(
                    None, geometry, x, y, max(tolerance * 1.5, 8.0))
                if axis is not None:
                    hover = ("ring", axis)
            else:
                geometry = data.get("figure_gizmo")
                mode = data.get("figure_mode", "move")
                accepted = {"move": ("move_x", "move_y"), "scale": ("scale",),
                            "rotate": ("rotate",)}.get(mode, ())
                if geometry:
                    handle, _distance = poses.figure_hit(
                        None, geometry, x, y, max(tolerance, 8.0))
                    if handle in accepted:
                        hover = ("figure", handle)
        if hover != self._pose_hover:
            self._pose_hover = hover
            self._update_poses_overlay()

    def sync_poses(self):
        """Refresh (or clear) the poses overlay after a tool/mode change."""
        if self.app.tools.poses_tool:
            self._update_poses_overlay()
        else:
            poses.set_preview(False)
            self._pose_layer = None
            self._pose_selected_bone = None
            self._pose_ring_axis = None
            self._pose_sel = None
            self._pose_joint = None
            self._pose_drag = None
            self._pose_dirty = False
            self._pose_cycle = False
            self._pose_empty = False
            self._pose_figure_mode = "move"
            self._pose_axis = None
            self._pose_primitive_geom = None
            self._pose_hover = None
            self._pose_overlay_data = None
            view = self.app.area.current_view()
            if view is not None:
                view.set_poses_overlay(None)

    def sync_symmetry(self):
        """Repaint when the symmetry guide changed, so the view redraws its axes."""
        guide = self.app.symmetry_guide()
        if guide != self._symmetry_last:
            self._symmetry_last = guide
            self.app.render()

    # --------------------------------------------------------------- liquify
    def _liquify_press(self, event, x, y):
        if self._layer_locked():
            return
        layer = self.app.active_layer()
        if layer is None:
            return
        self.app._push_undo("Liquify")
        self._liquify = warp.Liquify(layer)
        self._liquify_last = (x, y)
        self._liquify_stamp(x, y, 0.0, 0.0)
        self._apply_cursor(event.widget)
        self.app.render()

    def _liquify_stamp(self, x, y, step_x, step_y):
        tools = self.app.tools
        self._liquify.stamp(
            x, y, tools.brush_size / 2.0, tools.liquify_mode,
            tools.liquify_strength / 100.0, step_x, step_y,
        )
        self._liquify.layer.set_image(self._liquify.image)

    def _liquify_drag(self, x, y):
        if self._liquify is None:
            return
        last = self._liquify_last
        self._liquify_last = (x, y)
        if last is None:
            self._liquify_stamp(x, y, 0.0, 0.0)
        else:
            self._liquify_stamp(x, y, x - last[0], y - last[1])
        self.app.render()

    def _liquify_release(self):
        if self._liquify is None:
            return
        self._liquify = None
        self._liquify_last = None
        self.app.set_result("Liquified the active layer.")

    # ------------------------------------------------------------ mesh warp
    def sync_warp(self):
        """Start/stop the mesh-warp session with the tool, or drop a liquify drag."""
        if not self.app.tools.liquify_tool:
            self._liquify = None
            self._liquify_last = None
        if self.app.tools.warp_tool:
            if self._warp is None and self.app.current is not None:
                self.start_warp()
            elif self._warp is not None:
                self._update_warp_overlay()
        elif self._warp is not None:
            self.apply_warp()

    def start_warp(self):
        """Begin a mesh-warp session on the active layer.

        The undo entry is pushed lazily on the first node drag, so merely
        selecting the tool (or switching canvases) leaves no empty undo step.
        """
        if self._warp is not None:
            self._warp.release()
        self._clear_warp_overlay()
        layer = self.app.active_layer()
        if layer is None or layer.is_locked():
            self.app.set_result("Select an unlocked layer to warp.")
            return False
        box = layer.image.getbbox()
        if box is None:
            self.app.set_result("The active layer has no pixels to warp.")
            return False
        tools = self.app.tools
        self._warp = warp.MeshWarp(layer, box, tools.warp_cols, tools.warp_rows)
        self._warp_original = layer.image.copy()
        self._warp_pushed = False
        self._warp_node = None
        self._update_warp_overlay()
        return True

    def discard_warp(self):
        """Drop the warp session without touching pixels (used on canvas switch)."""
        if self._warp is not None:
            self._warp.release()
        self._warp = None
        self._warp_original = None
        self._warp_node = None
        self._warp_pushed = False
        self._liquify = None
        self._liquify_last = None
        self._clear_warp_overlay()

    def _clear_warp_overlay(self):
        view = self.app.area.current_view()
        if view is not None:
            view.set_warp_grid(None)

    def _update_warp_overlay(self):
        view = self.app.area.current_view()
        if view is None:
            return
        if self._warp is None:
            view.set_warp_grid(None)
        else:
            view.set_warp_grid(self._warp.grid_geometry())

    def _warp_press(self, event, x, y):
        if self._warp is None:
            return
        tolerance = self.WARP_HIT / max(event.widget.zoom, 1e-6)
        node = self._warp.hit_node(x, y, tolerance)
        self._warp_node = node
        if node is not None:
            self._warp_press_origin = (x, y)
            self._warp_start_offset = (
                float(self._warp.nodes[node[0], node[1], 0]),
                float(self._warp.nodes[node[0], node[1], 1]),
            )
        self._update_warp_overlay()

    def _warp_drag(self, x, y):
        if self._warp is None or self._warp_node is None:
            return
        if not self._warp_pushed:
            self.app._push_undo("Mesh warp")
            self._warp_pushed = True
        ox, oy = self._warp_press_origin
        sx, sy = self._warp_start_offset
        self._warp.set_node_offset(
            self._warp_node[0], self._warp_node[1],
            sx + (x - ox), sy + (y - oy),
        )
        self._warp.layer.set_image(self._warp.image)
        self._update_warp_overlay()
        self.app.render()

    def _warp_release(self):
        self._warp_node = None

    def apply_warp(self):
        if self._warp is None:
            return
        self._warp.release()
        self._warp = None
        self._warp_original = None
        self._warp_node = None
        self._warp_pushed = False
        self._clear_warp_overlay()
        self.app.render()
        self.app.set_result("Warped the active layer.")
        # Keep going when the tool is still active: begin a fresh session on the
        # newly warped pixels, so Apply commits and lets the user keep deforming.
        if self.app.tools.warp_tool and self.app.current is not None:
            self.start_warp()

    def cancel_warp(self):
        """Revert an in-progress warp; returns whether one was running."""
        if self._warp is None:
            return False
        layer = self._warp.layer
        if self._warp_original is not None and layer in self.app.current.layers:
            layer.set_image(self._warp_original)
        if self._warp_pushed:
            self.app._drop_undo()
        self._warp.release()
        self._warp = None
        self._warp_original = None
        self._warp_node = None
        self._warp_pushed = False
        self._clear_warp_overlay()
        self.app.render()
        self.app.set_result("Cancelled the warp.")
        return True

    def reset_warp(self):
        if self._warp is None:
            return
        self._warp.reset()
        self._warp.layer.set_image(self._warp.image)
        self._update_warp_overlay()
        self.app.render()

    def on_warp_action(self, action):
        """Handle the Warp settings buttons (reset/apply/cancel/rebuild)."""
        if action == "reset":
            self.reset_warp()
        elif action == "apply":
            self.apply_warp()
        elif action == "cancel":
            self.cancel_warp()
        elif action == "rebuild":
            self.start_warp()

    # ------------------------------------------------------------ puppet warp
    def sync_puppet(self):
        """Start/stop the puppet-warp session with the tool."""
        if self.app.tools.puppet_tool:
            if self._puppet is None and self.app.current is not None:
                self.start_puppet()
            elif self._puppet is not None:
                # Changed Influence/Expansion re-render the live session in place.
                tools = self.app.tools
                changed = (tools.puppet_influence != self._puppet.influence
                           or tools.puppet_expansion != self._puppet.expansion)
                if changed:
                    self._puppet.influence = tools.puppet_influence
                    self._puppet.expansion = tools.puppet_expansion
                    if self._puppet.pins:
                        self._puppet.layer.set_image(self._puppet.render())
                        self.app.render()
                self._update_puppet_overlay()
        elif self._puppet is not None:
            self.apply_puppet()

    def start_puppet(self):
        """Begin a puppet-warp session on the active layer.

        The undo entry is pushed lazily on the first pin drag, so merely
        selecting the tool (or switching canvases) leaves no empty undo step.
        """
        self._clear_puppet_overlay()
        layer = self.app.active_layer()
        if layer is None or layer.is_locked():
            self.app.set_result("Select an unlocked layer to puppet-warp.")
            return False
        box = layer.image.getbbox()
        if box is None:
            self.app.set_result("The active layer has no pixels to warp.")
            return False
        # A selection limits the warp to the selected pixels, like Clip Studio's
        # puppet warp; with none the whole layer content moves.
        clip = None
        selection = self._selection()
        if selection is not None and not selection.is_empty():
            ox, oy = getattr(layer, "offset", (0, 0))
            clip = Image.new("L", layer.image.size, 0)
            clip.paste(selection.mask, (-ox, -oy))
        tools = self.app.tools
        self._puppet = warp.PuppetWarp(
            layer, box, tools.puppet_influence, tools.puppet_expansion,
            clip=clip,
        )
        self._puppet_original = layer.image.copy()
        self._puppet_pushed = False
        self._puppet_reset_drag()
        self._update_puppet_overlay()
        return True

    def discard_puppet(self):
        """Drop the session without touching pixels (used on canvas switch)."""
        self._puppet = None
        self._puppet_original = None
        self._puppet_pushed = False
        self._puppet_reset_drag()
        self._clear_puppet_overlay()

    def _puppet_reset_drag(self):
        """Clear the transient drag/selection-band state."""
        if self._puppet_preview_job is not None:
            self.app.root.after_cancel(self._puppet_preview_job)
            self._puppet_preview_job = None
        self._puppet_mode = None
        self._puppet_press_origin = None
        self._puppet_start_pos = None
        self._puppet_start_angles = None
        self._puppet_rotate_center = None
        self._puppet_rotate_start = 0.0
        self._puppet_band = None
        self._puppet_band_additive = False
        self._puppet_moved = False
        self._puppet_manipulated = set()

    def _clear_puppet_overlay(self):
        view = self.app.area.current_view()
        if view is not None:
            view.set_puppet_overlay(None)

    def _update_puppet_overlay(self):
        view = self.app.area.current_view()
        if view is None:
            return
        if self._puppet is None:
            view.set_puppet_overlay(None)
            return
        show_mesh = bool(getattr(self.app.tools, "puppet_show_mesh", False))
        view.set_puppet_overlay({
            "pins": self._puppet.pin_geometry(),
            "mesh": self._puppet.mesh_polylines() if show_mesh else None,
            "show_mesh": show_mesh,
            "band": self._puppet_band if self._puppet_moved else None,
            "handle_radius": self.PUPPET_HANDLE / max(view.zoom, 1e-6),
        })

    def _puppet_press(self, event, x, y):
        if self._puppet is None and not self.start_puppet():
            return
        zoom = max(event.widget.zoom, 1e-6)
        shift = bool(event.state & 0x1)
        ctrl = bool(event.state & 0x4)
        handle_radius = self.PUPPET_HANDLE / zoom
        tolerance = self.PUPPET_HIT / zoom
        index = self._puppet.hit_handle(x, y, tolerance, handle_radius)
        if index is None:
            index = self._puppet.hit_ring(x, y, tolerance, handle_radius)
        if index is not None:
            if not self._puppet.pins[index]["selected"]:
                self._puppet.select_only(index)
            self._begin_puppet_rotate(index, x, y)
            return
        index = self._puppet.hit_pin(x, y, tolerance)
        if index is not None:
            if shift or ctrl:
                self._puppet.toggle_select(index)
            elif not self._puppet.pins[index]["selected"]:
                self._puppet.select_only(index)
            self._begin_puppet_move(x, y)
            return
        # Empty space: a tap drops a pin, a drag rubber-bands a selection.
        self._puppet_mode = "band"
        self._puppet_press_origin = (x, y)
        self._puppet_band = [x, y, x, y]
        self._puppet_band_additive = shift
        self._puppet_moved = False
        if not shift:
            self._puppet.clear_selection()
        self._update_puppet_overlay()

    def _begin_puppet_move(self, x, y):
        self._puppet_mode = "move"
        self._puppet_press_origin = (x, y)
        self._puppet_start_pos = {
            index: (pin["pos"][0], pin["pos"][1])
            for index, pin in enumerate(self._puppet.pins)
            if pin["selected"]
        }
        self._puppet_manipulated = set(self._puppet_start_pos)

    def _begin_puppet_rotate(self, index, x, y):
        self._puppet_mode = "rotate"
        px, py = self._puppet.pins[index]["pos"]
        self._puppet_rotate_center = (px, py)
        self._puppet_rotate_start = math.atan2(y - py, x - px)
        self._puppet_start_angles = {
            i: pin["angle"]
            for i, pin in enumerate(self._puppet.pins)
            if pin["selected"]
        }
        self._puppet_manipulated = set(self._puppet_start_angles)

    def _puppet_drag(self, x, y):
        if self._puppet is None:
            return
        if self._puppet_mode == "move":
            self._push_puppet_undo()
            ox, oy = self._puppet_press_origin
            dx, dy = x - ox, y - oy
            for index, (sx, sy) in self._puppet_start_pos.items():
                self._puppet.set_pin_pos(index, sx + dx, sy + dy)
            self._puppet_refresh()
        elif self._puppet_mode == "rotate":
            self._push_puppet_undo()
            px, py = self._puppet_rotate_center
            delta = math.atan2(y - py, x - px) - self._puppet_rotate_start
            for index, angle in self._puppet_start_angles.items():
                self._puppet.set_pin_angle(index, angle + delta)
            self._puppet_refresh()
        elif self._puppet_mode == "band":
            self._puppet_band[2] = x
            self._puppet_band[3] = y
            ox, oy = self._puppet_press_origin
            if abs(x - ox) > 3.0 or abs(y - oy) > 3.0:
                self._puppet_moved = True
            self._update_puppet_overlay()

    def _push_puppet_undo(self):
        if not self._puppet_pushed:
            self.app._push_undo("Puppet warp")
            self._puppet_pushed = True

    def _puppet_refresh(self):
        """Coalesce pointer events instead of rendering stale drag positions."""
        if self._puppet_preview_job is None:
            self._puppet_preview_job = self.app.root.after(
                16, self._render_puppet_preview)

    def _render_puppet_preview(self):
        self._puppet_preview_job = None
        if self._puppet is None:
            return
        self._puppet.layer.set_image(self._puppet.preview())
        self._update_puppet_overlay()
        self.app.render()

    def _puppet_release(self):
        mode = self._puppet_mode
        if self._puppet is not None and mode == "band":
            if self._puppet_moved:
                left = min(self._puppet_band[0], self._puppet_band[2])
                top = min(self._puppet_band[1], self._puppet_band[3])
                right = max(self._puppet_band[0], self._puppet_band[2])
                bottom = max(self._puppet_band[1], self._puppet_band[3])
                self._puppet.select_rect(
                    (left, top, right, bottom), self._puppet_band_additive,
                )
            else:
                self._puppet_place_pin(self._puppet_press_origin)
        if self._puppet is not None and (
                self._puppet.previewed or self._puppet_preview_job is not None):
            # Finish a reduced-resolution drag at full quality.
            self._puppet.layer.set_image(self._puppet.render())
            self.app.render()
        self._puppet_reset_drag()
        self._update_puppet_overlay()

    def _puppet_place_pin(self, point):
        if not self._puppet.contains(*point):
            self.app.set_result("Place puppet pins on the drawing's mesh.")
            return
        if len(self._puppet.pins) >= warp.MAX_PINS:
            self.app.set_result(f"At most {warp.MAX_PINS} pins.")
            return
        index = self._puppet.add_pin(point[0], point[1])
        self._puppet.select_only(index)

    def _puppet_delete_pin(self, x, y, view):
        """Remove the pin under ``(x, y)``; returns whether one was removed."""
        if self._puppet is None:
            return False
        tolerance = self.PUPPET_HIT / max(view.zoom, 1e-6)
        index = self._puppet.hit_pin(x, y, tolerance)
        if index is None:
            return False
        self._puppet.remove_pin(index)
        self._puppet.layer.set_image(self._puppet.image)
        self._puppet_reset_drag()
        self._update_puppet_overlay()
        self.app.render()
        return True

    def clear_puppet_pins(self):
        if self._puppet is None:
            return
        self._puppet_reset_drag()
        self._puppet.clear_pins()
        self._puppet.layer.set_image(self._puppet.image)
        self._update_puppet_overlay()
        self.app.render()

    def reset_puppet(self):
        if self._puppet is None:
            return
        self._puppet_reset_drag()
        self._puppet.reset()
        self._puppet.layer.set_image(self._puppet.image)
        self._update_puppet_overlay()
        self.app.render()

    def apply_puppet(self):
        if self._puppet is None:
            return
        if self._puppet.previewed or self._puppet_preview_job is not None:
            self._puppet.layer.set_image(self._puppet.render())
        self._puppet = None
        self._puppet_original = None
        self._puppet_pushed = False
        self._puppet_reset_drag()
        self._clear_puppet_overlay()
        self.app.render()
        self.app.set_result("Puppet-warped the active layer.")
        # Keep going when the tool is still active: begin a fresh session on the
        # newly warped pixels, so Apply commits and lets the user keep deforming.
        if self.app.tools.puppet_tool and self.app.current is not None:
            self.start_puppet()

    def cancel_puppet(self):
        """Revert an in-progress puppet warp; returns whether one was running."""
        if self._puppet is None:
            return False
        layer = self._puppet.layer
        if self._puppet_original is not None and layer in self.app.current.layers:
            layer.set_image(self._puppet_original)
        if self._puppet_pushed:
            self.app._drop_undo()
        self._puppet = None
        self._puppet_original = None
        self._puppet_pushed = False
        self._puppet_reset_drag()
        self._clear_puppet_overlay()
        self.app.render()
        self.app.set_result("Cancelled the puppet warp.")
        return True

    def _puppet_reorder(self, forward):
        """Bring the selected pins to the front (or send them to the back)."""
        if self._puppet is None:
            return
        indices = self._puppet.selected_indices()
        if not indices:
            self.app.set_result("Select a pin first.")
            return
        if forward:
            self._puppet.raise_pins(indices)
        else:
            self._puppet.lower_pins(indices)
        self._puppet.layer.set_image(self._puppet.render())
        self._update_puppet_overlay()
        self.app.render()

    def on_puppet_action(self, action):
        """Handle the Puppet settings buttons (clear/reset/apply/cancel/order)."""
        if action == "clear":
            self.clear_puppet_pins()
        elif action == "reset":
            self.reset_puppet()
        elif action == "apply":
            self.apply_puppet()
        elif action == "cancel":
            self.cancel_puppet()
        elif action == "front":
            self._puppet_reorder(True)
        elif action == "back":
            self._puppet_reorder(False)

    # ------------------------------------------------------------------- text
    def _text_spec(self):
        if self._text is None:
            return None
        layer = self._text["layer"]
        index = self._text["index"]
        if 0 <= index < len(layer.texts):
            return layer.texts[index]
        return None

    def current_text_style(self):
        """The style dict of the text object being edited, or ``None``.

        Lets the settings panel mirror the object being edited instead of the
        last style the user applied.
        """
        return self._text_spec()

    # Pickup radius (document pixels at zoom 1) for the transform handles, and
    # the distance within which a move snaps to a guide or canvas centre.
    TEXT_HANDLE_HIT = 8
    TEXT_SNAP = 6
    TEXT_CURSORS = {
        "rot": "exchange",
        "pivot": "crosshair",
        "nw": "size_nw_se", "se": "size_nw_se",
        "ne": "size_ne_sw", "sw": "size_ne_sw",
        "e": "size_we", "w": "size_we",
        "move": "fleur",
    }

    def _text_press(self, event, x, y):
        """Start or continue an inline text edit at document ``(x, y)``."""
        view = event.widget
        if self._text is not None:
            spec = self._text_spec()
            if spec is not None:
                tolerance = self.TEXT_HANDLE_HIT / max(view.zoom, 1e-6)
                handle = textrender.hit_handle(
                    spec, x, y, tolerance, self._text.get("rot_pivot"),
                    bool(event.state & 0x8))
                if handle is not None and handle != "move":
                    self._begin_text_drag(handle, x, y)
                    return
                if handle == "move":
                    # A plain press inside the block places the caret and starts
                    # a text-selection drag; Ctrl turns it into a block move.
                    caret = textrender.caret_from_point(spec, x, y)
                    self._text["caret"] = caret
                    if event.state & 0x4:
                        self._text["anchor"] = None
                        self._begin_text_drag("move", x, y)
                    else:
                        self._text["anchor"] = caret
                        self._begin_text_drag("text", x, y)
                    self._refresh_text_edit()
                    return
            self.commit_text()
        layer = self.app.active_layer()
        # Any text under the cursor opens for editing, wherever it lives: search
        # every visible layer front to back so a text object on its own layer is
        # still re-editable instead of stacking a new layer on each click.
        hit = self._pick_text(x, y)
        if hit is not None:
            hit_layer, index = hit
            # Make the owning layer active so the handles, undo entry and layer
            # list all act on the object that was clicked, and highlight it.
            if hit_layer is not layer:
                self.app.active = self.app.current.layers.index(hit_layer)
            self.app._text_sel = (self.app.active, index)
            self.app._refresh_layer_list()
            self._begin_text_edit(hit_layer, index, view, x, y)
            return
        if layer is None and not self.app.tools.text_new_layer:
            self.app.set_result("There is no layer. Add a layer first.")
            return
        if layer is not None and not self.app.tools.text_new_layer and layer.is_locked():
            if not layer.visible:
                self.app.set_result(f"Layer '{layer.name}' is hidden (locked).")
            else:
                self.app.set_result(f"Layer '{layer.name}' is locked.")
            return
        style = self.app.default_text_style()
        spec = textrender.new_spec("", x, y, **style)
        self.app._push_undo("Add text")
        created = None
        if self.app.tools.text_new_layer:
            doc = self.app.current
            active = doc.active_layer()
            created = doc.add_layer(
                name="Text",
                index=doc.layers.index(active) + 1
                if active is not None else len(doc.layers),
            )
            self.app.active = doc.layers.index(created)
            self.app._refresh_layer_list()
            layer = created
        layer.texts.append(spec)
        layer.invalidate_texts()
        self.app._text_sel = (self.app.active, len(layer.texts) - 1)
        self.app._refresh_layer_list()
        self._text = {
            "layer": layer, "index": len(layer.texts) - 1, "caret": 0,
            "anchor": None, "view": view, "job": None, "created_layer": created,
            "original": None, "pushed": True, "rot_pivot": None,
        }
        self._blink_on = True
        view.focus_set()
        self.app.render()
        self._refresh_text_edit()
        self._start_blink()
        self.app.set_result(
            "Type to add text; drag or Shift+arrows select, double/triple-click "
            "word/line, Ctrl+Z undo, Enter commits, Esc cancels; Alt-drag the "
            "orange origin to set the rotation pivot."
        )

    @staticmethod
    def _text_at(layer, x, y):
        """Index of the topmost text object under ``(x, y)``, or ``None``."""
        for index in range(len(layer.texts) - 1, -1, -1):
            if textrender.hit(layer.texts[index], x, y):
                return index
        return None

    def _pick_text(self, x, y):
        """The text object under ``(x, y)`` across visible layers, or ``None``.

        Returns ``(layer, index)``. Layers are searched front to back so the
        object drawn on top wins, and hidden layers (or hidden groups) are
        skipped, so clicking empty space never edits something the user cannot
        see.
        """
        doc = self.app.current
        if doc is None:
            return None
        for layer in reversed(doc.layers):
            if not self._layer_visible(layer):
                continue
            index = self._text_at(layer, x, y)
            if index is not None:
                return layer, index
        return None

    @staticmethod
    def _layer_visible(layer):
        """Whether ``layer`` and any group it belongs to are visible."""
        if not layer.visible:
            return False
        group = getattr(layer, "group", None)
        return group is None or getattr(group, "visible", True)

    def _begin_text_edit(self, layer, index, view, x, y):
        """Re-open a committed text object for editing, caret at ``(x, y)``."""
        spec = layer.texts[index]
        self._text = {
            "layer": layer, "index": index,
            "caret": textrender.caret_from_point(spec, x, y),
            "anchor": None, "view": view, "job": None, "created_layer": None,
            "original": textrender.clone(spec), "pushed": False,
            "rot_pivot": None,
        }
        self._blink_on = True
        self.app.tools.set_text_style(spec)
        view.focus_set()
        layer.invalidate_texts()
        self.app.render()
        self._refresh_text_edit()
        self._start_blink()
        self.app.set_result(
            "Editing text; drag or Shift+arrows select, double/triple-click "
            "word/line, Ctrl+Z undo, Enter commits, Esc cancels; Alt-drag the "
            "orange origin to set the rotation pivot."
        )

    def edit_text_object(self, layer, index):
        """Re-open a committed text object for editing from the layer panel."""
        if not (0 <= index < len(layer.texts)):
            return
        spec = layer.texts[index]
        self.commit_text()
        if spec not in layer.texts:
            return
        view = self.app.area.current_view()
        if view is None:
            return
        if not self.app.tools.text_tool:
            self.app.tools.select("text")
        self._begin_text_edit(
            layer, layer.texts.index(spec), view,
            spec.get("x", 0.0), spec.get("y", 0.0),
        )

    def _mark_text_dirty(self):
        """Record one undo entry for the first change of a text edit session."""
        if self._text is None or self._text.get("pushed"):
            return
        self._text["pushed"] = True
        self.app._push_undo("Edit text")

    def _begin_text_drag(self, handle, x, y):
        """Capture the state needed to move, scale, resize or rotate the draft."""
        spec = self._text_spec()
        if spec is None:
            return
        if handle not in ("move", "text", "pivot"):
            # Moving is confirmed on the first motion; the other handles only
            # start on an intentional grab, so record the change up front.
            # Repositioning the rotation origin changes no output, so it never
            # records an undo entry.
            self._mark_text_dirty()
        start = dict(spec)
        x0, y0, x1, y1 = textrender.local_rect(spec)
        center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
        pivot_local = self._text.get("rot_pivot") or center
        data = {
            "handle": handle,
            "press": (x, y),
            "start": start,
            "theta0": float(start.get("rotation", 0.0) or 0.0),
            "center_local": center,
            "pivot_local": pivot_local,
            "anchor0": (start["x"], start["y"]),
            "size0": int(start["size"]),
            "width0": textrender.box_width(spec),
        }
        if handle in textrender.CORNER_ORDER:
            opposite = textrender.CORNER_OPPOSITE[handle]
            data["corner_local"] = textrender.corner_local(spec, handle)
            data["opposite"] = opposite
            data["opp_local"] = textrender.corner_local(spec, opposite)
            data["opp_doc"] = textrender.to_doc(spec, *data["opp_local"])
        elif handle in textrender.EDGE_ORDER:
            # Grabbing an edge of an auto-width box first pins the box to its
            # current content width so the drag starts from what is on screen.
            if data["width0"] <= 0:
                data["width0"] = textrender.local_rect(spec)[2]
                spec["width"] = data["width0"]
        elif handle == "rot":
            data["pivot"] = textrender.to_doc(spec, *pivot_local)
            data["angle0"] = math.atan2(y - data["pivot"][1],
                                        x - data["pivot"][0])
        elif handle == "pivot":
            data["pivot_grab"] = (pivot_local[0] - textrender.to_local(spec, x, y)[0],
                                  pivot_local[1] - textrender.to_local(spec, x, y)[1])
        self._text_drag = data

    def _text_drag_move(self, x, y, shift=False):
        if self._text is None or self._text_drag is None:
            return
        spec = self._text_spec()
        if spec is None:
            return
        data = self._text_drag
        handle = data["handle"]
        theta = math.radians(data["theta0"])
        if handle == "pivot":
            lx, ly = textrender.to_local(spec, x, y)
            gx, gy = data["pivot_grab"]
            marker = (lx + gx, ly + gy)
            view = self._text.get("view")
            zoom = max(getattr(view, "zoom", 1.0) or 1.0, 1e-6)
            best = self.TEXT_HANDLE_HIT / zoom
            snapped = None
            for anchor in textrender.anchor_locals(spec):
                distance = math.hypot(anchor[0] - marker[0], anchor[1] - marker[1])
                if distance <= best:
                    best = distance
                    snapped = anchor
            self._text["rot_pivot"] = snapped if snapped is not None else marker
            self._refresh_text_edit()
            return
        if handle == "move":
            self._mark_text_dirty()
            ax = data["anchor0"][0] + (x - data["press"][0])
            ay = data["anchor0"][1] + (y - data["press"][1])
            spec["x"], spec["y"] = self._snap_text(spec, ax, ay)
        elif handle == "text":
            # Extend the selection from the anchor the press recorded.
            self._text["caret"] = textrender.caret_from_point(spec, x, y)
        elif handle == "rot":
            pivot_x, pivot_y = data["pivot"]
            angle = math.atan2(y - pivot_y, x - pivot_x) - data["angle0"]
            theta_deg = data["theta0"] + math.degrees(angle)
            if shift:
                theta_deg = round(theta_deg / 15.0) * 15.0
            spec["rotation"] = theta_deg
            cx, cy = data["pivot_local"]
            rx, ry = textrender.rotate_point(math.radians(theta_deg), cx, cy)
            spec["x"] = pivot_x - rx
            spec["y"] = pivot_y - ry
        elif handle in textrender.EDGE_ORDER:
            # Drag a side edge to set the layout box width, keeping the opposite
            # edge pinned so the box grows away from the grabbed handle.
            lx = textrender.to_local(data["start"], x, y)[0]
            width0 = data["width0"]
            if handle == "e":
                spec["width"] = max(textrender.MIN_BOX_WIDTH, lx)
            else:
                spec["width"] = max(textrender.MIN_BOX_WIDTH, width0 - lx)
                moved = width0 - spec["width"]
                rx, ry = textrender.rotate_point(theta, moved, 0.0)
                spec["x"] = data["anchor0"][0] + rx
                spec["y"] = data["anchor0"][1] + ry
        else:
            # Uniform scale about the opposite corner, keeping it pinned.
            local = textrender.to_local(data["start"], x, y)
            opp = data["opp_local"]
            corner = data["corner_local"]
            span = math.hypot(corner[0] - opp[0], corner[1] - opp[1])
            reach = math.hypot(local[0] - opp[0], local[1] - opp[1])
            factor = reach / span if span > 1e-6 else 1.0
            size = int(round(data["size0"] * factor))
            spec["size"] = max(textrender.MIN_SIZE, min(textrender.MAX_SIZE, size))
            if data["width0"] > 0:
                spec["width"] = max(
                    textrender.MIN_BOX_WIDTH, data["width0"] * factor
                )
            new_opp = textrender.corner_local(spec, data["opposite"])
            rx, ry = textrender.rotate_point(math.radians(data["theta0"]), *new_opp)
            spec["x"] = data["opp_doc"][0] - rx
            spec["y"] = data["opp_doc"][1] - ry
        self._refresh_text_edit()

    def _snap_text(self, spec, ax, ay):
        """Snap a moved anchor to the nearest guide or canvas centre line."""
        view = self._text.get("view") if self._text else None
        if view is None or not getattr(view, "doc", None):
            return ax, ay
        probe = dict(spec)
        probe["x"] = ax
        probe["y"] = ay
        quad = textrender.corners(probe)
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        tolerance = self.TEXT_SNAP / max(view.zoom, 1e-6)
        v_lines = [view.doc.width / 2.0]
        h_lines = [view.doc.height / 2.0]
        for orient, pos in view.guides:
            (v_lines if orient == "v" else h_lines).append(float(pos))
        return ax + self._snap_delta(xs, v_lines, tolerance), \
            ay + self._snap_delta(ys, h_lines, tolerance)

    @staticmethod
    def _snap_delta(coords, lines, tolerance):
        edges = (min(coords), (min(coords) + max(coords)) / 2.0, max(coords))
        best = 0.0
        found = False
        for edge in edges:
            for line in lines:
                delta = line - edge
                if abs(delta) <= tolerance and (not found or abs(delta) < abs(best)):
                    best = delta
                    found = True
        return best if found else 0.0

    def _hover_text(self, event):
        """Show the matching resize/rotate/move cursor over a handle."""
        view = event.widget
        spec = self._text_spec() if self._text is not None else None
        if spec is None:
            return
        x, y = self._doc_coords(event)
        tolerance = self.TEXT_HANDLE_HIT / max(view.zoom, 1e-6)
        handle = textrender.hit_handle(
            spec, x, y, tolerance, self._text.get("rot_pivot"),
            bool(event.state & 0x8))
        # Inside the block a plain drag selects text, so show the text cursor;
        # Ctrl turns the same spot into a move grab.
        if handle == "move" and not (event.state & 0x4):
            cursor = "xterm"
        else:
            cursor = self.TEXT_CURSORS.get(handle, "xterm")
        try:
            view.config(cursor=cursor)
        except tk.TclError:
            pass

    def on_text_key(self, event):
        """Feed keystrokes to the active inline text edit, if any."""
        if self._preview_showing(event):
            return None
        if self.app.tools.vector_tool and self._vector_nudge_key(event):
            return "break"
        if self._text is None:
            return None
        if event.keysym in ("Return", "KP_Enter") and not (event.state & 0x1):
            self.commit_text()
            return "break"
        if event.keysym == "Escape":
            self.cancel_text()
            return "break"
        keysym = event.keysym
        shift = bool(event.state & 0x1)
        if event.state & 0x4:  # a Control chord edits the text, not the canvas
            return self._text_control_key(keysym.lower(), shift)
        if keysym == "BackSpace":
            self._delete_text(back=True)
        elif keysym == "Delete":
            self._delete_text(back=False)
        elif keysym == "Left":
            self._move_caret(-1, extend=shift)
        elif keysym == "Right":
            self._move_caret(1, extend=shift)
        elif keysym == "Up":
            self._move_caret_vertical(-1, extend=shift)
        elif keysym == "Down":
            self._move_caret_vertical(1, extend=shift)
        elif keysym == "Home":
            self._caret_line_edge(start=True, extend=shift)
        elif keysym == "End":
            self._caret_line_edge(start=False, extend=shift)
        elif keysym == "KP_Enter":
            self._insert_text("\n")
        elif event.char and event.char.isprintable():
            self._insert_text(event.char)
        else:
            return None
        return "break"

    def _text_control_key(self, keysym, shift=False):
        """Handle a Control chord while editing text.

        Returns ``"break"`` for the chords the edit consumes and ``None`` for
        the rest, so app-level shortcuts (save, ...) still reach the window.
        Text editing owns its own undo history, so Ctrl+Z/Ctrl+Y act on the
        edit rather than falling through to the canvas undo.
        """
        if keysym == "a":
            self.select_all_text()
            return "break"
        if keysym == "c":
            self._copy_text(cut=False)
            return "break"
        if keysym == "x":
            self._copy_text(cut=True)
            return "break"
        if keysym == "v":
            self._paste_text()
            return "break"
        if keysym == "z":
            if shift:
                self._text_redo()
            else:
                self._text_undo()
            return "break"
        if keysym == "y":
            self._text_redo()
            return "break"
        if keysym == "left":
            self._move_caret_word(-1, extend=shift)
            return "break"
        if keysym == "right":
            self._move_caret_word(1, extend=shift)
            return "break"
        if keysym == "backspace":
            self._delete_word(back=True)
            return "break"
        if keysym == "delete":
            self._delete_word(back=False)
            return "break"
        if keysym == "home":
            self._caret_text_edge(start=True, extend=shift)
            return "break"
        if keysym == "end":
            self._caret_text_edge(start=False, extend=shift)
            return "break"
        return None

    def _text_range(self):
        """The selected ``(start, end)`` caret range, or ``None`` when collapsed."""
        if self._text is None:
            return None
        anchor = self._text.get("anchor")
        caret = self._text["caret"]
        if anchor is None or anchor == caret:
            return None
        return (min(anchor, caret), max(anchor, caret))

    def select_all_text(self):
        """Select the whole text of the object being edited."""
        spec = self._text_spec()
        if spec is None:
            return
        self._text["anchor"] = 0
        self._text["caret"] = len(spec["text"])
        self._refresh_text_edit()

    def _copy_text(self, cut):
        """Copy (or cut) the current text selection to the clipboard."""
        spec = self._text_spec()
        selection = self._text_range()
        if spec is None or selection is None:
            return
        start, end = selection
        view = self._text.get("view")
        if view is not None:
            view.clipboard_clear()
            view.clipboard_append(spec["text"][start:end])
        if cut:
            self._delete_text(back=False)

    def _paste_text(self):
        """Insert the clipboard text at the caret, replacing any selection."""
        spec = self._text_spec()
        view = self._text.get("view")
        if spec is None or view is None:
            return
        try:
            chunk = view.clipboard_get()
        except tk.TclError:
            return
        chunk = chunk.replace("\r\n", "\n").replace("\r", "\n")
        if chunk:
            self._insert_text(chunk)

    def _insert_text(self, chunk):
        spec = self._text_spec()
        if spec is None or not chunk:
            return
        self._mark_text_dirty()
        self._push_text_history("type" if len(chunk) == 1 else "paste")
        selection = self._text_range()
        if selection is not None:
            start, end = selection
        else:
            start = end = self._text["caret"]
        text = spec["text"]
        spec["text"] = text[:start] + chunk + text[end:]
        self._text["caret"] = start + len(chunk)
        self._text["anchor"] = None
        self._refresh_text_edit()

    def _delete_text(self, back):
        spec = self._text_spec()
        if spec is None:
            return
        selection = self._text_range()
        if selection is not None:
            # A selection deletes as a block, whichever key was pressed.
            self._mark_text_dirty()
            self._push_text_history("block")
            start, end = selection
            text = spec["text"]
            spec["text"] = text[:start] + text[end:]
            self._text["caret"] = start
            self._text["anchor"] = None
            self._refresh_text_edit()
            return
        caret = self._text["caret"]
        text = spec["text"]
        if back:
            if caret <= 0:
                return
            self._mark_text_dirty()
            self._push_text_history("back")
            spec["text"] = text[:caret - 1] + text[caret:]
            self._text["caret"] = caret - 1
        else:
            if caret >= len(text):
                return
            self._mark_text_dirty()
            self._push_text_history("forward")
            spec["text"] = text[:caret] + text[caret + 1:]
        self._refresh_text_edit()

    def _move_caret(self, delta, extend=False):
        spec = self._text_spec()
        if spec is None:
            return
        if extend:
            if self._text.get("anchor") is None:
                self._text["anchor"] = self._text["caret"]
        else:
            self._text["anchor"] = None
        caret = max(0, min(len(spec["text"]), self._text["caret"] + delta))
        self._text["caret"] = caret
        self._refresh_text_edit()

    def _caret_line_edge(self, start, extend=False):
        spec = self._text_spec()
        if spec is None:
            return
        text = spec["text"]
        caret = self._text["caret"]
        if extend:
            if self._text.get("anchor") is None:
                self._text["anchor"] = caret
        else:
            self._text["anchor"] = None
        if start:
            self._text["caret"] = text.rfind("\n", 0, caret) + 1
        else:
            end = text.find("\n", caret)
            self._text["caret"] = len(text) if end == -1 else end
        self._refresh_text_edit()

    def _start_selection(self, extend):
        """Anchor the selection at the caret, or clear it, before a move."""
        if extend:
            if self._text.get("anchor") is None:
                self._text["anchor"] = self._text["caret"]
        else:
            self._text["anchor"] = None

    def _move_caret_word(self, direction, extend=False):
        """Move the caret one word left/right, optionally extending."""
        spec = self._text_spec()
        if spec is None:
            return
        self._start_selection(extend)
        text = spec["text"]
        caret = self._text["caret"]
        if direction > 0:
            self._text["caret"] = textrender.word_right(text, caret)
        else:
            self._text["caret"] = textrender.word_left(text, caret)
        self._refresh_text_edit()

    def _move_caret_vertical(self, direction, extend=False):
        """Move the caret to the adjacent visual row, keeping its column."""
        spec = self._text_spec()
        if spec is None:
            return
        self._start_selection(extend)
        self._text["caret"] = textrender.caret_vertical(
            spec, self._text["caret"], direction
        )
        self._refresh_text_edit()

    def _caret_text_edge(self, start, extend=False):
        """Move the caret to the very start or end of the text (Ctrl+Home/End)."""
        spec = self._text_spec()
        if spec is None:
            return
        self._start_selection(extend)
        self._text["caret"] = 0 if start else len(spec["text"])
        self._refresh_text_edit()

    def _delete_word(self, back):
        """Delete the word before/after the caret (Ctrl+BackSpace/Delete)."""
        spec = self._text_spec()
        if spec is None:
            return
        if self._text_range() is not None:
            # A selection always deletes as a block.
            self._delete_text(back=back)
            return
        text = spec["text"]
        caret = self._text["caret"]
        if back:
            target = textrender.word_left(text, caret)
        else:
            target = textrender.word_right(text, caret)
        if target == caret:
            # Nothing to swallow on that side; fall back to a single character.
            self._delete_text(back=back)
            return
        self._mark_text_dirty()
        self._push_text_history("word-delete")
        if back:
            spec["text"] = text[:target] + text[caret:]
            self._text["caret"] = target
        else:
            spec["text"] = text[:caret] + text[target:]
        self._text["anchor"] = None
        self._refresh_text_edit()

    def _select_text_unit(self, x, y, word):
        """Select the word (double-click) or logical line (triple-click) at x, y."""
        spec = self._text_spec()
        if spec is None:
            return
        position = textrender.caret_from_point(spec, x, y)
        if word:
            start, end = textrender.word_bounds(spec["text"], position)
        else:
            start, end = textrender.line_bounds(spec["text"], position)
        if end <= start:
            return
        self._text["anchor"] = start
        self._text["caret"] = end
        self._text["_last_op"] = None
        self._refresh_text_edit()

    # ------------------------------------------------------- text edit history
    def _text_snapshot(self):
        """The state pushed onto the edit's undo history."""
        spec = self._text_spec()
        if spec is None:
            return None
        return (spec["text"], self._text["caret"], self._text.get("anchor"))

    def _push_text_history(self, op):
        """Record the pre-edit state, coalescing a run of the same quick edit."""
        if self._text is None:
            return
        history = self._text.setdefault("history", [])
        now = time.monotonic()
        same_run = (
            op == self._text.get("_last_op")
            and now - self._text.get("_last_time", 0.0) < 0.8
        )
        if not same_run:
            snapshot = self._text_snapshot()
            if snapshot is not None:
                history.append(snapshot)
                if len(history) > 200:
                    del history[0]
            self._text["future"] = []
        self._text["_last_op"] = op
        self._text["_last_time"] = now

    def _apply_text_snapshot(self, snapshot):
        spec = self._text_spec()
        if spec is None:
            return
        text, caret, anchor = snapshot
        spec["text"] = text
        self._text["caret"] = max(0, min(len(text), caret))
        self._text["anchor"] = anchor
        self._text["_last_op"] = None
        self._refresh_text_edit()

    def _text_undo(self):
        """Undo the last edit within the current text session."""
        if self._text is None:
            return
        history = self._text.setdefault("history", [])
        if not history:
            return
        current = self._text_snapshot()
        if current is not None:
            self._text.setdefault("future", []).append(current)
        self._apply_text_snapshot(history.pop())

    def _text_redo(self):
        """Redo the last undone edit within the current text session."""
        if self._text is None:
            return
        future = self._text.setdefault("future", [])
        if not future:
            return
        current = self._text_snapshot()
        if current is not None:
            self._text.setdefault("history", []).append(current)
        self._apply_text_snapshot(future.pop())

    def _refresh_text_edit(self):
        if self._text is None:
            return
        layer = self._text["layer"]
        layer.invalidate_texts()
        self._blink_on = True
        self.app.render()
        self._update_text_overlay()
        # Keep the caret solid while actively editing: restart the blink so it
        # only starts toggling once the user pauses.
        self._stop_blink()
        self._start_blink()

    def _update_text_overlay(self):
        if self._text is None:
            return
        spec = self._text_spec()
        if spec is None:
            return
        geometry = textrender.session_geometry(
            spec, self._text["caret"], self._text.get("anchor"),
            self._text.get("rot_pivot"),
        )
        geometry["caret_on"] = self._blink_on
        view = self._text.get("view")
        if view is not None and view.winfo_exists():
            view.set_text_session(geometry)

    def _start_blink(self):
        if self._text is None:
            return
        # Cancel any previous timer so callers can restart it idempotently.
        self._stop_blink()
        self._text["job"] = self._text["view"].after(500, self._blink)

    def _blink(self):
        if self._text is None:
            return
        self._blink_on = not self._blink_on
        view = self._text.get("view")
        if view is not None and view.winfo_exists():
            view.set_text_caret_on(self._blink_on)
            self._text["job"] = view.after(500, self._blink)

    def _stop_blink(self):
        if self._text is None:
            return
        job = self._text.get("job")
        view = self._text.get("view")
        if job is not None and view is not None:
            try:
                view.after_cancel(job)
            except Exception:
                pass
        self._text["job"] = None

    def _end_text_session(self):
        self._stop_blink()
        self._text_drag = None
        view = self._text.get("view") if self._text else None
        self._text = None
        if view is not None and view.winfo_exists():
            view.set_text_session(None)
        # Drop back to the last style the user chose for the next new text.
        self.app.tools.set_text_style(self.app.default_text_style())

    def commit_text(self):
        """Finish the inline edit, keeping the text (or dropping it if empty)."""
        if self._text is None:
            return
        layer = self._text["layer"]
        spec = self._text_spec()
        created = self._text.get("created_layer")
        original = self._text.get("original")
        pushed = self._text.get("pushed")
        self._end_text_session()
        empty = spec is None or not spec.get("text", "").strip()
        if empty and original is not None:
            # An existing text emptied to nothing is a delete, not a no-op.
            if not pushed:
                self.app._push_undo("Delete text")
            if spec is not None:
                try:
                    layer.texts.remove(spec)
                except ValueError:
                    pass
            layer.invalidate_texts()
            self.app._refresh_layer_list()
            self.app.render()
            self.app.set_result(f"Removed text on '{layer.name}'.")
            return
        if empty:
            if spec is not None:
                try:
                    layer.texts.remove(spec)
                except ValueError:
                    pass
            if pushed:
                self.app._drop_undo()
            self._discard_text_layer(created)
            layer.invalidate_texts()
            self.app._refresh_layer_list()
            self.app.render()
            return
        if not pushed and original is not None and spec == original:
            # Re-opened and closed without editing: nothing to record.
            layer.invalidate_texts()
            self.app.render()
            self.app.set_result("")
            return
        if not pushed:
            self.app._push_undo("Edit text")
        layer.invalidate_texts()
        self.app._refresh_layer_list()
        self.app.render()
        if original is None:
            self.app.set_result(f"Added text on '{layer.name}'.")
        else:
            self.app.set_result(f"Edited text on '{layer.name}'.")

    def _discard_text_layer(self, layer):
        """Remove the layer a cancelled or empty text session created."""
        if layer is None:
            return
        doc = self.app.current
        if layer not in doc.layers:
            return
        index = doc.layers.index(layer)
        del doc.layers[index]
        self.app.active = (
            min(max(0, index - 1), len(doc.layers) - 1) if doc.layers else 0
        )
        self.app._refresh_layer_list()

    def cancel_text(self):
        """Discard the inline edit and any undo entry it recorded.

        Returns whether an edit was in progress.
        """
        if self._text is None:
            return False
        layer = self._text["layer"]
        index = self._text["index"]
        spec = self._text_spec()
        created = self._text.get("created_layer")
        original = self._text.get("original")
        pushed = self._text.get("pushed")
        self._end_text_session()
        if original is not None:
            # Re-edited text: restore what the object looked like on entry.
            if 0 <= index < len(layer.texts):
                layer.texts[index] = textrender.normalize(original)
        elif spec is not None:
            try:
                layer.texts.remove(spec)
            except ValueError:
                pass
        if pushed:
            self.app._drop_undo()
        self._discard_text_layer(created)
        layer.invalidate_texts()
        self.app._refresh_layer_list()
        self.app.render()
        self.app.set_result("")
        return True

    def sync_text(self):
        """Commit an in-progress edit when the Text tool is no longer active."""
        if self._text is not None and not self.app.tools.text_tool:
            self.commit_text()

    def update_text_style(self, style):
        """Apply a style change to the text being edited."""
        if self._text is None:
            return
        spec = self._text_spec()
        if spec is None:
            return
        merged = dict(spec)
        for key in ("family", "size", "bold", "italic", "color", "align",
                    "line_spacing", "letter_spacing", "stroke", "shadow",
                    "wrap"):
            if key in style:
                merged[key] = style[key]
        if merged.get("wrap") and textrender.box_width(merged) <= 0:
            # Turning wrapping on with no box yet pins the box to the text so
            # the side handles have something to drag.
            merged["width"] = max(
                textrender.MIN_BOX_WIDTH, textrender.local_rect(spec)[2]
            )
        updated = textrender.normalize(merged)
        if updated == spec:
            return
        self._mark_text_dirty()
        self._text["layer"].texts[self._text["index"]] = updated
        self._refresh_text_edit()

    def _drop_duplicate_anchor(self):
        """Remove the anchor a double-click's second press just added."""
        path = self._path
        if path is None or len(path) < 2:
            return
        ax, ay = path.anchors[-1]
        bx, by = path.anchors[-2]
        if abs(ax - bx) <= 2 and abs(ay - by) <= 2:
            path.remove_last()

    def finish_path(self):
        """Commit the current path, keeping it editable on the layer.

        A finished path is stored as an editable object on the layer (like text
        and vector strokes) rather than baked into pixels, so it can be reopened
        and reshaped; 'Rasterize Paths' folds it into the pixels later. When an
        object is already open for editing, the finish writes the edited
        geometry and style back to it; otherwise a new object is appended. In
        fill mode the path is closed and the enclosed shape filled, in stroke
        mode it is stroked with the current brush size, and ``both`` fills and
        then strokes the outline. Fill output can carry an outline width, a soft
        feathered edge and an overall opacity.
        """
        self._path_dragging = None
        path = self._path
        if path is None:
            return
        self._path_move_origin = None
        self._path_move_start = None
        edit = self._path_edit
        tools = self.app.tools
        fill = tools.path_mode in ("fill", "both")
        minimum = 3 if fill else 2
        if edit is None and len(path) < minimum:
            self.cancel_path()
            what = "fill" if fill else "line"
            self.app.set_result(
                f"A path needs at least {minimum} points to {what}."
            )
            return
        if edit is not None:
            spec = self._path_edit_spec()
            if spec is None:
                self.cancel_path()
                return
            layer = edit["layer"]
            original = edit.get("original")
            pathspec.apply_geometry(spec, path)
            self._apply_path_style(spec)
            count = len(path)
            if original is not None and spec != original:
                self._mark_path_dirty()
            layer.invalidate_paths()
            layers = self.app.current.layers
            layer_index = layers.index(layer) if layer in layers else None
            self.app._path_sel = (
                (layer_index, edit["index"]) if layer_index is not None else None
            )
            self._path = None
            self._path_edit = None
            self._path_view = None
            self._path_draft = False
            self._update_path_preview()
            self.app._refresh_layer_list()
            self.app.render()
            self.app.set_result(
                f"Updated path ({count} points) on '{layer.name}'."
            )
            return
        if not tools.path_new_layer and self._layer_locked():
            self.cancel_path()
            return
        self.app._push_undo("Path")
        if tools.path_new_layer:
            doc = self.app.current
            layer = doc.add_layer(
                name="Path",
                index=(
                    doc.layers.index(doc.active_layer()) + 1
                    if doc.active_layer() is not None else len(doc.layers)
                ),
            )
            self.app.active = doc.layers.index(layer)
            self.app._refresh_layer_list()
        else:
            layer = self.app.active_layer()
        self.app.current.align_for_painting(layer)
        spec = pathspec.from_path(path, **tools.path_style())
        layer.paths.append(spec)
        layer.invalidate_paths()
        self.app._path_sel = (self.app.current.layers.index(layer),
                              len(layer.paths) - 1)
        count = len(path)
        if tools.path_mode == "both":
            verb = "Filled and stroked"
        elif tools.path_mode == "stroke":
            verb = "Stroked"
        else:
            verb = "Filled"
        self._path = None
        self._path_view = None
        self._update_path_preview()
        self.app._refresh_layer_list()
        self.app.render()
        self.app.set_result(
            f"{verb} path ({count} points) on '{layer.name}'."
        )

    def cancel_path(self):
        """Discard the in-progress path, restoring an edited object.

        Returns whether there was a session to discard, so Escape can peel one
        layer of state at a time.
        """
        self._path_dragging = None
        edit = self._path_edit
        spec = self._path_edit_spec()
        had_session = edit is not None or self._path is not None
        if edit is not None:
            original = edit.get("original")
            if spec is not None and original is not None:
                spec.clear()
                spec.update(original)
                edit["layer"].invalidate_paths()
            if edit.get("pushed"):
                self.app._drop_undo()
        self._path_edit = None
        self._clear_path_edit()
        if self._path is not None:
            self._path = None
            self._update_path_preview()
            self._path_view = None
        if edit is not None:
            self.app._refresh_layer_list()
            self.app.render()
        if had_session:
            self.app.set_result("")
        return had_session

    def sync_path(self):
        """Commit or discard the in-progress path when the Path tool is left."""
        if self.app.tools.path_tool or self._path is None:
            return
        if self._path_edit is not None:
            self.finish_path()
        else:
            self.cancel_path()

    # --------------------------------------------------------------- transform
    TRANSFORM_CURSORS = {
        "rot": "exchange",
        "pivot": "crosshair",
        "nw": "size_nw_se", "se": "size_nw_se",
        "ne": "size_ne_sw", "sw": "size_ne_sw",
        "n": "size_ns", "s": "size_ns",
        "e": "size_we", "w": "size_we",
        "move": "fleur",
    }

    # Minimum seconds between transform drag repaints. A scale/rotate drag
    # resamples the layer and re-rasterises its document-space overlays, which on
    # a large or scaled layer costs far more than one motion event's interval, so
    # doing it per event leaves the drag laggy and unresponsive. The gizmo frame
    # still tracks every event; only the expensive pixel/overlay work is rate
    # limited, and a release paints the exact final frame (see
    # ``TransformGizmo.finish``).
    TRANSFORM_RENDER_INTERVAL = 0.05

    def _transform_ready(self):
        return self.gizmo.active and self.gizmo.layer is self.app.active_layer()

    def _selection_transform_active(self):
        """True when the Select tool's "Transform selection" toggle is on."""
        tools = self.app.tools
        return bool(tools.select_tool
                    and getattr(tools, "select_transform", False))

    def sync_transform(self):
        """(Re)build the transform gizmo for the active tool and layer."""
        view = self.app.area.current_view()
        if self.app.tools.transform_tool:
            self.sel_gizmo.clear()
            self._sync_layer_transform(view)
            return
        self.gizmo.clear()
        if not self._selection_transform_active():
            self.sel_gizmo.clear()
            return
        self._sync_selection_transform(view)

    def _sync_layer_transform(self, view):
        if view is None:
            self.gizmo.clear()
            return
        layer = self.app.active_layer()
        if layer is None:
            self.gizmo.clear()
            return
        # Editable overlays (text, paths, vector items, poses) are kept live:
        # the gizmo carries their geometry through the transform too, so they no
        # longer have to be folded into pixels to stay aligned.
        if not self.gizmo.begin(self.app, view):
            # The active layer has nothing transformable (empty, hidden or a
            # locked master): drop the previous session so its frame does not
            # linger on screen with no live gizmo behind it.
            self.gizmo.clear()
            self.app.set_result("Nothing to transform on this layer.")
            return
        view.set_gizmo(self.gizmo.geometry())
        self.app.render()

    def _sync_selection_transform(self, view):
        if view is None:
            return
        selection = self._selection()
        if selection is None or selection.is_empty():
            self.sel_gizmo.clear()
            self.app.set_result("Make a selection, then transform it.")
            return
        if not self.sel_gizmo.begin_selection(selection, self.app.current.size):
            self.sel_gizmo.clear()
            return
        self.sel_gizmo.view = view
        self.sel_gizmo.on_change = self._set_selection
        view.set_gizmo(self.sel_gizmo.geometry())
        self.app.render()

    def cancel_transform(self):
        """Revert the in-progress drag; committed transforms are kept.

        Returns whether a drag was reverted.
        """
        changed = False
        if self.gizmo.active and self.gizmo.dragging:
            self.gizmo.cancel_drag(self.app)
            self._transform_pushed = False
            self.app.render()
            changed = True
        if self.sel_gizmo.active and self.sel_gizmo.dragging:
            if self.sel_gizmo.source_selection is not None:
                self._set_selection(self.sel_gizmo.source_selection)
            self.sel_gizmo.finish()
            self.sync_transform()
            changed = True
        return changed

    def _set_transform_cursor(self, view, handle):
        try:
            view.config(cursor=self.TRANSFORM_CURSORS.get(handle, "crosshair"))
        except tk.TclError:
            view.config(cursor="crosshair")

    def _hover_transform(self, event):
        view = event.widget
        handle = None
        if self._transform_ready():
            x, y = self._doc_coords(event)
            handle = self.gizmo.hit(x, y, self.gizmo.HANDLE_HIT / view.zoom)
        self._set_transform_cursor(view, handle)

    def _hover_selection_transform(self, event):
        view = event.widget
        handle = None
        if self.sel_gizmo.active:
            x, y = self._doc_coords(event)
            handle = self.sel_gizmo.hit(
                x, y, self.sel_gizmo.HANDLE_HIT / view.zoom)
        self._set_transform_cursor(view, handle)

    def _transform_press(self, event):
        if not self._transform_ready():
            self.sync_transform()
        if not self._transform_ready():
            return
        view = event.widget
        x, y = self._doc_coords(event)
        handle = self.gizmo.hit(x, y, self.gizmo.HANDLE_HIT / view.zoom)
        if handle is None:
            return
        self.gizmo.start(handle, x, y)
        self._transform_pushed = False
        # Paint the first drag frame at once rather than inheriting the previous
        # session's throttle timestamp.
        self._transform_render_at = 0.0

    def _transform_drag(self, event, x, y):
        if not self._transform_ready() or not self.gizmo.dragging:
            return
        if not self.gizmo.moving_pivot and not self._transform_pushed:
            self.app._push_undo("Transform")
            self._transform_pushed = True
        uniform = bool(event.state & 0x1)
        # Moving the rotation origin touches no pixels, and the first event of
        # any drag must paint immediately; otherwise rate-limit the resample and
        # overlay work so a fast drag is not stalled by a frame it cannot keep
        # up with. The gizmo frame is refreshed every event regardless.
        pivot = self.gizmo.moving_pivot
        now = time.monotonic()
        apply = pivot or (now - self._transform_render_at >= self.TRANSFORM_RENDER_INTERVAL)
        if self.gizmo.update(x, y, uniform, apply_pixels=apply, draft=True):
            event.widget.set_gizmo(self.gizmo.geometry())
            if apply:
                self._transform_render_at = now
                self.app.render()

    def _transform_release(self):
        if not self.gizmo.active:
            return
        if self.gizmo.moving_pivot:
            self.app.set_result("Moved the rotation origin.")
        elif self._transform_pushed:
            self.app.set_result("Transformed layer.")
        self.gizmo.finish()
        self._transform_pushed = False
        # A deferred move bakes only on release, so repaint the committed state.
        self.app.render()

    def _selection_transform_press(self, event):
        if not self.sel_gizmo.active:
            self.sync_transform()
        if not self.sel_gizmo.active:
            return
        view = event.widget
        x, y = self._doc_coords(event)
        handle = self.sel_gizmo.hit(x, y, self.sel_gizmo.HANDLE_HIT / view.zoom)
        if handle is None:
            return
        self.sel_gizmo.start(handle, x, y)
        self._sel_transform_dragged = False

    def _selection_transform_drag(self, event, x, y):
        if not self.sel_gizmo.active or not self.sel_gizmo.dragging:
            return
        if not self.sel_gizmo.moving_pivot:
            self._sel_transform_dragged = True
        uniform = bool(event.state & 0x1)
        if self.sel_gizmo.update(x, y, uniform):
            event.widget.set_gizmo(self.sel_gizmo.geometry())
            self.app.render()

    def _selection_transform_release(self):
        if not self.sel_gizmo.active:
            return
        if self.sel_gizmo.moving_pivot:
            self.app.set_result("Moved the rotation origin.")
        elif getattr(self, "_sel_transform_dragged", False):
            self.app.set_result("Transformed selection.")
        self.sel_gizmo.finish()
        self._sel_transform_dragged = False

    # ------------------------------------------------------------------ move
    def _begin_move(self, x, y):
        doc = self.app.current
        layer = self.app.active_layer()
        if doc is None or layer is None:
            self._move_layers = []
            return
        indices = self.app.layer_panel.selected_indices(len(doc.layers))
        if not indices:
            indices = [doc.layers.index(layer)]
        # The Move tool moves every selected layer at once, but never a locked
        # one (which includes the hidden ones); the press on a locked active
        # layer was already refused before we get here. A link layer counts as
        # locked for painting, but its position is per-instance, so it can still
        # be moved here (which shifts its offset, not the master).
        self._move_layers = []
        for item in (doc.layers[index] for index in indices):
            if item.is_locked() and not item.is_link:
                continue
            item._content_delta = (0, 0)
            self._move_layers.append((item, tuple(item.offset), {
                # Texts, vector items, path objects and pose figures live in
                # document space, so they are shifted with the pixels rather than
                # left pinned to the page (which would clip them as the layer's
                # window slides over them).
                # A link's stroke/fill/path content belongs to its master, so it
                # is not shifted.
                "texts": [(spec, spec["x"], spec["y"]) for spec in item.texts],
                # Pose ``x``/``y`` are derived from the shared scene anchor, so
                # the scene - not the specs - is what a move shifts.
                "pose_scene": self._move_pose_scene(item),
                "vector": [] if item.is_link else [
                    (spec, vector.item_snapshot(spec))
                    for spec in (list(getattr(item, "strokes", []))
                                 + list(getattr(item, "fills", [])))
                ],
                "paths": [] if item.is_link else [
                    (spec, pathspec.clone(spec))
                    for spec in list(getattr(item, "paths", []))
                ],
            }))
        # Freeze the view's content bounds for the drag so its scroll region does
        # not track the moved layer (see ``PaintApp._content_bounds``).
        if self._move_layers:
            self.app._move_bounds = self.app._content_bounds()
        self._move_origin = (x, y)
        self._move_pushed = False

    def _move_to(self, x, y):
        if not self._move_layers or self._move_origin is None:
            return
        if not self._move_pushed:
            self.app._push_undo("Move layers")
            self._move_pushed = True
        ox, oy = self._move_origin
        dx = x - ox
        dy = y - oy
        for layer, (start_x, start_y), _extra in self._move_layers:
            layer.offset = (start_x + dx, start_y + dy)
            # Document-space content (text objects, pose figures, vector lines
            # and fills, and path objects) is drawn at ``spec - offset``, so it
            # would stay pinned to the page as the offset slides. Translating
            # every spec here would instead re-rasterise every overlay on each
            # motion event (the path/pose move lag). Instead, record the drag
            # delta: the layer's own image is drawn at ``spec - content_offset``
            # with ``content_offset`` held at the drag's start, so it does not
            # change at all during the drag - only the placement slides over the
            # backdrop - and the specs are translated once on release.
            layer._content_delta = (dx, dy)
        self.app.render()

    def _move_pose_scene(self, layer):
        """Snapshot a layer's poses scene anchor for a move, or ``None``.

        A pose figure's ``x``/``y`` are recomputed from the shared scene anchor
        before every render, so a move shifts the scene - not the specs. A link's
        figures belong to its master, so its own scene is left alone.
        """
        if layer.is_link:
            return None
        scene = getattr(layer, "pose_scene", None)
        if not isinstance(scene, dict):
            return None
        return (scene, float(scene.get("x", 0.0)), float(scene.get("y", 0.0)))

    def _end_move(self):
        # Bake the drag delta into the moved layers' specs once, at release. The
        # overlays were drawn all drag at the start offset (see ``_move_to``), so
        # shifting the specs and clearing the delta leaves the layer's own image
        # unchanged on screen; only the now-stale overlay caches are dropped, so
        # they are rebuilt once rather than on every motion event.
        for layer, _start, extra in self._move_layers:
            dx, dy = layer._content_delta
            if dx or dy:
                for spec, spec_x, spec_y in extra["texts"]:
                    spec["x"] = spec_x + dx
                    spec["y"] = spec_y + dy
                if extra["texts"]:
                    layer.invalidate_texts()
                if extra["pose_scene"] is not None:
                    scene, scene_x, scene_y = extra["pose_scene"]
                    scene["x"] = scene_x + dx
                    scene["y"] = scene_y + dy
                    layer.invalidate_poses()
                for spec, snapshot in extra["vector"]:
                    vector.item_restore(spec, snapshot)
                    vector.item_translate(spec, dx, dy)
                if extra["vector"]:
                    layer.invalidate_vectors()
                for spec, snapshot in extra["paths"]:
                    pathspec.apply_geometry(spec, pathspec.to_path(snapshot))
                    pathspec.translate(spec, dx, dy)
                if extra["paths"]:
                    layer.invalidate_paths()
            layer._content_delta = (0, 0)
        self._move_layers = []
        self._move_origin = None
        self._move_pushed = False
        # Let the scroll region follow the new content bounds again now the drag
        # has ended.
        self.app._move_bounds = None

    # ----------------------------------------------------------------- guides
    # Guides are grabbed with the Move tool, so a brush stroke near one still
    # paints instead of yanking the guide.
    GUIDE_HIT = 5
    GUIDE_CURSORS = {"v": "size_we", "h": "size_ns"}

    def _guide_at(self, view, x, y):
        tolerance = self.GUIDE_HIT / max(view.zoom, 1e-6)
        return view.guide_hit(x, y, tolerance)

    def _begin_guide(self, event, x, y):
        view = event.widget
        index = self._guide_at(view, x, y)
        if index is None:
            return False
        self._guide_index = index
        self._guide_view = view
        self._guide_pushed = False
        return True

    def _drag_guide(self, x, y):
        if self._guide_index is None:
            return
        view = self._guide_view
        if not self._guide_pushed:
            view.push_guide_undo()
            self._guide_pushed = True
        orient = view.guides[self._guide_index][0]
        view.set_guide_pos(self._guide_index, x if orient == "v" else y)

    def _end_guide(self, event):
        view = self._guide_view
        index = self._guide_index
        pushed = self._guide_pushed
        self._guide_index = None
        self._guide_view = None
        self._guide_pushed = False
        if index is None:
            return
        inside = (
            0 <= event.x <= view.winfo_width()
            and 0 <= event.y <= view.winfo_height()
        )
        if inside:
            if pushed:
                self.app.set_result("Moved guide.")
            return
        # Released over a ruler or the surrounding chrome: remove the guide.
        if not pushed:
            view.push_guide_undo()
        view.pop_guide(index)
        self.app.set_result("Removed guide.")

    def _hover_guide(self, event):
        if self._guide_index is not None:
            return
        view = event.widget
        x, y = self._doc_coords(event)
        index = self._guide_at(view, x, y)
        if index is None:
            cursor = "fleur"
        else:
            cursor = self.GUIDE_CURSORS[view.guides[index][0]]
        try:
            view.config(cursor=cursor)
        except tk.TclError:
            pass

    # ------------------------------------------------------------------ pan
    def on_middle_press(self, event):
        # Shift+middle-drag rotates the 2D view about the viewport centre,
        # Ctrl+middle flips it horizontally about the same point and
        # Ctrl+Shift+middle flips it vertically, while plain middle-drag pans
        # the viewport. Alt+middle-drag still pans the pose scene's 3D camera
        # while posing.
        if (event.state & 0x1) and (event.state & 0x4):  # Shift+Ctrl: vertical flip.
            event.widget.flip_view_vertical()
            self.app.set_result("Flipped the view vertically.")
            return "break"
        if event.state & 0x1:  # Shift: begin a view rotation.
            self._begin_view_rotate(event)
            return "break"
        if event.state & 0x4:  # Control: mirror the view horizontally.
            event.widget.flip_view()
            self.app.set_result("Flipped the view horizontally.")
            return "break"
        if self.app.tools.poses_tool and (event.state & ALT_STATE_MASK):
            self._begin_pose_pan(event)
            return "break"
        self._begin_pan(event)
        return "break"

    def on_middle_drag(self, event):
        if self._view_rotate is not None:
            self._view_rotate_to(event)
            return "break"
        if self._pose_pan is not None:
            self._pose_pan_to(event)
            return "break"
        self._pan_to(event)
        return "break"

    def on_middle_release(self, _event):
        if self._view_rotate is not None:
            self._end_view_rotate()
            return "break"
        if self._pose_pan is not None:
            self._end_pose_pan()
            return "break"
        self._end_pan()
        return "break"

    def on_middle_reset(self, event):
        """A middle double-click returns the view to its unrotated, unflipped state."""
        if event.state & (0x1 | 0x4 | ALT_STATE_MASK):
            return "break"
        event.widget.reset_view()
        self.app.set_result("Reset the view rotation and flips.")
        return "break"

    def _begin_view_rotate(self, event):
        """Start rotating the view; the pointer's angle about the pivot drives it."""
        view = event.widget
        cx, cy = view.begin_view_rotate()
        self._view_rotate = {
            "view": view,
            "center": (cx, cy),
            "start_pointer": math.atan2(event.y - cy, event.x - cx),
            "start_angle": view.view_angle,
            "moved": False,
        }
        self.cursor_x = None
        self.cursor_y = None
        view.set_cursor_overlay(None)

    def _view_rotate_to(self, event):
        state = self._view_rotate
        if state is None:
            return
        view = state["view"]
        cx, cy = state["center"]
        angle = math.atan2(event.y - cy, event.x - cx)
        delta = angle - state["start_pointer"]
        if abs(delta) > 0.002:
            state["moved"] = True
        view.set_view_angle(state["start_angle"] + delta)

    def _end_view_rotate(self):
        state = self._view_rotate
        self._view_rotate = None
        if state is not None and state["moved"]:
            degrees = math.degrees(state["view"].view_angle) % 360.0
            self.app.set_result(f"Rotated the view to {degrees:.0f}\u00b0.")

    def _begin_pose_pan(self, event):
        """Start panning the pose scene's 3D camera from a middle-drag."""
        layer = self._pose_scene_layer()
        if layer is None:
            return
        view = event.widget
        scene = self._pose_scene(layer)
        self._pose_pan = {
            "view": view,
            "layer": layer,
            "scene": scene,
            "start": view.event_to_doc(event),
            "anchor": (float(scene["x"]), float(scene["y"])),
            "pushed": False,
        }
        self.cursor_x = None
        self.cursor_y = None
        view.set_cursor_overlay(None)

    def _pose_pan_to(self, event):
        state = self._pose_pan
        if state is None:
            return
        x, y = state["view"].event_to_doc(event)
        sx, sy = state["start"]
        dx = x - sx
        dy = y - sy
        if not state["pushed"] and (abs(dx) > 2 or abs(dy) > 2):
            self.app._push_undo("Pan scene")
            state["pushed"] = True
        scene = state["scene"]
        scene["x"] = state["anchor"][0] + dx
        scene["y"] = state["anchor"][1] + dy
        layer = state["layer"]
        poses.sync_scene(layer.poses, scene)
        layer.invalidate_poses()
        self._update_poses_overlay()
        self.app.render()

    def _end_pose_pan(self):
        state = self._pose_pan
        self._pose_pan = None
        if state is not None and state["pushed"]:
            self.app.set_result("Panned the scene.")

    def _begin_pan(self, event):
        view = event.widget
        self._pan_view = view
        self._pan_origin = (event.x, event.y)
        self.cursor_x = None
        self.cursor_y = None
        view.set_cursor_overlay(None)

    def _pan_to(self, event):
        if self._pan_view is None:
            return
        view = self._pan_view
        ox, oy = self._pan_origin
        dx = event.x - ox
        dy = event.y - oy
        if dx:
            view.xview_scroll(-dx, "units")
        if dy:
            view.yview_scroll(-dy, "units")
        self._pan_origin = (event.x, event.y)

    def _end_pan(self):
        self._pan_view = None
        self._pan_origin = None
