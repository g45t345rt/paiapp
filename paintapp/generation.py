"""The image generation panel."""

import os
import random
import re
import tkinter as tk
from tkinter import font as tkfont, ttk

from engine import paths as engine_paths

from . import dialogs, theme
from .aspect import aspect_ratio_label
from .constants import MAX_CANVAS_SIZE
from .scrollbar import AutoScrollbar
from .sizeinput import SizeInput

# The Preload hint's idle text; replaced by the load status while one runs.
PRELOAD_IDLE = "load the model now"


def _lora_name(path):
    """The file stem of a LoRA path, or ``""`` when it is unset."""
    return os.path.splitext(os.path.basename(path))[0] if path else ""


def _escape_prompt_segment(text):
    """Escape parentheses so a weighted row's text stays literal in a prompt."""
    return text.replace("(", "\\(").replace(")", "\\)")


# A classic prompt segment already written as ``(text:weight)``; moving to the
# weighted mode pulls that weight onto the row's slider instead of leaving the
# whole thing as literal text.
_WEIGHTED_SEGMENT = re.compile(r"^\((.*):([0-9]*\.?[0-9]+)\)$", re.S)


# The Generate panel's two orthogonal init choices. ``Source`` is what a run
# edits from (``Off`` is plain text-to-image); ``Region`` narrows which pixels
# may change and only applies once a source is chosen.
SOURCE_OFF = "Off"
SOURCE_CANVAS = "Canvas"
SOURCE_LAYER = "Selected Layer"
SOURCE_ASSET = "Selected Asset"
SOURCE_CHOICES = (SOURCE_OFF, SOURCE_CANVAS, SOURCE_LAYER, SOURCE_ASSET)

# How the positive and negative prompts are written. ``Default`` is the
# free-text box (the ``(text:weight)`` syntax still works there); ``Keywords``
# is the fast mode, where every segment has its own box and weight slider.
PROMPT_MODE_DEFAULT = "Default"
PROMPT_MODE_KEYWORDS = "Keywords"
PROMPT_MODE_CHOICES = (PROMPT_MODE_DEFAULT, PROMPT_MODE_KEYWORDS)
# Names stored by earlier builds, kept so existing projects still open in the
# right mode.
PROMPT_MODE_ALIASES = {
    "Classic": PROMPT_MODE_DEFAULT,
    "Weighted": PROMPT_MODE_KEYWORDS,
}


def _prompt_mode_value(value):
    """Map a stored prompt-mode name onto the current choices."""
    if value in PROMPT_MODE_CHOICES:
        return value
    return PROMPT_MODE_ALIASES.get(value, PROMPT_MODE_DEFAULT)
# How long a slider must rest after release before the fast mode regenerates,
# so a quick keyboard nudge or a double tap only starts one run.
FAST_REGENERATE_DELAY = 250
WEIGHT_MIN = 0.0
WEIGHT_MAX = 10.0
WEIGHT_STEP = 0.05
# The weighted row list grows with its rows but stops at this height and then
# scrolls, so a long list does not push the rest of the Generate panel away.
WEIGHTED_LIST_MAX_HEIGHT = 200
# The negative prompt auto-grows up to this many lines, then scrolls.
NEGATIVE_PROMPT_MAX_LINES = 8
# The weighted negative list is usually shorter; cap it lower than the positive
# list so both fit in the sidebar together.
WEIGHTED_NEGATIVE_MAX_HEIGHT = 140

REGION_WHOLE = "Whole image"
REGION_SELECTION = "Selection"
REGION_MASK = "Painted mask"
REGION_CHOICES = (REGION_WHOLE, REGION_SELECTION, REGION_MASK)

# Where a finished run is put: kept as a reference image beside the layer stack,
# written straight into the active layer, or inserted as a new layer above it.
OUTPUT_ASSETS = "In assets"
OUTPUT_LAYER = "In layer"
OUTPUT_NEW_LAYER = "New layer"
OUTPUT_CHOICES = (OUTPUT_ASSETS, OUTPUT_LAYER, OUTPUT_NEW_LAYER)

# The hires-fix second pass: enlarge the finished latent and re-denoise it a
# little at the larger size. ``Off`` disables it; the labels map onto
# ``engine.paths.UPSCALE_TYPES``.
UPSCALE_OFF = "Off"
UPSCALE_LABELS = {
    "none": UPSCALE_OFF,
    "latent_nearest": "Latent nearest",
    "latent_bilinear": "Latent bilinear",
    "latent_bicubic": "Latent bicubic",
    "latent_bislerp": "Latent bislerp",
    "pixel": "Pixel (VAE)",
}
UPSCALE_LABEL_TO_TYPE = {label: kind for kind, label in UPSCALE_LABELS.items()}

# The live denoising preview's decode modes. ``Latent2RGB`` is a fixed linear
# latent->RGB map (no VAE, always available); ``TAESD`` uses a tiny distilled
# decoder where one exists and falls back to Latent2RGB otherwise.
LIVE_LATENT2RGB = "Latent2RGB"
LIVE_TAESD = "TAESD"
LIVE_MODE_LABELS = {LIVE_LATENT2RGB: "latent2rgb", LIVE_TAESD: "taesd"}
LIVE_LABEL_TO_TYPE = {label: mode for label, mode in LIVE_MODE_LABELS.items()}
LIVE_TYPE_TO_LABEL = {mode: label for label, mode in LIVE_MODE_LABELS.items()}

# Per-schedule shape parameters (see ``engine.diffusion_samplers``). Each entry
# is the scheduler picker value, a caption, and its fields as
# ``(config key, label, default, min, max, step)``. Only the chosen schedule's
# fields are shown; the rest collapse, so a schedule with no parameters adds no
# height. ``beta``/``karras``/``polyexponential``/``laplace``/``linear_quadratic``
# are the schedules that read anything here.
SCHEDULER_PARAMETERS = (
    ("beta", "Beta \u03b1/\u03b2:", (
        ("beta_alpha", "\u03b1", 0.5, 0.01, 10.0, 0.05),
        ("beta_beta", "\u03b2", 0.7, 0.01, 10.0, 0.05),
    )),
    ("karras", "Karras \u03c1:", (
        ("karras_rho", "\u03c1", 7.0, 1.0, 100.0, 0.5),
    )),
    ("polyexponential", "Poly \u03c1:", (
        ("poly_rho", "\u03c1", 1.0, 0.05, 10.0, 0.05),
    )),
    ("laplace", "Laplace \u03bc/\u03b2:", (
        ("laplace_mu", "\u03bc", 0.0, -5.0, 5.0, 0.1),
        ("laplace_beta", "\u03b2", 0.5, 0.01, 5.0, 0.05),
    )),
    ("linear_quadratic", "Lin/quad:", (
        ("lq_threshold", "noise", 0.025, 0.0, 1.0, 0.005),
        ("lq_linear_steps", "steps", 0, 0, 50, 1),
    )),
)
SCHEDULER_PARAM_DEFAULTS = {
    key: default
    for _name, _caption, _fields in SCHEDULER_PARAMETERS
    for (key, _label, default, _lo, _hi, _step) in _fields
}


class ScrollArea(tk.Frame):
    """A vertically scrollable frame that grows only up to ``max_height``.

    The inner frame sits on a canvas sized to its own request height, capped at
    ``max_height``; past that the bar appears and the content scrolls. Used for
    the weighted prompt rows so a long list cannot push the panel's other
    fields off screen.
    """

    def __init__(self, parent, max_height):
        super().__init__(parent)
        self.max_height = int(max_height)
        self._height = 0
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0, height=1)
        theme.register(self.canvas, bg="panel_bg")
        self.scrollbar = AutoScrollbar(self, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.canvas)
        theme.register(self.inner, bg="panel_bg")
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_wheel(self.canvas)
        self._bind_wheel(self.inner)

    def _on_inner_configure(self, _event=None):
        """Size the canvas to the rows, clamped to ``max_height``."""
        # Tk keeps a canvas window frame's old requested height after its last
        # child is destroyed, which would leave an empty-list gap; measure zero
        # explicitly when there are no rows.
        content = self.inner.winfo_reqheight() if self.inner.pack_slaves() else 0
        target = max(1, min(self.max_height, content))
        if target != self._height:
            self._height = target
            self.canvas.configure(height=target)
        # Constrain the canvas window to the measured rows too: the inner frame
        # otherwise keeps its old requested size after the last row is gone and
        # the empty list still occupies a row's height.
        self.canvas.itemconfigure(self._window, height=max(1, content))
        width = max(1, self.canvas.winfo_width())
        self.canvas.configure(scrollregion=(0, 0, width, max(1, content)))
        self._refresh_scrollbar()

    def _refresh_scrollbar(self):
        """Re-evaluate the bar after the content shrinks as well as grows."""
        try:
            first, last = self.canvas.yview()
        except tk.TclError:
            return
        self.scrollbar.set(first, last)

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_wheel(self, widget):
        """Let the wheel scroll the list even over a row's entry or slider."""
        widget.bind("<MouseWheel>", self._on_wheel)
        for child in widget.winfo_children():
            self._bind_wheel(child)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    def see_bottom(self):
        """Re-measure and scroll to the newest row after a short idle pass."""
        self.refresh()
        self.canvas.yview_moveto(1.0)

    def refresh(self):
        """Re-measure after rows are removed, when the inner frame shrinks."""
        self.update_idletasks()
        self._on_inner_configure()


class PromptBox(tk.Frame):
    """A prompt ``tk.Text`` with a scrollbar and a drag-to-grow bottom grip.

    The grip changes the box's line height so long prompts can be seen without
    leaving the panel. Dragging reuses the theme's splitter colors, matching the
    sidebar resize handle. With ``auto_grow`` the box instead follows its text
    up to ``max_lines`` and then scrolls; ``min_lines`` is its resting height.
    """

    MIN_LINES = 2
    MAX_LINES = 30

    def __init__(self, parent, auto_grow=False, min_lines=None, max_lines=None):
        super().__init__(parent)
        self._drag = None
        self._auto_grow = bool(auto_grow)
        self._min_lines = int(min_lines or self.MIN_LINES)
        self._max_lines = int(max_lines or self.MAX_LINES)
        self._fit_after = None

        top = tk.Frame(self)
        top.pack(fill=tk.X)
        self.text = tk.Text(top, width=22, height=3, wrap=tk.WORD, undo=True)
        self.text.prompt_box = self
        scrollbar = AutoScrollbar(top, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        grip = tk.Frame(self, height=6, cursor="sb_v_double_arrow")
        theme.register(grip, bg="splitter")
        grip.pack(fill=tk.X)
        grip.pack_propagate(False)
        ridge = tk.Frame(grip, height=2)
        theme.register(ridge, bg="splitter_line")
        ridge.pack(fill=tk.X, padx=6, pady=2)
        for widget in (grip, ridge):
            widget.bind("<Enter>", lambda _e: self._highlight(True))
            widget.bind("<Leave>", lambda _e: self._highlight(False))
            widget.bind("<Button-1>", self._press)
            widget.bind("<B1-Motion>", self._motion)
            widget.bind("<ButtonRelease-1>", self._release)
        self._grip = grip
        self._ridge = ridge

        if self._auto_grow:
            self.text.bind("<<Modified>>", self._on_modified, add="+")
            self.text.bind("<Configure>", self._on_configure, add="+")
            self.schedule_fit()

    # --------------------------------------------------------------- resizing
    def schedule_fit(self):
        """Fit the height to the text once the pending edits settle."""
        if not self._auto_grow or self._fit_after is not None:
            return
        self._fit_after = self.after_idle(self._fit_height)

    def _fit_height(self):
        self._fit_after = None
        try:
            lines = int(self.text.count("1.0", "end-1c", "displaylines")[0])
        except (tk.TclError, TypeError, IndexError):
            return
        lines = max(self._min_lines, min(self._max_lines, lines or 1))
        if lines != int(self.text.cget("height")):
            self.text.configure(height=lines)

    def _on_modified(self, _event=None):
        self.schedule_fit()

    def _on_configure(self, _event=None):
        self.schedule_fit()

    def _linespace(self):
        """Pixel height of one text line, so a drag converts to whole lines."""
        try:
            font = tkfont.nametofont(self.text.cget("font"))
            return max(1, font.metrics("linespace"))
        except tk.TclError:
            return 16

    def _highlight(self, active):
        if not active and self._drag is not None:
            return
        self._grip.configure(
            bg=theme.color("splitter_hover" if active else "splitter")
        )
        self._ridge.configure(
            bg=theme.color("splitter_hover_line" if active else "splitter_line")
        )

    def _press(self, event):
        self._drag = (event.y_root, int(self.text.cget("height")))
        self._highlight(True)
        return "break"

    def _motion(self, event):
        if self._drag is None:
            return "break"
        start_y, start_lines = self._drag
        step = (event.y_root - start_y) / self._linespace()
        lines = start_lines + int(round(step))
        lines = max(self._min_lines, min(self._max_lines, lines))
        if lines != int(self.text.cget("height")):
            self.text.configure(height=lines)
        return "break"

    def _release(self, _event):
        self._drag = None
        return "break"


class GenerationPanel(tk.Frame):
    """Collects generation settings and hands them back as a config dict.

    The panel is purely a form: it owns the tk variables but leaves threading and
    inference to the coordinator, which subscribes via ``on_generate``. It also
    reports every field through ``on_settings`` as they change (and reloads them
    with :meth:`set_settings`), so the coordinator can remember the panel per
    canvas and per layer; the positive prompt is reported through
    ``on_positive_prompt`` and travels on the layer.
    """

    # The recipe dropdown's placeholder: the current values match no saved recipe.
    RECIPE_CUSTOM = "(custom)"

    def __init__(self, parent, on_generate, get_canvas_size=None, on_cancel=None,
                 on_keep=None, on_positive_prompt=None,
                 on_preload=None, on_edit_inpaint_mask=None, on_settings=None):
        super().__init__(parent)
        self._on_generate = on_generate
        self._get_canvas_size = get_canvas_size
        self._on_cancel = on_cancel if on_cancel is not None else (lambda: None)
        self._on_keep = on_keep if on_keep is not None else (lambda: None)
        self._on_positive_prompt = on_positive_prompt
        self._on_preload = on_preload if on_preload is not None else (lambda: None)
        self._on_edit_inpaint_mask = on_edit_inpaint_mask
        # Called with the full panel values whenever any field changes, so the
        # coordinator can store them on the active canvas and layer.
        self._on_settings = on_settings
        # Tracks whether the app is currently in inpaint-mask painting mode, so
        # the Edit button can toggle it without a round trip.
        self._inpaint_editing = False
        # Set while the source/region selectors are written programmatically, so
        # their traces do not echo the change straight back into the app.
        self._syncing_mode = False
        # Guards the programmatic writes made while a preset is applied so they
        # are not echoed straight back into the model's stored preset.
        self._applying = False
        self._preset_save_after = None
        self._preset_pending = None
        # Set while the size is written for the user (canvas sync or a recipe
        # apply), so those writes do not mark the current recipe as diverged.
        self._syncing_size = False
        # Set while the whole panel is written from a stored per-canvas/layer
        # setting, so those writes are not echoed back into that storage.
        self._loading = False
        # The weighted prompt's rows (one text box and weight slider each) and
        # the pending fast-mode regeneration timer. Positive and negative prompts
        # each have their own row list.
        self._weighted_rows = []
        self._weighted_negative_rows = []
        # The classic text the weighted rows were last compiled from/to, so a
        # switch back and forth keeps whichever side the user last edited.
        self._prompt_snapshot = None
        self._negative_snapshot = None
        self._regenerate_after = None
        # True while the queue holds a generate job for this panel, so a fast
        # regeneration can replace it instead of piling a second run behind it.
        self._has_generation_job = False
        # Why the last get_config() call returned None, for the caller to show.
        self.config_error = ""
        # The recipe names shown for the selected model, split into the ones the
        # user saved (removable) and the family's read-only built-ins.
        self._saved_recipe_names = set()
        self._builtin_recipe_names = set()
        self._build()
        self.bind("<Destroy>", self._on_destroy)

    def _build(self):
        model_row = tk.Frame(self)
        model_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(model_row, text="Model:").pack(side=tk.LEFT)
        self.model_choices, self._model_headers, values = self._model_catalog()
        self.model_var = tk.StringVar(value=self._initial_model() if self.model_choices else "")
        self._last_model = self.model_var.get()
        self.model_menu = ttk.Combobox(
            model_row, textvariable=self.model_var,
            values=values or ["(no model found)"], state="readonly", width=18,
        )
        self.model_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.model_menu.bind("<<ComboboxSelected>>", self._on_model_selected)
        # ttk builds the combobox drop-down lazily and rebuilds its rows on
        # every post, so the family headers are recolored from its <Map> handler
        # (set up here) rather than once.
        self._popdown = None
        self.after_idle(self._recolor_model_rows)

        preload_row = tk.Frame(self)
        preload_row.pack(fill=tk.X, pady=(2, 0))
        self.preload_btn = tk.Button(
            preload_row, text="Preload", command=self._on_preload,
            state=tk.NORMAL if self.model_choices else tk.DISABLED,
        )
        self.preload_btn.pack(side=tk.LEFT)
        self.preload_hint = tk.Label(preload_row, text=PRELOAD_IDLE, anchor="w")
        self.preload_hint.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        theme.register(self.preload_hint, fg="muted")
        # Bound after the hint exists so a model change can reset it.
        self.model_var.trace_add("write", self._on_model_change)

        # The selected model's LoRA stack, read only here (it is edited in Model
        # Settings); one row per LoRA with its assigned strength.
        lora_header = tk.Frame(self)
        lora_header.pack(fill=tk.X, pady=(4, 0))
        tk.Label(lora_header, text="LoRAs:", anchor="w").pack(side=tk.LEFT)
        self.lora_count = tk.Label(lora_header, text="", anchor="e")
        theme.register(self.lora_count, fg="muted")
        self.lora_count.pack(side=tk.RIGHT)
        self.lora_list = tk.Frame(self)
        self.lora_list.pack(fill=tk.X)

        mode_row = tk.Frame(self)
        mode_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(mode_row, text="Prompt mode:").pack(side=tk.LEFT)
        self.prompt_mode_var = tk.StringVar(value=PROMPT_MODE_DEFAULT)
        self.prompt_mode_menu = ttk.Combobox(
            mode_row, textvariable=self.prompt_mode_var,
            values=list(PROMPT_MODE_CHOICES), state="readonly", width=12,
        )
        self.prompt_mode_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.prompt_mode_var.trace_add("write", self._on_prompt_mode_change)
        self.auto_generate_var = tk.BooleanVar(value=False)
        self.auto_generate_check = tk.Checkbutton(
            self, text="Auto-generate on slider release",
            variable=self.auto_generate_var, anchor="w",
        )

        # Both prompt editors share one slot so switching modes does not move
        # the rest of the panel; only the active one is packed.
        self.prompt_area = tk.Frame(self)
        self.prompt_area.pack(fill=tk.X)

        self.classic_prompt_frame = tk.Frame(self.prompt_area)
        self.classic_prompt_frame.pack(fill=tk.X)
        tk.Label(
            self.classic_prompt_frame, text="Positive prompt:", anchor="w"
        ).pack(fill=tk.X, pady=(4, 0))
        self.positive_prompt_text = self._prompt_text(self.classic_prompt_frame)
        self.positive_prompt_text.bind("<<Modified>>", self._on_positive_modified)

        # Fast mode: one row per prompt segment. The compiled prompt is their
        # texts joined by commas with ``(text:weight)`` for weights other than 1.
        self.weighted_prompt_frame = tk.Frame(self.prompt_area)
        weighted_header = tk.Frame(self.weighted_prompt_frame)
        weighted_header.pack(fill=tk.X, pady=(4, 0))
        tk.Label(weighted_header, text="Positive prompt:", anchor="w").pack(side=tk.LEFT)
        weighted_hint = tk.Label(
            weighted_header, text="weight 0 disables a row", anchor="e",
        )
        theme.register(weighted_hint, fg="muted")
        weighted_hint.pack(side=tk.RIGHT)
        self.weighted_list = ScrollArea(
            self.weighted_prompt_frame, WEIGHTED_LIST_MAX_HEIGHT
        )
        self.weighted_list.pack(fill=tk.X)
        self.weighted_rows_frame = self.weighted_list.inner
        self.add_weighted_btn = tk.Button(
            self.weighted_prompt_frame, text="+ Add prompt",
            command=self._add_weighted_row,
        )
        self.add_weighted_btn.pack(fill=tk.X, pady=(2, 0))

        # Negative prompt: the classic box in Classic mode, weighted rows in the
        # fast mode, so both prompts can be tuned with sliders there.
        self.negative_area = tk.Frame(self)
        self.negative_area.pack(fill=tk.X)

        self.classic_negative_frame = tk.Frame(self.negative_area)
        self.classic_negative_frame.pack(fill=tk.X)
        tk.Label(
            self.classic_negative_frame, text="Negative prompt:", anchor="w"
        ).pack(fill=tk.X, pady=(4, 0))
        self.negative_var = tk.StringVar(value="")
        self.negative_text = self._prompt_text(
            self.classic_negative_frame, auto_grow=True, min_lines=3,
            max_lines=NEGATIVE_PROMPT_MAX_LINES,
        )
        self.negative_text.bind("<<Modified>>", self._on_negative_modified)
        self.negative_var.trace_add("write", self._on_negative_var)

        self.weighted_negative_frame = tk.Frame(self.negative_area)
        negative_header = tk.Frame(self.weighted_negative_frame)
        negative_header.pack(fill=tk.X, pady=(4, 0))
        tk.Label(
            negative_header, text="Negative prompt:", anchor="w"
        ).pack(side=tk.LEFT)
        negative_hint = tk.Label(
            negative_header, text="weight 0 disables a row", anchor="e",
        )
        theme.register(negative_hint, fg="muted")
        negative_hint.pack(side=tk.RIGHT)
        self.negative_list = ScrollArea(
            self.weighted_negative_frame, WEIGHTED_NEGATIVE_MAX_HEIGHT
        )
        self.negative_list.pack(fill=tk.X)
        self.weighted_negative_rows_frame = self.negative_list.inner
        self.add_weighted_negative_btn = tk.Button(
            self.weighted_negative_frame, text="+ Add negative prompt",
            command=self._add_weighted_negative_row,
        )
        self.add_weighted_negative_btn.pack(fill=tk.X, pady=(2, 0))

        recipe_row = tk.Frame(self)
        recipe_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(recipe_row, text="Recipe:").pack(side=tk.LEFT)
        self.recipe_var = tk.StringVar(value=self.RECIPE_CUSTOM)
        self.recipe_menu = ttk.Combobox(
            recipe_row, textvariable=self.recipe_var,
            values=[self.RECIPE_CUSTOM], state="readonly", width=12,
        )
        self.recipe_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.recipe_menu.bind("<<ComboboxSelected>>", self._on_recipe_selected)
        self.recipe_save_btn = tk.Button(
            recipe_row, text="Save as\u2026", command=self._save_recipe,
        )
        self.recipe_save_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.recipe_delete_btn = tk.Button(
            recipe_row, text="Delete", command=self._delete_recipe,
        )
        self.recipe_delete_btn.pack(side=tk.LEFT, padx=(2, 0))

        # Info note about what a family's built-in sampler actually uses; set by
        # :meth:`_refresh_h3_fields`. Sits between the recipe picker and the
        # Steps row, and is hidden while empty so most families add no height.
        self.h3_note = tk.Label(self, text="", anchor="w")
        theme.register(self.h3_note, fg="muted")

        opts = tk.Frame(self)
        opts.pack(fill=tk.X, pady=(4, 0))
        opts.columnconfigure(0, weight=1)
        self._opts_row = opts

        def row(index, text):
            bg = "stripe_bg" if index % 2 else "panel_bg"
            line = tk.Frame(opts)
            line.grid(row=index, column=0, sticky="ew", pady=(1, 1))
            line.columnconfigure(0, weight=1)
            theme.register(line, bg=bg)
            caption = tk.Label(line, text=text, anchor="w")
            caption.grid(row=0, column=0, sticky="w", padx=(2, 6), pady=2)
            theme.register(caption, bg=bg, fg="text")
            controls = tk.Frame(line)
            controls.grid(row=0, column=1, sticky="e", padx=(0, 2), pady=2)
            theme.register(controls, bg=bg)
            return controls

        # Steps keeps its spinbox in both prompt modes; Seed, CFG and Denoise
        # swap to sliders in the weighted fast mode. Each pair is one row.
        self._fast_controls = []

        self.steps_var = tk.IntVar(value=10)
        self.steps_spin = tk.Spinbox(
            row(0, "Steps:"), from_=1, to=50, width=4, textvariable=self.steps_var
        )
        self.steps_spin.pack(side=tk.LEFT)

        seed_field = row(1, "Seed:")
        self.seed_var = tk.IntVar(value=0)
        self.seed_spin = tk.Spinbox(
            seed_field, from_=0, to=2**31 - 1, width=9, textvariable=self.seed_var
        )
        self.seed_spin.pack(side=tk.LEFT, padx=(0, 4))
        self.seed_random_btn = tk.Button(
            seed_field, text="Random", command=self._random_seed
        )
        self.seed_random_btn.pack(side=tk.LEFT)
        self.seed_var.trace_add("write", self._on_seed_input_change)

        size_field = row(2, "Size:")
        self.size = SizeInput(
            size_field, 1024, 1024, labels=("Width:", "Height:"),
            minimum=64, maximum=MAX_CANVAS_SIZE, spinbox=True,
        )
        self.size.pack(side=tk.LEFT)
        self.gen_width_var = self.size.width_var
        self.gen_height_var = self.size.height_var
        # A recipe carries the output size, so a hand-edited size diverges from
        # a picked recipe just like an edited sampler field does.
        self.size.width_var.trace_add("write", self._on_size_change)
        self.size.height_var.trace_add("write", self._on_size_change)
        self.use_canvas_btn = tk.Button(
            size_field, text="Canvas", command=self._set_canvas_size
        )
        self.use_canvas_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.size_ratio_var = tk.StringVar(value="")
        self.size_ratio_label = tk.Label(
            size_field, textvariable=self.size_ratio_var, anchor="w",
        )
        self.size_ratio_label.pack(side=tk.LEFT, padx=(8, 0))
        theme.register(self.size_ratio_label, fg="muted")
        self.size.width_var.trace_add("write", self._update_size_ratio)
        self.size.height_var.trace_add("write", self._update_size_ratio)
        self._update_size_ratio()

        self.cfg_var = tk.DoubleVar(value=1.0)
        cfg_field = row(3, "CFG:")
        self.cfg_classic = tk.Frame(cfg_field)
        self.cfg_classic.pack(side=tk.LEFT)
        self.cfg_spin = tk.Spinbox(
            self.cfg_classic, from_=0.0, to=10.0, increment=0.1, width=5,
            textvariable=self.cfg_var,
        )
        self.cfg_spin.pack(side=tk.LEFT)
        self.fast_cfg_scale = self._fast_scale(
            cfg_field, self.cfg_var, 0.0, 10.0, 0.1, digits=3
        )
        self._fast_controls.append((self.cfg_classic, self.fast_cfg_scale))

        self.denoise_var = tk.DoubleVar(value=1.0)
        denoise_field = row(4, "Denoise:")
        self.denoise_classic = tk.Frame(denoise_field)
        self.denoise_classic.pack(side=tk.LEFT)
        self.denoise_spin = tk.Spinbox(
            self.denoise_classic, from_=0.0, to=1.0, increment=0.05, width=5,
            textvariable=self.denoise_var,
        )
        self.denoise_spin.pack(side=tk.LEFT)
        self.fast_denoise_scale = self._fast_scale(
            denoise_field, self.denoise_var, 0.0, 1.0, 0.05, digits=3
        )
        self._fast_controls.append((self.denoise_classic, self.fast_denoise_scale))

        self.sampler_var = tk.StringVar(value="euler")
        self.sampler_combo = ttk.Combobox(
            row(5, "Sampler:"), textvariable=self.sampler_var,
            values=list(engine_paths.SAMPLERS), state="readonly", width=14,
        )
        self.sampler_combo.pack(side=tk.LEFT)

        self.scheduler_var = tk.StringVar(value="simple")
        sched_field = row(6, "Scheduler:")
        self.scheduler_combo = ttk.Combobox(
            sched_field, textvariable=self.scheduler_var,
            values=list(engine_paths.SCHEDULERS), state="readonly", width=14,
        )
        self.scheduler_combo.pack(side=tk.LEFT)

        # Each schedule's shape parameters, shown under the picker. Only the
        # chosen schedule's row is shown; the others collapse, so a schedule with
        # no parameters adds no height.
        self._sched_param_vars = {}
        self._sched_param_rows = {}
        row_index = 8
        for name, caption, fields in SCHEDULER_PARAMETERS:
            field = row(row_index, caption)
            for position, (key, label, default, low, high, step) in enumerate(fields):
                if position:
                    tk.Label(field, text="\u00b7").pack(side=tk.LEFT, padx=3)
                tk.Label(field, text=label).pack(side=tk.LEFT)
                var = tk.DoubleVar(value=default)
                self._sched_param_vars[key] = var
                tk.Spinbox(
                    field, from_=low, to=high, increment=step, width=5,
                    textvariable=var,
                ).pack(side=tk.LEFT, padx=(2, 0))
            self._sched_param_rows[name] = field.master
            row_index += 1
        self.scheduler_var.trace_add("write", self._refresh_sched_params)
        self._refresh_sched_params()

        # Separate the sampler fields from the hires-fix pass below them.
        up_divider = tk.Frame(opts, height=1, bd=0, highlightthickness=0)
        up_divider.grid(row=13, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        theme.register(up_divider, bg="border")

        up_type_field = row(14, "Upscale:")
        self.upscale_var = tk.StringVar(value=UPSCALE_OFF)
        ttk.Combobox(
            up_type_field, textvariable=self.upscale_var,
            values=list(UPSCALE_LABELS.values()), state="readonly", width=14,
        ).pack(side=tk.LEFT)

        self.upscale_by_var = tk.DoubleVar(value=2.0)
        tk.Spinbox(
            row(15, "Hires scale:"), from_=1.0, to=4.0, increment=0.05, width=5,
            textvariable=self.upscale_by_var,
        ).pack(side=tk.LEFT)

        self.upscale_denoise_var = tk.DoubleVar(value=0.5)
        tk.Spinbox(
            row(16, "Hires denoise:"), from_=0.0, to=1.0, increment=0.05, width=5,
            textvariable=self.upscale_denoise_var,
        ).pack(side=tk.LEFT)

        divider = tk.Frame(self, height=1, bd=0, highlightthickness=0)
        divider.pack(fill=tk.X, pady=(6, 0))
        theme.register(divider, bg="border")

        source_row = tk.Frame(self)
        source_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(source_row, text="Source:").pack(side=tk.LEFT)
        self.source_var = tk.StringVar(value=SOURCE_OFF)
        self.source_menu = ttk.Combobox(
            source_row, textvariable=self.source_var, values=list(SOURCE_CHOICES),
            state="readonly", width=16,
        )
        self.source_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        region_row = tk.Frame(self)
        region_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(region_row, text="Region:").pack(side=tk.LEFT)
        self.region_var = tk.StringVar(value=REGION_WHOLE)
        self.region_menu = ttk.Combobox(
            region_row, textvariable=self.region_var, values=list(REGION_CHOICES),
            state="disabled", width=14,
        )
        self.region_menu.pack(side=tk.LEFT, padx=(4, 0))
        self.edit_mask_btn = tk.Button(
            region_row, text="Edit mask\u2026", command=self._on_edit_mask_clicked,
            state=tk.DISABLED,
        )
        self.edit_mask_btn.pack(side=tk.LEFT, padx=(6, 0))

        feather_row = tk.Frame(self)
        feather_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(feather_row, text="Feather:").pack(side=tk.LEFT)
        self.feather_var = tk.DoubleVar(value=0.0)
        self.feather_spin = tk.Spinbox(
            feather_row, from_=0.0, to=128.0, increment=1.0, width=5,
            textvariable=self.feather_var, state="disabled",
        )
        self.feather_spin.pack(side=tk.LEFT, padx=(4, 0))

        # A recipe carries the feather, so a hand edit diverges from a picked
        # recipe just like an edited sampler field or size does.
        self.feather_var.trace_add("write", self._on_size_change)

        # Inpainting a small region renders it cropped and upscaled so the mask
        # lands on enough latent cells to carry detail (then stitches it back).
        # Off falls back to editing the whole frame.
        crop_row = tk.Frame(self)
        crop_row.pack(fill=tk.X, pady=(2, 0))
        self.crop_region_var = tk.BooleanVar(value=False)
        self.crop_region_check = tk.Checkbutton(
            crop_row, text="Crop and stitch", variable=self.crop_region_var,
            anchor="w",
        )
        self.crop_region_check.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.source_var.trace_add("write", self._on_source_change)
        self.region_var.trace_add("write", self._on_region_change)
        self._refresh_mode_state()

        divider = tk.Frame(self, height=1, bd=0, highlightthickness=0)
        divider.pack(fill=tk.X, pady=(6, 0))
        theme.register(divider, bg="border")

        # Preview mode: a finished run is shown in the canvas viewport first, so
        # the destination is chosen from the preview's buttons rather than ahead
        # of time. On by default; the Output choice only applies without it.
        preview_row = tk.Frame(self)
        preview_row.pack(fill=tk.X, pady=(2, 0))
        self.preview_var = tk.BooleanVar(value=True)
        self.preview_check = tk.Checkbutton(
            preview_row, text="Preview before applying", variable=self.preview_var,
            anchor="w",
        )
        self.preview_check.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # A trace (rather than the button's own command) keeps the Output choice
        # in step with preview writes made elsewhere, e.g. set_settings/reset.
        self.preview_var.trace_add("write", self._on_preview_change)

        output_row = tk.Frame(self)
        output_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(output_row, text="Output:").pack(side=tk.LEFT)
        self.output_var = tk.StringVar(value=OUTPUT_ASSETS)
        self.output_menu = ttk.Combobox(
            output_row, textvariable=self.output_var, values=list(OUTPUT_CHOICES),
            state="readonly", width=16,
        )
        self.output_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self._refresh_output_state()

        # Live preview: show the denoising steps forming while the run samples.
        # Latent2RGB is the cheap, always-available decoder; TAESD uses a small
        # distilled autoencoder where one exists (falling back to Latent2RGB).
        live_row = tk.Frame(self)
        live_row.pack(fill=tk.X, pady=(2, 0))
        self.live_var = tk.BooleanVar(value=True)
        self.live_check = tk.Checkbutton(
            live_row, text="Live preview", variable=self.live_var,
            anchor="w", command=self._refresh_live_state,
        )
        self.live_check.pack(side=tk.LEFT)
        self.live_mode_var = tk.StringVar(value=LIVE_LATENT2RGB)
        self.live_mode_menu = ttk.Combobox(
            live_row, textvariable=self.live_mode_var, values=list(LIVE_MODE_LABELS),
            state="disabled", width=12,
        )
        self.live_mode_menu.pack(side=tk.RIGHT)
        self._refresh_live_state()

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(6, 8))
        self.generate_btn = tk.Button(buttons, text="Generate", command=self._on_generate)
        self.generate_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.keep_btn = tk.Button(
            buttons, text="Stop & Keep", command=self._on_keep, state=tk.DISABLED
        )
        self.keep_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.cancel_btn = tk.Button(
            buttons, text="Cancel", command=self._on_cancel, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=(4, 0))

        # Sticky sampler settings: watch the fields so edits are remembered per
        # model, and seed the panel from the default model's saved preset.
        self._watch_preset_fields()
        initial = self.model_choices.get(self.model_var.get())
        if initial is not None:
            self._apply_model_preset(initial[0], initial[4], initial[1])
        self._refresh_recipes()
        self._render_loras()
        # Watch every field so the coordinator can remember the panel on the
        # active canvas and layer.
        self._watch_settings_fields()

    def _model_catalog(self):
        """Build the grouped model list for the picker.

        Group names become non-selectable header rows (:meth:`_on_model_selected`
        reverts them); every model row maps its label to ``(family, path)``.
        """
        choices = {}
        headers = set()
        values = []
        for group, entries in engine_paths.list_models():
            header = f"\u2014 {group} \u2014"
            headers.add(header)
            values.append(header)
            for label, family, path, components, loras, key in entries:
                display = label if label not in choices else f"{group}: {label}"
                choices[display] = (family, path, components, loras, key)
                values.append(display)
        return choices, headers, values

    def _model_popdown(self):
        """The model combobox's drop-down path, created on first use.

        ``ttk`` builds the pop-down itself, so it is not a tkinter widget; it is
        addressed by its Tcl path. The pop-down is hidden until the picker is
        first posted and its rows are rebuilt every time it opens, so the
        recoloring is driven by a ``<Map>`` binding set up here.
        """
        exists = self._popdown is not None and self.model_menu.tk.call(
            "winfo", "exists", self._popdown
        )
        if not exists:
            try:
                self._popdown = self.model_menu.tk.call(
                    "ttk::combobox::PopdownWindow", self.model_menu
                )
            except tk.TclError:
                self._popdown = None
                return None
            self.model_menu.tk.call(
                "bind", self._popdown, "<Map>", "+"
                + self.model_menu.register(self._recolor_model_rows)
            )
        return self._popdown

    def _model_listbox(self):
        """The listbox Tcl path inside the pop-down, or ``None``."""
        popdown = self._model_popdown()
        return self._find_listbox(popdown) if popdown else None

    def _find_listbox(self, path):
        tkcmd = self.model_menu.tk
        children = tkcmd.splitlist(tkcmd.call("winfo", "children", path))
        for child in children:
            if tkcmd.call("winfo", "class", child) == "Listbox":
                return child
            found = self._find_listbox(child)
            if found is not None:
                return found
        return None

    def _recolor_model_rows(self, _event=None):
        """Tint the family header rows so they stand apart from the models.

        The dashes stay; a distinct foreground and background make the headers
        read as section labels even though a Tk listbox cannot bold a single
        row.
        """
        listbox = self._model_listbox()
        if listbox is None:
            return
        tkcmd = self.model_menu.tk
        header = (theme.color("accent"), theme.color("stripe_bg"))
        row = (theme.color("text"), theme.color("listbox_bg"))
        count = int(tkcmd.call(listbox, "size"))
        for index, value in enumerate(self.model_menu.cget("values")):
            if index >= count:
                break
            fg, bg = header if value in self._model_headers else row
            tkcmd.call(
                listbox, "itemconfigure", index,
                "-foreground", fg, "-background", bg,
            )

    def _initial_model(self):
        """The default anima model, falling back to the first catalog entry."""
        default_path = engine_paths.DEFAULTS["model"]
        for display, (_family, path, _components, _loras, _key) in self.model_choices.items():
            if path == default_path:
                return display
        return next(iter(self.model_choices), "")

    def refresh_models(self):
        """Rebuild the picker after the model settings change.

        Keeps the selected row by its identity (so two rows over one file stay
        distinct, and an edited label, component or LoRA stack does not drop the
        selection); a row that only just gained an identity is matched by its
        file path instead. Otherwise it falls back to the default model.
        """
        previous = self.model_choices.get(self.model_var.get())
        self.model_choices, self._model_headers, values = self._model_catalog()
        self.model_menu.configure(values=values or ["(no model found)"])
        display = None
        if previous is not None:
            display = next(
                (d for d, choice in self.model_choices.items()
                 if choice[4] == previous[4]),
                None,
            )
            if display is None:
                display = next(
                    (d for d, choice in self.model_choices.items()
                     if choice[1] == previous[1]),
                    None,
                )
        if display is None:
            display = self._initial_model()
        self.model_var.set(display)
        self._last_model = display
        self._render_loras()
        self.preload_btn.config(
            state=tk.NORMAL if self.model_choices else tk.DISABLED
        )

    def _prompt_text(self, parent, auto_grow=False, min_lines=None, max_lines=None):
        """A multiline prompt box that grows by dragging its bottom grip.

        Returns the inner ``tk.Text`` so callers treat it as a plain text box.
        ``auto_grow`` makes the box follow its content up to ``max_lines``.
        """
        box = PromptBox(
            parent, auto_grow=auto_grow, min_lines=min_lines, max_lines=max_lines
        )
        box.pack(fill=tk.X, pady=(2, 0))
        return box.text

    # -------------------------------------------------------- weighted prompt
    def _fast_scale(self, parent, var, from_, to, resolution, digits=None,
                    length=130):
        """A horizontal slider for the Keywords fast mode's rows."""
        scale = tk.Scale(
            parent, from_=from_, to=to, resolution=resolution,
            orient=tk.HORIZONTAL, variable=var, showvalue=True, length=length,
        )
        if digits is not None:
            scale.configure(digits=digits)
        self._bind_fast_release(scale)
        return scale

    def _bind_fast_release(self, scale):
        """Regenerate when the scale is released or a keyboard nudge ends."""
        scale.bind("<ButtonRelease-1>", self._on_fast_slider_release)
        scale.bind("<KeyRelease>", self._on_fast_slider_release)

    def _on_fast_slider_release(self, _event=None):
        self._on_fast_slider()

    def _on_seed_input_change(self, *_):
        """Auto-generate for a hand-edited seed box in Keywords mode."""
        if self._applying or self._loading:
            return
        if self.prompt_mode() != PROMPT_MODE_KEYWORDS:
            return
        if not self.auto_generate_var.get():
            return
        self._schedule_fast_regenerate()

    def prompt_mode(self):
        """``Default`` or ``Keywords`` (the fast keyword list)."""
        return _prompt_mode_value(self.prompt_mode_var.get())

    def _on_prompt_mode_change(self, *_):
        """Swap the prompt editors and carry the text across the two modes."""
        mode = self.prompt_mode()
        if not self._loading:
            if mode == PROMPT_MODE_KEYWORDS:
                self._sync_keywords_from_text(
                    self.positive_prompt(), self._weighted_rows,
                    self.weighted_list, self._add_weighted_row,
                    "_prompt_snapshot",
                )
                self._sync_keywords_from_text(
                    self.negative_var.get(), self._weighted_negative_rows,
                    self.negative_list, self._add_weighted_negative_row,
                    "_negative_snapshot",
                )
            else:
                if self._weighted_rows:
                    text = self.weighted_prompt()
                    self.set_prompt(text)
                    self._prompt_snapshot = text
                if self._weighted_negative_rows or self._negative_snapshot is not None:
                    text = self.weighted_negative_prompt()
                    self.negative_var.set(text)
                    self._negative_snapshot = text
        self._apply_prompt_mode_ui(mode)
        if not self._loading:
            self._report_weighted_prompt()

    def _apply_prompt_mode_ui(self, mode):
        """Show the classic editors or the weighted rows, not both."""
        weighted = mode == PROMPT_MODE_KEYWORDS
        if weighted:
            self.classic_prompt_frame.pack_forget()
            self.weighted_prompt_frame.pack(fill=tk.X)
            self.classic_negative_frame.pack_forget()
            self.weighted_negative_frame.pack(fill=tk.X)
            if not self._weighted_rows:
                self._add_weighted_row()
            self.weighted_list.refresh()
            self.negative_list.refresh()
        else:
            self.weighted_prompt_frame.pack_forget()
            self.classic_prompt_frame.pack(fill=tk.X)
            self.weighted_negative_frame.pack_forget()
            self.classic_negative_frame.pack(fill=tk.X)
        if weighted:
            self.auto_generate_check.pack(fill=tk.X, before=self.prompt_area)
        else:
            self.auto_generate_check.pack_forget()
        for classic, scale in self._fast_controls:
            if weighted:
                classic.pack_forget()
                scale.pack(side=tk.LEFT)
            else:
                scale.pack_forget()
                classic.pack(side=tk.LEFT)

    def _sync_keywords_from_text(self, text, rows, area, add, snapshot):
        """Refresh the row list when the classic text changed between switches.

        Rows are the source of truth while Keywords is active; switching to
        Default writes their compiled text back. If that text was edited in
        Default before switching again, the rows are rebuilt from it so the two
        modes cannot diverge.
        """
        if rows and getattr(self, snapshot) == text:
            return
        self._clear_weighted_rows(rows)
        area.refresh()
        for part in (text or "").split(","):
            part = part.strip()
            if not part:
                continue
            match = _WEIGHTED_SEGMENT.match(part)
            if match:
                add(match.group(1), float(match.group(2)))
            else:
                add(part)
        setattr(self, snapshot, text)

    def _add_weighted_row(self, text="", weight=1.0):
        """Append one positive prompt row."""
        return self._add_weighted_row_to(
            self._weighted_rows, self.weighted_list,
            self._on_weighted_text_change, text, weight,
        )

    def _add_weighted_negative_row(self, text="", weight=1.0):
        """Append one negative prompt row."""
        return self._add_weighted_row_to(
            self._weighted_negative_rows, self.negative_list,
            self._on_weighted_settings_change, text, weight,
        )

    def _add_weighted_row_to(self, rows, area, on_change, text="", weight=1.0):
        """Append one prompt box with its weight slider to ``rows``/``area``."""
        row = tk.Frame(area.inner)
        row.pack(fill=tk.X, pady=(2, 0))
        text_var = tk.StringVar(value=text)
        entry = tk.Entry(row, textvariable=text_var)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        try:
            weight = float(weight)
        except (TypeError, ValueError):
            weight = 1.0
        weight_var = tk.DoubleVar(value=min(WEIGHT_MAX, max(WEIGHT_MIN, weight)))
        scale = tk.Scale(
            row, from_=WEIGHT_MIN, to=WEIGHT_MAX, resolution=WEIGHT_STEP,
            orient=tk.HORIZONTAL, variable=weight_var, showvalue=True,
            length=95, digits=3,
        )
        scale.pack(side=tk.LEFT, padx=(4, 0))
        self._bind_fast_release(scale)
        data = {
            "frame": row, "text_var": text_var, "weight_var": weight_var,
            "entry": entry, "scale": scale, "rows": rows, "area": area,
        }
        remove = tk.Button(
            row, text="\u2715", width=2,
            command=lambda r=data: self._remove_weighted_row(r),
        )
        remove.pack(side=tk.LEFT, padx=(2, 0))
        data["remove_btn"] = remove
        rows.append(data)
        text_var.trace_add("write", lambda *_: on_change())
        entry.bind("<Return>", lambda _e: self._fast_regenerate())
        theme.restyle(row)
        area._bind_wheel(row)
        area.see_bottom()
        return data

    def _remove_weighted_row(self, data):
        """Drop one weighted row; the list may end up empty."""
        rows = data.get("rows", self._weighted_rows)
        area = data.get("area", self.weighted_list)
        if data not in rows:
            return
        rows.remove(data)
        data["frame"].destroy()
        area.refresh()
        self._on_settings_change()
        self._report_weighted_prompt()

    @staticmethod
    def _clear_weighted_rows(rows):
        for data in rows:
            data["frame"].destroy()
        rows.clear()

    def _set_weighted_rows(self, rows):
        """Replace the positive rows with stored ``{text, weight}`` entries."""
        self._clear_weighted_rows(self._weighted_rows)
        for item in rows or []:
            if isinstance(item, dict):
                self._add_weighted_row(
                    str(item.get("text", "")), item.get("weight", 1.0)
                )
        self.weighted_list.refresh()

    def _set_weighted_negative_rows(self, rows):
        """Replace the negative rows with stored ``{text, weight}`` entries."""
        self._clear_weighted_rows(self._weighted_negative_rows)
        for item in rows or []:
            if isinstance(item, dict):
                self._add_weighted_negative_row(
                    str(item.get("text", "")), item.get("weight", 1.0)
                )
        self.negative_list.refresh()

    def _on_weighted_text_change(self, *_):
        if self._loading or self._applying:
            return
        self._on_settings_change()
        self._report_weighted_prompt()

    def _on_weighted_settings_change(self, *_):
        if self._loading or self._applying:
            return
        self._on_settings_change()

    def _report_weighted_prompt(self, *_):
        """Keep the layer's stored prompt in step with the weighted list."""
        if self._loading or self._applying:
            return
        if self.prompt_mode() != PROMPT_MODE_KEYWORDS:
            return
        if self._on_positive_prompt is not None:
            self._on_positive_prompt(self.weighted_prompt())

    @staticmethod
    def _compile_weighted_rows(rows):
        """Compile rows to ``happy, (smile:1.5), ...``, dropping weight 0.

        A row at weight 0 is disabled: it is left out of the compiled prompt
        entirely, so the slider can switch a segment off without deleting it.
        """
        parts = []
        for data in rows:
            text = data["text_var"].get().strip()
            if not text:
                continue
            try:
                weight = float(data["weight_var"].get())
            except (tk.TclError, ValueError):
                weight = 1.0
            if weight <= 0:
                continue
            text = _escape_prompt_segment(text)
            if abs(weight - 1.0) < 1e-9:
                parts.append(text)
            else:
                parts.append(f"({text}:{weight:g})")
        return ", ".join(parts)

    def weighted_prompt(self):
        """The positive weighted rows compiled to prompt syntax."""
        return self._compile_weighted_rows(self._weighted_rows)

    def weighted_negative_prompt(self):
        """The negative weighted rows compiled to prompt syntax."""
        return self._compile_weighted_rows(self._weighted_negative_rows)

    def negative_prompt(self):
        """The run's negative prompt: the compiled fast list or the classic box."""
        if self.prompt_mode() == PROMPT_MODE_KEYWORDS:
            return self.weighted_negative_prompt()
        return self.negative_var.get()

    @staticmethod
    def _weighted_row_values(rows):
        values = []
        for data in rows:
            text = data["text_var"].get().strip()
            if not text:
                continue
            try:
                weight = float(data["weight_var"].get())
            except (tk.TclError, ValueError):
                weight = 1.0
            values.append({"text": text, "weight": weight})
        return values

    def weighted_rows(self):
        """The positive rows as ``[{"text": ..., "weight": ...}, ...]``."""
        return self._weighted_row_values(self._weighted_rows)

    def weighted_negative_rows(self):
        """The negative rows as ``[{"text": ..., "weight": ...}, ...]``."""
        return self._weighted_row_values(self._weighted_negative_rows)

    # ------------------------------------------------------ fast regeneration
    def _on_fast_slider(self, _value=None):
        """A slider moved: regenerate once it comes to rest (i.e. on release)."""
        if self._applying or self._loading:
            return
        if self.prompt_mode() != PROMPT_MODE_KEYWORDS:
            return
        if not self.auto_generate_var.get():
            return
        self._schedule_fast_regenerate()

    def _schedule_fast_regenerate(self):
        if self._regenerate_after is not None:
            try:
                self.after_cancel(self._regenerate_after)
            except tk.TclError:
                pass
        self._regenerate_after = self.after(
            FAST_REGENERATE_DELAY, self._fast_regenerate
        )

    def _fast_regenerate(self):
        """Run the current weighted prompt, replacing an in-flight run."""
        self._regenerate_after = None
        if self._loading or self._applying:
            return
        if self.prompt_mode() != PROMPT_MODE_KEYWORDS:
            return
        if not self.prompt():
            return
        self._on_settings_change()
        if self._has_generation_job:
            self._on_cancel()
        self._on_generate()

    # ------------------------------------------------------------------ public
    def prompt(self):
        """The run's positive prompt: the compiled fast list or the classic box."""
        if self.prompt_mode() == PROMPT_MODE_KEYWORDS:
            return self.weighted_prompt()
        return self.positive_prompt()

    def positive_prompt(self):
        return self.positive_prompt_text.get("1.0", "end-1c").strip()

    def set_prompt(self, text):
        """Load the active layer's positive prompt without echoing a change."""
        self._set_prompt_text(self.positive_prompt_text, text or "")

    def _set_prompt_text(self, widget, text):
        if widget.get("1.0", "end-1c") == text:
            return
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.edit_modified(False)
        # A programmatic rewrite (a canvas or layer switch) is not something the
        # user should be able to Ctrl+Z back into, so drop it from the box's own
        # undo stack; Ctrl+Z then only steps through what was typed here.
        widget.edit_reset()
        box = getattr(widget, "prompt_box", None)
        if box is not None:
            box.schedule_fit()

    def set_size(self, width, height):
        """Set the generation width/height (defaults to the active canvas size)."""
        self._syncing_size = True
        try:
            self.size.set_values(int(width), int(height))
        finally:
            self._syncing_size = False

    def settings(self):
        """The panel's fields as a dict for per-canvas/layer storage.

        Returns ``None`` while a bounded field is mid-edit (an empty size), so a
        half-typed value is not captured as a setting.
        """
        try:
            width, height = self.size.values()
            if width is None or height is None:
                return None
            return {
                "model": self.model_var.get(),
                "prompt_mode": self.prompt_mode_var.get(),
                "weighted_prompt": self.weighted_rows(),
                "weighted_negative": self.weighted_negative_rows(),
                "auto_generate": bool(self.auto_generate_var.get()),
                "negative": self.negative_var.get(),
                "steps": int(self.steps_var.get()),
                "seed": int(self.seed_var.get()),
                "width": int(width),
                "height": int(height),
                "cfg": float(self.cfg_var.get()),
                "denoise": float(self.denoise_var.get()),
                "sampler": self.sampler_var.get(),
                "scheduler": self.scheduler_var.get(),
                **self._sched_param_values(),
                "upscale_type": self.upscale_type(),
                "upscale_by": float(self.upscale_by_var.get()),
                "upscale_denoise": float(self.upscale_denoise_var.get()),
                "feather": self.feather(),
                "crop_region": self.crop_region(),
                "source": self.source_var.get(),
                "region": self.region_var.get(),
                "output": self.output_var.get(),
                "preview": self.preview(),
                "live_preview": self.live_preview_mode(),
            }
        except (tk.TclError, TypeError, ValueError):
            return None

    def set_settings(self, values):
        """Load stored per-canvas/layer settings without echoing a change.

        The model is set first (so its preset is applied) and then every stored
        field overwrites it, so a saved set wins over the model fallback. Fields
        the caller omitted are left as they are.
        """
        if not values:
            return
        self._loading = True
        try:
            model = values.get("model")
            if model and model in self.model_choices:
                self.model_var.set(model)
            self._set_preset_fields(values)
            if "prompt_mode" in values or "weighted_prompt" in values:
                mode = _prompt_mode_value(
                    values.get("prompt_mode", PROMPT_MODE_DEFAULT)
                )
                self.prompt_mode_var.set(mode)
                if "weighted_prompt" in values:
                    self._set_weighted_rows(values.get("weighted_prompt"))
                if "weighted_negative" in values:
                    self._set_weighted_negative_rows(values.get("weighted_negative"))
                if "auto_generate" in values:
                    self.auto_generate_var.set(bool(values["auto_generate"]))
                self._prompt_snapshot = None
                self._negative_snapshot = None
                self._apply_prompt_mode_ui(mode)
            if "seed" in values:
                self.seed_var.set(int(values["seed"]))
            if "negative" in values:
                self.negative_var.set(values["negative"])
            if "source" in values or "region" in values:
                self._apply_mode(
                    values.get("source", self.source_var.get()),
                    values.get("region", self.region_var.get()),
                )
            if "output" in values:
                output = values.get("output")
                self.output_var.set(output if output in OUTPUT_CHOICES else OUTPUT_ASSETS)
            if "preview" in values:
                self.preview_var.set(bool(values["preview"]))
            if "live_preview" in values:
                mode = values["live_preview"]
                self.set_live_preview(mode not in (None, "off"), mode)
            if "upscale_type" in values:
                self.upscale_var.set(
                    UPSCALE_LABELS.get(values["upscale_type"], UPSCALE_OFF)
                )
            if "upscale_by" in values:
                self.upscale_by_var.set(float(values["upscale_by"]))
            if "upscale_denoise" in values:
                self.upscale_denoise_var.set(float(values["upscale_denoise"]))
            self._refresh_output_state()
        finally:
            self._loading = False

    def asset_settings(self):
        """The panel's fields plus the positive prompt, for storing on an asset.

        This is :meth:`settings` with the layer's positive prompt folded in, so
        a finished run can record everything needed to reproduce it. Returns
        ``None`` while a bounded field is mid-edit, matching :meth:`settings`.
        """
        values = self.settings()
        if values is None:
            return None
        values = dict(values)
        values["positive_prompt"] = self.prompt()
        values["negative"] = self.negative_prompt()
        return values

    def apply_asset_settings(self, values):
        """Restore a generated asset's stored fields and prompt, without echoing.

        Loads the sampler/size fields through :meth:`set_settings` and the
        positive prompt through :meth:`set_prompt`; both are guarded against a
        change being reported back, so the active layer keeps its own stored
        settings. Unknown keys are ignored by :meth:`set_settings`.
        """
        if not values:
            return
        self.set_settings(values)
        if "positive_prompt" in values:
            self.set_prompt(values.get("positive_prompt", self.positive_prompt()))

    def reset_to_model_defaults(self):
        """Reset the panel to the selected model's defaults.

        Used when the active canvas and layer have no stored settings: the
        sampler fields take the model preset (or its family default) and the
        remaining fields return to their documented defaults. The size is left
        alone, so the caller can keep it following the canvas.
        """
        choice = self.model_choices.get(self.model_var.get())
        if choice is None:
            return
        self._loading = True
        try:
            self._apply_model_preset(choice[0], choice[4], choice[1])
            self.seed_var.set(0)
            self.prompt_mode_var.set(PROMPT_MODE_DEFAULT)
            self.auto_generate_var.set(False)
            self._set_weighted_rows([])
            self._set_weighted_negative_rows([])
            self._prompt_snapshot = None
            self._negative_snapshot = None
            self._apply_prompt_mode_ui(PROMPT_MODE_DEFAULT)
            self.negative_var.set("")
            self.feather_var.set(0.0)
            self.crop_region_var.set(False)
            self._apply_mode(SOURCE_OFF, REGION_WHOLE)
            self.output_var.set(OUTPUT_ASSETS)
            self.preview_var.set(True)
            self.set_live_preview(True, "latent2rgb")
            self.upscale_var.set(UPSCALE_OFF)
            self.upscale_by_var.set(2.0)
            self.upscale_denoise_var.set(0.5)
            self._refresh_output_state()
        finally:
            self._loading = False

    def _watch_settings_fields(self):
        """Report the whole panel to ``on_settings`` as fields change."""
        for var in (
            self.model_var, self.prompt_mode_var, self.auto_generate_var,
            self.negative_var, self.steps_var, self.seed_var,
            self.cfg_var, self.denoise_var, self.sampler_var, self.scheduler_var,
            *self._sched_param_vars.values(),
            self.size.width_var, self.size.height_var, self.feather_var,
            self.crop_region_var,
            self.source_var, self.region_var, self.output_var, self.preview_var,
            self.live_var, self.live_mode_var,
            self.upscale_var, self.upscale_by_var, self.upscale_denoise_var,
        ):
            var.trace_add("write", self._on_settings_change)

    def _on_settings_change(self, *_):
        """Hand the current panel to ``on_settings`` unless it is echoing."""
        if self._applying or self._loading or self._syncing_size:
            return
        if self._on_settings is None:
            return
        values = self.settings()
        if values is None:
            return
        self._on_settings(values)

    def source(self):
        """What the run edits from: ``Off`` (text-to-image), ``Canvas``,
        ``Selected Layer`` or ``Selected Asset``. An unrecognised value falls
        back to ``Off``."""
        value = self.source_var.get()
        return value if value in SOURCE_CHOICES else SOURCE_OFF

    def region(self):
        """Which pixels may change: ``Whole image``, ``Selection`` or
        ``Painted mask``.

        A region only narrows an edit that has a source, so this reads
        ``Whole image`` while the source is off.
        """
        if self.source() == SOURCE_OFF:
            return REGION_WHOLE
        value = self.region_var.get()
        return value if value in REGION_CHOICES else REGION_WHOLE

    def mode(self):
        """The run's ``(source, region)`` pair, so callers read both at once."""
        return self.source(), self.region()

    def output(self):
        """Where a finished run goes: ``In assets``, ``In layer`` or ``New layer``.

        An unrecognised value falls back to ``In assets``.
        """
        value = self.output_var.get()
        return value if value in OUTPUT_CHOICES else OUTPUT_ASSETS

    def preview(self):
        """True while a finished run is shown for review before being applied."""
        try:
            return bool(self.preview_var.get())
        except tk.TclError:
            return True

    def _on_preview_change(self, *_event):
        """Keep the Output choice in step with the preview toggle."""
        self._refresh_output_state()

    def _refresh_output_state(self):
        """Grey out the Output choice while the preview decides the destination."""
        self.output_menu.configure(
            state="disabled" if self.preview() else "readonly"
        )

    def _refresh_sched_params(self, *_event):
        """Show only the selected schedule's shape-parameter row."""
        rows = getattr(self, "_sched_param_rows", None)
        if not rows:
            return
        active = self.scheduler_var.get()
        for name, line in rows.items():
            if name == active:
                line.grid()
            else:
                line.grid_remove()

    def _sched_param_values(self):
        """The current per-schedule shape parameters as a dict.

        Returns ``{}`` while a value is mid-edit or before the rows exist.
        """
        values = {}
        try:
            for key, var in self._sched_param_vars.items():
                values[key] = float(var.get())
        except (tk.TclError, AttributeError):
            return {}
        return values

    def live_preview_mode(self):
        """The live denoising-preview decode mode: ``off``, ``latent2rgb`` or ``taesd``."""
        try:
            if not self.live_var.get():
                return "off"
        except tk.TclError:
            return "off"
        return LIVE_LABEL_TO_TYPE.get(self.live_mode_var.get(), "latent2rgb")

    def set_live_preview(self, enabled, mode=None):
        """Set the live preview toggle (and mode) without echoing a change."""
        try:
            self.live_var.set(bool(enabled))
        except tk.TclError:
            return
        if mode in LIVE_TYPE_TO_LABEL:
            self.live_mode_var.set(LIVE_TYPE_TO_LABEL[mode])
        self._refresh_live_state()

    def _refresh_live_state(self):
        """Enable the decode-mode picker only while live preview is on."""
        self.live_mode_menu.configure(
            state="readonly" if self.live_var.get() else "disabled"
        )

    def set_inpaint_mask_edit(self, enabled):
        """Mirror the app's inpaint-mask painting state into the panel.

        Turning painting on arms the painted-mask region (and a canvas source,
        since a region needs one to edit); turning it off falls back to the whole
        image. The app calls this both when the panel asks for a change and when
        the paint tool is toggled elsewhere, so an echo of the current state is
        dropped instead of looping.
        """
        enabled = bool(enabled)
        if enabled == self._inpaint_editing:
            return
        self._inpaint_editing = enabled
        if enabled:
            source = self.source()
            self._apply_mode(
                SOURCE_CANVAS if source == SOURCE_OFF else source, REGION_MASK,
            )
        else:
            self._apply_mode(self.source(), REGION_WHOLE)

    def _apply_mode(self, source, region):
        """Set both selectors programmatically without re-entering the app."""
        self._syncing_mode = True
        try:
            if source in SOURCE_CHOICES:
                self.source_var.set(source)
            if region in REGION_CHOICES:
                self.region_var.set(region)
        finally:
            self._syncing_mode = False
        self._refresh_mode_state()

    def _on_source_change(self, *_):
        if self._syncing_mode:
            return
        self._edit_mask_mode(self._edits_mask())
        self._refresh_mode_state()

    def _on_region_change(self, *_):
        if self._syncing_mode:
            return
        self._edit_mask_mode(self._edits_mask())
        self._refresh_mode_state()

    def _edits_mask(self):
        """True when the current choice arms painted-mask editing."""
        return self.source() != SOURCE_OFF and self.region() == REGION_MASK

    def _refresh_mode_state(self):
        """Gate the region controls on a real source.

        The region choice, the Edit button and the feather radius are only
        usable once a source is picked; without one there is nothing for a
        region to narrow or a mask to soften.
        """
        active = self.source() != SOURCE_OFF
        self.region_menu.configure(state="readonly" if active else "disabled")
        self.feather_spin.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.crop_region_check.configure(state=tk.NORMAL if active else tk.DISABLED)
        self.edit_mask_btn.configure(
            state=tk.NORMAL
            if (active and self.region() == REGION_MASK) else tk.DISABLED
        )

    def _on_edit_mask_clicked(self):
        """Toggle inpaint-mask painting without changing the region choice."""
        self._edit_mask_mode(not self._inpaint_editing)

    def _edit_mask_mode(self, enabled):
        self._inpaint_editing = bool(enabled)
        if self._on_edit_inpaint_mask is not None:
            self._on_edit_inpaint_mask(bool(enabled))

    def feather(self):
        """The selection-mask feather radius in document pixels (0 = hard edge)."""
        try:
            return max(0.0, float(self.feather_var.get()))
        except tk.TclError:
            return 0.0

    def crop_region(self):
        """True when a small inpaint region is rendered cropped and upscaled.

        Falls back to off (the documented default) when the widget is gone.
        """
        try:
            return bool(self.crop_region_var.get())
        except tk.TclError:
            return False

    def upscale_type(self):
        """The hires-fix method, or ``none`` (``Off``) when the pass is disabled."""
        return UPSCALE_LABEL_TO_TYPE.get(self.upscale_var.get(), "none")

    # ------------------------------------------------------------- controlnet
    # Control adapters live in their own sidebar panel
    # (:class:`paintapp.controlnet_panel.ControlNetPanel`); the coordinator merges them
    # into the run config. The Generate panel only reports the model family so
    # the panel can offer the right adapter list.

    def model_family(self):
        """The family of the selected model, or ``None`` (e.g. no models)."""
        return self._model_family()

    def _render_loras(self):
        """Rebuild the read-only LoRA list for the selected model.

        Each row shows the LoRA file stem and its assigned strength; a model
        with no LoRAs shows a muted ``(none)``.
        """
        for child in self.lora_list.winfo_children():
            child.destroy()
        choice = self.model_choices.get(self.model_var.get())
        loras = choice[3] if choice else None
        if not loras:
            empty = tk.Label(self.lora_list, text="(none)", anchor="w")
            theme.register(empty, fg="muted")
            empty.pack(fill=tk.X, padx=(2, 0))
            self.lora_count.config(text="")
            theme.restyle(self.lora_list)
            return
        for lora in loras:
            row = tk.Frame(self.lora_list)
            row.pack(fill=tk.X, padx=(2, 0), pady=(1, 0))
            name = tk.Label(row, text=_lora_name(lora.get("path")), anchor="w")
            theme.register(name, fg="text")
            name.pack(side=tk.LEFT, fill=tk.X, expand=True)
            strength = tk.Label(
                row, text=f"{float(lora.get('strength', 1.0)):g}", anchor="e",
            )
            theme.register(strength, fg="muted")
            strength.pack(side=tk.RIGHT)
        self.lora_count.config(text=f"{len(loras)}")
        theme.restyle(self.lora_list)

    @staticmethod
    def _number(var, label, cast):
        """Read a numeric tk variable, naming the field if it is mid-edit.

        Tk stores spinbox text verbatim, so an incomplete entry (for example
        ``"."`` in the denoise box) makes ``get()`` raise ``TclError``. Raising
        a plain ``ValueError`` with the field's label lets :meth:`get_config`
        report which value needs finishing.
        """
        try:
            return cast(var.get())
        except (tk.TclError, ValueError):
            raise ValueError(label) from None

    def get_config(self):
        """The generation run config, or ``None`` while a field is mid-edit.

        When it returns ``None``, :attr:`config_error` says which field still
        needs a complete value.
        """
        family, path, components, loras, _key = self.model_choices.get(
            self.model_var.get(), ("anima", None, {}, [], None)
        )
        components = components or {}
        width, height = self.size.values()
        try:
            config = {
                "family": family,
                "width": int(width),
                "height": int(height),
                "steps": self._number(self.steps_var, "Steps", int),
                "seed": self._number(self.seed_var, "Seed", int),
                "negative": self.negative_prompt(),
                "cfg": self._number(self.cfg_var, "CFG", float),
                "denoise": self._number(self.denoise_var, "Denoise", float),
                "sampler": self.sampler_var.get(),
                "scheduler": self.scheduler_var.get(),
                **self._sched_param_values(),
                "feather": self.feather(),
                "upscale_type": self.upscale_type(),
                "upscale_by": self._number(
                    self.upscale_by_var, "Hires scale", float
                ),
                "upscale_denoise": self._number(
                    self.upscale_denoise_var, "Hires denoise", float
                ),
                "live_preview": self.live_preview_mode(),
            }
        except ValueError as exc:
            self.config_error = f"Finish editing the {exc} value first."
            return None
        self.config_error = ""
        if family == "stable_diffusion":
            config["checkpoint"] = path
        elif family == "z_image":
            config["dit"] = path
            for slot in ("text", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        elif family == "qwen_image":
            config["dit"] = path
            for slot in ("text", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        elif family == "flux":
            config["dit"] = path
            for slot in ("clip", "t5", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        elif family == "krea2":
            config["dit"] = path
            for slot in ("text", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        elif family == "moss_image":
            config["dit"] = path
            for slot in ("text", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        elif family == "h3":
            config["dit"] = path
            for slot in ("text", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        elif family == "h3_image":
            config["dit"] = path
            for slot in ("text", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        else:
            config["model"] = path
            for slot in ("clip", "vae"):
                if components.get(slot):
                    config[slot] = components[slot]
        if loras is not None:
            config["loras"] = [
                {
                    "path": item.get("path"),
                    "strength": float(item.get("strength", 1.0) or 1.0),
                }
                for item in loras
                if isinstance(item, dict) and item.get("path")
            ]
        return config

    def set_preload_status(self, text, token="muted"):
        """Set the Preload hint text (``None`` restores the idle label)."""
        self.preload_hint.config(text=text or PRELOAD_IDLE)
        theme.register(self.preload_hint, fg=token)

    def set_busy(self, busy, cancellable=False, generating=None):
        """Reflect the job queue state for this panel.

        Generate stays enabled while a job runs: pressing it again queues
        another job behind the current one. Cancel lights up only while this
        panel has a queued or running job to stop, and Stop & Keep only while a
        generation is running (it stops the sampler and keeps the image so far).
        ``generating`` distinguishes a real generate job from a preload, so a
        fast-mode slider release only replaces a run, never cancels a load.
        """
        state = tk.NORMAL if (busy and cancellable) else tk.DISABLED
        self.cancel_btn.config(state=state)
        self.keep_btn.config(state=state)
        # Remember whether a generate job is live so a fast-mode slider release
        # can replace it rather than queue a second run behind it.
        self._has_generation_job = bool(busy if generating is None else generating)

    # ------------------------------------------------------------------ events
    def _random_seed(self):
        self.seed_var.set(random.randint(0, 2**31 - 1))

    def _on_positive_modified(self, _event=None):
        widget = self.positive_prompt_text
        if not widget.edit_modified():
            return
        widget.edit_modified(False)
        if self._on_positive_prompt is not None:
            self._on_positive_prompt(widget.get("1.0", "end-1c").strip())

    def _on_negative_modified(self, _event=None):
        """Mirror the negative text box into ``negative_var`` for persistence."""
        widget = self.negative_text
        if not widget.edit_modified():
            return
        widget.edit_modified(False)
        self.negative_var.set(widget.get("1.0", "end-1c"))

    def _on_negative_var(self, *_):
        """Push an externally set ``negative_var`` (e.g. project load) into the box."""
        widget = self.negative_text
        text = self.negative_var.get()
        if widget.get("1.0", "end-1c") == text:
            return
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.edit_modified(False)
        box = getattr(widget, "prompt_box", None)
        if box is not None:
            box.schedule_fit()

    def _set_canvas_size(self):
        if self._get_canvas_size is None:
            return
        width, height = self._get_canvas_size()
        self.size.set_values(int(width), int(height))

    def _update_size_ratio(self, *_args):
        """Keep the compact aspect-ratio label in step with the size fields."""
        width, height = self.size.values()
        if width is None or height is None or width < 1 or height < 1:
            self.size_ratio_var.set("—")
            return
        self.size_ratio_var.set(aspect_ratio_label(width, height))

    def _on_size_change(self, *_):
        """Drop a selected recipe name once the size no longer matches it.

        Size writes for the user (the canvas sync and a recipe apply) set
        ``_syncing_size``/``_applying``, so only a hand edit lands here.
        """
        if self._applying or self._syncing_size:
            return
        if self.recipe_var.get() != self.RECIPE_CUSTOM:
            self.recipe_var.set(self.RECIPE_CUSTOM)
            self._update_recipe_buttons()

    def _on_model_selected(self, _event=None):
        """Keep group headers non-selectable by snapping back to the last model."""
        if self.model_var.get() in self._model_headers:
            self.model_var.set(self._last_model)

    def _on_model_change(self, *_):
        choice = self.model_choices.get(self.model_var.get())
        if choice is None:  # a group header; _on_model_selected reverts it
            return
        self._last_model = self.model_var.get()
        self.set_preload_status(PRELOAD_IDLE, "muted")
        self._apply_model_preset(choice[0], choice[4], choice[1])
        self._refresh_recipes()
        self._render_loras()

    def _apply_model_preset(self, family, key, path):
        """Load a model's sampler settings, falling back to its family defaults.

        A model the user has tuned keeps its own steps/cfg/sampler/scheduler/
        denoise; one with no stored preset takes the family default, matching
        the historical behavior. The ``_applying`` guard keeps these writes out
        of the change-to-preset echo (see :meth:`_on_preset_change`).
        """
        defaults = engine_paths.family_defaults(family, path)
        stored = engine_paths.model_preset(key, path)
        preset = {
            "steps": stored.get("steps", defaults["steps"]),
            "cfg": stored.get("cfg", defaults["cfg"]),
            "sampler": stored.get("sampler", defaults["sampler"]),
            "scheduler": stored.get("scheduler", defaults["scheduler"]),
            "denoise": stored.get("denoise", 1.0),
        }
        for field, default in SCHEDULER_PARAM_DEFAULTS.items():
            preset[field] = stored.get(field, defaults.get(field, default))
        self._set_preset_fields(preset)
        self._refresh_h3_fields(family)

    def _refresh_h3_fields(self, family):
        """Note the sampler settings a family's built-in sampler runs at.

        MiniMax H3 always runs its own dual-schedule rectified-flow Euler at
        guidance 1 (see :func:`engine.pipeline._generate_h3`), so CFG, Sampler
        and Scheduler are inert; MiniMax H3 Image runs a 6-step PDD grid at
        guidance 1 (see :func:`engine.pipeline._generate_h3_image`), so Steps is
        inert too. Fibo 2.0 runs diffusers' own pipeline sampler, so its Sampler
        and Scheduler are inert. The fields stay editable anyway -- a LoRA or a
        future loader may still consume them -- so the note only records what
        the pipeline currently uses.
        """
        note = {
            "h3": "fixed guidance 1 / Euler",
            "h3_image": "fixed guidance 1 / 6-step PDD grid",
            "fibo": "uses the pipeline's own sampler",
        }.get(family, "")
        self.h3_note.configure(text=note)
        self.h3_note.pack_forget()
        if note:
            self.h3_note.pack(fill=tk.X, pady=(2, 0), before=self._opts_row)

    def _set_preset_fields(self, values):
        """Set the sampler fields from ``values`` without echoing into storage."""
        self._applying = True
        try:
            if "steps" in values:
                self.steps_var.set(values["steps"])
            if "cfg" in values:
                self.cfg_var.set(values["cfg"])
            if "sampler" in values:
                self.sampler_var.set(values["sampler"])
            if "scheduler" in values:
                self.scheduler_var.set(values["scheduler"])
            if "denoise" in values:
                self.denoise_var.set(values["denoise"])
            for key, var in self._sched_param_vars.items():
                if key in values:
                    var.set(float(values[key]))
            if "feather" in values:
                self.feather_var.set(values["feather"])
            if "crop_region" in values:
                self.crop_region_var.set(bool(values["crop_region"]))
            if "upscale_type" in values:
                self.upscale_var.set(
                    UPSCALE_LABELS.get(values["upscale_type"], UPSCALE_OFF)
                )
            if "upscale_by" in values:
                self.upscale_by_var.set(float(values["upscale_by"]))
            if "upscale_denoise" in values:
                self.upscale_denoise_var.set(float(values["upscale_denoise"]))
            if "width" in values or "height" in values:
                width, height = self.size.values()
                self.size.set_values(
                    int(values.get("width", width)),
                    int(values.get("height", height)),
                )
        finally:
            self._applying = False

    # --------------------------------------------------------------- recipes
    def _model_path(self):
        """The file path of the selected model, or ``None`` (e.g. no models)."""
        choice = self.model_choices.get(self.model_var.get())
        return choice[1] if choice else None

    def _model_key(self):
        """The preset/recipe key of the selected model (its identity or path)."""
        choice = self.model_choices.get(self.model_var.get())
        return choice[4] if choice else None

    def _model_family(self):
        """The family of the selected model, or ``None`` (e.g. no models)."""
        choice = self.model_choices.get(self.model_var.get())
        return choice[0] if choice else None

    def _refresh_recipes(self):
        """Rebuild the recipe dropdown for the selected model.

        The family's built-in recipes are listed first, then the model's own
        saved ones; a saved recipe of the same name shadows the built-in.
        """
        key = self._model_key()
        path = self._model_path()
        saved = engine_paths.model_recipes(key, path) if key else {}
        family = self._model_family()
        builtin = engine_paths.family_recipes(family, path) if family else {}
        self._saved_recipe_names = set(saved)
        self._builtin_recipe_names = set(builtin) - set(saved)
        names = sorted(self._builtin_recipe_names) + sorted(
            self._saved_recipe_names - self._builtin_recipe_names
        )
        self.recipe_var.set(self.RECIPE_CUSTOM)
        self.recipe_menu.configure(values=[self.RECIPE_CUSTOM] + names)
        self._update_recipe_buttons()

    def _on_recipe_selected(self, _event=None):
        """Apply the chosen recipe and make it the model's current preset."""
        name = self.recipe_var.get()
        if name == self.RECIPE_CUSTOM:
            self._update_recipe_buttons()
            return
        key = self._model_key()
        path = self._model_path()
        recipe = engine_paths.model_recipes(key, path).get(name)
        if recipe is None:
            recipe = engine_paths.family_recipes(self._model_family(), path).get(name)
        if recipe is None:
            return
        self._set_preset_fields(recipe)
        self._store_current_preset()
        self._update_recipe_buttons()

    def _save_recipe(self):
        """Prompt for a name and store the current fields as a model recipe."""
        key = self._model_key()
        values = self._read_recipe_values()
        if not key or values is None:
            return
        current = self.recipe_var.get()
        initial = "" if current == self.RECIPE_CUSTOM else current
        name = dialogs.ask_string(
            self.winfo_toplevel(), "Save Recipe", "Recipe name:",
            initialvalue=initial,
        )
        if name is None:
            return
        name = name.strip()
        if not name or name == self.RECIPE_CUSTOM:
            return
        engine_paths.save_model_recipe(key, name, values)
        engine_paths.save_model_preset(key, values)
        self._refresh_recipes()
        self.recipe_var.set(name)
        self._update_recipe_buttons()

    def _delete_recipe(self):
        """Remove the selected recipe from the model.

        The family's built-in recipes are read-only, so only a saved one is
        removable.
        """
        name = self.recipe_var.get()
        key = self._model_key()
        if not key or name == self.RECIPE_CUSTOM or name not in self._saved_recipe_names:
            return
        engine_paths.delete_model_recipe(key, name)
        self._refresh_recipes()

    def _store_current_preset(self):
        """Persist the current sampler fields as the model's sticky preset."""
        key = self._model_key()
        values = self._read_preset_values()
        if key and values is not None:
            engine_paths.save_model_preset(key, values)

    def _update_recipe_buttons(self):
        """Enable the recipe buttons only when they can act on a model/recipe."""
        has_model = bool(self._model_key())
        name = self.recipe_var.get()
        self.recipe_save_btn.config(state=tk.NORMAL if has_model else tk.DISABLED)
        self.recipe_delete_btn.config(
            state=tk.NORMAL
            if (has_model and name in self._saved_recipe_names)
            else tk.DISABLED
        )

    def _preset_vars(self):
        """The sticky fields mapped to the tk variables that hold them."""
        values = {
            "steps": self.steps_var,
            "cfg": self.cfg_var,
            "denoise": self.denoise_var,
            "sampler": self.sampler_var,
            "scheduler": self.scheduler_var,
        }
        values.update(self._sched_param_vars)
        return values

    def _watch_preset_fields(self):
        """Persist the sampler fields under the current model as they change."""
        for var in self._preset_vars().values():
            var.trace_add("write", self._on_preset_change)

    def _on_preset_change(self, *_):
        """Debounce a per-model preset save after a sampler field is edited.

        The target model and the values are captured now, so a save still lands
        on the model that was being edited even if the selection changes before
        the debounce fires.
        """
        if self._applying:
            return
        key = self._model_key()
        if not key:
            return
        values = self._read_preset_values()
        if values is None:
            # A field mid-edit (empty IntVar/DoubleVar); a later change retries.
            return
        # An edit diverges from any selected recipe, so the dropdown drops back
        # to "(custom)" (the values are still saved as the sticky preset).
        if self.recipe_var.get() != self.RECIPE_CUSTOM:
            self.recipe_var.set(self.RECIPE_CUSTOM)
            self._update_recipe_buttons()
        self._preset_pending = (key, values)
        if self._preset_save_after is not None:
            self.after_cancel(self._preset_save_after)
        self._preset_save_after = self.after(400, self._flush_preset)

    def _read_preset_values(self):
        """The current sampler fields as a dict, or ``None`` if one is mid-edit."""
        values = {}
        try:
            for field, var in self._preset_vars().items():
                values[field] = var.get()
        except tk.TclError:
            return None
        return values

    def _read_recipe_values(self):
        """The sampler fields plus the current size, for saving a recipe.

        Recipes capture the output size as well, so recalling one restores the
        render dimensions; the sticky per-model preset stays sampler-only so the
        generation size can keep following the active canvas.
        """
        values = self._read_preset_values()
        if values is None:
            return None
        width, height = self.size.values()
        if width is None or height is None:
            return None
        values["width"] = int(width)
        values["height"] = int(height)
        values["feather"] = self.feather()
        values["upscale_type"] = self.upscale_type()
        try:
            values["upscale_by"] = float(self.upscale_by_var.get())
            values["upscale_denoise"] = float(self.upscale_denoise_var.get())
        except tk.TclError:
            return None
        return values

    def _flush_preset(self):
        """Write the captured sampler settings to the captured model's preset."""
        self._preset_save_after = None
        pending = self._preset_pending
        self._preset_pending = None
        if pending is None:
            return
        key, values = pending
        engine_paths.save_model_preset(key, values)

    def _on_destroy(self, event):
        """Flush a still-pending preset save when the panel is torn down."""
        if event.widget is not self:
            return
        if self._regenerate_after is not None:
            try:
                self.after_cancel(self._regenerate_after)
            except tk.TclError:
                pass
            self._regenerate_after = None
        if self._preset_save_after is not None:
            try:
                self.after_cancel(self._preset_save_after)
            except tk.TclError:
                pass
            self._preset_save_after = None
        if self._preset_pending is not None:
            key, values = self._preset_pending
            self._preset_pending = None
            engine_paths.save_model_preset(key, values)

