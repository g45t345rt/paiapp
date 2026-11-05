"""The application Settings dialog.

Opened from :menuselection:`Edit --> Settings\u2026` (``Ctrl+,``), this dialog
collects the preferences that were previously only reachable from individual
menus, plus the keyboard shortcuts:

* **Hotkeys** - every global command listed by group. Click a key field and
  press a new combination to rebind it (``Esc`` cancels the capture); a clashing
  combination is refused and the command that already owns it is named. Each row
  has a Reset button and the tab has "Reset All to Defaults".
* **Interface** - theme (light / dark / follow the system), ruler and guide
  visibility (and whether guides are shared by every canvas or kept per
  canvas), the status-bar CPU/RAM/GPU readings, and the off-canvas preview's
  opacity and background color.
* **Canvas** - the size a new document starts at, the default brush size, and a
  button to restore the starter brush presets.
* **Generation** - the AI engine's memory strategy: fp8 precision mode, disk
  streaming (SSD seeker), the RAM budget and prefetch depth, text-encoder
  placement, SageAttention, VAE tiled decode, and the auto-unload timeout. These
  are written to the engine's own settings file and applied without a restart.

Changes are applied and written to ``settings.json`` (``theme``, ``interface``,
``canvas`` and ``shortcuts`` sections) when Save is chosen; Cancel leaves
everything as it was (shortcut edits made while the dialog was open are rolled
back, since those are bound live).
"""

import tkinter as tk
from tkinter import ttk

from engine import tuning as engine_tuning

from . import brushes, dialogs, theme
from .canvasarea import DEFAULT_GRID_SIZE, DEFAULT_OFFCANVAS_BG
from .constants import CANVAS_W, CANVAS_H, MAX_BRUSH_SIZE, MAX_CANVAS_SIZE
from .scrollbar import AutoScrollbar
from .shortcuts import display_all, from_event

# Theme choices shown in the Interface tab.
_THEME_CHOICES = (("Light", "light"), ("Dark", "dark"), ("System", "system"))

# Off-canvas opacity choices, as ``(label, value)``.
_OPACITY_CHOICES = (
    ("25%", 0.25), ("35%", 0.35), ("50%", 0.5), ("75%", 0.75), ("100%", 1.0),
)

# Defaults the Reset button restores.
_DEFAULT_OPACITY = 0.35
_DEFAULT_BRUSH_SIZE = 40

# Generation tab: value -> label maps for the two pickers (the stored value is
# the engine's, the label is what the user reads).
_PRECISION_LABELS = {value: label for label, value in engine_tuning.PRECISION_CHOICES}
_TEXT_ENCODER_LABELS = {
    value: label for label, value in engine_tuning.TEXT_ENCODER_CHOICES
}


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*(int(c) for c in rgb))


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


class AppSettingsDialog:
    """Modal editor for the app's interface, canvas and shortcut settings."""

    def __init__(self, root, app):
        self.root = root
        self.app = app
        self.shortcuts = app.shortcuts
        self._capturing = None
        self._hotkey_rows = {}
        self._shortcut_snapshot = self.shortcuts.snapshot()

        self.window = tk.Toplevel(root)
        self.window.title("Settings")
        self.window.transient(root)
        self.window.withdraw()
        self._build()
        self.window.bind("<Escape>", self._on_escape)
        self.window.protocol("WM_DELETE_WINDOW", self._cancel)
        theme.restyle(self.window)
        dialogs.center_window(root, self.window)
        self.window.grab_set()

    # ----------------------------------------------------------------- layout
    def _build(self):
        body = tk.Frame(self.window)
        body.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(body)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=(10, 0))

        self._build_hotkeys_tab(notebook)
        self._build_interface_tab(notebook)
        self._build_canvas_tab(notebook)
        self._build_generation_tab(notebook)

        buttons = tk.Frame(self.window, padx=12, pady=10)
        buttons.pack(fill=tk.X)
        tk.Button(buttons, text="Save", width=10, command=self._save).pack(side=tk.RIGHT)
        tk.Button(buttons, text="Cancel", width=10, command=self._cancel).pack(
            side=tk.RIGHT, padx=(0, 6)
        )
        tk.Button(buttons, text="Reset", width=10, command=self._reset).pack(side=tk.LEFT)

    def _new_tab(self, notebook, title, width=640, height=440):
        """Add a scrollable page to ``notebook`` and return its inner frame."""
        page = tk.Frame(notebook)
        notebook.add(page, text=title)

        canvas = tk.Canvas(
            page, highlightthickness=0, width=width, height=height,
            bg=theme.color("bg"),
        )
        scroll = AutoScrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = tk.Frame(canvas, padx=12, pady=10)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window_id, width=e.width),
        )
        return canvas, inner

    def _bind_wheel(self, widget, canvas):
        """Scroll ``canvas`` from anywhere over ``widget`` or its descendants."""
        widget.bind("<MouseWheel>", lambda _e, c=canvas: self._on_wheel(_e, c))
        for child in widget.winfo_children():
            self._bind_wheel(child, canvas)

    def _on_wheel(self, event, canvas):
        canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    def _section(self, parent, title, row):
        header = tk.Label(parent, text=title, anchor="w", padx=6, pady=3)
        theme.register(header, bg="header_bg", fg="header_fg")
        header.grid(row=row, column=0, columnspan=3, sticky="we", pady=(10, 0))
        return row + 1

    # -------------------------------------------------------------- hotkeys
    def _build_hotkeys_tab(self, notebook):
        canvas, inner = self._new_tab(notebook, "Hotkeys")
        self._hotkey_canvas = canvas
        self._hotkey_holder = tk.Frame(inner)
        self._hotkey_holder.pack(fill=tk.X)
        self._render_hotkeys()

        tk.Label(
            inner,
            text="Click a key field, then press the new combination. "
                 "Esc cancels the capture.",
            anchor="w", justify="left", wraplength=560, fg=theme.color("muted"),
        ).pack(fill=tk.X, pady=(10, 2))
        self._hotkey_status = tk.StringVar(value="")
        tk.Label(
            inner, textvariable=self._hotkey_status, anchor="w",
            justify="left", wraplength=560, fg=theme.color("error"),
        ).pack(fill=tk.X)
        tk.Button(
            inner, text="Reset All to Defaults", command=self._reset_all_shortcuts,
        ).pack(anchor="w", pady=(8, 0))
        self._bind_wheel(inner, canvas)

    def _render_hotkeys(self):
        for child in self._hotkey_holder.winfo_children():
            child.destroy()
        self._hotkey_rows = {}
        self._hotkey_holder.columnconfigure(0, weight=1)

        row = 0
        for group in self.shortcuts.groups():
            row = self._section(self._hotkey_holder, group, row)
            for entry in self.shortcuts.definitions():
                if entry["group"] != group:
                    continue
                self._hotkey_row(row, entry)
                row += 1
        theme.restyle(self._hotkey_holder)

    def _hotkey_row(self, row, entry):
        tk.Label(
            self._hotkey_holder, text=entry["label"], anchor="w", width=30,
        ).grid(row=row, column=0, sticky="w", pady=2)
        button = tk.Button(
            self._hotkey_holder, text=self._binding_text(entry["id"]), width=20,
            command=lambda sid=entry["id"]: self._start_capture(sid),
        )
        button.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        tk.Button(
            self._hotkey_holder, text="Reset", width=7,
            command=lambda sid=entry["id"]: self._reset_shortcut(sid),
        ).grid(row=row, column=2, sticky="w", padx=(6, 0), pady=2)
        self._hotkey_rows[entry["id"]] = button

    def _binding_text(self, shortcut_id):
        return display_all(self.shortcuts.bindings(shortcut_id))

    def _refresh_rows(self):
        for shortcut_id, button in self._hotkey_rows.items():
            if button.winfo_exists():
                button.configure(text=self._binding_text(shortcut_id))

    def _start_capture(self, shortcut_id):
        if self._capturing == shortcut_id:
            self._stop_capture()
            return
        self._capturing = shortcut_id
        self._hotkey_status.set("")
        for sid, button in self._hotkey_rows.items():
            if sid == shortcut_id:
                button.configure(text="Press keys\u2026")
            else:
                button.configure(text=self._binding_text(sid))
        self.window.bind("<KeyPress>", self._on_capture_key)
        self.window.focus_force()

    def _stop_capture(self):
        if self._capturing is not None:
            self.window.unbind("<KeyPress>")
            self._capturing = None
            self._refresh_rows()

    def _on_capture_key(self, event):
        if self._capturing is None:
            return None
        sequence = from_event(event)
        if sequence is None:
            return "break"
        shortcut_id = self._capturing
        ok, clash_label = self.shortcuts.set_binding(shortcut_id, sequence)
        if not ok:
            self._hotkey_status.set(
                f"{display_all([sequence])} is already used by \u201c{clash_label}\u201d."
            )
            return "break"
        self.shortcuts.apply()
        self._hotkey_status.set("")
        self._stop_capture()
        return "break"

    def _reset_shortcut(self, shortcut_id):
        self.shortcuts.reset(shortcut_id)
        self.shortcuts.apply()
        self._hotkey_status.set("")
        self._stop_capture()

    def _reset_all_shortcuts(self):
        self.shortcuts.reset_all()
        self.shortcuts.apply()
        self._hotkey_status.set("")
        self._stop_capture()

    # ------------------------------------------------------------ interface
    def _build_interface_tab(self, notebook):
        canvas, inner = self._new_tab(notebook, "Interface")
        inner.columnconfigure(0, weight=1)

        row = 0
        row = self._section(inner, "Appearance", row)
        stored_theme = theme.read_settings().get("theme")
        if stored_theme not in ("light", "dark", "system"):
            stored_theme = theme.current()
        self.theme_var = tk.StringVar(value=stored_theme)
        theme_row = tk.Frame(inner)
        theme_row.grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        for label, value in _THEME_CHOICES:
            tk.Radiobutton(
                theme_row, text=label, value=value, variable=self.theme_var,
            ).pack(side=tk.LEFT, padx=(0, 10))
        row += 1

        row = self._section(inner, "Canvas overlays", row)
        self.rulers_var = tk.BooleanVar(value=self.app.show_rulers)
        self.guides_var = tk.BooleanVar(value=self.app.show_guides)
        tk.Checkbutton(
            inner, text="Show rulers", variable=self.rulers_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        tk.Checkbutton(
            inner, text="Show guides", variable=self.guides_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        self.shared_guides_var = tk.BooleanVar(value=self.app.guides_shared)
        tk.Checkbutton(
            inner, text="Same guides on all canvases",
            variable=self.shared_guides_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        self.grid_var = tk.BooleanVar(value=self.app.show_grid)
        tk.Checkbutton(
            inner, text="Show grid", variable=self.grid_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        self.grid_snap_var = tk.BooleanVar(value=self.app.grid_snap)
        tk.Checkbutton(
            inner, text="Snap to grid", variable=self.grid_snap_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        tk.Label(inner, text="Grid size (px)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.grid_size_var = tk.StringVar(value=str(self.app.grid_size))
        tk.Spinbox(
            inner, from_=1, to=MAX_CANVAS_SIZE, width=8,
            textvariable=self.grid_size_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        row = self._section(inner, "Status bar", row)
        self.cpu_var = tk.BooleanVar(value=self.app.stat_visible("cpu"))
        self.ram_var = tk.BooleanVar(value=self.app.stat_visible("ram"))
        self.gpu_var = tk.BooleanVar(value=self.app.stat_visible("gpu"))
        for label, var in (("Show CPU usage", self.cpu_var),
                           ("Show RAM usage", self.ram_var),
                           ("Show GPU usage", self.gpu_var)):
            tk.Checkbutton(inner, text=label, variable=var, anchor="w").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=2,
            )
            row += 1

        row = self._section(inner, "Off-canvas preview", row)
        tk.Label(inner, text="Opacity", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.opacity_labels = {label: value for label, value in _OPACITY_CHOICES}
        current_opacity = self.app.offscreen_opacity
        opacity_label = min(
            self.opacity_labels,
            key=lambda lab: abs(self.opacity_labels[lab] - current_opacity),
        )
        self.opacity_var = tk.StringVar(value=opacity_label)
        ttk.Combobox(
            inner, textvariable=self.opacity_var, state="readonly", width=8,
            values=[label for label, _value in _OPACITY_CHOICES],
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        tk.Label(inner, text="Background color", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.bg_rgb = tuple(int(c) for c in self.app.offscreen_bg)
        self.bg_button = tk.Button(inner, width=10, command=self._choose_bg)
        theme.skip(self.bg_button)
        self._apply_bg_swatch()
        self.bg_button.grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        self._bind_wheel(inner, canvas)

    def _apply_bg_swatch(self):
        color = _rgb_to_hex(self.bg_rgb)
        self.bg_button.configure(bg=color, activebackground=color)

    def _choose_bg(self):
        chosen = dialogs.ask_color(
            self.window, _rgb_to_hex(self.bg_rgb),
            title="Off-Canvas Background Color", persist_recent=False,
        )
        if chosen:
            self.bg_rgb = _hex_to_rgb(chosen)
            self._apply_bg_swatch()

    # --------------------------------------------------------------- canvas
    def _build_canvas_tab(self, notebook):
        canvas, inner = self._new_tab(notebook, "Canvas")

        row = self._section(inner, "New document", 0)
        width, height = self.app.default_canvas_size
        self.width_var = tk.StringVar(value=str(width))
        self.height_var = tk.StringVar(value=str(height))
        tk.Label(inner, text="Default width", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        tk.Spinbox(
            inner, from_=1, to=MAX_CANVAS_SIZE, width=8,
            textvariable=self.width_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1
        tk.Label(inner, text="Default height", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        tk.Spinbox(
            inner, from_=1, to=MAX_CANVAS_SIZE, width=8,
            textvariable=self.height_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        row = self._section(inner, "Tools", row)
        tk.Label(inner, text="Default brush size", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.brush_var = tk.StringVar(value=str(self.app.brush_size))
        tk.Spinbox(
            inner, from_=1, to=MAX_BRUSH_SIZE, width=8,
            textvariable=self.brush_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        row = self._section(inner, "Brush presets", row)
        tk.Button(
            inner, text="Restore Default Brushes", command=self._restore_brushes,
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        self._bind_wheel(inner, canvas)

    # -------------------------------------------------------------- generation
    def _build_generation_tab(self, notebook):
        """The engine's placement/speed knobs (see :mod:`engine.tuning`).

        Every control is optional and defaults to today's automatic behaviour;
        the tab only adds control for users whose hardware wants a different
        trade (force fp8, more/less VRAM headroom, low-RAM disk streaming, tiled
        decode, unloading idle models).
        """
        canvas, inner = self._new_tab(notebook, "Generation")
        inner.columnconfigure(0, weight=1)
        current = engine_tuning.read()

        row = 0
        row = self._section(inner, "Precision & memory", row)
        tk.Label(inner, text="Precision / placement", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.precision_var = tk.StringVar(
            value=_PRECISION_LABELS.get(current["precision"], _PRECISION_LABELS["auto"])
        )
        ttk.Combobox(
            inner, textvariable=self.precision_var, state="readonly", width=34,
            values=[label for label, _v in engine_tuning.PRECISION_CHOICES],
        ).grid(row=row, column=1, columnspan=2, sticky="w", padx=(6, 0), pady=2)
        row += 1

        tk.Label(inner, text="VRAM safety reserve (GB)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.vram_reserve_var = tk.StringVar(value=f"{current['vram_reserve_gb']:.2f}")
        tk.Spinbox(
            inner, from_=0.0, to=32.0, increment=0.5, width=8,
            textvariable=self.vram_reserve_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1
        tk.Label(
            inner, text="Free VRAM kept for the VAE decode and fragmentation.",
            anchor="w", fg=theme.color("muted"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=(6, 0))
        row += 1

        tk.Label(inner, text="Streaming reserve (GB)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.stream_reserve_var = tk.StringVar(value=f"{current['stream_reserve_gb']:.2f}")
        tk.Spinbox(
            inner, from_=0.0, to=48.0, increment=0.5, width=8,
            textvariable=self.stream_reserve_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1
        tk.Label(
            inner, text="Wider headroom when a model streams its blocks.",
            anchor="w", fg=theme.color("muted"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=(6, 0))
        row += 1

        self.keep_resident_var = tk.BooleanVar(value=current["keep_resident"])
        tk.Checkbutton(
            inner, text="Keep as many blocks resident in VRAM as fit",
            variable=self.keep_resident_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        row = self._section(inner, "Speed", row)
        self.compile_vae_var = tk.BooleanVar(value=current["compile_vae"])
        tk.Checkbutton(
            inner, text="Compile the VAE decoder (torch.compile)",
            variable=self.compile_vae_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1

        self.sage_attention_var = tk.BooleanVar(value=current["sage_attention"])
        tk.Checkbutton(
            inner, text="Use SageAttention for the diffusion model (NVIDIA, faster)",
            variable=self.sage_attention_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        tk.Label(
            inner, text="Needs the optional 'sageattention' package and a CUDA GPU.\n"
                        "Without it, attention automatically uses PyTorch SDPA.",
            anchor="w", justify="left", fg=theme.color("muted"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=(6, 0))
        row += 1

        self.tiled_decode_var = tk.BooleanVar(value=current["tiled_decode"])
        tk.Checkbutton(
            inner, text="Decode the VAE in tiles (lower peak, slightly slower)",
            variable=self.tiled_decode_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        tk.Label(inner, text="Tile size (latent px)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.tile_size_var = tk.StringVar(value=str(current["tile_size"]))
        tk.Spinbox(
            inner, from_=128, to=2048, increment=64, width=8,
            textvariable=self.tile_size_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1
        tk.Label(inner, text="Tile overlap (latent px)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.tile_overlap_var = tk.StringVar(value=str(current["tile_overlap"]))
        tk.Spinbox(
            inner, from_=0, to=512, increment=16, width=8,
            textvariable=self.tile_overlap_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        tk.Label(inner, text="Text encoders", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.text_encoder_var = tk.StringVar(
            value=_TEXT_ENCODER_LABELS.get(
                current["text_encoder_device"], _TEXT_ENCODER_LABELS["cpu"]
            )
        )
        ttk.Combobox(
            inner, textvariable=self.text_encoder_var, state="readonly", width=34,
            values=[label for label, _v in engine_tuning.TEXT_ENCODER_CHOICES],
        ).grid(row=row, column=1, columnspan=2, sticky="w", padx=(6, 0), pady=2)
        row += 1

        row = self._section(inner, "Low-RAM streaming", row)
        self.disk_stream_var = tk.BooleanVar(value=current["disk_stream"])
        tk.Checkbutton(
            inner, text="Stream GGUF transformer weights from disk",
            variable=self.disk_stream_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        tk.Label(
            inner, text="Reads each block on demand instead of holding the whole\n"
                        "model in RAM. Slower, but runs far larger models.",
            anchor="w", justify="left", fg=theme.color("muted"),
        ).grid(row=row, column=0, columnspan=3, sticky="w", padx=(6, 0))
        row += 1

        tk.Label(inner, text="RAM budget (GB, 0 = auto)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.ram_budget_var = tk.StringVar(value=f"{current['ram_budget_gb']:.1f}")
        tk.Spinbox(
            inner, from_=0.0, to=512.0, increment=1.0, width=8,
            textvariable=self.ram_budget_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1
        tk.Label(inner, text="Prefetch depth (blocks, 0 = auto)", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.prefetch_var = tk.StringVar(value=str(current["prefetch_depth"]))
        tk.Spinbox(
            inner, from_=0, to=6, increment=1, width=8,
            textvariable=self.prefetch_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        row = self._section(inner, "Model caching", row)
        self.auto_unload_var = tk.BooleanVar(value=current["auto_unload"])
        tk.Checkbutton(
            inner, text="Unload models after a period of inactivity",
            variable=self.auto_unload_var, anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
        row += 1
        tk.Label(inner, text="Idle minutes before unloading", anchor="w").grid(
            row=row, column=0, sticky="w", pady=2,
        )
        self.auto_unload_minutes_var = tk.StringVar(
            value=str(current["auto_unload_minutes"])
        )
        tk.Spinbox(
            inner, from_=1, to=240, increment=1, width=8,
            textvariable=self.auto_unload_minutes_var,
        ).grid(row=row, column=1, sticky="w", padx=(6, 0), pady=2)
        row += 1

        self._bind_wheel(inner, canvas)

    def _generation_values(self):
        """The Generation tab's controls as an :mod:`engine.tuning` value dict."""
        return {
            "precision": next(
                (value for value, label in _PRECISION_LABELS.items()
                 if label == self.precision_var.get()), "auto",
            ),
            "text_encoder_device": next(
                (value for value, label in _TEXT_ENCODER_LABELS.items()
                 if label == self.text_encoder_var.get()), "cpu",
            ),
            "vram_reserve_gb": self._float_value(self.vram_reserve_var, 2.5, 0.0, 32.0),
            "stream_reserve_gb": self._float_value(self.stream_reserve_var, 4.0, 0.0, 48.0),
            "keep_resident": self.keep_resident_var.get(),
            "compile_vae": self.compile_vae_var.get(),
            "sage_attention": self.sage_attention_var.get(),
            "tiled_decode": self.tiled_decode_var.get(),
            "tile_size": self._int_value(self.tile_size_var, 512, 2048),
            "tile_overlap": self._int_value(self.tile_overlap_var, 64, 512),
            "disk_stream": self.disk_stream_var.get(),
            "ram_budget_gb": self._float_value(self.ram_budget_var, 0.0, 0.0, 512.0),
            "prefetch_depth": self._int_value(self.prefetch_var, 0, 6),
            "auto_unload": self.auto_unload_var.get(),
            "auto_unload_minutes": self._int_value(
                self.auto_unload_minutes_var, 10, 240
            ),
        }

    @staticmethod
    def _float_value(var, default, minimum, maximum):
        try:
            value = float(var.get())
        except (tk.TclError, ValueError):
            return default
        return max(minimum, min(value, maximum))

    def _reset_generation(self):
        defaults = engine_tuning.DEFAULTS
        self.precision_var.set(_PRECISION_LABELS.get(defaults["precision"], "auto"))
        self.text_encoder_var.set(
            _TEXT_ENCODER_LABELS.get(defaults["text_encoder_device"], "cpu")
        )
        self.vram_reserve_var.set(f"{defaults['vram_reserve_gb']:.2f}")
        self.stream_reserve_var.set(f"{defaults['stream_reserve_gb']:.2f}")
        self.keep_resident_var.set(defaults["keep_resident"])
        self.compile_vae_var.set(defaults["compile_vae"])
        self.sage_attention_var.set(defaults["sage_attention"])
        self.tiled_decode_var.set(defaults["tiled_decode"])
        self.tile_size_var.set(str(defaults["tile_size"]))
        self.tile_overlap_var.set(str(defaults["tile_overlap"]))
        self.disk_stream_var.set(defaults["disk_stream"])
        self.ram_budget_var.set(f"{defaults['ram_budget_gb']:.1f}")
        self.prefetch_var.set(str(defaults["prefetch_depth"]))
        self.auto_unload_var.set(defaults["auto_unload"])
        self.auto_unload_minutes_var.set(str(defaults["auto_unload_minutes"]))

    # -------------------------------------------------------------- actions
    def _apply(self):
        self.app.set_theme(self.theme_var.get())
        self.app.set_rulers(self.rulers_var.get())
        self.app.set_guides(self.guides_var.get())
        self.app.set_guides_shared(self.shared_guides_var.get())
        self.app.set_grid(self.grid_var.get())
        self.app.set_grid_snap(self.grid_snap_var.get())
        self.app.set_grid_size(
            self._int_value(self.grid_size_var, DEFAULT_GRID_SIZE, MAX_CANVAS_SIZE)
        )
        self.app.set_stat_visible("cpu", self.cpu_var.get())
        self.app.set_stat_visible("ram", self.ram_var.get())
        self.app.set_stat_visible("gpu", self.gpu_var.get())
        self.app.set_offscreen_opacity(
            self.opacity_labels.get(self.opacity_var.get(), _DEFAULT_OPACITY)
        )
        self.app.set_offscreen_bg(self.bg_rgb)
        self.app.default_canvas_size = (
            self._int_value(self.width_var, CANVAS_W, MAX_CANVAS_SIZE),
            self._int_value(self.height_var, CANVAS_H, MAX_CANVAS_SIZE),
        )
        self.app.brush_size = self._int_value(
            self.brush_var, _DEFAULT_BRUSH_SIZE, MAX_BRUSH_SIZE
        )
        # Persist and apply the generation-engine tuning immediately; the engine
        # reads it at the start of every generation too, so a saved change takes
        # effect without a restart.
        engine_tuning.write(self._generation_values())
        engine_tuning.apply()

    @staticmethod
    def _int_value(var, default, maximum):
        try:
            value = int(var.get())
        except (tk.TclError, ValueError):
            return default
        return max(1, min(value, maximum))

    def _reset(self):
        self.theme_var.set("system")
        self.rulers_var.set(True)
        self.guides_var.set(True)
        self.shared_guides_var.set(False)
        self.grid_var.set(False)
        self.grid_snap_var.set(False)
        self.grid_size_var.set(str(DEFAULT_GRID_SIZE))
        self.cpu_var.set(True)
        self.ram_var.set(True)
        self.gpu_var.set(True)
        self.opacity_var.set(next(
            label for label, value in _OPACITY_CHOICES if value == _DEFAULT_OPACITY
        ))
        self.bg_rgb = tuple(DEFAULT_OFFCANVAS_BG)
        self._apply_bg_swatch()
        self.width_var.set(str(CANVAS_W))
        self.height_var.set(str(CANVAS_H))
        self.brush_var.set(str(_DEFAULT_BRUSH_SIZE))
        self._reset_generation()
        self._reset_all_shortcuts()

    def _restore_brushes(self):
        """Re-add the starter brushes (acts immediately, like the shortcuts)."""
        if not dialogs.ask_yes_no(
            self.window, "Restore Default Brushes",
            "Re-add the starter brushes? A preset with the same name is "
            "replaced.",
        ):
            return
        count = brushes.restore_defaults()
        library = getattr(self.app, "brush_library", None)
        if library is not None:
            library.reload()
        self.app.set_result(f"Restored {count} default brushes.")

    def _save(self):
        self._stop_capture()
        self.shortcuts.save()
        self._apply()
        width, height = self.app.default_canvas_size
        interface = {
            "rulers": self.rulers_var.get(),
            "guides": self.guides_var.get(),
            "shared_guides": self.shared_guides_var.get(),
            "grid": self.grid_var.get(),
            "grid_snap": self.grid_snap_var.get(),
            "grid_size": self.app.grid_size,
            "stats": {
                "cpu": self.cpu_var.get(),
                "ram": self.ram_var.get(),
                "gpu": self.gpu_var.get(),
            },
            "offscreen_opacity": self.app.offscreen_opacity,
            "offscreen_bg": list(self.bg_rgb),
        }
        canvas = {
            "width": width,
            "height": height,
            "brush_size": self.app.brush_size,
        }
        settings = theme.read_settings()
        settings["interface"] = interface
        settings["canvas"] = canvas
        theme.write_settings(settings)
        self.window.destroy()

    def _cancel(self):
        self._stop_capture()
        self.shortcuts.restore(self._shortcut_snapshot)
        self.shortcuts.apply()
        self.window.destroy()

    def _on_escape(self, _event=None):
        if self._capturing is not None:
            self._stop_capture()
        else:
            self._cancel()
        return "break"
