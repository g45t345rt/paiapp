"""The color picker: wheel/square, hue and value sliders, entries, palette.

The picker offers two interchangeable ways to choose a hue and saturation: the
round HSV wheel (with a value slider) and the classic saturation/value square
(with a vertical hue strip). Both drive the same ``hue``/``sat``/``value``
state, so switching modes keeps the current color. Exact colors can be typed
into the hex and RGB fields, colors are remembered in a recent row, and the
palette can grow, shrink and be saved to or loaded from a JSON file.
"""

import colorsys
import json
import math
import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from . import theme
from .ui import PopupMenu
from .constants import (
    HUE_STRIP_W,
    PALETTE_COLORS,
    RECENT_COLORS_MAX,
    WHEEL_SIZE,
)

_HEX_DIGITS = set("0123456789abcdefABCDEF")


class ColorPicker(tk.Frame):
    """A self-contained HSV color chooser with a palette and recent colors.

    The chosen color is exposed as ``color`` (a ``#rrggbb`` string); the palette
    as ``palette`` and ``selected_swatch``. No callbacks are required: callers
    read these attributes when they need the current color.
    """

    def __init__(self, parent, on_change=None, persist_recent=True):
        super().__init__(parent)
        self._color = "#000000"
        self.hue = 0.0
        self.sat = 0.0
        self.value = 1.0
        self.on_change = on_change
        self.mode = "wheel"
        self._suppress_value = False
        self._syncing = False
        self._persist_recent = persist_recent
        self._recent_save_job = None
        self._menu_index = 0
        self._wheel_photo = None
        self._square_photo = None
        self._hue_photo = None
        self._wheel_base = None
        self._wheel_mask = None
        self._value_mask = None
        self._black = None
        self.palette = list(PALETTE_COLORS)
        self.selected_swatch = 0
        self.recent_colors = self._load_recent()
        self._build()
        self._update_swatch_borders()
        self._rebuild_recent()
        self._render_hue_strip()
        self._render_wheel()
        self._render_square()
        self._sync_entries()
        theme.subscribe(self._on_theme)

    # ------------------------------------------------------------------ build
    def _build(self):
        self._build_mode_row()
        self._build_pickers()
        self._build_fields()
        self._build_palette()
        self._build_recent()

    def _build_mode_row(self):
        mode_row = tk.Frame(self)
        mode_row.pack(fill=tk.X, pady=(4, 2))
        self.mode_var = tk.StringVar(value=self.mode)
        for label, value in (("Wheel", "wheel"), ("Square", "square")):
            tk.Radiobutton(
                mode_row, text=label, value=value, variable=self.mode_var,
                command=self._on_mode_change,
            ).pack(side=tk.LEFT)

    def _build_pickers(self):
        self.picker_host = tk.Frame(self)
        self.picker_host.pack(pady=(0, 2))

        self.wheel_frame = tk.Frame(self.picker_host)
        self.wheel_canvas = tk.Canvas(
            self.wheel_frame, width=WHEEL_SIZE, height=WHEEL_SIZE,
            highlightthickness=0, cursor="crosshair",
        )
        self.wheel_canvas.pack(side=tk.LEFT)
        self.wheel_canvas.bind("<Button-1>", self._on_wheel)
        self.wheel_canvas.bind("<B1-Motion>", self._on_wheel)
        self.wheel_canvas.bind("<ButtonRelease-1>", self._on_picker_release)

        value_col = tk.Frame(self.wheel_frame)
        value_col.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0))
        self.value_var = tk.IntVar(value=round(self.value * 100))
        self.value_scale = tk.Scale(
            value_col, from_=100, to=0, orient=tk.VERTICAL,
            variable=self.value_var, command=self._on_value_change,
            length=WHEEL_SIZE,
        )
        self.value_scale.pack(side=tk.TOP, fill=tk.Y)
        self.value_scale.bind("<ButtonRelease-1>", self._on_picker_release)
        tk.Label(value_col, text="Value").pack(side=tk.TOP)

        self.square_frame = tk.Frame(self.picker_host)
        self.square_canvas = tk.Canvas(
            self.square_frame, width=WHEEL_SIZE, height=WHEEL_SIZE,
            highlightthickness=0, cursor="crosshair",
        )
        self.square_canvas.pack(side=tk.LEFT)
        self.square_canvas.bind("<Button-1>", self._on_square)
        self.square_canvas.bind("<B1-Motion>", self._on_square)
        self.square_canvas.bind("<ButtonRelease-1>", self._on_picker_release)

        hue_col = tk.Frame(self.square_frame)
        hue_col.pack(side=tk.LEFT, fill=tk.Y, padx=(4, 0))
        self.hue_canvas = tk.Canvas(
            hue_col, width=HUE_STRIP_W, height=WHEEL_SIZE,
            highlightthickness=0, cursor="hand2",
        )
        self.hue_canvas.pack(side=tk.TOP)
        self.hue_canvas.bind("<Button-1>", self._on_hue)
        self.hue_canvas.bind("<B1-Motion>", self._on_hue)
        self.hue_canvas.bind("<ButtonRelease-1>", self._on_picker_release)
        tk.Label(hue_col, text="Hue").pack(side=tk.TOP)

        self.wheel_frame.pack()

    def _build_fields(self):
        hex_row = tk.Frame(self)
        hex_row.pack(fill=tk.X, pady=(4, 2))
        tk.Label(hex_row, text="Hex").pack(side=tk.LEFT)
        self.hex_var = tk.StringVar()
        self.hex_var.trace_add("write", self._on_hex_change)
        self.hex_entry = tk.Entry(hex_row, width=9, textvariable=self.hex_var)
        self.hex_entry.pack(side=tk.LEFT, padx=(4, 0))
        self.hex_entry.bind("<Return>", self._commit_hex)
        self.hex_entry.bind("<FocusOut>", self._commit_hex)

        rgb_row = tk.Frame(self)
        rgb_row.pack(fill=tk.X)
        self.r_var = tk.StringVar()
        self.g_var = tk.StringVar()
        self.b_var = tk.StringVar()
        for label, var in (("R", self.r_var), ("G", self.g_var), ("B", self.b_var)):
            tk.Label(rgb_row, text=label).pack(side=tk.LEFT)
            entry = tk.Entry(rgb_row, width=4, textvariable=var)
            entry.pack(side=tk.LEFT, padx=(2, 6))
            entry.bind("<Return>", self._commit_rgb)
            entry.bind("<FocusOut>", self._commit_rgb)

    def _build_palette(self):
        header = tk.Frame(self)
        header.pack(fill=tk.X, pady=(6, 0))
        tk.Label(header, text="Palette", anchor="w").pack(side=tk.LEFT)
        self.menu_button = tk.Button(
            header, text="...", width=2, takefocus=0,
            command=self._post_palette_menu,
        )
        self.menu_button.pack(side=tk.RIGHT)
        self.palette_menu = PopupMenu(self)
        self.palette_menu.add_command(label="Add swatch", command=self.add_swatch)
        self.palette_menu.add_command(
            label="Remove swatch", command=self.remove_selected_swatch
        )
        self.palette_menu.add_separator()
        self.palette_menu.add_command(
            label="Save palette...", command=self.save_palette
        )
        self.palette_menu.add_command(
            label="Load palette...", command=self.load_palette
        )
        self.palette_menu.add_separator()
        self.palette_menu.add_command(
            label="Reset palette", command=self.reset_palette
        )

        self.swatch_grid = tk.Frame(self)
        self.swatch_grid.pack(anchor="w", pady=(2, 4))

        self.swatch_menu = tk.Menu(self, tearoff=0)
        self.swatch_menu.add_command(
            label="Edit...", command=lambda: self._edit_swatch(self._menu_index)
        )
        self.swatch_menu.add_command(
            label="Assign current color",
            command=lambda: self._assign_index(self._menu_index),
        )
        self.swatch_menu.add_command(
            label="Insert after",
            command=lambda: self.insert_swatch(self._menu_index + 1),
        )
        self.swatch_menu.add_separator()
        self.swatch_menu.add_command(
            label="Remove", command=lambda: self.remove_swatch(self._menu_index)
        )
        self._rebuild_swatches()

    def _post_palette_menu(self):
        self.palette_menu.post(
            self.menu_button.winfo_rootx(),
            self.menu_button.winfo_rooty() + self.menu_button.winfo_height(),
        )

    def _build_recent(self):
        tk.Label(self, text="Recent", anchor="w").pack(fill=tk.X)
        self.recent_grid = tk.Frame(self)
        self.recent_grid.pack(anchor="w", pady=(2, 4))
        self.recent_boxes = []

    # ------------------------------------------------------------------ color
    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value
        self._sync_entries()

    def get_color(self):
        return self._color

    def set_color(self, color):
        self.color = color

    def _notify(self):
        if self.on_change is not None:
            self.on_change(self._color)

    def set_rgb(self, r, g, b):
        """Adopt an RGB triple: update HSV, marker and the selected swatch."""
        hue, sat, value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        self.hue, self.sat, self.value = hue * 360.0, sat, value
        self._sync_value_var()
        self.color = "#%02x%02x%02x" % (r, g, b)
        self._refresh_picker()
        self.assign_to_swatch()
        self._notify()

    def choose_color(self):
        def preview(color):
            self.set_rgb(*self._hex_to_rgb(color))

        from . import dialogs

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), self._normalize_hex(self._color) or "#000000",
            on_preview=preview,
        )
        if chosen is not None:
            self.set_rgb(*self._hex_to_rgb(chosen))
            self._record_recent()

    # ------------------------------------------------------------------ fields
    def _sync_entries(self):
        if not hasattr(self, "hex_var"):
            return
        value = self._normalize_hex(self._color)
        self._syncing = True
        try:
            self.hex_var.set(value or "")
            if value:
                r, g, b = self._hex_to_rgb(value)
                self.r_var.set(str(r))
                self.g_var.set(str(g))
                self.b_var.set(str(b))
        finally:
            self._syncing = False

    def _on_hex_change(self, *_args):
        if self._syncing:
            return
        value = self._normalize_hex(self.hex_var.get())
        if value and value != self._color:
            self.set_rgb(*self._hex_to_rgb(value))
            self._record_recent()

    def _commit_hex(self, _event=None):
        self._sync_entries()

    def _commit_rgb(self, _event=None):
        if self._syncing:
            return
        try:
            r = int(self.r_var.get())
            g = int(self.g_var.get())
            b = int(self.b_var.get())
        except ValueError:
            self._sync_entries()
            return
        if not all(0 <= channel <= 255 for channel in (r, g, b)):
            self._sync_entries()
            return
        if (r, g, b) != self._hex_to_rgb(self._normalize_hex(self._color) or "#000000"):
            self.set_rgb(r, g, b)
            self._record_recent()
        else:
            self._sync_entries()

    @staticmethod
    def _normalize_hex(color):
        """Return a canonical ``#rrggbb`` string, or ``None`` when invalid."""
        if not isinstance(color, str):
            return None
        value = color.strip().lstrip("#")
        if len(value) == 3 and all(c in _HEX_DIGITS for c in value):
            value = "".join(c * 2 for c in value)
        if len(value) == 6 and all(c in _HEX_DIGITS for c in value):
            return "#" + value.lower()
        return None

    @staticmethod
    def _hex_to_rgb(color):
        return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode):
        if mode not in ("wheel", "square") or mode == self.mode:
            return
        self.mode = mode
        self.mode_var.set(mode)
        self._on_mode_change()

    def _on_mode_change(self):
        self.mode = self.mode_var.get()
        if self.mode == "square":
            self.wheel_frame.pack_forget()
            self.square_frame.pack()
            self._render_square()
        else:
            self.square_frame.pack_forget()
            self.wheel_frame.pack()
            self._render_wheel()

    def _refresh_picker(self):
        if self.mode == "square":
            self._render_square()
        else:
            self._render_wheel()

    # ------------------------------------------------------------------ wheel
    def _on_wheel(self, event):
        size = WHEEL_SIZE
        c = (size - 1) / 2.0
        radius = size / 2.0
        dx, dy = event.x - c, event.y - c
        dist = math.hypot(dx, dy)
        if dist > radius:
            return
        self.hue = math.degrees(math.atan2(dy, dx)) % 360.0
        self.sat = min(dist / radius, 1.0)
        self._apply_hsv()

    def _on_square(self, event):
        size = WHEEL_SIZE
        x = min(max(event.x, 0), size - 1)
        y = min(max(event.y, 0), size - 1)
        self.sat = x / (size - 1)
        self.value = 1.0 - y / (size - 1)
        self._sync_value_var()
        self._apply_hsv()

    def _on_hue(self, event):
        size = WHEEL_SIZE
        y = min(max(event.y, 0), size - 1)
        self.hue = y / (size - 1) * 360.0
        self._render_square()
        self._apply_hsv()

    def _apply_hsv(self):
        r, g, b = colorsys.hsv_to_rgb(self.hue / 360.0, self.sat, self.value)
        self.color = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
        self._draw_marker()
        self.assign_to_swatch()
        self._notify()

    def _on_value_change(self, _value=None):
        if self._suppress_value:
            return
        self.value = max(0.0, min(1.0, self.value_var.get() / 100.0))
        self._apply_hsv()
        self._render_wheel()

    def _on_picker_release(self, _event=None):
        self._record_recent()

    def _sync_value_var(self):
        self._suppress_value = True
        try:
            self.value_var.set(round(self.value * 100))
        finally:
            self._suppress_value = False

    # ------------------------------------------------------------- rendering
    def _render_wheel(self):
        """Paint the wheel, scaling a cached full-value bitmap for brightness.

        The hue/saturation disc never changes, so it is rasterised once and
        only darkened with a lookup table on each value change. The area
        outside the disc shows the panel background, so it matches the theme.
        """
        size = WHEEL_SIZE
        if self._wheel_base is None:
            base = Image.new("RGB", (size, size), (0, 0, 0))
            mask = Image.new("L", (size, size), 0)
            base_px, mask_px = base.load(), mask.load()
            c = (size - 1) / 2.0
            radius = size / 2.0
            for y in range(size):
                dy = y - c
                for x in range(size):
                    dx = x - c
                    dist = math.hypot(dx, dy)
                    if dist <= radius + 0.5:
                        hue = math.degrees(math.atan2(dy, dx)) % 360.0
                        sat = min(dist / radius, 1.0)
                        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, 1.0)
                        base_px[x, y] = (int(r * 255), int(g * 255), int(b * 255))
                        coverage = min(1.0, max(0.0, radius - dist + 0.5))
                        mask_px[x, y] = int(coverage * 255)
            self._wheel_base, self._wheel_mask = base, mask
        backdrop = Image.new("RGB", (size, size), theme.color("bg") or "white")
        if self.value >= 0.999:
            disc = self._wheel_base
        else:
            value = self.value
            disc = Image.eval(self._wheel_base, lambda p, v=value: int(p * v))
        img = Image.composite(disc, backdrop, self._wheel_mask)
        self._wheel_photo = ImageTk.PhotoImage(img)
        self.wheel_canvas.delete("all")
        self.wheel_canvas.create_image(0, 0, anchor=tk.NW, image=self._wheel_photo)
        self._draw_wheel_marker()

    def _render_square(self):
        """Paint the saturation/value square for the current hue.

        A one-row white-to-hue gradient is stretched horizontally, then shaded
        top-to-bottom by a cached value mask, so no per-pixel loop is needed.
        """
        size = WHEEL_SIZE
        row = Image.new("RGB", (size, 1))
        px = row.load()
        for x in range(size):
            sat = x / (size - 1)
            r, g, b = colorsys.hsv_to_rgb(self.hue / 360.0, sat, 1.0)
            px[x, 0] = (int(r * 255), int(g * 255), int(b * 255))
        square = row.resize((size, size), Image.NEAREST)
        if self._value_mask is None:
            gradient = Image.linear_gradient("L").resize(
                (size, size), Image.BILINEAR
            )
            self._value_mask = Image.eval(gradient, lambda p: 255 - p)
            self._black = Image.new("RGB", (size, size), (0, 0, 0))
        square = Image.composite(square, self._black, self._value_mask)
        self._square_photo = ImageTk.PhotoImage(square)
        self.square_canvas.delete("all")
        self.square_canvas.create_image(0, 0, anchor=tk.NW, image=self._square_photo)
        self._draw_square_marker()

    def _render_hue_strip(self):
        size = WHEEL_SIZE
        img = Image.new("RGB", (HUE_STRIP_W, size))
        px = img.load()
        for y in range(size):
            r, g, b = colorsys.hsv_to_rgb(y / (size - 1), 1.0, 1.0)
            rgb = (int(r * 255), int(g * 255), int(b * 255))
            for x in range(HUE_STRIP_W):
                px[x, y] = rgb
        self._hue_photo = ImageTk.PhotoImage(img)
        self.hue_canvas.delete("all")
        self.hue_canvas.create_image(0, 0, anchor=tk.NW, image=self._hue_photo)
        self._draw_hue_marker()

    def _draw_marker(self):
        if self.mode == "square":
            self._draw_square_marker()
            self._draw_hue_marker()
        else:
            self._draw_wheel_marker()

    def _draw_wheel_marker(self):
        self.wheel_canvas.delete("marker")
        size = WHEEL_SIZE
        c = (size - 1) / 2.0
        radius = size / 2.0
        angle = math.radians(self.hue)
        x = c + math.cos(angle) * self.sat * radius
        y = c + math.sin(angle) * self.sat * radius
        self.wheel_canvas.create_oval(
            x - 5, y - 5, x + 5, y + 5, outline="#ffffff", width=2, tags="marker"
        )
        self.wheel_canvas.create_oval(
            x - 4, y - 4, x + 4, y + 4, outline="#000000", width=1, tags="marker"
        )

    def _draw_square_marker(self):
        self.square_canvas.delete("marker")
        size = WHEEL_SIZE
        x = self.sat * (size - 1)
        y = (1.0 - self.value) * (size - 1)
        self.square_canvas.create_oval(
            x - 5, y - 5, x + 5, y + 5, outline="#ffffff", width=2, tags="marker"
        )
        self.square_canvas.create_oval(
            x - 4, y - 4, x + 4, y + 4, outline="#000000", width=1, tags="marker"
        )

    def _draw_hue_marker(self):
        self.hue_canvas.delete("marker")
        size = WHEEL_SIZE
        y = self.hue / 360.0 * (size - 1)
        self.hue_canvas.create_rectangle(
            0, y - 2, HUE_STRIP_W, y + 2, outline="#ffffff", width=2, tags="marker"
        )
        self.hue_canvas.create_rectangle(
            0, y - 1, HUE_STRIP_W, y + 1, outline="#000000", width=1, tags="marker"
        )

    # ------------------------------------------------------------------ palette
    def set_palette(self, colors):
        self.palette = list(colors)
        if not self.palette:
            self.palette = list(PALETTE_COLORS)
        self.selected_swatch = max(
            0, min(self.selected_swatch, len(self.palette) - 1)
        )
        self._rebuild_swatches()

    def _rebuild_swatches(self):
        for box in getattr(self, "swatches", []):
            box.destroy()
        self.swatches = []
        for index, swatch_color in enumerate(self.palette):
            box = tk.Frame(
                self.swatch_grid, width=20, height=20, bg=swatch_color,
                highlightthickness=2, highlightbackground=theme.color("border"),
            )
            box.grid(row=index // 8, column=index % 8, padx=1, pady=1)
            box.pack_propagate(False)
            box.grid_propagate(False)
            # The swatch color is content: keep it in every theme.
            theme.skip_bg(box)
            box.bind("<Button-1>", lambda _e, i=index: self.select_swatch(i))
            box.bind("<Double-Button-1>", lambda _e, i=index: self._edit_swatch(i))
            box.bind("<Button-3>", lambda e, i=index: self._show_swatch_menu(e, i))
            self.swatches.append(box)
        self._update_swatch_borders()

    def add_swatch(self, color=None):
        """Append a palette slot holding ``color`` (the current color default)."""
        self.palette.append(color or self._color)
        self.selected_swatch = len(self.palette) - 1
        self._rebuild_swatches()

    def insert_swatch(self, index, color=None):
        """Insert a palette slot at ``index`` and select it."""
        index = max(0, min(index, len(self.palette)))
        self.palette.insert(index, color or self._color)
        self.selected_swatch = index
        self._rebuild_swatches()

    def remove_swatch(self, index):
        """Drop one palette slot, keeping at least one swatch."""
        if len(self.palette) <= 1 or not 0 <= index < len(self.palette):
            return
        del self.palette[index]
        self.selected_swatch = max(0, min(self.selected_swatch, len(self.palette) - 1))
        self._rebuild_swatches()

    def remove_selected_swatch(self):
        """Drop the currently selected palette slot."""
        self.remove_swatch(self.selected_swatch)

    def reset_palette(self):
        self.set_palette(list(PALETTE_COLORS))

    def save_palette(self):
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(), title="Save Palette",
            defaultextension=".json", filetypes=[("Palette", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.palette, handle, indent=2)
        except OSError:
            return

    def load_palette(self):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(), title="Load Palette",
            filetypes=[("Palette", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return
        if not isinstance(data, list):
            return
        colors = [
            normalized for normalized in
            (self._normalize_hex(item) for item in data) if normalized
        ]
        if colors:
            self.set_palette(colors)

    def _show_swatch_menu(self, event, index):
        self._menu_index = index
        try:
            self.swatch_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.swatch_menu.grab_release()

    def set_selected(self, index):
        self.selected_swatch = index
        self._update_swatch_borders()

    def select_swatch(self, index):
        self.selected_swatch = index
        color = self.palette[index]
        r, g, b = self._hex_to_rgb(color)
        hue, sat, value = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        self.hue, self.sat, self.value = hue * 360.0, sat, value
        self._sync_value_var()
        self.color = color
        self._update_swatch_borders()
        self._refresh_picker()
        self._notify()
        self._record_recent()

    def assign_to_swatch(self):
        """Write the current color into the selected palette slot."""
        if not hasattr(self, "swatches") or not self.palette:
            return
        self._assign_index(self.selected_swatch)

    def _assign_index(self, index):
        if not 0 <= index < len(self.swatches):
            return
        self.palette[index] = self._color
        self.swatches[index].config(bg=self._color)

    def _edit_swatch(self, index):
        if not 0 <= index < len(self.palette):
            return
        original = self.palette[index]

        def preview(color):
            self.palette[index] = color
            if index < len(self.swatches):
                self.swatches[index].config(bg=color)

        from . import dialogs

        chosen = dialogs.ask_color(
            self.winfo_toplevel(), original, on_preview=preview,
        )
        if chosen is None:
            return
        self.selected_swatch = index
        self.palette[index] = chosen
        self.swatches[index].config(bg=chosen)
        self.select_swatch(index)

    def _update_swatch_borders(self):
        if not hasattr(self, "swatches"):
            return
        accent = theme.color("accent")
        border = theme.color("border")
        for index, box in enumerate(self.swatches):
            highlight = accent if index == self.selected_swatch else border
            try:
                box.config(highlightbackground=highlight, highlightcolor=highlight)
            except tk.TclError:
                return

    # ------------------------------------------------------------------ recent
    def _load_recent(self):
        if not self._persist_recent:
            return []
        stored = theme.read_settings().get("recent_colors")
        if not isinstance(stored, list):
            return []
        colors = [
            normalized for normalized in
            (self._normalize_hex(item) for item in stored) if normalized
        ]
        return colors[:RECENT_COLORS_MAX]

    def _record_recent(self):
        color = self._normalize_hex(self._color)
        if not color:
            return
        if self.recent_colors and self.recent_colors[0] == color:
            return
        self.recent_colors = [color] + [
            c for c in self.recent_colors if c != color
        ]
        del self.recent_colors[RECENT_COLORS_MAX:]
        self._rebuild_recent()
        self._schedule_recent_save()

    def _rebuild_recent(self):
        """Redraw the recent row as a fixed grid of slots.

        Filled slots show a remembered color; the remaining slots stay as empty
        placeholders so the row keeps a stable size as colors are added.
        """
        for box in getattr(self, "recent_boxes", []):
            box.destroy()
        self.recent_boxes = []
        border = theme.color("border")
        empty_bg = theme.color("bg")
        for index in range(RECENT_COLORS_MAX):
            color = self.recent_colors[index] if index < len(self.recent_colors) else None
            box = tk.Frame(
                self.recent_grid, width=16, height=16,
                bg=color or empty_bg,
                highlightthickness=1, highlightbackground=border,
            )
            box.grid(row=index // 8, column=index % 8, padx=1, pady=1)
            box.pack_propagate(False)
            box.grid_propagate(False)
            if color is not None:
                theme.skip_bg(box)
                box.bind(
                    "<Button-1>", lambda _e, c=color: self._choose_recent(c)
                )
            self.recent_boxes.append(box)

    def _choose_recent(self, color):
        r, g, b = self._hex_to_rgb(color)
        self.set_rgb(r, g, b)
        self._record_recent()

    def _schedule_recent_save(self):
        if not self._persist_recent:
            return
        if self._recent_save_job is not None:
            try:
                self.after_cancel(self._recent_save_job)
            except (tk.TclError, ValueError):
                pass
        try:
            self._recent_save_job = self.after(800, self._save_recent)
        except tk.TclError:
            self._recent_save_job = None

    def _save_recent(self):
        self._recent_save_job = None
        settings = theme.read_settings()
        settings["recent_colors"] = list(self.recent_colors)
        theme.write_settings(settings)

    # ------------------------------------------------------------------ theme
    def _on_theme(self):
        self._update_swatch_borders()
        self._rebuild_recent()
        self._render_wheel()
