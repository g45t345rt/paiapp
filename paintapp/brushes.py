"""Brush presets: named parameter sets saved to a per-user library.

A preset is one tool's settings (``brush`` or ``eraser``) captured as a dict of
the same attribute names :class:`paintapp.drawing.ToolsPanel` persists, plus the
shared ``brush_size``. Each preset is stored as its own JSON file under the
per-user app folder (``%APPDATA%\\Pai\\brushes`` on Windows, ``~/.config/Pai``
elsewhere), beside ``settings.json``. A small starter set is written once so a
fresh install has usable brushes.

The module also builds the "Brushes" sidebar panel: a scrollable, thumbnail list
for the active tool, with Save / Save As / Rename / Delete.
"""

import json
import os
import uuid
import zipfile

import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

from engine import settings as engine_settings

from . import dialogs, theme
from .brush import Dynamics
from .constants import rgba
from .scrollbar import ArrowScrollbar

#: Marker in the preset file and the only format this module reads or writes.
FORMAT = "pai-brush"
VERSION = 1

#: Tools that have a preset library. Each paints with ``brush_size`` and the
#: shared stamping engine, so a preset is just that tool's own settings.
TOOLS = ("brush", "eraser", "clone", "smudge", "dodge", "burn")

#: Settings used to preview the tools that are not a plain brush or eraser, as
#: ``(hardness key, spacing key, opacity key or None)``.
_GENERIC_PREVIEW = {
    "clone": ("clone_hardness", "clone_spacing", "clone_opacity"),
    "smudge": ("smudge_hardness", "smudge_spacing", None),
    "dodge": ("dodge_hardness", "dodge_spacing", None),
    "burn": ("burn_hardness", "burn_spacing", None),
}

#: Name of the sentinel file that records the starter set was written, so it is
#: not re-added after the user deletes presets.
DEFAULTS_MARKER = ".defaults-v1"


def library_dir():
    """The folder holding the brush preset files."""
    return os.path.join(engine_settings.app_dir(), "brushes")


class BrushPreset:
    """A named, tool-specific set of brush parameters."""

    def __init__(self, name, tool, params, preset_id=None):
        self.id = preset_id or uuid.uuid4().hex
        self.name = str(name)
        self.tool = tool if tool in TOOLS else "brush"
        self.params = dict(params or {})

    def to_dict(self):
        return {
            "format": FORMAT,
            "version": VERSION,
            "id": self.id,
            "name": self.name,
            "tool": self.tool,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, data):
        """Build a preset from a stored dict, or ``None`` if it is not one."""
        if not isinstance(data, dict) or data.get("format") != FORMAT:
            return None
        tool = data.get("tool")
        if tool not in TOOLS:
            return None
        params = data.get("params")
        return cls(
            str(data.get("name") or "Brush"), tool,
            params if isinstance(params, dict) else {},
            str(data.get("id") or uuid.uuid4().hex),
        )


def _brush_params(size, opacity=100, hardness=100, flow=100, spacing=10,
                  stabilize=0, stabilize_mode="simple", antialias=True,
                  blend="Normal", tip="Round",
                  pattern="none", vel_size=0, vel_opacity=0, vel_flow=0,
                  vel_hardness=0, vel_spacing=0, taper_in=0, taper_out=0,
                  grain="none", grain_scale=100, grain_strength=0,
                  grain_mode="canvas", hue_var=0, sat_var=0, val_var=0,
                  pickup=0, wet_edge=0):
    return {
        "brush_size": size, "brush_opacity": opacity,
        "brush_hardness": hardness, "brush_flow": flow,
        "brush_spacing": spacing, "brush_stabilize": stabilize,
        "brush_stabilize_mode": stabilize_mode,
        "brush_antialias": antialias, "brush_blend": blend,
        "brush_tip": tip, "brush_pattern": pattern,
        "brush_velocity_size": vel_size, "brush_velocity_opacity": vel_opacity,
        "brush_velocity_flow": vel_flow, "brush_velocity_hardness": vel_hardness,
        "brush_velocity_spacing": vel_spacing,
        "brush_taper_in": taper_in, "brush_taper_out": taper_out,
        "brush_grain": grain, "brush_grain_scale": grain_scale,
        "brush_grain_strength": grain_strength, "brush_grain_mode": grain_mode,
        "brush_jitter_hue": hue_var, "brush_jitter_sat": sat_var,
        "brush_jitter_val": val_var, "brush_pickup": pickup,
        "brush_wet_edge": wet_edge,
    }


def _eraser_params(size, opacity=100, hardness=100, spacing=15, stabilize=0,
                   stabilize_mode="simple", antialias=True, blend="Normal",
                   tip="Round", pattern="none",
                   target="transparent", color="#000000", vel_size=0,
                   vel_opacity=0, vel_flow=0, vel_hardness=0, vel_spacing=0,
                   taper_in=0, taper_out=0, grain="none", grain_scale=100,
                   grain_strength=0, grain_mode="canvas", hue_var=0, sat_var=0,
                   val_var=0, pickup=0):
    return {
        "brush_size": size, "eraser_opacity": opacity,
        "eraser_hardness": hardness, "eraser_spacing": spacing,
        "eraser_stabilize": stabilize,
        "eraser_stabilize_mode": stabilize_mode,
        "eraser_antialias": antialias,
        "eraser_blend": blend, "eraser_tip": tip, "eraser_pattern": pattern,
        "eraser_target": target, "eraser_color": color,
        "eraser_velocity_size": vel_size, "eraser_velocity_opacity": vel_opacity,
        "eraser_velocity_flow": vel_flow, "eraser_velocity_hardness": vel_hardness,
        "eraser_velocity_spacing": vel_spacing,
        "eraser_taper_in": taper_in, "eraser_taper_out": taper_out,
        "eraser_grain": grain, "eraser_grain_scale": grain_scale,
        "eraser_grain_strength": grain_strength, "eraser_grain_mode": grain_mode,
        "eraser_jitter_hue": hue_var, "eraser_jitter_sat": sat_var,
        "eraser_jitter_val": val_var, "eraser_pickup": pickup,
    }


#: The starter brushes written to a fresh library. Order is display order.
DEFAULT_SPECS = [
    ("Hard Round", "brush", _brush_params(30, hardness=100, spacing=10)),
    ("Soft Round", "brush", _brush_params(60, hardness=0, flow=60, spacing=12,
                                           vel_size=20)),
    ("Airbrush", "brush", _brush_params(90, hardness=0, flow=15, opacity=80,
                                        spacing=8)),
    ("Ink Pen", "brush", _brush_params(18, hardness=100, spacing=10,
                                       vel_size=55, vel_opacity=35,
                                       taper_in=45, taper_out=35)),
    ("Pencil", "brush", _brush_params(8, hardness=85, flow=70, spacing=18,
                                      vel_size=30, vel_opacity=40)),
    ("Marker", "brush", _brush_params(24, hardness=90, spacing=12, vel_size=15,
                                      taper_out=15)),
    ("Charcoal", "brush", _brush_params(34, hardness=55, flow=85, spacing=12,
                                        vel_size=25, grain="paper",
                                        grain_scale=140, grain_strength=70)),
    ("Paper Pencil", "brush", _brush_params(6, hardness=80, flow=60, spacing=16,
                                            vel_size=35, vel_opacity=40,
                                            grain="noise", grain_strength=60)),
    ("Canvas Dry", "brush", _brush_params(40, hardness=70, flow=70, spacing=14,
                                          vel_size=20, grain="canvas",
                                          grain_scale=120, grain_strength=75,
                                          grain_mode="canvas")),
    ("Halftone", "brush", _brush_params(28, hardness=100, spacing=10,
                                        grain="halftone", grain_strength=65,
                                        grain_mode="stamp")),
    ("Watercolor", "brush", _brush_params(48, hardness=30, flow=45, opacity=70,
                                          spacing=8, pickup=65, wet_edge=55,
                                          grain="paper", grain_strength=30)),
    ("Speckle", "brush", _brush_params(20, hardness=85, spacing=18,
                                       hue_var=35, sat_var=25, val_var=30)),
    ("Hard Eraser", "eraser", _eraser_params(30, hardness=100, spacing=15)),
    ("Soft Eraser", "eraser", _eraser_params(70, hardness=0, opacity=70,
                                             spacing=12, vel_size=15)),
]


def default_presets():
    """The starter presets as fresh :class:`BrushPreset` objects."""
    return [BrushPreset(name, tool, params) for name, tool, params in DEFAULT_SPECS]


def preset_path(preset):
    """The file a preset is stored in (keyed by its stable id)."""
    return os.path.join(library_dir(), preset.id + ".json")


def load_presets():
    """Every readable preset in the library, sorted by tool then name."""
    directory = library_dir()
    presets = []
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return presets
    for entry in entries:
        if not entry.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, entry), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        preset = BrushPreset.from_dict(data)
        if preset is not None:
            presets.append(preset)
    presets.sort(key=lambda p: (p.tool != "brush", p.name.lower()))
    return presets


def save_preset(preset):
    """Write ``preset`` to the library. Returns its path."""
    directory = library_dir()
    os.makedirs(directory, exist_ok=True)
    path = preset_path(preset)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(preset.to_dict(), handle, indent=2)
    return path


def delete_preset(preset):
    """Remove ``preset`` from the library. Returns True when it was removed."""
    try:
        os.remove(preset_path(preset))
        return True
    except OSError:
        return False


def ensure_defaults():
    """Write the starter presets once, the first time the library is used."""
    directory = library_dir()
    marker = os.path.join(directory, DEFAULTS_MARKER)
    if os.path.exists(marker):
        return
    try:
        os.makedirs(directory, exist_ok=True)
        for preset in default_presets():
            save_preset(preset)
        with open(marker, "w", encoding="utf-8") as handle:
            handle.write("1")
    except OSError:
        pass


def restore_defaults():
    """Re-add the starter brushes on demand, replacing same-named presets.

    Unlike :func:`ensure_defaults` this ignores the one-shot marker, so it works
    after the user has edited or deleted the starters. A preset whose tool and
    name match a default is removed first, so repeated restores replace rather
    than pile up duplicates. Returns the number of defaults written.
    """
    directory = library_dir()
    try:
        os.makedirs(directory, exist_ok=True)
    except OSError:
        return 0
    defaults = default_presets()
    names = {(preset.tool, preset.name) for preset in defaults}
    for preset in load_presets():
        if (preset.tool, preset.name) in names:
            delete_preset(preset)
    written = 0
    for preset in defaults:
        save_preset(preset)
        written += 1
    try:
        with open(os.path.join(directory, DEFAULTS_MARKER), "w",
                  encoding="utf-8") as handle:
            handle.write("1")
    except OSError:
        pass
    return written


def dynamics_for(params, prefix, size):
    """Build a :class:`Dynamics` from a preset's velocity/taper params."""
    def value(key, default=0):
        try:
            return float(params.get(f"{prefix}_{key}", default) or 0)
        except (TypeError, ValueError):
            return float(default)

    return Dynamics(
        size=value("velocity_size") / 100.0,
        opacity=value("velocity_opacity") / 100.0,
        flow=value("velocity_flow") / 100.0,
        hardness=value("velocity_hardness") / 100.0,
        spacing=value("velocity_spacing") / 100.0,
        taper_in=size * value("taper_in") / 100.0,
        taper_out=size * value("taper_out") / 100.0,
    )


def _grain_args(params, prefix):
    """The grain and colour-dynamics keywords for the preview renderer."""
    return {
        "grain": params.get(f"{prefix}_grain", "none"),
        "grain_scale": _as_int(params.get(f"{prefix}_grain_scale"), 100) / 100.0,
        "grain_strength": _as_int(params.get(f"{prefix}_grain_strength"), 0) / 100.0,
        "grain_mode": params.get(f"{prefix}_grain_mode", "canvas"),
        "jitter_hue": _as_int(params.get(f"{prefix}_jitter_hue"), 0) / 100.0,
        "jitter_sat": _as_int(params.get(f"{prefix}_jitter_sat"), 0) / 100.0,
        "jitter_val": _as_int(params.get(f"{prefix}_jitter_val"), 0) / 100.0,
        "pickup": _as_int(params.get(f"{prefix}_pickup"), 0) / 100.0,
    }


def preset_preview_image(preset, color=(40, 40, 40)):
    """Render the preset's sample stroke (see :func:`drawing.brush_preview_image`)."""
    from .drawing import brush_preview_image
    params = preset.params
    size = _as_int(params.get("brush_size"), 30)
    if preset.tool == "eraser":
        target = params.get("eraser_target", "transparent")
        return brush_preview_image(
            size, _as_int(params.get("eraser_hardness"), 100), 100,
            _as_int(params.get("eraser_opacity"), 100),
            _as_int(params.get("eraser_spacing"), 25),
            str(params.get("eraser_tip", "Round")).lower(),
            params.get("eraser_pattern", "none"),
            dynamics_for(params, "eraser", size),
            kind="eraser-color" if target == "color" else "eraser",
            color=rgba(params.get("eraser_color", "#000000"))[:3],
            **_grain_args(params, "eraser"),
        )
    if preset.tool == "brush":
        return brush_preview_image(
            size, _as_int(params.get("brush_hardness"), 100),
            _as_int(params.get("brush_flow"), 100),
            _as_int(params.get("brush_opacity"), 100),
            _as_int(params.get("brush_spacing"), 25),
            str(params.get("brush_tip", "Round")).lower(),
            params.get("brush_pattern", "none"),
            dynamics_for(params, "brush", size),
            color=color,
            **_grain_args(params, "brush"),
        )
    # clone / smudge / dodge / burn: show the stamp shape they paint with.
    hardness_key, spacing_key, opacity_key = _GENERIC_PREVIEW.get(
        preset.tool, ("brush_hardness", "brush_spacing", None))
    return brush_preview_image(
        size, _as_int(params.get(hardness_key), 100), 100,
        _as_int(params.get(opacity_key), 100) if opacity_key else 100,
        _as_int(params.get(spacing_key), 25), "round", "none",
        dynamics_for(params, preset.tool, size), color=color,
    )


def _as_int(value, default):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


#: Width/height of a preset row's thumbnail on the Brushes panel.
THUMB_W = 128
THUMB_H = 30


class BrushLibraryPanel(tk.Frame):
    """The "Brushes" sidebar panel: a thumbnail list for the active tool."""

    def __init__(self, parent, tools, on_apply=None, on_status=None):
        super().__init__(parent)
        self._tools = tools
        self._on_apply = on_apply
        self._on_status = on_status
        self._tool = "brush"
        self._presets = []
        self._rows = {}
        self._names = {}
        self._thumbs = {}
        self._selected_id = None
        self._modified = False
        self._drag_id = None
        self._build()
        self.reload()

    # ------------------------------------------------------------- public API
    def set_tool(self, tool):
        """Show presets for ``tool`` (``"brush"`` or ``"eraser"``)."""
        tool = tool if tool in TOOLS else "brush"
        if tool == self._tool:
            return
        self._tool = tool
        self.reload()

    def reload(self):
        """Reload presets from disk, apply the search box, and rebuild."""
        ensure_defaults()
        presets = [p for p in load_presets() if p.tool == self._tool]
        query = self._search_var.get().strip().lower()
        if query:
            presets = [p for p in presets if query in p.name.lower()]
        self._presets = self._apply_order(presets)
        if self._selected_id is None:
            self._selected_id = self._last_used()
        self._render()

    # ------------------------------------------------------------ ordering
    def _order_key(self):
        return f"order_{self._tool}"

    def _apply_order(self, presets):
        """Sort ``presets`` into the order the user dragged them into."""
        stored = engine_settings.section("brushes").get(self._order_key())
        if not isinstance(stored, list):
            return presets
        rank = {str(pid): index for index, pid in enumerate(stored)}
        return sorted(presets,
                      key=lambda p: (rank.get(p.id, len(rank)), p.name.lower()))

    def _persist_order(self):
        section = engine_settings.section("brushes")
        section[self._order_key()] = [p.id for p in self._presets]
        engine_settings.set_section("brushes", section)

    # --------------------------------------------------------------- building
    def _build(self):
        self._search_var = tk.StringVar()
        search = tk.Entry(self, textvariable=self._search_var)
        search.pack(fill=tk.X, pady=(0, 4))
        search.bind("<KeyRelease>", lambda _e: self.reload())

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(
            body, highlightthickness=0, bd=0, height=220,
            bg=theme.color("listbox_bg"),
        )
        scroll = ArrowScrollbar(body, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._list = tk.Frame(self._canvas, bg=theme.color("listbox_bg"))
        # Pin the list background to the theme token so a restyle pass keeps it
        # (the canvas itself is skipped by ``theme.restyle``; ``_on_theme``
        # recolours it on a theme switch).
        theme.register(self._list, bg="listbox_bg")
        self._window = self._canvas.create_window((0, 0), window=self._list,
                                                  anchor="nw")
        self._list.bind("<Configure>", self._on_list_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(4, 0))
        for text, command in (("Save", self._save), ("Save As", self._save_as),
                              ("Rename", self._rename), ("Delete", self._delete)):
            tk.Button(buttons, text=text, command=command).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        pack = tk.Frame(self)
        pack.pack(fill=tk.X, pady=(2, 0))
        for text, command in (("Export Pack\u2026", self._export_pack),
                              ("Import Pack\u2026", self._import_pack)):
            tk.Button(pack, text=text, command=command).pack(
                side=tk.LEFT, fill=tk.X, expand=True, padx=1)
        theme.restyle(self)

    def _on_list_configure(self, _event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self._canvas.itemconfigure(self._window, width=event.width)

    def _on_theme(self):
        """Recolour the list and rebuild its rows for the active theme."""
        bg = theme.color("listbox_bg")
        try:
            self._canvas.config(bg=bg)
            self._list.config(bg=bg)
        except tk.TclError:
            return
        self._render()

    def _render(self):
        for child in self._list.winfo_children():
            child.destroy()
        self._rows.clear()
        self._names.clear()
        if not self._presets:
            empty = tk.Label(
                self._list, text="No presets yet. Save the current settings.",
                anchor="w", justify="left", wraplength=200,
            )
            theme.register(empty, bg="listbox_bg", fg="muted")
            empty.pack(fill=tk.X, padx=4, pady=4)
        for preset in self._presets:
            self._make_row(preset)
        self._update_selection()
        theme.restyle(self)

    def _make_row(self, preset):
        row = tk.Frame(self._list, bd=1, relief=tk.FLAT)
        row.pack(fill=tk.X, padx=2, pady=1)
        handle = tk.Label(row, text="\u22ee\u22ee", cursor="fleur", width=2)
        handle.pack(side=tk.LEFT)
        handle.bind("<ButtonPress-1>",
                    lambda _e, p=preset: self._on_drag_press(p))
        handle.bind("<B1-Motion>", self._on_drag_motion)
        handle.bind("<ButtonRelease-1>", self._on_drag_release)
        thumb = tk.Label(row, bd=0)
        theme.register(thumb, bg="listbox_bg")
        thumb.pack(side=tk.LEFT)
        photo = self._thumbs.get(preset.id)
        if photo is None:
            photo = ImageTk.PhotoImage(self._thumbnail(preset))
            self._thumbs[preset.id] = photo
        thumb.config(image=photo)
        name = tk.Label(row, text=preset.name, anchor="w")
        theme.register(name, bg="listbox_bg", fg="text")
        name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        for widget in (row, thumb, name):
            widget.bind("<Button-1>", lambda _e, p=preset: self._apply(p))
        self._rows[preset.id] = row
        self._names[preset.id] = name

    def _thumbnail(self, preset):
        image = preset_preview_image(preset)
        return image.convert("RGBA").resize((THUMB_W, THUMB_H), Image.LANCZOS)

    def _is_modified(self, preset):
        """True when the tool's live settings differ from the selected preset.

        Only the keys the preset actually stores are compared, so a preset saved
        before a new setting existed is not shown as edited just because that
        setting now has a default value.
        """
        if preset is None or preset.tool != self._tool:
            return False
        try:
            live = self._tools.tool_params(preset.tool)
        except Exception:
            return False
        return any(live.get(key) != value for key, value in preset.params.items())

    def refresh_modified(self):
        """Re-check whether the selected preset has been edited (shows a ``*``)."""
        modified = self._is_modified(self._selected())
        if modified != self._modified:
            self._update_selection()

    def _update_selection(self):
        preset = self._selected()
        self._modified = self._is_modified(preset)
        for preset_id, row in self._rows.items():
            selected = preset_id == self._selected_id
            row.config(relief=tk.SUNKEN if selected else tk.FLAT)
            label = self._names.get(preset_id)
            if label is None:
                continue
            name = next((p.name for p in self._presets
                         if p.id == preset_id), "")
            label.config(text=name + (" *" if selected and self._modified else ""))
        theme.restyle(self)

    def cycle(self, step):
        """Apply the previous/next preset for the active tool."""
        if not self._presets:
            return
        ids = [p.id for p in self._presets]
        if self._selected_id in ids:
            index = (ids.index(self._selected_id) + step) % len(ids)
        else:
            index = 0 if step > 0 else len(ids) - 1
        self._apply(self._presets[index])

    # ------------------------------------------------------------- reordering
    def _on_drag_press(self, preset):
        self._drag_id = preset.id
        return "break"

    def _row_at(self, x_root, y_root):
        """The preset row under the pointer, or ``None``."""
        widget = self.winfo_containing(x_root, y_root)
        rows = {id(row): preset_id for preset_id, row in self._rows.items()}
        while widget is not None:
            if id(widget) in rows:
                return widget, rows[id(widget)]
            widget = widget.master
        return None, None

    def _on_drag_motion(self, event):
        if self._drag_id is None:
            return "break"
        row = self._rows.get(self._drag_id)
        target, _preset_id = self._row_at(event.x_root, event.y_root)
        if row is None or target is None or target is row:
            return "break"
        midpoint = target.winfo_rooty() + target.winfo_height() // 2
        if event.y_root < midpoint:
            row.pack(before=target, fill=tk.X, padx=2, pady=1)
        else:
            row.pack(after=target, fill=tk.X, padx=2, pady=1)
        return "break"

    def _on_drag_release(self, _event):
        if self._drag_id is None:
            return "break"
        self._drag_id = None
        ordered = []
        for widget in self._list.winfo_children():
            for preset_id, row in self._rows.items():
                if row is widget:
                    ordered.append(preset_id)
                    break
        if len(ordered) == len(self._presets):
            rank = {pid: index for index, pid in enumerate(ordered)}
            self._presets.sort(key=lambda p: rank.get(p.id, 0))
            self._persist_order()
        return "break"

    # ---------------------------------------------------------------- actions
    def _selected(self):
        for preset in self._presets:
            if preset.id == self._selected_id:
                return preset
        return None

    def _apply(self, preset):
        self._selected_id = preset.id
        self._tools.apply_tool_params(preset.tool, preset.params)
        self._remember(preset)
        self._update_selection()
        if self._on_apply is not None:
            self._on_apply()

    def _save(self):
        preset = self._selected()
        if preset is None:
            self._save_as()
            return
        preset.params = self._tools.tool_params(preset.tool)
        save_preset(preset)
        self._thumbs.pop(preset.id, None)
        self.reload()

    def _save_as(self):
        name = dialogs.ask_string(self.winfo_toplevel(), "Save Brush",
                                  "Brush name:", initialvalue="New Brush")
        if not name:
            return
        preset = BrushPreset(name.strip(), self._tool,
                             self._tools.tool_params(self._tool))
        save_preset(preset)
        self._selected_id = preset.id
        self._remember(preset)
        self.reload()

    def _rename(self):
        preset = self._selected()
        if preset is None:
            return
        name = dialogs.ask_string(self.winfo_toplevel(), "Rename Brush",
                                  "Brush name:", initialvalue=preset.name)
        if not name:
            return
        preset.name = name.strip()
        save_preset(preset)
        self._render()

    def _delete(self):
        preset = self._selected()
        if preset is None:
            return
        if not dialogs.ask_yes_no(self.winfo_toplevel(), "Delete Brush",
                                  f"Delete '{preset.name}'?"):
            return
        delete_preset(preset)
        self._thumbs.pop(preset.id, None)
        if self._selected_id == preset.id:
            self._selected_id = None
        self.reload()

    # ------------------------------------------------------------- brush packs
    def _export_pack(self):
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(), title="Export Brush Pack",
            defaultextension=".zip", filetypes=[("Brush pack", "*.zip")])
        if not path:
            return
        presets = load_presets()
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as pack:
                for preset in presets:
                    pack.writestr(preset.id + ".json",
                                  json.dumps(preset.to_dict(), indent=2))
        except OSError:
            self._report("Could not write the brush pack.")
            return
        self._report(f"Exported {len(presets)} brushes.")

    def _import_pack(self):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(), title="Import Brush Pack",
            filetypes=[("Brush pack", "*.zip")])
        if not path:
            return
        count = 0
        try:
            with zipfile.ZipFile(path) as pack:
                for info in pack.infolist():
                    if not info.filename.endswith(".json"):
                        continue
                    try:
                        data = json.loads(pack.read(info.filename))
                    except (ValueError, OSError):
                        continue
                    preset = BrushPreset.from_dict(data)
                    if preset is not None:
                        save_preset(preset)
                        count += 1
        except (OSError, zipfile.BadZipFile):
            self._report("That file is not a brush pack.")
            return
        self.reload()
        self._report(f"Imported {count} brushes.")

    def _report(self, text):
        if self._on_status is not None:
            self._on_status(text)

    # ------------------------------------------------------------- last used
    def _last_used(self):
        return engine_settings.section("brushes").get(f"last_{self._tool}")

    def _remember(self, preset):
        section = engine_settings.section("brushes")
        section[f"last_{preset.tool}"] = preset.id
        engine_settings.set_section("brushes", section)
