"""User-remappable application shortcuts.

Every global keyboard shortcut lives in one registry here instead of being
scattered across ``root.bind(...)`` calls. The registry is a list of
:func:`definitions` - one entry per command with a stable ``id``, a menu
``label``, the ``group`` it belongs to, its default Tk sequences, the ``command``
it runs and whether it should stand down while a text field has focus.

:class:`ShortcutManager` owns the live bindings: it loads any user overrides
from ``settings.json`` (the ``shortcuts`` section), binds the effective
sequences on the root window, and writes overrides back on save. Keys are kept
as Tk event sequences (``"<Control-Shift-S>"``) and rendered for the UI with
:func:`display` (``"Ctrl+Shift+S"``); :func:`from_event` turns a pressed key
combination into a canonical sequence for the recorder in the Settings dialog.
"""

import tkinter as tk

from engine import settings as engine_settings

from .constants import ALT_STATE_MASK

# Keysyms that only represent a modifier being held; a recorder keeps waiting
# for a "real" key when one of these arrives.
_MODIFIER_KEYSYMS = {
    "Control_L", "Control_R", "Shift_L", "Shift_R", "Alt_L", "Alt_R",
    "Super_L", "Super_R", "Meta_L", "Meta_R", "Win_L", "Win_R",
    "Hyper_L", "Hyper_R", "Caps_Lock", "Num_Lock", "ISO_Level3_Shift",
    "Multi_key",
}

# Keysyms whose meaning already implies Shift, so the modifier is dropped when
# recording (otherwise Control+"+" would be stored as Control+Shift+plus and
# never match the Control+plus binding Tk expects).
_SHIFTED_SYMBOLS = {
    "plus", "underscore", "asterisk", "parenleft", "parenright", "braceleft",
    "braceright", "colon", "quotedbl", "less", "greater", "question", "bar",
    "asciitilde", "asciicircum", "ampersand", "dollar", "percent",
    "numbersign", "at", "exclam",
}

# Readable names for keysyms that read poorly when shown raw.
_KEY_LABELS = {
    "Escape": "Esc",
    "Return": "Enter",
    "Delete": "Del",
    "BackSpace": "Backspace",
    "space": "Space",
    "plus": "+",
    "equal": "=",
    "minus": "-",
    "comma": ",",
    "period": ".",
    "slash": "/",
    "backslash": "\\",
    "semicolon": ";",
    "apostrophe": "'",
    "grave": "`",
    "bracketleft": "[",
    "bracketright": "]",
    "Prior": "PageUp",
    "Next": "PageDown",
    "Tab": "Tab",
    "Home": "Home",
    "End": "End",
    "Insert": "Insert",
}


def definitions(app):
    """Return the shortcut registry for ``app`` in menu order.

    Each entry is a dict with ``id``, ``label``, ``group``, ``bindings`` (the
    current Tk sequences, seeded from the defaults), ``defaults``, ``command``
    and an optional ``guard`` flag. ``guard`` mirrors the app's old ``guard()``
    wrapper: the command is skipped while a text widget has focus so typing
    ``Ctrl+C`` into a prompt copies text instead of the active layer.
    """
    return [
        # ------------------------------------------------------------- global
        _def("escape", "Cancel, deselect, then Arrow tool", "Global",
             ["<Escape>"], app._escape),

        # --------------------------------------------------------------- file
        _def("file_new", "New Project", "File", ["<Control-n>"], app.new_document),
        _def("file_open_project", "Open Project\u2026", "File",
             ["<Control-Shift-O>"], app.open_project),
        _def("file_open_image", "Open Image\u2026", "File",
             ["<Control-o>"], app.open_image),
        _def("file_save_project", "Save Project", "File",
             ["<Control-s>"], app.save_project),
        _def("file_save_image", "Save Image As\u2026", "File",
             ["<Control-Shift-S>"], app.save_image),
        _def("file_export", "Export Images\u2026", "File",
             ["<Control-Shift-E>"], app.export_images),
        _def("file_exit", "Exit", "File", ["<Control-q>"], app._on_close),

        # --------------------------------------------------------------- edit
        _def("edit_settings", "Settings\u2026", "Edit",
             ["<Control-comma>"], app.open_settings),
        _def("edit_undo", "Undo", "Edit", ["<Control-z>"], app.undo, guard=True),
        _def("edit_redo", "Redo", "Edit",
             ["<Control-y>", "<Control-Shift-Z>"], app.redo, guard=True),
        _def("edit_cut", "Cut", "Edit", ["<Control-x>"], app.cut, guard=True),
        _def("edit_copy", "Copy", "Edit", ["<Control-c>"], app.copy, guard=True),
        _def("edit_paste", "Paste", "Edit", ["<Control-v>"], app.paste, guard=True),
        _def("edit_select_all", "Select All", "Edit",
             ["<Control-a>"], app.select_all, guard=True),
        _def("edit_delete", "Delete", "Edit",
             ["<Delete>", "<BackSpace>"], app.delete_forward, guard=True),
        _def("edit_clear", "Clear Layer", "Edit",
             ["<Control-Delete>"], app.clear),
        _def("edit_transform", "Transform Layer\u2026", "Edit",
             ["<Control-Shift-T>"], app.transform_layer),
        _def("edit_free_transform", "Free Transform", "Edit",
             ["<Control-Shift-F>"], app.toggle_transform),
        _def("edit_remove_bg", "Remove Background", "Edit",
             ["<Control-b>"], app.start_remove_background),
        _def("edit_upscale", "Upscale", "Edit", ["<Control-u>"], app.start_upscale),

        # -------------------------------------------------------------- tools
        # Single-letter tool keys mirror the usual raster-editor layout; each is
        # guarded so typing in a sidebar field never switches tools.
        _def("tool_arrow", "Arrow (No Tool)", "Tools",
             ["<a>"], app.toggle_arrow, guard=True),
        _def("tool_brush", "Brush", "Tools",
             ["<b>"], app.toggle_brush, guard=True),
        _def("tool_pick", "Eyedropper", "Tools",
             ["<i>"], app.toggle_eyedropper, guard=True),
        _def("tool_eraser", "Eraser", "Tools",
             ["<e>", "<Control-e>"], app.toggle_eraser, guard=True),
        _def("tool_fill", "Paint Bucket", "Tools",
             ["<g>"], app.toggle_fill, guard=True),
        _def("tool_select", "Select", "Tools",
             ["<m>", "<Control-t>"], app.toggle_select, guard=True),
        _def("tool_lasso", "Lasso", "Tools",
             ["<l>", "<Control-l>"], app.toggle_lasso, guard=True),
        _def("tool_path", "Path", "Tools",
             ["<p>", "<Control-Shift-P>"], app.toggle_path, guard=True),
        _def("tool_vector", "Vector", "Tools",
             ["<x>"], app.toggle_vector, guard=True),
        _def("tool_poses", "Poses", "Tools",
             ["<Shift-M>"], app.toggle_poses, guard=True),
        _def("tool_liquify", "Liquify", "Tools",
             ["<Shift-L>"], app.toggle_liquify, guard=True),
        _def("tool_warp", "Warp", "Tools",
             ["<Shift-W>"], app.toggle_warp, guard=True),
        _def("tool_puppet", "Puppet Warp", "Tools",
             ["<Shift-P>"], app.toggle_puppet, guard=True),
        _def("tool_move", "Move", "Tools",
             ["<v>"], app.toggle_move, guard=True),
        _def("tool_drag", "Drag (Pan)", "Tools",
             ["<h>", "<Control-d>"], app.toggle_drag, guard=True),
        _def("tool_text", "Text", "Tools",
             ["<t>", "<Control-Shift-X>"], app.toggle_text, guard=True),
        _def("tool_gradient", "Gradient", "Tools",
             ["<Shift-G>"], app.toggle_gradient, guard=True),
        _def("tool_shape", "Shape", "Tools",
             ["<u>"], app.toggle_shape, guard=True),
        _def("tool_clone", "Clone Stamp", "Tools",
             ["<s>"], app.toggle_clone, guard=True),
        _def("tool_smudge", "Smudge", "Tools",
             ["<r>"], app.toggle_smudge, guard=True),
        _def("tool_dodge", "Dodge", "Tools",
             ["<o>"], app.toggle_dodge, guard=True),
        _def("tool_burn", "Burn", "Tools",
             ["<Shift-O>"], app.toggle_burn, guard=True),
        _def("tool_wand", "Magic Wand", "Tools",
             ["<w>"], app.toggle_wand, guard=True),
        _def("tool_paste", "Paste Tool", "Tools",
             ["<y>"], app.toggle_paste, guard=True),
        _def("tool_crop", "Crop", "Tools",
             ["<c>"], app.toggle_crop, guard=True),

        # ------------------------------------------------------------ brushes
        _def("brush_prev_preset", "Previous Brush Preset", "Brushes",
             ["<bracketleft>"], app.prev_brush_preset, guard=True),
        _def("brush_next_preset", "Next Brush Preset", "Brushes",
             ["<bracketright>"], app.next_brush_preset, guard=True),

        # -------------------------------------------------------------- layer
        _def("layer_new", "New Layer", "Layer",
             ["<Control-Shift-L>"], app.add_layer),
        _def("layer_new_vector", "New Vector Layer", "Layer",
             ["<Control-Shift-V>"], app.add_vector_layer),
        _def("layer_from_selection", "New Layer from Selection", "Layer",
             ["<Control-j>"], app.layer_from_selection, guard=True),
        _def("layer_merge", "Merge Layers", "Layer",
             ["<Control-m>"], app.merge_layers),
        _def("layer_group", "Group Selected Layers", "Layer",
             ["<Control-Shift-U>"], app.group_selected_layers, guard=True),
        _def("layer_mask_edit", "Edit Mask", "Layer",
             ["<Control-Shift-M>"], app.toggle_mask_edit),
        _def("layer_isolate", "Isolate Layer", "Layer",
             ["<Control-Shift-I>"], app.isolate_layer),
        _def("layer_unisolate", "Unisolate Layer", "Layer",
             ["<Control-Shift-Alt-U>"], app.unisolate_layer),
        _def("layer_show_all", "Show All Layers", "Layer",
             ["<Control-Shift-Alt-I>"], app.show_all_layers),

        # -------------------------------------------------------------- canvas
        _def("canvas_duplicate", "Duplicate Canvas", "Canvas",
             ["<Control-Shift-N>"], app.duplicate_canvas),
        # Left/Right step through the tabs. Guarded so a focused text field keeps
        # its own caret movement; the in-canvas text tool already consumes the
        # arrows (it returns "break"), so this never steals them mid-edit.
        _def("canvas_prev", "Previous Canvas", "Canvas",
             ["<Left>"], lambda: app._cycle_canvas(-1), guard=True),
        _def("canvas_next", "Next Canvas", "Canvas",
             ["<Right>"], lambda: app._cycle_canvas(1), guard=True),
        _def("canvas_flip_h", "Flip Canvas Horizontal", "Canvas",
             ["<Control-Shift-H>"], app.flip_canvas_horizontal),
        _def("canvas_flip_v", "Flip Canvas Vertical", "Canvas",
             ["<Control-Shift-J>"], app.flip_canvas_vertical),
        _def("canvas_rotate_cw", "Rotate Canvas 90\u00b0 CW", "Canvas",
             ["<Control-Shift-period>"], app.rotate_canvas_cw),
        _def("canvas_rotate_ccw", "Rotate Canvas 90\u00b0 CCW", "Canvas",
             ["<Control-Shift-comma>"], app.rotate_canvas_ccw),

        # --------------------------------------------------------------- view
        _def("view_zoom_in", "Zoom In", "View",
             ["<Control-plus>", "<Control-equal>"], app.zoom_in),
        _def("view_zoom_out", "Zoom Out", "View",
             ["<Control-minus>"], app.zoom_out),
        _def("view_zoom_reset", "Reset Zoom", "View",
             ["<Control-Key-0>"], app.reset_zoom),
        _def("view_rotate_cw", "Rotate View Right", "View",
             ["<Control-bracketright>"], app.rotate_view_cw),
        _def("view_rotate_ccw", "Rotate View Left", "View",
             ["<Control-bracketleft>"], app.rotate_view_ccw),
        _def("view_flip", "Flip View Horizontal", "View",
             ["<Control-Shift-Alt-H>"], app.flip_view_horizontal),
        _def("view_flip_vertical", "Flip View Vertical", "View",
             ["<Control-Shift-Alt-J>"], app.flip_view_vertical),
        _def("view_reset", "Reset View", "View",
             ["<Control-Shift-Key-0>"], app.reset_view),
        _def("view_rulers", "Toggle Rulers", "View",
             ["<Control-Shift-R>"], app.toggle_rulers),
        _def("view_exploded", "Toggle Exploded Layers", "View",
             ["<Control-Shift-K>"], app.toggle_exploded),
        _def("view_guides", "Toggle Guides", "View",
             ["<Control-semicolon>"], app.toggle_guides),
        _def("view_theme", "Toggle Dark Mode", "View",
             ["<Control-Shift-D>"], app.toggle_theme),
        _def("view_gallery", "Toggle Canvas Gallery", "View",
             ["<Control-Shift-G>"], app.toggle_gallery),
        _def("view_gallery_side", "Toggle Gallery Sidebar", "View",
             ["<Control-Shift-Alt-G>"], app.toggle_gallery_side),
        _def("view_fullscreen", "Toggle Fullscreen", "View",
             ["<F11>"], app.toggle_fullscreen),

        # ----------------------------------------------------------- generate
        _def("gen_start", "Generate", "Generate", ["<Control-g>"], app.start_generate),
        _def("gen_cancel", "Cancel Generation", "Generate",
             ["<Control-period>"], app.cancel_generate),
    ]


def _def(shortcut_id, label, group, defaults, command, guard=False):
    return {
        "id": shortcut_id,
        "label": label,
        "group": group,
        "bindings": list(defaults),
        "defaults": list(defaults),
        "command": command,
        "guard": guard,
    }


# --------------------------------------------------------------- formatting
def display(sequence):
    """Render a Tk event sequence as a short label (``"<Control-n>"`` -> ``"Ctrl+N"``)."""
    if not sequence:
        return ""
    token = sequence.strip().strip("<>")
    mods = []
    key = None
    for part in token.split("-"):
        if not part:
            continue
        if part in ("Control", "Ctrl"):
            if "Ctrl" not in mods:
                mods.append("Ctrl")
        elif part in ("Alt", "Mod1"):
            if "Alt" not in mods:
                mods.append("Alt")
        elif part == "Shift":
            if "Shift" not in mods:
                mods.append("Shift")
        elif part == "Key":
            continue
        else:
            key = part
    label = ""
    if key is not None:
        label = _KEY_LABELS.get(key)
        if label is None:
            label = key.upper() if len(key) == 1 else key
    return "+".join(mods + [label]) if label else "+".join(mods)


def display_all(sequences):
    """Join several sequences (aliases) into one label, or ``"(none)"``."""
    labels = [display(seq) for seq in sequences if seq]
    return ", ".join(labels) if labels else "(none)"


def from_event(event):
    """Turn a ``<KeyPress>`` event into a canonical Tk sequence, or ``None``.

    ``None`` means the event carried no usable key (a bare modifier, a lock key,
    or an unprintable keysym), so a recorder should keep waiting.
    """
    keysym = event.keysym
    if keysym in _MODIFIER_KEYSYMS:
        return None

    state = event.state
    control = bool(state & 0x0004)
    shift = bool(state & 0x0001)
    alt = bool(state & ALT_STATE_MASK)

    if shift and keysym in _SHIFTED_SYMBOLS:
        shift = False
    if len(keysym) == 1 and keysym.isalpha():
        key = keysym.upper() if shift else keysym
    elif len(keysym) == 1 and keysym.isdigit():
        # Digits use the explicit "Key-" form (Tk's canonical spelling).
        key = f"Key-{keysym}"
    else:
        key = keysym

    mods = []
    if control:
        mods.append("Control")
    if alt:
        mods.append("Alt")
    if shift:
        mods.append("Shift")
    return "<" + "-".join(mods + [key]) + ">"


def normalize(sequence):
    """Wrap a bare keysym in ``<>`` and drop whitespace; ``""`` stays empty."""
    if not sequence:
        return ""
    sequence = str(sequence).strip()
    if not sequence:
        return ""
    if sequence.startswith("<") and sequence.endswith(">"):
        return sequence
    return f"<{sequence}>"


# ------------------------------------------------------------- the manager
class ShortcutManager:
    """Owns the live keyboard bindings and their persisted overrides."""

    def __init__(self, app):
        self.app = app
        self.root = app.root
        self._defs = definitions(app)
        self._by_id = {entry["id"]: entry for entry in self._defs}
        self._bound = {}  # shortcut id -> [(sequence, funcid)]
        self.load()

    # ------------------------------------------------------------- definitions
    def definitions(self):
        return list(self._defs)

    def groups(self):
        """The group names in first-seen order."""
        order = []
        for entry in self._defs:
            if entry["group"] not in order:
                order.append(entry["group"])
        return order

    def entry(self, shortcut_id):
        return self._by_id.get(shortcut_id)

    def bindings(self, shortcut_id):
        entry = self._by_id.get(shortcut_id)
        return list(entry["bindings"]) if entry else []

    def primary(self, shortcut_id):
        bindings = self.bindings(shortcut_id)
        return bindings[0] if bindings else ""

    def accelerator(self, shortcut_id):
        """The label a menu shows for ``shortcut_id`` (its first binding)."""
        return display(self.primary(shortcut_id))

    def is_customized(self, shortcut_id):
        entry = self._by_id.get(shortcut_id)
        return bool(entry and entry["bindings"] != entry["defaults"])

    # ---------------------------------------------------------------- storage
    def load(self):
        stored = engine_settings.section("shortcuts")
        for entry in self._defs:
            entry["defaults"] = list(entry.get("defaults", entry["bindings"]))
            custom = stored.get(entry["id"])
            if isinstance(custom, list):
                cleaned = [normalize(seq) for seq in custom if isinstance(seq, str)]
                entry["bindings"] = [seq for seq in cleaned if seq]
            else:
                entry["bindings"] = list(entry["defaults"])

    def save(self):
        """Persist only the shortcuts that differ from their defaults."""
        overrides = {
            entry["id"]: list(entry["bindings"])
            for entry in self._defs
            if entry["bindings"] != entry["defaults"]
        }
        settings = engine_settings.read()
        if overrides:
            settings["shortcuts"] = overrides
        else:
            settings.pop("shortcuts", None)
        engine_settings.write(settings)

    # --------------------------------------------------------------- editing
    def conflict(self, shortcut_id, sequence):
        """Return the entry already using ``sequence`` (other than ``shortcut_id``)."""
        if not sequence:
            return None
        for entry in self._defs:
            if entry["id"] == shortcut_id:
                continue
            if sequence in entry["bindings"]:
                return entry
        return None

    def set_binding(self, shortcut_id, sequence):
        """Point ``shortcut_id`` at ``sequence``. Returns ``(ok, message)``.

        A clash with another shortcut is refused so the caller can report which
        command already owns the combination.
        """
        entry = self._by_id.get(shortcut_id)
        sequence = normalize(sequence)
        if entry is None or not sequence:
            return False, ""
        clash = self.conflict(shortcut_id, sequence)
        if clash is not None:
            return False, clash["label"]
        entry["bindings"] = [sequence]
        return True, ""

    def clear(self, shortcut_id):
        entry = self._by_id.get(shortcut_id)
        if entry is not None:
            entry["bindings"] = []

    def reset(self, shortcut_id):
        entry = self._by_id.get(shortcut_id)
        if entry is not None:
            entry["bindings"] = list(entry["defaults"])

    def reset_all(self):
        for entry in self._defs:
            entry["bindings"] = list(entry["defaults"])

    def restore(self, snapshot):
        """Restore a ``{id: [sequences]}`` snapshot (used to undo a cancelled edit)."""
        for entry in self._defs:
            if entry["id"] in snapshot:
                entry["bindings"] = list(snapshot[entry["id"]])

    def snapshot(self):
        return {entry["id"]: list(entry["bindings"]) for entry in self._defs}

    # --------------------------------------------------------------- binding
    def apply(self):
        """Rebind every shortcut on the root window to match current state."""
        # Drop the sequences bound by the previous apply. The manager owns every
        # key binding on the root, so removing by sequence removes only its own.
        for sequence in {seq for entries in self._bound.values()
                         for seq, _funcid in entries}:
            try:
                self.root.unbind(sequence)
            except tk.TclError:
                pass
        self._bound = {}
        for entry in self._defs:
            handler = self._make_handler(entry)
            for sequence in entry["bindings"]:
                try:
                    funcid = self.root.bind(sequence, handler, add="+")
                except tk.TclError:
                    continue
                self._bound.setdefault(entry["id"], []).append((sequence, funcid))

    def _make_handler(self, entry):
        command = entry["command"]
        guard = entry.get("guard", False)

        def handler(_event):
            if guard and self.app._text_has_focus():
                return None
            command()
            return None

        return handler
