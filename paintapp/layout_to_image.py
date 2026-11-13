"""The Layout Control feature: draw boxes, get a layout prompt.

Two model families understand a layout prompt built from the same rows:

* **MOSS Image** renders a structured JSON *layout* prompt: a scene caption plus
  a list of items, each with a semantic ``label``, a ``caption`` describing it,
  an optional literal ``text`` to render, and a bounding box in
  ``abs_centralized`` ``yxyx`` coordinates (the canvas centre is ``(0, 0)``). See
  the model card's "Layout-to-image generation" section. Built by
  :func:`build_layout`.
* **Krea 2** (the ``*-bbox`` finetunes) renders a compact *grounding DSL*: one
  ``tag[x0,y0,x1,y1] description`` line per element on a fixed 0-1000 grid, x
  first, origin top-left. Its own model card ships the spec. Built by
  :func:`build_bbox`.

:class:`LayoutModel` is the plain data behind the feature (a list of rows, plus
optional meta tags and an enable switch). It survives the Layout Control panel
being rebuilt on every tool change, so it lives on the Tools panel and the
coordinator reads it for storage and for the run's prompt. :class:`LayoutEditor`
is a throwaway widget tree over that model, shown in the Layout Control panel's
sidebar container; it writes straight back into the model and reports each
change through ``on_change``. :func:`layout_prompt` dispatches to the format the
active family expects.
"""

import json
import tkinter as tk
from tkinter import ttk

from . import dialogs, theme
from .generation import PromptBox

# Families whose engine understands a layout prompt: MOSS Image (the JSON
# layout) and Krea 2 (the bbox grounding DSL, see :func:`build_bbox`).
SUPPORTED_FAMILIES = {"moss_image", "krea2"}

# The item labels the MOSS layout format documents; ``text`` (and ``title``)
# items carry a literal string to render, so their text box is enabled. The
# combobox accepts other labels typed by hand.
LABELS = ("object", "text")
TEXT_LABELS = {"text", "title", "label", "caption", "heading"}

# The Krea 2 grounding DSL names elements by tag; these labels map onto one (see
# :data:`_KREA2_TAGS`). ``text`` carries a literal string, so its box is enabled.
KREA2_LABELS = ("panel", "object", "text", "person", "anime", "furry")

# The label choices each family's editor offers, in display order.
FAMILY_LABELS = {
    "moss_image": LABELS,
    "krea2": KREA2_LABELS,
}

# A box narrower or shorter than this (document pixels) is treated as a stray
# click rather than a drawn box.
MIN_BOX = 4

# Box outline/label colours for the canvas overlay, cycled per row.
PALETTE = ("#3b82f6", "#22c55e", "#f97316", "#a855f7", "#14b8a6", "#ec4899")

# Krea 2 grounding-DSL element tag for each layout-row label. Unknown labels fall
# back to ``o`` (object). ``t`` is the only tag that carries verbatim text.
_KREA2_TAGS = {
    "p": "p", "panel": "p",
    "o": "o", "object": "o",
    "t": "t", "text": "t",
    "pe": "pe", "person": "pe",
    "ac": "ac", "anime": "ac",
    "fc": "fc", "furry": "fc", "anthro": "fc",
}

# Character tags take an id (``ac:1[..]``) that is reused to mean "the same one".
_KREA2_CHARACTER_TAGS = {"pe", "ac", "fc"}

# Emission order: panels first, then characters, objects and finally text.
_KREA2_TAG_ORDER = {"p": 0, "pe": 1, "ac": 1, "fc": 1, "o": 2, "t": 3}


def build_layout(rows, caption, canvas_size, output_size, meta=None):
    """The layout dict for ``rows`` (or ``None`` when there is nothing to build).

    ``rows`` is a list of ``{"label", "caption", "text", "bbox"}`` with ``bbox`` a
    document-pixel ``(left, top, right, bottom)`` box. Each box is scaled from the
    canvas size to the generation size and recentred to ``abs_centralized``
    ``yxyx``: rectangles become ``[y0, x0, y1, x1]``. ``meta`` is an optional list
    of tags folded into ``summary.meta``.
    """
    cw, ch = (max(1, int(canvas_size[0])), max(1, int(canvas_size[1])))
    ow, oh = (max(1, int(output_size[0])), max(1, int(output_size[1])))
    items = []
    for row in rows:
        box = row.get("bbox")
        if not box:
            continue
        left, top, right, bottom = box
        if right - left < MIN_BOX or bottom - top < MIN_BOX:
            continue
        label = (row.get("label") or "object").strip().lower()
        item = {
            "label": label,
            "caption": (row.get("caption") or "").strip(),
        }
        text = (row.get("text") or "").strip()
        if label in TEXT_LABELS and text:
            item["text"] = text
        item["bbox"] = [
            round(top / ch * oh - oh / 2),
            round(left / cw * ow - ow / 2),
            round(bottom / ch * oh - oh / 2),
            round(right / cw * ow - ow / 2),
        ]
        items.append(item)
    if not items:
        return None
    summary = {"caption": (caption or "").strip()}
    tags = [tag.strip() for tag in (meta or []) if tag.strip()]
    if tags:
        summary["meta"] = tags
    return {
        "summary": summary,
        "items": items,
        "bbox_coord_space": "abs_centralized",
        "bbox_format": "yxyx",
    }


def build_bbox(rows, caption, meta, canvas_size):
    """The Krea 2 grounding-DSL prompt for ``rows`` (``None`` when there are none).

    ``rows`` is a list of ``{"label", "caption", "text", "bbox"}`` with ``bbox`` a
    document-pixel ``(left, top, right, bottom)`` box. Boxes scale from
    ``canvas_size`` onto the model's fixed 0-1000 grid (x first, origin top-left).
    ``meta`` tags become header lines: a tag starting with ``@`` is the style
    line, one starting with ``~`` the background line (sigils stripped); other
    tags join the style line. Character rows (person / anime / furry) that share a
    Name reuse one numeric id, so the same character stays consistent across
    boxes. Text rows carry their verbatim string in quotes.
    """
    cw, ch = (max(1, int(canvas_size[0])), max(1, int(canvas_size[1])))
    elements = []
    by_name = {}
    next_id = 0
    for row in rows:
        box = row.get("bbox")
        if not box:
            continue
        left, top, right, bottom = box
        if right - left < MIN_BOX or bottom - top < MIN_BOX:
            continue
        tag = _KREA2_TAGS.get(_row_label(row), "o")
        x0, y0 = _grid(left, cw), _grid(top, ch)
        x1, y1 = _grid(right, cw), _grid(bottom, ch)
        if x0 == x1 or y0 == y1:
            continue
        head = tag
        if tag in _KREA2_CHARACTER_TAGS:
            name = (row.get("name") or "").strip().lower()
            cid = by_name.get(name) if name else None
            if cid is None:
                next_id += 1
                cid = next_id
                if name:
                    by_name[name] = cid
            head = f"{tag}:{cid}"
        line = f"{head}[{x0},{y0},{x1},{y1}]"
        if tag == "t":
            text = (row.get("text") or "").strip()
            if text:
                line += '"' + _escape_text(text) + '"'
        desc = " ".join((row.get("caption") or "").split())
        if desc:
            line += " " + desc
        elements.append((tag, line))
    if not elements:
        return None
    lines = []
    summary = " ".join((caption or "").split())
    if summary:
        lines.append(summary)
    lines.extend(_krea2_headers(meta))
    elements.sort(key=lambda pair: _KREA2_TAG_ORDER.get(pair[0], 2))
    lines.extend(line for _tag, line in elements)
    return "\n".join(lines)


def _row_label(row):
    return (row.get("label") or "object").strip().lower()


def _grid(value, size):
    """A document-pixel ``value`` on a ``size``-long axis, snapped to 0-1000."""
    return max(0, min(1000, round(value / size * 1000)))


def _escape_text(text):
    """Escape a verbatim string for the DSL's quoted text span."""
    return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _krea2_headers(meta):
    """Krea 2 header lines (``@`` style, ``~`` background) from the meta tags."""
    style, background, plain = [], [], []
    for tag in (meta or []):
        tag = (tag or "").strip()
        if not tag:
            continue
        if tag.startswith("~"):
            background.append(tag[1:].strip())
        elif tag.startswith("@"):
            style.append(tag[1:].strip())
        else:
            plain.append(tag)
    lines = []
    style_line = ", ".join(part for part in (style or plain) if part)
    if style_line:
        lines.append("@" + style_line)
    bg = " ".join(part for part in background if part)
    if bg:
        lines.append("~" + bg)
    return lines


def layout_prompt(rows, caption, meta, canvas_size, output_size, family="moss_image"):
    """The layout prompt for ``rows`` in ``family``'s format (``None`` when none).

    MOSS Image gets the JSON layout (:func:`build_layout`); Krea 2 gets the
    grounding DSL (:func:`build_bbox`).
    """
    if family == "krea2":
        return build_bbox(rows, caption, meta, canvas_size)
    layout = build_layout(rows, caption, canvas_size, output_size, meta)
    if layout is None:
        return None
    return json.dumps(layout, ensure_ascii=False)


def _clean_tags(values):
    if isinstance(values, str):
        values = values.split(",")
    return [tag.strip() for tag in (values or []) if tag and tag.strip()]


class LayoutModel:
    """The layout rows, meta tags and enable switch (no widgets)."""

    def __init__(self):
        self.rows = []
        self.meta = []
        self.enabled = False
        self.show_caption = False
        self._counter = 0

    def next_name(self):
        self._counter += 1
        return f"Item {self._counter}"

    def add_box(self, bbox, name=None, label="object", caption="", text="",
                color=None):
        row = {
            "name": name or self.next_name(),
            "label": label or "object",
            "text": text or "",
            "caption": caption or "",
            "color": color,
            "bbox": [int(v) for v in bbox] if bbox else None,
        }
        self.rows.append(row)
        return row

    def remove(self, row):
        if row in self.rows:
            self.rows.remove(row)

    def has_items(self):
        return any(row.get("bbox") for row in self.rows)

    def settings(self):
        """The rows as plain dicts for per-canvas/layer storage."""
        rows = []
        for row in self.rows:
            box = row.get("bbox")
            if not box:
                continue
            rows.append({
                "label": (row.get("label") or "object").strip(),
                "caption": (row.get("caption") or "").strip(),
                "text": (row.get("text") or "").strip(),
                "bbox": [int(v) for v in box],
                "name": (row.get("name") or "").strip(),
                "color": row.get("color"),
            })
        return rows

    def set_settings(self, rows, meta=None):
        self.rows = []
        self._counter = 0
        for item in (rows or []):
            if not isinstance(item, dict):
                continue
            self.add_box(
                item.get("bbox"), name=item.get("name"),
                label=item.get("label"), caption=item.get("caption", ""),
                text=item.get("text", ""), color=item.get("color"),
            )
        if meta is not None:
            self.meta = _clean_tags(meta)

    def meta_tags(self):
        return list(self.meta)

    def layout_text(self, family, caption, canvas_size, output_size):
        return layout_prompt(
            self.settings(), caption, self.meta, canvas_size, output_size, family
        )

    def overlay_items(self):
        """Canvas-overlay dicts for the current rows (see the view)."""
        items = []
        for index, row in enumerate(self.rows):
            box = row.get("bbox")
            if not box:
                continue
            items.append({
                "bbox": tuple(box),
                "color": row.get("color") or PALETTE[index % len(PALETTE)],
                "title": (row.get("name") or "").strip()
                         or (row.get("label") or "object").strip(),
                "text": (row.get("text") or "").strip(),
                "caption": ((row.get("caption") or "").strip()
                            if self.show_caption else ""),
            })
        return items


class LayoutEditor(tk.Frame):
    """The Layout Control panel's editor: rows, meta and enable switch."""

    def __init__(self, parent, model, supported=True, family="moss_image",
                 on_capture=None, on_change=None, on_enabled=None):
        super().__init__(parent)
        self.model = model
        self.supported = supported
        self.family = family
        self._on_capture = on_capture
        self._on_change = on_change
        self._on_enabled = on_enabled
        self._build()

    def _build(self):
        self.enabled_var = tk.BooleanVar(value=self.model.enabled)
        self.enabled_check = tk.Checkbutton(
            self, text="Enable layout", variable=self.enabled_var,
            command=self._on_enabled_toggle, anchor="w",
        )
        self.enabled_check.pack(fill=tk.X)
        self.show_caption_var = tk.BooleanVar(value=self.model.show_caption)
        self.show_caption_check = tk.Checkbutton(
            self, text="Show caption in box", variable=self.show_caption_var,
            command=self._on_show_caption_toggle, anchor="w",
        )
        self.show_caption_check.pack(fill=tk.X)
        self.add_btn = tk.Button(
            self, text="Add item from selection", command=self._on_add_from_selection,
        )
        self.add_btn.pack(fill=tk.X, pady=(4, 0))

        meta_row = tk.Frame(self)
        meta_row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(meta_row, text="Meta:", anchor="w").pack(side=tk.LEFT)
        self.meta_var = tk.StringVar(value=", ".join(self.model.meta))
        tk.Entry(meta_row, textvariable=self.meta_var).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.meta_var.trace_add("write", lambda *_: self._on_meta())

        self.hint_label = tk.Label(self, text="", anchor="w", wraplength=210,
                                   justify="left")
        self.hint_label.pack(fill=tk.X, pady=(4, 0))
        theme.register(self.hint_label, fg="muted")

        self.rows_frame = tk.Frame(self)
        self.rows_frame.pack(fill=tk.X)
        self._row_widgets = []
        for row in list(self.model.rows):
            self._build_row(row)

        theme.restyle(self)
        self._refresh_state()
        self._update_hint()

    # ------------------------------------------------------------------ rows
    def _build_row(self, row):
        frame = tk.Frame(self.rows_frame)
        frame.pack(fill=tk.X, pady=(4, 0))
        frame.columnconfigure(1, weight=1)

        tk.Label(frame, text="Name:", anchor="w").grid(
            row=0, column=0, sticky="nw", pady=(2, 0))
        name_var = tk.StringVar(value=row.get("name") or "")
        name_entry = tk.Entry(frame, textvariable=name_var)
        name_entry.grid(row=0, column=1, sticky="ew", padx=(4, 4), pady=(2, 0))
        remove = tk.Button(frame, text="\u2715", width=2,
                           command=lambda r=row: self._remove_row(r))
        remove.grid(row=0, column=2, sticky="ne")

        tk.Label(frame, text="Label:", anchor="w").grid(
            row=1, column=0, sticky="nw", pady=(4, 0))
        label_var = tk.StringVar(value=row.get("label") or "object")
        combo = ttk.Combobox(
            frame, textvariable=label_var, values=list(FAMILY_LABELS.get(
                self.family, LABELS)), width=10)
        combo.grid(row=1, column=1, sticky="w", padx=(4, 4), pady=(4, 0))

        tk.Label(frame, text="Color:", anchor="w").grid(
            row=2, column=0, sticky="nw", pady=(4, 0))
        color_btn = tk.Button(frame, text="", width=3)
        theme.skip(color_btn)
        color_btn.grid(row=2, column=1, sticky="w", padx=(4, 4), pady=(4, 0))

        text_label = tk.Label(frame, text="Text:", anchor="w")
        text_label.grid(row=3, column=0, sticky="nw", pady=(4, 0))
        text_box = PromptBox(frame)
        text_box.grid(row=3, column=1, columnspan=2, sticky="ew",
                      padx=(4, 4), pady=(4, 0))
        text_box.text.insert("1.0", row.get("text") or "")
        text_box.text.edit_modified(False)
        text_box.text.edit_reset()

        tk.Label(frame, text="Caption:", anchor="w").grid(
            row=4, column=0, sticky="nw", pady=(4, 0))
        caption_box = PromptBox(frame)
        caption_box.grid(row=4, column=1, columnspan=2, sticky="ew",
                         padx=(4, 4), pady=(4, 0))
        caption_box.text.insert("1.0", row.get("caption") or "")
        caption_box.text.edit_modified(False)
        caption_box.text.edit_reset()

        tk.Label(frame, text="Box:", anchor="w").grid(
            row=5, column=0, sticky="nw", pady=(4, 0))
        box_label = tk.Label(frame, text="", anchor="w")
        box_label.grid(row=5, column=1, columnspan=2, sticky="w",
                       padx=(4, 0), pady=(4, 0))
        theme.register(box_label, fg="muted")

        divider = tk.Frame(frame, height=1, bd=0, highlightthickness=0)
        divider.grid(row=6, column=0, columnspan=3, sticky="we", pady=(6, 0))
        theme.register(divider, bg="border")

        widgets = {
            "row": row, "frame": frame, "name_var": name_var,
            "name_entry": name_entry, "label_var": label_var, "combo": combo,
            "color_btn": color_btn,
            "text_text": text_box.text, "text_box": text_box,
            "text_label": text_label,
            "caption_text": caption_box.text, "box_label": box_label,
        }
        self._row_widgets.append(widgets)
        color_btn.config(command=lambda w=widgets: self._choose_color(w))

        name_var.trace_add("write", lambda *_a, w=widgets: self._on_name(w))
        label_var.trace_add("write", lambda *_a, w=widgets: self._on_label(w))
        text_box.text.bind(
            "<<Modified>>", lambda _e, w=widgets: self._on_text(w))
        caption_box.text.bind(
            "<<Modified>>", lambda _e, w=widgets: self._on_caption(w))
        self._set_box(widgets)
        self._sync_color(widgets)

    def _remove_row(self, row):
        self.model.remove(row)
        self._drop_row(row)
        self._notify()

    def _drop_row(self, row):
        """Remove one row's widgets without rebuilding the whole list."""
        for index, widgets in enumerate(self._row_widgets):
            if widgets["row"] is row:
                widgets["frame"].destroy()
                del self._row_widgets[index]
                break
        if not self._row_widgets:
            # Tk leaves a container's requested size stale when its last packed
            # child is destroyed, so nudge it back down to empty.
            self.rows_frame.configure(height=1)
        else:
            # The palette colour is positional, so the rows left behind may
            # need a new swatch now that the list has shifted.
            for widgets in self._row_widgets:
                self._sync_color(widgets)
        self._refresh_state()
        self._update_hint()

    def add_row(self, row):
        """Add one row's widgets without rebuilding the existing rows."""
        # The model may have been enabled by the caller (drawing a box arms the
        # tool), so keep the switch in step before enabling the new row.
        self.enabled_var.set(bool(self.model.enabled))
        self.show_caption_var.set(bool(self.model.show_caption))
        self._build_row(row)
        theme.restyle(self)
        self._refresh_state()
        self._update_hint()

    def refresh_boxes(self):
        """Update the box readouts after the model's boxes move.

        Dragging a box on the canvas only changes its ``bbox``, so the row
        widgets stay in place; rebuilding them would reset the panel's scroll
        position and throw away the widget state.
        """
        rows = list(self.model.rows)
        if len(rows) != len(self._row_widgets) or any(
                widgets["row"] is not row
                for widgets, row in zip(self._row_widgets, rows)):
            self._rebuild_rows()
            return
        for widgets in self._row_widgets:
            self._set_box(widgets)
        self._update_hint()

    def _rebuild_rows(self):
        for widgets in self._row_widgets:
            widgets["frame"].destroy()
        self._row_widgets = []
        for row in list(self.model.rows):
            self._build_row(row)
        if not self._row_widgets:
            # Tk leaves a container's requested size stale when its last packed
            # child is destroyed, so the emptied stack would keep the height of
            # the rows it just lost. Nudging the frame's height makes the
            # geometry manager recompute it back down to empty.
            self.rows_frame.configure(height=1)
        theme.restyle(self)
        self._refresh_state()
        self._update_hint()

    def _set_box(self, widgets):
        box = widgets["row"].get("bbox")
        if not box:
            widgets["box_label"].config(text="")
            return
        left, top, right, bottom = box
        widgets["box_label"].config(
            text=f"x {left}  y {top}  w {right - left}  h {bottom - top}"
        )

    def _row_color(self, widgets):
        """A row's overlay colour: its own, else the palette colour for its slot."""
        color = widgets["row"].get("color")
        if color:
            return color
        try:
            index = self.model.rows.index(widgets["row"])
        except ValueError:
            index = 0
        return PALETTE[index % len(PALETTE)]

    def _sync_color(self, widgets):
        color = self._row_color(widgets)
        widgets["color_btn"].config(bg=color, activebackground=color)

    def _choose_color(self, widgets):
        row = widgets["row"]
        initial = self._row_color(widgets)

        def preview(chosen):
            row["color"] = chosen
            widgets["color_btn"].config(bg=chosen, activebackground=chosen)
            self._notify()

        dialogs.ask_color(
            self.winfo_toplevel(), initial, title="Box Color", on_preview=preview,
        )

    # ------------------------------------------------------------- write-back
    def _on_name(self, widgets):
        widgets["row"]["name"] = widgets["name_var"].get().strip()
        self._notify()

    def _on_label(self, widgets):
        widgets["row"]["label"] = widgets["label_var"].get().strip()
        self._refresh_row_text(widgets)
        self._notify()

    def _on_text(self, widgets):
        widget = widgets["text_text"]
        if not widget.edit_modified():
            return
        widget.edit_modified(False)
        widgets["row"]["text"] = widget.get("1.0", "end-1c")
        self._notify()

    def _on_caption(self, widgets):
        widget = widgets["caption_text"]
        if not widget.edit_modified():
            return
        widget.edit_modified(False)
        widgets["row"]["caption"] = widget.get("1.0", "end-1c")
        self._notify()

    def _on_meta(self):
        self.model.meta = _clean_tags(self.meta_var.get())
        self._notify()

    def _on_show_caption_toggle(self):
        self.model.show_caption = bool(self.show_caption_var.get())
        self._notify()

    def _on_enabled_toggle(self):
        enabled = bool(self.enabled_var.get())
        self.model.enabled = enabled
        self._refresh_state()
        self._update_hint()
        self._notify()
        if self._on_enabled is not None:
            self._on_enabled(enabled)

    def _on_add_from_selection(self):
        if self._on_capture is None:
            return
        box = self._on_capture()
        if box is None:
            return
        row = self.model.add_box(box)
        self.add_row(row)
        self._notify()

    # ------------------------------------------------------------------ state
    def _refresh_state(self):
        # Layout Control works on every model: boxes can be drawn and edited
        # regardless of family, and a model that does not read layouts simply
        # ignores them at generation time. The widgets therefore follow the
        # enable switch alone, not whether the family is supported.
        enabled = self.model.enabled
        self.enabled_check.config(state=tk.NORMAL)
        self.add_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
        for widgets in self._row_widgets:
            widgets["combo"].config(state="normal" if enabled else "disabled")
            widgets["name_entry"].config(state=tk.NORMAL if enabled else tk.DISABLED)
            widgets["color_btn"].config(state=tk.NORMAL if enabled else tk.DISABLED)
            widgets["caption_text"].config(
                state=tk.NORMAL if enabled else tk.DISABLED)
            if enabled:
                self._refresh_row_text(widgets)
            else:
                widgets["text_text"].config(state=tk.DISABLED)
                widgets["text_label"].config(state=tk.DISABLED)

    def _refresh_row_text(self, widgets):
        label = (widgets["label_var"].get() or "object").strip().lower()
        state = tk.NORMAL if label in TEXT_LABELS else tk.DISABLED
        widgets["text_text"].config(state=state)
        widgets["text_label"].config(state=state)

    def _update_hint(self):
        if not self.supported:
            text = ("This model does not read layouts; boxes are kept but ignored "
                    "when generating. MOSS Image and Krea 2 use them.")
        elif not self.model.enabled:
            text = "Enable layout, then drag a box on the canvas to add an item."
        elif not self.model.has_items():
            text = "Drag a box on the canvas to add an item."
        else:
            text = ("Each box places one item; the Positive prompt is the scene "
                    "caption. Drag inside a box to draw another; Ctrl-drag to "
                    "move or resize one, clicking overlapping boxes again to "
                    "reach the one underneath.")
        self.hint_label.config(text=text)

    def _notify(self):
        if self._on_change is not None:
            self._on_change()
