"""The Region Prompts panel: per-region prompts for regional prompting.

A run may split its prompt by image region: each row here pairs a prompt with a
spatial mask, and generation conditions that prompt only on the image tokens
inside the mask. The run's own (global) prompt remains the shared base for every
token, so a region prompt refines what that part of the image should contain.

Each row owns its mask. Capturing the current selection (or the painted inpaint
mask) snapshots it *into that row*, so several rows can hold several distinct
regions instead of all reading the same live selection. The snapshot is stored as
a base64 PNG (see :func:`encode_mask`) so it survives a save/load, and the row
draws a thumbnail of it so the user can see which region belongs to which prompt.

The panel is a form only: it reports its rows through ``on_settings`` and asks the
coordinator for a snapshot through ``on_capture`` (the coordinator reads the
selection or painted mask off the document). The coordinator then folds the rows
into the run (see :func:`engine.pipeline.generate`, ``regions_list``).
"""

import base64
import io
import tkinter as tk

from PIL import Image, ImageTk

from . import theme
from .canvasarea import checkerboard
from .generation import PromptBox

# Where a region reads its mask from, reusing the masks the Generate panel
# already understands. These are also the capture sources the panel asks the
# coordinator for.
MASK_SELECTION = "Selection"
MASK_PAINTED = "Painted mask"

# Families whose engine supports regional prompting.
SUPPORTED_FAMILIES = {
    "anima", "stable_diffusion", "flux", "z_image", "qwen_image", "krea2",
    "moss_image",
}

# Longest edge of a row's mask thumbnail, in pixels.
THUMB = 40

# Tint the thumbnail paints the mask's region with, over the checkerboard.
MASK_TINT = (90, 170, 255)


def encode_mask(mask):
    """A base64 PNG of an 8-bit ``mask``, for a row's snapshot in settings."""
    buffer = io.BytesIO()
    mask.convert("L").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def decode_mask(text):
    """The 8-bit mask an :func:`encode_mask` string holds, or ``None``.

    A missing or malformed value decodes to ``None``.
    """
    if not isinstance(text, str) or not text:
        return None
    try:
        data = base64.b64decode(text, validate=True)
        return Image.open(io.BytesIO(data)).convert("L")
    except Exception:
        return None


def _mask_thumbnail(mask):
    """A ``PhotoImage`` preview of ``mask``: its region tinted over a checker."""
    thumb = mask.convert("L").copy()
    thumb.thumbnail((THUMB, THUMB), Image.LANCZOS)
    tint = Image.new("RGBA", thumb.size, MASK_TINT + (255,))
    tint.putalpha(thumb)
    board = checkerboard(thumb.size)
    board.alpha_composite(tint)
    return ImageTk.PhotoImage(board.convert("RGB"))


class RegionsPanel(tk.Frame):
    """A list of per-region prompt rows for a family that supports them."""

    def __init__(self, parent, on_settings=None, on_capture=None):
        super().__init__(parent)
        self._on_settings = on_settings
        self._on_capture = on_capture
        self._family = None
        self._rows = []
        self._mask_counter = 0
        self._loading = False
        self._build()

    def _build(self):
        self.rows_frame = tk.Frame(self)
        self.rows_frame.pack(fill=tk.X)

        bar = tk.Frame(self)
        bar.pack(fill=tk.X, pady=(4, 6))
        # A master switch: with it off the run ignores the rows entirely, so they
        # can be parked without clearing them or the prompts they carry.
        self.enabled_var = tk.BooleanVar(value=False)
        self.enabled_check = tk.Checkbutton(
            bar, text="Enable", variable=self.enabled_var,
            command=self._on_enabled_toggle,
        )
        self.enabled_check.pack(side=tk.LEFT)
        self.add_btn = tk.Button(bar, text="Add region prompt", command=self._on_add_row)
        self.add_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.hint_label = tk.Label(bar, text="", anchor="w")
        self.hint_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        theme.register(self.hint_label, fg="muted")
        # The panel is built before the app's first theme pass at startup, but
        # restyling here keeps it correct if it is ever built later (and matches
        # the dynamically added rows).
        theme.restyle(self)
        self._update_hint()

    def family(self):
        return self._family

    def supported(self):
        return self._family in SUPPORTED_FAMILIES

    def enabled(self):
        """Whether the run should fold the rows into its config."""
        return bool(self.enabled_var.get())

    def set_enabled(self, enabled):
        """Set the master switch without echoing a change (used when loading)."""
        self.enabled_var.set(bool(enabled))
        self._refresh_state()
        self._update_hint()

    def _on_enabled_toggle(self):
        self._refresh_state()
        self._update_hint()
        self._changed()

    def set_family(self, family):
        """Point the panel at ``family``; regional prompts are family-limited."""
        if family == self._family:
            return
        self._family = family
        if not self.supported():
            self._loading = True
            try:
                for row in list(self._rows):
                    self._remove_row(row, notify=False)
            finally:
                self._loading = False
        self._refresh_state()
        self._update_hint()

    def _on_add_row(self):
        """Add a row, prefilled with the current selection when there is one."""
        mask = self._request_mask(MASK_SELECTION, warn=False)
        self.add_row(mask=mask, source=MASK_SELECTION if mask is not None else None)
        self._changed()

    def add_row(self, prompt="", mask=None, name=None, source=None, strength=1.0):
        """Add one region row.

        ``mask`` may be a :class:`~PIL.Image.Image` to snapshot now, or an
        :func:`encode_mask` string to restore (``set_settings``). The masked row
        keeps the captured copy, so a later selection change cannot retarget it.
        ``strength`` scales how much the region's prompt pulls on its tokens
        relative to the run's base prompt (1.0 is neutral).
        """
        if not self.supported():
            return None
        frame = tk.Frame(self.rows_frame)
        frame.pack(fill=tk.X, pady=(2, 4))
        frame.columnconfigure(1, weight=1)

        row = {}

        # Line 1: the region's prompt (a multiline box, drag the grip to grow it)
        # and a remove button.
        tk.Label(frame, text="Prompt:", anchor="w").grid(
            row=0, column=0, sticky="nw", pady=(2, 0)
        )
        prompt_box = PromptBox(frame)
        prompt_box.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        prompt_box.text.insert("1.0", prompt)
        prompt_box.text.edit_modified(False)
        prompt_box.text.edit_reset()
        remove = tk.Button(frame, text="\u2715", width=2, command=lambda: self._remove_row(row))
        remove.grid(row=0, column=2, sticky="ne")

        # Line 2: the captured mask's thumbnail and name, so the several rows a
        # run can hold are told apart by what they actually cover.
        tk.Label(frame, text="Mask:", anchor="w").grid(row=1, column=0, sticky="nw", pady=(4, 0))
        preview = tk.Frame(frame)
        preview.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(4, 0), pady=(3, 0))
        preview.columnconfigure(1, weight=1)
        thumb = tk.Label(preview, anchor="w")
        thumb.grid(row=0, column=0, sticky="w")
        name_label = tk.Label(preview, text="No mask", anchor="w")
        name_label.grid(row=0, column=1, sticky="w", padx=(6, 0))
        theme.register(name_label, fg="muted")
        clear = tk.Button(preview, text="\u2715", width=2, command=lambda: self._clear_mask(row))
        clear.grid(row=0, column=2, sticky="e", padx=(4, 0))

        # Line 3: capture controls, indented under the mask preview.
        buttons = tk.Frame(frame)
        buttons.grid(row=2, column=1, columnspan=2, sticky="w", padx=(4, 0), pady=(2, 0))
        use_selection = tk.Button(
            buttons, text="From selection",
            command=lambda: self._capture(row, MASK_SELECTION),
        )
        use_selection.pack(side=tk.LEFT)
        use_painted = tk.Button(
            buttons, text="From painted",
            command=lambda: self._capture(row, MASK_PAINTED),
        )
        use_painted.pack(side=tk.LEFT, padx=(4, 0))

        # Line 4: how strongly the region's prompt steers its own tokens.
        tk.Label(frame, text="Strength:", anchor="w").grid(row=3, column=0, sticky="nw", pady=(6, 0))
        try:
            strength = float(strength)
        except (TypeError, ValueError):
            strength = 1.0
        strength_var = tk.DoubleVar(value=strength)
        strength_scale = tk.Scale(
            frame, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL,
            variable=strength_var, length=140,
        )
        strength_scale.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(4, 0))

        # A hairline under every row separates the several region prompts a run
        # can hold; the line rides on the row, so removing one keeps it right.
        divider = tk.Frame(frame, height=1, bd=0, highlightthickness=0)
        divider.grid(row=4, column=0, columnspan=3, sticky="we", pady=(6, 0))
        theme.register(divider, bg="border")

        row.update({
            "frame": frame,
            "prompt_text": prompt_box.text,
            "thumb": thumb,
            "name_label": name_label,
            "clear_btn": clear,
            "use_selection_btn": use_selection,
            "use_painted_btn": use_painted,
            "strength_scale": strength_scale,
            "strength_var": strength_var,
            "mask_b64": None,
            "mask_name": None,
            "source": None,
            "mask_thumb": None,
        })
        strength_var.trace_add("write", lambda *_: self._changed())
        prompt_box.text.bind(
            "<<Modified>>", lambda _e, r=row: self._on_prompt_modified(r)
        )
        # Rows are built after the app's initial theme pass, so style them here
        # or they keep the light default chrome in dark mode.
        theme.restyle(frame)
        self._rows.append(row)

        # Adopt a mask supplied at build time: a fresh capture, or a stored
        # snapshot from ``set_settings``.
        if mask is not None:
            self._adopt_mask(row, mask, name, source)

        self._refresh_state()
        self._update_hint()
        return row

    def _request_mask(self, source, warn=True):
        """Ask the coordinator for a snapshot of ``source``, or ``None``."""
        if self._on_capture is None:
            return None
        return self._on_capture(source, warn)

    def _capture(self, row, source):
        """Snapshot ``source`` into ``row`` (the user pressed a capture button)."""
        if row not in self._rows:
            return
        mask = self._request_mask(source, warn=True)
        if mask is None:
            return
        self._adopt_mask(row, mask, source=source)
        self._changed()

    def _adopt_mask(self, row, mask, name=None, source=None):
        """Store ``mask`` (a PIL image, or an encoded string) on ``row``."""
        if isinstance(mask, Image.Image):
            image = mask
            encoded = encode_mask(mask)
        else:
            image = decode_mask(mask)
            if image is None:
                return
            encoded = mask
        if not name:
            name = self._next_mask_name()
        row["mask_b64"] = encoded
        row["mask_name"] = name
        row["source"] = source
        row["mask_thumb"] = _mask_thumbnail(image)
        row["thumb"].config(image=row["mask_thumb"])
        row["name_label"].config(text=self._mask_text(row))

    def _clear_mask(self, row):
        """Drop ``row``'s snapshot (the user pressed the mask's clear button)."""
        row["mask_b64"] = None
        row["mask_name"] = None
        row["source"] = None
        row["mask_thumb"] = None
        row["thumb"].config(image="")
        row["name_label"].config(text="No mask")
        self._changed()

    def _next_mask_name(self):
        self._mask_counter += 1
        return f"Region {self._mask_counter}"

    def _sync_mask_counter(self):
        """Keep auto-names ahead of the ``Region N`` names already in the panel."""
        highest = 0
        for row in self._rows:
            name = row.get("mask_name") or ""
            prefix = "Region "
            if name.startswith(prefix):
                try:
                    highest = max(highest, int(name[len(prefix):]))
                except ValueError:
                    pass
        self._mask_counter = max(self._mask_counter, highest)

    @staticmethod
    def _mask_text(row):
        name = row.get("mask_name") or "Mask"
        source = row.get("source")
        return f"{name} \u00b7 {source.lower()}" if source else name

    def _on_prompt_modified(self, row):
        """Report a prompt-box edit through the panel's change callback."""
        widget = row["prompt_text"]
        if not widget.edit_modified():
            return
        widget.edit_modified(False)
        self._changed()

    def _remove_row(self, row, notify=True):
        if row not in self._rows:
            return
        self._rows.remove(row)
        row["frame"].destroy()
        if not self._rows:
            # Tk leaves a container's requested size stale when its last packed
            # child is destroyed, so the emptied stack would keep the height of
            # the rows it just lost. Nudging the frame's height makes the
            # geometry manager recompute it back down to empty.
            self.rows_frame.configure(height=1)
        self._refresh_state()
        if notify:
            self._changed()

    def _refresh_state(self):
        supported = self.supported()
        enabled = supported and self.enabled()
        self.enabled_check.config(state=tk.NORMAL if supported else tk.DISABLED)
        self.add_btn.config(state=tk.NORMAL if enabled else tk.DISABLED)
        for row in self._rows:
            state = tk.NORMAL if enabled else tk.DISABLED
            row["prompt_text"].config(state=state)
            row["use_selection_btn"].config(state=state)
            row["use_painted_btn"].config(state=state)
            row["clear_btn"].config(state=state)
            row["strength_scale"].config(state=state)

    def _update_hint(self):
        if not self.supported():
            text = "Regional prompts are not available for this model."
        elif not self.enabled():
            text = "Region prompts are off."
        elif not self._rows:
            text = "No region prompts."
        elif any(row.get("mask_b64") is None for row in self._rows):
            text = "Give every region a mask (from the selection or painted mask)."
        else:
            text = "Each region's prompt applies inside its own mask."
        self.hint_label.config(text=text)

    @staticmethod
    def _number(var, fallback):
        """A row's numeric field, or ``fallback`` when it is empty or malformed."""
        try:
            return float(var.get())
        except (tk.TclError, ValueError, AttributeError):
            return fallback

    def settings(self):
        """The rows as a list of dicts for storage (skips fully empty rows)."""
        rows = []
        for row in self._rows:
            prompt = row["prompt_text"].get("1.0", "end-1c").strip()
            mask = row.get("mask_b64")
            if not prompt and not mask:
                continue
            rows.append({
                "prompt": prompt,
                "mask": mask,
                "name": row.get("mask_name"),
                "source": row.get("source"),
                "strength": self._number(row.get("strength_var"), 1.0),
            })
        return rows

    def set_settings(self, rows):
        if rows is None:
            return
        self._loading = True
        try:
            for row in list(self._rows):
                self._remove_row(row, notify=False)
            if self.supported():
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    self.add_row(
                        prompt=item.get("prompt", ""),
                        mask=item.get("mask"),
                        name=item.get("name"),
                        source=item.get("source"),
                        strength=item.get("strength", 1.0),
                    )
            self._sync_mask_counter()
        finally:
            self._loading = False
        self._update_hint()

    def _changed(self):
        if self._loading or self._on_settings is None:
            return
        self._update_hint()
        self._on_settings(self.settings())