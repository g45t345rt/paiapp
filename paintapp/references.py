"""The References panel: reference images for a generation run.

Three independent reference features live here, each in its own collapsible
section:

* The Anima reference adapter: reference images drive character/style transfer
  through clean offset-RoPE reference frames plus a gated LoRA. Several images
  can be listed; they are stitched left-to-right into the one reference image the
  adapter reads, and two characters in one image are a single stitched
  reference. Only the ``anima`` family honours it; the adapters live in their own
  ``adapters/`` pool (not ``loras/``) because an ordinary LoRA applies on every
  run while this one is active only while a reference is attached.
* Reference images (multi-image edit): an ordered list for Qwen-Image, each
  encoded by the Qwen3-VL vision tower and substituted into the DiT's vision
  slots, and for MiniMax H3 Image, each encoded by the H3 image VAE and packed as
  a reference frame. A run conditions on several reference images at once. Only
  the ``qwen_image`` and ``h3_image`` families honour it.
* Krea 2 AnyPaint (arbitrary-mask inpainting/outpainting): a functional adapter
  LoRA for Krea 2 that turns the masked region of the canvas into a generated
  edit while preserving the rest. It reuses the Generate panel's Source and
  Region (mask) inputs; only the ``krea2`` family honours it. See
  :mod:`engine.krea2_anypaint`.

Both lists are built from the same rows. A row is a reference *slot*: pick what
"selected" means (a highlighted asset, the active layer, or the canvas) and press
**Assign** to capture that image into the row now. Because the image is captured
at assign time, two rows can hold two different assets -- the panel does not
re-resolve a shared "current selection" for every row at generate time.

The panel is a form only: it reports its fields through ``on_settings`` and asks
the coordinator to capture an image through ``on_capture`` (see
:meth:`~paintapp.app.PaintApp._capture_reference_selection`); the coordinator
folds the bindings into the run config and captures the images (see
:meth:`~paintapp.app.PaintApp.start_generate`). The family is pushed in by the
coordinator with :meth:`ReferencesPanel.set_family` (it owns the model picker).
"""

import os
import tkinter as tk
from tkinter import ttk

from engine import paths as engine_paths

from . import theme
from .generation import (
    SOURCE_ASSET,
    SOURCE_CANVAS,
    SOURCE_LAYER,
    _lora_name,
)

# A reference row's image source: a highlighted asset, the active layer, or the
# composited canvas. The row's Assign button captures the current selection.
REF_KINDS = (SOURCE_ASSET, SOURCE_LAYER, SOURCE_CANVAS)
# Legacy panel settings used a single "Off"/source string; kept only to read old
# projects back into a row.
REF_OFF = "Off"
# The reference guider's colour-drift correction, matching the ComfyUI node.
REF_CORRECTIONS = ("apg", "rescale", "none")

# A row's kind label <-> the canonical token stored in its binding.
_KIND_TOKENS = {SOURCE_ASSET: "asset", SOURCE_LAYER: "layer", SOURCE_CANVAS: "canvas"}
_TOKEN_KINDS = {token: label for label, token in _KIND_TOKENS.items()}
# Disclosure glyphs shared with the sidebar's panel headers.
_SECTION_OPEN = "\u25be"
_SECTION_CLOSED = "\u25b8"


class _Section:
    """A collapsible block: a clickable glyph header over a hideable body.

    The header carries an optional right-side control (Add / checkbox); the
    body holds the section's rows. A short summary (e.g. the reference count)
    rides the header so a collapsed section still says what it holds.
    """

    def __init__(self, parent, title, open_by_default=True):
        self.title = title
        self._summary = ""
        self.frame = tk.Frame(parent)
        self.header = tk.Frame(self.frame)
        self.header.pack(fill=tk.X)
        self.header.columnconfigure(0, weight=1)
        self.title_var = tk.StringVar()
        self.toggle = tk.Button(
            self.header, textvariable=self.title_var, anchor="w",
            relief=tk.FLAT, bd=0, padx=0, pady=0, cursor="hand2",
            command=self.toggle_open,
        )
        self.toggle.grid(row=0, column=0, sticky="we")
        self.body = tk.Frame(self.frame)
        self.open = bool(open_by_default)
        self._apply()

    def set_open(self, open_):
        open_ = bool(open_)
        if open_ != self.open:
            self.open = open_
            self._apply()

    def toggle_open(self):
        self.open = not self.open
        self._apply()

    def set_summary(self, summary):
        summary = summary or ""
        if summary != self._summary:
            self._summary = summary
            self._apply()

    def _apply(self):
        glyph = _SECTION_OPEN if self.open else _SECTION_CLOSED
        suffix = f" ({self._summary})" if self._summary else ""
        self.title_var.set(f"{glyph}  {self.title}{suffix}")
        if self.open:
            if not self.body.winfo_manager():
                self.body.pack(fill=tk.X)
        elif self.body.winfo_manager():
            self.body.pack_forget()


class ReferencesPanel(tk.Frame):
    """The Anima reference adapter and the Qwen multi-reference list.

    Each feature is a collapsible section; the panel folds the sections for
    other families down to their headers, and a header shows the section's
    reference count (or ``on`` for AnyPaint) so a folded section still reads.
    """

    def __init__(self, parent, on_settings=None, on_capture=None):
        super().__init__(parent)
        self._on_settings = on_settings
        # ``on_capture(kind)`` snapshots the current selection of ``kind``
        # ("asset"/"layer"/"canvas") into a binding, or returns ``None`` after
        # warning. Supplied by the coordinator.
        self._on_capture = on_capture
        self._family = None
        self._loading = False
        self._ref_lora_paths = {}
        self._anypaint_lora_paths = {}
        self._anima_refs = []
        self._qwen_refs = []
        self._build()

    def _build(self):
        # ------------------------------------------------ anima reference adapter
        self._anima_section = _Section(
            self, "Anima references", open_by_default=True,
        )
        self.ref_header = self._anima_section.header
        self.ref_add_btn = tk.Button(
            self.ref_header, text="Add", command=lambda: self._add_anima_reference(),
        )
        self.ref_add_btn.grid(row=0, column=1, sticky="e", padx=(4, 0))
        self._anima_section.frame.pack(fill=tk.X, pady=(4, 0))

        body = self._anima_section.body
        self.ref_rows = tk.Frame(body)
        self.ref_rows.pack(fill=tk.X)

        ref_lora_row = tk.Frame(body)
        ref_lora_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(ref_lora_row, text="Ref adapter:").pack(side=tk.LEFT)
        self.ref_lora_var = tk.StringVar(value="(none)")
        self.ref_lora_menu = ttk.Combobox(
            ref_lora_row, textvariable=self.ref_lora_var, values=["(none)"],
            state="readonly", width=18,
        )
        self.ref_lora_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        # Re-scan the adapters folder each time the picker is opened, so a file
        # downloaded (e.g. via Model Settings' Hugging Face browser) appears
        # without restarting the app.
        self.ref_lora_menu.bind(
            "<Button-1>", lambda _e: self._refresh_reference_loras(), add="+"
        )
        self._refresh_reference_loras()

        ref_params_row = tk.Frame(body)
        ref_params_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(ref_params_row, text="Ref strength:").pack(side=tk.LEFT)
        self.ref_strength_var = tk.DoubleVar(value=1.0)
        self.ref_strength_spin = tk.Spinbox(
            ref_params_row, from_=0.0, to=2.0, increment=0.05, width=5,
            textvariable=self.ref_strength_var,
        )
        self.ref_strength_spin.pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(ref_params_row, text="Ref guidance:").pack(side=tk.LEFT, padx=(6, 0))
        self.ref_guidance_var = tk.DoubleVar(value=4.5)
        self.ref_guidance_spin = tk.Spinbox(
            ref_params_row, from_=0.0, to=20.0, increment=0.1, width=5,
            textvariable=self.ref_guidance_var,
        )
        self.ref_guidance_spin.pack(side=tk.LEFT, padx=(4, 0))

        ref_correction_row = tk.Frame(body)
        ref_correction_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(ref_correction_row, text="Ref correction:").pack(side=tk.LEFT)
        self.ref_correction_var = tk.StringVar(value="apg")
        self.ref_correction_menu = ttk.Combobox(
            ref_correction_row, textvariable=self.ref_correction_var,
            values=list(REF_CORRECTIONS), state="readonly", width=8,
        )
        self.ref_correction_menu.pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(ref_correction_row, text="Offset:").pack(side=tk.LEFT, padx=(6, 0))
        self.ref_offset_var = tk.DoubleVar(value=1.0)
        self.ref_offset_spin = tk.Spinbox(
            ref_correction_row, from_=0.0, to=4.0, increment=0.05, width=5,
            textvariable=self.ref_offset_var,
        )
        self.ref_offset_spin.pack(side=tk.LEFT, padx=(4, 0))

        # Ref MP gets its own row: sharing the correction row pushed its spinbox
        # past the sidebar's right edge, where it could not be clicked.
        ref_mp_row = tk.Frame(body)
        ref_mp_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(ref_mp_row, text="Ref MP:").pack(side=tk.LEFT)
        self.ref_megapixels_var = tk.DoubleVar(value=1.0)
        self.ref_megapixels_spin = tk.Spinbox(
            ref_mp_row, from_=0.25, to=2.0, increment=0.05, width=5,
            textvariable=self.ref_megapixels_var,
        )
        self.ref_megapixels_spin.pack(side=tk.LEFT, padx=(4, 0))

        self._refresh_reference_state()

        qref_divider = tk.Frame(self, height=1, bd=0, highlightthickness=0)
        qref_divider.pack(fill=tk.X, pady=(6, 0))
        theme.register(qref_divider, bg="border")

        # -------------------------------------------------- reference images
        self._qwen_section = _Section(
            self, "Reference images", open_by_default=True,
        )
        self.qref_header = self._qwen_section.header
        self.qref_add_btn = tk.Button(
            self.qref_header, text="Add", command=lambda: self._add_qwen_reference(),
        )
        self.qref_add_btn.grid(row=0, column=1, sticky="e", padx=(4, 0))
        self._qwen_section.frame.pack(fill=tk.X, pady=(2, 0))
        self.qref_rows = tk.Frame(self._qwen_section.body)
        self.qref_rows.pack(fill=tk.X)
        self._refresh_qwen_reference_state()

        anypaint_divider = tk.Frame(self, height=1, bd=0, highlightthickness=0)
        anypaint_divider.pack(fill=tk.X, pady=(6, 0))
        theme.register(anypaint_divider, bg="border")

        # --------------------------------------------- krea 2 anypaint (mask edit)
        self._anypaint_section = _Section(
            self, "AnyPaint (Krea 2)", open_by_default=True,
        )
        self.ap_header = self._anypaint_section.header
        self.anypaint_var = tk.BooleanVar(value=False)
        self.anypaint_check = tk.Checkbutton(
            self.ap_header, variable=self.anypaint_var,
            command=self._on_anypaint_toggle,
        )
        self.anypaint_check.grid(row=0, column=1, sticky="e", padx=(4, 0))
        self._anypaint_section.frame.pack(fill=tk.X, pady=(2, 0))

        body = self._anypaint_section.body
        ap_lora_row = tk.Frame(body)
        ap_lora_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(ap_lora_row, text="Adapter:").pack(side=tk.LEFT)
        self.anypaint_lora_var = tk.StringVar(value="(none)")
        self.anypaint_lora_menu = ttk.Combobox(
            ap_lora_row, textvariable=self.anypaint_lora_var, values=["(none)"],
            state="readonly", width=18,
        )
        self.anypaint_lora_menu.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        self.anypaint_lora_menu.bind(
            "<Button-1>", lambda _e: self._refresh_anypaint_loras(), add="+"
        )
        self._refresh_anypaint_loras()

        ap_strength_row = tk.Frame(body)
        ap_strength_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(ap_strength_row, text="Adapter strength:").pack(side=tk.LEFT)
        self.anypaint_strength_var = tk.DoubleVar(value=1.0)
        self.anypaint_strength_spin = tk.Spinbox(
            ap_strength_row, from_=0.0, to=2.0, increment=0.05, width=5,
            textvariable=self.anypaint_strength_var,
        )
        self.anypaint_strength_spin.pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(
            body, text="Uses the Source and Region (mask) below.", anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))
        self._refresh_anypaint_state()

        for var in (
            self.ref_lora_var, self.ref_strength_var, self.ref_guidance_var,
            self.ref_correction_var, self.ref_offset_var,
            self.ref_megapixels_var, self.anypaint_lora_var,
            self.anypaint_strength_var,
        ):
            var.trace_add("write", self._on_settings_change)

    # ------------------------------------------------------------------- family
    def family(self):
        return self._family

    def set_family(self, family):
        """Point the panel at the selected model's family and refresh its gates."""
        changed = family != self._family
        self._family = family
        self._refresh_reference_state()
        self._refresh_qwen_reference_state()
        self._refresh_anypaint_state()
        if changed:
            self._sync_section_expansion()

    def _sync_section_expansion(self):
        """Open the active family's section; fold the other two away.

        Only run when the family actually changes, so a section the user
        collapsed by hand stays collapsed across ordinary canvas switches.
        """
        active = {
            "anima": getattr(self, "_anima_section", None),
            "qwen_image": getattr(self, "_qwen_section", None),
            "h3_image": getattr(self, "_qwen_section", None),
            "krea2": getattr(self, "_anypaint_section", None),
        }.get(self._family)
        for section in (
            getattr(self, "_anima_section", None),
            getattr(self, "_qwen_section", None),
            getattr(self, "_anypaint_section", None),
        ):
            if section is not None:
                section.set_open(section is active)

    def _refresh_section_titles(self):
        """Show each section's contents on its header (useful when collapsed)."""
        if hasattr(self, "_anima_section"):
            self._anima_section.set_summary(
                str(len(self._anima_refs)) if self._anima_refs else ""
            )
        if hasattr(self, "_qwen_section"):
            self._qwen_section.set_summary(
                str(len(self._qwen_refs)) if self._qwen_refs else ""
            )
        if hasattr(self, "_anypaint_section"):
            self._anypaint_section.set_summary(
                "on" if self.anypaint_var.get() else ""
            )

    # ------------------------------------------------------------ reference rows
    def _add_ref_row(self, container, store, binding=None):
        """Append a reference slot row (kind picker, Assign, label, Remove)."""
        frame = tk.Frame(container)
        frame.pack(fill=tk.X, pady=(1, 0))
        token = (binding or {}).get("source")
        kind_var = tk.StringVar(value=_TOKEN_KINDS.get(token, SOURCE_ASSET))
        combo = ttk.Combobox(
            frame, textvariable=kind_var, values=list(REF_KINDS),
            state="readonly", width=13,
        )
        combo.pack(side=tk.LEFT)
        row = {
            "frame": frame, "kind_var": kind_var, "combo": combo,
            "label_var": tk.StringVar(value=(binding or {}).get("label", "")),
            "binding": dict(binding) if binding else None,
            "store": store,
        }
        assign = tk.Button(
            frame, text="Assign", width=6,
            command=lambda r=row: self._assign_row(r),
        )
        assign.pack(side=tk.LEFT, padx=(4, 0))
        row["assign"] = assign
        label = tk.Label(frame, textvariable=row["label_var"], anchor="w")
        label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))
        row["label"] = label
        remove = tk.Button(
            frame, text="\u2715", width=2,
            command=lambda r=row: self._remove_ref_row(r),
        )
        remove.pack(side=tk.RIGHT)
        row["remove"] = remove
        store.append(row)
        # Rows are built after the app's initial theme.apply, so each one themes
        # itself here - otherwise it would keep Tk's default light colors.
        theme.restyle(frame)
        kind_var.trace_add("write", lambda *_: self._kind_changed(row))
        return row

    def _assign_row(self, row):
        """Capture the current selection of the row's kind into the row."""
        if self._on_capture is None:
            return
        token = _KIND_TOKENS.get(row["kind_var"].get(), "asset")
        binding = self._on_capture(token)
        if not binding:
            return
        row["binding"] = dict(binding)
        row["label_var"].set(binding.get("label", ""))
        self._on_settings_change()

    def _kind_changed(self, row):
        """A new kind invalidates the captured image until Assign is pressed."""
        row["binding"] = None
        row["label_var"].set("")
        self._on_settings_change()

    def _remove_ref_row(self, row):
        store = row["store"]
        store[:] = [r for r in store if r is not row]
        row["frame"].destroy()
        self._refresh_reference_state()
        self._refresh_qwen_reference_state()
        self._on_settings_change()

    @staticmethod
    def _row_bindings(store):
        """The rows' bindings: the captured binding, or a live kind with no id."""
        out = []
        for row in store:
            binding = row["binding"]
            if binding is None:
                binding = {"source": _KIND_TOKENS.get(row["kind_var"].get(), "asset")}
            out.append(dict(binding))
        return out

    def _set_rows(self, store, container, entries):
        """Replace a section's rows with ``entries`` (bindings or legacy strings)."""
        for row in list(store):
            row["frame"].destroy()
        store[:] = []
        for entry in entries or []:
            if isinstance(entry, str):
                if entry == REF_OFF:
                    continue
                binding = {"source": _KIND_TOKENS.get(entry, "asset")}
            elif isinstance(entry, dict):
                binding = entry
            else:
                continue
            self._add_ref_row(container, store, binding=binding)

    # ------------------------------------------------------- anima reference
    def _refresh_reference_loras(self):
        """Populate the reference-adapter picker from the adapters folder."""
        selected = self.ref_lora_path()
        self._ref_lora_paths = {}
        names = ["(none)"]
        for path in engine_paths.list_adapters():
            display = _lora_name(path)
            if display in self._ref_lora_paths:
                display = f"{display} ({os.path.basename(os.path.dirname(path))})"
            self._ref_lora_paths[display] = path
            names.append(display)
        self.ref_lora_menu.configure(values=names)
        # Keep the current choice across a refresh (a new file added since the
        # panel was built should not clear the selection).
        if selected and selected in self._ref_lora_paths.values():
            self._set_ref_lora(selected)

    def reference_bindings(self):
        """The Anima reference rows' bindings (callers gate by family)."""
        return self._row_bindings(self._anima_refs)

    def reference_enabled(self):
        """True when an Anima reference row exists and the family supports it."""
        return self._family == "anima" and bool(self._anima_refs)

    def ref_lora_path(self):
        """The selected reference-adapter LoRA file path, or ``None``."""
        return self._ref_lora_paths.get(self.ref_lora_var.get())

    def _set_ref_lora(self, path):
        """Select the reference-adapter LoRA whose file path is ``path``."""
        if not path:
            self.ref_lora_var.set("(none)")
            return
        for display, candidate in self._ref_lora_paths.items():
            if candidate == path:
                self.ref_lora_var.set(display)
                return
        self.ref_lora_var.set("(none)")

    def _add_anima_reference(self, binding=None):
        """Append an Anima reference slot row."""
        self._add_ref_row(self.ref_rows, self._anima_refs, binding)
        self._refresh_reference_state()
        self._on_settings_change()

    def _refresh_reference_state(self):
        """Grey the reference controls out for families that do not support them."""
        supported = self._family == "anima"
        self.ref_add_btn.configure(state=tk.NORMAL if supported else tk.DISABLED)
        assignable = supported and self._on_capture is not None
        for row in self._anima_refs:
            row["combo"].configure(state="readonly" if supported else "disabled")
            row["assign"].configure(state=tk.NORMAL if assignable else tk.DISABLED)
            row["remove"].configure(state=tk.NORMAL if supported else tk.DISABLED)
        active = supported and bool(self._anima_refs)
        for widget in (self.ref_lora_menu, self.ref_correction_menu):
            widget.configure(state="readonly" if active else "disabled")
        for widget in (self.ref_strength_spin, self.ref_guidance_spin,
                       self.ref_offset_spin, self.ref_megapixels_spin):
            widget.configure(state=tk.NORMAL if active else tk.DISABLED)
        self._refresh_section_titles()

    # -------------------------------------------------------- qwen references
    def _add_qwen_reference(self, binding=None):
        """Append a Qwen reference slot row."""
        self._add_ref_row(self.qref_rows, self._qwen_refs, binding)
        self._refresh_qwen_reference_state()
        self._on_settings_change()

    def qwen_reference_bindings(self):
        """The ordered Qwen reference bindings (callers gate by family)."""
        return self._row_bindings(self._qwen_refs)

    def _refresh_qwen_reference_state(self):
        """Grey the reference controls out for families that do not support them."""
        supported = self._family in ("qwen_image", "h3_image")
        assignable = supported and self._on_capture is not None
        self.qref_add_btn.configure(state=tk.NORMAL if supported else tk.DISABLED)
        for row in self._qwen_refs:
            row["combo"].configure(state="readonly" if supported else "disabled")
            row["assign"].configure(state=tk.NORMAL if assignable else tk.DISABLED)
            row["remove"].configure(state=tk.NORMAL if supported else tk.DISABLED)
        self._refresh_section_titles()

    # ----------------------------------------------------- krea 2 anypaint
    def _refresh_anypaint_loras(self):
        """Populate the AnyPaint adapter picker from the adapters folder."""
        selected = self.anypaint_lora_path()
        self._anypaint_lora_paths = {}
        names = ["(none)"]
        for path in engine_paths.list_adapters():
            display = _lora_name(path)
            if display in self._anypaint_lora_paths:
                display = f"{display} ({os.path.basename(os.path.dirname(path))})"
            self._anypaint_lora_paths[display] = path
            names.append(display)
        self.anypaint_lora_menu.configure(values=names)
        if selected and selected in self._anypaint_lora_paths.values():
            self._set_anypaint_lora(selected)

    def anypaint_enabled(self):
        """True when AnyPaint is on and the family supports it."""
        return self._family == "krea2" and bool(self.anypaint_var.get())

    def anypaint_lora_path(self):
        """The selected AnyPaint adapter LoRA file path, or ``None``."""
        return self._anypaint_lora_paths.get(self.anypaint_lora_var.get())

    def _set_anypaint_lora(self, path):
        """Select the AnyPaint adapter whose file path is ``path``."""
        if not path:
            self.anypaint_lora_var.set("(none)")
            return
        for display, candidate in self._anypaint_lora_paths.items():
            if candidate == path:
                self.anypaint_lora_var.set(display)
                return
        self.anypaint_lora_var.set("(none)")

    def _refresh_anypaint_state(self):
        """Grey the AnyPaint controls out for families that do not support them."""
        supported = self._family == "krea2"
        self.anypaint_check.configure(state=tk.NORMAL if supported else tk.DISABLED)
        active = supported and bool(self.anypaint_var.get())
        self.anypaint_lora_menu.configure(state="readonly" if active else "disabled")
        self.anypaint_strength_spin.configure(state=tk.NORMAL if active else tk.DISABLED)
        self._refresh_section_titles()

    def _on_anypaint_toggle(self):
        self._refresh_anypaint_state()
        self._on_settings_change()

    # ------------------------------------------------------------- settings
    def settings(self):
        """The panel's fields as a dict for per-canvas/layer storage."""
        try:
            return {
                "ref_refs": self.reference_bindings(),
                "ref_lora": self.ref_lora_path(),
                "ref_strength": float(self.ref_strength_var.get()),
                "ref_guidance": float(self.ref_guidance_var.get()),
                "ref_correction": self.ref_correction_var.get(),
                "ref_offset": float(self.ref_offset_var.get()),
                "ref_megapixels": float(self.ref_megapixels_var.get()),
                "qwen_refs": self.qwen_reference_bindings(),
                "anypaint": bool(self.anypaint_var.get()),
                "anypaint_lora": self.anypaint_lora_path(),
                "anypaint_strength": float(self.anypaint_strength_var.get()),
            }
        except (tk.TclError, TypeError, ValueError):
            return None

    def set_settings(self, values):
        """Load stored fields without echoing a change back into storage."""
        if not values:
            return
        self._loading = True
        try:
            if "ref_refs" in values:
                self._set_rows(self._anima_refs, self.ref_rows, values["ref_refs"])
            elif "ref_source" in values:
                # Legacy single-source setting: one row bound to the live kind.
                source = values["ref_source"]
                self._set_rows(
                    self._anima_refs, self.ref_rows,
                    [] if source == REF_OFF else [source],
                )
            if "ref_lora" in values:
                self._set_ref_lora(values["ref_lora"])
            if "ref_strength" in values:
                self.ref_strength_var.set(float(values["ref_strength"]))
            if "ref_guidance" in values:
                self.ref_guidance_var.set(float(values["ref_guidance"]))
            if "ref_correction" in values:
                correction = values["ref_correction"]
                self.ref_correction_var.set(
                    correction if correction in REF_CORRECTIONS else "apg"
                )
            if "ref_offset" in values:
                self.ref_offset_var.set(float(values["ref_offset"]))
            if "ref_megapixels" in values:
                self.ref_megapixels_var.set(float(values["ref_megapixels"]))
            if "qwen_refs" in values:
                self._set_rows(self._qwen_refs, self.qref_rows, values["qwen_refs"])
            if "anypaint" in values:
                self.anypaint_var.set(bool(values["anypaint"]))
            if "anypaint_lora" in values:
                self._set_anypaint_lora(values["anypaint_lora"])
            if "anypaint_strength" in values:
                self.anypaint_strength_var.set(float(values["anypaint_strength"]))
            self._refresh_reference_state()
            self._refresh_qwen_reference_state()
            self._refresh_anypaint_state()
        finally:
            self._loading = False

    def reset(self):
        """Return the panel to its documented defaults (no stored settings)."""
        self._loading = True
        try:
            self._set_rows(self._anima_refs, self.ref_rows, [])
            self.ref_lora_var.set("(none)")
            self.ref_strength_var.set(1.0)
            self.ref_guidance_var.set(4.5)
            self.ref_correction_var.set("apg")
            self.ref_offset_var.set(1.0)
            self.ref_megapixels_var.set(1.0)
            self._set_rows(self._qwen_refs, self.qref_rows, [])
            self.anypaint_var.set(False)
            self.anypaint_lora_var.set("(none)")
            self.anypaint_strength_var.set(1.0)
        finally:
            self._loading = False

    def apply_config(self, config):
        """Fold the reference fields into a run ``config`` for the current family."""
        if self._family == "anima" and self.reference_bindings():
            config["anima_ref_lora"] = self.ref_lora_path()
            config["anima_ref_strength"] = float(self.ref_strength_var.get())
            config["ref_guidance"] = float(self.ref_guidance_var.get())
            config["ref_correction"] = self.ref_correction_var.get()
            config["ref_offset_h"] = float(self.ref_offset_var.get())
            config["ref_megapixels"] = float(self.ref_megapixels_var.get())
        elif self._family in ("qwen_image", "h3_image") and self.qwen_reference_bindings():
            config["qwen_refs"] = self.qwen_reference_bindings()
        if self._family == "krea2" and self.anypaint_var.get():
            config["anypaint"] = True
            config["anypaint_lora"] = self.anypaint_lora_path()
            config["anypaint_strength"] = float(self.anypaint_strength_var.get())
        return config

    def _on_settings_change(self, *_):
        """Hand the current fields to ``on_settings`` unless it is echoing."""
        if self._loading or self._on_settings is None:
            return
        values = self.settings()
        if values is None:
            return
        self._on_settings(values)
