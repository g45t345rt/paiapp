"""The ControlNet panel: a stack of control adapters for a generation run.

Unlike the other generation fields, control adapters are not one-per-family.
A run may combine several: two Anima ControlNet-LLLite adapters (for example a
3D-pose adapter plus a regional-control adapter), or several SD ControlNets that
each contribute their own residuals. The panel is therefore a list of rows, and
each row carries everything one adapter needs: the model, where its control
image comes from, the control type (preprocessor), the strength and the step
range (start/end fraction) over which it applies.

The panel is a form only: it reports its rows through ``on_settings`` and the
coordinator resolves each row's hint image and folds the rows into the run
config (see :func:`~engine.pipeline.generate`).
"""

import os
import tkinter as tk
from tkinter import ttk

from engine import component_kinds as engine_kinds
from engine import paths as engine_paths
from engine import preprocess as engine_preprocess

from . import theme

# A row's "no adapter" value, used by the model dropdown.
NONE = "(none)"

# Where a row reads its control image from. ``Source`` reuses the run's own
# Source image, which is the historical single-ControlNet behaviour; the others
# let different adapters steer from different layers or assets (the 3D-pose
# mesh and the regional colour map, for instance).
HINT_SOURCE = "Source"
HINT_CANVAS = "Canvas"
HINT_LAYER = "Selected layer"
HINT_ASSET = "Selected asset"
HINT_CHOICES = (HINT_SOURCE, HINT_CANVAS, HINT_LAYER, HINT_ASSET)

# Families whose control adapters the panel can offer, mapped to the ``kind``
# the engine expects in a config entry.
FAMILY_KIND = {
    "stable_diffusion": "controlnet",
    "z_image": "controlnet",
    "anima": "lllite",
}

# Families whose control branch is a single model, so only one row is used.
SINGLE_ADAPTER_FAMILIES = {"z_image"}


# The control-adapter kinds (see :mod:`engine.component_kinds`) each family's
# dropdown offers. Every family's control adapters share the one ``controlnet``
# pool, so the detected kind decides which are listed -- Anima's
# ControlNet-LLLite adapters, or the classic ControlNets for Stable Diffusion
# and Z-Image -- and a selection cannot cross families.
def _control_kinds(family):
    if family == "anima":
        return ("lllite",)
    if family in ("stable_diffusion", "z_image"):
        return ("controlnet_sd15", "controlnet_sdxl", "controlnet_zimage")
    return ()


def _catalog(family):
    """Control adapter display label -> file path for ``family``.

    The family's control adapters all live in :func:`~engine.paths.controlnet_dir`;
    the detected kind (see :mod:`engine.component_kinds`) decides which the
    dropdown offers. A file whose kind cannot be read is still offered, matching
    the old per-folder listing that showed every file unfiltered.
    """
    wanted = _control_kinds(family)
    if not wanted:
        return {}
    choices = {}
    for path in engine_paths.list_controlnets():
        kind = engine_kinds.detect(path, engine_kinds.CONTROL)
        if kind is not None and kind not in wanted:
            continue
        stem = os.path.splitext(os.path.basename(path))[0]
        display = stem if stem not in choices else path
        choices[display] = path
    return choices


class ControlNetPanel(tk.Frame):
    """A stack of control-adapter rows for the selected model's family.

    The family is pushed in by the coordinator with :meth:`set_family` (it owns
    the model picker), so the panel can offer the right model list and nothing
    else. For an unsupported family the stack is emptied and adding is disabled.
    """

    def __init__(self, parent, on_settings=None):
        super().__init__(parent)
        # Called with the row list whenever a row changes, so the coordinator can
        # remember the stack on the active canvas and layer.
        self._on_settings = on_settings
        self._family = None
        self._rows = []
        # Guards programmatic writes (set_settings/rebuilds) from echoing back.
        self._loading = False
        self._build()

    def _build(self):
        self.rows_frame = tk.Frame(self)
        self.rows_frame.pack(fill=tk.X)

        bar = tk.Frame(self)
        bar.pack(fill=tk.X, pady=(4, 6))
        # A master switch: with it off the run ignores the stack entirely, so it
        # can be parked without clearing the rows or the model settings.
        self.enabled_var = tk.BooleanVar(value=False)
        self.enabled_check = tk.Checkbutton(
            bar, text="Enable", variable=self.enabled_var,
            command=self._on_enabled_toggle,
        )
        self.enabled_check.pack(side=tk.LEFT)
        self.add_btn = tk.Button(bar, text="Add controlnet", command=self.add_row)
        self.add_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.hint_label = tk.Label(bar, text="", anchor="w")
        self.hint_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        theme.register(self.hint_label, fg="muted")
        # The panel is built before the app's first theme pass at startup, but
        # restyling here keeps it correct if it is ever built later (and matches
        # the dynamically added rows).
        theme.restyle(self)
        self._update_hint()

    # ------------------------------------------------------------------ family
    def family(self):
        return self._family

    def supported(self):
        return self._family in FAMILY_KIND

    def enabled(self):
        """Whether the run should fold the stack into its config."""
        return bool(self.enabled_var.get())

    def set_enabled(self, enabled):
        """Set the master switch without echoing a change (used when loading)."""
        self.enabled_var.set(bool(enabled))
        self._refresh_add_state()
        self._update_hint()

    def _on_enabled_toggle(self):
        self._refresh_add_state()
        self._update_hint()
        self._changed()

    def set_family(self, family):
        """Point the panel at ``family``, rescaling the rows' model lists."""
        if family == self._family:
            return
        self._family = family
        self._loading = True
        try:
            for row in self._rows:
                self._rebuild_row_catalog(row)
            if family in SINGLE_ADAPTER_FAMILIES:
                # Keep only the first row; the family has a single control branch.
                while len(self._rows) > 1:
                    self._remove_row(self._rows[-1], notify=False)
        finally:
            self._loading = False
        self._refresh_add_state()
        self._update_hint()
        self._changed()

    # -------------------------------------------------------------------- rows
    def add_row(self, model=None, hint=None, preprocessor=None, strength=1.0,
                start=0.0, end=1.0):
        """Append an adapter row, optionally seeded from stored settings."""
        if not self.supported():
            return
        if self._family in SINGLE_ADAPTER_FAMILIES and self._rows:
            return
        frame = tk.Frame(self.rows_frame)
        frame.pack(fill=tk.X, pady=(2, 4))
        frame.columnconfigure(1, weight=1)

        row = {"frame": frame}

        def caption(index, text, pady=(3, 0)):
            label = tk.Label(frame, text=text, anchor="w")
            label.grid(row=index, column=0, sticky="w", pady=pady)
            return label

        # Line 1: the adapter model and a remove button.
        caption(0, "Model:", pady=(0, 0))
        model_menu = ttk.Combobox(frame, state="readonly", width=16)
        model_menu.grid(row=0, column=1, sticky="ew", padx=(4, 4))
        model_menu.bind("<<ComboboxSelected>>", lambda _e: self._changed())
        remove = tk.Button(frame, text="\u2715", width=2, command=lambda: self._remove_row(row))
        remove.grid(row=0, column=2, sticky="e")
        row["model_menu"] = model_menu

        # Line 2: where the control image comes from.
        caption(1, "Image:")
        hint_menu = ttk.Combobox(
            frame, textvariable=tk.StringVar(), values=list(HINT_CHOICES),
            state="readonly", width=12,
        )
        hint_menu.grid(row=1, column=1, columnspan=2, sticky="ew", padx=(4, 0), pady=(3, 0))
        row["hint_menu"] = hint_menu

        # Line 3: the control type (preprocessor applied to the control image).
        caption(2, "Preprocess:")
        type_menu = ttk.Combobox(
            frame, textvariable=tk.StringVar(),
            values=list(engine_preprocess.PREPROCESSORS), state="readonly", width=8,
        )
        type_menu.grid(row=2, column=1, columnspan=2, sticky="ew", padx=(4, 0), pady=(3, 0))
        row["type_menu"] = type_menu

        # Line 4: how strongly the adapter steers the run.
        caption(3, "Strength:")
        strength_spin = tk.Spinbox(
            frame, from_=0.0, to=2.0, increment=0.05, width=5,
        )
        strength_spin.grid(row=3, column=1, columnspan=2, sticky="ew", padx=(4, 0), pady=(3, 0))
        row["strength_spin"] = strength_spin

        # Line 5: the step range (as start/end fractions) the adapter applies over.
        caption(4, "Steps:")
        step_frame = tk.Frame(frame)
        step_frame.grid(row=4, column=1, columnspan=2, sticky="ew", padx=(4, 0), pady=(3, 0))
        step_frame.columnconfigure(0, weight=1)
        step_frame.columnconfigure(2, weight=1)
        start_spin = tk.Spinbox(step_frame, from_=0.0, to=1.0, increment=0.05, width=5)
        start_spin.grid(row=0, column=0, sticky="ew")
        tk.Label(step_frame, text="to", anchor="center").grid(row=0, column=1, padx=4)
        end_spin = tk.Spinbox(step_frame, from_=0.0, to=1.0, increment=0.05, width=5)
        end_spin.grid(row=0, column=2, sticky="ew")
        row["start_spin"] = start_spin
        row["end_spin"] = end_spin

        # A hairline under every row separates the several adapters a run can
        # stack; the line rides on the row, so removing one keeps it right.
        divider = tk.Frame(frame, height=1, bd=0, highlightthickness=0)
        divider.grid(row=5, column=0, columnspan=3, sticky="we", pady=(6, 0))
        theme.register(divider, bg="border")

        row["model_var"] = tk.StringVar()
        row["hint_var"] = tk.StringVar(value=hint or HINT_SOURCE)
        row["type_var"] = tk.StringVar(
            value=preprocessor or engine_preprocess.PREPROCESSORS[0]
        )
        row["strength_var"] = tk.DoubleVar(value=float(strength))
        row["start_var"] = tk.DoubleVar(value=float(start))
        row["end_var"] = tk.DoubleVar(value=float(end))

        model_menu.configure(textvariable=row["model_var"])
        hint_menu.configure(textvariable=row["hint_var"])
        type_menu.configure(textvariable=row["type_var"])
        strength_spin.configure(textvariable=row["strength_var"])
        start_spin.configure(textvariable=row["start_var"])
        end_spin.configure(textvariable=row["end_var"])

        for var in (
            row["model_var"], row["hint_var"], row["type_var"],
            row["strength_var"], row["start_var"], row["end_var"],
        ):
            var.trace_add("write", lambda *_: self._changed())

        # Rows are built after the app's initial theme pass, so style them here
        # or they keep the light default chrome in dark mode.
        theme.restyle(frame)
        self._rows.append(row)
        self._rebuild_row_catalog(row, preferred=model)
        self._refresh_add_state()
        if model is None:
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
        self._refresh_add_state()
        if notify:
            self._changed()

    def _rebuild_row_catalog(self, row, preferred=None):
        """Refresh a row's model list for the current family, keeping its pick."""
        choices = _catalog(self._family)
        row["choices"] = choices
        previous = preferred if preferred is not None else row["model_var"].get()
        row["model_menu"].configure(values=[NONE] + list(choices))
        if previous in choices:
            row["model_var"].set(previous)
        else:
            row["model_var"].set(NONE)

    def _refresh_add_state(self):
        supported = self.supported()
        enabled = supported and self.enabled()
        self.enabled_check.config(state=tk.NORMAL if supported else tk.DISABLED)
        allows_more = enabled and not (
            self._family in SINGLE_ADAPTER_FAMILIES and self._rows
        )
        self.add_btn.config(state=tk.NORMAL if allows_more else tk.DISABLED)
        for row in self._rows:
            for key in ("model_menu", "hint_menu", "type_menu"):
                row[key].config(state="readonly" if enabled else "disabled")
            for key in ("strength_spin", "start_spin", "end_spin"):
                row[key].config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _update_hint(self):
        if not self.supported():
            text = "ControlNet is not available for this model."
        elif not self.enabled():
            text = "ControlNet is off."
        elif self._family in SINGLE_ADAPTER_FAMILIES:
            text = "This model uses a single control adapter."
        elif self._rows:
            text = "Applied top to bottom."
        else:
            text = "No control adapters."
        self.hint_label.config(text=text)

    # ---------------------------------------------------------------- settings
    def settings(self):
        """The rows as a list of dicts for storage (skips empty rows)."""
        rows = []
        for row in self._rows:
            model = (row["choices"] or {}).get(row["model_var"].get())
            if not model:
                continue
            rows.append({
                "kind": FAMILY_KIND.get(self._family),
                "model": model,
                "display": row["model_var"].get(),
                "hint": row["hint_var"].get(),
                "preprocessor": row["type_var"].get(),
                "strength": self._number(row["strength_var"], 1.0),
                "start": self._number(row["start_var"], 0.0),
                "end": self._number(row["end_var"], 1.0),
            })
        return rows

    def set_settings(self, rows):
        """Rebuild the stack from stored rows without echoing a change."""
        if rows is None:
            return
        self._loading = True
        try:
            for row in list(self._rows):
                self._remove_row(row, notify=False)
            for item in rows:
                if not isinstance(item, dict) or not item.get("model"):
                    continue
                self.add_row(
                    model=item.get("display") or item.get("model"),
                    hint=item.get("hint"),
                    preprocessor=item.get("preprocessor"),
                    strength=item.get("strength", 1.0),
                    start=item.get("start", 0.0),
                    end=item.get("end", 1.0),
                )
        finally:
            self._loading = False
        self._update_hint()

    def refresh_models(self):
        """Rescan the family's control folder after the model settings change."""
        for row in self._rows:
            self._rebuild_row_catalog(row)

    @staticmethod
    def _number(var, fallback):
        try:
            return float(var.get())
        except tk.TclError:
            return fallback

    def _changed(self):
        if self._loading or self._on_settings is None:
            return
        self._update_hint()
        self._on_settings(self.settings())
