"""The background removal panel."""

import tkinter as tk
from tkinter import ttk

from engine import background as engine_background


class BackgroundPanel(tk.Frame):
    """Picks a local background-removal model and triggers the removal.

    Like :class:`~paintapp.generation.GenerationPanel` it is only a form: the
    coordinator owns the threading and applies the result to the active layer.
    """

    def __init__(self, parent, on_remove, on_cancel=None):
        super().__init__(parent)
        self._on_remove = on_remove
        self._on_cancel = on_cancel if on_cancel is not None else (lambda: None)
        self._build()

    def _build(self):
        choices = engine_background.list_models()
        self.bg_model_choices = {name: name for name in choices}

        row = tk.Frame(self)
        row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(row, text="Model:").pack(side=tk.LEFT)
        self.bg_model_var = tk.StringVar(value=choices[0] if choices else "")
        self.bg_menu = ttk.Combobox(
            row, textvariable=self.bg_model_var,
            values=list(choices) or ["(no model found)"], state="readonly", width=18,
        )
        self.bg_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.new_layer_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self,
            text="Create new layer (keep original)",
            variable=self.new_layer_var,
            anchor="w",
        ).pack(fill=tk.X, pady=(6, 0))

        # Restrict the cutout to the active selection, so a removal can be
        # limited to one region instead of clearing the whole layer.
        self.selection_only_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self,
            text="Apply only inside selection",
            variable=self.selection_only_var,
            anchor="w",
        ).pack(fill=tk.X, pady=(2, 0))

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(6, 8))
        self.remove_btn = tk.Button(buttons, text="Remove Background", command=self._on_remove)
        self.remove_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cancel_btn = tk.Button(
            buttons, text="Cancel", command=self._on_cancel, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=(4, 0))
        if not choices:
            self.remove_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ public
    def model(self):
        return self.bg_model_var.get()

    def refresh_models(self):
        """Rebuild the model list after the model settings change."""
        choices = engine_background.list_models()
        self.bg_model_choices = {name: name for name in choices}
        self.bg_menu.configure(values=list(choices) or ["(no model found)"])
        if self.bg_model_var.get() not in choices:
            self.bg_model_var.set(choices[0] if choices else "")
        self.remove_btn.config(state=tk.NORMAL if choices else tk.DISABLED)

    def new_layer(self):
        return bool(self.new_layer_var.get())

    def selection_only(self):
        """Whether the removal should be limited to the active selection."""
        return bool(self.selection_only_var.get())

    def set_busy(self, busy, cancellable=False):
        """Reflect the job queue state for this panel.

        Remove stays enabled so further removals can be queued; Cancel lights
        up only while this panel has a job that can still be stopped.
        """
        if not self.bg_model_choices:
            return
        self.cancel_btn.config(
            state=tk.NORMAL if (busy and cancellable) else tk.DISABLED
        )
