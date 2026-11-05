"""The upscale panel."""

import tkinter as tk
from tkinter import ttk


def _list_models():
    """Local upscale models, or ``[]`` when the engine cannot be imported.

    :mod:`engine.upscale` imports torch at module level, but this panel is built
    with the rest of the UI at startup. Importing it lazily keeps a core-only
    install (no torch) able to open the window, with the panel disabled.
    """
    try:
        from engine import upscale as engine_upscale
        return list(engine_upscale.list_models())
    except Exception:
        return []


class UpscalePanel(tk.Frame):
    """Picks a local upscale model and triggers the upscale.

    Like :class:`~paintapp.removal.BackgroundPanel` it is only a form: the
    coordinator owns the threading and applies the result to the active layer.
    """

    def __init__(self, parent, on_upscale, on_cancel=None):
        super().__init__(parent)
        self._on_upscale = on_upscale
        self._on_cancel = on_cancel if on_cancel is not None else (lambda: None)
        self._build()

    def _build(self):
        choices = _list_models()
        self.upscale_model_choices = {name: name for name in choices}

        row = tk.Frame(self)
        row.pack(fill=tk.X, pady=(4, 0))
        tk.Label(row, text="Model:").pack(side=tk.LEFT)
        self.upscale_model_var = tk.StringVar(value=choices[0] if choices else "")
        self.upscale_menu = ttk.Combobox(
            row, textvariable=self.upscale_model_var,
            values=list(choices) or ["(no model found)"], state="readonly", width=18,
        )
        self.upscale_menu.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.new_layer_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            self,
            text="Create new layer (keep original)",
            variable=self.new_layer_var,
            anchor="w",
        ).pack(fill=tk.X, pady=(6, 0))

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(6, 8))
        self.upscale_btn = tk.Button(buttons, text="Upscale", command=self._on_upscale)
        self.upscale_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cancel_btn = tk.Button(
            buttons, text="Cancel", command=self._on_cancel, state=tk.DISABLED
        )
        self.cancel_btn.pack(side=tk.LEFT, padx=(4, 0))
        if not choices:
            self.upscale_btn.config(state=tk.DISABLED)

    # ------------------------------------------------------------------ public
    def model(self):
        return self.upscale_model_var.get()

    def refresh_models(self):
        """Rebuild the model list after the model settings change."""
        choices = _list_models()
        self.upscale_model_choices = {name: name for name in choices}
        self.upscale_menu.configure(values=list(choices) or ["(no model found)"])
        if self.upscale_model_var.get() not in choices:
            self.upscale_model_var.set(choices[0] if choices else "")
        self.upscale_btn.config(state=tk.NORMAL if choices else tk.DISABLED)

    def new_layer(self):
        return bool(self.new_layer_var.get())

    def set_busy(self, busy, cancellable=False):
        """Reflect the job queue state for this panel.

        Upscale stays enabled so further upscales can be queued; Cancel lights
        up only while this panel has a job that can still be stopped.
        """
        if not self.upscale_model_choices:
            return
        self.cancel_btn.config(
            state=tk.NORMAL if (busy and cancellable) else tk.DISABLED
        )
