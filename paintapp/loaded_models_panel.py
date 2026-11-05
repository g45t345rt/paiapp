"""The Loaded Models panel.

Shows what the engine is holding in memory: the diffusion model, text encoders,
VAE and other components cached by the last generation, plus any background or
upscale model kept from earlier jobs. The panel reads the engine's caches, which
are published as finished objects, so refreshing while a job runs is safe; each
model has its own Unload button that drops just that cache, and the Unload All
button at the bottom drops every cache (the app refuses either while a job is
running). It is tabbed with Generate by default (see
:data:`paintapp.constants.DEFAULT_PANEL_GROUPS`).
"""

import os
import tkinter as tk

from . import theme

# Sidebar wrap width for the model/source labels, matching the other panels.
WRAP = 240

# Torch dtype names shortened for the narrow sidebar.
_DTYPES = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "float64": "fp64",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8",
    "float8_e4m3fnuz": "fp8",
}


def _records():
    """The engine's loaded-model records, or ``[]`` when none are available."""
    try:
        from engine import loaded_models
        return loaded_models.loaded()
    except Exception:
        return []


def _fmt_bytes(count):
    """A human-readable size such as ``4.2 GB``."""
    size = float(count or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _models_word(count):
    """``1 model`` / ``2 models``."""
    return f"{count} model{'s' if count != 1 else ''}"


def _basename(path):
    """The file name of a LoRA path, or ``""`` when it is unset."""
    return os.path.basename(path) if path else ""


def _detail(component):
    """The muted ``size . dtype . device`` tail shown after a component name."""
    parts = [_fmt_bytes(component.get("bytes"))]
    dtype = component.get("dtype")
    if dtype:
        parts.append(_DTYPES.get(dtype, dtype))
    device = component.get("device")
    if device:
        parts.append("GPU" if device.startswith("cuda") else "CPU")
    return " \u00b7 ".join(parts)


class LoadedModelsPanel(tk.Frame):
    """Lists the engine components currently cached in memory."""

    def __init__(self, parent, on_unload=None):
        super().__init__(parent)
        self._on_unload = on_unload
        self._build()
        self.refresh()

    def _build(self):
        self.load_status = tk.Label(self, anchor="w", justify="left", wraplength=WRAP)
        theme.register(self.load_status, fg="muted")
        # Packed on demand by ``set_load_status`` (hidden until a load starts).

        self.summary = tk.Label(self, anchor="w", justify="left", wraplength=WRAP)
        self.summary.pack(fill=tk.X, pady=(2, 2))
        theme.register(self.summary, fg="muted")
        self._summary_shown = True

        self.rows = tk.Frame(self)
        self.rows.pack(fill=tk.X)

        self.empty = tk.Label(
            self, text="Nothing loaded yet.", anchor="w", justify="left",
            wraplength=WRAP,
        )
        self.empty.pack(fill=tk.X, pady=(1, 0))
        theme.register(self.empty, fg="muted")
        self._empty_shown = True

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(6, 4))
        self._buttons = buttons
        self.refresh_btn = tk.Button(buttons, text="Refresh", command=self.refresh)
        self.refresh_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.unload_btn = tk.Button(
            buttons, text="Unload All", command=self._unload, state=tk.DISABLED,
        )
        self.unload_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

    # ------------------------------------------------------------------ public
    def set_load_status(self, text, token="muted"):
        """Show a load-status line above the list; ``None``/empty hides it."""
        if not text:
            self.load_status.pack_forget()
            return
        self.load_status.config(text=text)
        theme.register(self.load_status, fg=token)
        if not self.load_status.winfo_manager():
            # Keep the status line above the summary, or above the list when the
            # summary is hidden (nothing loaded).
            anchor = self.summary if self._summary_shown else self.rows
            self.load_status.pack(fill=tk.X, pady=(2, 0), before=anchor)
        theme.restyle(self.load_status)

    def refresh(self):
        """Rebuild the list from the engine's current caches."""
        for child in self.rows.winfo_children():
            child.destroy()

        records = _records()
        if not records:
            # Tk leaves a container's requested size stale when its last packed
            # child is destroyed, so the emptied list would keep the height of
            # the rows it just lost (a large gap above the buttons after an
            # unload). Nudging the frame's height makes the geometry manager
            # recompute it back down to empty.
            self.rows.configure(height=1)
            self._hide_summary()
            self._show_empty()
        else:
            self._hide_empty()
            total = sum(
                component.get("bytes", 0)
                for record in records
                for component in record["components"]
            ) + sum(
                lora.get("bytes", 0)
                for record in records
                for lora in record.get("loras", ())
            )
            components = sum(len(record["components"]) for record in records)
            loras = sum(len(record.get("loras", ())) for record in records)
            models = len(records)
            parts = [
                _models_word(models),
                f"{components} component{'s' if components != 1 else ''}",
            ]
            if loras:
                parts.append(f"{loras} LoRA{'s' if loras != 1 else ''}")
            parts.append(_fmt_bytes(total))
            self._show_summary(" \u00b7 ".join(parts))
            for record in records:
                self._add_record(record)

        self.unload_btn.config(state=tk.NORMAL if records else tk.DISABLED)

        # The rows are built after the initial theme.apply, so restyle them here
        # (like JobRow does); otherwise they keep Tk's default light colors.
        theme.restyle(self.rows)

    # --------------------------------------------------------------- internals
    def _unload(self, key=None, label=None):
        if self._on_unload is not None:
            self._on_unload(key, label)
            return
        try:
            from engine import loaded_models
            dropped = loaded_models.unload(key)
        except Exception as exc:
            self.set_load_status(f"Unload failed: {exc}", "error")
            return
        self.refresh()
        if dropped:
            self.set_load_status(f"Unloaded {label or _models_word(dropped)}.", "muted")
        else:
            self.set_load_status("Nothing to unload.", "warn")

    def _add_record(self, record):
        header = tk.Frame(self.rows)
        header.pack(fill=tk.X, pady=(4, 0))
        title = tk.Label(
            header, text=record["family"], anchor="w", justify="left",
            wraplength=WRAP,
        )
        title.pack(side=tk.LEFT, fill=tk.X, expand=True)
        theme.register(title, fg="accent")

        key = record.get("key")
        if key:
            button = tk.Button(
                header, text="Unload",
                command=lambda: self._unload(key, record["family"]),
            )
            button.pack(side=tk.RIGHT, padx=(4, 0))

        if record.get("source"):
            source = tk.Label(
                self.rows, text=record["source"], anchor="w", justify="left",
                wraplength=WRAP,
            )
            source.pack(fill=tk.X)
            theme.register(source, fg="muted")

        for component in record["components"]:
            row = tk.Label(
                self.rows,
                text=f"{component['name']} \u2014 {_detail(component)}",
                anchor="w", justify="left", wraplength=WRAP,
            )
            row.pack(fill=tk.X, padx=(10, 0))
            theme.register(row, fg="muted")

        for lora in record.get("loras", ()):
            strength = lora.get("strength", 1.0)
            scale = "" if strength == 1.0 else f" \u00d7{strength:g}"
            layers = lora.get("layers", 0)
            row = tk.Label(
                self.rows,
                text=f"LoRA {_basename(lora.get('source'))}{scale} \u2014 "
                     f"{_fmt_bytes(lora.get('bytes'))} \u00b7 "
                     f"{layers} layer{'s' if layers != 1 else ''}",
                anchor="w", justify="left", wraplength=WRAP,
            )
            row.pack(fill=tk.X, padx=(10, 0))
            theme.register(row, fg="muted")

    def _hide_empty(self):
        if self._empty_shown:
            self.empty.pack_forget()
            self._empty_shown = False

    def _show_empty(self):
        if not self._empty_shown:
            self.empty.pack(fill=tk.X, pady=(1, 0), before=self._buttons)
            self._empty_shown = True

    def _hide_summary(self):
        """Drop the summary line entirely (an empty label would still reserve
        a text row, leaving a blank gap above "Nothing loaded yet.")."""
        if self._summary_shown:
            self.summary.pack_forget()
            self._summary_shown = False

    def _show_summary(self, text):
        self.summary.config(text=text)
        if not self._summary_shown:
            self.summary.pack(fill=tk.X, pady=(2, 2), before=self.rows)
            self._summary_shown = True
