"""Modal dialogs that center themselves over the main window.

Tk's stock ``messagebox`` and ``simpledialog`` widgets are positioned by the
window manager, so they ignore the app's geometry. These wrappers build their
own ``Toplevel`` and center it over the main window so every dialog in the app
lines up the same way as the rename prompt.
"""

import tkinter as tk
from tkinter import filedialog, font as tkfont, ttk
import webbrowser

from . import colormanagement, fonts, theme
from .scrollbar import ArrowScrollbar, AutoScrollbar


# Returned by :func:`ask_string` when its optional delete button is chosen.
DELETED = object()


def center_window(root, window):
    """Center ``window`` over ``root`` and bring it to the front.

    Positions are computed in the virtual desktop's coordinate space so the
    dialog lands on the same monitor as the app. ``winfo_rootx``/``winfo_rooty``
    are negative for monitors left of or above the primary display, so clamp to
    the virtual root's bounds (``winfo_vroot*``) instead of ``0``. Negative
    positions must be written as ``+-N`` in a geometry string; a bare ``-N``
    would be read as an offset from the right/bottom edge. The window is
    withdrawn while the size is measured and only mapped once its geometry is
    set, so it never flashes at the window manager's default position first.
    """
    window.transient(root)
    window.withdraw()
    window.update_idletasks()
    width = window.winfo_reqwidth()
    height = window.winfo_reqheight()
    x = root.winfo_rootx() + (root.winfo_width() - width) // 2
    y = root.winfo_rooty() + (root.winfo_height() - height) // 2
    vx = root.winfo_vrootx()
    vy = root.winfo_vrooty()
    x = max(vx, min(x, vx + root.winfo_vrootwidth() - width))
    y = max(vy, min(y, vy + root.winfo_vrootheight() - height))
    window.geometry(f"{width}x{height}+{x}+{y}")
    window.deiconify()
    window.lift()
    window.focus_force()


def _make_window(root, title):
    window = tk.Toplevel(root)
    window.title(title)
    window.transient(root)
    window.resizable(False, False)
    window.withdraw()
    return window


def _run(root, window, on_cancel, focus=None):
    previous_grab = root.grab_current()
    window.protocol("WM_DELETE_WINDOW", on_cancel)
    window.bind("<Escape>", lambda _e: on_cancel())
    theme.restyle(window)
    center_window(root, window)
    if focus is not None:
        focus.focus_set()
    window.grab_set()
    root.wait_window(window)
    if previous_grab is not None and previous_grab.winfo_exists():
        previous_grab.grab_set()


def ask_string(root, title, prompt, initialvalue="", delete_label=None):
    """A centered single-line prompt.

    Returns the entered text, ``None`` when cancelled, or ``DELETED`` when
    ``delete_label`` is given and its button is chosen.
    """
    window = _make_window(root, title)
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text=prompt, anchor="w").grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 4)
    )
    value_var = tk.StringVar(value=initialvalue)
    entry = tk.Entry(frame, textvariable=value_var, width=28)
    entry.grid(row=1, column=0, columnspan=2, sticky="we")

    result = {"value": None}

    def on_ok(_event=None):
        result["value"] = value_var.get()
        window.destroy()

    def on_delete():
        result["value"] = DELETED
        window.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(row=2, column=0, columnspan=2, sticky="e", pady=(8, 0))
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )
    if delete_label:
        tk.Button(buttons, text=delete_label, width=8, command=on_delete).pack(
            side=tk.LEFT
        )

    window.bind("<Return>", on_ok)
    entry.selection_range(0, tk.END)
    _run(root, window, window.destroy, focus=entry)
    return result["value"]


def ask_transform(root, initial=None):
    """A centered prompt for arbitrary layer transforms.

    Returns a dict with ``scale_x``/``scale_y`` (fractions, ``1.0`` = 100%),
    ``angle`` (degrees clockwise), and ``flip_h``/``flip_v`` (booleans), or
    ``None`` when cancelled. Values entered as percentages are converted here so
    callers work in plain pixel-space fractions.
    """
    initial = initial or {}
    window = _make_window(root, "Transform Layer")
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    fields = [
        ("Scale X (%)", "scale_x", initial.get("scale_x", "100")),
        ("Scale Y (%)", "scale_y", initial.get("scale_y", "100")),
        ("Rotate (° CW)", "angle", initial.get("angle", "0")),
    ]
    entries = {}
    first_entry = None
    for row, (label, key, value) in enumerate(fields):
        tk.Label(frame, text=label, anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=2
        )
        var = tk.StringVar(value=str(value))
        entry = tk.Entry(frame, textvariable=var, width=12)
        entry.grid(row=row, column=1, sticky="we", pady=2)
        entries[key] = var
        if first_entry is None:
            first_entry = entry

    flip_h_var = tk.BooleanVar(value=bool(initial.get("flip_h", False)))
    flip_v_var = tk.BooleanVar(value=bool(initial.get("flip_v", False)))
    flips = tk.Frame(frame)
    flips.grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(4, 0))
    tk.Checkbutton(flips, text="Flip horizontal", variable=flip_h_var).pack(
        side=tk.LEFT
    )
    tk.Checkbutton(flips, text="Flip vertical", variable=flip_v_var).pack(
        side=tk.LEFT, padx=(6, 0)
    )

    error_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=error_var, fg="red", anchor="w").grid(
        row=len(fields) + 1, column=0, columnspan=2, sticky="w", pady=(2, 0)
    )

    result = {"value": None}

    def on_ok(_event=None):
        try:
            scale_x = float(entries["scale_x"].get()) / 100.0
            scale_y = float(entries["scale_y"].get()) / 100.0
            angle = float(entries["angle"].get())
        except ValueError:
            error_var.set("Enter numbers for scale and rotation.")
            return
        if scale_x <= 0 or scale_y <= 0:
            error_var.set("Scale must be greater than 0%.")
            return
        result["value"] = {
            "scale_x": scale_x,
            "scale_y": scale_y,
            "angle": angle,
            "flip_h": flip_h_var.get(),
            "flip_v": flip_v_var.get(),
        }
        window.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(
        row=len(fields) + 2, column=0, columnspan=2, sticky="e", pady=(8, 0)
    )
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    window.bind("<Return>", on_ok)
    if first_entry is not None:
        first_entry.selection_range(0, tk.END)
    _run(root, window, window.destroy, focus=first_entry)
    return result["value"]


def _message_dialog(root, title, message, buttons, note=None, checkbox=None):
    """Show a centered message box.

    ``buttons`` is ``[(label, value), ...]``. ``note`` is an optional muted
    caption shown under the button row, for extra context that should not
    widen the buttons themselves. ``checkbox`` is an optional
    ``(label, default)`` drawn above the buttons; when supplied the result is
    ``(value, checked)`` instead of just ``value``.
    """
    window = _make_window(root, title)
    frame = tk.Frame(window, padx=16, pady=16)
    frame.pack(fill=tk.BOTH, expand=True)
    tk.Label(frame, text=message, anchor="w", justify="left", wraplength=320).pack(
        anchor="w"
    )

    check_var = None
    if checkbox is not None:
        check_label, check_default = checkbox
        check_var = tk.BooleanVar(value=bool(check_default))
        tk.Checkbutton(
            frame, text=check_label, variable=check_var, anchor="w"
        ).pack(anchor="w", pady=(12, 0))

    result = {"value": None}

    def choose(value):
        result["value"] = value
        window.destroy()

    row = tk.Frame(frame)
    row.pack(anchor="e", pady=(12, 0))
    # Buttons share one width so the row stays even; never narrower than the
    # 8-character default, but wide enough that a long label is not clipped.
    width = max(8, max(len(label) for label, _ in buttons))
    default = None
    for label, value in buttons:
        button = tk.Button(row, text=label, width=width, command=lambda v=value: choose(v))
        button.pack(side=tk.LEFT, padx=(6, 0))
        if default is None:
            default = button
    if note:
        caption = tk.Label(frame, text=note, anchor="e", justify="right", wraplength=320)
        theme.register(caption, fg="muted")
        caption.pack(anchor="e", pady=(6, 0))
    if default is not None:
        window.bind("<Return>", lambda _e: default.invoke())

    _run(root, window, lambda: choose(None), focus=default)
    if check_var is not None:
        return result["value"], bool(check_var.get())
    return result["value"]


def ask_yes_no(root, title, message):
    """A centered yes/no confirmation. Returns ``True`` if Yes was chosen."""
    return bool(
        _message_dialog(root, title, message, [("Yes", True), ("No", False)])
    )


def ask_save_changes(root, title, message):
    """Ask whether to save pending changes before discarding them.

    Returns ``"save"``, ``"discard"`` or ``"cancel"`` (``None`` when the
    dialog is dismissed, which callers treat like ``"cancel"``).
    """
    return _message_dialog(
        root,
        title,
        message,
        [("Save", "save"), ("Don't Save", "discard"), ("Cancel", "cancel")],
    )


# Longest layer name the paste dialog's "This Layer" choice shows before it is
# shortened with an ellipsis, so an unusually long name cannot stretch it out.
_PASTE_LAYER_NAME_MAX = 32


def _shorten(text, limit):
    """Trim ``text`` to ``limit`` characters, ending with an ellipsis."""
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"


def ask_paste_dest(root, layer_name=None):
    """Ask where a paste should land and whether to fit it to the canvas.

    ``layer_name`` names the active layer, shown as a muted caption under the
    buttons so "This Layer" is unambiguous. Returns ``(dest, fit)`` where
    ``dest`` is ``"new_layer"``, ``"selected_layer"`` or ``"asset"`` and
    ``fit`` is whether "Fit to canvas" was checked, or ``None`` when the
    dialog is dismissed.
    """
    note = None
    if layer_name:
        note = f"Selected layer: {_shorten(layer_name, _PASTE_LAYER_NAME_MAX)}"
    dest, fit = _message_dialog(
        root,
        "Paste",
        "Where should the clipboard be pasted?",
        [
            ("New Layer", "new_layer"),
            ("This Layer", "selected_layer"),
            ("Assets", "asset"),
        ],
        note=note,
        checkbox=("Fit to canvas", False),
    )
    if dest is None:
        return None
    return dest, fit


def ask_remove_master(root, master_name, links):
    """Confirm dropping a master layer, listing every link that points at it.

    ``links`` is a list of ``(canvas_name, layer_name)`` pairs naming the link
    layers that reference the master. Removing the master unlinks each of them
    into an independent copy of its content. Returns ``True`` when the user
    chooses to unlink and remove, or ``None`` when cancelled.
    """
    window = _make_window(root, "Remove Master Layer")
    frame = tk.Frame(window, padx=16, pady=16)
    frame.pack(fill=tk.BOTH, expand=True)

    count = len(links)
    noun = "layer" if count == 1 else "layers"
    heading = f"{count} {noun} link to '{_shorten(master_name, _PASTE_LAYER_NAME_MAX)}'."
    tk.Label(frame, text=heading, anchor="w", justify="left", wraplength=360).pack(
        anchor="w"
    )
    caption = tk.Label(
        frame,
        text="Removing the master unlinks these layers into independent copies.",
        anchor="w", justify="left", wraplength=360,
    )
    theme.register(caption, fg="muted")
    caption.pack(anchor="w", pady=(4, 8))

    holder = tk.Frame(frame)
    holder.pack(fill=tk.BOTH, expand=True)
    body = tk.Listbox(holder, height=max(3, min(12, count)), width=48,
                      activestyle="none", highlightthickness=1, bd=0)
    scroll = AutoScrollbar(holder, command=body.yview)
    body.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    for canvas_name, layer_name in links:
        body.insert(tk.END, f"{canvas_name} \u203a {layer_name}")
    body.configure(state=tk.DISABLED, disabledforeground=theme.color("text"))

    result = {"value": None}

    def choose(value):
        result["value"] = value
        window.destroy()

    row = tk.Frame(frame)
    row.pack(anchor="e", pady=(12, 0))
    unlink_btn = tk.Button(
        row, text="Unlink & Remove", width=16, command=lambda: choose(True)
    )
    unlink_btn.pack(side=tk.RIGHT)
    tk.Button(row, text="Cancel", width=12, command=lambda: choose(None)).pack(
        side=tk.RIGHT, padx=(0, 6)
    )
    window.bind("<Return>", lambda _e: unlink_btn.invoke())

    _run(root, window, lambda: choose(None), focus=unlink_btn)
    return result["value"]


def show_info(root, title, message):
    """A centered informational message."""
    _message_dialog(root, title, message, [("OK", True)])


def show_error(root, title, message):
    """A centered error message."""
    _message_dialog(root, title, message, [("OK", True)])


def show_text(root, title, text):
    """Show ``text`` in full in a centered, resizable, read-only dialog."""
    window = _make_window(root, title)
    window.resizable(True, True)
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    holder = tk.Frame(frame)
    holder.pack(fill=tk.BOTH, expand=True)
    # Size to the message, but keep very long entries to a readable height.
    lines = 1 + text.count("\n") + len(text) // 70
    body = tk.Text(
        holder, width=72, height=max(3, min(16, lines)), wrap=tk.WORD,
    )
    scroll = AutoScrollbar(holder, command=body.yview)
    body.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    body.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    body.insert("1.0", text)
    body.configure(state=tk.DISABLED)

    def copy():
        window.clipboard_clear()
        window.clipboard_append(text)

    buttons = tk.Frame(frame)
    buttons.pack(fill=tk.X, pady=(8, 0))
    close_btn = tk.Button(buttons, text="Close", width=8, command=window.destroy)
    close_btn.pack(side=tk.RIGHT)
    tk.Button(buttons, text="Copy", width=8, command=copy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    _run(root, window, window.destroy, focus=close_btn)


def open_url(url):
    """Open ``url`` in the user's browser, ignoring failures."""
    try:
        webbrowser.open(url)
    except Exception:
        pass


def _link_dialog(root, title, heading, message, url, button_label):
    """A centered prompt whose primary button opens ``url`` in a browser."""
    window = _make_window(root, title)
    frame = tk.Frame(window, padx=16, pady=16)
    frame.pack(fill=tk.BOTH, expand=True)

    heading_font = tkfont.nametofont("TkDefaultFont").copy()
    heading_font.configure(weight="bold", size=heading_font.cget("size") + 2)
    tk.Label(frame, text=heading, font=heading_font, anchor="w").pack(anchor="w")

    tk.Label(frame, text=message, anchor="w", justify="left", wraplength=320).pack(
        anchor="w", pady=(8, 0)
    )

    row = tk.Frame(frame)
    row.pack(anchor="e", pady=(12, 0))

    def open_link():
        open_url(url)
        window.destroy()

    tk.Button(row, text=button_label, width=12, command=open_link).pack(side=tk.RIGHT)
    tk.Button(row, text="Maybe Later", width=12, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )
    window.bind("<Return>", lambda _e: open_link())

    _run(root, window, window.destroy)


def show_donate(root, url, message):
    """A centered donation prompt with a button that opens ``url`` in a browser."""
    _link_dialog(root, "Support Pai", "Support Pai", message, url, "Donate")


def show_contribute(root, url, message):
    """A centered contribution prompt with a button that opens ``url``."""
    _link_dialog(root, "Contribute", "Contribute to Pai", message, url, "Open Page")


_EXPORT_FORMATS = ("PNG", "JPEG", "WEBP", "GIF", "TIFF")
# Formats that can hold more than one frame, so "all canvases" can export an
# animation instead of one file per canvas.
_EXPORT_ANIMATABLE = ("PNG", "WEBP", "GIF")
# Formats that can store a CMYK colour space, for print-ready output.
_EXPORT_CMYK_FORMATS = ("JPEG", "TIFF")
# Print page presets: ``(id, label)`` in display order. ``custom`` reads its
# millimetre size from the dialog's own fields.
_EXPORT_PAGE_PRESETS = (
    ("none", "None"),
    ("a5", "A5 (148 \u00d7 210 mm)"),
    ("b5", "B5 (182 \u00d7 257 mm)"),
    ("a4", "A4 (210 \u00d7 297 mm)"),
    ("b4", "B4 (257 \u00d7 364 mm)"),
    ("letter", "Letter (216 \u00d7 279 mm)"),
    ("custom", "Custom\u2026"),
)
# Preset id -> portrait ``(width_mm, height_mm)``.
_EXPORT_PAGE_MM = {
    "a5": (148.0, 210.0),
    "b5": (182.0, 257.0),
    "a4": (210.0, 297.0),
    "b4": (257.0, 364.0),
    "letter": (215.9, 279.4),
}
_EXPORT_PAGE_LABELS = {pid: label for pid, label in _EXPORT_PAGE_PRESETS}
_EXPORT_PAGE_BY_LABEL = {label: pid for pid, label in _EXPORT_PAGE_PRESETS}
_EXPORT_PRINT_DPI = 300
_EXPORT_CUSTOM = "Custom\u2026"
_EXPORT_SCALE_CHOICES = ("25%", "50%", "100%", "200%", "400%", _EXPORT_CUSTOM)
_EXPORT_RESAMPLE = (
    ("Nearest (hard edges)", "nearest"),
    ("Box (fast)", "box"),
    ("Bilinear", "bilinear"),
    ("Bicubic", "bicubic"),
    ("Lanczos (best)", "lanczos"),
)
_EXPORT_RESAMPLE_BY_LABEL = {label: value for label, value in _EXPORT_RESAMPLE}
_EXPORT_RESAMPLE_LABELS = {value: label for label, value in _EXPORT_RESAMPLE}
_EXPORT_JPEG_SUBSAMPLING = (
    ("4:4:4 (best quality)", 0),
    ("4:2:2", 1),
    ("4:2:0 (smallest file)", 2),
)
_EXPORT_JPEG_SUB_BY_LABEL = {label: value for label, value in _EXPORT_JPEG_SUBSAMPLING}
_EXPORT_JPEG_SUB_LABELS = {value: label for label, value in _EXPORT_JPEG_SUBSAMPLING}
# Descriptive metadata keys and their dialog labels. Blank entries are omitted
# from the exported file.
_EXPORT_METADATA_FIELDS = (
    ("title", "Title"),
    ("author", "Author"),
    ("copyright", "Copyright"),
    ("description", "Description"),
    ("keywords", "Keywords"),
    ("software", "Software"),
    ("created", "Date (YYYY-MM-DD)"),
)
# Fixed pixel size of the export dialog's scrollable body. The window never
# resizes; the body scrolls and the footer stays pinned below it.
_EXPORT_BODY_W = 460
_EXPORT_BODY_H = 440
# Long hint labels wrap at this width so they do not widen the fixed dialog.
_EXPORT_HINT_WRAP = 420


def _human_size(nbytes):
    """A short ``"1.2 MB"``-style size read-out for the estimate label."""
    size = float(nbytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            break
        size /= 1024
    return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"


def ask_export(root, multiple_canvases=False, title="Export Images", action="Export",
               base_size=None, estimate=None, initial=None, canvas_name=None,
               canvas_count=None):
    """Prompt for image export options.

    Returns a dict describing the export:

    * ``scope`` - ``"current"`` or ``"all"`` canvases.
    * ``format`` - ``"PNG"``/``"JPEG"``/``"WEBP"``/``"GIF"``.
    * ``quality`` - 1-100 for JPEG/WEBP.
    * ``scale`` - a size multiplier, used when ``size`` is ``None``.
    * ``size`` - an explicit ``(width, height)`` in pixels, or ``None``.
    * ``resample`` - the Pillow resize filter id (``"lanczos"`` by default).
    * ``trim`` - crop fully transparent borders before sizing.
    * ``transparent`` - keep the alpha channel (always ``False`` for JPEG).
    * ``background`` - matte ``#rrggbb`` used when ``transparent`` is ``False``.
    * ``animate`` - merge every canvas into one animated file (only when
      ``scope`` is ``"all"`` and the format is animatable).
    * ``frame_delay`` - animation frame duration in milliseconds.
    * ``loop`` - whether an animation repeats forever.
    * ``png``/``jpeg``/``webp``/``gif`` - per-format option dicts.
    * ``dpi`` - embedded resolution as an int, or ``None``.
    * ``metadata`` - descriptive fields (``title``, ``author``, ``copyright``,
      ``description``, ``keywords``, ``software``, ``created``); blank entries
      are omitted from the file.
    * ``page`` - print page preset id (``"none"``, ``"a5"``, ``"b5"``, ``"a4"``,
      ``"b4"``, ``"letter"`` or ``"custom"``).
    * ``page_mm`` - the page ``(width, height)`` in millimetres, or ``None``.
    * ``bleed_mm`` - bleed margin in millimetres (``0`` for none).
    * ``crop_marks`` - draw corner crop marks in the bleed margin.
    * ``cmyk`` - convert to CMYK (only for JPEG/TIFF).
    * ``profile_target`` - profile id to convert the pixels to, or ``""`` to
      keep the working space.
    * ``profile_embed`` - embed the output profile in the file.
    * ``cmyk_profile`` - CMYK profile id for the conversion, or ``""`` for a
      naive ``convert("CMYK")``.

    Returns ``None`` when cancelled. ``title`` and ``action`` label the dialog
    and its confirm button so the same options serve both export and save.
    ``base_size`` is the current canvas size, used to seed the custom
    width/height fields. ``estimate(options)`` is an optional callback returning
    ``(bytes, (width, height))``; when given it drives a live size read-out that
    refreshes automatically as options change, or ``None``. ``initial`` is a
    previous options dict used to restore the last-used settings. ``canvas_name``
    names the canvas the output comes from and is shown in a pinned header at
    the top; ``canvas_count`` (with ``canvas_name``) lets that header report the
    canvas count instead when ``scope`` is ``"all"``.
    """
    window = _make_window(root, title)
    outer = tk.Frame(window, padx=12, pady=12)
    outer.pack(fill=tk.BOTH, expand=True)

    # A pinned header naming the canvas the output comes from, so it is obvious
    # before saving. It is packed first so it sits above the scrolling body and
    # stays visible. When exporting every canvas it reports the count instead
    # (see ``update_canvas_header`` below, once ``scope_var`` exists).
    canvas_header_var = tk.StringVar(
        value=f"Canvas: {canvas_name}" if canvas_name else ""
    )
    if canvas_name:
        canvas_header = tk.Label(outer, textvariable=canvas_header_var, anchor="w")
        canvas_header.pack(side=tk.TOP, fill=tk.X, pady=(0, 8))
        theme.register(canvas_header, fg="muted")

    # The body scrolls inside a fixed-size area; the footer stays pinned below
    # it. A plain ``ArrowScrollbar`` is used so the bar is always visible even
    # when a collapsed accordion fits without scrolling.
    body_holder = tk.Frame(outer)
    body_holder.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    canvas = tk.Canvas(
        body_holder, highlightthickness=0, bd=0,
        width=_EXPORT_BODY_W, height=_EXPORT_BODY_H, bg=theme.color("bg"),
    )
    scroll = ArrowScrollbar(body_holder, command=canvas.yview)
    canvas.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    frame = tk.Frame(canvas)
    theme.register(frame, bg="bg")
    frame_id = canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind(
        "<Configure>",
        lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
    )
    canvas.bind(
        "<Configure>",
        lambda e: canvas.itemconfigure(frame_id, width=e.width),
    )

    def on_wheel(event):
        canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    # A Toplevel binding reaches its descendants through bindtags, so the wheel
    # scrolls the body without a global ``bind_all`` (which would clobber the
    # sidebar's own wheel handler on close).
    window.bind("<MouseWheel>", on_wheel)

    base_w, base_h = base_size or (0, 0)
    aspect = (base_w / base_h) if base_w and base_h else 1.0

    initial = initial if isinstance(initial, dict) else {}

    def seed(key, default):
        value = initial.get(key)
        return default if value is None else value

    def seed_bool(key, default):
        return bool(seed(key, default))

    def seed_int(key, default, low, high):
        try:
            return max(low, min(high, int(seed(key, default))))
        except (TypeError, ValueError):
            return default

    def seed_sub(key):
        value = initial.get(key)
        return value if isinstance(value, dict) else {}

    def sub_int(opts, key, default, low, high):
        try:
            return max(low, min(high, int(opts.get(key, default))))
        except (TypeError, ValueError):
            return default

    def sub_bool(opts, key, default):
        return bool(opts.get(key, default))

    def seed_color(key, default):
        value = seed(key, default)
        hexdigits = "0123456789abcdefABCDEF"
        if (isinstance(value, str) and len(value) == 7 and value[0] == "#"
                and all(ch in hexdigits for ch in value[1:])):
            return value.lower()
        return default

    scale_label = "100%"
    try:
        scale_value = float(seed("scale", 1.0))
    except (TypeError, ValueError):
        scale_value = 1.0
    for choice in _EXPORT_SCALE_CHOICES:
        if choice != _EXPORT_CUSTOM and abs(
            int(choice.rstrip("%")) / 100.0 - scale_value
        ) < 1e-9:
            scale_label = choice
            break

    format_seed = seed("format", "PNG")
    scope_var = tk.StringVar(value="current")
    format_var = tk.StringVar(
        value=format_seed if format_seed in _EXPORT_FORMATS else "PNG"
    )
    scale_var = tk.StringVar(value=scale_label)
    width_var = tk.StringVar(value=str(base_w or 1))
    height_var = tk.StringVar(value=str(base_h or 1))
    lock_var = tk.BooleanVar(value=True)
    resample_var = tk.StringVar(value=_EXPORT_RESAMPLE_LABELS.get(
        seed("resample", "lanczos"), _EXPORT_RESAMPLE_LABELS["lanczos"]
    ))
    transparent_var = tk.BooleanVar(value=seed_bool("transparent", True))
    trim_var = tk.BooleanVar(value=seed_bool("trim", False))
    quality_var = tk.IntVar(value=seed_int("quality", 90, 1, 100))
    png_opts = seed_sub("png")
    png_compress_var = tk.IntVar(value=sub_int(png_opts, "compress_level", 6, 0, 9))
    png_optimize_var = tk.BooleanVar(value=sub_bool(png_opts, "optimize", False))
    png_interlace_var = tk.BooleanVar(value=sub_bool(png_opts, "interlace", False))
    jpeg_opts = seed_sub("jpeg")
    jpeg_progressive_var = tk.BooleanVar(value=sub_bool(jpeg_opts, "progressive", False))
    jpeg_optimize_var = tk.BooleanVar(value=sub_bool(jpeg_opts, "optimize", True))
    jpeg_subsampling_var = tk.StringVar(value=_EXPORT_JPEG_SUB_LABELS.get(
        sub_int(jpeg_opts, "subsampling", 2, 0, 2), _EXPORT_JPEG_SUB_LABELS[2]
    ))
    webp_opts = seed_sub("webp")
    webp_lossless_var = tk.BooleanVar(value=sub_bool(webp_opts, "lossless", False))
    webp_method_var = tk.IntVar(value=sub_int(webp_opts, "method", 4, 0, 6))
    gif_opts = seed_sub("gif")
    gif_optimize_var = tk.BooleanVar(value=sub_bool(gif_opts, "optimize", False))
    anim_var = tk.BooleanVar(value=seed_bool("animate", False))
    delay_var = tk.StringVar(value=str(seed_int("frame_delay", 100, 10, 10000)))
    loop_var = tk.BooleanVar(value=seed_bool("loop", True))
    dpi_seed = initial.get("dpi")
    dpi_var = tk.StringVar(
        value="" if dpi_seed in (None, "") else str(dpi_seed)
    )
    meta_seed = initial.get("metadata")
    if not isinstance(meta_seed, dict):
        meta_seed = {}
    meta_vars = {
        key: tk.StringVar(value=str(meta_seed.get(key) or ""))
        for key, _label in _EXPORT_METADATA_FIELDS
    }
    background = {"value": seed_color("background", "#ffffff")}
    page_seed = seed("page", "none")
    page_var = tk.StringVar(
        value=_EXPORT_PAGE_LABELS.get(page_seed, "None")
    )
    page_w_var = tk.StringVar(value=str(seed_int("page_w_mm", 210, 10, 5000)))
    page_h_var = tk.StringVar(value=str(seed_int("page_h_mm", 297, 10, 5000)))
    bleed_var = tk.StringVar(value=str(seed_int("bleed_mm", 0, 0, 50)))
    crop_var = tk.BooleanVar(value=seed_bool("crop_marks", False))
    cmyk_var = tk.BooleanVar(value=seed_bool("cmyk", False))
    updating = {"on": False}
    syncing = {"on": False}

    # Keep the pinned canvas header in step with the Scope choice: name the
    # current canvas, or report the count once "All canvases" is selected.
    if canvas_name and canvas_count:
        def update_canvas_header(*_args):
            if scope_var.get() == "all":
                canvas_header_var.set(f"Exporting all {canvas_count} canvases")
            else:
                canvas_header_var.set(f"Canvas: {canvas_name}")

        scope_var.trace_add("write", update_canvas_header)
        update_canvas_header()

    # ICC colour-management choices. Profile labels must be unique in a
    # combobox, so duplicate file names get a numeric suffix.
    profile_items = colormanagement.list_profiles()
    profile_by_label = {}
    for _ident, _label in profile_items:
        unique = _label
        suffix = 2
        while unique in profile_by_label:
            unique = f"{_label} ({suffix})"
            suffix += 1
        profile_by_label[unique] = _ident
    profile_labels = ["Working space"] + list(profile_by_label)
    profile_label_to_id = {"Working space": "", **profile_by_label}
    profile_id_to_label = {value: key for key, value in profile_label_to_id.items()}
    profile_var = tk.StringVar(value=profile_id_to_label.get(
        seed("profile_target", ""), "Working space"
    ))
    embed_var = tk.BooleanVar(value=seed_bool("profile_embed", True))
    cmyk_items = colormanagement.list_cmyk_profiles()
    cmyk_labels = ["Automatic (no profile)"] + [label for _ident, label in cmyk_items]
    cmyk_label_to_id = {"Automatic (no profile)": ""}
    for _ident, _label in cmyk_items:
        cmyk_label_to_id.setdefault(_label, _ident)
    cmyk_profile_var = tk.StringVar(value=next(
        (label for label, ident in cmyk_label_to_id.items()
         if ident == seed("cmyk_profile", "")),
        "Automatic (no profile)",
    ))

    # ---- Collapsible sections -------------------------------------------
    # The dialog grew a lot of sections, so each one is an accordion: a
    # clickable header over its body. ``add_section`` returns the holder to
    # pack and the body frame to build the rows into. Output, Background and
    # Format options start open; the rest start collapsed.
    def add_section(title, open_by_default=True):
        holder = tk.Frame(frame)
        header = tk.Button(holder, anchor="w", relief=tk.FLAT, bd=0)
        body = tk.Frame(holder, padx=8, pady=4)
        state = {"open": bool(open_by_default)}

        def apply():
            glyph = "\u25be" if state["open"] else "\u25b8"
            header.config(text=f"{glyph}  {title}")
            if state["open"]:
                body.pack(fill=tk.X)
            else:
                body.pack_forget()

        def toggle():
            state["open"] = not state["open"]
            apply()

        header.config(command=toggle)
        header.pack(fill=tk.X)
        apply()
        return holder, body

    # ---- Output ---------------------------------------------------------
    output_holder, output = add_section("Output", True)
    output_holder.pack(fill=tk.X)
    row = 0
    if multiple_canvases:
        tk.Label(output, text="Scope:", anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 8), pady=2
        )
        scope_row = tk.Frame(output)
        scope_row.grid(row=row, column=1, sticky="w", pady=2)
        tk.Radiobutton(
            scope_row, text="Current canvas", variable=scope_var, value="current",
        ).pack(side=tk.LEFT)
        tk.Radiobutton(
            scope_row, text="All canvases", variable=scope_var, value="all",
        ).pack(side=tk.LEFT, padx=(6, 0))
        row += 1

    tk.Label(output, text="Format:", anchor="w").grid(
        row=row, column=0, sticky="w", padx=(0, 8), pady=2
    )
    format_menu = ttk.Combobox(
        output, textvariable=format_var, values=list(_EXPORT_FORMATS),
        state="readonly", width=10,
    )
    format_menu.grid(row=row, column=1, sticky="w", pady=2)
    row += 1

    tk.Label(output, text="Scale:", anchor="w").grid(
        row=row, column=0, sticky="w", padx=(0, 8), pady=2
    )
    scale_menu = ttk.Combobox(
        output, textvariable=scale_var, values=list(_EXPORT_SCALE_CHOICES),
        state="readonly", width=10,
    )
    scale_menu.grid(row=row, column=1, sticky="w", pady=2)
    row += 1

    tk.Label(output, text="Size (px):", anchor="w").grid(
        row=row, column=0, sticky="w", padx=(0, 8), pady=2
    )
    size_row = tk.Frame(output)
    size_row.grid(row=row, column=1, sticky="w", pady=2)
    width_entry = tk.Entry(size_row, textvariable=width_var, width=7, justify="right")
    width_entry.pack(side=tk.LEFT)
    tk.Label(size_row, text="\u00d7").pack(side=tk.LEFT, padx=4)
    height_entry = tk.Entry(size_row, textvariable=height_var, width=7, justify="right")
    height_entry.pack(side=tk.LEFT)
    lock_check = tk.Checkbutton(size_row, text="Lock aspect", variable=lock_var)
    lock_check.pack(side=tk.LEFT, padx=(8, 0))
    row += 1

    tk.Label(output, text="Resample:", anchor="w").grid(
        row=row, column=0, sticky="w", padx=(0, 8), pady=2
    )
    ttk.Combobox(
        output, textvariable=resample_var,
        values=[label for label, _value in _EXPORT_RESAMPLE],
        state="readonly", width=22,
    ).grid(row=row, column=1, sticky="w", pady=2)

    # ---- Background -----------------------------------------------------
    bg_holder, background_frame = add_section("Background", True)
    bg_holder.pack(fill=tk.X, pady=(8, 0))
    transparent_check = tk.Checkbutton(
        background_frame, text="Transparent background", variable=transparent_var,
        anchor="w",
    )
    transparent_check.grid(row=0, column=0, columnspan=2, sticky="w")
    tk.Label(background_frame, text="Matte color:", anchor="w").grid(
        row=1, column=0, sticky="w", padx=(0, 8), pady=2
    )
    matte_row = tk.Frame(background_frame)
    matte_row.grid(row=1, column=1, sticky="w", pady=2)
    # The swatch is content (the chosen matte colour), so it is left unthemed.
    bg_swatch = tk.Label(matte_row, width=2, height=1, relief=tk.SUNKEN, bd=1)
    bg_swatch.pack(side=tk.LEFT, padx=(0, 6))
    theme.skip(bg_swatch)
    bg_button = tk.Button(matte_row, text="Choose\u2026", width=10)
    bg_button.pack(side=tk.LEFT)
    tk.Checkbutton(
        background_frame, text="Trim transparent edges", variable=trim_var,
        anchor="w",
    ).grid(row=2, column=0, columnspan=2, sticky="w")

    # ---- Animation ------------------------------------------------------
    # Created unpacked; ``sync`` shows it only for an animatable format with the
    # "all canvases" scope. It stays above the Format options section.
    anim_holder, animation_frame = add_section("Animation", False)
    tk.Checkbutton(
        animation_frame, text="Export all canvases as one animated file",
        variable=anim_var, anchor="w",
    ).grid(row=0, column=0, columnspan=3, sticky="w")
    tk.Label(animation_frame, text="Frame delay (ms):", anchor="w").grid(
        row=1, column=0, sticky="w", padx=(0, 8), pady=2
    )
    tk.Entry(animation_frame, textvariable=delay_var, width=8).grid(
        row=1, column=1, sticky="w", pady=2
    )
    tk.Checkbutton(
        animation_frame, text="Loop forever", variable=loop_var, anchor="w",
    ).grid(row=2, column=0, columnspan=3, sticky="w")

    # ---- Format options -------------------------------------------------
    options_holder, options_frame = add_section("Format options", True)
    options_holder.pack(fill=tk.X, pady=(8, 0))

    quality_row = tk.Frame(options_frame)
    quality_row.pack(fill=tk.X)
    tk.Label(quality_row, text="Quality:", anchor="w", width=12).pack(side=tk.LEFT)
    quality_scale = tk.Scale(
        quality_row, from_=1, to=100, orient=tk.HORIZONTAL, variable=quality_var,
        length=200,
    )
    quality_scale.pack(side=tk.LEFT)

    format_body = tk.Frame(options_frame)
    format_body.pack(fill=tk.X, pady=(4, 0))

    png_frame = tk.Frame(format_body)
    tk.Label(png_frame, text="Compression:", anchor="w", width=12).grid(
        row=0, column=0, sticky="w"
    )
    tk.Scale(
        png_frame, from_=0, to=9, orient=tk.HORIZONTAL, variable=png_compress_var,
        length=160,
    ).grid(row=0, column=1, sticky="w")
    tk.Checkbutton(png_frame, text="Optimize", variable=png_optimize_var).grid(
        row=1, column=0, sticky="w"
    )
    tk.Checkbutton(png_frame, text="Interlaced (Adam7)", variable=png_interlace_var).grid(
        row=1, column=1, sticky="w"
    )

    jpeg_frame = tk.Frame(format_body)
    tk.Checkbutton(jpeg_frame, text="Progressive", variable=jpeg_progressive_var).grid(
        row=0, column=0, sticky="w"
    )
    tk.Checkbutton(jpeg_frame, text="Optimize", variable=jpeg_optimize_var).grid(
        row=0, column=1, sticky="w"
    )
    tk.Label(jpeg_frame, text="Subsampling:", anchor="w", width=12).grid(
        row=1, column=0, sticky="w", pady=(2, 0)
    )
    ttk.Combobox(
        jpeg_frame, textvariable=jpeg_subsampling_var,
        values=[label for label, _value in _EXPORT_JPEG_SUBSAMPLING],
        state="readonly", width=22,
    ).grid(row=1, column=1, sticky="w", pady=(2, 0))

    webp_frame = tk.Frame(format_body)
    tk.Checkbutton(webp_frame, text="Lossless", variable=webp_lossless_var).grid(
        row=0, column=0, sticky="w"
    )
    tk.Label(webp_frame, text="Method:", anchor="w", width=12).grid(
        row=1, column=0, sticky="w", pady=(2, 0)
    )
    tk.Scale(
        webp_frame, from_=0, to=6, orient=tk.HORIZONTAL, variable=webp_method_var,
        length=160,
    ).grid(row=1, column=1, sticky="w", pady=(2, 0))

    gif_frame = tk.Frame(format_body)
    tk.Checkbutton(gif_frame, text="Optimize", variable=gif_optimize_var).grid(
        row=0, column=0, sticky="w"
    )
    gif_hint = tk.Label(
        gif_frame, text="GIF is limited to a 256-color palette.", anchor="w",
    )
    gif_hint.grid(row=1, column=0, sticky="w", pady=(2, 0))
    theme.register(gif_hint, fg="muted")

    tiff_frame = tk.Frame(format_body)
    tiff_hint = tk.Label(
        tiff_frame, text="Uncompressed TIFF; keeps CMYK for print.", anchor="w",
    )
    tiff_hint.grid(row=0, column=0, sticky="w")
    theme.register(tiff_hint, fg="muted")

    # ---- Metadata -------------------------------------------------------
    meta_holder, meta_frame = add_section("Metadata", False)
    meta_holder.pack(fill=tk.X, pady=(8, 0))
    tk.Label(meta_frame, text="DPI:", anchor="w").grid(
        row=0, column=0, sticky="w", padx=(0, 8)
    )
    tk.Entry(meta_frame, textvariable=dpi_var, width=8).grid(
        row=0, column=1, sticky="w"
    )
    dpi_hint = tk.Label(meta_frame, text="(blank to omit)", anchor="w")
    dpi_hint.grid(row=0, column=2, sticky="w", padx=(8, 0))
    theme.register(dpi_hint, fg="muted")
    meta_row = 1
    for key, label in _EXPORT_METADATA_FIELDS:
        tk.Label(meta_frame, text=f"{label}:", anchor="w").grid(
            row=meta_row, column=0, sticky="w", padx=(0, 8), pady=2
        )
        tk.Entry(meta_frame, textvariable=meta_vars[key], width=40).grid(
            row=meta_row, column=1, columnspan=2, sticky="w", pady=2
        )
        meta_row += 1
    meta_hint = tk.Label(
        meta_frame,
        text="Blank fields are omitted. GIF does not store this metadata.",
        anchor="w", justify="left", wraplength=_EXPORT_HINT_WRAP,
    )
    meta_hint.grid(row=meta_row, column=0, columnspan=3, sticky="w", pady=(2, 0))
    theme.register(meta_hint, fg="muted")

    # ---- Color ----------------------------------------------------------
    color_holder, color_frame = add_section("Color", False)
    color_holder.pack(fill=tk.X, pady=(8, 0))
    tk.Label(color_frame, text="Convert to:", anchor="w").grid(
        row=0, column=0, sticky="w", padx=(0, 8), pady=2
    )
    ttk.Combobox(
        color_frame, textvariable=profile_var, values=profile_labels,
        state="readonly", width=30,
    ).grid(row=0, column=1, sticky="w", pady=2)
    tk.Checkbutton(
        color_frame, text="Embed profile in file", variable=embed_var, anchor="w",
    ).grid(row=1, column=0, columnspan=2, sticky="w")
    tk.Label(color_frame, text="CMYK profile:", anchor="w").grid(
        row=2, column=0, sticky="w", padx=(0, 8), pady=2
    )
    cmyk_profile_menu = ttk.Combobox(
        color_frame, textvariable=cmyk_profile_var, values=cmyk_labels,
        state="disabled", width=30,
    )
    cmyk_profile_menu.grid(row=2, column=1, sticky="w", pady=2)
    color_hint = tk.Label(
        color_frame,
        text="Import .icc/.icm profiles from Canvas \u2192 Color Management\u2026",
        anchor="w", justify="left", wraplength=_EXPORT_HINT_WRAP,
    )
    color_hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 0))
    theme.register(color_hint, fg="muted")

    # ---- Print ----------------------------------------------------------
    print_holder, print_frame = add_section("Print", False)
    print_holder.pack(fill=tk.X, pady=(8, 0))
    tk.Label(print_frame, text="Page:", anchor="w").grid(
        row=0, column=0, sticky="w", padx=(0, 8), pady=2
    )
    page_menu = ttk.Combobox(
        print_frame, textvariable=page_var,
        values=[label for _pid, label in _EXPORT_PAGE_PRESETS],
        state="readonly", width=22,
    )
    page_menu.grid(row=0, column=1, sticky="w", pady=2)

    custom_row = tk.Frame(print_frame)
    custom_row.grid(row=1, column=0, columnspan=2, sticky="w", pady=2)
    tk.Label(custom_row, text="Custom (mm):", anchor="w").pack(side=tk.LEFT)
    page_w_entry = tk.Entry(
        custom_row, textvariable=page_w_var, width=6, justify="right"
    )
    page_w_entry.pack(side=tk.LEFT, padx=(6, 2))
    tk.Label(custom_row, text="\u00d7").pack(side=tk.LEFT)
    page_h_entry = tk.Entry(
        custom_row, textvariable=page_h_var, width=6, justify="right"
    )
    page_h_entry.pack(side=tk.LEFT, padx=(2, 0))

    tk.Label(print_frame, text="Bleed (mm):", anchor="w").grid(
        row=2, column=0, sticky="w", padx=(0, 8), pady=2
    )
    tk.Entry(print_frame, textvariable=bleed_var, width=8).grid(
        row=2, column=1, sticky="w", pady=2
    )
    tk.Checkbutton(
        print_frame, text="Crop marks", variable=crop_var, anchor="w",
    ).grid(row=3, column=0, sticky="w")
    cmyk_check = tk.Checkbutton(
        print_frame, text="CMYK (JPEG/TIFF)", variable=cmyk_var, anchor="w",
    )
    cmyk_check.grid(row=3, column=1, sticky="w")
    print_hint = tk.Label(
        print_frame,
        text="A page sizes the output at the DPI above (300 when blank); the "
             "Scale/Size fields are then ignored.",
        anchor="w", justify="left", wraplength=_EXPORT_HINT_WRAP,
    )
    print_hint.grid(row=4, column=0, columnspan=2, sticky="w", pady=(2, 0))
    theme.register(print_hint, fg="muted")

    # ---- Footer ---------------------------------------------------------
    # Pinned to ``outer`` (outside the scroll area) so it always stays visible.
    footer = tk.Frame(outer)
    footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 0))
    estimate_label = tk.Label(footer, text="", anchor="w")
    estimate_job = {"id": None}

    def cancel_estimate(*_args):
        if estimate_job["id"] is not None:
            try:
                window.after_cancel(estimate_job["id"])
            except tk.TclError:
                pass
            estimate_job["id"] = None

    def run_estimate():
        estimate_job["id"] = None
        if estimate is None:
            return
        try:
            outcome = estimate(collect())
        except Exception as exc:
            estimate_label.config(text=f"Estimate failed: {exc}")
            return
        if not outcome:
            estimate_label.config(text="")
            return
        nbytes, size = outcome
        estimate_label.config(
            text=f"\u2248 {_human_size(nbytes)}  ({size[0]}\u00d7{size[1]})"
        )

    def mark_stale(*_args):
        """Show the read-out as stale and recompute it once edits settle.

        Estimating encodes the image, so it is debounced rather than run on
        every keystroke or slider tick.
        """
        cancel_estimate()
        if estimate is None:
            return
        estimate_label.config(text="Estimating\u2026")
        estimate_job["id"] = window.after(300, run_estimate)

    def on_destroy(event):
        # A pending ``after`` outlives its widget, so stop it explicitly.
        if event.widget is window:
            cancel_estimate()

    window.bind("<Destroy>", on_destroy, add="+")

    def refresh_size(*_args):
        if updating["on"]:
            return
        updating["on"] = True
        try:
            custom = scale_var.get() == _EXPORT_CUSTOM
            if not custom:
                try:
                    factor = int(scale_var.get().rstrip("%")) / 100.0
                except ValueError:
                    factor = 1.0
                width_var.set(str(max(1, round(base_w * factor))))
                height_var.set(str(max(1, round(base_h * factor))))
            state = tk.NORMAL if custom else tk.DISABLED
            width_entry.config(state=state)
            height_entry.config(state=state)
            lock_check.config(state=state)
        finally:
            updating["on"] = False
        mark_stale()

    def on_width(*_args):
        if (updating["on"] or not lock_var.get()
                or scale_var.get() != _EXPORT_CUSTOM):
            return
        try:
            value = float(width_var.get())
        except ValueError:
            return
        updating["on"] = True
        try:
            height_var.set(str(max(1, round(value / aspect))))
        finally:
            updating["on"] = False
        mark_stale()

    def on_height(*_args):
        if (updating["on"] or not lock_var.get()
                or scale_var.get() != _EXPORT_CUSTOM):
            return
        try:
            value = float(height_var.get())
        except ValueError:
            return
        updating["on"] = True
        try:
            width_var.set(str(max(1, round(value * aspect))))
        finally:
            updating["on"] = False
        mark_stale()

    def apply_bg():
        bg_swatch.config(bg=background["value"])

    def choose_bg():
        def preview(chosen):
            background["value"] = chosen
            apply_bg()
            mark_stale()

        ask_color(window, background["value"], title="Background Color",
                  on_preview=preview)
        apply_bg()

    bg_button.config(command=choose_bg)
    apply_bg()

    def sync(*_args):
        if syncing["on"]:
            return
        syncing["on"] = True
        try:
            fmt = format_var.get()
            is_jpeg = fmt == "JPEG"
            if is_jpeg:
                transparent_var.set(False)
            transparent_check.config(state=tk.DISABLED if is_jpeg else tk.NORMAL)
            bg_button.config(
                state=tk.DISABLED if transparent_var.get() else tk.NORMAL
            )
            for sub, name in ((png_frame, "PNG"), (jpeg_frame, "JPEG"),
                              (webp_frame, "WEBP"), (gif_frame, "GIF"),
                              (tiff_frame, "TIFF")):
                if name == fmt:
                    sub.pack(fill=tk.X)
                else:
                    sub.pack_forget()
            quality_scale.config(state=tk.NORMAL)
            custom_page = page_var.get() == _EXPORT_PAGE_LABELS["custom"]
            page_state = tk.NORMAL if custom_page else tk.DISABLED
            page_w_entry.config(state=page_state)
            page_h_entry.config(state=page_state)
            supports_cmyk = fmt in _EXPORT_CMYK_FORMATS
            if not supports_cmyk:
                cmyk_var.set(False)
            cmyk_check.config(state=tk.NORMAL if supports_cmyk else tk.DISABLED)
            cmyk_profile_menu.config(
                state="readonly" if (supports_cmyk and cmyk_var.get()) else "disabled"
            )
            animatable = (multiple_canvases and scope_var.get() == "all"
                          and fmt in _EXPORT_ANIMATABLE)
            if animatable:
                anim_holder.pack(fill=tk.X, pady=(8, 0), before=options_holder)
            else:
                anim_holder.pack_forget()
        finally:
            syncing["on"] = False
        mark_stale()

    def collect():
        fmt = format_var.get()
        size = None
        if scale_var.get() == _EXPORT_CUSTOM:
            try:
                size = (max(1, int(float(width_var.get()))),
                        max(1, int(float(height_var.get()))))
            except ValueError:
                size = None
        scale = 1.0
        if size is None:
            try:
                scale = int(scale_var.get().rstrip("%")) / 100.0
            except ValueError:
                scale = 1.0
        dpi = None
        text = dpi_var.get().strip()
        if text:
            try:
                dpi = max(1, min(100000, int(round(float(text)))))
            except ValueError:
                dpi = None
        try:
            frame_delay = max(10, min(10000, int(float(delay_var.get()))))
        except ValueError:
            frame_delay = 100
        animate = (bool(anim_var.get()) and scope_var.get() == "all"
                   and fmt in _EXPORT_ANIMATABLE)
        page = _EXPORT_PAGE_BY_LABEL.get(page_var.get(), "none")
        page_mm = _EXPORT_PAGE_MM.get(page)
        if page == "custom":
            try:
                page_mm = (max(10.0, float(page_w_var.get())),
                           max(10.0, float(page_h_var.get())))
            except ValueError:
                page, page_mm = "none", None
        elif page_mm is None:
            page = "none"
        try:
            bleed = max(0.0, min(50.0, float(bleed_var.get())))
        except ValueError:
            bleed = 0.0
        return {
            "scope": scope_var.get(),
            "format": fmt,
            "quality": int(quality_var.get()),
            "scale": scale,
            "size": size,
            "resample": _EXPORT_RESAMPLE_BY_LABEL.get(resample_var.get(), "lanczos"),
            "trim": bool(trim_var.get()),
            "transparent": bool(transparent_var.get()) and fmt != "JPEG",
            "background": background["value"],
            "animate": animate,
            "frame_delay": frame_delay,
            "loop": bool(loop_var.get()),
            "png": {
                "compress_level": int(png_compress_var.get()),
                "optimize": bool(png_optimize_var.get()),
                "interlace": bool(png_interlace_var.get()),
            },
            "jpeg": {
                "progressive": bool(jpeg_progressive_var.get()),
                "optimize": bool(jpeg_optimize_var.get()),
                "subsampling": _EXPORT_JPEG_SUB_BY_LABEL.get(
                    jpeg_subsampling_var.get(), 2
                ),
            },
            "webp": {
                "lossless": bool(webp_lossless_var.get()),
                "method": int(webp_method_var.get()),
            },
            "gif": {
                "optimize": bool(gif_optimize_var.get()),
            },
            "dpi": dpi,
            "metadata": {
                key: var.get().strip() for key, var in meta_vars.items()
            },
            "page": page,
            "page_mm": page_mm,
            "bleed_mm": bleed,
            "crop_marks": bool(crop_var.get()),
            "cmyk": bool(cmyk_var.get()) and fmt in _EXPORT_CMYK_FORMATS,
            "profile_target": profile_label_to_id.get(profile_var.get(), ""),
            "profile_embed": bool(embed_var.get()),
            "cmyk_profile": cmyk_label_to_id.get(cmyk_profile_var.get(), ""),
        }

    result = {"value": None}

    def on_ok(_event=None):
        result["value"] = collect()
        window.destroy()

    buttons = tk.Frame(footer)
    buttons.pack(side=tk.RIGHT)
    tk.Button(buttons, text=action, width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )
    estimate_label.pack(side=tk.LEFT, padx=(0, 8))

    for variable in (format_var, transparent_var, webp_lossless_var, scope_var,
                     page_var, cmyk_var):
        variable.trace_add("write", sync)
    for variable in (quality_var, resample_var, dpi_var, png_compress_var,
                     png_optimize_var, png_interlace_var, jpeg_progressive_var,
                     jpeg_optimize_var, jpeg_subsampling_var, webp_method_var,
                     gif_optimize_var, trim_var, anim_var, delay_var, loop_var,
                     page_w_var, page_h_var, bleed_var, crop_var, cmyk_var,
                     profile_var, embed_var, cmyk_profile_var):
        variable.trace_add("write", mark_stale)
    scale_var.trace_add("write", refresh_size)
    width_var.trace_add("write", on_width)
    height_var.trace_add("write", on_height)

    refresh_size()
    sync()

    window.bind("<Return>", on_ok)
    _run(root, window, window.destroy, focus=format_menu)
    return result["value"]


def ask_color_management(root, working="sRGB", proof=None, proof_intent="relative",
                         make_default=False):
    """Prompt for the active canvas's ICC colour-management settings.

    Returns a dict with:

    * ``working_profile`` - the canvas working-space profile id.
    * ``proof_profile`` - the soft-proof profile id, or ``""`` when off.
    * ``proof_intent`` - the proof rendering intent name.
    * ``make_default`` - use the working profile for new canvases.

    Returns ``None`` when cancelled. Both pickers list the built-in sRGB profile
    and any ``.icc``/``.icm`` files in the user's profile folder; an Import
    button adds a profile from disk.
    """
    window = _make_window(root, "Color Management")
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    items = {"list": colormanagement.list_profiles()}

    def labels():
        return [label for _ident, label in items["list"]]

    def label_for(identifier):
        for ident, label in items["list"]:
            if ident == identifier:
                return label
        return "sRGB"

    def ident_for(label):
        for ident, name in items["list"]:
            if name == label:
                return ident
        return colormanagement.SRGB

    working_frame = tk.LabelFrame(frame, text="Working space", padx=8, pady=6)
    working_frame.pack(fill=tk.X)
    working_var = tk.StringVar(value=label_for(working))
    working_menu = ttk.Combobox(
        working_frame, textvariable=working_var, values=labels(),
        state="readonly", width=30,
    )
    working_menu.grid(row=0, column=0, sticky="w")
    working_hint = tk.Label(
        working_frame,
        text="Layers are composited in this space; opened images are converted "
             "into it.",
        anchor="w", justify="left",
    )
    working_hint.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
    theme.register(working_hint, fg="muted")

    def import_profile():
        path = filedialog.askopenfilename(
            parent=window, title="Import ICC Profile",
            filetypes=[("ICC profiles", "*.icc *.icm"), ("All files", "*.*")],
        )
        if not path:
            return
        identifier = colormanagement.import_profile(path)
        if not identifier:
            show_error(window, "Import Profile",
                       "That file is not a usable ICC profile.")
            return
        items["list"] = colormanagement.list_profiles()
        working_menu.config(values=labels())
        working_var.set(label_for(identifier))

    tk.Button(working_frame, text="Import\u2026", width=10, command=import_profile).grid(
        row=0, column=1, sticky="w", padx=(8, 0)
    )

    proof_frame = tk.LabelFrame(frame, text="Soft proof", padx=8, pady=6)
    proof_frame.pack(fill=tk.X, pady=(8, 0))
    proof_on_var = tk.BooleanVar(value=bool(proof))
    proof_var = tk.StringVar(value=label_for(proof) if proof else label_for("sRGB"))
    intent_var = tk.StringVar(value=colormanagement.intent_label(proof_intent))
    proof_check = tk.Checkbutton(
        proof_frame, text="Simulate an output profile", variable=proof_on_var,
        anchor="w",
    )
    proof_check.grid(row=0, column=0, columnspan=2, sticky="w")
    tk.Label(proof_frame, text="Profile:", anchor="w").grid(
        row=1, column=0, sticky="w", pady=2
    )
    proof_menu = ttk.Combobox(
        proof_frame, textvariable=proof_var, values=labels(),
        state="readonly", width=30,
    )
    proof_menu.grid(row=1, column=1, sticky="w", pady=2)
    tk.Label(proof_frame, text="Intent:", anchor="w").grid(
        row=2, column=0, sticky="w", pady=2
    )
    intent_menu = ttk.Combobox(
        proof_frame, textvariable=intent_var,
        values=[colormanagement.intent_label(n) for n in colormanagement.intent_names()],
        state="readonly", width=30,
    )
    intent_menu.grid(row=2, column=1, sticky="w", pady=2)

    def sync(*_args):
        state = tk.NORMAL if proof_on_var.get() else tk.DISABLED
        proof_menu.config(state="readonly" if proof_on_var.get() else tk.DISABLED)
        intent_menu.config(state="readonly" if proof_on_var.get() else tk.DISABLED)

    proof_on_var.trace_add("write", sync)
    sync()

    default_var = tk.BooleanVar(value=bool(make_default))
    tk.Checkbutton(
        frame, text="Use this working space for new canvases",
        variable=default_var, anchor="w",
    ).pack(fill=tk.X, pady=(8, 0))

    result = {"value": None}

    def on_ok(_event=None):
        result["value"] = {
            "working_profile": ident_for(working_var.get()),
            "proof_profile": ident_for(proof_var.get()) if proof_on_var.get() else "",
            "proof_intent": next(
                (name for name in colormanagement.intent_names()
                 if colormanagement.intent_label(name) == intent_var.get()),
                "relative",
            ),
            "make_default": bool(default_var.get()),
        }
        window.destroy()

    buttons = tk.Frame(frame)
    buttons.pack(fill=tk.X, pady=(10, 0))
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    window.bind("<Return>", on_ok)
    _run(root, window, window.destroy, focus=working_menu)
    return result["value"]


MIN_TEXT_SIZE = 4
MAX_TEXT_SIZE = 1000


def ask_text(root, initial=None):
    """A centered prompt for text-tool content and its font.

    Returns a dict with ``text``, ``family``, ``size``, ``bold``, ``italic`` and
    ``color`` (a ``#rrggbb`` string), or ``None`` when cancelled. Empty text is
    rejected so the caller never has to handle a no-op commit.
    """
    initial = initial or {}
    family_names = fonts.families()
    window = _make_window(root, "Add Text")
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(frame, text="Text", anchor="w").grid(
        row=0, column=0, columnspan=3, sticky="w"
    )
    text_holder = tk.Frame(frame)
    text_holder.grid(row=1, column=0, columnspan=3, sticky="we", pady=(2, 8))
    text_widget = tk.Text(text_holder, width=40, height=5, wrap=tk.WORD)
    text_scroll = AutoScrollbar(text_holder, command=text_widget.yview)
    text_widget.configure(yscrollcommand=text_scroll.set)
    text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    text_widget.insert("1.0", initial.get("text", ""))

    family_var = tk.StringVar(
        value=initial.get("family") or fonts.default_family()
    )
    tk.Label(frame, text="Font", anchor="w").grid(row=2, column=0, sticky="w")
    family_box = ttk.Combobox(
        frame, textvariable=family_var, values=family_names, state="readonly",
        width=20,
    )
    family_box.grid(row=2, column=1, sticky="we", pady=2)

    def cycle_font(step):
        if not family_names:
            return
        try:
            index = family_names.index(family_var.get())
        except ValueError:
            index = -1 if step > 0 else 0
        family_var.set(family_names[(index + step) % len(family_names)])
        refresh_preview()

    font_row = tk.Frame(frame)
    font_row.grid(row=2, column=2, sticky="e")
    tk.Button(font_row, text="\u25c2", width=3, command=lambda: cycle_font(-1)).pack(
        side=tk.LEFT
    )
    tk.Button(font_row, text="\u25b8", width=3, command=lambda: cycle_font(1)).pack(
        side=tk.LEFT, padx=(2, 0)
    )

    size_var = tk.StringVar(value=str(initial.get("size", 48)))
    tk.Label(frame, text="Size", anchor="w").grid(row=3, column=0, sticky="w")
    size_box = tk.Spinbox(
        frame, from_=MIN_TEXT_SIZE, to=MAX_TEXT_SIZE, width=6,
        textvariable=size_var,
    )
    size_box.grid(row=3, column=1, sticky="w", pady=2)

    style_row = tk.Frame(frame)
    style_row.grid(row=3, column=2, sticky="e")
    bold_var = tk.BooleanVar(value=bool(initial.get("bold", False)))
    italic_var = tk.BooleanVar(value=bool(initial.get("italic", False)))
    tk.Checkbutton(style_row, text="Bold", variable=bold_var).pack(side=tk.LEFT)
    tk.Checkbutton(style_row, text="Italic", variable=italic_var).pack(
        side=tk.LEFT, padx=(4, 0)
    )

    color = {"value": initial.get("color", "#000000") or "#000000"}
    color_btn = tk.Button(frame, text="Color", width=8)
    # The button doubles as a swatch, so its background is content, not chrome.
    theme.skip(color_btn)
    tk.Label(frame, text="Color", anchor="w").grid(row=4, column=0, sticky="w")

    preview = tk.Label(frame, text="Preview", anchor="w")
    preview.grid(row=5, column=0, columnspan=3, sticky="we", pady=(8, 0))

    def apply_color():
        color_btn.config(bg=color["value"], activebackground=color["value"])

    def choose_color():
        def preview(chosen):
            color["value"] = chosen
            apply_color()
            refresh_preview()

        ask_color(window, color["value"], title="Text Color", on_preview=preview)

    color_btn.config(command=choose_color)
    color_btn.grid(row=4, column=1, sticky="w", pady=2)
    apply_color()

    def current_size():
        try:
            size = int(size_var.get())
        except (tk.TclError, ValueError):
            return None
        return max(MIN_TEXT_SIZE, min(size, MAX_TEXT_SIZE))

    preview_font = {"obj": None}

    def refresh_preview(_event=None):
        family = family_var.get() or fonts.default_family()
        size = current_size() or 48
        preview_size = max(8, min(size, 32))
        spec = {
            "family": family,
            "size": preview_size,
        }
        if bold_var.get():
            spec["weight"] = "bold"
        if italic_var.get():
            spec["slant"] = "italic"
        try:
            preview_font["obj"] = tkfont.Font(**spec)
            preview.config(font=preview_font["obj"], fg=color["value"])
        except tk.TclError:
            pass
        preview.config(text=text_widget.get("1.0", "end-1c") or "Preview")

    family_box.bind("<<ComboboxSelected>>", refresh_preview)
    size_box.bind("<KeyRelease>", refresh_preview)
    text_widget.bind("<KeyRelease>", refresh_preview)
    bold_var.trace_add("write", lambda *_a: refresh_preview())
    italic_var.trace_add("write", lambda *_a: refresh_preview())
    refresh_preview()

    error_var = tk.StringVar(value="")
    tk.Label(frame, textvariable=error_var, fg="red", anchor="w").grid(
        row=6, column=0, columnspan=3, sticky="w", pady=(4, 0)
    )

    result = {"value": None}

    def on_ok(_event=None):
        text = text_widget.get("1.0", "end-1c")
        if not text.strip():
            error_var.set("Enter some text.")
            return
        size = current_size()
        if size is None:
            error_var.set("Enter a whole-number size.")
            return
        result["value"] = {
            "text": text,
            "family": family_var.get() or fonts.default_family(),
            "size": size,
            "bold": bold_var.get(),
            "italic": italic_var.get(),
            "color": color["value"],
        }
        window.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(row=7, column=0, columnspan=3, sticky="e", pady=(8, 0))
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    window.bind("<Control-Return>", on_ok)
    _run(root, window, window.destroy, focus=text_widget)
    return result["value"]


def ask_color(root, initial, title="Color", on_preview=None, persist_recent=True,
              on_pick=None):
    """Pick a color with the app's own picker, applying changes live.

    ``initial`` is the current ``#rrggbb`` string. ``on_preview(color)`` fires
    on every change so the caller can apply it immediately; Cancel restores
    ``initial`` through ``on_preview`` and returns ``None``. OK returns the
    chosen ``#rrggbb`` string. This replaces the native ``colorchooser`` dialog
    everywhere a color is chosen.

    ``on_pick`` (optional) enables a "Pick on Canvas" button. It is called with
    a ``use(color)`` callback; the dialog hides itself, the caller samples a
    color from the canvas, and then invokes ``use`` to show the dialog again
    with that color selected.
    """
    from .colorpicker import ColorPicker

    def to_rgb(color):
        return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

    initial = initial or "#000000"
    window = _make_window(root, title)
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    picker = ColorPicker(frame, persist_recent=persist_recent)
    picker.pack()
    picker.set_rgb(*to_rgb(initial))
    picker.on_change = on_preview

    result = {"value": None}

    def cancel():
        if on_preview is not None:
            on_preview(initial)
        window.destroy()

    def on_ok(_event=None):
        result["value"] = picker.color
        window.destroy()

    def start_pick():
        """Hide the dialog and let the caller sample a color from the canvas."""
        if on_pick is None:
            return
        window.grab_release()
        window.withdraw()

        def use(color):
            picker.set_rgb(*to_rgb(color))
            window.deiconify()
            window.lift()
            window.focus_force()
            window.grab_set()

        on_pick(use)

    buttons = tk.Frame(frame)
    buttons.pack(fill=tk.X, pady=(8, 0))
    if on_pick is not None:
        tk.Button(buttons, text="Pick on Canvas", command=start_pick).pack(
            side=tk.LEFT
        )
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=cancel).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    window.bind("<Return>", on_ok)
    _run(root, window, cancel)
    return result["value"]


def ask_offscreen_color(root, initial, default, on_preview):
    """Pick an off-canvas background color with live preview.

    ``initial`` is the current ``(r, g, b)`` and ``default`` the color restored
    by the Reset button. ``on_preview(rgb)`` fires on every change so the caller
    can repaint immediately instead of waiting for OK. Returns the chosen triple
    or ``None`` when cancelled, after restoring ``initial``.
    """
    from .colorpicker import ColorPicker

    def to_rgb(color):
        return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))

    initial = tuple(initial)
    default = tuple(default)
    window = _make_window(root, "Off-Canvas Background Color")
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    picker = ColorPicker(frame, persist_recent=False)
    picker.pack()
    picker.set_rgb(*initial)

    picker.on_change = lambda color: on_preview(to_rgb(color))

    def reset():
        picker.set_rgb(*default)

    def cancel():
        on_preview(initial)
        window.destroy()

    result = {"value": None}

    def on_ok(_event=None):
        result["value"] = to_rgb(picker.color)
        window.destroy()

    buttons = tk.Frame(frame)
    buttons.pack(fill=tk.X, pady=(8, 0))
    tk.Button(buttons, text="Reset", width=8, command=reset).pack(side=tk.LEFT)
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=cancel).pack(
        side=tk.RIGHT, padx=(0, 6)
    )

    window.bind("<Return>", on_ok)
    _run(root, window, cancel)
    return result["value"]


class _CurveEditor(tk.Canvas):
    """A draggable tone-curve editor for the Curves filter dialog.

    Points are ``[x, y]`` pairs in ``0..255`` (input, output). The first and last
    points keep x pinned to 0 and 255 but move vertically; interior points slide
    between their neighbours. Clicking empty space adds a point and right-clicking
    an interior point removes it. The drawn curve comes from the same
    ``filters.curve_lut`` used to apply the filter, so it matches the result.
    """

    SIZE = 200
    PAD = 10
    HIT = 7

    def __init__(self, parent, points, on_change=None):
        super().__init__(
            parent, width=self.SIZE, height=self.SIZE, highlightthickness=1,
            background="#1e1e1e",
        )
        from . import filters as registry
        self._registry = registry
        self.points = [list(point) for point in registry.coerce_curve(points)]
        self.on_change = on_change
        self._drag = None
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.bind("<Button-3>", self._on_remove)
        self._redraw()

    def _span(self):
        return self.SIZE - 2 * self.PAD

    def _to_canvas(self, x, y):
        span = self._span()
        return (self.PAD + x / 255.0 * span, self.PAD + (1.0 - y / 255.0) * span)

    def _to_graph(self, cx, cy):
        span = self._span()
        x = (cx - self.PAD) / span * 255.0
        y = (1.0 - (cy - self.PAD) / span) * 255.0
        return max(0.0, min(255.0, x)), max(0.0, min(255.0, y))

    def _redraw(self):
        self.delete("all")
        span = self._span()
        for i in range(5):
            gx = self.PAD + i * span / 4
            self.create_line(gx, self.PAD, gx, self.PAD + span, fill="#333333")
            self.create_line(self.PAD, gx, self.PAD + span, gx, fill="#333333")
        self.create_line(
            *self._to_canvas(0, 0), *self._to_canvas(255, 255),
            fill="#444444", dash=(4, 2),
        )
        lut = self._registry.curve_lut(self.points)
        coords = []
        for value in range(256):
            coords.extend(self._to_canvas(value, lut[value]))
        self.create_line(*coords, fill="#7ec8ff", width=2)
        for point in self.points:
            cx, cy = self._to_canvas(point[0], point[1])
            radius = 3
            self.create_rectangle(
                cx - radius, cy - radius, cx + radius, cy + radius,
                fill="#ffffff", outline="#101010",
            )

    def _nearest(self, cx, cy):
        best = None
        best_distance = float(self.HIT)
        for index, point in enumerate(self.points):
            px, py = self._to_canvas(point[0], point[1])
            distance = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
            if distance <= best_distance:
                best_distance = distance
                best = index
        return best

    def _on_press(self, event):
        hit = self._nearest(event.x, event.y)
        if hit is None:
            x, y = self._to_graph(event.x, event.y)
            if any(abs(point[0] - x) < 3 for point in self.points):
                return
            self.points.append([int(round(x)), int(round(y))])
            self.points.sort(key=lambda point: point[0])
            hit = self.points.index(
                min(self.points, key=lambda point: abs(point[0] - x))
            )
        self._drag = hit
        self._redraw()
        self._changed()

    def _on_motion(self, event):
        if self._drag is None:
            return
        index = self._drag
        x, y = self._to_graph(event.x, event.y)
        x = int(round(x))
        y = int(round(y))
        if index == 0:
            x = 0
        elif index == len(self.points) - 1:
            x = 255
        else:
            low = self.points[index - 1][0] + 1
            high = self.points[index + 1][0] - 1
            x = max(low, min(high, x))
        self.points[index] = [x, max(0, min(255, y))]
        self._redraw()
        self._changed()

    def _on_release(self, _event):
        self._drag = None

    def _on_remove(self, event):
        index = self._nearest(event.x, event.y)
        if index is None or index in (0, len(self.points) - 1):
            return
        del self.points[index]
        self._redraw()
        self._changed()

    def _changed(self):
        if self.on_change is not None:
            self.on_change()


def ask_filter(root, spec, delete_label=None, on_preview=None):
    """Prompt for a filter's parameters, using its registry definition.

    ``spec`` is a filter spec dict (see :mod:`paintapp.filters`). Returns a dict
    of the edited parameter values (typed per the registry), or ``None`` when
    cancelled. Filters without parameters return an empty dict. When
    ``delete_label`` is given, its button returns ``DELETED`` instead. When
    ``on_preview`` is given, it is called with the current parameter dict as the
    controls change (and once when the dialog opens) so the caller can render a
    live preview.
    """
    from . import filters as registry

    entry = registry.FILTERS.get(spec.get("type"))
    if entry is None:
        return None
    current = spec.get("params", {})

    window = _make_window(root, f"Filter: {entry['label']}")
    frame = tk.Frame(window, padx=12, pady=12)
    frame.pack(fill=tk.BOTH, expand=True)

    variables = {}
    labels = {}
    choice_labels = {}
    kinds = {param["key"]: param["kind"] for param in entry["params"]}
    first_focus = None
    row = 0
    for param in entry["params"]:
        key = param["key"]
        value = current.get(key, param["default"])
        kind = param["kind"]
        if kind == "bool":
            variable = tk.BooleanVar(value=bool(value))
            tk.Checkbutton(
                frame, text=param["label"], variable=variable, anchor="w",
                command=lambda: refresh(),
            ).grid(row=row, column=0, columnspan=3, sticky="w", pady=2)
            variables[key] = variable
            row += 1
        elif kind == "curve":
            tk.Label(frame, text=param["label"], anchor="w").grid(
                row=row, column=0, columnspan=3, sticky="w", pady=(4, 2)
            )
            row += 1
            editor = _CurveEditor(frame, value, on_change=lambda: refresh())
            editor.grid(row=row, column=0, columnspan=3, pady=2)
            variables[key] = editor
            tk.Label(
                frame,
                text="Drag points to shape the curve; click to add, "
                     "right-click to remove.",
                anchor="w", fg="#888888",
            ).grid(row=row + 1, column=0, columnspan=3, sticky="w")
            if first_focus is None:
                first_focus = editor
            row += 2
        elif kind == "color":
            tk.Label(frame, text=param["label"], anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 6), pady=2
            )
            variable = tk.StringVar(value=str(value))
            swatch = tk.Button(frame, width=10)

            def apply_swatch(button=swatch, var=variable):
                color = var.get() or "#000000"
                button.config(bg=color, activebackground=color)

            def choose(k=key, title=param["label"], apply=apply_swatch):
                def preview(color):
                    variables[k].set(color)
                    apply()
                    refresh()
                ask_color(frame.winfo_toplevel(), variables[k].get(),
                          title=title, on_preview=preview)
                apply()

            swatch.config(command=choose)
            swatch.grid(row=row, column=1, sticky="w", pady=2)
            theme.skip(swatch)
            apply_swatch()
            variables[key] = variable
            row += 1
        elif kind == "choice":
            tk.Label(frame, text=param["label"], anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 6), pady=2
            )
            options = param.get("options", [])
            label_for = {opt[0]: opt[1] for opt in options}
            value_for = {opt[1]: opt[0] for opt in options}
            variable = tk.StringVar(value=label_for.get(str(value), str(value)))
            combo = ttk.Combobox(
                frame, textvariable=variable, state="readonly", width=14,
                values=[opt[1] for opt in options],
            )
            combo.grid(row=row, column=1, sticky="w", pady=2)
            combo.bind("<<ComboboxSelected>>", lambda *_: refresh())
            variables[key] = variable
            choice_labels[key] = value_for
            row += 1
        else:
            tk.Label(frame, text=param["label"], anchor="w").grid(
                row=row, column=0, sticky="w", padx=(0, 6), pady=2
            )
            variable = tk.DoubleVar(value=float(value))
            value_var = tk.StringVar()
            scale = tk.Scale(
                frame, from_=param["min"], to=param["max"], orient=tk.HORIZONTAL,
                variable=variable, resolution=param["step"], length=180,
                command=lambda *_: refresh(),
            )
            scale.grid(row=row, column=1, sticky="we", pady=2)
            tk.Label(
                frame, textvariable=value_var, width=7, anchor="w",
            ).grid(row=row, column=2, sticky="w")
            labels[key] = value_var
            if first_focus is None:
                first_focus = scale
            variables[key] = variable
            row += 1

    if not entry["params"]:
        tk.Label(
            frame, text="This filter has no options.", anchor="w",
            fg="#888888",
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=2)

    def collect():
        params = {}
        for param in entry["params"]:
            key = param["key"]
            if param["kind"] == "bool":
                params[key] = bool(variables[key].get())
                continue
            if param["kind"] == "curve":
                params[key] = [
                    list(point) for point in variables[key].points
                ]
                continue
            if param["kind"] == "color":
                params[key] = str(variables[key].get())
                continue
            if param["kind"] == "choice":
                shown = str(variables[key].get())
                params[key] = choice_labels.get(key, {}).get(shown, shown)
                continue
            value = variables[key].get()
            value = max(param["min"], min(param["max"], value))
            params[key] = int(round(value)) if param["kind"] == "int" else float(value)
        return params

    def refresh(*_args):
        for key, value_var in labels.items():
            value = variables[key].get()
            if kinds.get(key) == "int":
                value_var.set(f"{int(round(value))}")
            else:
                value_var.set(f"{value:g}")
        if on_preview is not None:
            on_preview(collect())

    refresh()

    result = {"value": None}

    def on_ok(_event=None):
        result["value"] = collect()
        window.destroy()

    def on_delete():
        result["value"] = DELETED
        window.destroy()

    buttons = tk.Frame(frame)
    buttons.grid(
        row=row + 1, column=0, columnspan=3, sticky="e",
        pady=(8, 0),
    )
    tk.Button(buttons, text="OK", width=8, command=on_ok).pack(side=tk.RIGHT)
    tk.Button(buttons, text="Cancel", width=8, command=window.destroy).pack(
        side=tk.RIGHT, padx=(0, 6)
    )
    if delete_label:
        tk.Button(buttons, text=delete_label, width=8, command=on_delete).pack(
            side=tk.LEFT
        )

    window.bind("<Return>", on_ok)
    _run(root, window, window.destroy, focus=first_focus)
    return result["value"]

