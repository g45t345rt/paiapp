"""Light and dark themes for the application chrome.

The app builds its widgets with plain, hardcoded colors. Rather than making
every component theme-aware, this module keeps a pair of color tables and
restyles the whole widget tree in place:

* :func:`apply` walks the tree from the root and, per widget class, sets the
  chrome options (background, foreground, trough, selection, ...) from the
  active table.
* Content widgets that must keep their own colors - palette swatches, the
  canvas pixels, colored legend chips - are excluded with :func:`skip` or
  :func:`skip_bg`, or given explicit tokens with :func:`register`.

Only *default* foregrounds are replaced, so labels that intentionally use a
custom color (error red, muted grey) keep it. ttk widgets
are handled through named styles in :func:`configure_styles`.
"""

import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk

from engine import settings as engine_settings

LIGHT = {
    "bg": "#f0f0f0",
    "panel_bg": "#f0f0f0",
    "stripe_bg": "#e7e7e7",
    "text": "#000000",
    "muted": "#888888",
    "button_bg": "#f0f0f0",
    "button_active": "#dbe9f6",
    "button_hover": "#d0e4f5",
    "border": "#999999",
    "trough": "#e6e6e6",
    "entry_bg": "#ffffff",
    "listbox_bg": "#ffffff",
    "sel_bg": "#0078d7",
    "sel_fg": "#ffffff",
    "accent": "#0078d7",
    "error": "#cc0000",
    # Message severities: a foreground and a matching background tint used by
    # the status bar and the Logs panel. ``info`` is the neutral default.
    "info": "#0b6fc2",
    "info_bg": "#e9f2fb",
    "success": "#1a7f37",
    "success_bg": "#e6f4ea",
    "warn": "#8a6100",
    "warn_bg": "#fdf4dd",
    "error_bg": "#fbe9e9",
    # Sidebar chrome.
    "header_bg": "#dcdcdc",
    "header_fg": "#333333",
    "header_active_bg": "#c8dcf0",
    "splitter": "#e6e6e6",
    "splitter_line": "#b0b0b0",
    "splitter_hover": "#d0e4f5",
    "splitter_hover_line": "#0078d7",
    "float_border": "#7a7a7a",
    "ghost_blend": "#eeeeee",
    # Canvas notebook tabs.
    "notebook_bg": "#d9d9d9",
    "tab_bg": "#c0c0c0",
    "tab_fg": "#555555",
    "tab_active_bg": "#d0d0d0",
    "tab_active_fg": "#333333",
    "tab_selected_bg": "#0078d7",
    "tab_selected_fg": "#ffffff",
    "tab_border": "#999999",
    # Canvas ruler strips.
    "ruler_bg": "#f5f5f5",
    "ruler_tick": "#8a8a8a",
    "ruler_text": "#666666",
    "guide_line": "#00a0d8",
    "grid_line": "#9a9a9a",
}

DARK = {
    "bg": "#2b2b2b",
    "panel_bg": "#2b2b2b",
    "stripe_bg": "#333333",
    "text": "#e6e6e6",
    "muted": "#9a9a9a",
    "button_bg": "#3a3a3a",
    "button_active": "#4a4a4a",
    "button_hover": "#4a4a4a",
    "border": "#555555",
    "trough": "#3a3a3a",
    "entry_bg": "#1f1f1f",
    "listbox_bg": "#252526",
    "sel_bg": "#094771",
    "sel_fg": "#ffffff",
    "accent": "#4a9eff",
    "error": "#ff6b6b",
    # Message severities: a foreground and a matching background tint used by
    # the status bar and the Logs panel. ``info`` is the neutral default.
    "info": "#4a9eff",
    "info_bg": "#17283a",
    "success": "#3fb950",
    "success_bg": "#12291b",
    "warn": "#d29922",
    "warn_bg": "#2e2612",
    "error_bg": "#3a1d1d",
    # Sidebar chrome.
    "header_bg": "#3a3a3a",
    "header_fg": "#dcdcdc",
    "header_active_bg": "#4a6b8a",
    "splitter": "#333333",
    "splitter_line": "#555555",
    "splitter_hover": "#3f5a75",
    "splitter_hover_line": "#4a9eff",
    "float_border": "#666666",
    "ghost_blend": "#2b2b2b",
    # Canvas notebook tabs.
    "notebook_bg": "#2b2b2b",
    "tab_bg": "#3a3a3a",
    "tab_fg": "#c8c8c8",
    "tab_active_bg": "#454545",
    "tab_active_fg": "#ffffff",
    "tab_selected_bg": "#094771",
    "tab_selected_fg": "#ffffff",
    "tab_border": "#555555",
    # Canvas ruler strips.
    "ruler_bg": "#333333",
    "ruler_tick": "#7a7a7a",
    "ruler_text": "#9a9a9a",
    "guide_line": "#28c0ff",
    "grid_line": "#5a5a5a",
}

THEMES = {"light": LIGHT, "dark": DARK}

# Severity levels for status and log messages, mapping each to its
# ``(foreground, background)`` theme tokens. ``info`` is the neutral default for
# anything that is not a warning, a success, or an error.
LEVELS = {
    "info": ("info", "info_bg"),
    "success": ("success", "success_bg"),
    "warn": ("warn", "warn_bg"),
    "error": ("error", "error_bg"),
}

# Options applied to every Tk menu (ignored by the native Windows menubar, but
# honored elsewhere and by popup menus).
MENU_OPTS = {
    "bg": "panel_bg",
    "fg": "text",
    "activebackground": "sel_bg",
    "activeforeground": "sel_fg",
    "selectcolor": "entry_bg",
    "disabledforeground": "muted",
}

_TTK_CLASSES = {
    "TCombobox", "TProgressbar", "TScrollbar", "TSpinbox", "TSizegrip",
    "TNotebook", "TButton", "TLabel", "TFrame", "TEntry", "TCheckbutton",
    "TRadiobutton", "TScale", "TLabelframe", "TMenubutton",
}

# Foreground values that mean "the platform default" and may be replaced.
_DEFAULT_FG = {
    "black", "#000000", "#000", "systembuttontext", "systemwindowtext",
    "systemmenutext",
}

_state = {"name": "light", "root": None}
_owned = {}      # widget -> {option: token} set through register()
_no_bg = set()   # widgets whose background must be left alone
_skip = set()    # widgets excluded from restyling entirely
_controlled = {}  # widget -> set(options) this module has taken over
_subscribers = []


# --------------------------------------------------------------- persistence
def read_settings():
    """Return the persisted settings dict, or ``{}`` when none is stored."""
    return engine_settings.read()


def write_settings(settings):
    """Persist ``settings`` (a dict) to the app's settings file."""
    engine_settings.write(settings)


def os_preference():
    """Return the OS-wide light/dark preference ("light" or "dark").

    Reads the Windows apps theme, the macOS interface style, or the
    freedesktop color-scheme hint in turn. Falls back to ``"light"`` whenever
    the preference cannot be determined, matching the app's historical default.
    """
    try:
        if os.name == "nt":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if value else "dark"
        if sys.platform == "darwin":
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                capture_output=True, text=True, timeout=2,
            )
            return "dark" if result.stdout.strip() == "Dark" else "light"
        result = subprocess.run(
            ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
            capture_output=True, text=True, timeout=2,
        )
        return "dark" if "dark" in result.stdout.strip().lower() else "light"
    except Exception:
        return "light"


def load_preference():
    """Return the saved theme name.

    A saved ``"light"``/``"dark"`` is used as-is. When the user has chosen
    ``"system"`` (or never chosen a theme), fall back to the operating system's
    light/dark preference so a fresh install matches the desktop.
    """
    name = read_settings().get("theme")
    if name in THEMES:
        return name
    return os_preference()


def save_preference(name):
    """Remember ``name`` ("light", "dark" or "system") for the next launch."""
    settings = read_settings()
    settings["theme"] = name
    write_settings(settings)


# ------------------------------------------------------- native window frame
def _dwm_colorref(hex_color):
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return (b << 16) | (g << 8) | r


def _attach_titlebar_remap(window):
    """Reapply the title bar colors once ``window`` is mapped.

    Windows ignores the DWM caption/border/text attributes when they are set
    before a toplevel is first shown, which is the case for the centered modal
    dialogs (they are themed during construction, then mapped by the main
    loop). Reapplying on ``<Map>`` makes dialog title bars follow the theme.
    """
    if getattr(window, "_titlebar_remap_bound", False):
        return
    window._titlebar_remap_bound = True

    def on_map(_event):
        _set_titlebar_dark(window, is_dark())

    window.bind("<Map>", on_map, add="+")


def _set_titlebar_dark(window, dark):
    """Ask Windows to draw ``window``'s title bar and border in the theme.

    Uses the DWM immersive-dark-mode attribute (Windows 10 1809+) and the
    caption/border color attributes (Windows 11). Silently does nothing on
    other platforms or when the calls are unavailable.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
    except Exception:
        return
    try:
        if not window.winfo_ismapped():
            _attach_titlebar_remap(window)
        window.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(window.winfo_id())
        if not hwnd:
            hwnd = window.winfo_id()
        dwmapi = ctypes.windll.dwmapi
        value = ctypes.c_int(1 if dark else 0)
        for attribute in (20, 19):  # DWMWA_USE_IMMERSIVE_DARK_MODE
            if dwmapi.DwmSetWindowAttribute(
                hwnd, attribute, ctypes.byref(value), ctypes.sizeof(value)
            ) == 0:
                break
        colors = {
            34: "#2b2b2b" if dark else None,   # DWMWA_BORDER_COLOR
            35: "#1f1f1f" if dark else None,   # DWMWA_CAPTION_COLOR
            36: "#e6e6e6" if dark else None,   # DWMWA_TEXT_COLOR
        }
        for attribute, hex_color in colors.items():
            color = -1 if hex_color is None else _dwm_colorref(hex_color)
            arg = ctypes.c_int(color)
            try:
                dwmapi.DwmSetWindowAttribute(
                    hwnd, attribute, ctypes.byref(arg), ctypes.sizeof(arg)
                )
            except Exception:
                pass
    except Exception:
        pass


def refresh_native_frame(window):
    """Reapply the native title bar colors after ``window``'s frame is rebuilt.

    Windows discards a toplevel's DWM dark-mode attributes when the native
    frame is recreated, which happens when the window enters or leaves
    fullscreen. Calling this once the new frame exists keeps dark mode from
    reverting to a white caption after a fullscreen round trip.
    """
    _set_titlebar_dark(window, is_dark())


# ------------------------------------------------------------------- tokens
def current():
    return _state["name"]


def is_dark():
    return _state["name"] == "dark"


def color(token):
    return THEMES[_state["name"]].get(token)


def subscribe(callback):
    """Register ``callback`` to run after every theme change."""
    _subscribers.append(callback)


# --------------------------------------------------------------- registration
def register(widget, **options):
    """Pin ``widget`` options to theme tokens and apply them immediately."""
    _owned.setdefault(widget, {}).update(options)
    _configure(widget, _owned[widget])


def skip(widget):
    """Exclude ``widget`` from all theming (content that carries its own colors)."""
    _skip.add(widget)


def skip_bg(widget):
    """Leave ``widget``'s background as-is while still theming its other options."""
    _no_bg.add(widget)


def _attach_hover(button):
    """Highlight ``button`` while the pointer is over it.

    Tk does not repaint a classic ``Button``'s background on pointer-over on
    Windows (``activebackground`` only applies to the pressed state there), so
    the highlight is swapped in explicitly and the base color restored on
    leaving. Both colors are read at event time, so a theme switch is picked
    up without rebinding.
    """
    if getattr(button, "_hover_bound", False):
        return
    button._hover_bound = True

    def on_enter(_event):
        if str(button.cget("state")) == "disabled":
            return
        button.configure(bg=color("button_hover"))

    def on_leave(_event):
        button.configure(bg=color("button_bg"))

    button.bind("<Enter>", on_enter, add="+")
    button.bind("<Leave>", on_leave, add="+")


# ------------------------------------------------------------------ restyle
def init(root, name="light"):
    """Adopt ``root``, select ``name`` and prepare the ttk styles."""
    _state["root"] = root
    if name in THEMES:
        _state["name"] = name
    configure_styles(root)


def apply(name=None):
    """Switch to ``name`` (when given) and restyle every live widget."""
    if name in THEMES:
        _state["name"] = name
    _prune()
    root = _state["root"]
    configure_styles(root)
    if root is not None:
        restyle(root)
        try:
            root.after_idle(lambda: _set_titlebar_dark(root, is_dark()))
        except tk.TclError:
            pass
    for callback in list(_subscribers):
        try:
            callback()
        except tk.TclError:
            pass


def restyle(widget):
    """Restyle ``widget`` and its descendants for the active theme."""
    _apply_widget(widget)
    for child in widget.winfo_children():
        restyle(child)


def _configure(widget, options):
    config = {}
    for option, token in options.items():
        value = color(token)
        if value is not None:
            config[option] = value
    if not config:
        return
    try:
        widget.configure(**config)
    except tk.TclError:
        for option, value in config.items():
            try:
                widget.configure(**{option: value})
            except tk.TclError:
                pass


def _alive(widget):
    try:
        return bool(widget.winfo_exists())
    except tk.TclError:
        return False


def _prune():
    for store in (_owned, _controlled):
        for widget in [w for w in store if not _alive(w)]:
            del store[widget]
    for store in (_skip, _no_bg):
        for widget in [w for w in store if not _alive(w)]:
            store.discard(widget)


def _want_fg(widget):
    controlled = _controlled.setdefault(widget, set())
    if "fg" in controlled:
        return True
    try:
        value = str(widget.cget("fg")).strip().lower()
    except tk.TclError:
        return False
    if value in _DEFAULT_FG:
        controlled.add("fg")
        return True
    return False


def _class_options(widget, cls):
    options = {}
    if cls in ("Tk", "Toplevel", "Frame", "Labelframe"):
        options["bg"] = "bg"
        if cls == "Labelframe" and _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Label":
        options["bg"] = "bg"
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls in ("Button", "Menubutton"):
        options.update(bg="button_bg", activebackground="button_active",
                       activeforeground="text", disabledforeground="muted",
                       highlightbackground="border", highlightcolor="accent")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls in ("Checkbutton", "Radiobutton"):
        options.update(bg="bg", activebackground="bg", selectcolor="entry_bg",
                       disabledforeground="muted", highlightbackground="border",
                       highlightcolor="accent")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Entry":
        options.update(bg="entry_bg", insertbackground="text",
                       disabledbackground="bg", disabledforeground="muted",
                       highlightbackground="border", highlightcolor="accent")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Text":
        options.update(bg="entry_bg", insertbackground="text",
                       selectbackground="sel_bg", selectforeground="sel_fg",
                       highlightbackground="border", highlightcolor="accent")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Listbox":
        options.update(bg="listbox_bg", selectbackground="sel_bg",
                       selectforeground="sel_fg", highlightbackground="border",
                       highlightcolor="accent")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Scale":
        options.update(bg="bg", troughcolor="trough", activebackground="accent",
                       highlightbackground="border")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Spinbox":
        options.update(bg="entry_bg", buttonbackground="button_bg",
                       insertbackground="text", disabledbackground="bg",
                       disabledforeground="muted", highlightbackground="border",
                       highlightcolor="accent")
        if _want_fg(widget):
            options["fg"] = "text"
    elif cls == "Scrollbar":
        options.update(bg="border", troughcolor="trough",
                       activebackground="accent", highlightbackground="bg",
                       highlightcolor="accent")
    elif cls == "Menu":
        options.update(bg="panel_bg", activebackground="sel_bg",
                       activeforeground="sel_fg", selectcolor="entry_bg",
                       disabledforeground="muted")
        if _want_fg(widget):
            options["fg"] = "text"
    return options


def _apply_widget(widget):
    if widget in _skip:
        return
    cls = widget.winfo_class()
    if cls in ("Tk", "Toplevel"):
        _set_titlebar_dark(widget, is_dark())
    if cls == "Canvas" or cls in _TTK_CLASSES:
        return
    options = _class_options(widget, cls)
    if widget in _no_bg:
        options.pop("bg", None)
    options.update(_owned.get(widget, {}))
    _configure(widget, options)
    if cls == "Button":
        _attach_hover(widget)


# ------------------------------------------------------------- ttk styles
def configure_styles(root=None):
    root = root or _state["root"]
    if root is None:
        return
    t = THEMES[_state["name"]]
    style = ttk.Style(root)
    try:
        style.theme_use("default")
    except tk.TclError:
        pass

    style.configure(
        "TCombobox",
        fieldbackground=t["entry_bg"], background=t["button_bg"],
        foreground=t["text"], arrowcolor=t["text"], bordercolor=t["border"],
        selectbackground=t["entry_bg"], selectforeground=t["text"],
        focuscolor=t["entry_bg"], focuswidth=0,
        borderwidth=0, padding=(4, 3),
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", t["entry_bg"]), ("disabled", t["bg"])],
        foreground=[("readonly", t["text"]), ("disabled", t["muted"])],
        background=[("readonly", t["button_bg"]), ("disabled", t["bg"])],
        selectbackground=[("readonly", t["entry_bg"]), ("disabled", t["bg"])],
        selectforeground=[("readonly", t["text"]), ("disabled", t["muted"])],
        focuscolor=[("readonly", t["entry_bg"]), ("disabled", t["bg"])],
        arrowcolor=[("disabled", t["muted"])],
    )
    style.configure(
        "TProgressbar",
        background=t["accent"], troughcolor=t["border"], bordercolor=t["border"],
    )
    style.configure(
        "TScrollbar", background=t["border"], troughcolor=t["trough"],
        arrowcolor=t["text"], bordercolor=t["border"],
    )
    style.configure(
        "TSpinbox", fieldbackground=t["entry_bg"], background=t["button_bg"],
        foreground=t["text"], arrowcolor=t["text"], bordercolor=t["border"],
    )
    style.map(
        "TSpinbox",
        fieldbackground=[("disabled", t["bg"])],
        foreground=[("disabled", t["muted"])],
        background=[("disabled", t["bg"])],
        arrowcolor=[("disabled", t["muted"])],
    )
    style.configure("TSizegrip", background=t["bg"])

    # Treeviews (the Hugging Face browser) follow the listbox tokens so their
    # rows and headings read like the rest of the chrome. The default theme's
    # raised heading bevel is flattened to a token color.
    style.configure(
        "Treeview",
        background=t["listbox_bg"], fieldbackground=t["listbox_bg"],
        foreground=t["text"], bordercolor=t["border"], borderwidth=0,
        rowheight=22,
    )
    style.map(
        "Treeview",
        background=[("selected", t["sel_bg"])],
        foreground=[("selected", t["sel_fg"])],
    )
    style.configure(
        "Treeview.Heading",
        background=t["header_bg"], foreground=t["header_fg"],
        bordercolor=t["border"], borderwidth=0, relief="flat", padding=(4, 3),
    )
    style.map(
        "Treeview.Heading",
        background=[("active", t["header_active_bg"])],
    )

    # Notebooks (the Model Settings tabs) reuse the canvas tab tokens so the
    # dialog reads like the rest of the chrome. lightcolor/darkcolor flatten the
    # default theme's raised-tab bevel so only the token colors show.
    style.configure(
        "TNotebook", background=t["bg"], bordercolor=t["border"], borderwidth=0,
        tabmargins=(2, 2, 2, 0),
    )
    style.configure(
        "TNotebook.Tab", background=t["tab_bg"], foreground=t["tab_fg"],
        bordercolor=t["tab_border"], lightcolor=t["tab_bg"], darkcolor=t["tab_bg"],
        borderwidth=0, padding=(14, 5),
    )
    # Drop the "Notebook.focus" element from the tab layout so the selected tab
    # is not ringed by a dashed focus rectangle (the default theme's keyboard
    # focus indicator). Element names match the forced "default" theme.
    style.layout(
        "TNotebook.Tab",
        [("Notebook.tab", {"sticky": "nswe", "children": [
            ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
                ("Notebook.label", {"side": "top", "sticky": ""}),
            ]}),
        ]})],
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", t["tab_active_bg"]), ("active", t["tab_active_bg"])],
        foreground=[("selected", t["tab_active_fg"]), ("active", t["tab_active_fg"])],
    )

    # The combobox popdown listbox is built by ttk when it is first posted, so
    # it never passes through :func:`restyle`. Seed its colors through the
    # option database instead; these apply to each listbox as it is created.
    style.configure(
        "ComboboxPopdownFrame", background=t["border"],
        relief="flat", borderwidth=1,
    )
    for pattern, value in (
        ("*TCombobox*Listbox.background", t["listbox_bg"]),
        ("*TCombobox*Listbox.foreground", t["text"]),
        ("*TCombobox*Listbox.selectBackground", t["sel_bg"]),
        ("*TCombobox*Listbox.selectForeground", t["sel_fg"]),
        ("*TCombobox*Listbox.highlightThickness", "0"),
        ("*TCombobox*Listbox.borderWidth", "0"),
        ("*TCombobox*Listbox.relief", "flat"),
    ):
        root.option_add(pattern, value)

    # A readonly combobox changes its value on the mouse wheel by default
    # (ttk::combobox::Scroll). Replacing that class binding with a no-op stops
    # the pointer from silently changing the selection as it sweeps across a
    # dropdown. It must not return ``"break"``: that would also stop the event
    # at the ``TCombobox`` bindtag, so the ``all``-tag wheel handler the
    # sidebars use to scroll the rail would never run under the pointer.
    root.bind_class("TCombobox", "<MouseWheel>", lambda _event: None)
    root.bind_class("TCombobox", "<Button-4>", lambda _event: None)
    root.bind_class("TCombobox", "<Button-5>", lambda _event: None)
