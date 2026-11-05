"""Searchable font picker with live preview rows and favorites.

The Text settings panel used to offer a plain combobox of family names. This
replaces it with a button that opens a floating popup: a search box filters the
system families, every row renders its own name in that font (so the list is its
own preview), and rows can be starred. Favorites and recently used families are
kept in the app settings so they float to the top of an empty search.

``FontPicker`` is drop-in for a one-line chooser: ``get()`` returns the current
family and ``on_change(family)`` fires when the user picks one.
"""

import tkinter as tk
from tkinter import ttk

from . import fonts, theme

MAX_ROWS = 150
PREVIEW_SIZE = 15
RECENT_LIMIT = 8


# --------------------------------------------------------------- persistence
def favorites():
    """Families starred by the user, in the order they were added."""
    stored = theme.read_settings().get("text_favorite_fonts")
    return [str(name) for name in stored] if isinstance(stored, list) else []


def recents():
    """Recently chosen families, most recent first."""
    stored = theme.read_settings().get("text_recent_fonts")
    return [str(name) for name in stored] if isinstance(stored, list) else []


def toggle_favorite(family):
    """Star ``family`` if it is not a favorite, otherwise unstar it."""
    settings = theme.read_settings()
    stored = settings.get("text_favorite_fonts")
    names = [str(name) for name in stored] if isinstance(stored, list) else []
    names = [name for name in names if name != family] if family in names else names + [family]
    settings["text_favorite_fonts"] = names
    theme.write_settings(settings)


def remember(family):
    """Record ``family`` as the most recently used one."""
    settings = theme.read_settings()
    stored = settings.get("text_recent_fonts")
    names = [str(name) for name in stored] if isinstance(stored, list) else []
    names = [name for name in names if name != family]
    names.insert(0, family)
    settings["text_recent_fonts"] = names[:RECENT_LIMIT]
    theme.write_settings(settings)


# ------------------------------------------------------------------- widget
class FontPicker(tk.Frame):
    """A one-line font chooser that opens a searchable, previewing popup."""

    def __init__(self, parent, family=None, on_change=None):
        super().__init__(parent)
        self._on_change = on_change
        self._family = family or fonts.default_family()
        self._button = tk.Button(
            self, text=self._family, anchor="w", command=self._toggle,
        )
        self._button.pack(fill=tk.X)
        self._popup = None
        self.bind("<Destroy>", self._on_destroy, add="+")

    # ------------------------------------------------------------------ api
    def get(self):
        return self._family

    def set(self, family):
        self._family = family
        self._button.config(text=family)

    # --------------------------------------------------------------- popup
    def _toggle(self):
        if self._popup is not None:
            self._close()
            return
        self._popup = _FontPopup(self, self._family, self._choose)

    def _choose(self, family):
        self.set(family)
        remember(family)
        if self._on_change is not None:
            self._on_change(family)
        self._close()

    def _close(self):
        if self._popup is not None:
            popup, self._popup = self._popup, None
            popup.close()

    def _on_destroy(self, event):
        if event.widget is self:
            self._close()


class _FontPopup:
    """The floating search + preview list for one :class:`FontPicker`."""

    def __init__(self, picker, current, on_choose):
        self.picker = picker
        self.on_choose = on_choose
        self.rows = []      # (family, row_frame, name_label, star_label)
        self.index = 0
        self.query = ""

        self.top = tk.Toplevel(picker)
        self.top.overrideredirect(True)
        self.top.withdraw()
        try:
            self.top.attributes("-topmost", True)
        except tk.TclError:
            pass

        self.entry = tk.Entry(self.top)
        self.entry.pack(fill=tk.X, padx=4, pady=4)
        self.entry.bind("<KeyRelease>", self._on_key)
        self.entry.bind("<Down>", lambda _e: self._move(1))
        self.entry.bind("<Up>", lambda _e: self._move(-1))
        self.entry.bind("<Return>", lambda _e: self._accept())
        self.entry.bind("<Escape>", lambda _e: self.close())
        self.entry.bind("<FocusOut>", self._on_focus_out)

        bg = self._bg()
        self.canvas = tk.Canvas(
            self.top, highlightthickness=0, width=250, height=250, bg=bg,
        )
        self.scroll = ttk.Scrollbar(
            self.top, orient="vertical", command=self.canvas.yview,
        )
        self.canvas.configure(yscrollcommand=self.scroll.set)
        self.scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.inner = tk.Frame(self.canvas, bg=bg)
        self._window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<MouseWheel>", self._on_wheel)

        theme.restyle(self.top)
        self._build_rows("")
        self._place(current)
        self.entry.focus_set()

    # ------------------------------------------------------------ lifecycle
    def close(self):
        try:
            self.top.destroy()
        except tk.TclError:
            pass

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    def _on_focus_out(self, _event):
        focused = self.top.focus_get()
        if focused is self.picker._button:
            return
        try:
            if focused is not None and focused.winfo_toplevel() is self.top:
                return
        except tk.TclError:
            pass
        self.close()

    def _place(self, _current):
        picker = self.picker
        x = picker.winfo_rootx()
        y = picker.winfo_rooty() + picker.winfo_height()
        width = max(250, picker.winfo_width())
        self.top.geometry(f"{width}x290+{x}+{y}")
        self.top.deiconify()

    # --------------------------------------------------------------- rows
    def _bg(self):
        return theme.color("panel_bg") or "#f0f0f0"

    def _build_rows(self, query):
        for child in self.inner.winfo_children():
            child.destroy()
        self.rows = []
        self.query = query
        all_fonts = fonts.families()
        bookmarked = set(favorites())
        needle = query.strip().lower()

        def header(text):
            tk.Label(
                self.inner, text=text, anchor="w",
                bg=theme.color("stripe_bg") or "#e7e7e7",
                fg=theme.color("muted") or "#888888",
                padx=6, pady=3,
            ).pack(fill=tk.X, pady=(6, 1))

        def row(family):
            if len(self.rows) >= MAX_ROWS:
                return
            bg = self._bg()
            frame = tk.Frame(self.inner, bg=bg)
            frame.pack(fill=tk.X)
            name = tk.Label(
                frame, text=family, anchor="w", bg=bg,
                font=(family, PREVIEW_SIZE),
            )
            name.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 2), pady=1)
            star = tk.Label(
                frame, text="\u2605" if family in bookmarked else "\u2606",
                width=2, bg=bg, fg=theme.color("accent"),
            )
            star.pack(side=tk.RIGHT)
            anchor_index = len(self.rows)
            for widget in (frame, name):
                widget.bind("<Button-1>", lambda _e, f=family: self.on_choose(f))
                widget.bind("<Enter>", lambda _e, i=anchor_index: self._highlight(i))
            star.bind("<Button-1>", lambda _e, f=family: self._toggle_favorite(f))
            self.rows.append((family, frame, name, star))

        if needle:
            matched = [
                name for name in all_fonts
                if needle in name.lower() or needle in fonts.category(name).lower()
            ]
            if matched:
                header("Results")
                for family in matched:
                    row(family)
        else:
            fav = [name for name in all_fonts if name in bookmarked]
            if fav:
                header("Favorites")
                for family in fav:
                    row(family)
            recent = [name for name in recents() if name in all_fonts]
            if recent:
                header("Recent")
                for family in recent:
                    row(family)
            header("All fonts")
            for family in all_fonts:
                row(family)

        self.index = 0
        self._highlight(0)
        self.canvas.yview_moveto(0)

    def _toggle_favorite(self, family):
        toggle_favorite(family)
        current = self.rows[self.index][0] if self.rows else ""
        self._build_rows(self.query)
        for position, (name, *_rest) in enumerate(self.rows):
            if name == current:
                self._highlight(position)
                break
        return "break"

    # ----------------------------------------------------------- selection
    def _highlight(self, index):
        if not self.rows:
            return
        index = max(0, min(len(self.rows) - 1, index))
        self.index = index
        normal_bg = self._bg()
        selected_bg = theme.color("sel_bg") or "#0078d7"
        selected_fg = theme.color("sel_fg") or "#ffffff"
        text_fg = theme.color("text") or "#000000"
        accent = theme.color("accent") or "#0078d7"
        for position, (_family, frame, name, star) in enumerate(self.rows):
            if position == index:
                frame.configure(bg=selected_bg)
                name.configure(bg=selected_bg, fg=selected_fg)
                star.configure(bg=selected_bg, fg=selected_fg)
            else:
                frame.configure(bg=normal_bg)
                name.configure(bg=normal_bg, fg=text_fg)
                star.configure(bg=normal_bg, fg=accent)
        self._see(index)

    def _see(self, index):
        if not self.rows:
            return
        frame = self.rows[index][1]
        self.inner.update_idletasks()
        total = max(1, self.inner.winfo_height())
        top = frame.winfo_y() / total
        bottom = (frame.winfo_y() + frame.winfo_height()) / total
        first, last = self.canvas.yview()
        if top < first:
            self.canvas.yview_moveto(top)
        elif bottom > last:
            self.canvas.yview_moveto(max(0.0, bottom - (last - first)))

    def _move(self, delta):
        if self.rows:
            self._highlight((self.index + delta) % len(self.rows))
        return "break"

    def _accept(self):
        if self.rows:
            self.on_choose(self.rows[self.index][0])
        return "break"

    def _on_key(self, event):
        if event.keysym in ("Up", "Down", "Return", "Escape"):
            return None
        self._build_rows(self.entry.get())
        return None
