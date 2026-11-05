"""The scrollable log panel that collects status messages.

The status bar shows a single transient line, so anything it reports is gone
the moment the next action overwrites it. This panel keeps a scrolling history
of those messages, each colored by its severity, so a warning or an error can
still be found after it has scrolled off the bar.
"""

import tkinter as tk

from . import dialogs, theme
from .scrollbar import ArrowScrollbar

# Leading glyph per severity level; the fg color comes from ``theme.LEVELS``.
_MARKS = {
    "info": "\u2022",
    "success": "\u2713",
    "warn": "\u26a0",
    "error": "\u2717",
}

# How many messages to keep; the oldest lines fall off the bottom.
MAX_ENTRIES = 200


class LogPanel(tk.Frame):
    """A read-only list of status messages, newest first, colored by severity."""

    def __init__(self, parent):
        super().__init__(parent)
        self._entries = []
        self._build()
        theme.subscribe(self._restyle)

    def _build(self):
        list_frame = tk.Frame(self)
        list_frame.pack(fill=tk.X)
        scrollbar = ArrowScrollbar(list_frame)
        self.list = tk.Listbox(
            list_frame, height=8, exportselection=False, activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        scrollbar.set_command(self.list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.list.bind("<Double-Button-1>", self._on_double_click)

        buttons = tk.Frame(self)
        buttons.pack(fill=tk.X, pady=(6, 4))
        self.clear_btn = tk.Button(buttons, text="Clear Log", command=self.clear)
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # ------------------------------------------------------------------ public
    def append(self, kind, text):
        """Add one message under ``kind`` (info/success/warn/error).

        Newest messages go to the top; the view scrolls back to the top so the
        latest line is always visible.
        """
        if not text:
            return
        kind = kind if kind in theme.LEVELS else "info"
        self._entries.insert(0, (kind, text))
        self.list.insert(0, self._label(kind, text))
        self._color_row(0, kind)
        self._trim()
        self.list.see(0)
        self.list.yview_moveto(0.0)

    def entries(self):
        """A snapshot of the logged ``(kind, text)`` pairs, newest first."""
        return list(self._entries)

    def clear(self):
        """Remove every logged message."""
        self._entries = []
        self.list.delete(0, tk.END)

    # --------------------------------------------------------------- internals
    @staticmethod
    def _label(kind, text):
        return f"{_MARKS.get(kind, _MARKS['info'])}  {text}"

    def _color_row(self, row, kind):
        fg, _bg = theme.LEVELS.get(kind, theme.LEVELS["info"])
        try:
            self.list.itemconfig(row, foreground=theme.color(fg))
        except tk.TclError:
            pass

    def _trim(self):
        """Drop the oldest lines once the log grows past :data:`MAX_ENTRIES`."""
        while len(self._entries) > MAX_ENTRIES:
            self._entries.pop()
            self.list.delete(self.list.size() - 1)

    def _on_double_click(self, event):
        """Open the clicked line in a dialog so long messages can be read."""
        row = self.list.nearest(event.y)
        if not 0 <= row < len(self._entries):
            return
        box = self.list.bbox(row)
        if box is None or not box[1] <= event.y <= box[1] + box[3]:
            return
        kind, text = self._entries[row]
        dialogs.show_text(
            self.winfo_toplevel(), f"Log: {kind.title()}", text
        )

    def _restyle(self):
        """Recolor every row after a theme switch."""
        for row, (kind, _text) in enumerate(self._entries):
            self._color_row(row, kind)
