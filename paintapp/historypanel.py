"""The undo/redo history list panel."""

import tkinter as tk

from .scrollbar import ArrowScrollbar


class HistoryPanel(tk.Frame):
    """A clickable list of document states; selecting one jumps to it."""

    def __init__(self, parent, on_select):
        super().__init__(parent)
        self._on_select = on_select
        self._syncing = False
        self._indices = []
        self._build()

    def _build(self):
        list_frame = tk.Frame(self)
        list_frame.pack(fill=tk.X)
        scrollbar = ArrowScrollbar(list_frame)
        self.list = tk.Listbox(
            list_frame, height=6, exportselection=False, activestyle="none",
            yscrollcommand=scrollbar.set,
        )
        scrollbar.set_command(self.list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.list.bind("<<ListboxSelect>>", self._handle_select)

    def refresh(self, history):
        """Rebuild the list from ``history``, newest edit first.

        ``history`` is the active canvas's stack; ``None`` (no canvas open)
        clears the list.
        """
        self._syncing = True
        self.list.delete(0, tk.END)
        self._indices = []
        if history is None:
            self._syncing = False
            return
        rows = history.timeline()
        current = history.position
        for index in range(len(rows) - 1, -1, -1):
            row = rows[index]
            self._indices.append(index)
            label = row["label"]
            self.list.insert(tk.END, f"{label}  (current)" if row["current"] else label)
        if rows:
            row = len(rows) - 1 - current
            self.list.selection_clear(0, tk.END)
            self.list.selection_set(row)
            self.list.activate(row)
            self.list.see(row)
        self._syncing = False

    def _handle_select(self, _event=None):
        if self._syncing:
            return
        selection = self.list.curselection()
        if selection and selection[0] < len(self._indices):
            self._on_select(self._indices[selection[0]])
