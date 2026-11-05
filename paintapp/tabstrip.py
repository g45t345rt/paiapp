"""A scrollable row of canvas tabs with an overflow list.

The stock ``ttk.Notebook`` tab row cannot scroll: once the open canvases
outgrow the window its tabs shrink and then clip, with no way to reach the
hidden ones. This strip draws its own tabs on a canvas instead, so the row
scrolls sideways with the wheel or the chevron buttons, every canvas stays
reachable, and a dropdown lists them all in one click.

Tabs keep a readable width: a long name is ellipsized rather than squeezing
its neighbours away, and the trailing "+" tab opens the New Canvas flow like
the notebook page it replaces.
"""

import tkinter as tk
from tkinter import font as tkfont

from . import theme
from .ui import PopupMenu

# Tab row geometry, in pixels.
TAB_HEIGHT = 32
TAB_PAD = 12          # horizontal padding around a tab's label
TAB_MIN_W = 52
TAB_MAX_W = 220
ADD_TAB_W = 28

SCROLL_STEP = 60      # pixels moved by one wheel notch or chevron press
DRAG_SLOP = 6         # pointer travel before a press becomes a drag
MAX_MENU_LABEL = 48   # longest canvas name shown in the overflow list


class TabStrip(tk.Frame):
    """A scrolling row of canvas tabs.

    ``on_select(index)`` fires when the user picks a tab, ``on_rename(index)``
    on a tab double-click, ``on_add()`` for the trailing "+" tab,
    ``on_context(index, x_root, y_root)`` on a tab right-click,
    ``on_move(from_index, to_index)`` while a tab is dragged and
    ``on_move_commit()`` once it is dropped. Programmatic changes through
    :meth:`set_titles` and :meth:`select` never fire ``on_select``.
    """

    def __init__(self, parent, on_select=None, on_rename=None, on_add=None,
                 on_move=None, on_move_commit=None, on_context=None):
        super().__init__(parent)
        self._on_select = on_select
        self._on_rename = on_rename
        self._on_add = on_add
        self._on_move = on_move
        self._on_move_commit = on_move_commit
        self._on_context = on_context

        self._titles = []
        self._labels = []
        self._widths = []
        self._slots = []                       # (x, width, index), unscrolled
        self._add_slot = (0, ADD_TAB_W)
        self._content_w = ADD_TAB_W
        self._offset = 0
        self._selected = 0
        self._hover = None                     # tab index, "add" or None
        self._drag_from = None
        self._drag_start = None
        self._drag_pos = None
        self._drag_moved = False
        self._pending_reveal = None
        self._last_size = (0, 0)
        self._font = tkfont.nametofont("TkDefaultFont")

        self.left = tk.Button(self, text="\u2039", width=2, takefocus=0,
                              command=lambda: self.scroll(-1))
        self.right = tk.Button(self, text="\u203a", width=2, takefocus=0,
                               command=lambda: self.scroll(1))
        self.overflow = tk.Button(self, text="\u25be", width=2, takefocus=0,
                                  command=self._post_overflow)
        self.canvas = tk.Canvas(self, height=TAB_HEIGHT, bd=0,
                                highlightthickness=0, takefocus=0)

        self.left.pack(side=tk.LEFT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.right.pack(side=tk.LEFT, fill=tk.Y)
        self.overflow.pack(side=tk.LEFT, fill=tk.Y)

        self.canvas.bind("<Configure>", self._on_configure)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<Double-Button-1>", self._on_double)
        self.canvas.bind("<Button-3>", self._on_context_click)
        self.canvas.bind("<Motion>", self._on_hover)
        self.canvas.bind("<Leave>", self._on_leave)
        # Wheel events are bound on the widget, not across the app, so the
        # pointer has to be over the strip; "break" keeps the event from
        # reaching the global (sidebar) wheel handler.
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Shift-MouseWheel>", self._on_wheel)

        self._apply_theme()
        theme.subscribe(self._apply_theme)

    # ----------------------------------------------------------------- driving
    def set_titles(self, titles, selected=None):
        """Replace the shown tabs, marking ``selected`` as the active canvas."""
        self._titles = list(titles)
        if selected is not None:
            self._selected = int(selected)
        if self._titles:
            self._selected = max(0, min(self._selected, len(self._titles) - 1))
        else:
            self._selected = 0
        self._measure()
        self._layout()
        self._clamp()
        self._reveal_selected()
        self._render()
        self._sync_arrows()

    def select(self, index, notify=False):
        """Mark ``index`` active and scroll it into view.

        Only ``notify`` reports the change through ``on_select``; the app drives
        the strip itself when it switches canvases.
        """
        if not self._titles:
            return
        index = max(0, min(int(index), len(self._titles) - 1))
        self._selected = index
        self._reveal_selected()
        self._render()
        self._sync_arrows()
        if notify and self._on_select is not None:
            self._on_select(index)

    def scroll(self, direction):
        """Scroll the row one step to the left (``-1``) or right (``1``)."""
        limit = self._max_offset()
        if limit <= 0:
            self._sync_arrows()
            return
        target = max(0, min(limit, self._offset + direction * SCROLL_STEP))
        if target == self._offset:
            return
        self._offset = target
        self._render()
        self._sync_arrows()

    # ----------------------------------------------------------------- metrics
    def _measure(self):
        limit = TAB_MAX_W - 2 * TAB_PAD
        self._labels = []
        self._widths = []
        for title in self._titles:
            label = self._ellipsize(title, limit)
            self._labels.append(label)
            width = self._font.measure(label) + 2 * TAB_PAD
            self._widths.append(max(TAB_MIN_W, min(TAB_MAX_W, width)))

    def _ellipsize(self, text, limit):
        """Shorten ``text`` with an ellipsis so it fits ``limit`` pixels."""
        if self._font.measure(text) <= limit:
            return text
        low, high = 0, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if self._font.measure(text[:middle] + "\u2026") <= limit:
                low = middle
            else:
                high = middle - 1
        return text[:low].rstrip() + "\u2026"

    def _layout(self):
        self._slots = []
        x = 0
        for index, width in enumerate(self._widths):
            self._slots.append((x, width, index))
            x += width
        self._add_slot = (x, ADD_TAB_W)
        self._content_w = x + ADD_TAB_W

    def _view_w(self):
        width = self.canvas.winfo_width()
        return width if width > 1 else 0

    def _max_offset(self):
        view = self._view_w()
        if view <= 0:
            return 0
        return max(0, self._content_w - view)

    def _clamp(self):
        self._offset = max(0, min(self._offset, self._max_offset()))

    def _reveal_selected(self):
        """Scroll the selected tab into view (once the strip has a width)."""
        if self._drag_from is not None:
            return
        view = self._view_w()
        if view <= 0:
            self._pending_reveal = self._selected
            return
        self._pending_reveal = None
        if not self._slots:
            return
        x, width, _index = self._slots[self._selected]
        if x < self._offset:
            self._offset = x
        elif x + width > self._offset + view:
            self._offset = x + width - view
        self._clamp()

    # ------------------------------------------------------------------ input
    def _slot_at(self, content_x):
        for x, width, _index in self._slots:
            if x <= content_x < x + width:
                return _index
        return None

    def _hit(self, canvas_x):
        """What sits under ``canvas_x``: ``("tab", i)``, ``("add", None)`` or None."""
        content_x = canvas_x + self._offset
        index = self._slot_at(content_x)
        if index is not None:
            return ("tab", index)
        x, width = self._add_slot
        if x <= content_x < x + width:
            return ("add", None)
        return None

    def _target_at(self, canvas_x):
        """Index the dragged tab should occupy for the pointer at ``canvas_x``."""
        if not self._slots:
            return None
        content_x = canvas_x + self._offset
        index = self._slot_at(content_x)
        if index is not None:
            return index
        if content_x < self._slots[0][0]:
            return 0
        return len(self._slots) - 1

    def _on_press(self, event):
        self._drag_start = (event.x, event.y)
        self._drag_moved = False
        hit = self._hit(event.x)
        self._drag_from = hit[1] if hit is not None and hit[0] == "tab" else None
        self._drag_pos = (event.x, event.y)

    def _on_motion(self, event):
        if self._drag_start is None:
            return
        self._drag_pos = (event.x, event.y)
        if not self._drag_moved:
            dx = abs(event.x - self._drag_start[0])
            dy = abs(event.y - self._drag_start[1])
            if dx < DRAG_SLOP and dy < DRAG_SLOP:
                return
            self._drag_moved = True
        if self._drag_from is None:
            return
        target = self._target_at(event.x)
        if target is not None and target != self._drag_from:
            if self._on_move is not None:
                self._on_move(self._drag_from, target)
            self._drag_from = target
        # Keep the ghost under the pointer even when the tab itself does not
        # move (the reorder above already repaints when it does).
        self._render()

    def _on_release(self, event):
        moved = self._drag_moved
        self._drag_from = None
        self._drag_start = None
        self._drag_pos = None
        self._drag_moved = False
        # Drop the drag ghost even when the drop changes nothing (a reorder
        # repaints on its own, a cancelled drag does not).
        self._render()
        if moved:
            if self._on_move_commit is not None:
                self._on_move_commit()
            return
        hit = self._hit(event.x)
        if hit is None:
            return
        kind, index = hit
        if kind == "add":
            if self._on_add is not None:
                # Deferred so the click finishes before the size dialog opens.
                self.after_idle(self._on_add)
        else:
            self.select(index, notify=True)

    def _on_double(self, event):
        hit = self._hit(event.x)
        if hit is None or hit[0] != "tab" or self._on_rename is None:
            return
        index = hit[1]
        self.after_idle(lambda: self._on_rename(index))

    def _on_context_click(self, event):
        """Right-click: target the tab under the pointer, then report it.

        The clicked tab is made active first so the menu acts on the canvas the
        user pointed at, then ``on_context`` posts the action list.
        """
        hit = self._hit(event.x)
        if hit is None or hit[0] != "tab" or self._on_context is None:
            return None
        index = hit[1]
        if index != self._selected:
            self.select(index, notify=True)
        x_root, y_root = event.x_root, event.y_root
        self.after_idle(lambda: self._on_context(index, x_root, y_root))
        return "break"

    def _on_hover(self, event):
        hit = self._hit(event.x)
        target = None
        if hit is not None:
            target = hit[1] if hit[0] == "tab" else "add"
        if target != self._hover:
            self._hover = target
            self._render()

    def _on_leave(self, _event):
        if self._hover is not None:
            self._hover = None
            self._render()

    def _on_wheel(self, event):
        if event.delta == 0:
            return None
        self.scroll(-1 if event.delta > 0 else 1)
        return "break"

    def _on_configure(self, event):
        if event.widget is not self.canvas:
            return
        size = (event.width, event.height)
        if size == self._last_size:
            return
        self._last_size = size
        self._clamp()
        if self._pending_reveal is not None:
            self._reveal_selected()
        self._render()
        self._sync_arrows()

    # -------------------------------------------------------------- overflow
    def _post_overflow(self):
        """List every canvas under the dropdown button."""
        if not self._titles:
            return
        menu = PopupMenu(self)
        for index, title in enumerate(self._titles):
            if len(title) > MAX_MENU_LABEL:
                title = title[:MAX_MENU_LABEL - 1] + "\u2026"
            menu.add_checkbutton(
                label=title,
                variable=tk.BooleanVar(value=index == self._selected),
                command=lambda index=index: self.select(index, notify=True),
            )
        menu.post(self.overflow.winfo_rootx(),
                  self.overflow.winfo_rooty() + self.overflow.winfo_height())

    # --------------------------------------------------------------- drawing
    def _sync_arrows(self):
        limit = self._max_offset()
        self.left.config(state=tk.NORMAL if self._offset > 0 else tk.DISABLED)
        self.right.config(
            state=tk.NORMAL if self._offset < limit else tk.DISABLED
        )

    def _apply_theme(self):
        if not self.winfo_exists():
            return
        background = theme.color("notebook_bg")
        if background:
            try:
                self.canvas.configure(bg=background)
            except tk.TclError:
                return
        self._render()
        self._sync_arrows()

    def _render(self):
        canvas = self.canvas
        canvas.delete("all")
        view = self._view_w() or 10 ** 6
        tab_bg = theme.color("tab_bg")
        tab_fg = theme.color("tab_fg")
        active_bg = theme.color("tab_active_bg")
        active_fg = theme.color("tab_active_fg")
        selected_bg = theme.color("tab_selected_bg")
        selected_fg = theme.color("tab_selected_fg")
        border = theme.color("tab_border")

        for (x, width, index), label in zip(self._slots, self._labels):
            left = x - self._offset
            if left + width < 0 or left > view:
                continue
            selected = index == self._selected
            hovered = index == self._hover
            fill = selected_bg if selected else (active_bg if hovered else tab_bg)
            foreground = (selected_fg if selected
                          else (active_fg if hovered else tab_fg))
            canvas.create_rectangle(left, 0, left + width, TAB_HEIGHT,
                                    fill=fill, outline=border)
            canvas.create_text(left + width / 2, TAB_HEIGHT / 2, text=label,
                               fill=foreground, anchor="center")

        x, width = self._add_slot
        left = x - self._offset
        if not (left + width < 0 or left > view):
            hovered = self._hover == "add"
            canvas.create_rectangle(left, 0, left + width, TAB_HEIGHT,
                                    fill=active_bg if hovered else tab_bg,
                                    outline=border)
            canvas.create_text(left + width / 2, TAB_HEIGHT / 2, text="+",
                               fill=active_fg if hovered else tab_fg,
                               anchor="center")

        if self._drag_moved and self._drag_from is not None and self._drag_pos:
            self._draw_ghost(self._drag_from, self._drag_pos[0])

    def _draw_ghost(self, index, pointer_x):
        """Draw a floating, translucent copy of the dragged tab.

        The row reorders live, so the real tab already sits under the pointer.
        The ghost is a stippled, dashed-edged copy drawn on top of it, which
        reads as "this is the tab being moved" without hiding the row.
        """
        if not (0 <= index < len(self._labels)):
            return
        canvas = self.canvas
        width = self._widths[index]
        left = pointer_x - width / 2
        canvas.create_rectangle(
            left, 2, left + width, TAB_HEIGHT - 2,
            fill=theme.color("tab_selected_bg"), outline=theme.color("accent"),
            dash=(4, 3), width=1, stipple="gray50", tags="ghost",
        )
        canvas.create_text(
            left + width / 2, TAB_HEIGHT / 2, text=self._labels[index],
            fill=theme.color("tab_selected_fg"), anchor="center", tags="ghost",
        )
