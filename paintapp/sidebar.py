"""The resizable, scrollable sidebar rail that hosts the tool panels.

Besides the rail itself this module provides :class:`PanelHost`, the header +
content wrapper around one panel, and :class:`PanelGroup`, the container that
stacks several panels as tabs behind a single header.

The rail can be reorganized by the user: panels are dragged by their header to
reorder them, dropped onto another panel to tab the two together, detached into
a floating window that can be moved and resized anywhere (including outside the
main window) and docked back afterwards, or sent to the sidebar on the other
side of the canvas.

Two sidebars share the body frame, one on each side of the canvas. Panels are
created with the *body* as their master and packed into whichever rail holds
them, so a panel can move between the left and right rail without being
rebuilt: Tk allows a widget to be packed into any descendant of its parent, so
the same host (and every child and its state) is simply re-packed. The same
rule lets a host be packed into a :class:`PanelGroup` body, which is itself a
descendant of the body. Detaching uses Tk's ``wm manage`` command, which
promotes the item into a real top-level OS window without losing the widget
hierarchy (so every child and its state survives the round trip); ``wm forget``
returns it to the rail. Older Tk builds without ``wm manage`` fall back to an
in-window pane that is ``place``d over the body.
"""

import tkinter as tk

try:
    import ctypes
except ImportError:  # pragma: no cover - ctypes is stdlib on supported builds
    ctypes = None

from . import theme
from .constants import (
    SIDEBAR_W,
    SIDEBAR_MIN_W,
    SIDEBAR_SPLITTER_W,
    SIDEBAR_SCROLLBAR_W,
    SIDEBAR_HEADER_H,
    PANEL_FLOAT_MIN_W,
    PANEL_FLOAT_MIN_H,
    PANEL_FLOAT_PAD,
    SIDEBAR_GRIP,
)

from .scrollbar import ArrowScrollbar

FLOAT_GLYPH = "\u2197"
DOCK_GLYPH = "\u2199"
HANDLE_GLYPH = "\u2261"
UNGROUP_GLYPH = "\u229f"
COLLAPSE_GLYPH_EXPANDED = "\u25be"
COLLAPSE_GLYPH_COLLAPSED = "\u25b8"
# Shown on a panel header: the direction it will move to the opposite sidebar.
MOVE_LEFT_GLYPH = "\u2190"
MOVE_RIGHT_GLYPH = "\u2192"

SCROLL_UNIT = 40

# How far below an item's top edge counts as its header band: dropping a dragged
# panel there tabs the two together instead of reordering.
MERGE_BAND = SIDEBAR_HEADER_H + 8

# ``event.state`` bit set while Shift is held (Windows/Tk).
SHIFT_MASK = 0x0001


class _RailItem:
    """Window and drag behaviour shared by :class:`PanelHost` and groups.

    Both rail items are ``tk.Frame`` subclasses built with the shared body as
    their master. The methods here treat ``self`` as an item that owns a
    ``dock`` (:class:`Sidebar`), a ``floating`` flag and the widgets needed to
    move or resize itself as a real window.
    """

    floating = False

    # Built lazily the first time the item floats: a viewport that lets a
    # detached container be scrolled when its content outgrows the window.
    _fviewport = None
    _fscroll = None
    _finner = None
    _foffset = 0

    # A grouped panel floated as a companion window: its own move/resize
    # gestures are disabled because the owning :class:`PanelGroup` drives the
    # geometry (see :meth:`_sync_host_window`).
    _companion = False
    # The panel a floating group currently shows in its companion window.
    _managed_host = None

    # ------------------------------------------------------ detached window
    def _manage(self):
        """Promote this frame into a standalone top-level window.

        ``wm manage`` is Tk's supported way to reparent a widget between
        toplevels: it turns the frame itself into a window whose children (and
        their state) are preserved, so the item keeps working as a detached
        tool window that can be dragged off the main window. Raises
        ``tk.TclError`` on Tk builds that lack the command.
        """
        self.tk.call("wm", "manage", self._w)
        try:
            self.tk.call("wm", "overrideredirect", self._w, True)
        except tk.TclError:
            pass
        try:
            self.tk.call("wm", "attributes", self._w, "-topmost", True)
        except tk.TclError:
            pass

    def _unmanage(self):
        """Return a detached window to an ordinary frame ready for docking."""
        try:
            self.tk.call("wm", "attributes", self._w, "-topmost", False)
        except tk.TclError:
            pass
        try:
            self.tk.call("wm", "overrideredirect", self._w, False)
        except tk.TclError:
            pass
        self.tk.call("wm", "forget", self._w)

    def _set_window_geometry(self, spec):
        self.tk.call("wm", "geometry", self._w, spec)

    def _force_repaint(self):
        """Make Windows redraw the region an in-window pane just vacated.

        Moving a ``place``d pane over sibling widgets does not reliably
        invalidate the area it leaves behind, so a fast drag shows stale copies
        of the pane. Asking the OS to erase and repaint the body (and its
        children) keeps the moved pane clean.
        """
        try:
            if ctypes is None:
                return
            hwnd = self.dock.body.winfo_id()
            RDW_INVALIDATE = 0x0001
            RDW_ERASE = 0x0004
            RDW_ALLCHILDREN = 0x0080
            RDW_UPDATENOW = 0x0100
            ctypes.windll.user32.RedrawWindow(
                hwnd, None, None,
                RDW_INVALIDATE | RDW_ERASE | RDW_ALLCHILDREN | RDW_UPDATENOW,
            )
        except Exception:
            pass

    # --------------------------------------------------- floating scrollbar
    def _float_content(self):
        """The widget moved into the scrolling viewport while floating."""
        raise NotImplementedError

    def _start_float_scroll(self):
        """Move this item's content into a scrolling viewport while floating.

        The rail scrolls its docked panels, so a panel taller than the rail is
        still reachable. A detached container had no scroll of its own, so once
        the window was shorter than the panel its content was clipped and
        unreachable. Reparent the content into a placed inner frame under a
        vertical scrollbar (the same model the rail uses) so a floating panel
        can be scrolled too.
        """
        content = self._float_content()
        if self._fviewport is None:
            viewport = tk.Frame(self)
            theme.register(viewport, bg="bg")
            scrollbar = ArrowScrollbar(
                viewport, command=self._on_float_scrollbar,
                width=SIDEBAR_SCROLLBAR_W,
            )
            # Keep the scrollbar on the trailing edge, clear of the resize
            # grip that sits in the opposite corner.
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            inner = tk.Frame(viewport)
            theme.register(inner, bg="bg")
            inner.place(x=0, y=0)
            inner.bind("<Configure>", lambda _e: self._update_float_scroll())
            viewport.bind("<Configure>", self._on_float_viewport_configure)
            self._fviewport = viewport
            self._fscroll = scrollbar
            self._finner = inner
        content.pack_forget()
        content.pack(in_=self._finner, fill=tk.X)
        self._foffset = 0
        self._fviewport.pack(fill=tk.BOTH, expand=True)
        # The content's real parent is the item, but it is geometry-managed by
        # a descendant of that item; reparenting leaves it behind the viewport
        # in the stacking order, so Windows never paints it (a floating panel
        # showed an empty body with only a sliver of its last child). Raise it
        # above the viewport, after the viewport is mapped, so it is drawn.
        content.lift()
        self.after_idle(self._update_float_scroll)

    def _stop_float_scroll(self):
        """Return the content to a docked layout and drop the float scroll."""
        if self._fviewport is None:
            return
        content = self._float_content()
        content.pack_forget()
        self._fviewport.pack_forget()
        content.pack(fill=tk.X)
        self._foffset = 0
        self._fscroll.set(0.0, 1.0)

    def _on_float_viewport_configure(self, event):
        # Leave the scrollbar its own column so the raised content never
        # underlaps (and hides) it.
        self._finner.place_configure(
            width=max(1, event.width - SIDEBAR_SCROLLBAR_W)
        )
        self._update_float_scroll()

    def _update_float_scroll(self):
        """Clamp the offset and refresh the scrollbar after a size change."""
        if self._fviewport is None or not self.floating:
            return
        total = self._finner.winfo_reqheight()
        view = self._fviewport.winfo_height()
        max_offset = max(0, total - view)
        self._foffset = max(0, min(self._foffset, max_offset))
        self._finner.place_configure(y=-self._foffset)
        span = float(total) or 1.0
        self._fscroll.set(
            self._foffset / span, min(1.0, (self._foffset + view) / span)
        )

    def _float_scroll_by(self, delta):
        self._foffset += delta
        self._update_float_scroll()

    def _on_float_scrollbar(self, *args):
        if not args:
            return
        action = args[0]
        if action == "moveto":
            self._foffset = int(
                float(args[1]) * self._finner.winfo_reqheight()
            )
        elif action == "scroll":
            number = int(args[1])
            what = args[2] if len(args) > 2 else "units"
            step = (
                self._fviewport.winfo_height() if what == "pages"
                else SCROLL_UNIT
            )
            self._foffset += number * step
        self._update_float_scroll()

    # --------------------------------------------------------- header dragging
    def _drag_start(self, event):
        self._drag = {"x_root": event.x_root, "y_root": event.y_root}
        if self.floating:
            self._drag["x"] = self.winfo_x()
            self._drag["y"] = self.winfo_y()
            self._drag["w"] = self.winfo_width()
            self._drag["h"] = self.winfo_height()
            self.lift()
            # Keep a group's companion window above the (opaque) group frame it
            # covers, or lifting the group would hide the panel.
            if self._managed_host is not None:
                tk.Misc.lift(self._managed_host)
        else:
            self._dock_index = self.dock.row_index_of(self)
            self.dock.begin_drag_ghost(self, event.x_root, event.y_root)

    def _drag_motion(self, event):
        if self._drag is None:
            return
        if self.floating:
            x = self._drag["x"] + (event.x_root - self._drag["x_root"])
            y = self._drag["y"] + (event.y_root - self._drag["y_root"])
            if self._windowed:
                # A real window can be moved anywhere, including off the app.
                self._set_window_geometry(f"+{int(x)}+{int(y)}")
                self._sync_host_window(x=int(x), y=int(y))
            else:
                body = self.dock.body
                x = max(0, min(x, max(0, body.winfo_width() - 60)))
                y = max(0, min(y, max(0, body.winfo_height() - SIDEBAR_HEADER_H)))
                self.place_configure(x=int(x), y=int(y))
                self._force_repaint()
        else:
            self.dock.move_drag_ghost(event.x_root, event.y_root)
            self.dock.drag_over(self, event.x_root, event.y_root)

    def _drag_end(self, _event):
        if self._drag is not None and not self.floating:
            self.dock.end_drag_ghost()
            self.dock.drop_item(self)
        self._drag = None

    # ------------------------------------------------- header-wide dragging
    def _header_press(self, event):
        """Press on the header: drag a floating item, else run the click action.

        A floating item is moved by dragging anywhere on its header, since there
        is nothing else the header could do; a docked item keeps its usual
        header click (collapsing, for a panel) and is dragged by the handle.
        """
        if self.floating:
            self._drag_start(event)
        else:
            self._header_click(event)

    def _header_click(self, _event):
        """What a plain header click does while docked (overridden to collapse)."""
        return

    def _bind_header_drag(self, *widgets):
        """Make ``widgets`` press/drag/release the item's own header band."""
        for widget in widgets:
            widget.bind("<Button-1>", self._header_press)
            widget.bind("<B1-Motion>", self._drag_motion)
            widget.bind("<ButtonRelease-1>", self._drag_end)

    # --------------------------------------------------------- pane resizing
    def _resize_start(self, event):
        self._resize = {
            "x_root": event.x_root, "y_root": event.y_root,
            "w": self.winfo_width(), "h": self.winfo_height(),
        }

    def _resize_motion(self, event):
        if not self.floating or self._resize is None:
            return
        width = self._resize["w"] + (event.x_root - self._resize["x_root"])
        height = self._resize["h"] + (event.y_root - self._resize["y_root"])
        width = max(PANEL_FLOAT_MIN_W, width)
        height = max(PANEL_FLOAT_MIN_H, height)
        if self._windowed:
            x0, y0 = self.winfo_x(), self.winfo_y()
            self._set_window_geometry(
                f"{int(width)}x{int(height)}+{x0}+{y0}"
            )
            self._sync_host_window(
                x=x0, y=y0, width=int(width), height=int(height)
            )
        else:
            self.place_configure(width=width, height=height)
            self._force_repaint()

    def _sync_host_window(self, x=None, y=None, width=None, height=None):
        """Keep a floating group's companion window under its tab header.

        A group floats as two synced top-levels: the group frame draws the tab
        header and the resize gutter, while the active panel is promoted to its
        own window covering the body area. Moving or resizing the group must
        move and size that companion too.
        """
        host = self._managed_host
        if host is None:
            return
        # The header's offset inside the group is its border plus padding. Read
        # it from the options rather than live geometry: right after the group
        # is promoted the child layout has not settled and the offsets are 0.
        inset_x = int(self.cget("highlightthickness")) + int(self.cget("padx"))
        inset_y = int(self.cget("highlightthickness")) + int(self.cget("pady"))
        header_h = self.header.winfo_height()
        if header_h <= 1:
            header_h = SIDEBAR_HEADER_H
        if x is None:
            x = self.winfo_x()
        if y is None:
            y = self.winfo_y()
        if width is None:
            width = self.winfo_width()
        if height is None:
            height = self.winfo_height()
        host._set_window_geometry("%dx%d+%d+%d" % (
            max(1, int(width - 2 * inset_x)),
            max(1, int(height - 2 * inset_y - header_h)),
            int(x + inset_x),
            int(y + inset_y + header_h),
        ))
        # The group frame is opaque and spans the companion's area, so the
        # companion must stay above it.
        tk.Misc.lift(host)

    def _resize_band(self, event):
        """Which of the floating item's resize edges the pointer is within.

        Only the padding gutter around the item's children is exposed to the
        item frame, so these bindings fire on that gutter alone: a press on a
        button or field goes to that child, never to the resize gesture.
        """
        if not self.floating or self._companion:
            return False, False
        band = PANEL_FLOAT_PAD + 2
        right = event.x >= self.winfo_width() - band
        bottom = event.y >= self.winfo_height() - band
        return right, bottom

    def _resize_cursor(self, event):
        """Show the resize cursor while the pointer is on a resize edge."""
        right, bottom = self._resize_band(event)
        if right and bottom:
            cursor = "bottom_right_corner"
        elif right:
            cursor = "sb_h_double_arrow"
        elif bottom:
            cursor = "sb_v_double_arrow"
        else:
            cursor = ""
        if self.cget("cursor") != cursor:
            self.configure(cursor=cursor)

    def _resize_press(self, event):
        """Begin a resize when a press lands on the bottom or right edge."""
        right, bottom = self._resize_band(event)
        if right or bottom:
            self._resize_start(event)

    def _bind_edge_resize(self):
        """Resize a floating item by dragging its bottom or right edge.

        The item frame is covered by its own children apart from the padding
        gutter, so these bindings only ever fire there; the visible corner grip
        mirrors the same gesture. Nothing overlays the panel's buttons, so none
        of them can lose a click to the resize handle.
        """
        self.bind("<Motion>", self._resize_cursor)
        self.bind("<Leave>", lambda _e: self.configure(cursor=""))
        self.bind("<Button-1>", self._resize_press)
        self.bind("<B1-Motion>", self._resize_motion)
        self.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_resize", None))

    def _build_grip(self):
        """Build the passive corner grip that marks the resize edge.

        A flat triangle drawn in the item's bottom-right padding gutter: it is
        sized to fit that gutter, so it never sits over the panel content (a
        button at the panel's bottom-right corner keeps every click). Dragging
        it resizes like the edges do; it runs no other action.
        """
        self.grip = tk.Canvas(
            self, width=SIDEBAR_GRIP, height=SIDEBAR_GRIP,
            highlightthickness=0, bd=0, takefocus=0,
            cursor="bottom_right_corner",
        )
        theme.register(self.grip, bg="bg")
        theme.subscribe(self._draw_grip)
        self._draw_grip()
        self.grip.bind("<Button-1>", self._resize_start)
        self.grip.bind("<B1-Motion>", self._resize_motion)
        self.grip.bind("<ButtonRelease-1>", lambda _e: setattr(self, "_resize", None))

    def _draw_grip(self):
        """Paint the corner triangle in the current theme's border color."""
        try:
            if not self.grip.winfo_exists():
                return
        except tk.TclError:
            return
        size = SIDEBAR_GRIP
        self.grip.delete("all")
        color = theme.color("float_border")
        self.grip.create_polygon(
            size, 0, size, size, 0, size, fill=color, outline=color,
        )

    # ------------------------------------------------------ collapse control
    def set_collapse_enabled(self, enabled):
        """Show or hide the header collapse control.

        Collapsing only makes sense for an item sitting in a rail, so the
        control is hidden while the item floats as its own window or in-window
        pane, where it would only get in the way of moving the panel.
        """
        if enabled:
            if not self.collapse_btn.winfo_manager():
                self.collapse_btn.pack(side=tk.LEFT, before=self._collapse_anchor)
        elif self.collapse_btn.winfo_manager():
            self.collapse_btn.pack_forget()


class PanelHost(_RailItem, tk.Frame):
    """A titled, draggable wrapper around one sidebar panel.

    The panel widget itself is built into :attr:`content`. The header is used
    to drag the host (reordering it while docked, moving it while floating) and
    carries the dock/float toggle.
    """

    def __init__(self, dock, key, title):
        # The host's master is the shared body frame, not the rail it currently
        # sits in. That lets the same widget be packed into either sidebar's
        # scrolling frame (or into a group body) without being rebuilt.
        super().__init__(dock.body, bd=0, highlightthickness=0)
        self.dock = dock
        self.key = key
        self.title = title
        self.side = dock.side
        self.floating = False
        self.collapsed = False
        self._dock_index = None
        self._drag = None
        self._resize = None
        self._windowed = False
        self._header_visible = True
        self._grouped = False
        self._group_collapsed = False
        self._build()
        theme.register(self, bg="bg", highlightbackground="float_border")
        self._bind_edge_resize()

        self.content = tk.Frame(self)
        self.content.pack(fill=tk.X)

    # ------------------------------------------------------------------ build
    def _build(self):
        header = tk.Frame(self, height=SIDEBAR_HEADER_H)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        self.header = header
        theme.register(header, bg="header_bg")

        self.handle = tk.Label(
            header, text=HANDLE_GLYPH, cursor="fleur", padx=3,
        )
        self.handle.pack(side=tk.LEFT)
        theme.register(self.handle, bg="header_bg", fg="header_fg")

        self.collapse_btn = tk.Label(
            header, text=COLLAPSE_GLYPH_EXPANDED, padx=2,
        )
        self.collapse_btn.pack(side=tk.LEFT)
        theme.register(self.collapse_btn, bg="header_bg", fg="header_fg")
        self.collapse_btn.bind("<Button-1>", lambda _e: self.toggle_collapse())

        self.title_label = tk.Label(
            header, text=self.title, anchor="w",
        )
        self.title_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        theme.register(self.title_label, bg="header_bg", fg="header_fg")
        self._collapse_anchor = self.title_label

        self.side_btn = tk.Label(
            header, text=self._side_glyph(), padx=4,
        )
        self.side_btn.pack(side=tk.RIGHT)
        theme.register(self.side_btn, bg="header_bg", fg="header_fg")
        self.side_btn.bind("<Button-1>", lambda _e: self.dock.toggle_side(self))

        self.float_btn = tk.Label(
            header, text=FLOAT_GLYPH, padx=4,
        )
        self.float_btn.pack(side=tk.RIGHT)
        theme.register(self.float_btn, bg="header_bg", fg="header_fg")
        self.float_btn.bind("<Button-1>", lambda _e: self.dock.toggle_float(self))

        # The handle always drags the panel. While docked a header click
        # collapses/expands it; while floating the whole header drags and
        # clicking it does nothing else (see :meth:`_header_press`).
        self.handle.bind("<Button-1>", self._drag_start)
        self.handle.bind("<B1-Motion>", self._drag_motion)
        self.handle.bind("<ButtonRelease-1>", self._drag_end)

        for widget in (header, self.handle, self.title_label):
            widget.bind("<Enter>", lambda _e: self._highlight(True))
            widget.bind("<Leave>", lambda _e: self._highlight(False))
        self._bind_header_drag(header, self.title_label)

        self._build_grip()

    def _highlight(self, active):
        color = theme.color("header_active_bg" if active else "header_bg")
        self.header.config(bg=color)
        self.handle.config(bg=color)
        self.collapse_btn.config(bg=color)
        self.title_label.config(bg=color)
        self.side_btn.config(bg=color)
        self.float_btn.config(bg=color)

    # ------------------------------------------------------------- panel side
    def _side_glyph(self):
        """The arrow the move button shows: where the panel will go."""
        return MOVE_RIGHT_GLYPH if self.side == "left" else MOVE_LEFT_GLYPH

    def set_side(self, side):
        """Record which sidebar holds this panel and refresh the move button."""
        self.side = side if side in ("left", "right") else "right"
        self.side_btn.config(text=self._side_glyph())

    # --------------------------------------------------------- header visibility
    def set_header_visible(self, visible):
        """Show or hide the panel's own header.

        A panel that is tabbed into a :class:`PanelGroup` hides its header and
        lets the group draw the tab bar; leaving the group restores it.
        """
        visible = bool(visible)
        if visible == self._header_visible:
            return
        self._header_visible = visible
        if visible:
            if self.content.winfo_manager():
                self.header.pack(fill=tk.X, before=self.content)
            else:
                self.header.pack(fill=tk.X)
        else:
            self.header.pack_forget()

    # -------------------------------------------------------------- collapsing
    def _header_click(self, _event):
        """A docked panel's header click collapses or expands it."""
        self.toggle_collapse()

    def toggle_collapse(self):
        """Show or hide the panel body, keeping only its header.

        A floating panel cannot be collapsed: its body is the entire point of
        the detached window, and collapsing on a stray header click gets in the
        way of moving it.
        """
        if self.floating:
            return
        self.set_collapsed(not self.collapsed)

    def set_collapsed(self, collapsed):
        """Set the collapsed state (header-only when ``collapsed``)."""
        self.collapsed = bool(collapsed)
        if self.collapsed:
            self.content.pack_forget()
            self.collapse_btn.config(text=COLLAPSE_GLYPH_COLLAPSED)
        else:
            self.content.pack(fill=tk.X)
            self.collapse_btn.config(text=COLLAPSE_GLYPH_EXPANDED)

    def _float_content(self):
        return self.content


class PanelGroup(_RailItem, tk.Frame):
    """Several panels stacked as tabs under a single header.

    A group is created by dropping a dragged panel (or another group) onto a
    panel's header. It owns the panel keys it holds in :attr:`keys` and shows
    only :attr:`active`; the sidebar packs the active :class:`PanelHost` into
    :attr:`body`. The group's own header offers collapse, float, ungroup and
    move-to-other-sidebar controls and can be dragged like a panel header.
    """

    def __init__(self, dock, gid):
        super().__init__(dock.body, bd=0, highlightthickness=0)
        self.dock = dock
        self.gid = gid
        self.rail_id = "g%d" % gid
        self.side = dock.side
        self.floating = False
        self.collapsed = False
        self.keys = []
        self.active = None
        self._dock_index = None
        self._drag = None
        self._resize = None
        self._windowed = False
        self._tab_labels = {}
        self._tab_keys = []
        self._build()
        theme.register(self, bg="bg", highlightbackground="float_border")
        self._bind_edge_resize()

    # ------------------------------------------------------------------ build
    def _build(self):
        header = tk.Frame(self, height=SIDEBAR_HEADER_H)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        self.header = header
        theme.register(header, bg="header_bg")

        self.handle = tk.Label(header, text=HANDLE_GLYPH, cursor="fleur", padx=3)
        self.handle.pack(side=tk.LEFT)
        theme.register(self.handle, bg="header_bg", fg="header_fg")

        self.collapse_btn = tk.Label(header, text=COLLAPSE_GLYPH_EXPANDED, padx=2)
        self.collapse_btn.pack(side=tk.LEFT)
        theme.register(self.collapse_btn, bg="header_bg", fg="header_fg")
        self.collapse_btn.bind("<Button-1>", lambda _e: self.toggle_collapse())

        self.tabs = tk.Frame(header)
        theme.register(self.tabs, bg="header_bg")
        self.tabs.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._collapse_anchor = self.tabs

        self.ungroup_btn = tk.Label(header, text=UNGROUP_GLYPH, padx=4)
        self.ungroup_btn.pack(side=tk.RIGHT)
        theme.register(self.ungroup_btn, bg="header_bg", fg="header_fg")
        self.ungroup_btn.bind("<Button-1>", lambda _e: self.dock.ungroup(self))

        self.side_btn = tk.Label(header, text=self._side_glyph(), padx=4)
        self.side_btn.pack(side=tk.RIGHT)
        theme.register(self.side_btn, bg="header_bg", fg="header_fg")
        self.side_btn.bind("<Button-1>", lambda _e: self.dock.toggle_side(self))

        self.float_btn = tk.Label(header, text=FLOAT_GLYPH, padx=4)
        self.float_btn.pack(side=tk.RIGHT)
        theme.register(self.float_btn, bg="header_bg", fg="header_fg")
        self.float_btn.bind("<Button-1>", lambda _e: self.dock.toggle_float(self))

        self.handle.bind("<Button-1>", self._drag_start)
        self.handle.bind("<B1-Motion>", self._drag_motion)
        self.handle.bind("<ButtonRelease-1>", self._drag_end)

        # Unlike a lone panel, a group's header is not a collapse target: only
        # its disclosure button collapses it, and a header click does nothing
        # while docked. So it must not raise the panel-style whole-header hover
        # highlight, which reads as "click me to collapse". Each control lights
        # up on its own instead; ``_highlight`` is still used to flag the group
        # as a drop target while dragging.
        self._bind_control_hover(
            self.handle, self.collapse_btn, self.ungroup_btn,
            self.side_btn, self.float_btn,
        )
        # While floating, the whole header (including the tab strip) moves the
        # group; while docked a header click does nothing (only the handle drags).
        self._bind_header_drag(header, self.tabs)

        self.body = tk.Frame(self)
        theme.register(self.body, bg="bg")
        self.body.pack(fill=tk.X)

        self._build_grip()
        # Any geometry change re-seats the companion window; while docked the
        # sync is a no-op (there is no companion).
        self.bind("<Configure>", lambda _e: self._sync_host_window())

    # ------------------------------------------------------------------ header
    def _highlight(self, active):
        """Flag the whole group during a drag (drop target), not on hover.

        The band-wide tint here reads as a drop target while a panel is dragged
        onto the group; it is deliberately not wired to ``<Enter>``/``<Leave>``,
        since a group header is not a collapse target.
        """
        color = theme.color("header_active_bg" if active else "header_bg")
        for widget in (self.header, self.handle, self.collapse_btn, self.tabs,
                       self.ungroup_btn, self.side_btn, self.float_btn):
            widget.config(bg=color)
        self._apply_tab_colors(
            self.active, "header_active_bg" if active else "header_bg"
        )

    def _bind_control_hover(self, *widgets):
        """Light up each header control on its own, never the whole header.

        A group header is not clickable-to-collapse, so a band-wide hover would
        wrongly advertise that; hovering a single control only tints that
        control.
        """
        for widget in widgets:
            widget.bind(
                "<Enter>",
                lambda _e, w=widget: w.config(bg=theme.color("header_active_bg")),
            )
            widget.bind(
                "<Leave>",
                lambda _e, w=widget: w.config(bg=theme.color("header_bg")),
            )

    def _side_glyph(self):
        return MOVE_RIGHT_GLYPH if self.side == "left" else MOVE_LEFT_GLYPH

    def set_side(self, side):
        self.side = side if side in ("left", "right") else "right"
        self.side_btn.config(text=self._side_glyph())

    # ------------------------------------------------------------------- tabs
    def sync_tabs(self, items, active):
        """Rebuild (or refresh) the tab labels for ``items`` ``(key, title)``."""
        keys = [key for key, _title in items]
        if list(self._tab_keys) != keys:
            for child in self.tabs.winfo_children():
                child.destroy()
            self._tab_labels = {}
            for key, title in items:
                label = tk.Label(self.tabs, text=title, padx=6)
                label.pack(side=tk.LEFT)
                label.bind(
                    "<Button-1>",
                    lambda _e, k=key: self._on_tab_click(k),
                )
                label.bind(
                    "<Enter>", lambda _e, k=key: self._hover_tab(k, True),
                )
                label.bind(
                    "<Leave>", lambda _e, k=key: self._hover_tab(k, False),
                )
                self._tab_labels[key] = label
            self._tab_keys = keys
        else:
            for key, title in items:
                self._tab_labels[key].config(text=title)
        self._apply_tab_colors(active, "header_bg")

    def _on_tab_click(self, key):
        """Click a tab: show it, or collapse the group when it is already shown.

        A tab is the visible content's header, so clicking the active tab
        toggles the group's collapse exactly like clicking a lone panel's header
        toggles its panel, rather than the whole group header band doing so.
        Clicking any other tab switches which panel is shown (and expands the
        group if it was collapsed), so a single click always reveals content.
        """
        if key == self.active:
            self.toggle_collapse()
        else:
            self.dock.set_group_active(self, key)
            if self.collapsed:
                self.dock.set_group_collapsed(self, False)

    def _hover_tab(self, key, active):
        """Tint the tab under the cursor, leaving the selected tab alone.

        A collapsed group draws no selected tab (see :meth:`_apply_tab_colors`),
        so its active tab is hoverable like any other while closed.
        """
        label = self._tab_labels.get(key)
        if label is None or (key == self.active and not self.collapsed):
            return
        theme.register(
            label,
            bg="header_active_bg" if active else "header_bg",
            fg="header_fg",
        )

    def _apply_tab_colors(self, active, normal_token):
        # A collapsed group shows no tab content, so no tab reads as selected:
        # drop the highlight while closed and restore it on expand.
        selected = None if self.collapsed else active
        for key, label in self._tab_labels.items():
            if key == selected:
                theme.register(label, bg="tab_selected_bg", fg="tab_selected_fg")
            else:
                theme.register(label, bg=normal_token, fg="header_fg")

    # -------------------------------------------------------------- collapsing
    def toggle_collapse(self):
        if self.floating:
            return
        self.dock.set_group_collapsed(self, not self.collapsed)

    def set_collapsed(self, collapsed):
        self.collapsed = bool(collapsed)
        if self.collapsed:
            self.body.pack_forget()
            self.collapse_btn.config(text=COLLAPSE_GLYPH_COLLAPSED)
        else:
            self.body.pack(fill=tk.X)
            self.collapse_btn.config(text=COLLAPSE_GLYPH_EXPANDED)
        self._apply_tab_colors(self.active, "header_bg")

    def _float_content(self):
        return self.body


class Sidebar(tk.Frame):
    """A vertical rail with a drag-to-resize splitter and mouse-wheel scrolling.

    ``side`` is ``"left"`` or ``"right"`` and controls which edge the splitter
    sits on (the canvas-facing edge) and how a resize drag is measured. Panels
    are registered with :meth:`add_panel`, which returns the content frame they
    should be built into. The sidebar owns their order, their grouping and their
    docked/floating state.

    Two sidebars work as a pair: :attr:`other` points at the sidebar on the
    opposite side, so a panel or a whole group can be sent across the canvas
    with :meth:`toggle_side`. :attr:`on_move` is called after any such move so
    the app can record the new side.
    """

    def __init__(self, parent, root, side="right"):
        super().__init__(parent)
        self._body = parent
        self.body = parent
        self._side = side if side in ("left", "right") else "right"
        self.side = self._side
        self.other = None
        self.on_move = None
        self._grid_opts = {}

        self._hosts = {}
        # Display order: each entry is a PanelHost or a PanelGroup.
        self._rows = []
        self._floating = []
        self._group_of = {}
        self._next_group = 0
        self._merge_target = None
        self._ghost = None

        splitter = tk.Frame(
            self, width=SIDEBAR_SPLITTER_W, cursor="sb_h_double_arrow",
        )
        theme.register(splitter, bg="splitter")
        splitter.pack_propagate(False)
        grip = tk.Frame(splitter, width=1)
        theme.register(grip, bg="splitter_line")
        grip.pack(fill=tk.Y, expand=True)
        self._dragging = False
        for widget in (splitter, grip):
            widget.bind("<Enter>", lambda _e: self._highlight_splitter(True))
            widget.bind("<Leave>", lambda _e: self._highlight_splitter(False))
            widget.bind("<Button-1>", self._start_resize)
            widget.bind("<B1-Motion>", self._on_resize)
            widget.bind("<ButtonRelease-1>", self._end_resize)
        self._splitter = splitter
        self._grip = grip

        rail = tk.Frame(self, width=SIDEBAR_W)
        rail.pack_propagate(False)
        self.rail = rail
        # Keep the splitter on the canvas-facing edge: to the right of the rail
        # for a left sidebar, to its left for a right sidebar.
        if self._side == "left":
            rail.pack(side=tk.LEFT, fill=tk.Y)
            splitter.pack(side=tk.LEFT, fill=tk.Y)
        else:
            splitter.pack(side=tk.LEFT, fill=tk.Y)
            rail.pack(side=tk.LEFT, fill=tk.Y)

        # A plain clipping frame holds the content frame, which is ``place``d
        # inside it and scrolled by moving its ``y``. Tk's canvas scrolling
        # moves embedded widget windows with a blit; at wheel speed the rows
        # it leaves behind are not repainted in time, so the rail appeared to
        # duplicate panels ("past drawn" copies). Moving a single child window
        # with ``place`` is repainted atomically by the OS, so fast scrolling
        # stays clean.
        scrollbar = ArrowScrollbar(
            rail, command=self._on_scrollbar, width=SIDEBAR_SCROLLBAR_W,
        )
        # Keep the scrollbar on the outer edge, away from the canvas.
        scrollbar.pack(
            side=tk.LEFT if self._side == "left" else tk.RIGHT, fill=tk.Y
        )
        self._scrollbar = scrollbar

        viewport = tk.Frame(rail)
        theme.register(viewport, bg="bg")
        viewport.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._viewport = viewport

        self.inner = tk.Frame(viewport)
        theme.register(self.inner, bg="bg")
        self.inner.place(x=0, y=0, width=SIDEBAR_W)
        self._offset = 0
        self._content_h = 0
        self.inner.bind("<Configure>", lambda _e: self._update_scroll())
        viewport.bind("<Configure>", self._on_viewport_configure)

        # Both sidebars bind the wheel; each only acts on its own rail (see
        # :meth:`_on_wheel`), so ``add="+"`` lets the other handler pass through.
        root.bind_all("<MouseWheel>", self._on_wheel, add="+")
        # Panels are populated after ``add_panel`` returns, so bind the
        # list/text wheel policy once the widget tree has settled.
        self.after_idle(self.bind_wheel_targets)

    # ------------------------------------------------------------------ panels
    def add_panel(self, key, title, collapsed=False, index=None):
        """Register a panel and return the frame its content should live in.

        ``index`` places the row at that position in the rail (clamped to the
        current length); the default appends it below the existing panels.
        """
        host = PanelHost(self, key, title)
        host.set_side(self._side)
        host.set_collapsed(collapsed)
        self._hosts[key] = host
        if index is None:
            index = len(self._rows)
        index = max(0, min(index, len(self._rows)))
        self._rows.insert(index, host)
        self._repack()
        self._sync_visibility()
        return host.content

    def host(self, key):
        """The :class:`PanelHost` registered under ``key``, or ``None``."""
        return self._hosts.get(key)

    # --------------------------------------------------------------- ordering
    def row_index_of(self, item):
        """The docked-row index of ``item`` (``len(rows)`` when not docked)."""
        try:
            return self._rows.index(item)
        except ValueError:
            return len(self._rows)

    def group_of(self, key):
        """The :class:`PanelGroup` holding ``key``, or ``None`` if separate."""
        return self._group_of.get(key)

    def _remove_item_row(self, item):
        if item in self._rows:
            self._rows.remove(item)

    def _item_keys(self, item):
        if isinstance(item, PanelGroup):
            return list(item.keys)
        return [item.key]

    def _all_keys(self):
        keys = []
        for item in self._rows:
            keys.extend(self._item_keys(item))
        return keys

    # --------------------------------------------------------------- placement
    def mount(self, **grid_opts):
        """Grid the rail into its body and remember how to show it again."""
        self._grid_opts = dict(grid_opts)
        self._sync_visibility()

    def _sync_visibility(self):
        """Show the rail while it has docked panels, hide it when empty.

        A sidebar with no docked panels would otherwise leave an empty strip
        (and splitter) beside the canvas, so it is removed from the grid until
        a panel is moved back into it.
        """
        has_panels = bool(self._rows)
        if has_panels and not self.winfo_manager():
            if self._grid_opts:
                self.grid(**self._grid_opts)
            self.after_idle(self._update_scroll)
        elif not has_panels and self.winfo_manager():
            self.grid_remove()

    def toggle_side(self, item):
        """Send ``item`` (a panel or a whole group) to the opposite sidebar."""
        if self.other is None:
            return
        if isinstance(item, PanelGroup):
            self.move_group(item, self.other)
        else:
            self.move_panel(item, self.other)

    def move_panel(self, host, dest, index=None):
        """Re-dock ``host`` into sidebar ``dest`` (optionally at ``index``).

        The host keeps its master, children and state; only the frame it is
        packed into changes, so no panel content is rebuilt. A panel that is
        tabbed into a group is first detached, so it moves out on its own.
        """
        if dest is self or dest is None:
            return
        if host.floating:
            self.dock_item(host)
        if self._group_of.get(host.key) is not None:
            self.detach_key(host.key)
        self._remove_item_row(host)
        self._hosts.pop(host.key, None)
        host.dock = dest
        host.set_side(dest.side)
        dest._hosts[host.key] = host
        if index is None:
            index = len(dest._rows)
        index = max(0, min(index, len(dest._rows)))
        dest._rows.insert(index, host)
        self._repack()
        dest._repack()
        self._sync_visibility()
        dest._sync_visibility()
        dest.bind_wheel_targets()
        if self.on_move is not None:
            self.on_move(host)

    def move_group(self, group, dest):
        """Re-dock the whole tab group ``group`` into sidebar ``dest``."""
        if dest is self or dest is None:
            return
        if group.floating:
            self.dock_item(group)
        if group not in self._rows:
            return
        self._remove_item_row(group)
        hosts = {}
        for key in list(group.keys):
            hosts[key] = self._hosts.pop(key)
            self._group_of.pop(key, None)
        group.dock = dest
        group.set_side(dest.side)
        dest._rows.append(group)
        for key, host in hosts.items():
            host.dock = dest
            host.set_side(dest.side)
            dest._hosts[key] = host
            dest._group_of[key] = group
        self._repack()
        dest._repack()
        self._sync_visibility()
        dest._sync_visibility()
        dest.bind_wheel_targets()
        if self.on_move is not None:
            for host in hosts.values():
                self.on_move(host)

    # ------------------------------------------------------------- tab groups
    def _make_group(self):
        self._next_group += 1
        return PanelGroup(self, self._next_group)

    def _adopt(self, group, keys):
        """Add ``keys`` to ``group`` (expanding any collapsed members)."""
        for key in keys:
            host = self._hosts.get(key)
            if host is None:
                continue
            if not host._grouped:
                host._group_collapsed = host.collapsed
                host._grouped = True
            host.set_collapsed(False)
            self._group_of[key] = group
            if key not in group.keys:
                group.keys.append(key)

    def merge_items(self, dragged, target):
        """Tab ``dragged`` and ``target`` together into one group."""
        self._set_merge_target(None)
        if dragged is target:
            return
        dkeys = self._item_keys(dragged)
        tkeys = self._item_keys(target)
        if set(dkeys) & set(tkeys):
            return
        if isinstance(target, PanelGroup):
            group = target
            self._remove_item_row(dragged)
            incoming = dkeys
        elif isinstance(dragged, PanelGroup):
            group = dragged
            self._remove_item_row(group)
            # Recompute the target's slot after the removed row, then let the
            # group take it so the panel that was dropped onto stays put.
            target_index = self.row_index_of(target)
            self._remove_item_row(target)
            self._rows.insert(min(target_index, len(self._rows)), group)
            incoming = tkeys
        else:
            group = self._make_group()
            self._remove_item_row(dragged)
            target_index = self.row_index_of(target)
            self._remove_item_row(target)
            self._rows.insert(min(target_index, len(self._rows)), group)
            group.keys.append(target.key)
            host = self._hosts[target.key]
            host._group_collapsed = host.collapsed
            host._grouped = True
            host.set_collapsed(False)
            self._group_of[target.key] = group
            incoming = dkeys
        self._adopt(group, incoming)
        if group.keys:
            group.active = incoming[-1] if incoming else group.keys[0]
        self._repack()
        self._sync_visibility()

    def set_group_active(self, group, key):
        """Show tab ``key`` of ``group``.

        Only the group's own body is re-packed, not the whole rail: a full
        :meth:`_repack` would unmap and remap every panel in the sidebar, which
        reads as the UI being rebuilt (and flickers) on each tab click.
        """
        if key not in group.keys:
            return
        group.active = key
        self._render_group(group)
        active = self._hosts.get(group.active)
        if active is not None:
            # The host's real parent is the shared body, while the group is a
            # later child of it, so raise the tab content above the group frame
            # (see :meth:`_repack`).
            active.lift()
        self._update_scroll()

    def set_group_collapsed(self, group, collapsed):
        group.set_collapsed(collapsed)
        self._update_scroll()

    def ungroup(self, group):
        """Split ``group`` back into individual panels in place."""
        if group.floating:
            self.dock_item(group)
        if group not in self._rows:
            return
        index = self._rows.index(group)
        keys = list(group.keys)
        self._rows.pop(index)
        for offset, key in enumerate(keys):
            host = self._hosts.get(key)
            if host is None:
                continue
            self._group_of.pop(key, None)
            host._grouped = False
            host.set_collapsed(host._group_collapsed)
            self._rows.insert(index + offset, host)
        group.destroy()
        self._repack()
        self._sync_visibility()

    def detach_key(self, key):
        """Remove ``key`` from its group, dissolving a group of one.

        The detached host is left unplaced so the caller can float it or move
        it to another sidebar.
        """
        group = self._group_of.pop(key, None)
        if group is None:
            return None
        host = self._hosts.get(key)
        if key in group.keys:
            group.keys.remove(key)
        if host is not None:
            host._grouped = False
            host.set_collapsed(host._group_collapsed)
        if len(group.keys) <= 1:
            gindex = self._rows.index(group) if group in self._rows else None
            if gindex is not None:
                self._rows.pop(gindex)
            if group.keys:
                remaining = group.keys[0]
                rh = self._hosts.get(remaining)
                self._group_of.pop(remaining, None)
                if rh is not None:
                    rh._grouped = False
                    rh.set_collapsed(rh._group_collapsed)
                    if gindex is not None:
                        self._rows.insert(gindex, rh)
            group.destroy()
        elif group.active == key:
            group.active = group.keys[0]
        return host

    def _render_group(self, group):
        """Show the group's active member and refresh its tabs.

        A docked (or in-window pane) group packs the active member into its
        body; a group floated as a real window shows it in a synced companion
        window instead (see :meth:`_start_group_companion`).
        """
        items = [
            (key, self._hosts[key].title)
            for key in group.keys if key in self._hosts
        ]
        group.sync_tabs(items, group.active)
        if group.floating and group._windowed:
            self._start_group_companion(group)
            return
        for key in group.keys:
            host = self._hosts.get(key)
            if host is None:
                continue
            host.pack_forget()
            if key == group.active:
                host.set_header_visible(False)
                host.pack(in_=group.body, fill=tk.X)
                # The host's real parent is the shared body, while the group is
                # a later child of that body, so the group frame would otherwise
                # stack over (and, being opaque, hide) its own tab content. Raise
                # the active host above the group after it is placed.
                host.lift()
            else:
                host.set_header_visible(True)

    # ------------------------------------------------------- floating groups
    def _start_group_companion(self, group):
        """Show ``group``'s active member in a companion window under it.

        A group's member panels are parented to the shared body, so Tk cannot
        move them into the group's own toplevel. Instead the active member is
        promoted to a second borderless window, sized to the group's body area
        and kept in sync as the group is moved or resized. Its own move/resize
        gestures are disabled so the group remains the one window the user
        drives.
        """
        host = self._hosts.get(group.active)
        current = group._managed_host
        if current is host and host is not None:
            host.set_header_visible(False)
            group._sync_host_window()
            return
        if current is not None:
            self._stop_group_companion(group)
        if host is None:
            return
        host.pack_forget()
        host.set_header_visible(False)
        host.set_collapse_enabled(False)
        host.floating = True
        host._companion = True
        # No gutter: the group frame already draws the border and the resize
        # band around the companion, so the panel fills its window edge to edge.
        host.configure(highlightthickness=0, padx=0, pady=0)
        host._start_float_scroll()
        try:
            host._manage()
        except tk.TclError:
            host._companion = False
            host.floating = False
            host._stop_float_scroll()
            host.set_collapse_enabled(True)
            host.set_header_visible(True)
            return
        group._managed_host = host
        group._sync_host_window()
        tk.Misc.lift(host)
        # The group's own geometry may not have settled yet, so re-sync once
        # Tk has mapped and placed it.
        group.after_idle(group._sync_host_window)

    def _stop_group_companion(self, group):
        """Return a floating group's companion member to an ordinary frame."""
        host = group._managed_host
        if host is None:
            return
        group._managed_host = None
        host._stop_float_scroll()
        try:
            host._unmanage()
        except tk.TclError:
            pass
        host.floating = False
        host._companion = False
        host.set_collapse_enabled(True)
        host.set_header_visible(True)
        host.configure(highlightthickness=0, padx=0, pady=0)

    # ------------------------------------------------------------------ repack
    def _repack(self):
        for host in self._hosts.values():
            host.pack_forget()
        for item in self._rows:
            if isinstance(item, PanelGroup):
                item.pack_forget()
        for item in self._rows + self._floating:
            if isinstance(item, PanelGroup):
                self._render_group(item)
        for item in self._rows:
            if isinstance(item, PanelGroup):
                item.pack(in_=self.inner, fill=tk.X)
                active = self._hosts.get(item.active)
                if active is not None:
                    active.lift()
            else:
                item.set_header_visible(True)
                item.pack(in_=self.inner, fill=tk.X)
        self._update_scroll()

    # ------------------------------------------------------------------ docking
    def toggle_float(self, item):
        if item.floating:
            self.dock_item(item)
        else:
            self.float_item(item)

    def float_item(self, item):
        if item.floating:
            return
        item._dock_index = self.row_index_of(item)
        # A floating item is never collapsible, so make sure it starts expanded
        # (its collapse control is hidden below).
        item.set_collapsed(False)
        self._remove_item_row(item)
        item.pack_forget()
        item.floating = True
        item.set_collapse_enabled(False)
        item.float_btn.config(text=DOCK_GLYPH)
        item.configure(highlightthickness=1, padx=PANEL_FLOAT_PAD, pady=PANEL_FLOAT_PAD)
        self._floating.append(item)
        self._sync_visibility()

        offset = 40 + 24 * (len(self._floating) % 8)
        width = max(PANEL_FLOAT_MIN_W, self._body.winfo_width() // 3)
        height = max(PANEL_FLOAT_MIN_H, item.winfo_reqheight())

        if isinstance(item, PanelGroup):
            # The group's active member must leave the group body before the
            # group is promoted: its parent is the shared body in the main
            # window, and Tk refuses to pack a widget into a frame that lives in
            # a different toplevel. It is shown in a synced companion window
            # instead (see :meth:`_start_group_companion`).
            active = self._hosts.get(item.active)
            if active is not None:
                active.pack_forget()
        else:
            # Give the detached container its own scrollbar before it is sized,
            # so content taller than the window can still be reached.
            item._start_float_scroll()

        item._windowed = False
        try:
            item._manage()
            item._windowed = True
        except tk.TclError:
            # Older Tk without ``wm manage``: keep it as an in-window pane.
            pass

        if item._windowed:
            x = self._body.winfo_rootx() + offset
            y = self._body.winfo_rooty() + offset
            item._set_window_geometry(
                f"{int(width)}x{int(height)}+{int(x)}+{int(y)}"
            )
        else:
            if isinstance(item, PanelGroup):
                item._start_float_scroll()
            item.place(x=offset, y=offset, width=width, height=height)
        # The grip sits in the item's own padding gutter at the very corner,
        # clear of the scrollbar and the panel content, so it cannot cover a
        # button; the padding is also the band the edges are resized from.
        # ``place`` is measured from the inner (padded) area, so the gutter is
        # reached with a positive offset.
        item.grip.place(
            relx=1.0, rely=1.0, anchor="se",
            x=PANEL_FLOAT_PAD, y=PANEL_FLOAT_PAD,
        )
        # ``Canvas.lift`` raises a canvas *item*, so raise the widget itself.
        tk.Misc.lift(item.grip)
        item.lift()
        if isinstance(item, PanelGroup):
            # Show the active member (a companion window when windowed) and
            # refresh the tabs after the group itself is raised.
            self._render_group(item)

    def dock_item(self, item):
        if not item.floating:
            return
        if isinstance(item, PanelGroup):
            self._stop_group_companion(item)
        item.grip.place_forget()
        item._stop_float_scroll()
        if item._windowed:
            try:
                item._unmanage()
            except tk.TclError:
                pass
            item._windowed = False
        else:
            item.place_forget()
        item.floating = False
        item.set_collapse_enabled(True)
        item.float_btn.config(text=FLOAT_GLYPH)
        item.configure(highlightthickness=0, padx=0, pady=0)
        if item in self._floating:
            self._floating.remove(item)
        index = item._dock_index if item._dock_index is not None else len(self._rows)
        index = max(0, min(index, len(self._rows)))
        item._dock_index = None
        self._rows.insert(index, item)
        self._repack()
        self._sync_visibility()

    # ------------------------------------------------------------- drag ghost
    def begin_drag_ghost(self, item, x_root, y_root):
        """Show a washed-out snapshot of an item that follows the cursor.

        Used while reordering a docked item in the rail; a detached item is a
        real window and is moved directly. Tkinter cannot give an individual
        widget an alpha, so the snapshot is blended toward the panel background
        (a stand-in for opacity). It lives in a borderless top-level window
        rather than a ``place``d label inside the body: an in-window sibling
        that is moved over the rail's scrolling canvas leaves fragments of its
        previous position behind on Windows, so the panel appears to be drawn
        several times while it is dragged. A real window is moved atomically by
        the OS and never smears the widgets beneath it. Returns ``True`` when
        the snapshot is shown.
        """
        self.end_drag_ghost()
        try:
            from PIL import Image, ImageGrab, ImageTk
        except Exception:
            return False
        width, height = item.winfo_width(), item.winfo_height()
        if width <= 1 or height <= 1:
            return False
        sx, sy = item.winfo_rootx(), item.winfo_rooty()
        box = (sx, sy, sx + width, sy + height)
        try:
            shot = ImageGrab.grab(bbox=box, all_screens=True)
        except TypeError:
            try:
                shot = ImageGrab.grab(bbox=box)
            except Exception:
                return False
        except Exception:
            return False
        if shot is None:
            return False
        shot = shot.convert("RGB")
        washed = Image.blend(
            Image.new("RGB", shot.size, theme.color("ghost_blend")), shot, 0.6
        )
        photo = ImageTk.PhotoImage(washed)
        window = tk.Toplevel(self._body)
        window.overrideredirect(True)
        try:
            window.attributes("-topmost", True)
        except tk.TclError:
            pass
        overlay = tk.Label(
            window, image=photo, bd=0,
            highlightthickness=1, highlightbackground=theme.color("float_border"),
        )
        overlay.pack()
        window.geometry(f"{width}x{height}+{sx}+{sy}")
        self._ghost = {
            "widget": window, "photo": photo,
            "x": sx, "y": sy,
            "x_root": x_root, "y_root": y_root,
        }
        return True

    def move_drag_ghost(self, x_root, y_root):
        ghost = self._ghost
        if ghost is None:
            return
        x = ghost["x"] + (x_root - ghost["x_root"])
        y = ghost["y"] + (y_root - ghost["y_root"])
        ghost["widget"].geometry(f"+{int(x)}+{int(y)}")

    def end_drag_ghost(self):
        ghost = self._ghost
        if ghost is not None:
            ghost["widget"].destroy()
            self._ghost = None

    # ------------------------------------------------------- drag to reorganize
    def drag_over(self, item, x_root, y_root):
        """React to a docked item dragged over the rail.

        A drop inside another item's header band merges the two into a tab
        group (with the target highlighted); anywhere else reorders the item.
        """
        target = self._merge_target_for(item, x_root, y_root)
        if target is not None:
            self._set_merge_target(target)
            return
        self._set_merge_target(None)
        self.reorder_dragged(item, y_root)

    def drop_item(self, item):
        target = self._merge_target
        self._set_merge_target(None)
        if target is not None:
            self.merge_items(item, target)

    def _set_merge_target(self, target):
        if target is self._merge_target:
            return
        if self._merge_target is not None:
            try:
                self._merge_target._highlight(False)
            except tk.TclError:
                pass
        self._merge_target = target
        if target is not None:
            try:
                target._highlight(True)
            except tk.TclError:
                pass

    def _item_at(self, x_root, y_root):
        """The docked row whose box contains ``(x_root, y_root)``, or ``None``."""
        left = self.inner.winfo_rootx()
        right = left + self.inner.winfo_width()
        if not (left <= x_root <= right):
            return None
        for item in self._rows:
            try:
                top = item.winfo_rooty()
                height = item.winfo_height()
            except tk.TclError:
                continue
            if top <= y_root < top + height:
                return item
        return None

    def _merge_target_for(self, item, x_root, y_root):
        target = self._item_at(x_root, y_root)
        if target is None or target is item:
            return None
        if isinstance(item, PanelHost) and self._group_of.get(item.key) is target:
            return None
        try:
            top = target.winfo_rooty()
        except tk.TclError:
            return None
        if y_root - top > MERGE_BAND:
            return None
        return target

    def reorder_dragged(self, item, y_root):
        """Move ``item`` to the slot under the pointer while dragging."""
        if item not in self._rows:
            return
        rest = [row for row in self._rows if row is not item]
        insert = 0
        for other in rest:
            mid = other.winfo_rooty() + other.winfo_height() / 2.0
            if y_root > mid:
                insert += 1
        new_rows = rest[:insert] + [item] + rest[insert:]
        if new_rows != self._rows:
            self._rows = new_rows
            self._repack()

    # ------------------------------------------------------------------ rail
    def _start_resize(self, _event):
        self._dragging = True
        self.configure(cursor="sb_h_double_arrow")
        self._highlight_splitter(True)

    def _end_resize(self, _event):
        self._dragging = False
        self.configure(cursor="")
        self._highlight_splitter(True)

    def _highlight_splitter(self, active):
        if active or not self._dragging:
            self._splitter.configure(
                bg=theme.color("splitter_hover" if active else "splitter")
            )
            self._grip.configure(
                bg=theme.color("splitter_hover_line" if active else "splitter_line")
            )

    def _wheel_steps(self, event):
        """Wheel notches as unit steps, matching the rail's own direction."""
        steps = int(-event.delta / 120)
        if steps == 0 and event.delta != 0:
            steps = -1 if event.delta > 0 else 1
        return steps

    def _host_of(self, widget):
        """The :class:`PanelHost` a widget is nested in, if any."""
        node = widget
        while node is not None:
            if isinstance(node, PanelHost):
                return node
            node = getattr(node, "master", None)
        return None

    def _list_wheel(self, event):
        """Plain wheel over a rail list/text scrolls the rail, not the widget.

        Docked panels surrender the wheel to the sidebar so sweeping down the
        rail is never interrupted by a list grabbing the scroll mid-sweep. A
        floating panel keeps native widget scrolling, since it is not part of
        the rail.
        """
        host = self._host_of(event.widget)
        if host is not None and host.floating:
            return None
        self._scroll_by(self._wheel_steps(event) * SCROLL_UNIT)
        return "break"

    def _list_ctrl_wheel(self, event):
        """Ctrl+wheel scrolls the list/text widget itself."""
        event.widget.yview_scroll(self._wheel_steps(event), "units")
        return "break"

    def bind_wheel_targets(self):
        """Give rail Listbox/Text widgets the flipped wheel policy.

        Called once the panels are built; widgets created later, or panels
        moved in from the other sidebar, are picked up on the next call.
        """
        for host in self._hosts.values():
            for widget in self._walk_descendants(host):
                if isinstance(widget, (tk.Listbox, tk.Text)):
                    widget.bind("<MouseWheel>", self._list_wheel)
                    widget.bind("<Control-MouseWheel>", self._list_ctrl_wheel)

    def _walk_descendants(self, widget):
        """Yield same-window descendants, skipping detached toplevels."""
        for child in widget.winfo_children():
            if isinstance(child, tk.Toplevel):
                continue
            yield child
            yield from self._walk_descendants(child)

    def _on_wheel(self, event):
        try:
            widget = self.winfo_containing(event.x_root, event.y_root)
        except KeyError:
            # winfo_containing resolves the widget path via nametowidget,
            # which can fail when the pointer is over a transient window
            # such as a ttk.Combobox popdown (KeyError: 'popdown').
            return
        while widget is not None:
            if isinstance(widget, (tk.Listbox, tk.Text)):
                return
            if widget in (self.rail, self.inner, self._viewport):
                self._scroll_by(int(-event.delta / 120) * SCROLL_UNIT)
                return
            if isinstance(widget, _RailItem):
                # Only the sidebar that currently owns the item acts on the
                # wheel; the paired sidebar's handler passes the event through.
                # A docked item scrolls its rail, a floating one its viewport.
                if widget.dock is self:
                    units = int(-event.delta / 120) * SCROLL_UNIT
                    item = widget
                    if isinstance(widget, PanelHost):
                        # A grouped panel's host keeps its own (non-floating)
                        # flag while the group it belongs to is detached, and
                        # its parent chain never passes through the group, so
                        # resolve the group here to reach the float viewport.
                        group = self._group_of.get(widget.key)
                        if group is not None:
                            item = group
                    if item.floating:
                        # A group floated as a real window has no viewport of
                        # its own; its active member's companion window scrolls.
                        if item._fviewport is not None:
                            item._float_scroll_by(units)
                        elif item._managed_host is not None:
                            item._managed_host._float_scroll_by(units)
                    else:
                        self._scroll_by(units)
                return
            widget = getattr(widget, "master", None)

    # --------------------------------------------------------------- scrolling
    def _content_height(self):
        return self.inner.winfo_reqheight()

    def _max_offset(self):
        return max(0, self._content_height() - self._viewport.winfo_height())

    def _update_scroll(self):
        """Clamp the offset and refresh the scrollbar after a size change."""
        self._content_h = self._content_height()
        max_offset = self._max_offset()
        if self._offset > max_offset:
            self._offset = max_offset
            self.inner.place_configure(y=-self._offset)
        self._sync_scrollbar()

    def _sync_scrollbar(self):
        total = float(self._content_h) or 1.0
        view = self._viewport.winfo_height()
        self._scrollbar.set(
            self._offset / total,
            min(1.0, (self._offset + view) / total),
        )

    def _scroll_by(self, delta):
        self._offset += delta
        self._scroll_to(self._offset)

    def _scroll_to(self, offset):
        offset = max(0, min(offset, self._max_offset()))
        self._offset = offset
        self.inner.place_configure(y=-offset)
        self._content_h = self._content_height()
        self._sync_scrollbar()

    def _on_scrollbar(self, *args):
        if not args:
            return
        action = args[0]
        if action == "moveto":
            self._scroll_to(int(float(args[1]) * self._content_height()))
        elif action == "scroll":
            number = int(args[1])
            what = args[2] if len(args) > 2 else "units"
            step = self._viewport.winfo_height() if what == "pages" else SCROLL_UNIT
            self._scroll_by(number * step)

    def _on_viewport_configure(self, event):
        self.inner.place_configure(width=event.width)
        self._update_scroll()

    def _on_resize(self, event):
        body = self._body
        if self._side == "left":
            # The left edge is pinned to the body; the cursor's distance from it
            # is the new rail width.
            new_w = event.x_root - self.winfo_rootx()
        else:
            # The right edge is pinned to the body; measure inward from it.
            right_edge = body.winfo_rootx() + body.winfo_width()
            new_w = right_edge - event.x_root - 8
        # Leave room for the canvas and the sidebar on the other side.
        max_w = max(SIDEBAR_MIN_W, body.winfo_width() // 2)
        new_w = max(SIDEBAR_MIN_W, min(new_w, max_w))
        self.rail.configure(width=new_w)
        self._viewport.configure(width=new_w)
