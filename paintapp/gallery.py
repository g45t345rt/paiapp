"""A zoomable, pannable overview of every canvas in the project.

The gallery is an alternative to the canvas tab editor: instead of showing one
document at a time, it lays every canvas out as a thumbnail in a grid or a
single row. The user can zoom and pan the whole board, click a thumbnail to
select it, and double-click (or press Enter) to open it in the editor.

The view keeps its own camera - a ``zoom`` factor and a screen-space ``pan`` -
so thumbnails are laid out once in "world" coordinates and then mapped to the
widget with :meth:`_to_screen`. Panning moves the existing canvas items (cheap)
while zooming re-renders the cached thumbnails at the new scale.
"""

import math
import tkinter as tk

from PIL import Image, ImageTk

from . import theme
from .canvasarea import checkerboard
from .constants import (
    GALLERY_DOCK_W,
    GALLERY_DOCK_MIN_W,
    SIDEBAR_SCROLLBAR_W,
    SIDEBAR_SPLITTER_W,
)
from .scrollbar import ArrowScrollbar

# Largest thumbnail edge, in world pixels (before zoom).
THUMB = 220
GAP = 28
LABEL_H = 22

MIN_ZOOM = 0.05
MAX_ZOOM = 8.0
ZOOM_STEP = 1.25


class GalleryView(tk.Frame):
    """A board of canvas thumbnails with grid/row modes, zoom and pan.

    ``on_activate(index)`` opens the document at ``index`` in the editor and
    ``on_close()`` leaves the gallery. ``on_zoom()`` fires whenever the camera
    zoom changes, so the host can mirror it in the status bar.
    ``on_context(index, x_root, y_root)`` posts the thumbnail's context menu,
    which the host fills with rename/duplicate/delete and other canvas actions.
    """

    def __init__(self, parent, on_activate, on_close=None, on_zoom=None,
                 on_context=None):
        super().__init__(parent)
        self._on_activate = on_activate
        self._on_close = on_close
        self._on_zoom = on_zoom
        self._on_context = on_context

        self.documents = []
        self.mode = "grid"
        self.zoom = 1.0
        self._pan = (0.0, 0.0)
        self.selected = 0
        self._items = []
        self._bounds = (0, 0, 1, 1)
        self._cols = 1
        self._bases = {}
        self._photos = {}
        self._drag = None
        self._moved = False
        self._needs_fit = True
        self._last_size = (0, 0)

        self._build_toolbar()
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=theme.color("bg"))
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self._bind()
        # The gallery is built lazily, after the app's initial theme pass, so
        # style it now; later theme changes reach it through the tree walk.
        theme.restyle(self)
        theme.subscribe(self._on_theme)

    # ---------------------------------------------------------------- chrome
    def _build_toolbar(self):
        bar = tk.Frame(self)
        bar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 0))
        self._bar = bar

        tk.Button(bar, text="Grid", width=6,
                  command=lambda: self.set_mode("grid")).pack(side=tk.LEFT)
        tk.Button(bar, text="Row", width=6,
                  command=lambda: self.set_mode("row")).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(bar, text="Zoom In", width=8,
                  command=self.zoom_in).pack(side=tk.LEFT, padx=(12, 0))
        tk.Button(bar, text="Zoom Out", width=8,
                  command=self.zoom_out).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(bar, text="Fit", width=5,
                  command=self.fit).pack(side=tk.LEFT, padx=(4, 0))
        tk.Button(bar, text="Open", width=6,
                  command=self.open_selected).pack(side=tk.LEFT, padx=(12, 0))
        tk.Button(bar, text="Close", width=6,
                  command=self._close).pack(side=tk.LEFT, padx=(4, 0))

        self._info_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._info_var, anchor="e").pack(
            side=tk.RIGHT, padx=6
        )

    def _bind(self):
        c = self.canvas
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_motion)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", self._on_double)
        c.bind("<Button-3>", self._on_context_click)
        c.bind("<ButtonPress-2>", self._on_press)
        c.bind("<B2-Motion>", self._on_motion)
        c.bind("<ButtonRelease-2>", self._on_release_pan)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Configure>", self._on_configure)
        c.bind("<Left>", lambda _e: self._move_selection(-1))
        c.bind("<Right>", lambda _e: self._move_selection(1))
        c.bind("<Up>", lambda _e: self._move_selection(-self._cols))
        c.bind("<Down>", lambda _e: self._move_selection(self._cols))
        c.bind("<Return>", lambda _e: self.open_selected())

    def focus_canvas(self):
        """Give the board keyboard focus so arrows and Enter work."""
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass

    # --------------------------------------------------------------- driving
    def set_documents(self, documents, selected=0):
        """Show ``documents``, marking ``selected`` as the current canvas."""
        self.documents = list(documents)
        self.selected = max(0, min(int(selected), len(self.documents) - 1))
        self._bases = {}
        self._photos = {}
        self._needs_fit = True
        self._relayout()
        if self.canvas.winfo_width() > 1:
            self._needs_fit = False
            self.fit()
        else:
            self._render()

    def set_mode(self, mode):
        """Switch between ``"grid"`` and ``"row"`` layout and refit."""
        mode = "row" if mode == "row" else "grid"
        if mode == self.mode:
            return
        self.mode = mode
        self._relayout()
        self.fit()

    def _close(self):
        if self._on_close is not None:
            self._on_close()

    def open_selected(self):
        if self.documents and self._on_activate is not None:
            self._on_activate(self.selected)

    # ----------------------------------------------------------------- layout
    def _relayout(self):
        """Recompute every thumbnail's world rectangle from the current size."""
        self._items = []
        self._bases = {}
        count = len(self.documents)
        if count == 0:
            self._bounds = (0, 0, 1, 1)
            self._cols = 1
            return
        view_w = self.canvas.winfo_width()
        view_h = self.canvas.winfo_height()
        view_w = view_w if view_w > 1 else 900
        view_h = view_h if view_h > 1 else 600

        if self.mode == "row":
            cols = count
        else:
            cols = max(1, int((view_w - GAP) / (THUMB + GAP)))
            cols = min(cols, count)
        self._cols = cols
        rows = math.ceil(count / cols)

        cell_w = THUMB + GAP
        cell_h = THUMB + LABEL_H + GAP
        for index, doc in enumerate(self.documents):
            row, col = divmod(index, cols)
            scale = min(THUMB / doc.width, THUMB / doc.height)
            w = max(1, round(doc.width * scale))
            h = max(1, round(doc.height * scale))
            x = col * cell_w + (cell_w - w) / 2
            y = row * cell_h + (THUMB - h) / 2
            self._items.append({
                "index": index, "doc": doc, "x": x, "y": y, "w": w, "h": h,
            })
        self._bounds = (0, 0, cols * cell_w, rows * cell_h)

    # --------------------------------------------------------------- camera
    def _to_screen(self, x, y):
        return x * self.zoom + self._pan[0], y * self.zoom + self._pan[1]

    def _view_size(self):
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        return (w if w > 1 else 900), (h if h > 1 else 600)

    def fit(self):
        """Zoom so every thumbnail is visible, centered in the viewport."""
        if not self._items:
            self._render()
            return
        view_w, view_h = self._view_size()
        right, bottom = self._bounds[2], self._bounds[3]
        margin = 40
        zoom_x = (view_w - 2 * margin) / right if right else 1.0
        zoom_y = (view_h - 2 * margin) / bottom if bottom else 1.0
        self.zoom = max(MIN_ZOOM, min(MAX_ZOOM, min(zoom_x, zoom_y)))
        self._pan = (
            (view_w - right * self.zoom) / 2,
            (view_h - bottom * self.zoom) / 2,
        )
        self._photos = {}
        self._render()

    def _zoom_at(self, px, py, factor):
        new = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom * factor))
        if abs(new - self.zoom) < 1e-9:
            return
        world_x = (px - self._pan[0]) / self.zoom
        world_y = (py - self._pan[1]) / self.zoom
        self.zoom = new
        self._pan = (px - world_x * new, py - world_y * new)
        self._photos = {}
        self._render()

    def zoom_in(self):
        w, h = self._view_size()
        self._zoom_at(w / 2, h / 2, ZOOM_STEP)

    def zoom_out(self):
        w, h = self._view_size()
        self._zoom_at(w / 2, h / 2, 1 / ZOOM_STEP)

    # -------------------------------------------------------------- rendering
    def _base(self, item):
        """The thumbnail's RGB base image, cached per index at world size."""
        index = item["index"]
        cached = self._bases.get(index)
        if cached is not None:
            return cached
        source = item["doc"].composite_transparent()
        if source.size != (item["w"], item["h"]):
            source = source.resize((item["w"], item["h"]), Image.LANCZOS)
        board = checkerboard((item["w"], item["h"]))
        board.alpha_composite(source)
        image = board.convert("RGB")
        self._bases[index] = image
        return image

    def _photo(self, item):
        """The thumbnail rendered at the current zoom, cached per index."""
        index = item["index"]
        cached = self._photos.get(index)
        if cached is not None and abs(cached[0] - self.zoom) < 1e-6:
            return cached[1]
        base = self._base(item)
        width = max(1, round(item["w"] * self.zoom))
        height = max(1, round(item["h"] * self.zoom))
        if base.size != (width, height):
            resample = Image.NEAREST if self.zoom >= 1.0 else Image.LANCZOS
            base = base.resize((width, height), resample)
        photo = ImageTk.PhotoImage(base)
        self._photos[index] = (self.zoom, photo)
        return photo

    def _render(self):
        canvas = self.canvas
        canvas.delete("all")
        text_color = theme.color("text")
        border = theme.color("border")
        accent = theme.color("accent")

        if not self._items:
            canvas.create_text(
                canvas.winfo_width() // 2 or 450,
                canvas.winfo_height() // 2 or 300,
                text="No canvases", fill=theme.color("muted"),
            )
        for item in self._items:
            sx, sy = self._to_screen(item["x"], item["y"])
            width = item["w"] * self.zoom
            height = item["h"] * self.zoom
            canvas.create_image(sx, sy, anchor=tk.NW, image=self._photo(item))
            selected = item["index"] == self.selected
            canvas.create_rectangle(
                sx, sy, sx + width, sy + height,
                outline=accent if selected else border,
                width=3 if selected else 1,
            )
            canvas.create_text(
                sx + width / 2, sy + height + LABEL_H / 2,
                text=item["doc"].name, fill=text_color, anchor=tk.CENTER,
            )

        count = len(self.documents)
        plural = "" if count == 1 else "es"
        self._info_var.set(f"{count} canvas{plural} · {self.zoom * 100:.0f}%")
        if self._on_zoom is not None:
            self._on_zoom()

    # ------------------------------------------------------------------ input
    def _hit(self, x, y):
        """Index of the thumbnail under widget position ``(x, y)`` or ``None``."""
        for item in reversed(self._items):
            sx, sy = self._to_screen(item["x"], item["y"])
            if sx <= x <= sx + item["w"] * self.zoom and \
                    sy <= y <= sy + item["h"] * self.zoom:
                return item["index"]
        return None

    def _on_press(self, event):
        self.focus_canvas()
        self._drag = (event.x, event.y)
        self._moved = False

    def _on_motion(self, event):
        if self._drag is None:
            return
        dx = event.x - self._drag[0]
        dy = event.y - self._drag[1]
        if not self._moved and abs(dx) < 4 and abs(dy) < 4:
            return
        self._moved = True
        self._drag = (event.x, event.y)
        self._pan = (self._pan[0] + dx, self._pan[1] + dy)
        self.canvas.move("all", dx, dy)

    def _on_release(self, event):
        moved = self._moved
        had_drag = self._drag is not None
        self._drag = None
        self._moved = False
        if not had_drag or moved:
            return
        index = self._hit(event.x, event.y)
        if index is not None:
            self.selected = index
            self._render()

    def _on_release_pan(self, event):
        self._drag = None
        self._moved = False

    def _on_double(self, event):
        index = self._hit(event.x, event.y)
        if index is not None:
            self.selected = index
            self.open_selected()

    def _on_context_click(self, event):
        index = self._hit(event.x, event.y)
        if index is None or self._on_context is None:
            return None
        self.focus_canvas()
        if index != self.selected:
            self.selected = index
            self._render()
        x_root, y_root = event.x_root, event.y_root
        self.after_idle(lambda: self._on_context(index, x_root, y_root))
        return "break"

    def _on_wheel(self, event):
        delta = event.delta
        if delta == 0:
            return
        self._zoom_at(event.x, event.y, ZOOM_STEP ** (delta / 120.0))

    def _on_configure(self, event):
        if event.widget is not self.canvas:
            return
        size = (event.width, event.height)
        if size == self._last_size:
            return
        self._last_size = size
        self._relayout()
        if self._needs_fit:
            self._needs_fit = False
            self.fit()
        else:
            self._render()

    def _move_selection(self, delta):
        count = len(self.documents)
        if count == 0:
            return "break"
        self.selected = (self.selected + delta) % count
        self._ensure_visible(self.selected)
        self._render()
        # Consume the arrow so the window-level canvas shortcuts do not also
        # fire while the thumbnail board has focus.
        return "break"

    def _ensure_visible(self, index):
        item = self._items[index]
        view_w, view_h = self._view_size()
        sx, sy = self._to_screen(item["x"], item["y"])
        width = item["w"] * self.zoom
        height = item["h"] * self.zoom
        if sx < 0 or sy < 0 or sx + width > view_w or sy + height > view_h:
            self._pan = (
                (view_w - width) / 2 - item["x"] * self.zoom,
                (view_h - height) / 2 - item["y"] * self.zoom,
            )

    def _on_theme(self):
        try:
            self.canvas.config(bg=theme.color("bg"))
        except tk.TclError:
            return
        self._render()


GALLERY_SIDE_MARGIN = 8
GALLERY_SIDE_GAP = 12
GALLERY_SIDE_LABEL_H = 18
GALLERY_SIDE_WHEEL = 3


class GallerySidebar(tk.Frame):
    """A vertical, scrollable filmstrip of canvas thumbnails for the dock view.

    Unlike :class:`GalleryView` (the zoomable full-screen board), this is a
    plain list: every canvas is scaled to the rail's full width and stacked top
    to bottom, with a scrollbar for the overflow. There is no zoom, pan or
    grid/row mode - widening the rail simply makes each thumbnail larger.

    ``on_activate(index)`` switches the current canvas (on click or Enter) and
    ``on_context(index, x_root, y_root)`` posts its context menu. The strip
    stays open across activation, so the user can browse canvases while the
    editor follows along.
    """

    def __init__(self, parent, on_activate, on_context=None):
        super().__init__(parent)
        self._on_activate = on_activate
        self._on_context = on_context

        self.documents = []
        self.selected = 0
        self._items = []
        self._photos = {}
        self._content_h = 0
        self._last_w = 0
        self._drag = None
        self._moved = False

        self.canvas = tk.Canvas(
            self, highlightthickness=0, bg=theme.color("bg"),
            yscrollincrement=16,
        )
        self.scrollbar = ArrowScrollbar(
            self, command=self.canvas.yview, width=SIDEBAR_SCROLLBAR_W,
        )
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self._bind()
        theme.restyle(self)
        theme.subscribe(self._on_theme)

    # --------------------------------------------------------------- input
    def _bind(self):
        c = self.canvas
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_motion)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", self._on_double)
        c.bind("<Button-3>", self._on_context_click)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Configure>", self._on_configure)
        c.bind("<Up>", lambda _e: self._move(-1))
        c.bind("<Down>", lambda _e: self._move(1))
        c.bind("<Return>", lambda _e: self.open_selected())

    def focus_canvas(self):
        """Give the list keyboard focus so arrows and Enter work."""
        try:
            self.canvas.focus_set()
        except tk.TclError:
            pass

    # --------------------------------------------------------------- driving
    def set_documents(self, documents, selected=0):
        """Show ``documents``, marking ``selected`` as the current canvas.

        The scroll position is kept where it can, so a refresh after an edit
        does not jerk the list back to the top.
        """
        fraction = self.canvas.yview()[0] if self.documents else 0.0
        self.documents = list(documents)
        self.selected = max(0, min(int(selected), len(self.documents) - 1))
        self._photos = {}
        self._layout()
        self._render()
        if fraction:
            self.canvas.yview_moveto(fraction)

    def open_selected(self):
        if self.documents and self._on_activate is not None:
            self._on_activate(self.selected)

    def _activate(self, index):
        """Select ``index`` and switch to it, keeping the strip open."""
        if index is None:
            return
        if index != self.selected:
            self.selected = index
            self._render()
        if self._on_activate is not None:
            self._on_activate(index)

    # ---------------------------------------------------------------- layout
    def _rail_width(self):
        width = self.canvas.winfo_width()
        return width if width > 1 else GALLERY_DOCK_W

    def _layout(self):
        """Recompute every thumbnail's rectangle from the current rail width."""
        self._items = []
        width = self._rail_width()
        self._last_w = width
        avail = max(40, width - 2 * GALLERY_SIDE_MARGIN - 2)
        y = GALLERY_SIDE_MARGIN
        for index, doc in enumerate(self.documents):
            scale = avail / max(1, doc.width)
            height = max(1, round(doc.height * scale))
            self._items.append({
                "index": index, "doc": doc,
                "x": GALLERY_SIDE_MARGIN, "y": y, "w": avail, "h": height,
            })
            y += height + GALLERY_SIDE_LABEL_H + GALLERY_SIDE_GAP
        self._content_h = max(0, y - GALLERY_SIDE_GAP + GALLERY_SIDE_MARGIN)

    # ------------------------------------------------------------- rendering
    def _base(self, item):
        source = item["doc"].composite_transparent()
        if source.size != (item["w"], item["h"]):
            source = source.resize((item["w"], item["h"]), Image.LANCZOS)
        board = checkerboard((item["w"], item["h"]))
        board.alpha_composite(source)
        return board.convert("RGB")

    def _photo(self, item):
        index = item["index"]
        cached = self._photos.get(index)
        if cached is not None and cached[0] == item["w"]:
            return cached[1]
        photo = ImageTk.PhotoImage(self._base(item))
        self._photos[index] = (item["w"], photo)
        return photo

    def _render(self):
        canvas = self.canvas
        canvas.delete("all")
        text_color = theme.color("text")
        border = theme.color("border")
        accent = theme.color("accent")
        if not self._items:
            canvas.create_text(
                self._rail_width() // 2, 40,
                text="No canvases", fill=theme.color("muted"),
            )
        for item in self._items:
            canvas.create_image(
                item["x"], item["y"], anchor=tk.NW, image=self._photo(item),
            )
            selected = item["index"] == self.selected
            canvas.create_rectangle(
                item["x"] - 1, item["y"] - 1,
                item["x"] + item["w"] + 1, item["y"] + item["h"] + 1,
                outline=accent if selected else border,
                width=3 if selected else 1,
            )
            canvas.create_text(
                item["x"] + item["w"] / 2,
                item["y"] + item["h"] + GALLERY_SIDE_LABEL_H / 2,
                text=item["doc"].name, fill=text_color,
            )
        height = max(self._content_h, canvas.winfo_height())
        canvas.configure(scrollregion=(0, 0, self._last_w, height))
        # Canvas does not always re-fire ``yscrollcommand`` when the scroll
        # region changes, so push the fractions to the scrollbar directly or an
        # overflowing list would keep its bar hidden.
        self.scrollbar.set(*canvas.yview())

    # ------------------------------------------------------------------- hit
    def _hit(self, y):
        """Index of the thumbnail at widget y ``y``, or ``None``."""
        cy = self.canvas.canvasy(y)
        for item in self._items:
            if item["y"] <= cy <= item["y"] + item["h"]:
                return item["index"]
        return None

    # ---------------------------------------------------------------- events
    def _on_press(self, event):
        self.focus_canvas()
        self._drag = (event.x, event.y)
        self._moved = False

    def _on_motion(self, event):
        if self._drag is None:
            return
        if not self._moved and abs(event.x - self._drag[0]) < 4 \
                and abs(event.y - self._drag[1]) < 4:
            return
        self._moved = True

    def _on_release(self, event):
        moved, self._moved = self._moved, False
        self._drag = None
        if moved:
            return
        self._activate(self._hit(event.y))

    def _on_double(self, event):
        # A single click already switches the canvas; a double-click just
        # repeats it, so there is nothing extra to do.
        self._activate(self._hit(event.y))

    def _on_context_click(self, event):
        index = self._hit(event.y)
        if index is None or self._on_context is None:
            return None
        if index != self.selected:
            self.selected = index
            self._render()
        x_root, y_root = event.x_root, event.y_root
        self.after_idle(lambda: self._on_context(index, x_root, y_root))
        return "break"

    def _on_wheel(self, event):
        if event.delta == 0:
            return
        steps = -int(event.delta / 120) * GALLERY_SIDE_WHEEL
        self.canvas.yview_scroll(steps, "units")

    def _on_configure(self, event):
        if event.widget is not self.canvas or event.width == self._last_w:
            return
        fraction = self.canvas.yview()[0] if self.documents else 0.0
        self._layout()
        self._render()
        if fraction:
            self.canvas.yview_moveto(fraction)

    def _move(self, delta):
        count = len(self.documents)
        if count == 0:
            return "break"
        self._activate((self.selected + delta) % count)
        self._scroll_selected_visible()
        return "break"

    def _scroll_selected_visible(self):
        if not self._items:
            return
        item = self._items[self.selected]
        top = self.canvas.canvasy(0)
        bottom = top + self.canvas.winfo_height()
        if item["y"] < top:
            self.canvas.yview_moveto(item["y"] / max(1, self._content_h))
        elif item["y"] + item["h"] > bottom:
            target = item["y"] + item["h"] - self.canvas.winfo_height()
            self.canvas.yview_moveto(max(0, target) / max(1, self._content_h))

    def _on_theme(self):
        try:
            self.canvas.config(bg=theme.color("bg"))
        except tk.TclError:
            return
        self._render()


class GalleryDock(tk.Frame):
    """A resizable rail that docks the thumbnail list beside the canvas.

    It mirrors the shape of a tool sidebar - a splitter on the canvas-facing
    edge and a fixed-width rail on the outside - and hosts ``content`` (the
    :class:`GallerySidebar`), so the gallery can be shown as a viewport next to
    the editor rather than replacing it. Dragging the splitter resizes the
    rail; the hosted widget's master is the canvas area, so it can be moved
    into this rail without being rebuilt.
    """

    def __init__(self, parent, content, side="left", width=GALLERY_DOCK_W):
        super().__init__(parent)
        self._body = parent
        self._side = side if side in ("left", "right") else "left"
        self.content = content
        self._dragging = False

        rail = tk.Frame(self, width=width)
        rail.pack_propagate(False)
        self.rail = rail

        splitter = tk.Frame(
            self, width=SIDEBAR_SPLITTER_W, cursor="sb_h_double_arrow",
        )
        theme.register(splitter, bg="splitter")
        splitter.pack_propagate(False)
        grip = tk.Frame(splitter, width=1)
        theme.register(grip, bg="splitter_line")
        grip.pack(fill=tk.Y, expand=True)
        self._splitter = splitter
        self._grip = grip
        for widget in (splitter, grip):
            widget.bind("<Enter>", lambda _e: self._highlight_splitter(True))
            widget.bind("<Leave>", lambda _e: self._highlight_splitter(False))
            widget.bind("<Button-1>", self._start_resize)
            widget.bind("<B1-Motion>", self._on_resize)
            widget.bind("<ButtonRelease-1>", self._end_resize)

        # Keep the splitter on the canvas-facing edge: to the right of the rail
        # for a left dock, to its left for a right dock.
        if self._side == "left":
            rail.pack(side=tk.LEFT, fill=tk.Y)
            splitter.pack(side=tk.LEFT, fill=tk.Y)
        else:
            splitter.pack(side=tk.LEFT, fill=tk.Y)
            rail.pack(side=tk.LEFT, fill=tk.Y)

        content.pack(in_=rail, fill=tk.BOTH, expand=True)
        theme.restyle(self)

    # ------------------------------------------------------------- splitter
    def _highlight_splitter(self, active):
        if active or not self._dragging:
            self._splitter.configure(
                bg=theme.color("splitter_hover" if active else "splitter")
            )
            self._grip.configure(
                bg=theme.color("splitter_hover_line" if active else "splitter_line")
            )

    def _start_resize(self, _event):
        self._dragging = True
        self.configure(cursor="sb_h_double_arrow")
        self._highlight_splitter(True)

    def _end_resize(self, _event):
        self._dragging = False
        self.configure(cursor="")
        self._highlight_splitter(True)

    def _on_resize(self, event):
        if self._side == "left":
            # The outer (left) edge is pinned; the pointer's distance from it is
            # the new rail width.
            new_w = event.x_root - self.rail.winfo_rootx()
        else:
            right_edge = self.rail.winfo_rootx() + self.rail.winfo_width()
            new_w = right_edge - event.x_root
        # Leave room for the canvas itself.
        max_w = max(GALLERY_DOCK_MIN_W, self._body.winfo_width() // 2)
        new_w = max(GALLERY_DOCK_MIN_W, min(new_w, max_w))
        self.rail.configure(width=new_w)

