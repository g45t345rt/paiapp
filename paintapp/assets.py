"""The per-canvas Assets panel.

A canvas keeps a list of reference images - finished generations and files the
user added - beside its layer stack (see :class:`paintapp.document.Asset`). This
panel lists the active canvas's assets as a scrollable column of thumbnails, so
the user can pick one and insert it onto a layer, save it to disk, rename or
delete it. It is purely presentational: every mutation is delegated to a
callback supplied by the coordinator.
"""

import tkinter as tk

from PIL import Image, ImageTk

from . import theme
from .canvasarea import checkerboard
from .scrollbar import ArrowScrollbar
from .ui import PopupMenu

# Longest thumbnail edge, in pixels. Rows are a fixed height so the list stays a
# tidy column regardless of an asset's aspect ratio.
THUMB = 48
ROW_PAD = 2

# Longest edge of the drag ghost, in pixels. It is a little larger than a row's
# thumbnail so it reads as something being carried, not as a second list row.
GHOST = 64

# How far the pointer must move before a press turns into a drag, in pixels.
DRAG_SLOP = 6


class AssetsPanel(tk.Frame):
    """A thumbnail list of the active canvas's assets.

    ``on_add()`` asks the coordinator to open an image file, and
    ``on_add_layer()`` asks it to store the selected layer as a new asset. The
    remaining callbacks receive the index of the selected asset: ``on_preview(index)`` shows
    it in the canvas viewport for review, ``on_insert(index)`` puts it onto a new
    layer, ``on_save(index)`` writes it to a file, ``on_rename(index)`` renames it
    and ``on_delete(index)`` removes it. The panel keeps no asset state of its own
    beyond the current selection, so it is rebuilt from :meth:`refresh` whenever
    the canvas or its assets change.

    A row can also be dragged out of the panel: a washed-out ghost of the asset
    follows the cursor and the coordinator is told where it is, so it can
    highlight a drop target, then told when the drag ends so it can accept the
    drop. ``on_drag(index, x_root, y_root)`` fires on every move while dragging
    and ``on_drop(index, x_root, y_root)`` once when the button is released.

    ``on_select(index)`` fires when the user changes the selected row (not on
    :meth:`refresh`), so the coordinator can react to the pick - for instance by
    restoring a generated asset's settings into the Generate panel.
    """

    def __init__(self, parent, on_add, on_insert=None, on_save=None,
                 on_rename=None, on_delete=None, on_drag=None, on_drop=None,
                 on_preview=None, on_select=None, on_add_layer=None):
        super().__init__(parent)
        self._on_add = on_add
        self._on_add_layer = on_add_layer
        self._on_insert = on_insert
        self._on_save = on_save
        self._on_rename = on_rename
        self._on_delete = on_delete
        self._on_drag = on_drag
        self._on_drop = on_drop
        self._on_preview = on_preview
        self._on_select = on_select
        self._assets = []
        self._photos = []
        self._rows = []
        self.selected = None
        self._press = None
        self._drag_index = None
        self._ghost = None
        self._build()
        theme.subscribe(self._on_theme)

    # ------------------------------------------------------------------ build
    def _build(self):
        bar = tk.Frame(self)
        bar.pack(fill=tk.X)
        self.add_btn = tk.Button(bar, text="Add Image\u2026", command=self._add)
        self.add_btn.pack(side=tk.LEFT)
        self.add_layer_btn = tk.Button(
            bar, text="Add Layer", command=self._add_layer
        )
        self.add_layer_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.count_var = tk.StringVar(value="0 assets")
        count = tk.Label(bar, textvariable=self.count_var, anchor="e")
        count.pack(side=tk.RIGHT)
        theme.register(count, fg="muted")

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.canvas = tk.Canvas(
            body, highlightthickness=0, height=160, bg=theme.color("listbox_bg"),
            takefocus=1,
        )
        scrollbar = ArrowScrollbar(body, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Delete removes the selected asset while the list has focus; the
        # "break" keeps the global Delete shortcut from also acting on a layer.
        self.canvas.bind("<Delete>", self._on_delete_key)
        self.canvas.bind("<BackSpace>", self._on_delete_key)
        self.inner = tk.Frame(self.canvas, bg=theme.color("listbox_bg"))
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_ctrl_wheel(self.canvas)

    # ----------------------------------------------------------------- public
    def refresh(self, assets, selected=None):
        """Rebuild the list for ``assets``, keeping ``selected`` if it exists."""
        self._assets = list(assets or [])
        if selected is None:
            selected = self.selected
        if selected is None or not (0 <= selected < len(self._assets)):
            selected = 0 if self._assets else None
        self.selected = selected
        self._rebuild()

    def selected_index(self):
        return self.selected

    # ------------------------------------------------------------------ rows
    def _rebuild(self):
        # A rebuild tears the rows down, so any in-flight drag loses its origin
        # row; drop the press/ghost state rather than leave it dangling.
        self._press = None
        self._drag_index = None
        self._end_ghost()
        for child in self.inner.winfo_children():
            child.destroy()
        self._photos = []
        self._rows = []
        bg = theme.color("listbox_bg")
        fg = theme.color("text")
        muted = theme.color("muted")
        for index, asset in enumerate(self._assets):
            row = tk.Frame(self.inner, bg=bg)
            row.pack(fill=tk.X, pady=1)
            photo = ImageTk.PhotoImage(self._thumb(asset.image))
            self._photos.append(photo)
            thumb = tk.Label(row, image=photo, bg=bg, bd=0)
            thumb.pack(side=tk.LEFT, padx=ROW_PAD, pady=ROW_PAD)
            text_col = tk.Frame(row, bg=bg)
            text_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, ROW_PAD))
            name_label = tk.Label(
                text_col, text=asset.name, anchor="w", bg=bg, fg=fg,
            )
            name_label.pack(fill=tk.X)
            info_label = tk.Label(
                text_col, text=self._info(asset), anchor="w", bg=bg, fg=muted,
            )
            info_label.pack(fill=tk.X)
            widgets = (row, thumb, text_col, name_label, info_label)
            for widget in widgets:
                widget.bind(
                    "<Button-1>", lambda e, i=index: self._on_press(e, i)
                )
                widget.bind("<B1-Motion>", self._on_motion)
                widget.bind("<ButtonRelease-1>", self._on_release)
                widget.bind(
                    "<Double-Button-1>", lambda _e, i=index: self._activate(i)
                )
                widget.bind(
                    "<Button-3>", lambda e, i=index: self._context(e, i)
                )
            self._bind_ctrl_wheel(row)
            self._rows.append({
                "row": row,
                "thumb": thumb,
                "text": text_col,
                "name": name_label,
                "info": info_label,
            })
        self._apply_selection()
        self._update_count()

    def _info(self, asset):
        """The secondary line under an asset's name: size and where it came from."""
        width, height = asset.image.size
        source = (
            asset.source if asset.source in ("generated", "layer")
            else "imported"
        )
        return f"{width} \u00d7 {height}  \u00b7  {source}"

    def _thumb(self, image):
        """An ``RGB`` thumbnail of ``image`` over a checkerboard, fitted to a box."""
        source = image.convert("RGBA")
        source.thumbnail((THUMB, THUMB), Image.LANCZOS)
        board = checkerboard(source.size)
        board.alpha_composite(source)
        return board.convert("RGB")

    def _update_count(self):
        count = len(self._assets)
        self.count_var.set(f"{count} asset" if count == 1 else f"{count} assets")

    def _apply_selection(self):
        sel_bg = theme.color("sel_bg")
        sel_fg = theme.color("sel_fg")
        bg = theme.color("listbox_bg")
        fg = theme.color("text")
        muted = theme.color("muted")
        for index, entry in enumerate(self._rows):
            selected = index == self.selected
            row_bg = sel_bg if selected else bg
            name_fg = sel_fg if selected else fg
            info_fg = sel_fg if selected else muted
            entry["row"].config(bg=row_bg)
            entry["thumb"].config(bg=row_bg)
            entry["text"].config(bg=row_bg)
            entry["name"].config(bg=row_bg, fg=name_fg)
            entry["info"].config(bg=row_bg, fg=info_fg)

    def _select(self, index):
        if not (0 <= index < len(self._assets)):
            return
        changed = index != self.selected
        self.selected = index
        self._apply_selection()
        if changed and self._on_select is not None:
            self._on_select(index)

    def _activate(self, index):
        self._select(index)
        # Forget the press that started the double click so it cannot drag.
        self._press = None
        self._drag_index = None
        if self._on_preview is not None:
            self._on_preview(index)

    # ------------------------------------------------------------------- drag
    def _on_press(self, event, index):
        """Select the row and remember where the press landed, for a drag."""
        self.canvas.focus_set()
        self._select(index)
        self._press = {
            "index": index, "x_root": event.x_root, "y_root": event.y_root,
        }
        self._drag_index = None

    def _on_delete_key(self, _event):
        """Delete the selected asset, keeping the global Delete off a layer."""
        self._delete()
        return "break"

    def _on_motion(self, event):
        """Start the ghost once the pointer has moved far enough, then drag it."""
        press = self._press
        if press is None:
            return
        if self._drag_index is None:
            dx = event.x_root - press["x_root"]
            dy = event.y_root - press["y_root"]
            if dx * dx + dy * dy < DRAG_SLOP * DRAG_SLOP:
                return
            if not self._begin_ghost(press["index"], event.x_root, event.y_root):
                self._press = None
                return
            self._drag_index = press["index"]
        self._move_ghost(event.x_root, event.y_root)
        if self._on_drag is not None:
            self._on_drag(self._drag_index, event.x_root, event.y_root)

    def _on_release(self, event):
        """Finish a drag: drop the ghost and offer the asset to the coordinator."""
        press = self._press
        index = self._drag_index
        self._press = None
        self._drag_index = None
        if press is None or index is None:
            return
        self._end_ghost()
        if self._on_drop is not None:
            self._on_drop(index, event.x_root, event.y_root)

    def _begin_ghost(self, index, x_root, y_root):
        """Show a washed-out snapshot of the asset that follows the cursor.

        Tkinter cannot give an individual widget an alpha, so (as the sidebar's
        panel ghost does) the thumbnail is blended toward the panel background
        as a stand-in for opacity. It lives in a borderless top-level window
        rather than a ``place``d label, so the OS moves it atomically and leaves
        no smears on the widgets beneath. Returns ``True`` when it is shown.
        """
        self._end_ghost()
        if not (0 <= index < len(self._assets)):
            return False
        try:
            source = self._assets[index].image.convert("RGBA")
            source.thumbnail((GHOST, GHOST), Image.LANCZOS)
            board = checkerboard(source.size)
            board.alpha_composite(source)
            washed = Image.blend(
                board.convert("RGB"),
                Image.new("RGB", board.size, theme.color("ghost_blend")),
                0.4,
            )
            photo = ImageTk.PhotoImage(washed)
            window = tk.Toplevel(self)
            window.overrideredirect(True)
            try:
                window.attributes("-topmost", True)
            except tk.TclError:
                pass
            overlay = tk.Label(
                window, image=photo, bd=0, highlightthickness=1,
                highlightbackground=theme.color("float_border"),
            )
            overlay.pack()
        except tk.TclError:
            return False
        self._ghost = {
            "widget": window, "photo": photo,
            "width": washed.width, "height": washed.height,
        }
        self._move_ghost(x_root, y_root)
        return True

    def _move_ghost(self, x_root, y_root):
        ghost = self._ghost
        if ghost is None:
            return
        x = x_root - ghost["width"] // 2
        y = y_root - ghost["height"] // 2
        ghost["widget"].geometry(f"+{int(x)}+{int(y)}")

    def _end_ghost(self):
        ghost = self._ghost
        if ghost is not None:
            try:
                ghost["widget"].destroy()
            except tk.TclError:
                pass
            self._ghost = None

    # ---------------------------------------------------------------- actions
    def _add(self):
        if self._on_add is not None:
            self._on_add()

    def _add_layer(self):
        """Ask the coordinator to store the selected layer as a new asset."""
        if self._on_add_layer is not None:
            self._on_add_layer()

    def _preview(self):
        if self._on_preview is not None and self.selected is not None:
            self._on_preview(self.selected)

    def _insert(self):
        if self._on_insert is not None and self.selected is not None:
            self._on_insert(self.selected)

    def _save(self):
        if self._on_save is not None and self.selected is not None:
            self._on_save(self.selected)

    def _rename(self):
        if self._on_rename is not None and self.selected is not None:
            self._on_rename(self.selected)

    def _delete(self):
        if self._on_delete is not None and self.selected is not None:
            self._on_delete(self.selected)

    def _context(self, event, index):
        """Post a small menu of this asset's actions under the cursor."""
        self._select(index)
        menu = PopupMenu(self)
        menu.add_command("Preview", command=self._preview)
        menu.add_command("Insert as Layer", command=self._insert)
        menu.add_command("Save As\u2026", command=self._save)
        menu.add_command("Rename\u2026", command=self._rename)
        menu.add_separator()
        menu.add_command("Delete", command=self._delete)
        menu.post(event.x_root, event.y_root)
        return "break"

    # ---------------------------------------------------------------- scrolling
    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _on_ctrl_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    def _bind_ctrl_wheel(self, widget):
        """Let Ctrl+wheel scroll the list from anywhere over it.

        A plain wheel is left alone, so it scrolls the sidebar rail as usual. A
        canvas only receives a binding over itself, not over the row widgets
        embedded in it, so each row (and every child) gets the Ctrl binding too.
        """
        widget.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        for child in widget.winfo_children():
            self._bind_ctrl_wheel(child)

    def _on_theme(self):
        bg = theme.color("listbox_bg")
        try:
            self.canvas.config(bg=bg)
            self.inner.config(bg=bg)
        except tk.TclError:
            return
        self._rebuild()
