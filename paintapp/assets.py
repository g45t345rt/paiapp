"""The project-wide Assets panel.

A project keeps one list of reference images - finished generations and files
the user added - shared by all of its canvases (see
:class:`paintapp.document.AssetLibrary`). This panel lists them as a scrollable
column of thumbnails, organised with search, filters, sort, groups, tags and
favorites, so the user can pick one and insert it onto a layer of any canvas,
save it to disk, rename or delete it. It is purely presentational: every mutation
is delegated to a callback supplied by the coordinator.

Rows are identified by the stable :attr:`Asset.id`, never by their position, so
filtering and grouping can reorder the view without breaking the coordinator's
mapping back to the library.
"""

import tkinter as tk

from PIL import Image, ImageTk

from . import assetview, theme
from .canvasarea import checkerboard
from .scrollbar import ArrowScrollbar
from .ui import PopupMenu

# Longest thumbnail edge, in pixels. Rows are a fixed height so the list stays a
# tidy column regardless of an asset's aspect ratio.
THUMB = 48
ROW_PAD = 2
GROUP_INDENT = 12

# Longest edge of the drag ghost, in pixels. It is a little larger than a row's
# thumbnail so it reads as something being carried, not as a second list row.
GHOST = 64

# How far the pointer must move before a press turns into a drag, in pixels.
DRAG_SLOP = 6


class AssetsPanel(tk.Frame):
    """A thumbnail list of the project's assets.

    ``on_add()`` asks the coordinator to open an image file, and
    ``on_add_layer()`` asks it to store the selected layer as a new asset. The
    other callbacks receive stable asset ids: ``on_preview(id)`` shows an asset
    in the canvas viewport, ``on_insert(id)`` puts it onto a new layer,
    ``on_save(id)`` writes it to a file, ``on_rename(id)`` renames it and
    ``on_delete(ids)`` removes the selected assets. The panel keeps only view
    state (search, filters, sort, selection), so :meth:`refresh` can rebuild it
    whenever the project's assets change.

    Organization callbacks: ``on_assign_group(ids, group_id)`` moves assets into
    a group, ``on_reorder(ids, group_id, before_id)`` applies an internal drag,
    ``on_new_group()``/``on_rename_group(id)``/``on_delete_group(id)`` manage
    groups, ``on_favorite(ids)`` flips favorites, ``on_tags(ids)`` edits tags and
    ``on_collapse_group(id, collapsed)`` records a group's collapse state.

    A row can also be dragged out of the panel: a washed-out ghost of the asset
    follows the cursor and the coordinator is told where it is, so it can
    highlight a drop target, then told when the drag ends so it can accept the
    drop. ``on_drag(id, x_root, y_root)`` fires on every move while dragging and
    ``on_drop(id, x_root, y_root)`` once when the button is released. A drag that
    stays inside the panel reorders assets or moves them between groups instead.

    ``on_select(id)`` fires when the user changes the selected row (not on
    :meth:`refresh`), so the coordinator can react to the pick - for instance by
    restoring a generated asset's settings into the Generate panel.
    """

    def __init__(self, parent, on_add, on_insert=None, on_save=None,
                 on_rename=None, on_delete=None, on_drag=None, on_drop=None,
                 on_preview=None, on_select=None, on_add_layer=None,
                 on_assign_group=None, on_reorder=None, on_move_group=None,
                 on_new_group=None, on_rename_group=None, on_delete_group=None,
                 on_favorite=None, on_tags=None, on_collapse_group=None):
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
        self._on_assign_group = on_assign_group
        self._on_reorder = on_reorder
        self._on_move_group = on_move_group
        self._on_new_group = on_new_group
        self._on_rename_group = on_rename_group
        self._on_delete_group = on_delete_group
        self._on_favorite = on_favorite
        self._on_tags = on_tags
        self._on_collapse_group = on_collapse_group

        self._assets = []
        self._groups = []
        self._photos = {}
        self._rows = []
        self._selected_ids = set()
        self._primary_id = None
        self._anchor_id = None
        self._press = None
        self._group_press = None
        self._drag_kind = None
        self._drop_target = None
        self._ghost = None
        self._line = None
        self._search_job = None
        self._state = assetview.AssetViewState()
        self._source_vars = {
            key: tk.BooleanVar(value=False)
            for key, _label in assetview.SOURCE_LABELS
        }
        self._favorites_var = tk.BooleanVar(value=False)
        self._build()
        theme.subscribe(self._on_theme)

    # ------------------------------------------------------------------ build
    def _build(self):
        search_bar = tk.Frame(self)
        search_bar.pack(fill=tk.X, pady=(4, 0))
        label = tk.Label(search_bar, text="Search")
        label.pack(side=tk.LEFT)
        theme.register(label, fg="muted")
        self.search_var = tk.StringVar()
        self.search_entry = tk.Entry(search_bar, textvariable=self.search_var)
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 2))
        self.search_entry.bind("<KeyRelease>", self._on_search_key)
        self.search_entry.bind("<Escape>", self._clear_search)
        self.search_entry.bind("<Control-f>", self._focus_search)
        clear = tk.Button(
            search_bar, text="\u00d7", width=2, command=self._clear_search,
        )
        clear.pack(side=tk.RIGHT)

        tools = tk.Frame(self)
        tools.pack(fill=tk.X, pady=(2, 0))
        self.filter_btn = tk.Button(
            tools, text="Filter \u25be", command=self._post_filter_menu,
        )
        self.filter_btn.pack(side=tk.LEFT)
        self.sort_btn = tk.Button(
            tools, text="Sort \u25be", command=self._post_sort_menu,
        )
        self.sort_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.group_btn = tk.Button(
            tools, text="+ Group", command=self._new_group,
        )
        self.group_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.favorite_btn = tk.Checkbutton(
            tools, text="\u2605", variable=self._favorites_var,
            command=self._on_favorites_changed, indicatoron=False, width=2,
        )
        self.favorite_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.clear_btn = tk.Button(
            tools, text="Clear", command=self._clear_filters,
        )
        self.clear_btn.pack(side=tk.RIGHT)

        self.filter_var = tk.StringVar(value="")
        self.filter_label = tk.Label(self, textvariable=self.filter_var,
                                     anchor="w")
        self.filter_label.pack(fill=tk.X, pady=(2, 0))
        theme.register(self.filter_label, fg="muted")

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, pady=(4, 0))
        self.canvas = tk.Canvas(
            body, highlightthickness=0, height=220, bg=theme.color("listbox_bg"),
            takefocus=1,
        )
        scrollbar = ArrowScrollbar(body, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Delete removes the selected assets while the list has focus; the
        # "break" keeps the global Delete shortcut from also acting on a layer.
        self.canvas.bind("<Delete>", self._on_delete_key)
        self.canvas.bind("<BackSpace>", self._on_delete_key)
        self.canvas.bind("<Control-f>", self._focus_search)
        self.canvas.bind("<Control-a>", self._select_all)
        self.canvas.bind("<F2>", self._on_rename_key)
        self.inner = tk.Frame(self.canvas, bg=theme.color("listbox_bg"))
        self._window = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_ctrl_wheel(self.canvas)

        footer = tk.Frame(self)
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=(4, 0))
        self.add_btn = tk.Button(
            footer, text="Add Image\u2026", command=self._add,
        )
        self.add_btn.pack(side=tk.LEFT)
        self.add_layer_btn = tk.Button(
            footer, text="Add Selected Layer", command=self._add_layer,
        )
        self.add_layer_btn.pack(side=tk.LEFT, padx=(4, 0))
        self.count_var = tk.StringVar(value="0 assets")
        count = tk.Label(footer, textvariable=self.count_var, anchor="e")
        count.pack(side=tk.RIGHT)
        theme.register(count, fg="muted")

        self._sync_controls()

    # ----------------------------------------------------------------- public
    def refresh(self, assets, groups=None, selected_id=None):
        """Rebuild the list for ``assets`` and ``groups``.

        View state (search, filters, sort) survives; selection is preserved by
        asset id. ``selected_id`` forces the selection, and the first visible
        asset is selected silently when nothing valid remains.
        """
        self._assets = list(assets or [])
        self._groups = list(groups or [])
        self._prune_filters()
        valid_ids = {asset.id for asset in self._assets}
        if selected_id is not None and selected_id in valid_ids:
            self._selected_ids = {selected_id}
            self._primary_id = selected_id
            self._anchor_id = selected_id
        else:
            self._selected_ids &= valid_ids
            if self._primary_id not in valid_ids:
                self._primary_id = None
            if self._anchor_id not in valid_ids:
                self._anchor_id = None
        self._photos = {
            asset_id: photo for asset_id, photo in self._photos.items()
            if asset_id in valid_ids
        }
        self._rebuild()
        visible = self._visible_asset_ids()
        if visible and not (self._selected_ids & set(visible)):
            self._selected_ids = {visible[0]}
            self._primary_id = visible[0]
            self._anchor_id = visible[0]
            self._apply_selection()
        elif not visible:
            self._selected_ids = set()
            self._primary_id = None
            self._anchor_id = None
            self._apply_selection()

    def select_asset(self, asset_id, reveal=False):
        """Select ``asset_id``. With ``reveal``, clear filters that hide it."""
        if reveal and asset_id not in self._visible_asset_ids():
            self._cancel_search_job()
            self._state.clear_filters()
            asset = self._asset_by_id(asset_id)
            if asset is not None:
                group = self._group_by_id(getattr(asset, "group", None))
                if group is not None and self._state.is_collapsed(group):
                    self._state.set_collapsed(group, False)
                elif group is None and self._state.is_collapsed(None):
                    self._state.set_collapsed(None, False)
            self._sync_controls()
            self._rebuild()
        if asset_id not in self._visible_asset_ids():
            return
        self._selected_ids = {asset_id}
        self._primary_id = asset_id
        self._anchor_id = asset_id
        self._apply_selection()
        self._scroll_to_asset(asset_id)

    def selected_ids(self):
        """The selected asset ids, in current view order."""
        visible = self._visible_asset_ids()
        return [asset_id for asset_id in visible if asset_id in self._selected_ids]

    def selected_asset_id(self):
        """The primary selected asset id, or ``None``."""
        if self._primary_id in self._selected_ids:
            return self._primary_id
        selected = self.selected_ids()
        return selected[0] if selected else None

    def default_group_id(self):
        """The single filter-selected group new assets should join, if any."""
        ids = self._state.group_ids
        if len(ids) == 1 and None not in ids:
            return next(iter(ids))
        return None

    # ------------------------------------------------------------- view state
    def _asset_by_id(self, asset_id):
        for asset in self._assets:
            if asset.id == asset_id:
                return asset
        return None

    def _group_by_id(self, group_id):
        for group in self._groups:
            if group.id == group_id:
                return group
        return None

    def _visible_asset_ids(self):
        return [
            entry["asset"].id for entry in self._rows
            if entry.get("kind") == "asset"
        ]

    def _prune_filters(self):
        group_ids = {group.id for group in self._groups} | {None}
        self._state.group_ids &= group_ids
        if self._state.tag is not None:
            known = {tag.casefold() for tag in assetview.all_tags(self._assets)}
            if self._state.tag.casefold() not in known:
                self._state.tag = None
        sorts = {key for key, _label in assetview.SORTS}
        if self._state.sort not in sorts:
            self._state.sort = assetview.SORT_MANUAL

    def _sync_controls(self):
        for key, var in self._source_vars.items():
            var.set(key in self._state.sources)
        self._favorites_var.set(self._state.favorites_only)
        self.search_var.set(self._state.query)
        self._update_filter_summary()

    def _update_filter_summary(self):
        parts = []
        if self._state.query.strip():
            parts.append(f"search: {self._state.query.strip()}")
        if self._state.sources:
            labels = dict(assetview.SOURCE_LABELS)
            parts.append("source: " + ", ".join(
                labels.get(key, key) for key in sorted(self._state.sources)
            ))
        if self._state.group_ids:
            names = []
            for group_id in self._state.group_ids:
                if group_id is None:
                    names.append("Ungrouped")
                else:
                    group = self._group_by_id(group_id)
                    if group is not None:
                        names.append(group.name)
            if names:
                parts.append("group: " + ", ".join(names))
        if self._state.tag:
            parts.append(f"tag: {self._state.tag}")
        if self._state.favorites_only:
            parts.append("favorites")
        self.filter_var.set(" \u00b7 ".join(parts))

    def _on_search_key(self, _event):
        self._cancel_search_job()
        self._search_job = self.after(120, self._apply_query)

    def _cancel_search_job(self):
        if self._search_job is not None:
            try:
                self.after_cancel(self._search_job)
            except tk.TclError:
                pass
            self._search_job = None

    def _apply_query(self):
        self._search_job = None
        self._state.query = self.search_var.get()
        self._rebuild()

    def _clear_search(self, _event=None):
        self._cancel_search_job()
        self.search_var.set("")
        self._state.query = ""
        self._rebuild()
        return "break"

    def _focus_search(self, _event=None):
        self.search_entry.focus_set()
        self.search_entry.select_range(0, tk.END)
        return "break"

    def _on_filters_changed(self):
        self._state.sources = {
            key for key, var in self._source_vars.items() if var.get()
        }
        self._update_filter_summary()
        self._rebuild()

    def _on_favorites_changed(self):
        self._state.favorites_only = bool(self._favorites_var.get())
        self._update_filter_summary()
        self._rebuild()

    def _toggle_group_filter(self, group_id, selected):
        if selected:
            self._state.group_ids.add(group_id)
        else:
            self._state.group_ids.discard(group_id)
        self._update_filter_summary()
        self._rebuild()

    def _set_tag(self, tag):
        self._state.tag = tag or None
        self._update_filter_summary()
        self._rebuild()

    def _set_sort(self, sort):
        self._state.sort = sort
        self._rebuild()

    def _clear_filters(self, _event=None):
        self._cancel_search_job()
        self._state.clear_filters()
        self._sync_controls()
        self._rebuild()
        return "break"

    def _post_filter_menu(self, _event=None):
        menu = PopupMenu(self)
        sources = PopupMenu(menu)
        for key, label in assetview.SOURCE_LABELS:
            sources.add_checkbutton(
                label, self._source_vars[key],
                command=self._on_filters_changed,
            )
        menu.add_cascade("Sources", sources)

        groups = PopupMenu(menu)
        for group in self._groups:
            var = tk.BooleanVar(value=group.id in self._state.group_ids)
            groups.add_checkbutton(
                group.name, var,
                command=lambda gid=group.id, v=var: self._toggle_group_filter(
                    gid, v.get()
                ),
            )
        if self._groups:
            groups.add_separator()
        ungrouped = tk.BooleanVar(value=None in self._state.group_ids)
        groups.add_checkbutton(
            "Ungrouped", ungrouped,
            command=lambda: self._toggle_group_filter(None, ungrouped.get()),
        )
        menu.add_cascade("Groups", groups)

        tags = PopupMenu(menu)
        tag_var = tk.StringVar(value=self._state.tag or "")
        tags.add_radio("All tags", tag_var, "",
                       command=lambda: self._set_tag(None))
        for tag in assetview.all_tags(self._assets):
            tags.add_radio(
                tag, tag_var, tag, command=lambda t=tag: self._set_tag(t),
            )
        menu.add_cascade("Tags", tags)

        menu.add_separator()
        menu.add_checkbutton(
            "Favorites only", self._favorites_var,
            command=self._on_favorites_changed,
        )
        menu.add_command(
            "Clear filters", command=self._clear_filters,
            enabled=self._state.has_filters,
        )
        try:
            x = self.filter_btn.winfo_rootx()
            y = self.filter_btn.winfo_rooty() + self.filter_btn.winfo_height()
        except tk.TclError:
            return
        menu.post(x, y)

    def _post_sort_menu(self, _event=None):
        menu = PopupMenu(self)
        var = tk.StringVar(value=self._state.sort)
        for key, label in assetview.SORTS:
            menu.add_radio(
                label, var, key, command=lambda k=key: self._set_sort(k),
            )
        try:
            x = self.sort_btn.winfo_rootx()
            y = self.sort_btn.winfo_rooty() + self.sort_btn.winfo_height()
        except tk.TclError:
            return
        menu.post(x, y)

    # ------------------------------------------------------------------ rows
    def _rebuild(self):
        # A rebuild tears the rows down, so any in-flight drag loses its origin
        # row; drop the press/ghost state rather than leave it dangling.
        self._press = None
        self._group_press = None
        self._drag_kind = None
        self._drop_target = None
        self._end_ghost()
        self._hide_line()
        for child in self.inner.winfo_children():
            child.destroy()
        self._rows = []
        bg = theme.color("listbox_bg")
        fg = theme.color("text")
        muted = theme.color("muted")
        header_bg = theme.color("button_bg")
        rows = assetview.build_rows(self._assets, self._groups, self._state)
        if not rows:
            message = (
                "No assets yet" if not self._assets else "No assets match"
            )
            empty = tk.Label(
                self.inner, text=message, bg=bg, fg=muted, anchor="w",
            )
            empty.pack(fill=tk.X, padx=6, pady=6)
            self._rows.append({"kind": "empty", "frame": empty})
        for row in rows:
            if row["kind"] == "group":
                self._build_group_row(row, header_bg, fg, muted)
            else:
                self._build_asset_row(row, bg, fg, muted)
        self._apply_selection()
        self._update_count()
        self._update_filter_summary()

    def _build_group_row(self, row, bg, fg, muted):
        group = row["group"]
        frame = tk.Frame(self.inner, bg=bg)
        frame.pack(fill=tk.X, pady=(2, 0))
        arrow = tk.Label(
            frame, text="\u25b8" if row["collapsed"] else "\u25be",
            bg=bg, fg=muted, width=2,
        )
        arrow.pack(side=tk.LEFT)
        name = tk.Label(
            frame, text=group.name if group is not None else "Ungrouped",
            bg=bg, fg=fg, anchor="w",
        )
        name.pack(side=tk.LEFT, fill=tk.X, expand=True)
        count = tk.Label(frame, text=str(row["count"]), bg=bg, fg=muted)
        count.pack(side=tk.RIGHT, padx=(0, 4))
        entry = {
            "kind": "group", "group": group, "frame": frame,
            "arrow": arrow, "name": name, "count": count,
        }
        self._rows.append(entry)
        for widget in (frame, arrow, name, count):
            widget.bind(
                "<Button-1>", lambda e, g=group: self._on_group_press(e, g)
            )
            widget.bind("<B1-Motion>", self._on_group_motion)
            widget.bind("<ButtonRelease-1>", self._on_group_release)
            widget.bind(
                "<Button-3>", lambda e, g=group: self._context_group(e, g)
            )
        self._bind_ctrl_wheel(frame)

    def _build_asset_row(self, row, bg, fg, muted):
        asset = row["asset"]
        depth = row.get("depth", 0)
        frame = tk.Frame(self.inner, bg=bg)
        frame.pack(fill=tk.X, pady=1, padx=(GROUP_INDENT if depth else 0, 0))
        photo = self._thumb_photo(asset)
        thumb = tk.Label(frame, image=photo, bg=bg, bd=0)
        thumb.pack(side=tk.LEFT, padx=ROW_PAD, pady=ROW_PAD)
        star = tk.Label(
            frame, text="\u2605" if asset.favorite else "\u2606",
            bg=bg, fg=theme.color("accent") if asset.favorite else muted,
            width=2, cursor="hand2",
        )
        star.pack(side=tk.RIGHT, padx=(0, ROW_PAD))
        text_col = tk.Frame(frame, bg=bg)
        text_col.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, ROW_PAD))
        name_label = tk.Label(
            text_col, text=asset.name, anchor="w", bg=bg, fg=fg,
        )
        name_label.pack(fill=tk.X)
        info_label = tk.Label(
            text_col, text=self._info(asset), anchor="w", bg=bg, fg=muted,
        )
        info_label.pack(fill=tk.X)
        entry = {
            "kind": "asset", "asset": asset, "frame": frame, "thumb": thumb,
            "text": text_col, "name": name_label, "info": info_label,
            "star": star,
        }
        self._rows.append(entry)
        for widget in (frame, thumb, text_col, name_label, info_label):
            widget.bind(
                "<Button-1>", lambda e, i=asset.id: self._on_press(e, i)
            )
            widget.bind("<B1-Motion>", self._on_motion)
            widget.bind("<ButtonRelease-1>", self._on_release)
            widget.bind(
                "<Double-Button-1>", lambda _e, i=asset.id: self._activate(i)
            )
            widget.bind(
                "<Button-3>", lambda e, i=asset.id: self._context(e, i)
            )
        star.bind(
            "<Button-1>", lambda e, i=asset.id: self._toggle_favorite_one(e, i)
        )
        self._bind_ctrl_wheel(frame)

    def _thumb_photo(self, asset):
        photo = self._photos.get(asset.id)
        if photo is None:
            photo = ImageTk.PhotoImage(self._thumb(asset.image))
            self._photos[asset.id] = photo
        return photo

    def _info(self, asset):
        """The secondary line under an asset's name: size, source and tags."""
        width, height = asset.image.size
        info = (
            f"{width} \u00d7 {height}  \u00b7  "
            f"{assetview.asset_source(asset)}"
        )
        if asset.tags:
            info += "  \u00b7  " + ", ".join(asset.tags)
        return info

    def _thumb(self, image):
        """An ``RGB`` thumbnail of ``image`` over a checkerboard, fitted to a box."""
        source = image.convert("RGBA")
        source.thumbnail((THUMB, THUMB), Image.LANCZOS)
        board = checkerboard(source.size)
        board.alpha_composite(source)
        return board.convert("RGB")

    def _update_count(self):
        total = len(self._assets)
        shown = len(self._visible_asset_ids())
        if shown != total:
            text = f"{shown} of {total} assets"
        else:
            text = "1 asset" if total == 1 else f"{total} assets"
        self.count_var.set(text)

    def _apply_selection(self):
        sel_bg = theme.color("sel_bg")
        sel_fg = theme.color("sel_fg")
        bg = theme.color("listbox_bg")
        fg = theme.color("text")
        muted = theme.color("muted")
        accent = theme.color("accent")
        for entry in self._rows:
            if entry.get("kind") != "asset":
                continue
            asset = entry["asset"]
            selected = asset.id in self._selected_ids
            row_bg = sel_bg if selected else bg
            name_fg = sel_fg if selected else fg
            info_fg = sel_fg if selected else muted
            entry["frame"].config(bg=row_bg)
            entry["thumb"].config(bg=row_bg)
            entry["text"].config(bg=row_bg)
            entry["name"].config(bg=row_bg, fg=name_fg)
            entry["info"].config(bg=row_bg, fg=info_fg)
            entry["star"].config(
                bg=row_bg,
                fg=(sel_fg if selected else accent) if asset.favorite else (
                    sel_fg if selected else muted
                ),
            )

    def _scroll_to_asset(self, asset_id):
        for entry in self._rows:
            if entry.get("kind") != "asset" or entry["asset"].id != asset_id:
                continue
            try:
                self.update_idletasks()
                total = max(self.inner.winfo_height(), 1)
                y = entry["frame"].winfo_y()
                height = max(self.canvas.winfo_height(), 1)
                fraction = (y - height / 2) / total
                self.canvas.yview_moveto(max(0.0, min(1.0, fraction)))
            except tk.TclError:
                pass
            return

    # -------------------------------------------------------------- selection
    def _select(self, asset_id, mode="replace"):
        visible = self._visible_asset_ids()
        if asset_id not in visible:
            return
        before = self.selected_asset_id()
        if mode == "toggle":
            if asset_id in self._selected_ids:
                self._selected_ids.discard(asset_id)
            else:
                self._selected_ids.add(asset_id)
            self._primary_id = asset_id
        elif mode == "range" and self._anchor_id in visible:
            start = visible.index(self._anchor_id)
            end = visible.index(asset_id)
            low, high = sorted((start, end))
            self._selected_ids = set(visible[low:high + 1])
            self._primary_id = asset_id
        else:
            self._selected_ids = {asset_id}
            self._primary_id = asset_id
            self._anchor_id = asset_id
        if mode != "range":
            self._anchor_id = asset_id
        if not self._selected_ids:
            self._primary_id = None
        self._apply_selection()
        primary = self.selected_asset_id()
        if primary != before and primary is not None and self._on_select is not None:
            self._on_select(primary)

    def _select_all(self, _event=None):
        ids = self._visible_asset_ids()
        if not ids:
            return "break"
        self._selected_ids = set(ids)
        if self._primary_id not in self._selected_ids:
            self._primary_id = ids[0]
        self._anchor_id = ids[0]
        self._apply_selection()
        return "break"

    def _activate(self, asset_id):
        self._select(asset_id, "replace")
        # Forget the press that started the double click so it cannot drag.
        self._press = None
        self._drag_kind = None
        if self._on_preview is not None:
            self._on_preview(asset_id)

    # ------------------------------------------------------------------- drag
    def _on_press(self, event, asset_id):
        """Select the row and remember where the press landed, for a drag."""
        self.canvas.focus_set()
        mode = "replace"
        if event.state & 0x0004:
            mode = "toggle"
        elif event.state & 0x0001:
            mode = "range"
        # Pressing an already-selected row keeps the whole selection so a drag
        # can carry it; a click without a drag collapses to that one row.
        keep_selection = mode == "replace" and asset_id in self._selected_ids
        if not keep_selection:
            self._select(asset_id, mode)
        self._press = {
            "id": asset_id, "x_root": event.x_root, "y_root": event.y_root,
            "ids": list(self._selected_ids), "collapse": keep_selection,
        }
        self._drag_kind = None

    def _on_delete_key(self, _event):
        """Delete the selected assets, keeping the global Delete off a layer."""
        self._delete_selected()
        return "break"

    def _on_rename_key(self, _event):
        asset_id = self.selected_asset_id()
        if asset_id is not None:
            self._rename(asset_id)
        return "break"

    def _on_motion(self, event):
        """Start a ghost (drag-out) or an insertion line (reorder), then move it."""
        press = self._press
        if press is None:
            return
        if self._drag_kind is None:
            dx = event.x_root - press["x_root"]
            dy = event.y_root - press["y_root"]
            if dx * dx + dy * dy < DRAG_SLOP * DRAG_SLOP:
                return
            if not self._inside_panel(event.x_root, event.y_root):
                if not self._begin_ghost(press["id"], event.x_root, event.y_root):
                    self._press = None
                    return
                self._drag_kind = "out"
            elif self._state.sort == assetview.SORT_MANUAL:
                self._drag_kind = "asset"
                self._begin_ghost(
                    press["id"], event.x_root, event.y_root,
                    count=len(press["ids"]),
                )
            else:
                # Sorted views have no manual position to drop into.
                self._drag_kind = "none"
        if self._drag_kind == "out":
            self._move_ghost(event.x_root, event.y_root)
            if self._on_drag is not None:
                self._on_drag(press["id"], event.x_root, event.y_root)
        elif self._drag_kind == "asset":
            self._move_ghost(event.x_root, event.y_root)
            self._update_asset_drop(event.y_root)

    def _on_release(self, event):
        """Finish an internal reorder or a drag-out at the pointer."""
        press = self._press
        kind = self._drag_kind
        target = self._drop_target
        self._press = None
        self._drag_kind = None
        self._drop_target = None
        self._hide_line()
        if press is None:
            return
        if kind == "out":
            self._end_ghost()
            if self._on_drop is not None:
                self._on_drop(press["id"], event.x_root, event.y_root)
        elif kind == "asset":
            self._end_ghost()
            if target is not None and self._on_reorder is not None:
                group_id, before_id = target
                ids = [
                    asset_id for asset_id in self._visible_asset_ids()
                    if asset_id in press["ids"]
                ] or [press["id"]]
                self._on_reorder(ids, group_id, before_id)
        elif kind is None and press.get("collapse"):
            self._select(press["id"], "replace")

    def _update_asset_drop(self, y_root):
        target, line_y = self._asset_drop_target(y_root)
        self._drop_target = target
        if target is None:
            self._hide_line()
            return
        self._show_line(line_y)

    def _asset_drop_target(self, y_root):
        entries = [entry for entry in self._rows if entry.get("frame") is not None
                   and entry.get("kind") != "empty"]
        last = None
        for index, entry in enumerate(entries):
            frame = entry["frame"]
            try:
                top = frame.winfo_rooty()
                height = max(frame.winfo_height(), 1)
                bottom = top + height
                local_top = frame.winfo_y()
                local_bottom = local_top + height
            except tk.TclError:
                continue
            last = (entry, local_bottom)
            if not (top <= y_root <= bottom):
                continue
            upper = y_root < top + height / 2
            if entry["kind"] == "group":
                group = entry["group"]
                return (group.id if group is not None else None, None), local_top
            asset = entry["asset"]
            if upper:
                return (asset.group, asset.id), local_top
            before = None
            for following in entries[index + 1:]:
                if following["kind"] == "asset":
                    before = following["asset"].id
                    break
                if following["kind"] == "group":
                    break
            return (asset.group, before), local_bottom
        if last is not None:
            entry, local_bottom = last
            if entry["kind"] == "asset":
                return (entry["asset"].group, None), local_bottom
        return (None, None), 0

    def _begin_ghost(self, asset_id, x_root, y_root, count=1):
        """Show a washed-out snapshot of the asset that follows the cursor.

        Tkinter cannot give an individual widget an alpha, so (as the sidebar's
        panel ghost does) the thumbnail is blended toward the panel background
        as a stand-in for opacity. It lives in a borderless top-level window
        rather than a ``place``d label, so the OS moves it atomically and leaves
        no smears on the widgets beneath. When ``count`` is greater than one, a
        small badge shows how many assets are being carried. Returns ``True``
        when it is shown.
        """
        self._end_ghost()
        asset = self._asset_by_id(asset_id)
        if asset is None:
            return False
        badge = None
        try:
            source = asset.image.convert("RGBA")
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
            if count > 1:
                badge = tk.Label(
                    window, text=str(count), bd=0, padx=3,
                    bg=theme.color("accent"), fg=theme.color("sel_fg"),
                )
                badge.place(relx=1.0, rely=1.0, anchor="se", x=2, y=2)
        except tk.TclError:
            return False
        self._ghost = {
            "widget": window, "photo": photo, "badge": badge,
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

    def _inside_panel(self, x_root, y_root):
        try:
            left = self.winfo_rootx()
            top = self.winfo_rooty()
            right = left + self.winfo_width()
            bottom = top + self.winfo_height()
        except tk.TclError:
            return False
        return left <= x_root <= right and top <= y_root <= bottom

    def _show_line(self, y):
        self._hide_line()
        try:
            width = max(self.canvas.winfo_width(), 1)
            self._line = self.canvas.create_line(
                0, y, width, y, fill=theme.color("accent"), width=2,
            )
        except tk.TclError:
            self._line = None

    def _hide_line(self):
        if self._line is not None:
            try:
                self.canvas.delete(self._line)
            except tk.TclError:
                pass
            self._line = None

    # ---------------------------------------------------------- group dragging
    def _on_group_press(self, event, group):
        self.canvas.focus_set()
        self._group_press = {
            "group": group, "x_root": event.x_root, "y_root": event.y_root,
        }
        self._drag_kind = None

    def _on_group_motion(self, event):
        press = self._group_press
        if press is None:
            return
        if self._drag_kind is None:
            dx = event.x_root - press["x_root"]
            dy = event.y_root - press["y_root"]
            if dx * dx + dy * dy < DRAG_SLOP * DRAG_SLOP:
                return
            if press["group"] is None:
                self._drag_kind = "none"
                return
            self._drag_kind = "group"
        if self._drag_kind == "group":
            before_id, line_y = self._group_drop_target(event.y_root)
            self._drop_target = before_id
            self._show_line(line_y)

    def _on_group_release(self, event):
        press = self._group_press
        kind = self._drag_kind
        target = self._drop_target
        self._group_press = None
        self._drag_kind = None
        self._drop_target = None
        self._hide_line()
        if press is None:
            return
        group = press["group"]
        if kind == "group" and self._on_move_group is not None:
            self._on_move_group(group.id, target)
            return
        if kind is not None:
            return
        # A plain click toggles the group open or closed.
        collapsed = self._state.toggle_collapsed(group)
        if group is not None and self._on_collapse_group is not None:
            self._on_collapse_group(group.id, collapsed)
        self._rebuild()

    def _group_drop_target(self, y_root):
        entries = [
            entry for entry in self._rows
            if entry.get("kind") == "group" and entry.get("group") is not None
        ]
        for index, entry in enumerate(entries):
            frame = entry["frame"]
            try:
                top = frame.winfo_rooty()
                height = max(frame.winfo_height(), 1)
                bottom = top + height
                local_top = frame.winfo_y()
                local_bottom = local_top + height
            except tk.TclError:
                continue
            if not (top <= y_root <= bottom):
                continue
            upper = y_root < top + height / 2
            if upper:
                return entry["group"].id, local_top
            if index + 1 < len(entries):
                return entries[index + 1]["group"].id, local_bottom
            return None, local_bottom
        if entries:
            last = entries[-1]["frame"]
            return None, last.winfo_y() + max(last.winfo_height(), 1)
        return None, 0

    # ---------------------------------------------------------------- actions
    def _add(self):
        if self._on_add is not None:
            self._on_add()

    def _add_layer(self):
        """Ask the coordinator to store the selected layer as a new asset."""
        if self._on_add_layer is not None:
            self._on_add_layer()

    def _preview(self, asset_id):
        if self._on_preview is not None and asset_id is not None:
            self._on_preview(asset_id)

    def _insert(self, asset_id):
        if self._on_insert is not None and asset_id is not None:
            self._on_insert(asset_id)

    def _save(self, asset_id):
        if self._on_save is not None and asset_id is not None:
            self._on_save(asset_id)

    def _rename(self, asset_id):
        if self._on_rename is not None and asset_id is not None:
            self._on_rename(asset_id)

    def _delete(self, asset_ids):
        if self._on_delete is not None and asset_ids:
            self._on_delete(list(asset_ids))

    def _delete_selected(self):
        ids = self.selected_ids()
        if ids:
            self._delete(ids)

    def _favorite(self, asset_ids):
        if self._on_favorite is not None and asset_ids:
            self._on_favorite(list(asset_ids))

    def _tags(self, asset_ids):
        if self._on_tags is not None and asset_ids:
            self._on_tags(list(asset_ids))

    def _assign(self, asset_ids, group_id):
        if self._on_assign_group is not None and asset_ids:
            self._on_assign_group(list(asset_ids), group_id)

    def _toggle_favorite_one(self, _event, asset_id):
        self._select(asset_id, "replace")
        self._favorite([asset_id])
        return "break"

    def _toggle_favorite_selected(self):
        ids = self.selected_ids()
        if ids:
            self._favorite(ids)

    def _new_group(self):
        if self._on_new_group is not None:
            self._on_new_group()

    def _rename_group(self, group_id):
        if self._on_rename_group is not None and group_id is not None:
            self._on_rename_group(group_id)

    def _delete_group(self, group_id):
        if self._on_delete_group is not None and group_id is not None:
            self._on_delete_group(group_id)

    def _context(self, event, asset_id):
        """Post a menu of actions for the clicked asset (or the selection)."""
        if asset_id not in self._selected_ids:
            self._select(asset_id, "replace")
        ids = self.selected_ids() or [asset_id]
        primary = asset_id if asset_id in ids else ids[0]
        assets = [self._asset_by_id(asset_id) for asset_id in ids]
        assets = [asset for asset in assets if asset is not None]
        favorite = not all(asset.favorite for asset in assets)
        menu = PopupMenu(self)
        menu.add_command("Preview", command=lambda: self._preview(primary))
        menu.add_command("Insert as Layer", command=lambda: self._insert(primary))
        menu.add_command("Save As\u2026", command=lambda: self._save(primary))
        menu.add_command(
            "Rename\u2026", command=lambda: self._rename(primary),
            enabled=len(ids) == 1,
        )
        menu.add_separator()
        menu.add_command(
            "Add to Favorites" if favorite else "Remove from Favorites",
            command=lambda: self._favorite(ids),
        )
        menu.add_command("Edit Tags\u2026", command=lambda: self._tags(ids))
        move = PopupMenu(menu)
        move.add_command(
            "Ungrouped", command=lambda: self._assign(ids, None),
        )
        for group in self._groups:
            move.add_command(
                group.name,
                command=lambda gid=group.id: self._assign(ids, gid),
            )
        move.add_separator()
        move.add_command("New Group\u2026", command=self._new_group)
        menu.add_cascade("Move to Group", move)
        menu.add_separator()
        menu.add_command("Delete", command=lambda: self._delete(ids))
        menu.post(event.x_root, event.y_root)
        return "break"

    def _context_group(self, event, group):
        """Post a menu for a group header (or the ungrouped pseudo-group)."""
        menu = PopupMenu(self)
        menu.add_command("New Group\u2026", command=self._new_group)
        menu.add_command(
            "Rename Group\u2026", command=lambda: self._rename_group(group.id),
            enabled=group is not None,
        )
        menu.add_command(
            "Delete Group", command=lambda: self._delete_group(group.id),
            enabled=group is not None,
        )
        menu.add_separator()
        collapsed = self._state.is_collapsed(group)
        menu.add_command(
            "Expand" if collapsed else "Collapse",
            command=lambda: self._toggle_group(group),
        )
        menu.post(event.x_root, event.y_root)
        return "break"

    def _toggle_group(self, group):
        collapsed = self._state.toggle_collapsed(group)
        if group is not None and self._on_collapse_group is not None:
            self._on_collapse_group(group.id, collapsed)
        self._rebuild()

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
        # Thumbnails blend an alpha image onto the panel background, so they must
        # be rendered again after a theme switch.
        self._photos = {}
        self._rebuild()
