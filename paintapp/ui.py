"""Menu bar and status bar construction."""

import time
import tkinter as tk
from tkinter import font as tkfont, ttk

from . import filters
from . import theme
from .constants import PANELS

# Grid cell presets offered by View > Grid Size, as ``(label, document px)``.
GRID_SIZE_PRESETS = (
    ("8 px", 8), ("16 px", 16), ("32 px", 32), ("64 px", 64), ("128 px", 128),
)

# Opacity presets offered by View > Off-Canvas Opacity, as ``(label, fraction)``.
# Kept in step with the Settings dialog's choices (which include the 35% default).
OFFCANVAS_OPACITY_PRESETS = (
    ("25%", 0.25), ("35%", 0.35), ("50%", 0.5), ("75%", 0.75), ("100%", 1.0),
)

# The status bar's result message is shortened with an ellipsis once it grows
# past this many pixels (and past half the bar's width), so a long message never
# stretches across the entire bottom bar and hides the readouts beside it. The
# full text still lands in the Logs panel.
RESULT_MAX_WIDTH = 480
RESULT_MIN_WIDTH = 140


class _ProgressMeter:
    """One determinate progress bar with a percent label and elapsed timer.

    The status bar holds two of these: one at its far-left edge for a
    project open/save/export and one next to the status label for the running
    job (a generation, an upscale, a background removal or a model preload).
    Both use the same height, matching the surrounding
    status labels (and the ttk.Progressbar the job bar replaced). Each meter owns
    a frame, its widgets, and its own one-second timer.
    """

    BAR_WIDTH = 120
    BAR_HEIGHT = 22

    def __init__(self, parent, show_when_idle=False, bar_height=None):
        self._show_when_idle = show_when_idle
        self._bar_w = self.BAR_WIDTH
        self._bar_h = bar_height or self.BAR_HEIGHT
        # A frame keeps the meter's widgets together.
        self.frame = tk.Frame(parent)
        # The bar itself is drawn from two frames - a themed trough with an
        # accent fill whose width follows the fraction - so its height can be
        # set explicitly instead of following the ttk theme's fixed thickness.
        self.bar = tk.Frame(
            self.frame, width=self._bar_w, height=self._bar_h,
            highlightthickness=1,
        )
        self.bar.pack_propagate(False)
        theme.register(
            self.bar, bg="trough",
            highlightbackground="border", highlightcolor="border",
        )
        self._fill = tk.Frame(self.bar)
        theme.register(self._fill, bg="accent")
        self._inner_w = self._bar_w - 2
        self._inner_h = self._bar_h - 2
        self._frac = 0.0
        # The text labels are given no padding or border so their height stays
        # close to the bar.
        self.label = tk.Label(
            self.frame, text="--", width=4, anchor="w",
            borderwidth=0, pady=0,
        )
        # Elapsed time for the running job, ticked once a second so a stalled
        # load is still visibly progressing rather than frozen at one percent.
        self.time_var = tk.StringVar(value="")
        self.time_label = tk.Label(
            self.frame, textvariable=self.time_var, width=7, anchor="w",
            borderwidth=0, pady=0,
        )
        self._started_at = None
        self._tick = None
        self._packed = False
        if show_when_idle:
            self._pack()

    def _draw(self):
        width = int(round(self._inner_w * self._frac))
        self._fill.place(x=0, y=0, width=width, height=self._inner_h)

    def _pack(self):
        if not self._packed:
            self.frame.pack(side=tk.TOP, fill=tk.X)
            self.bar.pack(side=tk.LEFT, padx=4)
            self.label.pack(side=tk.LEFT)
            self.time_label.pack(side=tk.LEFT)
            self._draw()
            self._packed = True

    def _unpack(self):
        if self._packed:
            self.frame.pack_forget()
            self._packed = False

    def start(self):
        self._pack()
        self._frac = 0.0
        self._draw()
        self.label.config(text="0%")
        self._start_timer()

    def set(self, frac):
        self._pack()
        self._frac = max(0.0, min(1.0, frac))
        self._draw()
        self.label.config(text=f"{self._frac * 100:.0f}%")
        if self._started_at is None:
            self._start_timer()

    def reset(self):
        self._frac = 0.0
        self._draw()
        self.label.config(text="--")
        self._stop_timer()
        if not self._show_when_idle:
            self._unpack()

    def _start_timer(self):
        self._stop_timer()
        self._started_at = time.monotonic()
        self.time_var.set("0s")
        self._tick_once()

    def _stop_timer(self):
        if self._tick is not None:
            try:
                self.time_label.after_cancel(self._tick)
            except Exception:
                pass
            self._tick = None
        self._started_at = None
        self.time_var.set("")

    def _tick_once(self):
        if self._started_at is None:
            return
        elapsed = int(time.monotonic() - self._started_at)
        minutes, seconds = divmod(elapsed, 60)
        if minutes:
            self.time_var.set(f"{minutes}:{seconds:02d}")
        else:
            self.time_var.set(f"{seconds}s")
        self._tick = self.time_label.after(1000, self._tick_once)


class StatusBar(tk.Frame):
    """The bottom bar: result message, canvas size, status, and progress."""

    def __init__(self, parent, canvas_size_text):
        super().__init__(parent)
        # A visible resize grip in the window's bottom-right corner. The stock
        # window border is only a few pixels wide there, which makes the corner
        # easy to miss; this gives a clear, grabbable target.
        self.sizegrip = ttk.Sizegrip(self)
        self.sizegrip.pack(side=tk.RIGHT, anchor="se")

        # The save/load meter, at the far-left edge: a project open/save/export.
        # It shows "--" when it has nothing to report.
        load_area = tk.Frame(self)
        load_area.pack(side=tk.LEFT)
        self.load_meter = _ProgressMeter(load_area, show_when_idle=True)

        self.result_var = tk.StringVar(value="")
        self.result = tk.Label(
            self, textvariable=self.result_var, anchor="w",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2,
        )
        self.result.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # The untruncated message and the bar width it was fitted to.
        self._result_text = ""
        self._result_width = 0
        self._result_font = tkfont.nametofont("TkDefaultFont")
        self.bind("<Configure>", self._fit_result)

        self.cursor_var = tk.StringVar(value="--, --")
        self.cursor_label = tk.Label(
            self, textvariable=self.cursor_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2, width=12,
        )
        self.cursor_label.pack(side=tk.LEFT)

        self.canvas_size_var = tk.StringVar(value=canvas_size_text)
        self.canvas_size_label = tk.Label(
            self, textvariable=self.canvas_size_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2,
        )
        self.canvas_size_label.pack(side=tk.LEFT)

        self.zoom_var = tk.StringVar(value="100%")
        self.zoom_label = tk.Label(
            self, textvariable=self.zoom_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2, width=9,
        )
        self.zoom_label.pack(side=tk.LEFT)

        self.layer_var = tk.StringVar(value="")
        self.layer_label = tk.Label(
            self, textvariable=self.layer_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2, width=18,
        )
        self.layer_label.pack(side=tk.LEFT)

        self.tool_var = tk.StringVar(value="")
        self.tool_label = tk.Label(
            self, textvariable=self.tool_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2, width=16,
        )
        self.tool_label.pack(side=tk.LEFT)

        self.status = tk.Label(
            self, text="ready", anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2,
        )
        self.status.pack(side=tk.LEFT)

        # The job meter, next to the status label: a generation, an upscale, a
        # background removal or a model preload.
        job_area = tk.Frame(self)
        job_area.pack(side=tk.LEFT)
        self.job_meter = _ProgressMeter(job_area, show_when_idle=True)

        self.gpu_var = tk.StringVar(value="GPU --")
        self.gpu_label = tk.Label(
            self, textvariable=self.gpu_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2,
        )
        self.gpu_label.pack(side=tk.RIGHT)

        self.cpu_var = tk.StringVar(value="CPU --")
        self.cpu_label = tk.Label(
            self, textvariable=self.cpu_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2,
        )
        self.cpu_label.pack(side=tk.RIGHT)

        self.ram_var = tk.StringVar(value="RAM --")
        self.ram_label = tk.Label(
            self, textvariable=self.ram_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2,
        )
        self.ram_label.pack(side=tk.RIGHT)

        self.modified_var = tk.StringVar(value="")
        self.modified_label = tk.Label(
            self, textvariable=self.modified_var, anchor="e",
            relief=tk.SUNKEN, borderwidth=1, padx=6, pady=2, width=10,
        )
        self.modified_label.pack(side=tk.RIGHT)

        # CPU/RAM/GPU readings can be hidden from the View menu. They are kept
        # in one ordered map so they can always be laid out right-to-left as
        # GPU, CPU, RAM regardless of which ones are currently visible.
        self._stat_labels = {
            "cpu": self.cpu_label,
            "ram": self.ram_label,
            "gpu": self.gpu_label,
        }
        self._stat_visible = {"cpu": True, "ram": True, "gpu": True}

    def set_result(self, text, kind="info"):
        """Show a result message in the status bar, tinted by its ``kind``.

        ``kind`` is a :data:`~paintapp.theme.LEVELS` key (``info``, ``success``,
        ``warn`` or ``error``), so a warning reads amber and an error red. An
        empty message restores the plain, untinted chrome.
        """
        self._result_text = text
        self._fit_result()
        if not text:
            theme.register(self.result, bg="bg", fg="text")
            return
        fg, bg = theme.LEVELS.get(kind, theme.LEVELS["info"])
        theme.register(self.result, bg=bg, fg=fg)

    def _fit_result(self, event=None):
        """Show the result message, shortened to the bar's width budget."""
        width = event.width if event is not None else self.winfo_width()
        if event is not None and width == self._result_width:
            return
        self._result_width = width
        text = self._result_text
        if not text:
            self.result_var.set("")
            return
        # Never let the message own more than half the bar, and keep enough
        # room for a few characters on a very narrow window.
        limit = max(RESULT_MIN_WIDTH, min(RESULT_MAX_WIDTH, width // 2))
        if self._result_font.measure(text) <= limit:
            self.result_var.set(text)
            return
        ellipsis = "\u2026"
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._result_font.measure(text[:mid] + ellipsis) <= limit:
                lo = mid
            else:
                hi = mid - 1
        self.result_var.set(text[:lo] + ellipsis if lo else ellipsis)

    def set_status(self, text):
        self.status.config(text=text)

    def set_cursor(self, x, y):
        if x is None or y is None:
            self.cursor_var.set("--, --")
        else:
            self.cursor_var.set(f"{x}, {y}")

    def set_canvas_size(self, text):
        self.canvas_size_var.set(text)

    def set_zoom(self, view):
        text = f"zoom {view.zoom * 100:.0f}%"
        # Surface a non-default view transform, so a rotated/flipped canvas is
        # never mistaken for a permanent edit.
        angle = getattr(view, "view_angle", 0.0) * 57.29577951308232
        flip_h = getattr(view, "view_flip", False)
        flip_v = getattr(view, "view_flip_v", False)
        if flip_h or flip_v or abs(angle) > 1e-6:
            text += f"  rot {angle % 360:.0f}\u00b0"
            if flip_h and flip_v:
                text += " flip hv"
            elif flip_h:
                text += " flip h"
            elif flip_v:
                text += " flip v"
        self.zoom_var.set(text)

    def set_layer(self, name, masked=False):
        self.layer_var.set(f"{name} · mask" if masked else name)

    def set_tool(self, name, brush=None):
        self.tool_var.set(name if brush is None else f"{name} {brush}px")

    def set_modified(self, flag):
        self.modified_var.set("unsaved *" if flag else "saved")

    # The far-left save/load meter: a project open/save/export.
    def set_load_progress(self, frac):
        self.load_meter.set(frac)

    def reset_load_progress(self):
        self.load_meter.reset()

    # The job meter next to the status label: a generation, upscale, removal or
    # model preload.
    def set_job_progress(self, frac):
        self.job_meter.set(frac)

    def reset_job_progress(self):
        self.job_meter.reset()

    def set_system_stats(self, cpu, gpu, ram=None):
        self.cpu_var.set("CPU --" if cpu is None else f"CPU {cpu:.0f}%")
        self.gpu_var.set(f"GPU {_memory_text(gpu)}")
        self.ram_var.set(f"RAM {_memory_text(ram)}")

    def set_stat_visible(self, kind, visible):
        """Show or hide one CPU/RAM/GPU reading in the status bar."""
        if kind not in self._stat_labels:
            return
        self._stat_visible[kind] = bool(visible)
        self._layout_stats()

    def _layout_stats(self):
        # Re-pack the visible readings so they always end up, from the right
        # edge inward, as GPU, CPU, RAM, then the modified indicator.
        for label in self._stat_labels.values():
            label.pack_forget()
        anchor = self.modified_label
        for kind in ("ram", "cpu", "gpu"):
            if self._stat_visible[kind]:
                label = self._stat_labels[kind]
                label.pack(side=tk.RIGHT, before=anchor)
                anchor = label


def _memory_text(pair):
    if not pair:
        return "--"
    used, total = pair
    return f"{used:.1f}/{total:.1f} GB"


# How long an open submenu lingers before closing once the pointer is no longer
# over it or its own row (ms). Mirrors the grace period native menus use.
CLOSE_DELAY = 300

# Non-zero while a menu window is being laid out. ``post`` calls ``win.update()``
# to flush the popup's position, and that can pump pending <Motion> events back
# into the menu handlers before the new window is registered - causing nested
# re-posts that leak popup windows. Handlers stand down while this is set.
_POSTING = 0


class PopupMenu:
    """A Tk-drawn dropdown menu.

    ``tk.Menu`` popups are drawn by the native Windows menu API, whose frame
    ignores every Tk color option and stays light in dark mode. This class
    renders the same menu with ordinary Tk widgets so the whole popup - frame
    included - follows the active theme.
    """

    def __init__(self, parent):
        if isinstance(parent, PopupMenu):
            parent = parent._parent
        self._parent = parent
        self._items = []
        self._window = None
        self._rows = []
        self._active_row = None
        self._child = None
        # The menu this one cascades out of; ``None`` for a top-level menu. Only
        # the top-level menu holds the pointer grab (see ``post``) so it keeps
        # receiving motion while a submenu is open and the parent rows can still
        # be hovered and selected.
        self._owner = None
        # Pending "tuck the submenus away" timer, owned by the top-level menu.
        self._close_job = None
        # Optional callback run just before posting, to refresh dynamic labels.
        self._before_post = None
        # The widget that had the keyboard focus before the top-level menu was
        # posted, restored when the menu chain closes (see ``_close_all_menus``).
        self._previous_focus = None

    # --------------------------------------------------------------- building
    def add_command(self, label, accelerator=None, command=None, enabled=None):
        """Add a command row.

        ``enabled`` may be a ``bool`` or a zero-argument callable returning one.
        A callable is re-evaluated every time the menu is posted, so a row that
        would do nothing (no selection, nothing to undo, a locked layer, ...)
        can mute itself against the current state.
        """
        item = {
            "kind": "command", "label": label,
            "accel": accelerator, "command": command, "enabled": enabled,
        }
        self._items.append(item)
        return item

    def set_before_post(self, callback):
        """Register ``callback(menu)`` to run just before the menu is posted.

        Rows are rebuilt from ``_items`` on every post, so this lets a caller
        relabel a command row (such as Expand/Collapse) to match current state.
        """
        self._before_post = callback

    def add_separator(self):
        self._items.append({"kind": "separator"})

    def add_checkbutton(self, label, variable, command=None, accelerator=None,
                        enabled=None):
        self._items.append({
            "kind": "check", "label": label, "accel": accelerator,
            "variable": variable, "command": command, "enabled": enabled,
        })

    def add_radio(self, label, variable, value, command=None, accelerator=None,
                  enabled=None):
        """Add a radio row, ticked while ``variable`` equals ``value``.

        Unlike a checkbutton row, picking a radio row does not flip
        ``variable`` itself: it sets the shared variable to ``value`` and runs
        ``command``, so a set of rows can mark the one current choice.
        """
        self._items.append({
            "kind": "radio", "label": label, "accel": accelerator,
            "variable": variable, "value": value, "command": command,
            "enabled": enabled,
        })

    def add_cascade(self, label, menu, enabled=None):
        self._items.append({
            "kind": "cascade", "label": label, "menu": menu, "enabled": enabled,
        })

    @staticmethod
    def _eval_enabled(item):
        enabled = item.get("enabled")
        if enabled is None:
            return True
        if callable(enabled):
            try:
                return bool(enabled())
            except Exception:
                # A broken predicate must not take the whole menu down.
                return True
        return bool(enabled)

    # ----------------------------------------------------------------- posting
    def post(self, x, y):
        self._close_child()
        self._destroy()
        if self._owner is None:
            # The popup takes the input focus window so <Escape> reaches it, so
            # remember what had focus before, to hand it back on close.
            try:
                self._previous_focus = self._parent.winfo_toplevel().focus_get()
            except tk.TclError:
                self._previous_focus = None
        if self._before_post is not None:
            self._before_post(self)
        # Freeze each row's enabled state for this posting, so hit-testing and
        # activation agree with what was drawn.
        for item in self._items:
            item["_enabled"] = self._eval_enabled(item)
        panel = theme.color("panel_bg")
        border = theme.color("border")
        text = theme.color("text")
        muted = theme.color("muted")

        win = tk.Toplevel(self._parent)
        win.overrideredirect(True)
        win.withdraw()
        # Own the popup to the app window (rather than making it -topmost, which
        # would keep it floating above every other application): the window
        # manager then keeps it above its owner but lets other apps cover it.
        try:
            win.transient(self._parent.winfo_toplevel())
        except tk.TclError:
            pass
        win.configure(bg=border)
        inner = tk.Frame(win, bg=panel)
        inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

        self._rows = []
        self._active_row = None
        font = tkfont.nametofont("TkDefaultFont")
        for item in self._items:
            row = self._build_row(inner, item, font, [panel, text, muted])
            self._rows.append((item, row))

        win.update_idletasks()
        width = win.winfo_reqwidth()
        height = win.winfo_reqheight()
        # Clamp to the virtual desktop, not the primary monitor: on a
        # multi-monitor setup a secondary display extends beyond (or before) the
        # primary one, so clamping to 0..screenwidth would yank the popup back
        # to the primary screen.
        vx = win.winfo_vrootx()
        vy = win.winfo_vrooty()
        vw = win.winfo_vrootwidth()
        vh = win.winfo_vrootheight()
        x = max(vx, min(x, vx + vw - width))
        y = max(vy, min(y, vy + vh - height))
        win.geometry(f"+{x}+{y}")
        win.deiconify()
        # Register the window before flushing it, so any handler that runs while
        # update() pumps events sees this menu as posted (and bails out).
        self._window = win
        _OPEN.add(self)
        # Flush the new position to the native window and lay out the rows, so
        # hit-testing works even before the next idle turn.
        global _POSTING
        _POSTING += 1
        try:
            win.update()
        finally:
            _POSTING -= 1

        if not win.winfo_exists():
            # A handler pumped during the flush tore this popup down again;
            # there is nothing left to bind or show.
            self._window = None
            _OPEN.discard(self)
            return

        win.bind("<Motion>", self._on_motion)
        win.bind("<Button-1>", self._on_click)
        win.bind("<Button-3>", self._on_right_click)
        win.bind("<Escape>", lambda _e: _close_all_menus())
        # Only the top-level menu grabs the pointer. A local grab redirects
        # events from the whole application to the grab window, so the top menu
        # keeps receiving motion over its submenus; a submenu grabbing instead
        # would freeze the parent and make its rows unselectable while open.
        if self._owner is None:
            try:
                win.grab_set()
            except tk.TclError:
                pass
        win.lift()
        # Only the top-level menu takes the keyboard focus. It is the one that
        # handles <Escape>, and keeping focus on it means opening or closing a
        # submenu never moves the application's focus window - a submenu that
        # took it and was then destroyed left the app with no focus window, so
        # later clicks on entries were dropped.
        if self._owner is None:
            win.focus_force()

    # ---------------------------------------------------------------- internals
    def _build_row(self, parent, item, font, colors):
        panel, text, muted = colors
        if item["kind"] == "separator":
            line = tk.Frame(parent, height=1, bg=theme.color("border"))
            line.pack(fill=tk.X, padx=4, pady=3)
            return line

        row = tk.Frame(parent, bg=panel)
        row.pack(fill=tk.X)
        row.columnconfigure(1, weight=1)
        enabled = item.get("_enabled", True)
        label_fg = text if enabled else muted
        if item["kind"] == "check":
            indicator = "\u2713" if item["variable"].get() else ""
        elif item["kind"] == "radio":
            indicator = "\u2713" if item["variable"].get() == item["value"] else ""
        else:
            indicator = ""
        accel = item.get("accel")
        if callable(accel):
            # A callable accelerator is re-read on every posting, so a shortcut
            # rebound in Settings shows its new keys without a restart.
            try:
                accel = accel()
            except Exception:
                accel = ""
        marker = "\u25b8" if item["kind"] == "cascade" else (accel or "")
        cells = [
            tk.Label(row, text=indicator, bg=panel, fg=label_fg, width=2,
                     anchor="center", font=font),
            tk.Label(row, text=item["label"], bg=panel, fg=label_fg, anchor="w",
                     font=font, padx=0),
            tk.Label(row, text=marker, bg=panel, fg=muted, anchor="e",
                     font=font, padx=0),
        ]
        cells[0].grid(row=0, column=0, sticky="w", padx=(6, 0), pady=4)
        cells[1].grid(row=0, column=1, sticky="we", padx=(0, 14))
        cells[2].grid(row=0, column=2, sticky="e", padx=(0, 8))
        row._cells = cells
        row._enabled = enabled
        for widget in (row, *cells):
            widget.bind("<Enter>", lambda _e, r=row: self._set_active(r))
        return row

    @staticmethod
    def _row_enabled(row):
        return getattr(row, "_enabled", True)

    def _set_active(self, row):
        if _POSTING:
            # ``post`` flushes the new popup with ``win.update()``, which pumps
            # the <Enter> events raised while its rows were being laid out.
            # Acting on them re-enters ``_open_child`` and can tear down the
            # window that is still being posted, so stand down as the <Motion>
            # and click handlers already do.
            return
        if not self._row_enabled(row):
            # A muted row never highlights or opens its submenu.
            if self._active_row is not None:
                self._set_row_state(self._active_row, False)
                self._active_row = None
            self._schedule_close(self)
            return
        if row is not self._active_row:
            self._set_row_state(self._active_row, False)
            self._set_row_state(row, True)
            self._active_row = row
        # Keep the submenu state in sync even when the highlight did not move:
        # re-hovering the same row must still schedule its submenu to close.
        item = next((it for it, r in self._rows if r is row), None)
        if item and item["kind"] == "cascade":
            self._open_child(item["menu"], row)
        else:
            # Close through a short delay rather than at once, so a diagonal
            # move from a cascade row into its submenu (which skims across the
            # rows in between) does not slam the submenu shut mid-travel.
            self._schedule_close(self)

    def _set_row_state(self, row, active):
        if row is None or not self._rows or not getattr(row, "_cells", None):
            return
        bg = theme.color("sel_bg") if active else theme.color("panel_bg")
        fg = theme.color("sel_fg") if active else theme.color("text")
        # Recolor the row frame too, so the grid padding between the cells is
        # highlighted as well; otherwise each cell is a separate blue block and
        # the accelerator/marker column looks like a detached second box.
        row.configure(bg=bg)
        for widget in row._cells:
            widget.configure(bg=bg, fg=fg)

    def _row_at(self, x_root, y_root):
        for _item, row in self._rows:
            if row is None or not row.winfo_exists():
                continue
            if getattr(row, "_cells", None) is None:
                continue
            rx, ry = row.winfo_rootx(), row.winfo_rooty()
            if rx <= x_root < rx + row.winfo_width() and \
                    ry <= y_root < ry + row.winfo_height():
                return row
        return None

    def _root(self):
        """The top-level menu of this cascade chain (the pointer-grab holder)."""
        node = self
        while node._owner is not None:
            node = node._owner
        return node

    def _hit(self, x_root, y_root):
        """Find the deepest posted menu (and row) under a screen point.

        The top-level menu holds the grab, so it receives motion over every open
        menu; walking the chain lets it route the event to whichever menu and
        row the pointer is actually over.
        """
        node = self._root()
        hit = None
        while node is not None:
            win = node._window
            if win is not None and win.winfo_exists():
                x, y = win.winfo_rootx(), win.winfo_rooty()
                if x <= x_root < x + win.winfo_width() and \
                        y <= y_root < y + win.winfo_height():
                    # Deeper menus win, since they are drawn on top.
                    hit = (node, node._row_at(x_root, y_root))
            node = node._child
        return hit

    def _cancel_close(self):
        """Cancel a pending submenu close, if any."""
        root = self._root()
        if root._close_job is not None:
            try:
                root._parent.after_cancel(root._close_job)
            except (tk.TclError, ValueError):
                pass
            root._close_job = None

    def _schedule_close(self, menu):
        """Tuck ``menu``'s submenu (and deeper ones) away after a short delay.

        Deferring the close keeps a quick diagonal move from a cascade row into
        its submenu from being cut off by the rows it skims on the way, while
        still hiding the submenu once the pointer has really left.
        """
        root = self._root()
        self._cancel_close()

        def do_close():
            root._close_job = None
            menu._close_child()
            # A submenu takes the input focus window when it opens; handing it
            # back to the still-open menu keeps <Escape> working and stops the
            # whole application from being left without a focus window (which
            # would drop later clicks on entries).
            win = root._window
            if win is not None and win.winfo_exists():
                try:
                    win.focus_force()
                except tk.TclError:
                    pass

        root._close_job = root._parent.after(CLOSE_DELAY, do_close)

    def _on_motion(self, event):
        if _POSTING:
            return
        hit = self._hit(event.x_root, event.y_root)
        if hit is None:
            # Pointer left every open menu: drop the highlight and tuck the
            # open submenus away again. A fast move can skip the rows entirely,
            # so this must not depend on landing on a specific row.
            self._clear_active()
            self._schedule_close(self._root())
            return
        menu, row = hit
        if row is None or not menu._row_enabled(row):
            # Over a menu but not a selectable row (padding, separator, or a
            # muted row that would do nothing if picked).
            self._clear_active()
            self._schedule_close(menu)
            return
        self._cancel_close()
        menu._set_active(row)

    def _clear_active(self):
        node = self._root()
        while node is not None:
            node._set_row_state(node._active_row, False)
            node._active_row = None
            node = node._child

    def _on_click(self, event):
        if _POSTING:
            return
        hit = self._hit(event.x_root, event.y_root)
        if hit is None or hit[1] is None:
            _close_all_menus()
            return
        menu, row = hit
        menu._activate(row)

    def _on_right_click(self, event):
        """A right-click outside the menu dismisses the whole chain.

        The top-level menu holds the pointer grab, so a right-click anywhere
        else in the application - such as on the canvas that posted a context
        menu - arrives here instead of reaching that widget. Without this the
        click would be swallowed and the menu would stay open; dismissing on
        right-click lets a second right-click close the menu (matching the way
        a left-click already does). A right-click on a row is ignored, so it
        never triggers a command the way a left-click would.
        """
        if _POSTING:
            return
        if self._hit(event.x_root, event.y_root) is None:
            _close_all_menus()

    def _activate(self, row):
        if not self._row_enabled(row):
            # Clicking a muted row is a no-op; leave the menu open.
            return
        item = next((it for it, r in self._rows if r is row), None)
        if item is None:
            _close_all_menus()
            return
        if item["kind"] == "cascade":
            self._open_child(item["menu"], row)
        elif item["kind"] == "check":
            item["variable"].set(not item["variable"].get())
            _close_all_menus()
            if item["command"] is not None:
                item["command"]()
        elif item["kind"] == "radio":
            item["variable"].set(item["value"])
            _close_all_menus()
            if item["command"] is not None:
                item["command"]()
        elif item["kind"] == "command":
            _close_all_menus()
            if item["command"] is not None:
                item["command"]()

    def _open_child(self, menu, row):
        if self._child is menu and menu._window is not None:
            return
        self._cancel_close()
        self._close_child()
        self._child = menu
        menu._owner = self
        x = self._window.winfo_rootx() + self._window.winfo_width() - 2
        y = row.winfo_rooty() - 2
        menu.post(x, y)
        # If the submenu could not open to the right (no room before the edge
        # of the desktop) it was clamped; flip it to the left of its parent.
        win = menu._window
        if win.winfo_rootx() != x:
            left = max(win.winfo_vrootx(),
                       self._window.winfo_rootx() - win.winfo_width() + 2)
            win.geometry(f"+{left}+{y}")
            win.update_idletasks()

    def _close_child(self):
        if self._child is not None:
            self._child._close_child()
            self._child._destroy()
            self._child = None

    def _destroy(self):
        self._cancel_close()
        _OPEN.discard(self)
        win = self._window
        self._window = None
        self._rows = []
        self._active_row = None
        if win is None:
            return
        try:
            win.grab_release()
        except tk.TclError:
            pass
        if not win.winfo_exists():
            return
        # A topmost, overrideredirect toplevel that is simply destroyed leaves
        # its pixels behind - the window manager never repaints the area it
        # covered, so the submenu looks "stuck". Hide it and shift it off-screen
        # first, which forces that region to be repainted.
        try:
            win.withdraw()
            win.geometry("+20000+20000")
            win.update_idletasks()
        except tk.TclError:
            pass
        win.destroy()


def _close_all_menus():
    """Dismiss every open popup menu and return focus to the main window."""
    open_menus = list(_OPEN)
    if not open_menus:
        return
    root = next((m for m in open_menus if m._owner is None), open_menus[0])
    previous_focus = root._previous_focus
    target = root._parent.winfo_toplevel()
    for menu in open_menus:
        menu._close_child()
    for menu in open_menus:
        menu._destroy()
    # The popup owned the input focus window; destroying it leaves the whole
    # application without one, so a later click on an entry is dropped. Force
    # the focus window back to the owning toplevel, then restore the widget
    # that had focus before the menu was opened.
    try:
        target.focus_force()
    except tk.TclError:
        pass
    try:
        if previous_focus is not None and previous_focus.winfo_exists():
            previous_focus.focus_set()
    except tk.TclError:
        pass


# Every popup menu that is currently posted, so a click outside can dismiss the
# whole chain (and any submenus) at once.
_OPEN = set()


class Tooltip:
    """A small themed label shown after the pointer rests on ``widget``.

    Icon-only buttons carry their meaning in the tooltip, so this stays
    dependency-free and theme-aware: the popup is a bare ``Toplevel`` filled
    with the panel colors, like :class:`PopupMenu`. Call :meth:`set_text` to
    retarget a button whose meaning depends on the current selection.
    """

    DELAY = 500

    # A tooltip wraps once it grows past this many pixels, so a long layer or
    # group name reads on several lines instead of stretching across the
    # screen or running off its edge.
    MAX_WIDTH = 360

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self._job = None
        self._window = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def set_text(self, text):
        self.text = text
        if self._window is not None:
            self._hide()

    # ------------------------------------------------------------- internals
    def _schedule(self, _event=None):
        self._cancel()
        if not self.text:
            return
        try:
            self._job = self.widget.after(self.DELAY, self._show)
        except tk.TclError:
            self._job = None

    def _cancel(self):
        if self._job is not None:
            try:
                self.widget.after_cancel(self._job)
            except (tk.TclError, ValueError):
                pass
            self._job = None

    def _show(self):
        self._job = None
        if self._window is not None or not self.text:
            return
        try:
            if str(self.widget.cget("state")) == "disabled":
                return
        except tk.TclError:
            return
        win = tk.Toplevel(self.widget)
        win.overrideredirect(True)
        win.withdraw()
        try:
            win.transient(self.widget.winfo_toplevel())
        except tk.TclError:
            pass
        win.configure(bg=theme.color("border"))
        tk.Label(
            win, text=self.text, bg=theme.color("panel_bg"),
            fg=theme.color("text"), padx=6, pady=2, justify="left",
            wraplength=self.MAX_WIDTH,
        ).pack(padx=1, pady=1)
        win.update_idletasks()
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        # Keep the popup on the virtual desktop (see PopupMenu.post).
        vx, vy = win.winfo_vrootx(), win.winfo_vrooty()
        vw, vh = win.winfo_vrootwidth(), win.winfo_vrootheight()
        if x + win.winfo_reqwidth() > vx + vw:
            x = vx + vw - win.winfo_reqwidth()
        if y + win.winfo_reqheight() > vy + vh:
            y = self.widget.winfo_rooty() - win.winfo_reqheight() - 4
        win.geometry(f"+{x}+{y}")
        win.deiconify()
        self._window = win

    def _hide(self, _event=None):
        self._cancel()
        win = self._window
        self._window = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass


class _MenuButton(tk.Menubutton):
    """A themed top-level menu button that posts a Tk popup menu.

    The native Windows menu bar ignores background colors, so the app builds
    its own bar from these buttons instead of ``root.config(menu=...)``.
    """

    def __init__(self, parent, text):
        super().__init__(
            parent, text=text, relief=tk.FLAT, bd=0,
            highlightthickness=0, padx=8, pady=3, anchor="w",
        )
        theme.register(
            self, bg="bg", fg="text",
            activebackground="sel_bg", activeforeground="sel_fg",
        )
        self.bind("<Enter>", lambda _e: self.configure(bg=theme.color("button_active")))
        self.bind("<Leave>", lambda _e: self.configure(bg=theme.color("bg")))

    def set_menu(self, menu):
        self._menu = menu
        self.bind("<Button-1>", self._post_menu)

    def _post_menu(self, _event):
        self._menu.post(
            self.winfo_rootx(), self.winfo_rooty() + self.winfo_height() - 1)
        return "break"


def _menu_vars(app):
    """Create the Tk variables the menus share, once per application.

    The menu bar and the canvas context menu each build their own menu
    widgets, so the variables they tick live on ``app`` rather than being
    recreated per build - two copies would otherwise disagree about which
    tool or view option is current.
    """
    if not hasattr(app, "fullscreen_var"):
        app.fullscreen_var = tk.BooleanVar(value=app.is_fullscreen())
    if not hasattr(app, "offscreen_var"):
        app.offscreen_var = tk.BooleanVar(value=app.show_offscreen)
    if not hasattr(app, "exploded_var"):
        app.exploded_var = tk.BooleanVar(value=app.show_exploded)
    if not hasattr(app, "ruler_var"):
        app.ruler_var = tk.BooleanVar(value=app.show_rulers)
    if not hasattr(app, "guide_var"):
        app.guide_var = tk.BooleanVar(value=app.show_guides)
    if not hasattr(app, "shared_guides_var"):
        app.shared_guides_var = tk.BooleanVar(value=app.guides_shared)
    if not hasattr(app, "grid_var"):
        app.grid_var = tk.BooleanVar(value=app.show_grid)
    if not hasattr(app, "grid_snap_var"):
        app.grid_snap_var = tk.BooleanVar(value=app.grid_snap)
    if not hasattr(app, "dark_var"):
        app.dark_var = tk.BooleanVar(value=theme.is_dark())
    if not hasattr(app, "gallery_var"):
        app.gallery_var = tk.BooleanVar(value=False)
    if not hasattr(app, "gallery_side_var"):
        app.gallery_side_var = tk.BooleanVar(value=False)
    if not hasattr(app, "tool_var"):
        app.tool_var = tk.StringVar(value="arrow")
    for kind in ("cpu", "ram", "gpu"):
        name = f"{kind}_stat_var"
        if not hasattr(app, name):
            setattr(app, name, tk.BooleanVar(value=app.stat_visible(kind)))


def build_menus(app, parent):
    """Build one instance of every top-level menu, in menu-bar order.

    Returns ``(label, menu)`` pairs. The menu bar calls this once; the canvas
    context menu calls it again so it gets its own widgets - a posted menu
    remembers the parent it cascaded from, so the two copies must not share
    instances.
    """
    _menu_vars(app)

    # Predicates for graying out rows that would do nothing in the current
    # state. They are passed as ``enabled=`` to the menu builders below and
    # re-evaluated every time a menu is posted.
    def has_selection():
        selection = app.selection
        return selection is not None and not selection.is_empty()

    def has_canvas():
        return app.current is not None

    def layer_editable():
        layer = app.active_layer()
        return layer is not None and not layer.is_locked()

    def filter_editable():
        """True when a filter can be added: a group is selected, else the
        active layer must be editable."""
        if app._selected_group() is not None:
            return True
        return layer_editable()

    def layer_has_filters():
        layer = app.active_layer()
        return bool(layer is not None and layer.filters)

    def has_filters():
        """True when the current filter target (group or layer) has filters."""
        group = app._selected_group()
        if group is not None:
            return bool(group.filters)
        return layer_has_filters()

    def has_text():
        layer = app.active_layer()
        return bool(layer is not None and layer.texts)

    def has_paths():
        layer = app.active_layer()
        return bool(layer is not None and getattr(layer, "paths", None))

    def has_vector():
        layer = app.active_layer()
        return bool(layer is not None and (
            getattr(layer, "strokes", None) or getattr(layer, "fills", None)))

    def has_poses():
        layer = app.active_layer()
        return bool(layer is not None and getattr(layer, "poses", None))

    def has_mask():
        layer = app.active_layer()
        return bool(layer is not None and layer.mask is not None)

    def has_group():
        return app._selected_group() is not None

    def selected_layer_count():
        return len(app.layer_panel.selected_indices(len(app.layers)))

    def can_move(delta):
        blocks = app._layer_blocks()
        for slot, (_group, indices) in enumerate(blocks):
            if app.active in indices:
                return 0 <= slot + delta < len(blocks)
        return False

    def can_clip():
        if not app.layers:
            return False
        indices = app.layer_panel.selected_indices(len(app.layers)) or [app.active]
        return any(app.layers[i].clip or i > 0 for i in indices)

    def accel(shortcut_id):
        """A menu accelerator that reads the current binding when posted."""
        return lambda: app.shortcuts.accelerator(shortcut_id)

    menus = []

    file_menu = PopupMenu(parent)
    file_menu.add_command(label="New", accelerator=accel("file_new"), command=app.new_document)
    file_menu.add_command(label="Open Project...", accelerator=accel("file_open_project"), command=app.open_project)
    file_menu.add_command(label="Open Image...", accelerator=accel("file_open_image"), command=app.open_image)
    file_menu.add_command(label="Close Project", command=app.close_project)
    file_menu.add_separator()
    file_menu.add_command(label="Save Project", accelerator=accel("file_save_project"), command=app.save_project)
    file_menu.add_command(label="Save Project As...", command=app.save_project_as)
    file_menu.add_separator()
    file_menu.add_command(label="Save Image As...", accelerator=accel("file_save_image"), command=app.save_image)
    file_menu.add_command(label="Export Images...", accelerator=accel("file_export"), command=app.export_images)
    export_menu = PopupMenu(file_menu)
    export_menu.add_command(label="Vector (SVG)...", command=app.export_vector)
    file_menu.add_cascade(label="Export", menu=export_menu)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", accelerator=accel("file_exit"), command=app._on_close)
    menus.append(("File", file_menu))

    edit_menu = PopupMenu(parent)
    edit_menu.add_command(
        label="Undo", accelerator=accel("edit_undo"), command=app.undo,
        enabled=lambda: app.can_undo,
    )
    edit_menu.add_command(
        label="Redo", accelerator=accel("edit_redo"), command=app.redo,
        enabled=lambda: app.can_redo,
    )
    edit_menu.add_separator()
    edit_menu.add_command(
        label="Cut", accelerator=accel("edit_cut"), command=app.cut,
        enabled=layer_editable,
    )
    edit_menu.add_command(label="Copy", accelerator=accel("edit_copy"), command=app.copy)
    edit_menu.add_command(label="Paste", accelerator=accel("edit_paste"), command=app.paste)
    edit_menu.add_command(
        label="Copy Vector as SVG", command=app.copy_vector_svg,
        enabled=has_vector,
    )
    edit_menu.add_separator()
    edit_menu.add_command(
        label="Fill with Color", command=app.fill, enabled=layer_editable,
    )
    edit_menu.add_command(
        label="Clear Layer", accelerator=accel("edit_clear"), command=app.clear,
        enabled=layer_editable,
    )
    edit_menu.add_command(label="Pick Color...", command=app.choose_color)
    edit_menu.add_command(
        label="Edit Text...", command=app.edit_selected_text,
        enabled=lambda: app._text_target() is not None,
    )
    edit_menu.add_separator()
    transform_menu = PopupMenu(edit_menu)
    transform_menu.add_command(
        label="Flip Horizontal", command=app.flip_horizontal,
        enabled=layer_editable,
    )
    transform_menu.add_command(
        label="Flip Vertical", command=app.flip_vertical, enabled=layer_editable,
    )
    transform_menu.add_command(
        label="Rotate 90\u00b0 CW", command=app.rotate_cw, enabled=layer_editable,
    )
    transform_menu.add_command(
        label="Rotate 90\u00b0 CCW", command=app.rotate_ccw, enabled=layer_editable,
    )
    transform_menu.add_separator()
    transform_menu.add_command(
        label="Transform Layer...", accelerator=accel("edit_transform"),
        command=app.transform_layer, enabled=layer_editable,
    )
    transform_menu.add_command(
        label="Free Transform (Gizmo)", accelerator=accel("edit_free_transform"),
        command=app.toggle_transform,
    )
    edit_menu.add_cascade(label="Transform", menu=transform_menu)
    edit_menu.add_separator()
    edit_menu.add_command(
        label="Settings\u2026", accelerator=accel("edit_settings"),
        command=app.open_settings,
    )
    menus.append(("Edit", edit_menu))

    select_menu = PopupMenu(parent)
    select_menu.add_command(
        label="Select All", accelerator=accel("edit_select_all"),
        command=app.select_all,
    )
    select_menu.add_command(
        label="Deselect", accelerator=accel("escape"), command=app.deselect,
        enabled=has_selection,
    )
    select_menu.add_command(
        label="Invert Selection", command=app.invert_selection,
        enabled=has_selection,
    )
    select_menu.add_separator()
    select_menu.add_command(
        label="Grow Selection", command=app.grow_selection,
        enabled=has_selection,
    )
    select_menu.add_command(
        label="Shrink Selection", command=app.shrink_selection,
        enabled=has_selection,
    )
    select_menu.add_command(
        label="Feather Selection", command=app.feather_selection,
        enabled=has_selection,
    )
    select_menu.add_separator()
    select_menu.add_command(
        label="Delete Selection", accelerator=accel("edit_delete"),
        command=app.delete_selection,
        enabled=lambda: has_selection() and layer_editable(),
    )
    menus.append(("Select", select_menu))

    layers_menu = PopupMenu(parent)

    new_menu = PopupMenu(layers_menu)
    new_menu.add_command(
        label="New Layer", accelerator=accel("layer_new"),
        command=app.add_layer, enabled=has_canvas,
    )
    new_menu.add_command(
        label="New Vector Layer", accelerator=accel("layer_new_vector"),
        command=app.add_vector_layer, enabled=has_canvas,
    )
    new_menu.add_command(
        label="New Layer from Selection", accelerator=accel("layer_from_selection"),
        command=app.layer_from_selection,
        enabled=lambda: has_selection() and bool(app.layers),
    )
    layers_menu.add_cascade(label="New", menu=new_menu)
    layers_menu.add_separator()
    layers_menu.add_command(
        label="Rename Layer...", command=app.rename_layer,
        enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Add to Assets", command=app.add_layer_to_assets,
        enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Alpha to Selection", command=app.alpha_to_selection,
        enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Delete Layer", accelerator=accel("edit_delete"),
        command=app.delete_layer, enabled=lambda: len(app.layers) >= 1,
    )
    layers_menu.add_separator()
    layers_menu.add_command(
        label="Show/Hide Layer", command=app.toggle_visible,
        enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Lock/Unlock Layer", command=app.toggle_lock,
        enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Isolate/Unisolate Layer", accelerator=accel("layer_isolate"),
        command=app.toggle_isolate, enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Show All Layers", command=app.show_all_layers,
        enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Clip to Layer Below", command=app.toggle_clip, enabled=can_clip,
    )
    layers_menu.add_separator()
    layers_menu.add_command(
        label="Move Up", command=lambda: app.move_layer(1),
        enabled=lambda: can_move(1),
    )
    layers_menu.add_command(
        label="Move Down", command=lambda: app.move_layer(-1),
        enabled=lambda: can_move(-1),
    )
    layers_menu.add_separator()
    layers_menu.add_command(
        label="Select Layer Above", accelerator="Up",
        command=lambda: app.select_layer(-1), enabled=lambda: bool(app.layers),
    )
    layers_menu.add_command(
        label="Select Layer Below", accelerator="Down",
        command=lambda: app.select_layer(1), enabled=lambda: bool(app.layers),
    )
    layers_menu.add_separator()
    layers_menu.add_command(
        label="Merge Layers", accelerator=accel("layer_merge"), command=app.merge_layers,
        enabled=lambda: selected_layer_count() >= 2,
    )
    layers_menu.add_separator()

    group_menu = PopupMenu(layers_menu)
    group_menu.add_command(
        label="Group Selected Layers",
        accelerator=accel("layer_group"),
        command=app.group_selected_layers,
        enabled=lambda: selected_layer_count() >= 2,
    )
    group_menu.add_command(
        label="New Group", command=app.add_group, enabled=has_canvas,
    )
    group_menu.add_separator()
    group_menu.add_command(
        label="Ungroup", command=app.ungroup_selected_group, enabled=has_group,
    )
    group_menu.add_command(
        label="Flatten Group", command=app.flatten_selected_group,
        enabled=has_group,
    )
    group_menu.add_command(
        label="Rename Group...", command=app.rename_selected_group,
        enabled=has_group,
    )
    group_menu.add_command(
        label="Show/Hide Group", command=app.toggle_group_visible,
        enabled=has_group,
    )
    collapse_item = group_menu.add_command(
        label="Expand/Collapse Group", command=app.toggle_group_collapse,
        enabled=has_group,
    )

    def _sync_group_collapse(_menu):
        group = app._selected_group()
        collapsed = bool(group is not None and group.collapsed)
        collapse_item["label"] = "Expand Group" if collapsed else "Collapse Group"

    group_menu.set_before_post(_sync_group_collapse)
    group_menu.add_command(
        label="Delete Group", command=app.delete_selected_group,
        enabled=has_group,
    )
    layers_menu.add_cascade(label="Group", menu=group_menu)

    mask_menu = PopupMenu(layers_menu)
    mask_menu.add_command(
        label="Add Layer Mask", command=app.add_layer_mask,
        enabled=lambda: layer_editable() and not has_mask(),
    )
    mask_menu.add_command(
        label="Add Layer Mask (Hidden)",
        command=lambda: app.add_layer_mask(invert=True),
        enabled=lambda: layer_editable() and not has_mask(),
    )
    mask_menu.add_command(
        label="Apply Layer Mask", command=app.apply_layer_mask,
        enabled=lambda: layer_editable() and has_mask(),
    )
    mask_menu.add_command(
        label="Delete Layer Mask", command=app.delete_layer_mask,
        enabled=lambda: layer_editable() and has_mask(),
    )
    mask_menu.add_separator()
    mask_menu.add_command(
        label="Edit Mask", accelerator=accel("layer_mask_edit"),
        command=app.toggle_mask_edit, enabled=has_mask,
    )
    layers_menu.add_cascade(label="Mask", menu=mask_menu)

    rasterize_menu = PopupMenu(layers_menu)
    rasterize_menu.add_command(
        label="Rasterize Texts", command=app.rasterize_selected_texts,
        enabled=lambda: layer_editable() and has_text(),
    )
    rasterize_menu.add_command(
        label="Rasterize Paths", command=app.rasterize_selected_paths,
        enabled=lambda: layer_editable() and has_paths(),
    )
    rasterize_menu.add_command(
        label="Rasterize Vector", command=app.rasterize_vector,
        enabled=lambda: layer_editable() and has_vector(),
    )
    rasterize_menu.add_command(
        label="Rasterize Poses", command=app.rasterize_poses,
        enabled=lambda: layer_editable() and has_poses(),
    )
    layers_menu.add_cascade(label="Rasterize", menu=rasterize_menu)
    layers_menu.add_command(
        label="Convert Paths to Vector Lines",
        command=app.convert_paths_to_vector,
        enabled=lambda: layer_editable() and has_paths(),
    )
    layers_menu.add_separator()
    layers_menu.add_command(
        label="Apply Filters", command=app.bake_filters,
        enabled=lambda: layer_editable() and layer_has_filters(),
    )
    menus.append(("Layer", layers_menu))

    filter_menu = PopupMenu(parent)
    for category_key, category_label in filters.CATEGORIES:
        category_menu = PopupMenu(filter_menu)
        for filter_id in filters.filters_in_category(category_key):
            category_menu.add_command(
                label=filters.FILTERS[filter_id]["label"],
                command=lambda f=filter_id: app.add_filter(f),
                enabled=filter_editable,
            )
        filter_menu.add_cascade(
            label=category_label, menu=category_menu, enabled=filter_editable,
        )
    filter_menu.add_separator()
    filter_menu.add_command(
        label="Edit Filter...", command=app.edit_selected_filter,
        enabled=lambda: (
            app._filter_target() is not None
            or app._group_filter_target() is not None
        ),
    )
    filter_menu.add_separator()
    filter_menu.add_command(
        label="Apply Filters", command=app.bake_filters,
        enabled=lambda: has_filters() and filter_editable(),
    )
    filter_menu.add_command(
        label="Clear Filters", command=app.clear_filters,
        enabled=lambda: has_filters() and filter_editable(),
    )
    menus.append(("Filter", filter_menu))

    canvas_menu = PopupMenu(parent)
    canvas_menu.add_command(label="New Canvas...", command=app.add_canvas)
    canvas_menu.add_command(
        label="Duplicate Canvas", command=app.duplicate_canvas,
        accelerator=accel("canvas_duplicate"), enabled=has_canvas,
    )
    canvas_menu.add_command(
        label="Rename Canvas...", command=app.rename_canvas, enabled=has_canvas,
    )
    canvas_menu.add_command(
        label="Resize / Expand Canvas\u2026", command=app.resize_canvas,
        enabled=has_canvas,
    )
    canvas_menu.add_command(
        label="Color Management\u2026", command=app.open_color_management,
        enabled=has_canvas,
    )
    transform_canvas_menu = PopupMenu(canvas_menu)
    transform_canvas_menu.add_command(
        label="Flip Horizontal", accelerator=accel("canvas_flip_h"),
        command=app.flip_canvas_horizontal, enabled=has_canvas,
    )
    transform_canvas_menu.add_command(
        label="Flip Vertical", accelerator=accel("canvas_flip_v"),
        command=app.flip_canvas_vertical, enabled=has_canvas,
    )
    transform_canvas_menu.add_separator()
    transform_canvas_menu.add_command(
        label="Rotate 90\u00b0 CW", accelerator=accel("canvas_rotate_cw"),
        command=app.rotate_canvas_cw, enabled=has_canvas,
    )
    transform_canvas_menu.add_command(
        label="Rotate 90\u00b0 CCW", accelerator=accel("canvas_rotate_ccw"),
        command=app.rotate_canvas_ccw, enabled=has_canvas,
    )
    canvas_menu.add_cascade(label="Transform Canvas", menu=transform_canvas_menu)
    canvas_menu.add_separator()
    canvas_menu.add_command(
        label="Crop to Selection", command=app.crop_to_selection,
        enabled=has_selection,
    )
    canvas_menu.add_command(
        label="Crop to Content", command=app.crop_to_content, enabled=has_canvas,
    )
    canvas_menu.add_command(
        label="Fit Canvas to Layers", command=app.fit_canvas_to_layers,
        enabled=has_canvas,
    )
    canvas_menu.add_separator()
    canvas_menu.add_command(
        label="Slice Guides to Canvases", command=app.slice_guides_to_canvases,
        enabled=app.can_slice,
    )
    canvas_menu.add_command(
        label="Slice Guides to Layers", command=app.slice_guides_to_layers,
        enabled=app.can_slice,
    )
    canvas_menu.add_separator()
    canvas_menu.add_command(
        label="Delete Canvas", command=app.delete_canvas,
        enabled=lambda: len(app.canvases) >= 1,
    )
    canvas_menu.add_separator()
    canvas_menu.add_command(
        label="Next Canvas", command=lambda: app._cycle_canvas(1),
        accelerator=accel("canvas_next"),
        enabled=lambda: len(app.canvases) > 1,
    )
    canvas_menu.add_command(
        label="Previous Canvas", command=lambda: app._cycle_canvas(-1),
        accelerator=accel("canvas_prev"),
        enabled=lambda: len(app.canvases) > 1,
    )
    menus.append(("Canvas", canvas_menu))

    view_menu = PopupMenu(parent)
    view_menu.add_command(label="Zoom In", accelerator=accel("view_zoom_in"), command=app.zoom_in)
    view_menu.add_command(label="Zoom Out", accelerator=accel("view_zoom_out"), command=app.zoom_out)
    view_menu.add_command(label="Reset Zoom", accelerator=accel("view_zoom_reset"), command=app.reset_zoom)
    view_menu.add_separator()
    view_menu.add_command(label="Rotate View Right", accelerator=accel("view_rotate_cw"), command=app.rotate_view_cw)
    view_menu.add_command(label="Rotate View Left", accelerator=accel("view_rotate_ccw"), command=app.rotate_view_ccw)
    view_menu.add_command(label="Flip View Horizontal", accelerator=accel("view_flip"), command=app.flip_view_horizontal)
    view_menu.add_command(label="Flip View Vertical", accelerator=accel("view_flip_vertical"), command=app.flip_view_vertical)
    view_menu.add_command(label="Reset View", accelerator=accel("view_reset"), command=app.reset_view)
    view_menu.add_separator()
    app.fullscreen_var = tk.BooleanVar(value=app.is_fullscreen())
    view_menu.add_checkbutton(
        label="Fullscreen",
        accelerator=accel("view_fullscreen"),
        variable=app.fullscreen_var,
        command=app.toggle_fullscreen,
    )
    view_menu.add_separator()
    app.offscreen_var = tk.BooleanVar(value=app.show_offscreen)
    view_menu.add_checkbutton(
        label="Show Off-Canvas Content",
        variable=app.offscreen_var,
        command=app.toggle_offscreen,
    )
    view_menu.add_checkbutton(
        label="Exploded Layers",
        accelerator=accel("view_exploded"),
        variable=app.exploded_var,
        command=app.toggle_exploded,
    )
    app.opacity_var = tk.DoubleVar(value=app.offscreen_opacity)
    opacity_menu = PopupMenu(view_menu)
    for label, value in OFFCANVAS_OPACITY_PRESETS:
        opacity_menu.add_radio(
            label=label,
            variable=app.opacity_var,
            value=value,
            command=lambda v=value: app.set_offscreen_opacity(v),
        )
    view_menu.add_cascade(label="Off-Canvas Opacity", menu=opacity_menu)
    view_menu.add_command(
        label="Off-Canvas Background Color...",
        command=app.choose_offscreen_bg,
    )
    view_menu.add_separator()
    app.ruler_var = tk.BooleanVar(value=app.show_rulers)
    view_menu.add_checkbutton(
        label="Show Rulers",
        accelerator=accel("view_rulers"),
        variable=app.ruler_var,
        command=app.toggle_rulers,
    )
    app.guide_var = tk.BooleanVar(value=app.show_guides)
    view_menu.add_checkbutton(
        label="Show Guides",
        accelerator=accel("view_guides"),
        variable=app.guide_var,
        command=app.toggle_guides,
    )
    app.shared_guides_var = tk.BooleanVar(value=app.guides_shared)
    view_menu.add_checkbutton(
        label="Same Guides on All Canvases",
        variable=app.shared_guides_var,
        command=app.toggle_shared_guides,
    )
    view_menu.add_command(
        label="Clear Guides", command=app.clear_guides,
        enabled=lambda: bool(app.guides),
    )
    view_menu.add_separator()
    app.grid_var = tk.BooleanVar(value=app.show_grid)
    view_menu.add_checkbutton(
        label="Show Grid",
        variable=app.grid_var,
        command=app.toggle_grid,
    )
    app.grid_snap_var = tk.BooleanVar(value=app.grid_snap)
    view_menu.add_checkbutton(
        label="Snap to Grid",
        variable=app.grid_snap_var,
        command=app.toggle_grid_snap,
    )
    grid_size_menu = PopupMenu(view_menu)
    app.grid_size_var = tk.IntVar(value=app.grid_size)
    for label, value in GRID_SIZE_PRESETS:
        grid_size_menu.add_radio(
            label=label,
            variable=app.grid_size_var,
            value=value,
            command=lambda v=value: app.set_grid_size(v),
        )
    view_menu.add_cascade(label="Grid Size", menu=grid_size_menu)
    view_menu.add_separator()
    app.dark_var = tk.BooleanVar(value=theme.is_dark())
    view_menu.add_checkbutton(
        label="Dark Mode",
        accelerator=accel("view_theme"),
        variable=app.dark_var,
        command=app.toggle_theme,
    )
    view_menu.add_separator()
    app.gallery_var = tk.BooleanVar(value=False)
    view_menu.add_checkbutton(
        label="Canvas Gallery",
        accelerator=accel("view_gallery"),
        variable=app.gallery_var,
        command=app.toggle_gallery,
    )
    app.gallery_side_var = tk.BooleanVar(value=False)
    view_menu.add_checkbutton(
        label="Gallery Sidebar",
        accelerator=accel("view_gallery_side"),
        variable=app.gallery_side_var,
        command=app.toggle_gallery_side,
    )
    view_menu.add_separator()
    # Pick which side of the canvas each panel lives on. The panel header also
    # carries a move button; both routes go through ``app.move_panel``.
    panels_menu = PopupMenu(view_menu)
    for key, title in PANELS:
        panel_menu = PopupMenu(panels_menu)
        panel_menu.add_command(
            label="Move to Left Sidebar",
            command=lambda k=key: app.move_panel(k, "left"),
        )
        panel_menu.add_command(
            label="Move to Right Sidebar",
            command=lambda k=key: app.move_panel(k, "right"),
        )
        panels_menu.add_cascade(label=title, menu=panel_menu)
    view_menu.add_cascade(label="Sidebar Panels", menu=panels_menu)
    view_menu.add_separator()
    app.cpu_stat_var = tk.BooleanVar(value=app.stat_visible("cpu"))
    view_menu.add_checkbutton(
        label="Show CPU Usage",
        variable=app.cpu_stat_var,
        command=lambda: app.set_stat_visible("cpu", app.cpu_stat_var.get()),
    )
    app.ram_stat_var = tk.BooleanVar(value=app.stat_visible("ram"))
    view_menu.add_checkbutton(
        label="Show RAM Usage",
        variable=app.ram_stat_var,
        command=lambda: app.set_stat_visible("ram", app.ram_stat_var.get()),
    )
    app.gpu_stat_var = tk.BooleanVar(value=app.stat_visible("gpu"))
    view_menu.add_checkbutton(
        label="Show GPU Usage",
        variable=app.gpu_stat_var,
        command=lambda: app.set_stat_visible("gpu", app.gpu_stat_var.get()),
    )
    menus.append(("View", view_menu))

    tools_menu = PopupMenu(parent)
    app.tool_var = tk.StringVar(value="arrow")
    tools_menu.add_radio(label="Arrow (No Tool)", variable=app.tool_var, value="arrow", accelerator=accel("tool_arrow"), command=app.toggle_arrow)
    tools_menu.add_separator()
    tools_menu.add_radio(label="Brush", variable=app.tool_var, value="brush", accelerator=accel("tool_brush"), command=app.toggle_brush)
    tools_menu.add_radio(label="Pick", variable=app.tool_var, value="pick", accelerator=accel("tool_pick"), command=app.toggle_eyedropper)
    tools_menu.add_radio(label="Eraser", variable=app.tool_var, value="eraser", accelerator=accel("tool_eraser"), command=app.toggle_eraser)
    tools_menu.add_radio(label="Paint Bucket", variable=app.tool_var, value="fill", accelerator=accel("tool_fill"), command=app.toggle_fill)
    tools_menu.add_separator()
    tools_menu.add_radio(label="Select", variable=app.tool_var, value="select", accelerator=accel("tool_select"), command=app.select_rect)
    tools_menu.add_radio(label="Lasso", variable=app.tool_var, value="lasso", accelerator=accel("tool_lasso"), command=app.toggle_lasso)
    tools_menu.add_radio(label="Magic Wand", variable=app.tool_var, value="wand", accelerator=accel("tool_wand"), command=app.toggle_wand)
    tools_menu.add_radio(label="Paste", variable=app.tool_var, value="paste", accelerator=accel("tool_paste"), command=app.toggle_paste)
    tools_menu.add_radio(label="Path", variable=app.tool_var, value="path", accelerator=accel("tool_path"), command=app.toggle_path)
    tools_menu.add_radio(label="Vector", variable=app.tool_var, value="vector", accelerator=accel("tool_vector"), command=app.toggle_vector)
    tools_menu.add_radio(label="Poses", variable=app.tool_var, value="poses", accelerator=accel("tool_poses"), command=app.toggle_poses)
    tools_menu.add_radio(label="Liquify", variable=app.tool_var, value="liquify", accelerator=accel("tool_liquify"), command=app.toggle_liquify)
    tools_menu.add_radio(label="Warp", variable=app.tool_var, value="warp", accelerator=accel("tool_warp"), command=app.toggle_warp)
    tools_menu.add_radio(label="Puppet Warp", variable=app.tool_var, value="puppet", accelerator=accel("tool_puppet"), command=app.toggle_puppet)
    tools_menu.add_radio(label="Crop", variable=app.tool_var, value="crop", accelerator=accel("tool_crop"), command=app.toggle_crop)
    tools_menu.add_separator()
    tools_menu.add_radio(label="Move", variable=app.tool_var, value="move", accelerator=accel("tool_move"), command=app.toggle_move)
    tools_menu.add_radio(label="Drag (Pan)", variable=app.tool_var, value="drag", accelerator=accel("tool_drag"), command=app.toggle_drag)
    tools_menu.add_separator()
    tools_menu.add_radio(label="Text", variable=app.tool_var, value="text", accelerator=accel("tool_text"), command=app.toggle_text)
    tools_menu.add_radio(label="Gradient", variable=app.tool_var, value="gradient", accelerator=accel("tool_gradient"), command=app.toggle_gradient)
    tools_menu.add_radio(label="Shape", variable=app.tool_var, value="shape", accelerator=accel("tool_shape"), command=app.toggle_shape)
    tools_menu.add_separator()
    tools_menu.add_radio(label="Clone Stamp", variable=app.tool_var, value="clone", accelerator=accel("tool_clone"), command=app.toggle_clone)
    tools_menu.add_radio(label="Smudge", variable=app.tool_var, value="smudge", accelerator=accel("tool_smudge"), command=app.toggle_smudge)
    tools_menu.add_radio(label="Dodge", variable=app.tool_var, value="dodge", accelerator=accel("tool_dodge"), command=app.toggle_dodge)
    tools_menu.add_radio(label="Burn", variable=app.tool_var, value="burn", accelerator=accel("tool_burn"), command=app.toggle_burn)
    menus.append(("Tools", tools_menu))

    generate_menu = PopupMenu(parent)
    generate_menu.add_command(label="Generate", accelerator=accel("gen_start"), command=app.start_generate)
    generate_menu.add_command(
        label="Cancel Generation", accelerator=accel("gen_cancel"), command=app.cancel_generate,
        enabled=lambda: app.busy,
    )
    generate_menu.add_separator()
    generate_menu.add_command(
        label="Remove Background", accelerator=accel("edit_remove_bg"),
        command=app.start_remove_background, enabled=layer_editable,
    )
    generate_menu.add_command(
        label="Upscale", accelerator=accel("edit_upscale"), command=app.start_upscale,
        enabled=layer_editable,
    )
    generate_menu.add_separator()
    generate_menu.add_command(label="Models\u2026", command=app.open_model_settings)
    menus.append(("Generate", generate_menu))

    help_menu = PopupMenu(parent)
    help_menu.add_command(label="About", command=app.show_about)
    help_menu.add_command(label="Discord", command=app.open_discord)
    help_menu.add_command(label="Contribute", command=app.show_contribute)
    help_menu.add_command(label="Donate", command=app.show_donate)
    menus.append(("Help", help_menu))

    return menus


def build_menubar(app):
    """Build the application menu bar from the shared menu definitions."""
    root = app.root
    menubar = tk.Frame(root)
    theme.register(menubar, bg="bg")
    for label, menu in build_menus(app, menubar):
        button = _MenuButton(menubar, label)
        button.set_menu(menu)
        button.pack(side=tk.LEFT)
    menubar.pack(side=tk.TOP, fill=tk.X)
    edge = tk.Frame(root, height=1)
    theme.register(edge, bg="border")
    edge.pack(side=tk.TOP, fill=tk.X)

    # Native menus dismiss when their window is moved or resized; do the same so
    # an open popup never drifts away from the menu bar.
    def _on_root_configure(event):
        if event.widget is root:
            _close_all_menus()

    root.bind("<Configure>", _on_root_configure, add="+")

    # Keyboard shortcuts are registered and bound by ``app.shortcuts`` (see
    # :mod:`paintapp.shortcuts`) so the Settings dialog can remap them. The
    # manager re-reads any saved overrides and is applied right after this menu
    # is built.
    return menubar
