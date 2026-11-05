"""A serial job queue for long-running processes.

Image generation, background removal, and any future heavy operation are
expensive on the GPU, so running several at once only makes each one slower and
can exhaust memory. This module funnels them through a single worker thread: the
sidebar's :class:`JobQueuePanel` shows what is running and what is waiting, and
each job can be cancelled while it is queued or running.

The pieces are deliberately small:

* :class:`Job` packages one unit of work - a ``runner`` for the worker thread
  and a ``finisher`` callback the Tk thread runs once the result is ready.
* :class:`JobQueue` owns the worker thread and the ordering. It never touches
  widgets: it reports every state change by calling a ``post`` function, which
  the app points at its own Tk message queue.
* :class:`JobQueuePanel` renders the queue in the sidebar and forwards the
  cancel/clear clicks back to the app.
"""

import itertools
import threading
import time
import tkinter as tk
from tkinter import ttk

from engine.cancel import CancelToken, Cancelled

from . import icons, theme
from .constants import QUEUE_LIST_MAX_H
from .scrollbar import ArrowScrollbar

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"
CANCELLED = "cancelled"

FINISHED = (DONE, ERROR, CANCELLED)

_ACTIVE = (QUEUED, RUNNING)

# Per-status presentation: the leading icon (a vendored Tabler name, plus the
# text glyph used when the rasteriser is missing) and the theme token for its
# color.
_MARKS = {
    QUEUED: ("clock", "\u2022", "muted"),
    RUNNING: ("player-play", "\u25b6", "accent"),
    DONE: ("check", "\u2713", "text"),
    ERROR: ("x", "\u2717", "error"),
    CANCELLED: ("ban", "\u2013", "muted"),
}

# The row's action button, per state: pending (a stop was already requested),
# stop a cancellable job, or dismiss a finished one.
_BTN_PENDING = ("dots", "\u2026")
_BTN_STOP = ("player-stop", "\u25a0")
_BTN_DISMISS = ("x", "\u2715")

# Status icons are a little smaller than the layer-panel badges.
MARK_ICON = 14

_IDS = itertools.count(1)


def _set_icon(label, name, glyph, color, text_width=2):
    """Draw ``name`` (or the text ``glyph``) on ``label``, tinted ``color``.

    A label showing an image reads ``width`` in pixels rather than text units,
    so the icon path clears it and the text fallback restores the fixed slot.
    """
    image = icons.photo(name, MARK_ICON, color) if name else None
    if image is not None:
        label.config(image=image, text="", fg=color, width=0)
    else:
        label.config(image="", text=glyph, fg=color, width=text_width)


def format_duration(seconds):
    """A short ``14s`` / ``1:05`` label for a finished job's runtime."""
    if seconds is None:
        return ""
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{seconds:.0f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}:{secs:02d}"


class Job:
    """One queued unit of work.

    ``runner(progress, cancel)`` runs on the queue's worker thread and returns
    the result. It should call ``progress(frac, label, sub)`` as it advances -
    ``frac`` is the overall fraction and ``sub`` the current sub-process's own
    fraction (or ``None``) - and raise :class:`~engine.cancel.Cancelled`
    (through the token) to abort.
    ``finisher(result, error, cancelled)`` runs on the Tk thread and applies the
    outcome to the document, so it may touch widgets freely.
    """

    def __init__(self, kind, label, runner, finisher, detail=""):
        self.id = next(_IDS)
        self.kind = kind
        self.label = label
        self.detail = detail
        self.runner = runner
        self.finisher = finisher
        self.cancel_token = CancelToken()
        self.status = QUEUED
        self.cancelling = False
        self.keeping = False
        self.progress = 0.0
        self.sub_progress = None
        self.progress_label = ""
        self.error = None
        self.started_at = None
        self.finished_at = None
        self.elapsed = None

    @property
    def active(self):
        return self.status in _ACTIVE

    @property
    def finished(self):
        return self.status in FINISHED

    @property
    def cancellable(self):
        return self.status in _ACTIVE and not self.cancelling and not self.keeping


class JobQueue:
    """Runs :class:`Job` objects one at a time on a single worker thread.

    Every state transition is announced through ``post`` (the app passes
    ``self.queue.put``), so this class stays free of any Tk dependency.
    """

    def __init__(self, post):
        self._post = post
        self._jobs = []
        self._pending = []
        self._current = None
        self._cond = threading.Condition()
        self._closed = False
        self._worker = threading.Thread(
            target=self._loop, name="pai-job-queue", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------ queries
    def jobs(self):
        """A snapshot of every job, oldest first."""
        with self._cond:
            return list(self._jobs)

    def get(self, job_id):
        with self._cond:
            return self._find(job_id)

    @property
    def current(self):
        with self._cond:
            return self._current

    def has_running(self):
        return self.current is not None

    def has_active(self, kind=None):
        """True when a job of ``kind`` is queued or running."""
        with self._cond:
            return any(
                job.active and (kind is None or job.kind == kind)
                for job in self._jobs
            )

    def running_or_last(self, kind):
        """The running ``kind`` job, else the newest queued one, else ``None``."""
        with self._cond:
            for job in self._jobs:
                if job.kind == kind and job.status == RUNNING:
                    return job
            for job in reversed(self._jobs):
                if job.kind == kind and job.status == QUEUED:
                    return job
            return None

    def counts(self):
        """``(running, queued)`` totals across every kind."""
        with self._cond:
            running = sum(1 for job in self._jobs if job.status == RUNNING)
            queued = sum(1 for job in self._jobs if job.status == QUEUED)
            return running, queued

    # ----------------------------------------------------------------- mutators
    def submit(self, job):
        """Append ``job`` and wake the worker."""
        with self._cond:
            if self._closed:
                return
            self._jobs.append(job)
            self._pending.append(job)
            self._cond.notify()
        self._post(("jobstate", job.id))

    def cancel(self, job_id):
        """Cancel one job: drop it if queued, trip its token if running.

        A running job is flagged ``cancelling`` and posted at once, so the UI
        can show feedback the moment the click lands even though the worker
        only settles the job once the engine reaches its next safe point.
        """
        notify = False
        with self._cond:
            job = self._find(job_id)
            if job is None:
                return
            if job.status == QUEUED:
                job.status = CANCELLED
                try:
                    self._pending.remove(job)
                except ValueError:
                    pass
                notify = True
            elif job.status == RUNNING:
                if not job.cancelling:
                    job.cancelling = True
                    notify = True
                job.cancel_token.cancel()
        if notify:
            self._post(("jobstate", job_id))

    def keep(self, job_id):
        """Ask a running job to stop early and keep its current result.

        Unlike :meth:`cancel` this is not an abort: the engine returns the
        partial result and the job settles ``done``, so the finisher applies it
        as usual. A no-op unless the job is running.
        """
        notify = False
        with self._cond:
            job = self._find(job_id)
            if job is None or job.status != RUNNING:
                return
            if not job.keeping:
                job.keeping = True
                notify = True
            job.cancel_token.request_keep()
        if notify:
            self._post(("jobstate", job_id))

    def cancel_all(self):
        """Cancel every queued and running job."""
        with self._cond:
            changed = []
            for job in self._jobs:
                if job.status == QUEUED:
                    job.status = CANCELLED
                    changed.append(job.id)
                    try:
                        self._pending.remove(job)
                    except ValueError:
                        pass
            current = self._current
            if current is not None:
                if not current.cancelling:
                    current.cancelling = True
                    changed.append(current.id)
                current.cancel_token.cancel()
            self._cond.notify()
        for job_id in changed:
            self._post(("jobstate", job_id))

    def clear_finished(self):
        """Forget every finished job so the panel stays short."""
        with self._cond:
            self._jobs = [job for job in self._jobs if not job.finished]

    def remove(self, job_id):
        """Forget one finished job (a no-op while it is still active)."""
        with self._cond:
            self._jobs = [
                job for job in self._jobs
                if not (job.id == job_id and job.finished)
            ]

    def close(self):
        with self._cond:
            self._closed = True
            self._cond.notify()

    # ---------------------------------------------------------------- internals
    def _find(self, job_id):
        for job in self._jobs:
            if job.id == job_id:
                return job
        return None

    def _loop(self):
        while True:
            with self._cond:
                while not self._pending and not self._closed:
                    self._cond.wait()
                if self._closed:
                    return
                job = self._pending.pop(0)
                if job.status != QUEUED:
                    continue
                job.status = RUNNING
                job.progress = 0.0
                job.started_at = time.monotonic()
                self._current = job
            self._post(("jobstate", job.id))
            self._run(job)

    def _run(self, job):
        def progress(frac, label="", sub=None):
            job.progress = frac
            job.sub_progress = sub
            job.progress_label = label
            self._post(("jobprogress", job.id, frac, label, sub))

        try:
            result = job.runner(progress, job.cancel_token)
        except Cancelled:
            self._settle(job, CANCELLED)
            self._post(("jobdone", job.id, None, None, True))
            return
        except Exception as exc:
            job.error = f"Error: {exc}"
            self._settle(job, ERROR)
            self._post(("jobdone", job.id, None, job.error, False))
            return
        self._settle(job, DONE)
        self._post(("jobdone", job.id, result, None, False))

    def _settle(self, job, status):
        with self._cond:
            job.status = status
            job.finished_at = time.monotonic()
            if job.started_at is not None:
                job.elapsed = job.finished_at - job.started_at
            if self._current is job:
                self._current = None


class JobRow(tk.Frame):
    """One row of the queue panel: glyph, label, progress, and a button.

    Rows are built after the app's initial :func:`theme.apply`, so each one
    restyles itself once it exists - otherwise it would keep Tk's default light
    colors in dark mode.
    """

    def __init__(self, parent, job, on_action, stripe=False):
        super().__init__(parent)
        self.job_id = job.id
        self._on_action = on_action
        self._status = job.status
        self._bg = "stripe_bg" if stripe else "panel_bg"
        self._build()
        theme.restyle(self)
        self.update(job)

    def _build(self):
        self.columnconfigure(1, weight=1)
        theme.register(self, bg=self._bg)

        self.mark = tk.Label(self, width=2, anchor="n")
        self.mark.grid(row=0, column=0, rowspan=2, sticky="n", pady=(1, 0))
        theme.register(self.mark, bg=self._bg)

        self.name = tk.Label(self, anchor="w", justify="left")
        self.name.grid(row=0, column=1, sticky="ew")
        theme.register(self.name, bg=self._bg, fg="text")

        self.button = tk.Label(self, width=2, cursor="arrow")
        self.button.grid(row=0, column=2, sticky="ne")
        theme.register(self.button, bg=self._bg, fg="muted")
        self.button.bind("<Button-1>", self._clicked)
        self.button.bind("<Enter>", self._hover_on)
        self.button.bind("<Leave>", self._hover_off)

        self.detail = tk.Label(self, anchor="w", justify="left")
        self.detail.grid(row=1, column=1, sticky="ew")
        theme.register(self.detail, bg=self._bg, fg="muted")

        self.time = tk.Label(self, anchor="e", width=5)
        self.time.grid(row=1, column=2, sticky="e")
        theme.register(self.time, bg=self._bg, fg="muted")

        self.progress = ttk.Progressbar(
            self, mode="determinate", maximum=100, value=0, length=80
        )
        self.progress.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(2, 0))

    def update(self, job):
        """Refresh the row from ``job``'s current state."""
        self._status = job.status
        name, glyph, token = _MARKS.get(job.status, _MARKS[QUEUED])
        theme.register(self.mark, fg=token)
        _set_icon(self.mark, name, glyph, theme.color(token))
        theme.register(self.button, fg="muted")
        self.name.config(text=job.label)

        if job.status == RUNNING:
            if job.cancelling:
                self.detail.config(text="Cancelling\u2026")
            elif job.keeping:
                self.detail.config(text="Stopping\u2026 keeping result")
            else:
                self.detail.config(text=job.progress_label or "Running...")
            theme.register(self.detail, fg="muted")
            self.progress.grid()
            self.progress.config(value=max(0.0, min(1.0, job.progress)) * 100)
        else:
            self.progress.grid_remove()
            if job.status == ERROR:
                self.detail.config(text=job.error or "Failed")
                theme.register(self.detail, fg="error")
            elif job.status == DONE:
                self.detail.config(text=job.detail or "Done")
                theme.register(self.detail, fg="muted")
            elif job.status == CANCELLED:
                self.detail.config(text="Cancelled")
                theme.register(self.detail, fg="muted")
            else:
                self.detail.config(text=job.detail or "Queued")
                theme.register(self.detail, fg="muted")

        self.time.config(
            text=format_duration(job.elapsed) if job.finished else ""
        )

        if job.cancelling or job.keeping:
            # Already asked to stop; the button is inert until the worker
            # settles the job, so its icon reads as pending rather than active.
            name, glyph = _BTN_PENDING
        else:
            name, glyph = _BTN_STOP if job.cancellable else _BTN_DISMISS
        _set_icon(self.button, name, glyph, theme.color("muted"))

    def set_progress(self, frac):
        self.progress.config(value=max(0.0, min(1.0, frac)) * 100)

    def _clicked(self, _event):
        self._on_action(self.job_id)

    def _hover_on(self, _event):
        self.button.configure(bg=theme.color("button_hover"))

    def _hover_off(self, _event):
        self.button.configure(bg=theme.color(self._bg))


class JobQueuePanel(tk.Frame):
    """The sidebar list of queued, running, and finished jobs."""

    def __init__(self, parent, on_action, on_cancel_all, on_clear_finished):
        super().__init__(parent)
        self._on_action = on_action
        self._on_cancel_all = on_cancel_all
        self._on_clear_finished = on_clear_finished
        self._jobs = {}
        self._rows = {}
        self._build()
        theme.subscribe(self._restyle)

    def _build(self):
        self.summary = tk.Label(self, anchor="w")
        self.summary.pack(fill=tk.X, pady=(2, 2))
        theme.register(self.summary, fg="muted")

        # The job rows live in a capped, scrolling viewport: a long queue
        # scrolls inside the panel instead of pushing the rest of the rail
        # off-screen. The canvas sizes itself to its content up to
        # :data:`QUEUE_LIST_MAX_H` and carries a permanent scrollbar; the wheel
        # over it scrolls the rail as usual, and Ctrl+wheel scrolls the queue.
        # Built first, packed by :meth:`_hide_empty`.
        self.list_frame = tk.Frame(self)
        self.canvas = tk.Canvas(
            self.list_frame, highlightthickness=0, height=1,
            bg=theme.color("panel_bg"),
        )
        theme.register(self.canvas, bg="panel_bg")
        scrollbar = ArrowScrollbar(self.list_frame, command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.rows = tk.Frame(self.canvas)
        theme.register(self.rows, bg="panel_bg")
        self._window = self.canvas.create_window(
            (0, 0), window=self.rows, anchor="nw"
        )
        self.rows.bind("<Configure>", self._on_rows_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_ctrl_wheel(self.canvas)

        self.empty = tk.Label(self, text="No jobs queued.", anchor="w")
        theme.register(self.empty, fg="muted")

        self.buttons = tk.Frame(self)
        self.buttons.pack(fill=tk.X, pady=(6, 4))
        self.clear_btn = tk.Button(
            self.buttons, text="Clear Finished", command=self._on_clear_finished
        )
        self.clear_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.cancel_all_btn = tk.Button(
            self.buttons, text="Cancel All", command=self._on_cancel_all
        )
        self.cancel_all_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 0))

        self._empty_shown = False
        self._show_empty()

    # ---------------------------------------------------------------- scrolling
    def _on_rows_configure(self, _event=None):
        """Fit the viewport to its content, up to the panel's cap."""
        height = self.rows.winfo_reqheight()
        self.canvas.configure(height=max(1, min(height, QUEUE_LIST_MAX_H)))
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self._window, width=event.width)

    def _bind_ctrl_wheel(self, widget):
        """Let Ctrl+wheel scroll the queue from anywhere over it.

        A plain wheel is left alone, so it scrolls the sidebar rail as usual. A
        canvas only receives a binding over itself, not over the row widgets
        embedded in it, so each row (and every child) gets the Ctrl binding too.
        """
        widget.bind("<Control-MouseWheel>", self._on_ctrl_wheel)
        for child in widget.winfo_children():
            self._bind_ctrl_wheel(child)

    def _on_ctrl_wheel(self, event):
        self.canvas.yview_scroll(int(-event.delta / 120) or 0, "units")
        return "break"

    # ------------------------------------------------------------------ public
    def add(self, job):
        self._jobs[job.id] = job
        self._hide_empty()
        row = JobRow(
            self.rows, job, self._on_action, stripe=len(self._rows) % 2 == 1
        )
        slaves = self.rows.pack_slaves()
        if slaves:
            row.pack(fill=tk.X, pady=1, before=slaves[0])
        else:
            row.pack(fill=tk.X, pady=1)
        self._rows[job.id] = row
        self._bind_ctrl_wheel(row)
        self.canvas.yview_moveto(0.0)
        self._update_summary()

    def update(self, job):
        self._jobs[job.id] = job
        row = self._rows.get(job.id)
        if row is None:
            self.add(job)
            return
        row.update(job)
        self._update_summary()

    def set_progress(self, job_id, frac, label=""):
        row = self._rows.get(job_id)
        if row is not None:
            row.set_progress(frac)
            if label:
                row.detail.config(text=label)

    def rebuild(self, jobs):
        """Replace every row from ``jobs`` (used after a bulk prune)."""
        for row in self._rows.values():
            row.destroy()
        self._rows = {}
        self._jobs = {}
        for job in reversed(jobs):
            self._jobs[job.id] = job
            row = JobRow(
                self.rows, job, self._on_action, stripe=len(self._rows) % 2 == 1
            )
            row.pack(fill=tk.X, pady=1)
            self._rows[job.id] = row
            self._bind_ctrl_wheel(row)
        if jobs:
            self._hide_empty()
        else:
            self._show_empty()
        self.canvas.yview_moveto(0.0)
        self._update_summary()

    # --------------------------------------------------------------- internals
    def _hide_empty(self):
        """Show the job list and drop the placeholder label."""
        if self._empty_shown:
            self.empty.pack_forget()
            self._empty_shown = False
        if not self.list_frame.winfo_manager():
            self.list_frame.pack(fill=tk.X, before=self.buttons)

    def _show_empty(self):
        """Hide the job list and show the placeholder label."""
        if not self._empty_shown:
            self.empty.pack(fill=tk.X, pady=(1, 0), before=self.buttons)
            self._empty_shown = True
        if self.list_frame.winfo_manager():
            self.list_frame.pack_forget()

    def _update_summary(self):
        running = sum(1 for job in self._jobs.values() if job.status == RUNNING)
        queued = sum(1 for job in self._jobs.values() if job.status == QUEUED)
        if not running and not queued:
            self.summary.config(text="Idle")
        else:
            self.summary.config(text=f"{running} running \u00b7 {queued} queued")
        self.cancel_all_btn.config(
            state=tk.NORMAL if (running or queued) else tk.DISABLED
        )

    def _restyle(self):
        """Recompute the status glyph colors after a theme switch."""
        for job_id, row in self._rows.items():
            job = self._jobs.get(job_id)
            if job is not None:
                row.update(job)
        self._update_summary()
