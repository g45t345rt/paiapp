"""Tests for the coalesced live paint repaint
(run: .venv\\Scripts\\python.exe tests\\test_paint_throttle.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402
from paintapp.drawing import DrawingController  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _Root:
    """A stand-in for ``tk.Tk`` that records and runs ``after`` jobs."""

    def __init__(self):
        self.jobs = {}
        self.cancelled = []
        self._next = 0

    def after(self, ms, fn):
        self._next += 1
        self.jobs[self._next] = fn
        return self._next

    def after_cancel(self, job):
        self.cancelled.append(job)
        self.jobs.pop(job, None)

    def run(self, job):
        self.jobs.pop(job)()


def scheduler():
    """A ``PaintApp`` with the real scheduler but a recording render."""
    app = PaintApp.__new__(PaintApp)
    app.root = _Root()
    app.renders = 0
    app._stroke_render_job = None
    app.render = lambda: setattr(app, "renders", app.renders + 1)
    return app


def test_schedule_coalesces_the_burst():
    # A burst of motion events inside the timer window must queue one repaint,
    # not one per event, so the loop keeps draining pointer samples.
    app = scheduler()
    for _ in range(8):
        app.schedule_live_render()
    check("a burst queues one repaint", len(app.root.jobs), 1)
    app.root.run(app._stroke_render_job)
    check("the queued repaint paints once", app.renders, 1)
    check("the job slot is cleared", app._stroke_render_job, None)
    app.schedule_live_render()
    check("a later schedule queues again", len(app.root.jobs), 1)


def test_flush_paints_now_and_cancels():
    app = scheduler()
    app.schedule_live_render()
    app.flush_live_render()
    check("flush paints immediately", app.renders, 1)
    check("flush clears the job slot", app._stroke_render_job, None)
    check("flush cancels the queued repaint", app.root.cancelled, [1])
    check("no repaint is left queued", len(app.root.jobs), 0)


class _Tools:
    """Every tool flag reads as off unless a test sets it."""

    def __getattr__(self, name):
        return False


class _View:
    def __init__(self):
        self.zoom = 0.2

    def event_to_doc(self, event):
        return float(event.x), float(event.y)


class _Event:
    def __init__(self, widget, x, y, state=0x1):
        self.widget = widget
        self.x = x
        self.y = y
        self.state = state


class _CtrlApp:
    def __init__(self):
        self.tools = _Tools()
        self.scheduled = 0
        self.rendered = 0

    def schedule_live_render(self):
        self.scheduled += 1

    def render(self):
        self.rendered += 1


def controller():
    """A controller with the stroke machinery stubbed but the drag path real."""
    ctrl = DrawingController.__new__(DrawingController)
    ctrl.app = _CtrlApp()
    ctrl.last_x = 0
    ctrl.last_y = 0
    ctrl.cursor_x = 0
    ctrl.cursor_y = 0
    ctrl._continue_paint = lambda *a, **k: None
    ctrl._paint_speed = lambda *a, **k: None
    ctrl._apply_cursor = lambda *a, **k: None
    return ctrl


def test_drag_coalesces_through_the_app():
    ctrl = controller()
    view = _View()
    for i in range(10):
        ctrl.on_drag(_Event(view, 10 + i, 10 + i))
    check("every drag asks the app to schedule a repaint",
          ctrl.app.scheduled, 10)
    check("the drag never renders inline", ctrl.app.rendered, 0)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
