"""Unit tests for controller-owned tool overlay cleanup.

Run: .venv\\Scripts\\python.exe tests\\test_tool_overlays.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.canvasarea import CanvasArea, CanvasView  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


CONTROLLER_OVERLAYS = {
    "set_cursor_overlay",
    "set_selection_label",
    "set_gizmo",
    "set_path",
    "set_vector_overlay",
    "set_poses_overlay",
    "set_warp_grid",
    "set_puppet_overlay",
    "set_text_session",
    "set_shape_preview",
    "set_gradient_preview",
    "set_crop_preview",
}


class Recorder:
    """A stand-in view that records which setters ``clear_tool_overlays`` uses."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith("set_"):
            def record(*args, **kwargs):
                self.calls.append((name, args))
            return record
        raise AttributeError(name)


def test_clear_tool_overlays_drops_controller_state():
    view = Recorder()
    CanvasView.clear_tool_overlays(view)
    cleared = {name for name, _args in view.calls}
    check("every controller-owned overlay is cleared", cleared, CONTROLLER_OVERLAYS)
    check("every clear passes None", {args for _name, args in view.calls}, {(None,)})


def test_clear_tool_overlays_keeps_per_canvas_state():
    view = Recorder()
    CanvasView.clear_tool_overlays(view)
    cleared = {name for name, _args in view.calls}
    check("the canvas selection is kept", "set_selection" in cleared, False)
    check("the canvas layout boxes are kept", "set_layout_boxes" in cleared, False)
    check("the canvas generation preview is kept",
          "set_generation_preview" in cleared, False)
    check("the canvas mask preview is kept", "set_mask_preview" in cleared, False)


class FakeView:
    def __init__(self):
        self.doc = None
        self.selection = "old"
        self.cleared = 0

    def set_selection(self, selection):
        self.selection = selection

    def clear_tool_overlays(self):
        self.cleared += 1


class FakeDoc:
    widget = None


class FakePage:
    def __init__(self):
        self.forgotten = False
        self.packed = False

    def pack_forget(self):
        self.forgotten = True

    def pack(self, **kwargs):
        self.packed = True


def test_show_page_clears_the_hidden_view():
    area = CanvasArea.__new__(CanvasArea)
    pages = [FakePage(), FakePage()]
    views = [FakeView(), FakeView()]
    area._pages = pages
    area._views = views
    area._packed = 0
    CanvasArea._show_page(area, 1)
    check("the hidden page is unpacked", pages[0].forgotten, True)
    check("the hidden view drops its tool overlays", views[0].cleared, 1)
    check("the shown page is packed", pages[1].packed, True)
    check("the shown view keeps its overlays", views[1].cleared, 0)


def test_show_page_leaves_the_current_view_alone():
    area = CanvasArea.__new__(CanvasArea)
    pages = [FakePage()]
    views = [FakeView()]
    area._pages = pages
    area._views = views
    area._packed = 0
    CanvasArea._show_page(area, 0)
    check("showing the packed page is a no-op", views[0].cleared, 0)
    check("showing the packed page keeps it packed", pages[0].packed, False)


def test_retarget_drops_overlays_on_every_view():
    area = CanvasArea.__new__(CanvasArea)
    views = [FakeView(), FakeView()]
    area._views = views
    area.refresh_titles = lambda documents: None
    CanvasArea.retarget(area, [FakeDoc(), FakeDoc()])
    check("every re-pointed view drops its tool overlays",
          [view.cleared for view in views], [1, 1])
    check("every re-pointed view drops its selection",
          [view.selection for view in views], [None, None])


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
