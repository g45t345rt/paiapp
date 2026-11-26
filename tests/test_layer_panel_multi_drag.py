"""Checks for multi-row layer reordering, list keys, auto-scroll and reveal.

A press on a row that is already part of a multi-row selection keeps the whole
selection so a drag from a selected layer row can carry every selected row; a
click without a drag then collapses to the pressed row. The coordinator moves
all selected layer blocks together (group runs included) and applies the same
group-membership rules the single-layer drag uses. Dragging near the list's top
or bottom edge scrolls it, and selecting a layer from the viewport (the Arrow
tool or an exploded-view tile) scrolls its row into view. The focused list also
offers F2 rename, Ctrl+A select-all-layers and Ctrl+Up/Ctrl+Down stack moves.

Run: .venv\\Scripts\\python.exe tests\\test_layer_panel_multi_drag.py
"""

import os
import sys
import time
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import Canvas  # noqa: E402
from paintapp.layers import LayerPanel  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_panel(rows, selected=(), active=0):
    """A LayerPanel with only the row model and stubbed visual callbacks."""
    probe = object.__new__(LayerPanel)
    probe._rows = list(rows)
    probe._selected = list(selected)
    probe._active = active
    probe._anchor = None
    probe._press = None
    probe._drag_row = None
    probe._drag_rows = None
    probe._ghost = None
    probe._drop_line = None
    probe._selection_kind = "layer"
    probe._selected_filter = None
    probe._selected_text = None
    probe._selected_path = None
    probe._selected_gradient = None
    probe._selected_vector = None
    probe._selected_group = None
    probe._selected_group_filter = None
    probe.calls = []
    probe.focus_list = lambda: None
    probe._apply_selection_colors = lambda: None
    probe._dispatch = lambda row: probe.calls.append(("dispatch", row))
    probe._handle_move = lambda delta: probe.calls.append(("move", delta))
    probe._handle_rename = lambda: probe.calls.append(("rename",))
    probe._end_ghost = lambda: None
    probe._clear_drop_line = lambda: None
    probe._show_drop = lambda x, y: None
    probe._move_ghost = lambda x, y: None
    probe._update_autoscroll = lambda x, y: probe.calls.append(
        ("autoscroll", x, y))
    probe._stop_autoscroll = lambda: None
    probe._begin_ghost = lambda row, x, y, count=1: (
        probe.calls.append(("ghost", row, count)) or True
    )

    def select_rows(rows, target):
        probe._selected = sorted(set(rows))
        probe.calls.append(("select", probe._selected, target))

    probe._select_rows = select_rows
    return probe


def test_click_on_selected_row_keeps_then_collapses():
    rows = [("layer", 3), ("layer", 2), ("layer", 1), ("layer", 0)]
    panel = fake_panel(rows, selected=[0, 1, 2])
    panel._on_press(SimpleNamespace(x_root=10, y_root=10, state=0), 1)
    check("press keeps the multi-row selection", panel._selected, [0, 1, 2])
    check("press marks a pending collapse", panel._press.get("collapse"), True)
    check("press still routes the clicked row",
          ("dispatch", 1) in panel.calls, True)
    panel._on_release(SimpleNamespace(x_root=10, y_root=10))
    check("release without a drag collapses to the row", panel._selected, [1])
    check("collapse routes the row again",
          ("dispatch", 1) in panel.calls, True)


def test_click_outside_selection_collapses_immediately():
    rows = [("layer", 2), ("layer", 1), ("layer", 0)]
    panel = fake_panel(rows, selected=[0, 1])
    panel._on_press(SimpleNamespace(x_root=10, y_root=10, state=0), 2)
    check("outside click replaces the selection", panel._selected, [2])
    check("outside click has no pending collapse",
          panel._press.get("collapse"), False)


def test_drag_sources_carry_the_selection():
    group = object()
    rows = [
        ("layer", 2), ("layer", 1), ("filter", 1, 0),
        ("layer", 0), ("group", group),
    ]
    panel = fake_panel(rows, selected=[0, 1, 2, 4])
    check("layers and group headers are carried",
          panel._reorder_sources(1), [0, 1, 4])
    check("an unselected row drags alone", panel._reorder_sources(3), [3])
    check("a group press drags only itself", panel._reorder_sources(4), [4])


def test_drag_release_reports_the_pressed_source_once():
    rows = [("layer", 2), ("layer", 1), ("layer", 0)]
    panel = fake_panel(rows, selected=[0, 1])
    seen = []
    panel._on_reorder = lambda source, target, above: seen.append(
        (source, target, above))
    panel._press = {"row": 0, "x_root": 0, "y_root": 0, "collapse": True}
    panel._drag_row = 0
    panel._drag_rows = [0, 1]
    panel._drop_at = lambda x, y: (("layer", 0), False, 2)
    panel._on_release(SimpleNamespace(x_root=0, y_root=0))
    check("drop calls the coordinator once",
          seen, [(("layer", 2), ("layer", 0), False)])
    check("a dropped drag does not collapse", panel._selected, [0, 1])

    seen.clear()
    panel._press = {"row": 0, "x_root": 0, "y_root": 0, "collapse": True}
    panel._drag_row = 0
    panel._drag_rows = [0, 1]
    panel._drop_at = lambda x, y: (("layer", 1), True, 1)
    panel._on_release(SimpleNamespace(x_root=0, y_root=0))
    check("dropping on a carried row is ignored", seen, [])


def test_motion_starts_a_multi_row_ghost():
    rows = [("layer", 2), ("layer", 1), ("layer", 0)]
    panel = fake_panel(rows, selected=[0, 1])
    panel._on_press(SimpleNamespace(x_root=0, y_root=0, state=0), 0)
    panel._on_motion(SimpleNamespace(x_root=20, y_root=20))
    check("ghost carries the selection count", ("ghost", 0, 2) in panel.calls,
          True)
    check("drag rows are recorded", panel._drag_rows, [0, 1])
    check("motion updates the auto-scroll",
          ("autoscroll", 20, 20) in panel.calls, True)


def test_motion_on_a_non_draggable_row_collapses():
    rows = [("layer", 1), ("filter", 1, 0)]
    panel = fake_panel(rows, selected=[0, 1])
    panel._on_press(SimpleNamespace(x_root=0, y_root=0, state=0), 1)
    check("press keeps the selection", panel._selected, [0, 1])
    panel._on_motion(SimpleNamespace(x_root=20, y_root=20))
    check("drag attempt on a filter row collapses to it",
          panel._selected, [1])
    check("the press is dropped", panel._press, None)


def test_select_layer_reveals_its_row():
    rows = [("layer", 2), ("group", object()), ("layer", 1), ("layer", 0)]
    panel = fake_panel(rows)
    panel._sync_button_state = lambda: None
    revealed = []
    panel.reveal_row = lambda row: revealed.append(row)
    panel.select_layer(1)
    check("select_layer highlights the owning row", panel._selected, [2])
    check("select_layer scrolls the row into view", revealed, [2])


def test_select_all_key_selects_listed_layers():
    group = object()
    rows = [
        ("group", group), ("layer", 3), ("filter", 3, 0),
        ("layer", 2), ("text", 2, 0), ("layer", 1), ("layer", 0),
    ]
    panel = fake_panel(rows, selected=[5], active=2)
    check("Ctrl+A returns break", panel._on_select_all(), "break")
    check("Ctrl+A selects every layer row", panel._selected, [1, 3, 5, 6])
    check("Ctrl+A anchors on the active layer", panel._anchor, 3)
    check("Ctrl+A dispatches the active layer",
          ("select", [1, 3, 5, 6], 3) in panel.calls, True)

    panel = fake_panel(rows, active=99)
    panel._on_select_all()
    check("an unlisted active layer falls back to the top row",
          panel._anchor, 1)

    panel = fake_panel([("filter", 0, 0), ("group", group)])
    panel._on_select_all()
    check("Ctrl+A with no layer rows selects nothing", panel._selected, [])


def test_rename_and_move_keys_use_the_row_handlers():
    panel = fake_panel([("layer", 1), ("layer", 0)])
    check("F2 returns break", panel._on_rename_key(), "break")
    check("F2 asks for a rename", ("rename",) in panel.calls, True)
    check("Ctrl+Up returns break", panel._on_move_key(1), "break")
    check("Ctrl+Down returns break", panel._on_move_key(-1), "break")
    check("move keys pass the stack direction",
          [call for call in panel.calls if call[0] == "move"],
          [("move", 1), ("move", -1)])


def test_select_vector_row_reveals_its_row():
    rows = [("layer", 1), ("vector", 1, 0), ("layer", 0)]
    panel = fake_panel(rows)
    panel._sync_button_state = lambda: None
    revealed = []
    panel.reveal_row = lambda row: revealed.append(row)
    check("vector row was found", panel.select_vector_row(1, 0), True)
    check("select_vector_row scrolls the row into view", revealed, [1])


def fake_app(doc, selected, groups=()):
    panel = SimpleNamespace(
        selected_indices=lambda count: list(selected),
        selected_groups=lambda: list(groups),
    )
    app = SimpleNamespace(
        current=doc, active=doc.active, layers=doc.layers, layer_panel=panel,
        calls=[],
    )
    app._multi_layer_selection = lambda: (
        PaintApp._multi_layer_selection(app))
    app._layer_blocks = lambda: PaintApp._layer_blocks(app)
    app._push_undo = lambda label="Edit": app.calls.append(("undo", label))
    app._clear_row_selections = lambda: app.calls.append(("clear",))
    app._refresh_layer_list = lambda selected=None: app.calls.append(
        ("refresh", selected))
    app.render = lambda: app.calls.append(("render",))
    app.set_result = lambda message: app.calls.append(("result", message))
    return app


def canvas_with_layers(names):
    doc = Canvas("Test", (4, 4))
    for name in names:
        doc.add_layer(name=name)
    return doc


def test_multi_layer_drag_moves_the_selected_block():
    doc = canvas_with_layers(["A", "B", "C", "D"])
    doc.active = 0
    app = fake_app(doc, selected=[1, 2])
    PaintApp._reorder_selected_layers(app, ("layer", 0), above=False)
    check("selected block moved below A",
          [layer.name for layer in doc.layers], ["B", "C", "A", "D"])
    check("active layer followed the move", app.active, 2)
    check("one undo step was recorded",
          app.calls[0], ("undo", "Reorder layers"))
    check("the moved selection is restored",
          next(call for call in app.calls if call[0] == "refresh"),
          ("refresh", [0, 1]))


def test_reorder_layers_routes_a_multi_selection():
    doc = canvas_with_layers(["A", "B", "C", "D"])
    doc.active = 0
    app = fake_app(doc, selected=[1, 2])
    app._reorder_selected_layers = lambda target, above: (
        PaintApp._reorder_selected_layers(app, target, above))
    PaintApp.reorder_layers(app, ("layer", 2), ("layer", 0), above=False)
    check("the drag moved the whole selection",
          [layer.name for layer in doc.layers], ["B", "C", "A", "D"])


def test_multi_layer_drag_on_selected_block_is_a_noop():
    doc = canvas_with_layers(["A", "B", "C", "D"])
    doc.active = 3
    app = fake_app(doc, selected=[1, 2])
    order = [layer.name for layer in doc.layers]
    PaintApp._reorder_selected_layers(app, ("layer", 1), above=True)
    check("dropping inside the carried selection changes nothing",
          [layer.name for layer in doc.layers], order)
    check("no undo step for a no-op", app.calls, [])


def test_multi_layer_drag_dropped_on_group_joins_it():
    doc = canvas_with_layers(["A", "B", "C"])
    group = doc.new_group("G", members=(doc.layers[1],))
    doc.active = 0
    app = fake_app(doc, selected=[0, 2])
    PaintApp._reorder_selected_layers(app, ("group", group), above=True)
    check("both bare layers joined the target group",
          all(layer.group is group for layer in doc.layers), True)
    check("the group stayed contiguous",
          [layer.name for layer in doc.layers], ["B", "A", "C"])


def test_multi_layer_drag_dropped_on_empty_group_fills_it():
    doc = canvas_with_layers(["A", "B", "C"])
    group = doc.add_group("Empty")
    doc.active = 0
    app = fake_app(doc, selected=[0, 2])
    PaintApp._reorder_selected_layers(app, ("group", group), above=False)
    check("the selected layers joined the empty group",
          [layer.group is group for layer in doc.layers],
          [False, True, True])
    check("the group landed at the top of the stack",
          [layer.name for layer in doc.layers], ["B", "A", "C"])


def test_group_header_selection_moves_with_the_layers():
    doc = canvas_with_layers(["A", "B", "C", "D"])
    group = doc.new_group("G", members=(doc.layers[1],))
    doc.active = 3
    app = fake_app(doc, selected=[0, 2], groups=[group])
    PaintApp._reorder_selected_layers(app, ("layer", 3), above=True)
    check("the group block moved above D with the layers",
          [layer.name for layer in doc.layers], ["D", "A", "B", "C"])
    check("the group is still intact", doc.layers[2].group is group, True)


def test_select_layer_scrolls_the_real_list():
    import tkinter as tk

    from paintapp import theme

    try:
        root = tk.Tk()
        theme.init(root)
    except Exception as exc:  # no display, or Tk missing
        print(f"ok  real list scroll (no display, skipped: {exc})")
        return
    try:
        frame = tk.Frame(root)
        frame.pack(fill=tk.BOTH, expand=True)
        callback = lambda *_args, **_kwargs: None
        panel = LayerPanel(frame, *([callback] * 13))
        panel.pack(fill=tk.BOTH, expand=True)
        root.update()
        check("the list binds F2", bool(panel.canvas.bind("<F2>")), True)
        check("the list binds Ctrl+A",
              bool(panel.canvas.bind("<Control-a>")), True)
        check("the list binds Ctrl+Up",
              bool(panel.canvas.bind("<Control-Up>")), True)
        check("the list binds Ctrl+Down",
              bool(panel.canvas.bind("<Control-Down>")), True)
        doc = Canvas("Big", (4, 4))
        for index in range(40):
            doc.add_layer(name=f"Layer {index}")
        panel.refresh(doc.layers, len(doc.layers) - 1)
        panel.canvas.yview_moveto(0.0)
        root.update()
        check("real list starts at the top", panel.canvas.yview()[0], 0.0)
        panel.select_layer(0)
        root.update()
        check("select_layer scrolled the list to the layer",
              panel.canvas.yview()[0] > 0.0, True)

        panel.canvas.yview_moveto(0.5)
        root.update()
        panel._drag_row = 0
        panel._drag_rows = [0]
        left = panel.canvas.winfo_rootx()
        top = panel.canvas.winfo_rooty()
        bottom = top + panel.canvas.winfo_height()
        middle_x = left + 10

        panel.canvas.yview_moveto(0.0)
        root.update()
        panel._update_autoscroll(middle_x, top + 4)
        check("no auto-scroll at the top", panel._autoscroll_job, None)
        panel.canvas.yview_moveto(0.5)
        root.update()
        panel._update_autoscroll(middle_x, top + 4)
        check("drag near the top starts auto-scrolling",
              panel._autoscroll_job is not None, True)
        check("auto-scroll direction is up", panel._autoscroll_dir, -1)
        before = panel.canvas.yview()[0]
        deadline = time.monotonic() + 2.0
        while (time.monotonic() < deadline
               and panel.canvas.yview()[0] >= before):
            root.update()
            time.sleep(0.01)
        check("the timer scrolled the list up",
              panel.canvas.yview()[0] < before, True)
        check("auto-scroll keeps ticking", panel._autoscroll_job is not None,
              True)

        panel._update_autoscroll(middle_x, (top + bottom) // 2)
        check("leaving the edge stops auto-scrolling",
              panel._autoscroll_job, None)
        panel._update_autoscroll(left - 30, top + 4)
        check("dragging outside the column does not auto-scroll",
              panel._autoscroll_job, None)

        panel.canvas.yview_moveto(1.0)
        root.update()
        panel._update_autoscroll(middle_x, bottom - 4)
        check("no auto-scroll at the bottom", panel._autoscroll_job, None)
        panel.canvas.yview_moveto(0.5)
        root.update()
        panel._update_autoscroll(middle_x, bottom - 4)
        check("auto-scroll direction is down", panel._autoscroll_dir, 1)
        before = panel.canvas.yview()[0]
        deadline = time.monotonic() + 2.0
        while (time.monotonic() < deadline
               and panel.canvas.yview()[0] <= before):
            root.update()
            time.sleep(0.01)
        check("the timer scrolled the list down",
              panel.canvas.yview()[0] > before, True)
        panel._stop_autoscroll()
        panel._drag_row = None
        panel._drag_rows = None
    finally:
        root.destroy()


def main():
    test_click_on_selected_row_keeps_then_collapses()
    test_click_outside_selection_collapses_immediately()
    test_drag_sources_carry_the_selection()
    test_drag_release_reports_the_pressed_source_once()
    test_motion_starts_a_multi_row_ghost()
    test_motion_on_a_non_draggable_row_collapses()
    test_select_layer_reveals_its_row()
    test_select_vector_row_reveals_its_row()
    test_select_all_key_selects_listed_layers()
    test_rename_and_move_keys_use_the_row_handlers()
    test_multi_layer_drag_moves_the_selected_block()
    test_reorder_layers_routes_a_multi_selection()
    test_multi_layer_drag_on_selected_block_is_a_noop()
    test_multi_layer_drag_dropped_on_group_joins_it()
    test_multi_layer_drag_dropped_on_empty_group_fills_it()
    test_group_header_selection_moves_with_the_layers()
    test_select_layer_scrolls_the_real_list()
    if FAILED:
        print("\n".join(FAILED))
        raise SystemExit(1)
    print("\nall layer panel multi-drag checks passed")


if __name__ == "__main__":
    main()
