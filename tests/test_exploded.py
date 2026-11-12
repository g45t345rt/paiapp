"""Regression tests for the exploded-grid view (run: .venv\\Scripts\\python.exe tests\\test_exploded.py).

``paintapp.exploded`` lays a canvas's layers out side by side in a grid, and
``Canvas.composite_layout`` flattens that arrangement while clipping each layer
to its own cell. The subtle parts are that cells must not overlap, that a
layer's content lands inside its cell (not at its document offset), and that the
view's coordinate shift maps a pointer in a cell back onto the active layer's
own pixels - which is what lets the normal tools edit one tile. These tests
check the layout, the flatten and the shift round-trip.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw  # noqa: E402

from paintapp import exploded  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def build_canvas():
    canvas = Canvas("t", (120, 100))
    canvas.layers = []
    canvas.add_layer("a")
    canvas.add_layer("b")
    canvas.add_layer("c")
    ImageDraw.Draw(canvas.layers[0].image).rectangle(
        (10, 10, 40, 40), fill=(255, 0, 0, 255)
    )
    ImageDraw.Draw(canvas.layers[1].image).rectangle(
        (5, 5, 70, 30), fill=(0, 255, 0, 255)
    )
    ImageDraw.Draw(canvas.layers[2].image).rectangle(
        (2, 2, 20, 60), fill=(0, 0, 255, 255)
    )
    return canvas


def test_cells_do_not_overlap():
    layout = exploded.compute(build_canvas(), columns=2)
    check("columns", layout.columns, 2)
    check("cell count", len(layout.cells), 3)
    for i in range(len(layout.cells)):
        for j in range(i + 1, len(layout.cells)):
            a = layout.cells[i]
            b = layout.cells[j]
            disjoint = a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1]
            check(f"cells {i}/{j} disjoint", disjoint, True)
    check("size fits cells", layout.size[0],
          max(cell[2] for cell in layout.cells))


def test_cell_lookup():
    layout = exploded.compute(build_canvas(), columns=2)
    for index, (left, top, right, bottom) in enumerate(layout.cells):
        cx = (left + right) // 2
        cy = (top + bottom) // 2
        check(f"cell_at {index}", layout.cell_at(cx, cy), index)
    check("cell_at gap", layout.cell_at(-5, -5), None)


def test_content_lands_in_cell():
    canvas = build_canvas()
    layout = exploded.compute(canvas, columns=2)
    image, origin = canvas.composite_layout(
        layout, (0, 0, layout.size[0], layout.size[1]), 1.0
    )
    check("origin", origin, (0, 0))
    check("composite size", image.size, layout.size)
    for index, (left, top, right, bottom) in enumerate(layout.cells):
        cell = image.crop((left, top, right, bottom))
        bbox = cell.getbbox()
        check(f"cell {index} has content", bbox is not None, True)
        if bbox is not None:
            check(f"cell {index} content inside", (
                bbox[0] >= 0 and bbox[1] >= 0
                and bbox[2] <= right - left and bbox[3] <= bottom - top
            ), True)


def test_scaled_composite():
    canvas = build_canvas()
    layout = exploded.compute(canvas, columns=2)
    image, _origin = canvas.composite_layout(
        layout, (0, 0, layout.size[0], layout.size[1]), 0.5
    )
    check("scaled width", image.width, max(1, round(layout.size[0] * 0.5)))


def test_hidden_layer_still_shown():
    """The exploded view ignores visibility: every layer gets its content."""
    canvas = build_canvas()
    canvas.layers[1].visible = False
    layout = exploded.compute(canvas, columns=2)
    image, _origin = canvas.composite_layout(
        layout, (0, 0, layout.size[0], layout.size[1]), 1.0
    )
    left, top, right, bottom = layout.cells[1]
    check("hidden cell has content", image.crop(
        (left, top, right, bottom)).getbbox() is not None, True)


def test_shift_round_trip():
    """A pointer in layer i's cell maps back to layer i's own pixels."""
    canvas = build_canvas()
    layout = exploded.compute(canvas, columns=2)
    for index in range(len(canvas.layers)):
        layer = canvas.layers[index]
        dx, dy = layout.offsets[index]
        shift = (layer.offset[0] - dx, layer.offset[1] - dy)
        # A point a little inside the cell, at cell-relative (20, 15).
        cell_left, cell_top = layout.cells[index][0], layout.cells[index][1]
        layout_x, layout_y = cell_left + 20, cell_top + 15
        doc_x, doc_y = layout_x + shift[0], layout_y + shift[1]
        local_x = doc_x - layer.offset[0]
        local_y = doc_y - layer.offset[1]
        check(f"layer {index} local x", local_x, 20)
        check(f"layer {index} local y", local_y, 15)


def test_cells_fit_full_images():
    """Cells are at least a whole layer image, so content is never clipped."""
    canvas = Canvas("t", (64, 48))
    canvas.layers = []
    canvas.add_layer("a")
    canvas.add_layer("b")
    ImageDraw.Draw(canvas.layers[0].image).rectangle(
        (0, 0, 63, 47), fill=(255, 0, 0, 255)
    )
    ImageDraw.Draw(canvas.layers[1].image).rectangle(
        (0, 0, 63, 47), fill=(0, 255, 0, 255)
    )
    layout = exploded.compute(canvas, columns=2)
    image, _origin = canvas.composite_layout(
        layout, (0, 0, layout.size[0], layout.size[1]), 1.0
    )
    for index, (left, top, right, bottom) in enumerate(layout.cells):
        cell = image.crop((left, top, right, bottom))
        check(f"cell {index} fully covered",
              cell.getbbox(), (0, 0, right - left, bottom - top))


def test_gaps_show_background():
    """The exploded board paints gaps with the background, cells with checker."""
    from paintapp.canvasarea import compose_region

    composite = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    composite.paste((255, 0, 0, 255), (0, 0, 60, 40))
    cells = [(0, 0, 100, 100), (140, 0, 240, 100)]
    board = compose_region(
        composite, (240, 100), (0, 0), 1.0, (0, 0), (240, 100),
        (80, 80, 80), 1.0, None, cells,
    )
    check("gap is background", board.getpixel((120, 50)), (80, 80, 80))
    check("cell shows checker", board.getpixel((80, 80))[0] >= 200, True)


def _build_view(canvas, layout):
    """A CanvasView showing ``layout``, or ``(None, None)`` without a display."""
    import tkinter as tk

    from paintapp.canvasarea import CanvasView

    try:
        root = tk.Tk()
    except tk.TclError:
        SKIPPED.append("exploded generation preview (no display)")
        return None, None
    root.withdraw()
    view = CanvasView(root, canvas)
    view.set_layout(layout)
    return root, view


def test_toggle_keeps_camera():
    """Opening and closing the grid keeps the zoom and the stacked camera."""
    import tkinter as tk

    from paintapp.canvasarea import CanvasView

    try:
        root = tk.Tk()
    except tk.TclError:
        SKIPPED.append("exploded camera (no display)")
        return
    root.withdraw()
    canvas = build_canvas()
    layout = exploded.compute(canvas, columns=2)
    view = CanvasView(root, canvas)
    view.config(scrollregion=(0, 0, 4000, 4000))
    view.zoom = 2.0
    view.xview_moveto(0.3)
    view.yview_moveto(0.4)
    before = view.view_state()
    canvas.active = 1
    view.enter_exploded(layout)
    sx, sy = view._active_shift()
    shifted = view._pending_anchor
    check("explode keeps zoom", view.zoom, before["zoom"])
    check("explode maps anchor into the cell", (
        shifted[0], shifted[1],
    ), (
        before["anchor"][0] - sx, before["anchor"][1] - sy,
    ))
    check("explode shows the grid", view.exploded, True)
    view.leave_exploded()
    check("collapse keeps zoom", view.zoom, before["zoom"])
    check("collapse restores anchor", view._pending_anchor, before["anchor"])
    check("collapse clears the grid", view.exploded, False)
    root.destroy()


def test_preview_maps_to_active_cell():
    """A pending generation preview is centred on the active layer's tile."""
    canvas = build_canvas()
    layout = exploded.compute(canvas, columns=2)
    built = _build_view(canvas, layout)
    if built[0] is None:
        return
    root, view = built
    image = Image.new("RGBA", (60, 50), (255, 0, 0, 255))
    view.set_generation_preview(image, (30, 25), fit=False)
    canvas.active = 0
    check("preview origin cell 0", view._preview_layout_origin(), (30, 25))
    canvas.active = 1
    check("preview origin cell 1", view._preview_layout_origin(), (190, 25))
    root.destroy()


def test_preview_drawn_on_active_cell():
    """The overlay pastes the preview onto the active cell, not the document."""
    canvas = build_canvas()
    layout = exploded.compute(canvas, columns=2)
    built = _build_view(canvas, layout)
    if built[0] is None:
        return
    root, view = built
    image = Image.new("RGBA", (60, 50), (255, 0, 0, 255))
    view.set_generation_preview(image, (30, 25), fit=False)
    canvas.active = 1
    board = Image.new("RGB", (400, 200), (255, 255, 255))
    out = view._overlay_generation_preview(
        board, (0, 0), (400, 200), view._preview_layout_origin()
    )
    check("preview over cell 1", out.getpixel((200, 35)), (255, 0, 0))
    dim = out.getpixel((40, 35))
    check("document position is only dimmed",
          dim[0] == dim[1] == dim[2] and dim[0] < 200, True)
    root.destroy()


def main():
    test_cells_do_not_overlap()
    test_cell_lookup()
    test_content_lands_in_cell()
    test_hidden_layer_still_shown()
    test_scaled_composite()
    test_shift_round_trip()
    test_cells_fit_full_images()
    test_gaps_show_background()
    test_toggle_keeps_camera()
    test_preview_maps_to_active_cell()
    test_preview_drawn_on_active_cell()
    print()
    for note in SKIPPED:
        print(f"skipped  {note}")
    if FAILED:
        print(f"{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        return 1
    print("all exploded-view tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())