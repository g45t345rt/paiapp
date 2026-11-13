"""Checks for layer selection with the Arrow tool on the canvas.

Clicking with the Arrow tool should make the topmost visible layer with a
pixel under the pointer the active layer and mirror that in the layers panel.
Alt-click cycles down through the overlapping layers at that point and wraps
back to the top. Hidden layers, hidden/transparent groups and transparent
pixels are skipped so a click never selects something the user cannot see.

Run: .venv\\Scripts\\python.exe tests\\test_arrow_layer_select.py
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.document import Group, Layer  # noqa: E402
from paintapp.drawing import DrawingController  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def layer(name, box, offset=(0, 0), visible=True, opacity=1.0, group=None):
    """An RGBA layer with an opaque ``box`` painted at its own origin."""
    item = Layer(name, (100, 100), (0, 0, 0, 0))
    item.image.paste((255, 0, 0, 255), box)
    item.offset = offset
    item.visible = visible
    item.opacity = opacity
    item.group = group
    return item


def controller(layers, active=0):
    seen = []
    doc = SimpleNamespace(layers=layers, active=active)
    app = SimpleNamespace(
        current=doc, active=active,
        render=lambda: seen.append("render"),
        set_result=lambda message: seen.append(message),
    )

    def select(index):
        app.active = index
        seen.append(("select", index))

    app._on_layer_select = select
    app.layer_panel = SimpleNamespace(
        select_layer=lambda index: seen.append(("panel", index))
    )
    probe = object.__new__(DrawingController)
    probe.app = app
    return probe, app, seen


def test_topmost_layer_wins():
    bottom = layer("bottom", (0, 0, 90, 90))
    top = layer("top", (40, 40, 90, 90), offset=(10, 10))
    probe, _, _ = controller([bottom, top])
    check("overlap picks the top layer", probe._layer_at(55, 55), 1)
    check("single layer picks the bottom", probe._layer_at(5, 5), 0)


def test_transparency_and_empty_space_are_skipped():
    sparse = layer("sparse", (0, 0, 20, 20))
    probe, _, _ = controller([sparse])
    check("transparent pixel is a miss", probe._layer_at(80, 80), None)
    check("outside the canvas is a miss", probe._layer_at(-5, 80), None)


def test_hidden_layers_and_groups_are_skipped():
    hidden = layer("hidden", (0, 0, 100, 100), visible=False)
    group = Group("group", 1)
    group.visible = False
    grouped = layer("grouped", (0, 0, 100, 100), group=group)
    probe, _, _ = controller([hidden, grouped])
    check("hidden layer is a miss", probe._layer_at(50, 50), None)


def test_click_mirrors_the_panel():
    bottom = layer("bottom", (0, 0, 100, 100))
    top = layer("top", (40, 40, 90, 90), offset=(10, 10))
    probe, app, seen = controller([bottom, top])
    probe._arrow_press(55, 55)
    check("active layer follows the click", app.active, 1)
    check("document selection updated", ("select", 1) in seen, True)
    check("panel selection updated", ("panel", 1) in seen, True)
    check("canvas repainted", "render" in seen, True)
    check("click reports the layer", "Selected layer 'top'." in seen, True)


def test_click_on_active_layer_does_nothing():
    only = layer("only", (0, 0, 100, 100))
    probe, _, seen = controller([only])
    probe._arrow_press(50, 50)
    check("no reselect for the active layer", seen, [])


def test_alt_click_cycles_overlapping_layers():
    bottom = layer("bottom", (0, 0, 100, 100))
    middle = layer("middle", (0, 0, 100, 100))
    top = layer("top", (0, 0, 100, 100))
    probe, app, _ = controller([bottom, middle, top], active=2)
    probe._arrow_press(50, 50)
    check("plain click selects the top", app.active, 2)
    probe._arrow_press(50, 50, cycle=True)
    check("alt-click selects the layer below", app.active, 1)
    probe._arrow_press(50, 50, cycle=True)
    check("alt-click continues down", app.active, 0)
    probe._arrow_press(50, 50, cycle=True)
    check("alt-click wraps to the top", app.active, 2)
    probe._arrow_press(50, 50)
    check("a plain click returns to the top", app.active, 2)


def test_alt_click_starts_at_the_top_from_outside():
    bottom = layer("bottom", (0, 0, 100, 100))
    top = layer("top", (0, 0, 100, 100))
    probe, app, _ = controller([bottom, top], active=0)
    probe._arrow_press(50, 50, cycle=True)
    check("alt-click with no hit active starts at the top", app.active, 1)


def main():
    test_topmost_layer_wins()
    test_transparency_and_empty_space_are_skipped()
    test_hidden_layers_and_groups_are_skipped()
    test_click_mirrors_the_panel()
    test_click_on_active_layer_does_nothing()
    test_alt_click_cycles_overlapping_layers()
    test_alt_click_starts_at_the_top_from_outside()
    if FAILED:
        print("\n".join(FAILED))
        raise SystemExit(1)
    print("\nall arrow layer selection checks passed")


if __name__ == "__main__":
    main()
