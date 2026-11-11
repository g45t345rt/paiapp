"""Checks for Up/Down arrow navigation of the layer list.

The list is drawn top-most-first, so the arrow keys should walk the column the
user sees: ``<Up>`` selects the layer above the active one, ``<Down>`` the layer
below, skipping group headers and filter, text and path rows, and staying put at
either end. The panel's row model is exercised without a Tk root, and the app
action is driven through a lightweight fake.

Run: .venv\\Scripts\\python.exe tests\\test_layer_keyboard_nav.py
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402
from paintapp.layers import LayerPanel  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def panel(rows, active):
    """A LayerPanel with only the row model populated (no Tk widgets built)."""
    probe = object.__new__(LayerPanel)
    probe._rows = list(rows)
    probe._active = active
    return probe


# Document order is bottom-first, but the list renders top-most-first, so row 0
# is the highest layer.
ROWS = [
    ("layer", 2),
    ("group", object()),
    ("layer", 1),
    ("filter", 1, 0),
    ("layer", 0),
]


def test_step_layer_walks_visible_layers():
    probe = panel(ROWS, active=1)
    check("up selects layer above", probe.step_layer(-1), 2)
    check("down selects layer below", probe.step_layer(1), 0)
    check("down skips the filter row", probe.step_layer(1), 0)


def test_step_layer_stops_at_the_ends():
    top = panel(ROWS, active=2)
    check("up at the top stays put", top.step_layer(-1), None)
    bottom = panel(ROWS, active=0)
    check("down at the bottom stays put", bottom.step_layer(1), None)


def test_step_layer_without_layer_rows():
    probe = panel([("group", object())], active=0)
    check("no layers to step to", probe.step_layer(1), None)


def test_on_arrow_selects_and_reveals():
    probe = panel(ROWS, active=1)
    seen = []
    probe._select_rows = lambda rows, target: seen.append(("select", rows, target))
    probe.reveal_row = lambda row: seen.append(("reveal", row))
    check("arrow returns break", probe._on_arrow(-1), "break")
    # Up from layer 1 (row 2) lands on layer 2 (row 0).
    check("up selected the row above", ("select", [0], 0) in seen, True)
    check("up revealed that row", ("reveal", 0) in seen, True)
    check("panel active followed", probe._active, 2)


def test_on_arrow_at_the_end_does_nothing():
    probe = panel(ROWS, active=2)
    seen = []
    probe._select_rows = lambda rows, target: seen.append(("select", rows, target))
    probe.reveal_row = lambda row: seen.append(("reveal", row))
    check("arrow returns break at the top", probe._on_arrow(-1), "break")
    check("nothing selected at the top", seen, [])


def fake_app(active, panel_obj, layers):
    """A stand-in whose attributes satisfy ``PaintApp.select_layer`` alone."""
    calls = []
    app = SimpleNamespace(
        current=object(), active=active, layers=layers, layer_panel=panel_obj,
        calls=calls,
    )
    app._on_layer_select = lambda index: (
        setattr(app, "active", index), calls.append(("select", index)))
    app._refresh_layer_list = lambda: calls.append(("refresh",))
    app.layer_panel.reveal_layer = lambda index: calls.append(("reveal", index))
    return app


def test_select_layer_applies_the_move():
    layers = [object(), object(), object()]
    probe = panel(ROWS, active=1)
    app = fake_app(1, probe, layers)
    PaintApp.select_layer(app, -1)
    check("up moved active to layer 2", app.active, 2)
    check("up reveal", ("reveal", 2) in app.calls, True)
    check("up refreshed the list", ("refresh",) in app.calls, True)


def test_select_layer_at_the_end_does_nothing():
    layers = [object(), object(), object()]
    probe = panel(ROWS, active=2)
    app = fake_app(2, probe, layers)
    PaintApp.select_layer(app, -1)
    check("no move past the top", app.active, 2)
    check("no refresh at the top", app.calls, [])


def main():
    test_step_layer_walks_visible_layers()
    test_step_layer_stops_at_the_ends()
    test_step_layer_without_layer_rows()
    test_on_arrow_selects_and_reveals()
    test_on_arrow_at_the_end_does_nothing()
    test_select_layer_applies_the_move()
    test_select_layer_at_the_end_does_nothing()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print(" ", line)
        return 1
    print("\nall layer keyboard navigation checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
