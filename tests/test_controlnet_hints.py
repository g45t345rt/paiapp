"""Checks for the ControlNet panel's hint sources and assign bindings.

Builds the real widget on a withdrawn Tk root and drives a row's Image kind,
the Assign capture and the settings round-trip; then resolves bindings through
the real ``PaintApp`` helpers with a minimal stand-in coordinator. Skipped
(with a note) when no display is available.

Run: .venv\\Scripts\\python.exe tests\\test_controlnet_hints.py
"""

import os
import sys
from types import SimpleNamespace
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import controlnet_panel  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402
from paintapp.controlnet_panel import (  # noqa: E402
    HINT_ASSET, HINT_CANVAS, HINT_LAYER, HINT_SOURCE,
)

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _build_panel(capture=None):
    import tkinter as tk

    from paintapp.controlnet_panel import ControlNetPanel

    root = tk.Tk()
    root.withdraw()
    panel = ControlNetPanel(root, on_settings=lambda _rows: None, on_capture=capture)
    panel.set_family("stable_diffusion")
    panel.set_enabled(True)
    return root, panel


def _add_fake_row(panel, **kwargs):
    with mock.patch.object(
        controlnet_panel, "_catalog", return_value={"fake": "/fake.safetensors"}
    ):
        panel.add_row(model="fake", **kwargs)


def test_assign_then_round_trip():
    captured = []

    def capture(kind):
        captured.append(kind)
        if kind == "layer":
            return {"source": "layer", "layer": "L1", "label": "Layer: L1"}
        return {"source": "asset", "asset": 7, "label": "Asset: face"}

    try:
        root, panel = _build_panel(capture=capture)
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"panel test skipped: {exc}")
        return
    try:
        _add_fake_row(panel)
        row = panel._rows[0]
        check("default hint", row["hint_var"].get(), HINT_SOURCE)
        check("no binding stored", "binding" in panel.settings()[0], False)
        check("target hidden for source", row["target_frame"].winfo_manager(), "")

        # Selecting a layer kind reveals Assign; assigning pins the layer.
        row["hint_var"].set(HINT_LAYER)
        check("target shown for layer", row["target_frame"].winfo_manager(), "grid")
        panel._assign_row(row)
        check("captured kind", captured[-1], "layer")
        check("target label", row["target_var"].get(), "Layer: L1")
        check("layer binding stored", panel.settings()[0].get("binding"),
              {"source": "layer", "layer": "L1", "label": "Layer: L1"})

        # A different kind drops the pin until the new kind is assigned.
        row["hint_var"].set(HINT_ASSET)
        check("kind change clears", "binding" in panel.settings()[0], False)
        panel._assign_row(row)
        check("asset binding stored", panel.settings()[0].get("binding"),
              {"source": "asset", "asset": 7, "label": "Asset: face"})

        # Source and Canvas have nothing to pin, so Target hides again.
        row["hint_var"].set(HINT_CANVAS)
        check("canvas clears pin", "binding" in panel.settings()[0], False)
        check("target hidden for canvas", row["target_frame"].winfo_manager(), "")

        # Stored rows rebuild with their binding and kind intact.
        with mock.patch.object(
            controlnet_panel, "_catalog",
            return_value={"fake": "/fake.safetensors"},
        ):
            panel.set_settings([{
                "kind": "controlnet",
                "model": "/fake.safetensors",
                "display": "fake",
                "hint": HINT_LAYER,
                "binding": {"source": "layer", "layer": "L2", "label": "Layer: L2"},
                "preprocessor": "none",
                "strength": 1.0,
                "start": 0.0,
                "end": 1.0,
            }])
        rebuilt = panel.settings()[0]
        check("rebuilt hint", rebuilt.get("hint"), HINT_LAYER)
        check("rebuilt binding", rebuilt.get("binding"),
              {"source": "layer", "layer": "L2", "label": "Layer: L2"})
        check("rebuilt label", panel._rows[0]["target_var"].get(), "Layer: L2")

        # A binding whose kind no longer matches the hint is dropped on load.
        with mock.patch.object(
            controlnet_panel, "_catalog",
            return_value={"fake": "/fake.safetensors"},
        ):
            panel.set_settings([{
                "kind": "controlnet",
                "model": "/fake.safetensors",
                "display": "fake",
                "hint": HINT_SOURCE,
                "binding": {"source": "layer", "layer": "L2", "label": "Layer: L2"},
            }])
        check("stale binding dropped", "binding" in panel.settings()[0], False)
    finally:
        root.destroy()


def test_two_rows_pin_different_layers():
    names = iter(("L1", "L2"))

    def capture(kind):
        name = next(names)
        return {"source": "layer", "layer": name, "label": f"Layer: {name}"}

    try:
        root, panel = _build_panel(capture=capture)
    except Exception as exc:
        SKIPPED.append(f"two-row test skipped: {exc}")
        return
    try:
        _add_fake_row(panel)
        _add_fake_row(panel)
        for row in panel._rows:
            row["hint_var"].set(HINT_LAYER)
            panel._assign_row(row)
        check("distinct pins", [r["binding"]["layer"] for r in panel.settings()],
              ["L1", "L2"])
    finally:
        root.destroy()


class _HintApp:
    """A minimal coordinator for the unbound resolution helpers."""

    _binding_image = PaintApp._binding_image
    _resolve_control_hints = PaintApp._resolve_control_hints

    def __init__(self):
        self.current = SimpleNamespace(
            layers=[SimpleNamespace(name="L1"), SimpleNamespace(name="L2")],
        )
        self.asset_library = SimpleNamespace(
            assets=[SimpleNamespace(id=1, name="a"), SimpleNamespace(id=2, name="b")],
        )
        self.warnings = []

    def composite(self):
        return "canvas"

    def active_layer(self):
        return self.current.layers[1]

    def selected_asset(self):
        return self.asset_library.assets[1]

    def _layer_init_image(self, layer):
        return f"layer:{layer.name}"

    def _asset_init_image(self, asset):
        return f"asset:{asset.id}"

    def set_result(self, text, _kind=None):
        self.warnings.append(text)


def test_resolve_hints_uses_pins():
    app = _HintApp()
    rows = [
        {"hint": HINT_SOURCE},
        {"hint": HINT_CANVAS},
        {"hint": HINT_LAYER, "binding": {"source": "layer", "layer": "L1"}},
        {"hint": HINT_ASSET, "binding": {"source": "asset", "asset": 1}},
    ]
    check("pinned resolution", app._resolve_control_hints(rows, "init"),
          ["init", "canvas", "layer:L1", "asset:1"])

    # An unassigned row (old project) falls back to the current selection.
    check("unassigned fallback",
          app._resolve_control_hints([{"hint": HINT_LAYER}], "init"),
          ["layer:L2"])
    check("unassigned asset fallback",
          app._resolve_control_hints([{"hint": HINT_ASSET}], "init"),
          ["asset:2"])

    # Source with no init image warns instead of producing an image.
    app.warnings.clear()
    check("source needs init", app._resolve_control_hints([{"hint": HINT_SOURCE}], None),
          None)
    check("source warning", app.warnings[-1],
          "A control adapter set to Source needs a Source image.")


def test_binding_image_pins_exact_item():
    app = _HintApp()
    check("pinned layer", app._binding_image({"source": "layer", "layer": "L1"}),
          "layer:L1")
    check("missing layer falls back",
          app._binding_image({"source": "layer", "layer": "gone"}), "layer:L2")
    check("pinned asset", app._binding_image({"source": "asset", "asset": 1}),
          "asset:1")
    check("missing asset falls back",
          app._binding_image({"source": "asset", "asset": 99}), "asset:2")
    check("canvas binding", app._binding_image({"source": "canvas"}), "canvas")


def main():
    for test in (test_assign_then_round_trip, test_two_rows_pin_different_layers,
                 test_resolve_hints_uses_pins, test_binding_image_pins_exact_item):
        try:
            test()
        except Exception as exc:  # keep going so one failure is not fatal
            import traceback

            FAILED.append(f"{test.__name__} raised: {exc}\n{traceback.format_exc()}")

    for note in SKIPPED:
        print(f"skip {note}")
    if FAILED:
        print("\nFAILURES:")
        for line in FAILED:
            print(" -", line)
        return 1
    print("\nall controlnet hint tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
