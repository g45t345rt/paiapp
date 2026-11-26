"""Checks that entering a mask mode deactivates every other tool.

Mask editing and inpaint-mask painting are tool flags of their own, so turning
one on must clear all the others. A hard-coded list of tools used to miss
Gradient, Shape, Clone, Smudge, Dodge, Burn, Crop, Layout and the 3D tool, so
enabling the Mask tool from one of them left that tool active: clicks kept
running the old tool and the Mask button never lit up. Selecting Brush first
cleared the leftover flag, which is why that worked around the bug.

The panel is built with ``object.__new__`` and the coordinator is a lightweight
fake, so no Tk root or canvas is needed.

Run: .venv\\Scripts\\python.exe tests\\test_mask_tool_activation.py
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402
from paintapp.drawing import ToolsPanel  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def tools_probe(active):
    """A ToolsPanel with its flag table populated and ``active`` selected."""
    probe = object.__new__(ToolsPanel)
    for flag in ToolsPanel.TOOL_FLAGS.values():
        setattr(probe, flag, False)
    setattr(probe, ToolsPanel.TOOL_FLAGS[active], True)
    probe.mask_value = 0
    probe._sync_buttons = lambda: None
    return probe


class FakeApp:
    """The coordinator state the two mask-mode methods touch."""

    def __init__(self, tools):
        self.tools = tools
        self.layer_panel = SimpleNamespace(set_mask_edit=lambda enabled: None)
        self.current = SimpleNamespace(inpaint_mask_surface=lambda: None)
        self._mask_preview_state = (False, False)
        self.renders = 0
        self.modes = 0

    def active_layer(self):
        return SimpleNamespace(mask=object())

    def _active_layer_locked(self, mask=False):
        return False

    def _on_mode_change(self):
        self.modes += 1

    def render(self):
        self.renders += 1

    def set_result(self, _text):
        pass


def test_mask_edit_clears_every_tool():
    for key in ToolsPanel.TOOL_FLAGS:
        if key in ("mask", "inpaint_mask"):
            continue
        tools = tools_probe(key)
        app = FakeApp(tools)
        PaintApp.set_mask_edit(app, True)
        check(f"mask edit clears {key}",
              getattr(tools, ToolsPanel.TOOL_FLAGS[key]), False)
        check(f"mask edit is the only tool after {key}",
              tools._active_tool(), "mask")
        check(f"mask edit flag is set after {key}", tools.mask_edit, True)
        check(f"mask edit reports done after {key}", app.modes, 1)


def test_inpaint_mask_edit_clears_every_tool():
    for key in ToolsPanel.TOOL_FLAGS:
        if key in ("mask", "inpaint_mask"):
            continue
        tools = tools_probe(key)
        app = FakeApp(tools)
        PaintApp.set_inpaint_mask_edit(app, True)
        check(f"inpaint mask clears {key}",
              getattr(tools, ToolsPanel.TOOL_FLAGS[key]), False)
        check(f"inpaint mask is the only tool after {key}",
              tools._active_tool(), "inpaint_mask")
        check(f"layer mask edit stays off after {key}", tools.mask_edit, False)
        check(f"inpaint mask flag is set after {key}",
              tools.inpaint_mask_edit, True)
        check(f"inpaint mask saves the previous paint value after {key}",
              app._mask_value_before_inpaint, 0)
        check(f"inpaint mask paints the marked value after {key}",
              tools.mask_value, 255)
        check(f"inpaint mask reports done after {key}", app.modes, 1)


def test_reentering_inpaint_mask_keeps_the_saved_value():
    tools = tools_probe("inpaint_mask")
    tools.mask_value = 255
    app = FakeApp(tools)
    app._mask_value_before_inpaint = 0
    PaintApp.set_inpaint_mask_edit(app, True)
    check("re-entering keeps the saved value",
          app._mask_value_before_inpaint, 0)
    check("re-entering keeps the marked value", tools.mask_value, 255)


def test_leaving_mask_edit_keeps_other_tools_alone():
    tools = tools_probe("mask")
    app = FakeApp(tools)
    PaintApp.set_mask_edit(app, False)
    check("leaving mask edit clears the flag", tools.mask_edit, False)
    # No tool was active behind the mask, so the coordinator falls back to Arrow.
    check("leaving mask edit selects arrow", tools.arrow_tool, True)


def main():
    test_mask_edit_clears_every_tool()
    test_inpaint_mask_edit_clears_every_tool()
    test_reentering_inpaint_mask_keeps_the_saved_value()
    test_leaving_mask_edit_keeps_other_tools_alone()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        return 1
    print("\nall mask tool activation checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
