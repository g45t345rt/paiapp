"""Checks that tool settings update in place instead of rebuilding.

Rebuilding the settings panel on every control change destroyed the widget the
user was interacting with and reset the sidebar's scroll. The Shape and Select
modes now build their rows once and only show/hide them. Run:
.venv\\Scripts\\python.exe tests\\test_tool_settings_inplace.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_mode_changes_update_in_place():
    import tkinter as tk

    from paintapp.drawing import ToolsPanel

    try:
        root = tk.Tk()
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"tool settings test skipped: {exc}")
        return
    try:
        panel = ToolsPanel(
            root, lambda: None, lambda *_a: None, lambda: None)
        panel.pack(fill=tk.BOTH, expand=True)
        root.update()

        panel.select("shape")
        root.update()
        fill_btn = panel._shape_fill_btn
        panel._shape_mode_var.set("stroke")
        panel._on_shape_mode()
        root.update()
        check("shape swatch widget survives a mode change",
              fill_btn.winfo_exists(), 1)
        check("shape stroke swatch is shown",
              panel._shape_stroke_btn.winfo_manager(), "pack")
        check("shape fill swatch is hidden",
              panel._shape_fill_btn.winfo_manager(), "")

        # Re-syncing the active tool must not rebuild its panel either.
        panel._sync_buttons()
        root.update()
        check("same-tool sync keeps the settings widgets",
              fill_btn.winfo_exists(), 1)

        # Restoring loaded settings still has to rebuild the panel so the
        # controls show the restored values.
        panel.apply_settings({})
        root.update()
        check("applying settings still rebuilds the panel",
              fill_btn.winfo_exists(), 0)
        fill_btn = panel._shape_fill_btn

        panel.select("select")
        root.update()
        hint = panel._select_hint
        panel.set_select_mode("wand")
        root.update()
        check("select mode change keeps the hint widget",
              hint.winfo_exists(), 1)
        check("select wand rows are shown",
              panel._select_wand_rows[0].winfo_manager(), "pack")
        check("select rect rows are hidden",
              panel._select_rect_rows[0].winfo_manager(), "")

        # A real tool switch still builds a fresh panel.
        panel.select("brush")
        root.update()
        check("switching tools rebuilds settings", hint.winfo_exists(), 0)
    finally:
        root.destroy()


def main():
    test_mode_changes_update_in_place()
    for note in SKIPPED:
        print(f"skip  {note}")
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        return 1
    print("\nall tool settings checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
