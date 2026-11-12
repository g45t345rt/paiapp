"""Checks that tool settings update in place instead of rebuilding.

Rebuilding the settings panel on every control change destroyed the widget the
user was interacting with and reset the sidebar's scroll. Mode switches, brush
presets, and project settings now update the existing controls in place. Run:
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

        # Restoring loaded settings updates the same controls in place.
        panel.apply_settings({"shape_opacity": 55, "shape_mode": "both"})
        root.update()
        check("applying settings keeps the settings widgets",
              fill_btn.winfo_exists(), 1)
        check("applying settings updates a slider",
              panel._shape_opacity_var.get(), 55)
        check("applying settings updates a mode",
              panel._shape_mode_var.get(), "both")
        check("applying settings updates the mode rows",
              panel._shape_fill_btn.winfo_manager(), "pack")

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


def test_presets_and_settings_update_in_place():
    import tkinter as tk

    from paintapp import icons
    from paintapp.drawing import ToolsPanel

    # Photos belong to the Tk root that made them; the first test used another.
    icons.clear()
    try:
        root = tk.Tk()
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"tool preset test skipped: {exc}")
        return
    try:
        panel = ToolsPanel(
            root, lambda: None, lambda *_a: None, lambda: None)
        panel.pack(fill=tk.BOTH, expand=True)
        panel.select("brush")
        root.update()
        preview = panel._brush_preview
        blend_combo = None
        for widget in panel._walk_settings_widgets(panel.settings):
            if getattr(widget, "_setting_var", None) is panel._brush_blend_var:
                blend_combo = widget

        panel.apply_tool_params("brush", {
            "brush_opacity": 42, "brush_blend": "multiply", "brush_size": 33,
        })
        root.update()
        check("preset keeps the brush widgets", preview.winfo_exists(), 1)
        check("preset updates the opacity attribute", panel.brush_opacity, 42)
        check("preset updates the opacity control",
              panel._brush_opacity_var.get(), 42)
        check("preset updates the size control", panel.size_var.get(), 33)
        check("preset updates a menu value",
              panel._brush_blend_var.get(), "multiply")
        check("preset updates the menu label",
              blend_combo._display_var.get(),
              panel._option_label("multiply", blend_combo._setting_options))

        panel.apply_settings({
            "brush_hardness": 17, "brush_blend": "screen",
            "select_mode": "wand", "wand_tolerance": 88,
        })
        root.update()
        check("settings load keeps the brush widgets",
              preview.winfo_exists(), 1)
        check("settings load updates a slider",
              panel._brush_hardness_var.get(), 17)
        check("settings load updates a menu label",
              blend_combo._display_var.get(),
              panel._option_label("screen", blend_combo._setting_options))

        # A mode changed by the loaded settings updates its rows in place.
        panel.select("select")
        root.update()
        hint = panel._select_hint
        panel.apply_settings({"select_mode": "wand", "wand_tolerance": 12})
        root.update()
        check("settings load keeps the select hint", hint.winfo_exists(), 1)
        check("settings load shows the new mode rows",
              panel._select_wand_rows[0].winfo_manager(), "pack")
        check("settings load updates the mode value",
              panel._select_mode_var.get(), "wand")
    finally:
        root.destroy()


def main():
    test_mode_changes_update_in_place()
    test_presets_and_settings_update_in_place()
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
