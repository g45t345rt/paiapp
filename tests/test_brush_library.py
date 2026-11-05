"""Unit tests for the brush preset library
(run: .venv\\Scripts\\python.exe tests\\test_brush_library.py)."""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Point the per-user app folder at a throwaway directory before anything reads
# it (``engine.settings.app_dir`` resolves ``%APPDATA%`` on each call).
_TMP = tempfile.mkdtemp(prefix="pai-brushes-")
os.environ["APPDATA"] = _TMP

from paintapp import brushes  # noqa: E402
from paintapp.brush import Dynamics  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


BRUSH_KEYS = {
    "brush_size", "brush_opacity", "brush_hardness", "brush_flow",
    "brush_spacing", "brush_stabilize", "brush_stabilize_mode",
    "brush_antialias", "brush_blend",
    "brush_tip", "brush_pattern", "brush_velocity_size", "brush_velocity_opacity",
    "brush_velocity_flow", "brush_velocity_hardness", "brush_velocity_spacing",
    "brush_taper_in", "brush_taper_out", "brush_grain", "brush_grain_scale",
    "brush_grain_strength", "brush_grain_mode", "brush_jitter_hue",
    "brush_jitter_sat", "brush_jitter_val", "brush_pickup", "brush_wet_edge",
}
ERASER_KEYS = {
    "brush_size", "eraser_opacity", "eraser_hardness", "eraser_spacing",
    "eraser_stabilize", "eraser_stabilize_mode", "eraser_antialias",
    "eraser_blend", "eraser_target",
    "eraser_color", "eraser_tip", "eraser_pattern", "eraser_velocity_size",
    "eraser_velocity_opacity", "eraser_velocity_flow", "eraser_velocity_hardness",
    "eraser_velocity_spacing", "eraser_taper_in", "eraser_taper_out",
    "eraser_grain", "eraser_grain_scale", "eraser_grain_strength",
    "eraser_grain_mode", "eraser_jitter_hue", "eraser_jitter_sat",
    "eraser_jitter_val", "eraser_pickup",
}


def test_defaults_are_complete():
    presets = brushes.default_presets()
    check("starter set present", len(presets) >= 4, True)
    for preset in presets:
        expected = BRUSH_KEYS if preset.tool == "brush" else ERASER_KEYS
        check(f"{preset.name} keys", set(preset.params) == expected, True)


def test_textured_presets_and_preview():
    presets = {p.name: p for p in brushes.default_presets()}
    check("Charcoal preset exists", "Charcoal" in presets, True)
    check("Charcoal uses paper grain", presets["Charcoal"].params["brush_grain"],
          "paper")
    check("Halftone uses stamp grain",
          presets["Halftone"].params["brush_grain_mode"], "stamp")
    check("Watercolor has pickup", presets["Watercolor"].params["brush_pickup"],
          65)
    check("Watercolor has a wet edge",
          presets["Watercolor"].params["brush_wet_edge"], 55)
    image = brushes.preset_preview_image(presets["Canvas Dry"])
    check("textured preview renders", image.size, (224, 56))


def test_from_dict_rejects_foreign():
    check("foreign format rejected",
          brushes.BrushPreset.from_dict({"format": "nope"}), None)
    check("non-dict rejected", brushes.BrushPreset.from_dict("x"), None)
    check("bad tool rejected",
          brushes.BrushPreset.from_dict(
              {"format": brushes.FORMAT, "tool": "wand"}), None)


def test_round_trip_and_delete():
    preset = brushes.BrushPreset("Test Pen", "brush",
                                 {"brush_size": 12, "brush_taper_in": 40})
    brushes.save_preset(preset)
    loaded = {p.id: p for p in brushes.load_presets()}
    check("saved preset loads", preset.id in loaded, True)
    check("params survive round trip", loaded[preset.id].params,
          {"brush_size": 12, "brush_taper_in": 40})
    check("name survives round trip", loaded[preset.id].name, "Test Pen")
    check("delete removes it", brushes.delete_preset(preset), True)
    check("gone after delete",
          preset.id in {p.id for p in brushes.load_presets()}, False)


def test_ensure_defaults_runs_once():
    brushes.ensure_defaults()
    first = {p.name for p in brushes.load_presets()}
    marker = os.path.join(brushes.library_dir(), brushes.DEFAULTS_MARKER)
    check("marker written", os.path.exists(marker), True)
    # A user-deleted preset must not come back on the next launch.
    hard = next(p for p in brushes.load_presets() if p.name == "Hard Round")
    brushes.delete_preset(hard)
    brushes.ensure_defaults()
    names = {p.name for p in brushes.load_presets()}
    check("deleted default stays deleted", "Hard Round" in names, False)
    check("others remain", "Ink Pen" in names and "Ink Pen" in first, True)


def test_dynamics_for_scales_taper():
    dyn = brushes.dynamics_for(
        {"brush_velocity_size": 50, "brush_taper_in": 100, "brush_taper_out": 50},
        "brush", 20)
    check("dynamics is a Dynamics", isinstance(dyn, Dynamics), True)
    check("velocity size mapped", dyn.size, 0.5)
    check("taper_in scaled by size", dyn.taper_in, 20.0)
    check("taper_out scaled by size", dyn.taper_out, 10.0)


def test_tool_params_round_trip():
    try:
        import tkinter as tk
    except ImportError:
        print("ok  tool params (tk unavailable, skipped)")
        return
    try:
        root = tk.Tk()
    except tk.TclError:
        print("ok  tool params (no display, skipped)")
        return
    root.withdraw()
    from paintapp.drawing import ToolsPanel
    noop = lambda *a, **k: None
    tools = ToolsPanel(root, noop, noop, noop)
    tools.brush_size = 37
    tools.brush_hardness = 42
    tools.brush_taper_in = 25
    tools.brush_velocity_size = 60
    params = tools.tool_params("brush")
    check("params capture size", params["brush_size"], 37)
    check("params capture taper", params["brush_taper_in"], 25)
    # Wipe them, then re-apply and confirm they come back.
    tools.brush_hardness = 100
    tools.brush_taper_in = 0
    tools.brush_velocity_size = 0
    tools.apply_tool_params("brush", params)
    check("apply restores hardness", tools.brush_hardness, 42)
    check("apply restores taper", tools.brush_taper_in, 25)
    check("apply restores velocity", tools.brush_velocity_size, 60)
    check("apply restores size", tools.brush_size, 37)
    check("unknown tool yields no params", tools.tool_params("wand"), {})
    root.destroy()


def test_brushes_panel_start_state():
    from paintapp import constants
    check("brushes is a registered panel",
          "brushes" in [key for key, _title in constants.PANELS], True)
    # Brushes starts collapsed (header only); the user opens it when needed.
    check("brushes starts collapsed",
          "brushes" in constants.DEFAULT_COLLAPSED_PANELS, True)
    check("brushes docks left by default",
          constants.DEFAULT_PANEL_SIDES.get("brushes"), "left")


def _tk_panel():
    """Build a ToolsPanel + BrushLibraryPanel, or ``(None, None, None)``."""
    try:
        import tkinter as tk
    except ImportError:
        return None, None, None
    try:
        root = tk.Tk()
    except tk.TclError:
        return None, None, None
    root.withdraw()
    from paintapp.drawing import ToolsPanel
    from paintapp.brushes import BrushLibraryPanel
    noop = lambda *a, **k: None
    tools = ToolsPanel(root, noop, noop, noop)
    panel = BrushLibraryPanel(root, tools, on_apply=noop)
    return root, tools, panel


def test_more_tools_have_presets():
    root, tools, panel = _tk_panel()
    if root is None:
        print("ok  more tools (no display, skipped)")
        return
    keys = set(tools.tool_params("clone")) - {"brush_size"}
    check("clone exposes its own params", "clone_hardness" in keys, True)
    check("clone excludes brush fields",
          any(k.startswith("brush_") for k in keys), False)
    preset = brushes.BrushPreset("Clone A", "clone", tools.tool_params("clone"))
    brushes.save_preset(preset)
    panel.set_tool("clone")
    check("clone preset listed", preset.name in [p.name for p in panel._presets],
          True)
    brushes.delete_preset(preset)
    root.destroy()


def test_search_filters_the_list():
    root, tools, panel = _tk_panel()
    if root is None:
        print("ok  search (no display, skipped)")
        return
    panel.set_tool("brush")
    total = len(panel._presets)
    panel._search_var.set("ink")
    panel.reload()
    check("search narrows to matches",
          all("ink" in p.name.lower() for p in panel._presets)
          and 0 < len(panel._presets) <= total, True)
    panel._search_var.set("")
    panel.reload()
    check("clearing the search restores the list", len(panel._presets), total)
    root.destroy()


def test_reorder_persists():
    root, tools, panel = _tk_panel()
    if root is None:
        print("ok  reorder (no display, skipped)")
        return
    panel.set_tool("brush")
    original = [p.id for p in panel._presets]
    reversed_ids = list(reversed(original))
    panel._presets = [p for p in reversed(panel._presets)]
    panel._persist_order()  # what a drag-off commits
    panel._selected_id = None
    panel.reload()
    check("dragged order is remembered", [p.id for p in panel._presets],
          reversed_ids)
    root.destroy()


def test_cycle_and_modified():
    root, tools, panel = _tk_panel()
    if root is None:
        print("ok  cycle/modified (no display, skipped)")
        return
    panel.set_tool("brush")
    panel.cycle(1)
    check("cycle selects a preset", panel._selected_id is not None, True)
    tools.brush_hardness = (tools.brush_hardness + 7) % 100 + 1
    panel.refresh_modified()
    check("edited preset is marked modified", panel._modified, True)
    panel._save()  # saving the current settings clears the mark
    check("save clears modified", panel._modified, False)
    root.destroy()


def test_export_import_pack():
    root, tools, panel = _tk_panel()
    if root is None:
        print("ok  brush pack (no display, skipped)")
        return
    path = os.path.join(tempfile.mkdtemp(prefix="pai-pack-"), "pack.zip")
    saved, opened = (brushes.filedialog.asksaveasfilename,
                     brushes.filedialog.askopenfilename)
    brushes.filedialog.asksaveasfilename = lambda **k: path
    brushes.filedialog.askopenfilename = lambda **k: path
    try:
        before = len(brushes.load_presets())
        panel._export_pack()
        check("pack file written", os.path.exists(path), True)
        victim = brushes.load_presets()[0]
        brushes.delete_preset(victim)
        check("preset removed before import",
              len(brushes.load_presets()), before - 1)
        panel._import_pack()
        check("import restores presets", len(brushes.load_presets()), before)
    finally:
        brushes.filedialog.asksaveasfilename = saved
        brushes.filedialog.askopenfilename = opened
    root.destroy()


def test_brushes_list_uses_theme_background():
    """The Brushes list must paint its own background from the theme token: the
    scroll canvas is skipped by ``theme.restyle``, so an unthemed panel shows the
    default Tk white instead of the rail list colour."""
    root, tools, panel = _tk_panel()
    if root is None:
        print("ok  brushes theme (no display, skipped)")
        return
    from paintapp import theme
    # Switch to the dark theme and let the panel recolour, the case that showed
    # the white container; the dark token differs from any Tk default.
    theme.init(root, "dark")
    panel._on_theme()
    expected = theme.color("listbox_bg")
    check("brushes list canvas is themed",
          str(panel._canvas.cget("bg")), expected)
    check("brushes list frame is themed",
          str(panel._list.cget("bg")), expected)
    root.destroy()


def test_restore_defaults_replaces_not_duplicates():
    """Restore re-adds deleted starters and resets edited ones in place, rather
    than piling up duplicate-named presets."""
    brushes.restore_defaults()
    presets = {p.name: p for p in brushes.load_presets() if p.tool == "brush"}
    edited = presets["Hard Round"]
    edited.params["brush_size"] = 99
    brushes.save_preset(edited)
    brushes.delete_preset(presets["Pencil"])

    count = brushes.restore_defaults()
    after = brushes.load_presets()
    names = [p.name for p in after if p.tool == "brush"]
    check("restore rewrites every default", count, len(brushes.DEFAULT_SPECS))
    check("restore re-adds a deleted default", "Pencil" in names, True)
    check("restore does not duplicate defaults", len(names), len(set(names)))
    restored = next(p for p in after if p.name == "Hard Round")
    check("restore resets an edited default",
          restored.params.get("brush_size"), 30)


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
