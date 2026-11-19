"""Checks for the Generate panel's Keywords (weighted) prompt mode.

Builds the real widget on a withdrawn Tk root and drives the two prompt modes:
the comma split into keyword rows, the compiled ``(text:weight)`` prompt for
both positive and negative prompts, adding/removing rows, weight 0 disabling a
row, mode-switch syncing, the settings round-trip, the height caps, and the
optional slider-release regeneration. Skipped (with a note) when no display is
available.

Run: .venv\\Scripts\\python.exe tests\\test_weighted_prompt_ui.py
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


def _build_panel(on_generate=None, show=False):
    import tkinter as tk

    from paintapp.generation import GenerationPanel

    root = tk.Tk()
    if show:
        root.geometry("320x700+2000+2000")
    else:
        root.withdraw()
    panel = GenerationPanel(
        root,
        on_generate=on_generate or (lambda: None),
        on_positive_prompt=lambda _text: None,
        on_settings=lambda _values: None,
    )
    if show:
        panel.pack(fill="both", expand=True)
    return root, panel


def test_keyword_prompt():
    try:
        root, panel = _build_panel()
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"keyword prompt test skipped: {exc}")
        return
    try:
        from paintapp.generation import PROMPT_MODE_DEFAULT, PROMPT_MODE_KEYWORDS

        check("starts default", panel.prompt_mode(), PROMPT_MODE_DEFAULT)
        check("auto-generate is off", panel.auto_generate_var.get(), False)
        check("auto-generate hidden in default",
              panel.auto_generate_check.winfo_manager(), "")
        panel.set_prompt("happy, smile")
        check("default prompt", panel.prompt(), "happy, smile")

        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        check("auto-generate shown in keywords",
              panel.auto_generate_check.winfo_manager(), "pack")
        check("split into rows", [row["text"] for row in panel.weighted_rows()],
              ["happy", "smile"])
        check("weights default to 1", panel.prompt(), "happy, smile")

        panel._weighted_rows[0]["weight_var"].set(2.0)
        check("compiled weight", panel.prompt(), "(happy:2), smile")

        panel._weighted_rows[0]["weight_var"].set(0.0)
        check("zero weight disables the row", panel.prompt(), "smile")
        panel._weighted_rows[0]["weight_var"].set(2.0)

        panel._weighted_rows[0]["text_var"].set("red (hat)")
        check("parentheses escaped", panel.prompt(), "(red \\(hat\\):2), smile")

        panel._add_weighted_row("glow", 0.5)
        check("added row", panel.prompt(),
              "(red \\(hat\\):2), smile, (glow:0.5)")
        panel._remove_weighted_row(panel._weighted_rows[1])
        check("removed row", panel.prompt(), "(red \\(hat\\):2), (glow:0.5)")

        stored = panel.settings()
        check("settings mode", stored.get("prompt_mode"), PROMPT_MODE_KEYWORDS)
        check("settings rows", stored.get("weighted_prompt"), [
            {"text": "red (hat)", "weight": 2.0},
            {"text": "glow", "weight": 0.5},
        ])

        panel.prompt_mode_var.set(PROMPT_MODE_DEFAULT)
        root.update()
        check("auto-generate hidden in default",
              panel.auto_generate_check.winfo_manager(), "")
        check("switching back compiles", panel.prompt(),
              "(red \\(hat\\):2), (glow:0.5)")

        second = _build_panel()[1]
        second.set_settings(stored)
        root.update()
        check("restored mode", second.prompt_mode(), PROMPT_MODE_KEYWORDS)
        check("restored rows", second.weighted_rows(), stored["weighted_prompt"])
    finally:
        root.destroy()


def test_negative_keyword_prompt():
    try:
        root, panel = _build_panel()
    except Exception as exc:
        SKIPPED.append(f"negative keyword test skipped: {exc}")
        return
    try:
        from paintapp.generation import PROMPT_MODE_KEYWORDS

        panel.negative_var.set("blurry, noise")
        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        check("negative split", [row["text"] for row in
                                 panel.weighted_negative_rows()],
              ["blurry", "noise"])
        panel._weighted_negative_rows[0]["weight_var"].set(2.0)
        check("negative compiled", panel.negative_prompt(), "(blurry:2), noise")
        panel._weighted_negative_rows[1]["weight_var"].set(0.0)
        check("negative zero disables", panel.negative_prompt(), "(blurry:2)")

        stored = panel.settings()
        check("negative rows stored", stored.get("weighted_negative"), [
            {"text": "blurry", "weight": 2.0},
            {"text": "noise", "weight": 0.0},
        ])

        panel.prompt_mode_var.set("Default")
        root.update()
        check("negative pushed back", panel.negative_var.get(), "(blurry:2)")
    finally:
        root.destroy()


def test_modes_stay_in_sync():
    try:
        root, panel = _build_panel()
    except Exception as exc:
        SKIPPED.append(f"mode sync test skipped: {exc}")
        return
    try:
        from paintapp.generation import PROMPT_MODE_DEFAULT, PROMPT_MODE_KEYWORDS

        panel.set_prompt("happy, smile")
        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        panel._weighted_rows[0]["weight_var"].set(2.0)

        panel.prompt_mode_var.set(PROMPT_MODE_DEFAULT)
        root.update()
        check("keywords pushed to default", panel.positive_prompt(),
              "(happy:2), smile")

        panel.set_prompt("calm, soft")
        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        check("edited default rebuilds rows",
              [row["text"] for row in panel.weighted_rows()], ["calm", "soft"])
        check("edited default prompt", panel.prompt(), "calm, soft")

        panel.prompt_mode_var.set(PROMPT_MODE_DEFAULT)
        root.update()
        check("rows survive unchanged switch", panel.positive_prompt(), "calm, soft")
    finally:
        root.destroy()


def test_existing_weights_survive_the_switch():
    try:
        root, panel = _build_panel()
    except Exception as exc:
        SKIPPED.append(f"weight switch test skipped: {exc}")
        return
    try:
        from paintapp.generation import PROMPT_MODE_KEYWORDS

        panel.set_prompt("(happy:2), smile")
        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        check("existing weight moved onto the slider", panel.weighted_rows(), [
            {"text": "happy", "weight": 2.0},
            {"text": "smile", "weight": 1.0},
        ])
        check("prompt unchanged by the move", panel.prompt(), "(happy:2), smile")
    finally:
        root.destroy()


def test_list_and_negative_prompt_are_capped():
    try:
        root, panel = _build_panel()
    except Exception as exc:
        SKIPPED.append(f"max-height test skipped: {exc}")
        return
    try:
        from paintapp.generation import (
            NEGATIVE_PROMPT_MAX_LINES, PROMPT_MODE_KEYWORDS,
            WEIGHTED_LIST_MAX_HEIGHT,
        )

        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        for index in range(12):
            panel._add_weighted_row(f"prompt {index}")
        root.update_idletasks()
        canvas = panel.weighted_list.canvas
        check("weighted list capped",
              canvas.winfo_reqheight() <= WEIGHTED_LIST_MAX_HEIGHT, True)
        bbox = canvas.bbox("all")
        check("weighted list content overflows", bbox[3] > canvas.winfo_height(), True)

        for data in list(panel._weighted_rows[1:]):
            panel._remove_weighted_row(data)
        root.update()
        check("scrollbar hides when rows fit",
              panel.weighted_list.scrollbar.winfo_ismapped(), 0)

        for data in list(panel._weighted_rows):
            panel._remove_weighted_row(data)
        root.update()
        check("empty positive list collapses",
              panel.weighted_list.canvas.winfo_reqheight() <= 1, True)
        panel._add_weighted_row("back")
        root.update_idletasks()
        check("add prompt shows the row immediately",
              panel.weighted_list.canvas.winfo_reqheight() > 1, True)

        for index in range(12):
            panel._add_weighted_negative_row(f"bad {index}")
        root.update_idletasks()
        check("negative list capped",
              panel.negative_list.canvas.winfo_reqheight()
              <= 140, True)
        for data in list(panel._weighted_negative_rows):
            panel._remove_weighted_row(data)
        root.update()
        check("empty negative list collapses",
              panel.negative_list.canvas.winfo_reqheight() <= 1, True)
        panel._add_weighted_negative_row("again")
        root.update_idletasks()
        check("add negative prompt shows the row immediately",
              panel.negative_list.canvas.winfo_reqheight() > 1, True)
    finally:
        root.destroy()


def test_negative_prompt_autogrow():
    try:
        root, panel = _build_panel()
    except Exception as exc:
        SKIPPED.append(f"negative autogrow test skipped: {exc}")
        return
    try:
        from paintapp.generation import NEGATIVE_PROMPT_MAX_LINES

        panel.negative_var.set("\n".join(f"line {i}" for i in range(14)))
        root.update()
        height = int(panel.negative_text.cget("height"))
        check("negative prompt capped", height <= NEGATIVE_PROMPT_MAX_LINES, True)
        check("negative prompt grew", height > 3, True)
    finally:
        root.destroy()


def test_slider_release_regenerates_when_enabled():
    calls = []
    try:
        root, panel = _build_panel(
            on_generate=lambda: calls.append("gen"), show=True
        )
    except Exception as exc:
        SKIPPED.append(f"slider release test skipped: {exc}")
        return
    try:
        from paintapp.generation import PROMPT_MODE_DEFAULT, PROMPT_MODE_KEYWORDS

        panel.set_prompt("happy")
        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()
        panel._weighted_rows[0]["weight_var"].set(2.0)
        panel._on_fast_slider_release()
        root.after(500, root.quit)
        root.mainloop()
        check("off by default", calls, [])

        panel.auto_generate_var.set(True)
        panel._weighted_rows[0]["scale"].event_generate("<ButtonRelease-1>")
        root.after(500, root.quit)
        root.mainloop()
        check("one generation per release", calls, ["gen"])

        # In Default mode a slider (which is hidden) must not regenerate.
        panel.prompt_mode_var.set(PROMPT_MODE_DEFAULT)
        root.update()
        panel._on_fast_slider_release()
        root.after(500, root.quit)
        root.mainloop()
        check("default mode does not auto-run", calls, ["gen"])
    finally:
        root.destroy()


def test_seed_input_auto_generates():
    calls = []
    try:
        root, panel = _build_panel(
            on_generate=lambda: calls.append("gen"), show=True
        )
    except Exception as exc:
        SKIPPED.append(f"seed input test skipped: {exc}")
        return
    try:
        from paintapp.generation import PROMPT_MODE_DEFAULT, PROMPT_MODE_KEYWORDS

        panel.set_prompt("happy")
        panel.prompt_mode_var.set(PROMPT_MODE_KEYWORDS)
        root.update()

        panel.seed_var.set(123)
        root.after(500, root.quit)
        root.mainloop()
        check("seed edit ignored while auto is off", calls, [])

        panel.auto_generate_var.set(True)
        panel.seed_var.set(456)
        root.after(500, root.quit)
        root.mainloop()
        check("seed edit regenerates when auto is on", calls, ["gen"])

        panel.prompt_mode_var.set(PROMPT_MODE_DEFAULT)
        root.update()
        panel.seed_var.set(789)
        root.after(500, root.quit)
        root.mainloop()
        check("seed edit ignored in default mode", calls, ["gen"])
    finally:
        root.destroy()


def main():
    test_keyword_prompt()
    test_negative_keyword_prompt()
    test_modes_stay_in_sync()
    test_existing_weights_survive_the_switch()
    test_list_and_negative_prompt_are_capped()
    test_negative_prompt_autogrow()
    test_seed_input_auto_generates()
    test_slider_release_regenerates_when_enabled()
    for note in SKIPPED:
        print(f"skip {note}")
    if FAILED:
        print("\n".join(FAILED))
        print(f"\n{len(FAILED)} weighted prompt check(s) failed")
        return 1
    print("\nall weighted prompt checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
