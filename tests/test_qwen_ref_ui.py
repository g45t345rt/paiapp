"""Checks for the References panel's reference rows.

Builds the real widget on a withdrawn Tk root and drives the rows: adding,
removing, ordering, the assign-selected capture, the settings round-trip and the
family gate (rows only read back for their own family). Skipped (with a note)
when no display is available.

Run: .venv\\Scripts\\python.exe tests\\test_qwen_ref_ui.py
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


def _build_panel(family="qwen_image", capture=None):
    import tkinter as tk

    from paintapp.references import ReferencesPanel

    root = tk.Tk()
    root.withdraw()
    panel = ReferencesPanel(root, on_settings=lambda _values: None, on_capture=capture)
    panel.set_family(family)
    return root, panel


def test_reference_rows():
    try:
        root, panel = _build_panel()
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"panel test skipped: {exc}")
        return
    try:
        panel._add_qwen_reference({"source": "asset", "asset": 1, "label": "Asset: a"})
        panel._add_qwen_reference({"source": "canvas", "label": "Canvas"})
        panel._add_qwen_reference({"source": "layer", "layer": "L1", "label": "Layer: L1"})
        check("added in order", panel.qwen_reference_bindings(), [
            {"source": "asset", "asset": 1, "label": "Asset: a"},
            {"source": "canvas", "label": "Canvas"},
            {"source": "layer", "layer": "L1", "label": "Layer: L1"},
        ])

        rows = list(panel._qwen_refs)
        panel._remove_ref_row(rows[1])
        check("removed middle", panel.qwen_reference_bindings(), [
            {"source": "asset", "asset": 1, "label": "Asset: a"},
            {"source": "layer", "layer": "L1", "label": "Layer: L1"},
        ])

        panel._set_rows(panel._qwen_refs, panel.qref_rows, [
            {"source": "canvas", "label": "Canvas"},
            {"source": "asset", "asset": 2, "label": "Asset: b"},
        ])
        check("set order", panel.qwen_reference_bindings(), [
            {"source": "canvas", "label": "Canvas"},
            {"source": "asset", "asset": 2, "label": "Asset: b"},
        ])
        check("settings carries refs", panel.settings().get("qwen_refs"), [
            {"source": "canvas", "label": "Canvas"},
            {"source": "asset", "asset": 2, "label": "Asset: b"},
        ])

        # Legacy string entries (old projects) become live-kind bindings.
        panel._set_rows(panel._qwen_refs, panel.qref_rows, ["Canvas", "Selected Asset"])
        check("legacy strings", panel.qwen_reference_bindings(),
              [{"source": "canvas"}, {"source": "asset"}])
    finally:
        root.destroy()


def test_assign_selected():
    captured = []

    def capture(kind):
        captured.append(kind)
        if kind == "asset":
            return {"source": "asset", "asset": 10, "label": "Asset: face"}
        return {"source": kind, "label": kind.title()}

    try:
        root, panel = _build_panel(capture=capture)
    except Exception as exc:
        SKIPPED.append(f"assign test skipped: {exc}")
        return
    try:
        panel._add_qwen_reference()
        row = panel._qwen_refs[0]
        check("default kind", row["kind_var"].get(), "Selected Asset")
        panel._assign_row(row)
        check("captured kind", captured[-1], "asset")
        check("binding stored", panel.qwen_reference_bindings()[0].get("asset"), 10)

        # Two rows can hold two different assets.
        panel._add_qwen_reference()
        row2 = panel._qwen_refs[1]
        panel._assign_row(row2)
        check("two distinct bindings", [b.get("asset") for b in
                                        panel.qwen_reference_bindings()], [10, 10])

        # Changing a row's kind clears the captured image until re-assigned.
        row2["kind_var"].set("Canvas")
        check("kind change clears", panel.qwen_reference_bindings()[1],
              {"source": "canvas"})
    finally:
        root.destroy()


def test_family_gate():
    try:
        root, panel = _build_panel(family="anima")
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"family gate test skipped: {exc}")
        return
    try:
        panel._add_qwen_reference({"source": "canvas", "label": "Canvas"})
        panel._add_anima_reference({"source": "canvas", "label": "Canvas"})

        # The anima family folds only the Anima keys into the run config.
        config = panel.apply_config({})
        check("anima applies refs", "anima_ref_lora" in config, True)
        check("anima skips qwen", "qwen_refs" in config, False)

        # The qwen family folds only the Qwen refs.
        panel.set_family("qwen_image")
        config = panel.apply_config({})
        check("qwen applies refs", config.get("qwen_refs"),
              [{"source": "canvas", "label": "Canvas"}])
        check("qwen skips anima", "anima_ref_lora" in config, False)

        # Both lists are stored regardless of the active family.
        check("anima bindings persist", panel.reference_bindings(),
              [{"source": "canvas", "label": "Canvas"}])
    finally:
        root.destroy()


def test_collapsible_sections():
    try:
        root, panel = _build_panel(family="anima")
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"section test skipped: {exc}")
        return
    try:
        # The active family's section opens; the other two fold to headers.
        check("anima open", panel._anima_section.open, True)
        check("qwen folded", panel._qwen_section.open, False)
        check("anypaint folded", panel._anypaint_section.open, False)

        # The header rides the reference count, so a folded section still says
        # what it holds.
        panel._add_anima_reference({"source": "canvas", "label": "Canvas"})
        check("count in header", "(1)" in panel._anima_section.title_var.get(), True)

        # A manual collapse hides the body but keeps the header first in its
        # frame, so expanding later restores the order.
        panel._anima_section.toggle_open()
        check("manual collapse", panel._anima_section.open, False)
        check("body hidden", panel._anima_section.body.winfo_manager(), "")
        check("header stays first",
              panel._anima_section.frame.pack_slaves()[0] is
              panel._anima_section.header, True)
        panel._anima_section.toggle_open()
        check("manual expand", panel._anima_section.open, True)
        check("body shown", panel._anima_section.body.winfo_manager(), "pack")

        # Switching to another family moves the open section.
        panel.set_family("qwen_image")
        check("qwen open", panel._qwen_section.open, True)
        check("anima folded", panel._anima_section.open, False)

        # A repeated set_family for the same family keeps a manual choice.
        panel._qwen_section.toggle_open()
        panel.set_family("qwen_image")
        check("manual choice kept", panel._qwen_section.open, False)
    finally:
        root.destroy()


def test_stitch_side_by_side():
    from PIL import Image

    from paintapp.app import _stitch_side_by_side

    a = Image.new("RGB", (40, 100), (255, 0, 0))
    b = Image.new("RGB", (60, 100), (0, 0, 255))
    check("stitch width", _stitch_side_by_side([a, b], gap=8).size, (108, 100))
    # A shorter image is scaled up to the tallest one's height.
    c = Image.new("RGB", (30, 50), (0, 255, 0))
    out = _stitch_side_by_side([a, c], gap=0)
    check("stitch common height", out.size[1], 100)
    check("stitch scaled width", out.size[0], 40 + 60)
    check("single passes through", _stitch_side_by_side([a]).size, (40, 100))


def main():
    for test in (test_reference_rows, test_assign_selected, test_family_gate,
                 test_collapsible_sections, test_stitch_side_by_side):
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
    print("\nall qwen reference UI tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
