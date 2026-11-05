"""Unit tests for the Region Prompts panel's per-row mask snapshots.

Run: .venv\\Scripts\\python.exe tests\\test_regions.py

The decode/encode pair is pure and always runs. The panel half builds the real
widget on a withdrawn Tk root and drives capture, storage and reload; it is
skipped (with a note) when no display is available.
"""

import os
import sys
from types import SimpleNamespace

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import regions_panel as regions  # noqa: E402

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def rectangle(size, box):
    mask = Image.new("L", size, 0)
    mask.paste(255, box)
    return mask


def test_encode_decode_round_trip():
    mask = rectangle((16, 16), (2, 3, 9, 11))
    decoded = regions.decode_mask(regions.encode_mask(mask))
    check("round trip type", decoded.mode, "L")
    check("round trip size", decoded.size, (16, 16))
    check("round trip bbox", decoded.getbbox(), (2, 3, 9, 11))
    check("round trip inside", decoded.getpixel((4, 5)), 255)
    check("round trip outside", decoded.getpixel((12, 12)), 0)


def test_decode_rejects_missing():
    check("empty decodes to none", regions.decode_mask(""), None)
    check("none decodes to none", regions.decode_mask(None), None)
    check("garbage decodes to none", regions.decode_mask("not a snapshot"), None)


def _build_panel(captures):
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    calls = []

    def on_capture(source, warn=True):
        calls.append((source, warn))
        return captures.get(source)

    panel = regions.RegionsPanel(root, on_capture=on_capture)
    panel.set_family("stable_diffusion")
    panel.set_enabled(True)
    return root, panel, calls


def test_panel_rows_own_distinct_masks():
    try:
        root, panel, calls = _build_panel({
            regions.MASK_SELECTION: rectangle((8, 8), (0, 0, 4, 4)),
            regions.MASK_PAINTED: rectangle((8, 8), (4, 4, 8, 8)),
        })
    except Exception as exc:  # no display, or Tk missing
        SKIPPED.append(f"panel tests skipped: {exc}")
        return

    try:
        first = panel.add_row(prompt="a cat")
        second = panel.add_row(prompt="a dog")
        panel._capture(first, regions.MASK_SELECTION)
        panel._capture(second, regions.MASK_PAINTED)

        rows = panel.settings()
        check("two rows stored", len(rows), 2)
        check("first prompt", rows[0]["prompt"], "a cat")
        check("first source", rows[0]["source"], regions.MASK_SELECTION)
        check("second source", rows[1]["source"], regions.MASK_PAINTED)
        check("rows hold distinct masks", rows[0]["mask"] != rows[1]["mask"], True)
        check("default strength", rows[0]["strength"], 1.0)

        first["strength_var"].set(0.5)
        check("strength stored", panel.settings()[0]["strength"], 0.5)

        decoded = regions.decode_mask(rows[0]["mask"])
        check("first mask bbox", decoded.getbbox(), (0, 0, 4, 4))
        decoded2 = regions.decode_mask(rows[1]["mask"])
        check("second mask bbox", decoded2.getbbox(), (4, 4, 8, 8))

        # Reloading the stored rows must reproduce the prompts and masks.
        panel.set_settings(panel.settings())
        reloaded = panel.settings()
        check("reload keeps prompts", [r["prompt"] for r in reloaded], ["a cat", "a dog"])
        check("reload keeps strength", reloaded[0]["strength"], 0.5)
        check(
            "reload keeps masks",
            [regions.decode_mask(r["mask"]).getbbox() for r in reloaded],
            [(0, 0, 4, 4), (4, 4, 8, 8)],
        )
    finally:
        root.destroy()


def test_panel_unassigned_row_is_skipped_when_empty():
    try:
        root, panel, calls = _build_panel({
            regions.MASK_SELECTION: rectangle((8, 8), (0, 0, 4, 4)),
        })
    except Exception as exc:
        SKIPPED.append(f"panel tests skipped: {exc}")
        return

    try:
        row = panel.add_row(prompt="")
        check("empty row not stored", panel.settings(), [])
        panel._capture(row, regions.MASK_SELECTION)
        rows = panel.settings()
        check("masked row stored", len(rows), 1)
        check("masked row has mask", regions.decode_mask(rows[0]["mask"]).getbbox(), (0, 0, 4, 4))
        panel._clear_mask(row)
        check("cleared empty row dropped", panel.settings(), [])
    finally:
        root.destroy()


def test_capture_region_mask():
    from paintapp.app import PaintApp

    results = []
    source = rectangle((8, 8), (0, 0, 4, 4))
    fake = SimpleNamespace(
        current=SimpleNamespace(
            has_inpaint_mask=lambda: False,
            inpaint_mask=Image.new("L", (8, 8), 0),
        ),
        selection=SimpleNamespace(is_empty=lambda: False, mask=source),
        set_result=lambda msg, kind="info": results.append((msg, kind)),
    )
    got = PaintApp._capture_region_mask(fake, regions.MASK_SELECTION)
    check("capture selection bbox", got.getbbox(), (0, 0, 4, 4))
    check("capture is a detached copy", got is not source, True)

    painted = SimpleNamespace(
        current=SimpleNamespace(
            has_inpaint_mask=lambda: True,
            inpaint_mask=rectangle((8, 8), (4, 4, 8, 8)),
        ),
        selection=None,
        set_result=lambda msg, kind="info": results.append((msg, kind)),
    )
    got = PaintApp._capture_region_mask(painted, regions.MASK_PAINTED)
    check("capture painted bbox", got.getbbox(), (4, 4, 8, 8))

    empty = SimpleNamespace(
        current=SimpleNamespace(has_inpaint_mask=lambda: False),
        selection=None,
        set_result=lambda msg, kind="info": results.append((msg, kind)),
    )
    check(
        "capture with nothing warns",
        PaintApp._capture_region_mask(empty, regions.MASK_PAINTED),
        None,
    )
    check("capture warns", results[-1][1], "warn")


def test_resolve_region_masks():
    from paintapp.app import PaintApp

    results = []
    fake = SimpleNamespace(
        set_result=lambda msg, kind="info": results.append((msg, kind)),
    )
    rows = [
        {"prompt": "a cat", "mask": regions.encode_mask(rectangle((8, 8), (0, 0, 4, 4)))},
        {"prompt": "a dog", "mask": regions.encode_mask(rectangle((8, 8), (4, 4, 8, 8))),
         "strength": 0.25},
    ]
    out = PaintApp._resolve_region_masks(fake, rows)
    check("resolve returns both regions", len(out), 2)
    check("resolve decodes first", out[0]["mask"].getbbox(), (0, 0, 4, 4))
    check("resolve decodes second", out[1]["mask"].getbbox(), (4, 4, 8, 8))
    check("resolve keeps prompt", out[1]["prompt"], "a dog")
    check("resolve defaults strength", out[0]["strength"], 1.0)
    check("resolve keeps strength", out[1]["strength"], 0.25)

    bad = PaintApp._resolve_region_masks(fake, [{"prompt": "x", "mask": None}])
    check("resolve refuses a maskless row", bad, None)
    check("resolve warns on a maskless row", results[-1][1], "warn")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    for note in SKIPPED:
        print(f"skip {note}")
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()