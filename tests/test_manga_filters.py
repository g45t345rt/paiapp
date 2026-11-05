"""Unit tests for the manga tone filters (run: .venv\\Scripts\\python.exe tests\\test_manga_filters.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

from paintapp import filters  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def solid(gray):
    return Image.new("RGBA", (32, 32), (gray, gray, gray, 255))


def test_registered():
    check("halftone present", "halftone" in filters.FILTERS, True)
    check("screentone present", "screentone" in filters.FILTERS, True)
    check("halftone category", filters.FILTERS["halftone"]["category"], "manga")
    check("screentone category", filters.FILTERS["screentone"]["category"], "manga")
    check(
        "manga in categories",
        ("manga", "Manga") in filters.CATEGORIES,
        True,
    )
    check(
        "manga menu order",
        filters.filters_in_category("manga"),
        ["halftone", "screentone"],
    )


def test_choice_defaults():
    spec = filters.new_spec("halftone")
    check("halftone shape default", spec["params"]["shape"], "dot")
    spec = filters.new_spec("screentone")
    check("screentone pattern default", spec["params"]["pattern"], "dot")
    check("screentone color default", spec["params"]["color"], "#000000")


def test_normalize_round_trip():
    spec = filters.new_spec("screentone")
    spec["params"]["pattern"] = "cross"
    spec["params"]["color"] = "#336699"
    spec["params"]["strength"] = 40
    loaded = filters.normalize_spec(spec)
    check("type kept", loaded["type"], "screentone")
    check("choice kept", loaded["params"]["pattern"], "cross")
    check("color kept", loaded["params"]["color"], "#336699")
    check("int kept", loaded["params"]["strength"], 40)


def test_normalize_bad_choice_falls_back():
    loaded = filters.normalize_spec(
        {"type": "halftone", "enabled": True, "params": {"shape": "bogus"}}
    )
    check("bogus choice kept as string", loaded["params"]["shape"], "bogus")
    # apply must still work with an unknown shape (falls back to dot).
    out = filters.apply_chain(solid(128), [loaded])
    check("bogus shape applies", out.mode, "RGBA")


def test_halftone_extremes():
    spec = filters.new_spec("halftone")
    white = filters.apply_chain(solid(255), [spec])
    check("halftone keeps white white", white.getpixel((0, 0)), (255, 255, 255, 255))
    black = filters.apply_chain(solid(0), [spec])
    check("halftone makes black black", black.getpixel((0, 0)), (0, 0, 0, 255))
    mid = filters.apply_chain(solid(128), [spec])
    values = {mid.getpixel((x, y)) for x in range(mid.width)
              for y in range(mid.height)}
    check("halftone midtone mixes ink and paper", len(values) > 1, True)


def test_halftone_shapes():
    for shape in ("dot", "line", "cross"):
        spec = filters.new_spec("halftone")
        spec["params"]["shape"] = shape
        out = filters.apply_chain(solid(128), [spec])
        check(f"halftone shape {shape} runs", out.size, (32, 32))


def test_screentone_gating():
    spec = filters.new_spec("screentone")
    spec["params"]["color"] = "#ff0000"
    white = filters.apply_chain(solid(255), [spec])
    check("screentone leaves white untouched", white.getpixel((0, 0)),
          (255, 255, 255, 255))
    black = filters.apply_chain(solid(0), [spec])
    values = {black.getpixel((x, y)) for x in range(black.width)
              for y in range(black.height)}
    check("screentone tints black areas red", (255, 0, 0, 255) in values, True)


def test_screentone_patterns():
    for pattern in ("dot", "line", "cross", "grid"):
        spec = filters.new_spec("screentone")
        spec["params"]["pattern"] = pattern
        out = filters.apply_chain(solid(90), [spec])
        check(f"screentone pattern {pattern} runs", out.size, (32, 32))


def test_describe():
    spec = filters.new_spec("halftone")
    check("halftone describes", filters.describe(spec).startswith("Halftone"), True)
    spec = filters.new_spec("screentone")
    check("screentone describes", filters.describe(spec).startswith("Screentone"), True)


def test_gpu_screens_match_cpu():
    import numpy as np
    from paintapp import gpuimage
    if not gpuimage.available():
        print("ok  screens gpu unavailable (skipped)")
        return
    rows = np.linspace(0, 255, 40, dtype=np.uint8)[:, None]
    cols = np.linspace(0, 255, 48, dtype=np.uint8)[None, :]
    arr = np.empty((40, 48, 4), dtype=np.uint8)
    arr[..., 0] = rows
    arr[..., 1] = cols
    arr[..., 2] = 128
    arr[..., 3] = 200
    source = Image.fromarray(arr, "RGBA")

    def compare(spec):
        gpu_out = np.asarray(filters.apply_chain(source, [spec]), dtype=int)
        original = filters.gpuimage.available
        filters.gpuimage.available = lambda: False
        try:
            cpu_out = np.asarray(filters.apply_chain(source, [spec]), dtype=int)
        finally:
            filters.gpuimage.available = original
        difference = np.abs(gpu_out - cpu_out)
        fraction = float((difference.max(axis=2) > 0).mean())
        return fraction, float(difference.mean())

    # numpy evaluates the screen coordinates in float64 while GLSL is float32, so
    # the two can round to opposite sides of a mask edge. That only ever flips a
    # boundary pixel; the body of the screen must agree.
    for shape in ("dot", "line", "cross"):
        spec = filters.new_spec("halftone")
        spec["params"]["shape"] = shape
        fraction, mean = compare(spec)
        check(f"halftone {shape} gpu matches cpu ({fraction:.3f} edge)",
              fraction < 0.05 and mean < 1.0, True)
    for pattern in ("dot", "line", "cross", "grid"):
        spec = filters.new_spec("screentone")
        spec["params"]["pattern"] = pattern
        spec["params"]["color"] = "#3366cc"
        spec["params"]["strength"] = 70
        fraction, mean = compare(spec)
        check(f"screentone {pattern} gpu matches cpu ({fraction:.3f} edge)",
              fraction < 0.05 and mean < 1.0, True)


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
