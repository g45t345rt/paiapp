"""Unit tests for the shared GPU ops (run: .venv\\Scripts\\python.exe tests\\test_gpuops.py)."""

import os
import sys

import numpy as np
from PIL import Image, ImageChops, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import gpuops  # noqa: E402
from paintapp import gpuimage  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def random_mask(width=97, height=73, seed=0):
    rng = np.random.default_rng(seed)
    values = (rng.random((height, width)) < 0.5).astype(np.uint8) * 255
    # A blob plus an edge-touching run, so border handling is exercised too.
    values[10:40, 20:60] = 255
    values[0, :] = 255
    return Image.fromarray(values, "L")


def matching(gpu, pil):
    return np.array_equal(np.asarray(gpu), np.asarray(pil))


def test_morphology_available():
    check("gl available", gpuimage.available(), True)


def test_dilate_matches_pil_maxfilter():
    mask = random_mask()
    for radius in (1, 2, 3, 5):
        gpu = gpuops.dilate(mask, radius)
        if gpu is None:
            print("ok  dilate skipped (no gl)")
            return
        pil = mask.filter(ImageFilter.MaxFilter(2 * radius + 1))
        check(f"dilate r={radius} matches PIL MaxFilter", matching(gpu, pil), True)


def test_erode_matches_pil_minfilter():
    mask = random_mask(seed=1)
    for radius in (1, 2, 3, 5):
        gpu = gpuops.erode(mask, radius)
        if gpu is None:
            print("ok  erode skipped (no gl)")
            return
        pil = mask.filter(ImageFilter.MinFilter(2 * radius + 1))
        check(f"erode r={radius} matches PIL MinFilter", matching(gpu, pil), True)


def test_zero_radius_is_none():
    check("radius 0 has no work", gpuops.morphology(random_mask(), 0, "min"), None)


def test_selection_grow_shrink_match_pil():
    from paintapp import selection
    mask = random_mask(seed=2)
    base = selection.Selection.from_mask(mask.size, mask)
    for radius in (1, 2, 3):
        grown = base.grown(radius).mask
        shrunk = base.shrunk(radius).mask
        check(f"grown r={radius} matches PIL",
              matching(grown, mask.filter(ImageFilter.MaxFilter(2 * radius + 1))), True)
        check(f"shrunk r={radius} matches PIL",
              matching(shrunk, mask.filter(ImageFilter.MinFilter(2 * radius + 1))), True)


def test_fill_gap_matches_cpu_fallback():
    from paintapp import fill
    image = random_mask(seed=3).convert("RGBA")
    for gap in (1, 2, 3):
        gpu = fill.region_mask(image, (30, 20), tolerance=10, contiguous=True, gap=gap)
        if gpu is None:
            print("ok  fill gap skipped (no gl)")
            return
        original = gpuops.gpuimage.available
        gpuops.gpuimage.available = lambda: False
        try:
            cpu = fill.region_mask(image, (30, 20), tolerance=10, contiguous=True, gap=gap)
        finally:
            gpuops.gpuimage.available = original
        check(f"fill gap={gap} matches cpu", matching(gpu, cpu), True)


def _mean_abs(a, b):
    return float(np.abs(np.asarray(a, "i2") - np.asarray(b, "i2")).mean())


def test_brush_dab_matches_the_cpu_path():
    from paintapp import brush
    for size, hardness, tip in ((96, 100, "round"), (80, 35, "round"),
                                (72, 100, "square")):
        cpu = brush._build_dab(size, hardness, True, tip)
        gpu = gpuops.brush_dab(size, hardness, True, tip)
        check(f"gpu dab rendered ({size})", gpu is not None, True)
        if gpu is None:
            continue
        check(f"gpu dab size ({size})", gpu.size, cpu.size)
        mid = size // 2
        check(f"gpu dab centre is solid ({size})",
              gpu.getpixel((mid, mid)) >= 250, True)
        if tip == "round":
            check(f"round dab corner is empty ({size})", gpu.getpixel((0, 0)), 0)
        check(f"gpu dab matches the cpu path ({size})",
              _mean_abs(cpu, gpu) < 2.0, True)


def test_grain_overlay_matches_the_cpu_path():
    from paintapp import brush
    tile = brush.grain_tile("canvas", 0.7, 1.0)
    rng = np.random.default_rng(3)
    for w, h, ox, oy in ((64, 64, 13, 29), (97, 73, 5, 9), (256, 256, 300, -40)):
        cov = Image.fromarray((rng.random((h, w)) * 255).astype("uint8"), "L")
        cpu = ImageChops.multiply(cov, brush._grain_region(tile, ox, oy, w, h))
        gpu = gpuops.grain_overlay(cov, tile, ox, oy)
        check(f"grain gpu rendered ({w}x{h})", gpu is not None, True)
        if gpu is None:
            continue
        check(f"grain gpu matches cpu ({w}x{h})", _mean_abs(cpu, gpu) < 1.0, True)


def test_large_dab_uses_the_gpu():
    from paintapp import brush
    dab = brush.make_dab(128, 100, True, "round")
    check("large dab size", dab.size, (128, 128))
    small = brush.make_dab(16, 100, True, "round")
    check("small dab size", small.size, (16, 16))


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
