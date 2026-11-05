"""Unit tests for the blend modes (run: .venv\\Scripts\\python.exe tests\\test_blend_modes.py)."""

import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import brush  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def close(name, got, want, tol=2):
    if abs(got - want) > tol:
        FAILED.append(f"{name}: got {got} want {want} (+/-{tol})")
    else:
        print(f"ok  {name}")


def solid(rgb, size=(1, 1)):
    return Image.new("RGB", size, rgb)


def blended(mode, base, src):
    return brush._blend(mode, solid(base), solid(src)).getpixel((0, 0))


def test_every_mode_resolves():
    base = solid((128, 64, 200), (3, 2))
    src = solid((80, 180, 40), (3, 2))
    for key in brush.BLEND_MODES:
        out = brush._blend(key, base, src)
        check(f"{key} returns an RGB image", out.mode, "RGB")
        check(f"{key} keeps the size", out.size, (3, 2))


def test_normal_returns_the_source():
    check("normal is the source", blended("normal", (10, 20, 30), (200, 100, 50)),
          (200, 100, 50))


def test_difference_and_subtract():
    check("difference is |base-src|", blended("difference", (200, 40, 90),
                                              (60, 100, 90)), (140, 60, 0))
    check("subtract is base-src", blended("subtract", (200, 40, 90),
                                          (60, 100, 40)), (140, 0, 50))


def test_color_dodge_and_burn():
    # dodge = base / (1 - src): 128 / (1 - 0.5) = 255
    check("color dodge lightens past the source", blended(
        "color_dodge", (128, 128, 128), (128, 128, 128)), (255, 255, 255))
    # burn = 1 - (1-base)/src: 1 - 0.5/0.75 = 0.333 -> 85
    close("color burn darkens", blended("color_burn", (128, 128, 128),
                                        (191, 191, 191))[0], 85)
    check("burn of a black source is black", blended(
        "color_burn", (128, 128, 128), (0, 0, 0)), (0, 0, 0))


def test_exclusion():
    # b + s - 2bs at b=0.5, s=0.25 -> 0.5 + 0.25 - 0.25 = 0.5 -> 128
    close("exclusion formula", blended("exclusion", (128, 128, 128),
                                       (64, 64, 64))[0], 128)


def test_luminosity_takes_the_source_lightness():
    # Luminosity keeps base hue/sat and takes source lightness: a dark base
    # blended with a light source should read light.
    out = blended("luminosity", (40, 40, 200), (235, 235, 235))
    check("luminosity brightens toward the source", sum(out) > 640, True)


def test_hue_takes_the_source_hue():
    # A red source tints a blue base red while keeping the base hue-free
    # lightness (which HSL defines as the max/min midpoint, not the mean).
    out = blended("hue", (60, 60, 200), (255, 0, 0))
    check("hue shifts toward red", out[0] > out[1] and out[0] > out[2], True)
    close("hue keeps the base lightness", (max(out) + min(out)) / 2,
          (60 + 200) / 2, 3)


def test_composite_color_uses_the_mode():
    base = Image.new("RGBA", (2, 1), (200, 200, 200, 255))
    coverage = Image.new("L", (2, 1), 255)
    normal = brush.composite_color(base, (40, 40, 40), coverage, "normal")
    multiply = brush.composite_color(base, (40, 40, 40), coverage, "multiply")
    check("normal repaints the colour", normal.getpixel((0, 0))[:3], (40, 40, 40))
    check("multiply darkens past the paint colour",
          multiply.getpixel((0, 0))[0] < 40, True)


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
