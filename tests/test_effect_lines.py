"""Unit tests for the brush effect-line patterns (run: .venv\\Scripts\\python.exe tests\\test_effect_lines.py)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import brush  # noqa: E402
from paintapp.document import Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def ink_total(image):
    """Total non-zero pixels in an ``"L"`` image."""
    return sum(image.histogram()[1:])


def stroke_alpha(pattern, size=16):
    layer = Layer("t", (64, 64), (0, 0, 0, 0))
    stroke = brush.Stroke(
        layer, "brush", size=size, hardness=100, flow=100, opacity=100,
        spacing=15, antialias=True, tip="Round", pattern=pattern,
        color=(0, 0, 0),
    )
    stroke.begin(8, 32)
    stroke.move_to(56, 32)
    return layer.image.getchannel("A")


def test_patterns_registered():
    for key in ("none", "speed", "focus", "flash", "sparkle"):
        check(f"pattern {key} registered", key in brush.EFFECT_PATTERNS, True)


def test_effect_dab_shape():
    for pattern in ("speed", "focus", "flash", "sparkle"):
        dab = brush.make_effect_dab(24, pattern, 45, 100, True)
        check(f"{pattern} dab size", dab.size, (24, 24))
        check(f"{pattern} dab mode", dab.mode, "L")
        check(f"{pattern} dab has ink", ink_total(dab) > 0, True)


def test_effect_patterns_differ():
    speed = brush.make_effect_dab(24, "speed", 0, 100, False).tobytes()
    flash = brush.make_effect_dab(24, "flash", 0, 100, False).tobytes()
    sparkle = brush.make_effect_dab(24, "sparkle", 0, 100, False).tobytes()
    check("speed differs from flash", speed != flash, True)
    check("flash differs from sparkle", flash != sparkle, True)


def test_unknown_pattern_falls_back():
    dab = brush.make_effect_dab(16, "bogus", 0, 100, True)
    check("bogus pattern still yields a dab", ink_total(dab) > 0, True)


def test_cache_reuses_same_angle_step():
    first = brush.make_effect_dab(24, "speed", 45, 100, True)
    same = brush.make_effect_dab(24, "speed", 47, 100, True)
    later = brush.make_effect_dab(24, "speed", 90, 100, True)
    check("nearby angles share a cached dab", first is same, True)
    check("distant angle is a different dab", first is later, False)


def test_stroke_defaults_to_none():
    layer = Layer("t", (16, 16), (0, 0, 0, 0))
    stroke = brush.Stroke(
        layer, "brush", size=8, hardness=100, flow=100, opacity=100,
        spacing=25, antialias=True, tip="Round",
    )
    check("default pattern is none", stroke.pattern, "none")


def test_each_pattern_paints():
    for pattern in ("none", "speed", "focus", "flash", "sparkle"):
        alpha = stroke_alpha(pattern)
        check(f"{pattern} paints pixels", ink_total(alpha) > 0, True)


def test_effect_differs_from_plain_brush():
    plain = stroke_alpha("none").tobytes()
    for pattern in ("speed", "focus", "flash", "sparkle"):
        other = stroke_alpha(pattern).tobytes()
        check(f"{pattern} differs from plain", other != plain, True)


def test_focus_angle_points_at_origin():
    layer = Layer("t", (64, 64), (0, 0, 0, 0))
    stroke = brush.Stroke(
        layer, "brush", size=10, hardness=100, flow=100, opacity=100,
        spacing=25, antialias=True, tip="Round", pattern="focus",
    )
    stroke.begin(8, 32)
    angle = stroke._stamp_angle(56, 32, 1, 0)
    # The origin lies to the left, so the focus lines point left (180 degrees).
    check("focus points at origin", round(abs(angle)), 180)


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
