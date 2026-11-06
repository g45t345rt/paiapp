"""Tests for the layer blend modes (run: .venv\\Scripts\\python.exe tests\\test_layer_blend_modes.py).

Every mode offered in the layer menu must actually composite, over both an
opaque backdrop and a transparent one. Blending over transparency used to read
as black for the modes that mix with the backdrop (multiply, darken, overlay,
subtract, soft_light, ...), because a bare region was treated as a black RGB
pixel; it must fall back to the layer's own color there, matching the tools.

The known-answer tests below compute each mode's result independently from its
published formula instead of trusting ``brush._blend`` or Pillow, so a mode that
silently falls back to "normal" (or blends with the wrong operand) fails here.
"""

import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import brush, document  # noqa: E402

FAILED = []

BASE = (200, 100, 50)
SRC = (60, 180, 40)


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def composite(mode, backdrop):
    """Paste one full-opacity layer with ``mode`` over ``backdrop`` (RGBA)."""
    layer = document.Layer("l", backdrop.size, (*SRC, 255))
    layer.blend = mode
    out = backdrop.copy()
    document._paste_layer(out, layer, 0, 0)
    return out.getpixel((0, 0))


def composite_pair(mode, base, src):
    """Blend ``src`` over opaque ``base`` with ``mode``; return RGB."""
    backdrop = Image.new("RGBA", (2, 2), (*base, 255))
    layer = document.Layer("l", (2, 2), (*src, 255))
    layer.blend = mode
    document._paste_layer(backdrop, layer, 0, 0)
    return backdrop.getpixel((0, 0))[:3]


def close(got, want, tol=2):
    return all(abs(g - w) <= tol for g, w in zip(got, want))


def hsl(rgb):
    """Independent RGB -> HSL (all components 0..1)."""
    r, g, b = (c / 255.0 for c in rgb)
    top, bottom = max(r, g, b), min(r, g, b)
    light = (top + bottom) * 0.5
    delta = top - bottom
    if delta < 1e-9:
        return 0.0, 0.0, light
    sat = delta / (1.0 - abs(2.0 * light - 1.0))
    if top == r:
        hue = ((g - b) / delta) % 6.0
    elif top == g:
        hue = (b - r) / delta + 2.0
    else:
        hue = (r - g) / delta + 4.0
    return (hue / 6.0) % 1.0, sat, light


def _luminance(rgb):
    """Rec.709 luminance of an RGB triple (matches ``brush._luminance``)."""
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def expected_math(mode, base, src):
    """Published blend formula, computed in 0..255 with integer rounding."""
    if mode in ("darker_color", "lighter_color"):
        # Whole-pixel choice by luminance, not a per-channel formula.
        if mode == "darker_color":
            return base if _luminance(base) <= _luminance(src) else src
        return base if _luminance(base) >= _luminance(src) else src
    a = [c / 255.0 for c in base]
    b = [c / 255.0 for c in src]
    if mode == "multiply":
        out = [x * y for x, y in zip(a, b)]
    elif mode == "screen":
        out = [1.0 - (1.0 - x) * (1.0 - y) for x, y in zip(a, b)]
    elif mode == "darken":
        out = [min(x, y) for x, y in zip(a, b)]
    elif mode == "lighten":
        out = [max(x, y) for x, y in zip(a, b)]
    elif mode == "difference":
        out = [abs(x - y) for x, y in zip(a, b)]
    elif mode == "add":
        out = [min(1.0, x + y) for x, y in zip(a, b)]
    elif mode == "subtract":
        out = [max(0.0, x - y) for x, y in zip(a, b)]
    elif mode == "exclusion":
        out = [x + y - 2.0 * x * y for x, y in zip(a, b)]
    elif mode == "color_dodge":
        out = [min(1.0, x / max(1e-6, 1.0 - y)) for x, y in zip(a, b)]
    elif mode == "color_burn":
        out = [max(0.0, 1.0 - (1.0 - x) / max(1e-6, y)) for x, y in zip(a, b)]
    elif mode == "overlay":
        out = [2.0 * x * y if x < 0.5 else 1.0 - 2.0 * (1.0 - x) * (1.0 - y)
               for x, y in zip(a, b)]
    elif mode == "hard_light":
        out = [2.0 * x * y if y < 0.5 else 1.0 - 2.0 * (1.0 - x) * (1.0 - y)
               for x, y in zip(a, b)]
    elif mode == "linear_burn":
        out = [max(0.0, x + y - 1.0) for x, y in zip(a, b)]
    elif mode == "linear_light":
        out = [max(0.0, min(1.0, x + 2.0 * y - 1.0)) for x, y in zip(a, b)]
    elif mode == "vivid_light":
        def vivid(x, y):
            if y <= 0.5:
                v = 1.0 - (1.0 - x) / max(1e-6, 2.0 * y)
            else:
                v = x / max(1e-6, 2.0 - 2.0 * y)
            return max(0.0, min(1.0, v))
        out = [vivid(x, y) for x, y in zip(a, b)]
    elif mode == "pin_light":
        out = [max(0.0, min(1.0,
                              min(x, 2.0 * y) if y <= 0.5 else max(x, 2.0 * y - 1.0)))
               for x, y in zip(a, b)]
    elif mode == "hard_mix":
        out = [1.0 if x + y >= 1.0 else 0.0 for x, y in zip(a, b)]
    elif mode == "divide":
        out = [max(0.0, min(1.0, x / max(1e-6, y))) for x, y in zip(a, b)]
    else:
        raise ValueError(mode)
    return tuple(int(round(v * 255.0)) for v in out)


# Exact-formula modes, checked against the published math.
MATH_MODES = (
    "multiply", "screen", "darken", "lighten", "difference", "add", "subtract",
    "exclusion", "color_dodge", "color_burn", "overlay", "hard_light",
    "linear_burn", "linear_light", "vivid_light", "pin_light", "hard_mix",
    "divide", "darker_color", "lighter_color",
)


def test_every_menu_mode_has_an_implementation():
    """No mode in the menu may silently fall back to normal (which would make
    the menu entry a no-op)."""
    for mode, _label in document.BLEND_MODES:
        check(f"{mode} is a known blend id", document.normalize_blend(mode), mode)


def test_modes_change_the_composite():
    """Each non-normal mode must alter the result over an opaque backdrop."""
    opaque = Image.new("RGBA", (2, 2), (*BASE, 255))
    for mode, _label in document.BLEND_MODES:
        got = composite(mode, opaque)
        if mode == "normal":
            check("normal shows the source", got[:3], SRC)
        elif mode in MATH_MODES and expected_math(mode, BASE, SRC) == SRC:
            # Darker/Lighter Color legitimately equal "normal" when the source
            # is already the darker/lighter pixel; a silent fallback is
            # indistinguishable for this input, so the formula tests cover it.
            continue
        else:
            check(f"{mode} differs from normal", got[:3] != SRC, True)


def test_each_math_mode_matches_its_formula():
    """Each deterministic mode must produce the value its formula predicts."""
    for mode in MATH_MODES:
        want = expected_math(mode, BASE, SRC)
        got = composite_pair(mode, BASE, SRC)
        check(f"{mode} matches formula {want}", close(got, want), True)


def test_each_math_mode_identity_points():
    """Black/white operands pin each mode to an unambiguous result."""
    identities = [
        # (mode, base, src, expected RGB)
        ("multiply", (255, 255, 255), (0, 0, 0), (0, 0, 0)),
        ("multiply", (255, 255, 255), (128, 128, 128), (128, 128, 128)),
        ("screen", (0, 0, 0), (255, 255, 255), (255, 255, 255)),
        ("darken", (200, 100, 50), (0, 0, 0), (0, 0, 0)),
        ("lighten", (200, 100, 50), (255, 255, 255), (255, 255, 255)),
        ("difference", (200, 200, 200), (200, 200, 200), (0, 0, 0)),
        ("add", (200, 100, 50), (100, 200, 250), (255, 255, 255)),
        ("subtract", (50, 40, 30), (100, 100, 100), (0, 0, 0)),
        ("color_dodge", (200, 100, 50), (0, 0, 0), BASE),
        ("color_dodge", (200, 100, 50), (255, 255, 255), (255, 255, 255)),
        ("color_burn", (200, 100, 50), (255, 255, 255), BASE),
        ("color_burn", (200, 100, 50), (0, 0, 0), (0, 0, 0)),
        # New standard modes.
        ("linear_burn", (255, 255, 255), (255, 255, 255), (255, 255, 255)),
        ("linear_burn", (0, 0, 0), (255, 255, 255), (0, 0, 0)),
        ("linear_light", (0, 0, 0), (255, 255, 255), (255, 255, 255)),
        ("linear_light", (200, 100, 50), (0, 0, 0), (0, 0, 0)),
        ("vivid_light", (200, 100, 50), (255, 255, 255), (255, 255, 255)),
        ("vivid_light", (200, 100, 50), (0, 0, 0), (0, 0, 0)),
        ("pin_light", (200, 100, 50), (0, 0, 0), (0, 0, 0)),
        ("pin_light", (200, 100, 50), (255, 255, 255), (255, 255, 255)),
        ("hard_mix", (255, 255, 255), (0, 0, 0), (255, 255, 255)),
        ("hard_mix", (127, 127, 127), (127, 127, 127), (0, 0, 0)),
        ("hard_mix", (128, 128, 128), (128, 128, 128), (255, 255, 255)),
        ("divide", (200, 100, 50), (255, 255, 255), BASE),
        # Darker/Lighter Color take the whole pixel, not a per-channel mix.
        ("darker_color", (60, 60, 60), (200, 200, 200), (60, 60, 60)),
        ("darker_color", (200, 200, 200), (60, 60, 60), (60, 60, 60)),
        ("lighter_color", (60, 60, 60), (200, 200, 200), (200, 200, 200)),
        ("lighter_color", (200, 200, 200), (60, 60, 60), (200, 200, 200)),
    ]
    for mode, base, src, want in identities:
        got = composite_pair(mode, base, src)
        check(f"{mode} identity base={base} src={src}", close(got, want), True)


def test_soft_light_direction():
    """Soft Light must lighten with a white source and darken with a black one."""
    for base in ((60, 60, 60), (150, 120, 90), (230, 220, 200)):
        white = composite_pair("soft_light", base, (255, 255, 255))
        black = composite_pair("soft_light", base, (0, 0, 0))
        check(f"soft_light {base} lightens toward white",
              all(w >= b - 2 for w, b in zip(white, base)), True)
        check(f"soft_light {base} darkens toward black",
              all(b <= s + 2 for b, s in zip(black, base)), True)


def test_hsl_modes_take_the_expected_channel():
    """Hue/Saturation/Color/Luminosity each replace only the named channel."""
    base, src = BASE, SRC
    base_h, base_s, base_l = hsl(base)
    src_h, src_s, src_l = hsl(src)

    def near(a, b, tol=0.03):
        d = abs(a - b)
        return min(d, 1.0 - d) <= tol

    checks = [
        ("hue", "h", src_h, "s", base_s, "l", base_l),
        ("saturation", "h", base_h, "s", src_s, "l", base_l),
        ("color", "h", src_h, "s", src_s, "l", base_l),
        ("luminosity", "h", base_h, "s", base_s, "l", src_l),
    ]
    for mode, _hn, want_h, _sn, want_s, _ln, want_l in checks:
        got = composite_pair(mode, base, src)
        gh, gs, gl = hsl(got)
        check(f"{mode} takes hue {want_h:.3f}", near(gh, want_h), True)
        check(f"{mode} takes saturation {want_s:.3f}", near(gs, want_s), True)
        check(f"{mode} takes lightness {want_l:.3f}", near(gl, want_l), True)


def test_blend_over_transparency_keeps_the_layer_color():
    """Over a fully transparent backdrop there is nothing to blend against, so
    every mode must show the layer's own pixels instead of a blended black."""
    clear = Image.new("RGBA", (2, 2), (0, 0, 0, 0))
    for mode, _label in document.BLEND_MODES:
        check(f"{mode} over transparency", composite(mode, clear)[:3], SRC)


def test_partially_transparent_backdrop_still_blends():
    """A backdrop pixel with alpha 1..254 still has a color to blend against.

    Only alpha 0 has no color. Passing the raw alpha to ``Image.composite`` used
    to linearly mix the layer's own color into these pixels, so an opaque layer
    over a half-transparent backdrop came out washed out instead of the pure
    blend result.
    """
    for alpha in (1, 64, 128, 200):
        for mode in MATH_MODES:
            # Rebuild each time: _paste_layer mutates the backdrop in place.
            backdrop = Image.new("RGBA", (2, 2), (*BASE, alpha))
            layer = document.Layer("l", (2, 2), (*SRC, 255))
            layer.blend = mode
            document._paste_layer(backdrop, layer, 0, 0)
            got = backdrop.getpixel((0, 0))
            want = expected_math(mode, BASE, SRC)
            check(f"{mode} over alpha={alpha} uses blend color", close(got[:3], want), True)
            check(f"{mode} over alpha={alpha} keeps opaque alpha", got[3], 255)


def test_partial_layer_alpha_mixes_with_the_blend():
    """A half-transparent blended layer is the average of blend and backdrop."""
    for mode in MATH_MODES:
        backdrop = Image.new("RGBA", (2, 2), (*BASE, 255))
        layer = document.Layer("l", (2, 2), (*SRC, 128))
        layer.blend = mode
        document._paste_layer(backdrop, layer, 0, 0)
        got = backdrop.getpixel((0, 0))[:3]
        want = tuple(round(m * 128 / 255 + b * (255 - 128) / 255)
                     for m, b in zip(expected_math(mode, BASE, SRC), BASE))
        check(f"{mode} with 50% layer alpha averages with base", close(got, want, 2), True)
    base = Image.new("RGB", (2, 2), BASE)
    src = Image.new("RGB", (2, 2), SRC)
    for mode in ("hue", "saturation", "color", "luminosity"):
        backdrop = Image.new("RGBA", (2, 2), (*BASE, 255))
        layer = document.Layer("l", (2, 2), (*SRC, 128))
        layer.blend = mode
        document._paste_layer(backdrop, layer, 0, 0)
        got = backdrop.getpixel((0, 0))[:3]
        blended = brush._blend(mode, base, src).getpixel((0, 0))
        want = tuple(round(m * 128 / 255 + b * (255 - 128) / 255)
                     for m, b in zip(blended, BASE))
        check(f"{mode} with 50% layer alpha averages with base", close(got, want, 2), True)


def test_extra_modes_match_the_brush_implementation():
    """The layer-only modes reuse ``brush._blend``, so both paths must agree."""
    base = Image.new("RGB", (2, 2), BASE)
    src = Image.new("RGB", (2, 2), SRC)
    for mode in ("color_dodge", "color_burn", "exclusion", "linear_burn",
                 "linear_light", "vivid_light", "pin_light", "hard_mix",
                 "divide", "darker_color", "lighter_color", "hue",
                 "saturation", "color", "luminosity"):
        check(f"{mode} is in the layer menu",
              any(key == mode for key, _ in document.BLEND_MODES), True)
        expected = brush._blend(mode, base, src).getpixel((0, 0))
        check(f"{mode} matches brush._blend", composite(
            mode, Image.new("RGBA", (2, 2), (*BASE, 255)))[:3], expected)


def test_grouped_menu_matches_the_blend_list():
    """The grouped menu must offer every mode once, under a divider.

    The layer blend combobox shows display-only headers between groups; those
    headers are not blend ids and selection must be ignored by the handler. This
    locks the menu to :data:`document.BLEND_MODES` so a newly added mode cannot
    be omitted from the grouped picker.
    """
    labels = document.blend_mode_labels()
    headers = document.blend_headers()
    modes = [label for mode, label in document.BLEND_MODES]

    check("every mode appears once in the menu",
          [label for label in labels if label not in headers], modes)
    check("every divider is a header",
          all(label in headers for label in labels if label not in modes), True)
    check("every header is a divider",
          all(document.is_blend_divider(label) for label in headers), True)
    check("a divider opens each group",
          all(index + 1 < len(labels) and labels[index + 1] not in headers
              for index, label in enumerate(labels) if label in headers), True)
    check("grouped label set has no duplicates",
          len(labels) == len(set(labels)), True)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall layer blend tests passed")


if __name__ == "__main__":
    main()
