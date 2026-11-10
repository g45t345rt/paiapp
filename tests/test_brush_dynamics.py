"""Unit tests for brush velocity dynamics and stroke taper
(run: .venv\\Scripts\\python.exe tests\\test_brush_dynamics.py)."""

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


def alpha_sum(image):
    """Sum of every alpha value in an ``"L"`` image (total paint laid down)."""
    return sum(value * count for value, count in enumerate(image.histogram()))


def make_layer():
    return Layer("t", (128, 64), (0, 0, 0, 0))


def paint(steps, dynamics, size=16, start=(16, 32), end=(112, 32)):
    """Paint a straight stroke as ``steps`` equal move_to segments."""
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "brush", size=size, hardness=100, flow=100, opacity=100,
        spacing=25, antialias=True, tip="Round", dynamics=dynamics,
    )
    stroke.begin(*start)
    x0, y0 = start
    x1, y1 = end
    for i in range(1, steps + 1):
        t = i / steps
        stroke.move_to(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)
    return layer, stroke


def test_default_dynamics_inactive():
    check("default dynamics inactive", brush.Dynamics().active(), False)


def test_dynamics_active_flag():
    check("size makes dynamics active", brush.Dynamics(size=0.5).active(), True)
    check("taper makes dynamics active", brush.Dynamics(taper=10).active(), True)


def test_default_matches_base_stroke():
    base = paint(8, None)[0].image.getchannel("A").tobytes()
    same = paint(8, brush.Dynamics())[0].image.getchannel("A").tobytes()
    check("inactive dynamics is byte-identical", base, same)


def paint_speed(steps, dynamics, speed, size=16, start=(16, 32),
                end=(112, 32)):
    """Paint as ``steps`` equal move_to segments, each given an explicit speed."""
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "brush", size=size, hardness=100, flow=100, opacity=100,
        spacing=25, antialias=True, tip="Round", dynamics=dynamics,
    )
    stroke.begin(*start)
    x0, y0 = start
    x1, y1 = end
    for i in range(1, steps + 1):
        t = i / steps
        stroke.move_to(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, speed=speed)
    return layer, stroke


def test_faster_thins_the_stroke():
    slow = ink_total(paint(40, brush.Dynamics(size=1.0))[0].image.getchannel("A"))
    fast = ink_total(paint(1, brush.Dynamics(size=1.0))[0].image.getchannel("A"))
    check("faster stroke carries less ink", fast < slow, True)


def test_explicit_speed_drives_dynamics():
    # Identical geometry, only the reported speed differs: the fast stroke must
    # thin even though every segment is the same length.
    slow = alpha_sum(
        paint_speed(8, brush.Dynamics(size=1.0), 0.0)[0].image.getchannel("A"))
    fast = alpha_sum(
        paint_speed(8, brush.Dynamics(size=1.0), 1.0)[0].image.getchannel("A"))
    check("explicit fast speed thins the stroke", fast < slow, True)


def test_speed_reduces_luminance():
    slow = alpha_sum(
        paint(40, brush.Dynamics(opacity=1.0))[0].image.getchannel("A"))
    fast = alpha_sum(
        paint(1, brush.Dynamics(opacity=1.0))[0].image.getchannel("A"))
    check("faster stroke lays down less paint", fast < slow, True)


def test_taper_reduces_ink():
    plain = ink_total(paint(8, brush.Dynamics())[0].image.getchannel("A"))
    tapered = ink_total(
        paint(8, brush.Dynamics(taper=20))[0].image.getchannel("A"))
    check("taper removes ink at the ends", tapered < plain, True)
    check("taper keeps the stroke", tapered > 0, True)


def test_split_taper():
    in_only = brush.Dynamics(taper_in=24, taper_out=0)
    out_only = brush.Dynamics(taper_in=0, taper_out=24)
    check("taper in/out stored", (in_only.taper_in, in_only.taper_out),
          (24.0, 0.0))
    base = ink_total(paint(8, brush.Dynamics())[0].image.getchannel("A"))
    tin = ink_total(paint(8, in_only)[0].image.getchannel("A"))
    check("taper_in removes start ink", tin < base, True)
    layer, stroke = paint(8, out_only)
    before = ink_total(layer.image.getchannel("A"))
    check("taper_out leaves the stroke until finish", before, base)
    stroke.finish()
    after = ink_total(layer.image.getchannel("A"))
    check("taper_out fades the tip on finish", after < before, True)


def test_end_taper_applies_on_finish():
    # Compare two finished strokes so the tail completion finish also does is
    # held constant and only the taper differs.
    tapered, taper_stroke = paint(8, brush.Dynamics(taper=16))
    taper_stroke.finish()
    plain, plain_stroke = paint(8, brush.Dynamics())
    plain_stroke.finish()
    check("finish fades the tip",
          ink_total(tapered.image.getchannel("A"))
          < ink_total(plain.image.getchannel("A")), True)
    check("finish keeps the stroke",
          ink_total(tapered.image.getchannel("A")) > 0, True)


def test_finish_without_taper_does_not_fade():
    # The smoothed path ends at the last sample's midpoint; finish extends it to
    # the sample (adding ink) but, with no taper, must never remove any.
    layer, stroke = paint(8, brush.Dynamics())
    before = ink_total(layer.image.getchannel("A"))
    stroke.finish()
    after = ink_total(layer.image.getchannel("A"))
    check("no taper means finish does not fade", after >= before, True)


def test_fast_stroke_curves_instead_of_chording():
    # Two long samples meeting at a corner are joined by a quadratic bezier, so
    # the painted path bows inside the corner instead of chording straight across
    # it - the difference between a faceted fast stroke and a smooth one.
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "brush", size=4, hardness=100, flow=100, opacity=100,
        spacing=10, antialias=True, tip="Round",
    )
    path = []
    original = stroke._advance

    def record(x, y, step):
        path.append((x, y))
        original(x, y, step)

    stroke._advance = record
    stroke.begin(0, 0)
    stroke.move_to(40, 0)
    stroke.move_to(40, 40)
    # The straight chord from (20, 0) to (40, 20) is y = x - 20; the curve must
    # dip well inside it.
    check("fast stroke bows inside the corner",
          any(y < (x - 20) - 3 for x, y in path), True)


def test_preview_renders_a_stroke():
    from paintapp import drawing  # imports Tk widgets but creates no root
    image = drawing.brush_preview_image(
        size=18, hardness=100, flow=100, opacity=100, spacing=25, tip="round",
        pattern="none",
        dynamics=brush.Dynamics(size=0.5, taper_in=12, taper_out=12),
    )
    check("preview size", image.size,
          (drawing.BRUSH_PREVIEW_W, drawing.BRUSH_PREVIEW_H))
    check("preview has ink", min(image.convert("L").getdata()) < 150, True)


def test_preview_eraser_punches_through():
    from paintapp import drawing
    image = drawing.brush_preview_image(
        size=18, hardness=100, flow=100, opacity=100, spacing=25, tip="round",
        pattern="none", dynamics=brush.Dynamics(), kind="eraser",
        color=(230, 80, 80),
    )
    # The erased band must show the light background through the fill.
    check("eraser preview keeps background visible",
          any(px[0] > 200 and px[3] > 0 for px in image.getdata()), True)


def paint_grain(grain, strength, mode, size=16, steps=8):
    """Paint a straight stroke with a grain texture applied."""
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "brush", size=size, hardness=100, flow=100, opacity=100,
        spacing=20, antialias=True, tip="Round",
        grain=grain, grain_strength=strength, grain_mode=mode,
    )
    stroke.begin(16, 32)
    for i in range(1, steps + 1):
        t = i / steps
        stroke.move_to(16 + (112 - 16) * t, 32)
    return layer, stroke


def test_grain_reduces_ink():
    plain = ink_total(
        paint_grain("none", 0.0, "canvas")[0].image.getchannel("A"))
    for grain in ("noise", "paper", "canvas", "halftone"):
        grained = ink_total(
            paint_grain(grain, 0.8, "canvas")[0].image.getchannel("A"))
        check(f"{grain} grain carves ink", grained < plain, True)
        check(f"{grain} grain keeps the stroke", grained > 0, True)


def test_zero_strength_grain_is_a_noop():
    plain = paint_grain("none", 0.0, "canvas")[0].image.getchannel("A").tobytes()
    zero = paint_grain("canvas", 0.0, "canvas")[0].image.getchannel("A").tobytes()
    check("zero-strength grain leaves the stroke alone", zero, plain)


def test_canvas_grain_is_position_stable():
    # The same stroke at the same document position must reveal the same tooth,
    # which is what keeps canvas grain from crawling as you paint.
    first = paint_grain("canvas", 0.8, "canvas")[0].image.getchannel("A").tobytes()
    second = paint_grain("canvas", 0.8, "canvas")[0].image.getchannel("A").tobytes()
    check("canvas grain is deterministic", first, second)


def test_stamp_grain_travels_with_the_dab():
    layer, _ = paint_grain("halftone", 0.8, "stamp")
    marked = ink_total(layer.image.getchannel("A"))
    check("stamp grain still paints", marked > 0, True)


def _distinct_colors(layer):
    seen = set()
    px = layer.image.load()
    for y in range(layer.image.height):
        for x in range(layer.image.width):
            r, g, b, a = px[x, y]
            if a > 40:
                seen.add((r, g, b))
    return seen


def paint_color(color, **kwargs):
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "brush", size=18, hardness=100, flow=100, opacity=100,
        spacing=20, antialias=True, tip="Round", color=color, **kwargs,
    )
    stroke.begin(16, 32)
    for i in range(1, 7):
        stroke.move_to(16 + 16 * i, 32)
    return layer


def test_color_jitter_varies_the_dab():
    plain = _distinct_colors(paint_color((30, 60, 200)))
    jitter = _distinct_colors(paint_color((30, 60, 200), jitter_hue=0.6,
                                          jitter_sat=0.5, jitter_val=0.5))
    check("jitter adds colours", len(jitter) > len(plain), True)


def test_pickup_mixes_with_the_canvas():
    layer = make_layer()
    layer.image.paste((210, 30, 30, 255), (0, 0, 128, 64))
    stroke = brush.Stroke(
        layer, "brush", size=18, hardness=100, flow=100, opacity=100,
        spacing=20, antialias=True, tip="Round", color=(0, 0, 255), pickup=0.7,
    )
    stroke.begin(16, 32)
    stroke.move_to(96, 32)
    # Some deposited colour should be pulled toward the red underneath.
    check("pickup leans toward the canvas colour",
          any(r > 100 and b < 220 for r, g, b in _distinct_colors(layer)), True)


def test_color_defaults_are_a_noop():
    base = paint_color((30, 60, 200)).image.tobytes()
    zero = paint_color((30, 60, 200), jitter_hue=0.0, jitter_sat=0.0,
                       jitter_val=0.0, pickup=0.0).image.tobytes()
    check("zero colour dynamics leaves the stroke alone", zero, base)


def test_colour_map_blends_overlapping_dabs():
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "brush", size=16, hardness=100, flow=100, opacity=100,
        spacing=20, antialias=True, tip="Round", color=(255, 0, 0), pickup=0.5,
    )
    stroke.begin(32, 32)
    stroke.color = (0, 0, 255)
    stroke._stamp(32, 32)
    r, _g, b = stroke._color_map.getpixel((32, 32))
    check("overlap keeps some of the first colour", r > 20, True)
    check("overlap takes some of the second colour", b > 20, True)


def _wet_stroke(strength, use_kwarg=True):
    layer = make_layer()
    kwargs = {"wet_edge": strength} if use_kwarg else {}
    stroke = brush.Stroke(
        layer, "brush", size=24, hardness=0, flow=100, opacity=100,
        spacing=25, antialias=True, tip="Round", **kwargs,
    )
    stroke.begin(16, 32)
    for i in range(1, 7):
        stroke.move_to(16 + i * 16, 32)
    return layer


def test_wet_edge_adds_pigment():
    plain = alpha_sum(_wet_stroke(0.0).image.getchannel("A"))
    wet = alpha_sum(_wet_stroke(0.9).image.getchannel("A"))
    check("wet edge adds pigment at the rim", wet > plain, True)


def test_wet_edge_off_matches_no_kwarg():
    off = _wet_stroke(0.0, True).image.getchannel("A").tobytes()
    absent = _wet_stroke(0.0, False).image.getchannel("A").tobytes()
    check("wet_edge=0 matches a stroke without the argument", off, absent)


def test_colour_eraser_jitter_varies():
    layer = make_layer()
    layer.image.paste((120, 120, 120, 255), (0, 0, 128, 64))
    stroke = brush.Stroke(
        layer, "eraser-color", size=18, hardness=100, flow=100, opacity=100,
        spacing=20, antialias=True, tip="Round", color=(200, 60, 60),
        jitter_hue=0.6, jitter_sat=0.5, jitter_val=0.5,
    )
    stroke.begin(16, 32)
    for i in range(1, 7):
        stroke.move_to(16 + 16 * i, 32)
    check("colour eraser jitters its colours",
          len(_distinct_colors(layer)) > 3, True)


def test_none_grain_has_no_tile():
    check("none grain yields no tile", brush.grain_tile("none", 0.5), None)


def test_smudge_finish_is_safe():
    layer = make_layer()
    stroke = brush.Stroke(
        layer, "smudge", size=16, hardness=50, flow=60, opacity=100,
        spacing=25, antialias=True, tip="Round", dynamics=brush.Dynamics(taper=16),
    )
    stroke.begin(16, 32)
    stroke.move_to(112, 32)
    stroke.finish()


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
