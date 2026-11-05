"""Regression tests for the cached layer flatten (run: .venv\\Scripts\\python.exe tests\\test_composite_cache.py).

``Canvas._flatten_cached`` serves the viewport composite from a document-anchored
backdrop cache so a zoom or pan resamples a couple of buffers instead of
re-compositing every layer (the many-layer zoom lag). The cache is subtle: the
layers above the active one are only pre-flattened when they cannot blend, and a
stroke recomposites the active layer over the cached backdrop. These tests drive
both ``composite_region`` and ``composite_overview`` through the cache and check
that the result is pixel-identical to a whole-stack ``_flatten`` at the same
region and scale.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageChops, ImageDraw  # noqa: E402

from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def build_canvas():
    """A small stack mixing blend modes, opacity and offsets on both sides of
    the active layer, so the below/above split has every case to get wrong."""
    canvas = Canvas("t", (200, 150))
    canvas.layers = []
    specs = [
        ((10, 20, 180, 130), (0, 0), 1.0, "normal", (255, 0, 0, 180)),
        ((0, 0, 120, 90), (30, 10), 0.6, "multiply", (0, 200, 0, 200)),
        ((40, 40, 200, 150), (-15, 5), 0.85, "screen", (0, 0, 255, 160)),
        ((0, 0, 200, 150), (0, 0), 1.0, "normal", (255, 255, 0, 100)),
        ((5, 5, 90, 70), (20, 15), 0.5, "overlay", (10, 220, 220, 210)),
    ]
    for box, offset, opacity, blend, fill in specs:
        layer = canvas.add_layer(name="L")
        img = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
        ImageDraw.Draw(img).rectangle(box, fill=fill)
        layer.set_image(img)
        layer.offset = offset
        layer.opacity = opacity
        layer.blend = blend
    return canvas


def regions():
    return [
        (0, 0, 200, 150),      # whole canvas
        (20, 10, 160, 140),    # interior
        (0, 0, 80, 60),        # top-left corner
        (-30, -20, 220, 170),  # over-extends the canvas (clipped by the caller)
    ]


def reference(canvas, region, scale):
    """The cached path's reference: a whole-stack flatten clipped to the canvas,
    then (for scale < 1) resampled as one image. The cache composites at document
    resolution and resamples once, rather than downscaling every layer."""
    rl = max(0, region[0])
    rt = max(0, region[1])
    rr = min(canvas.width, region[2])
    rb = min(canvas.height, region[3])
    full = canvas._flatten(canvas.layers, (rl, rt, rr, rb), scale=1.0)
    if scale == 1.0:
        return full
    return full.resize(
        (max(1, int(round((rr - rl) * scale))),
         max(1, int(round((rb - rt) * scale)))),
        Image.BOX,
    )


def test_region_cache_matches_flatten():
    """A cached ``composite_region`` equals the whole-stack reference, for every
    active layer row, region and scale."""
    for active in range(5):
        canvas = build_canvas()
        canvas.active = active
        for region in regions():
            for scale in (1.0, 0.5):
                canvas.composite_region(region, active, scale)  # prime the cache
                got, _ = canvas.composite_region(region, active, scale)
                ref = reference(canvas, region, scale)
                check(
                    f"region active={active} region={region} scale={scale}",
                    ImageChops.difference(got.convert("RGBA"),
                                          ref.convert("RGBA")).getbbox(),
                    None,
                )


def test_overview_cache_matches_flatten():
    """A cached ``composite_overview`` equals a whole-stack flatten over the
    content bounds (with the off-canvas veil disabled)."""
    canvas = build_canvas()
    canvas.active = 2
    content = canvas.content_bounds()
    region = (content[0] - 10, content[1] - 10,
              content[2] + 10, content[3] + 10)
    canvas.composite_overview(1.0, region, 1.0, 2)
    got, origin = canvas.composite_overview(1.0, region, 1.0, 2)
    ref = canvas._flatten(canvas.layers, content, scale=1.0)
    check("overview origin clips to content",
          origin, (content[0], content[1]))
    check(
        "overview cached equals flatten",
        ImageChops.difference(got.convert("RGBA"), ref.convert("RGBA")).getbbox(),
        None,
    )


def test_stroke_composites_active_layer_live():
    """Editing the active layer after priming the cache must show through: the
    active layer is never part of the cached backdrop."""
    canvas = build_canvas()
    canvas.active = 2
    region = (0, 0, 200, 150)
    canvas.composite_region(region, 2, 1.0)
    before, _ = canvas.composite_region(region, 2, 1.0)

    # Paint a filled rectangle onto the active layer's own pixels.
    active = canvas.layers[2]
    draw = ImageDraw.Draw(active.begin_write())
    draw.rectangle((0, 0, 200, 150), fill=(255, 0, 255, 255))
    active.invalidate_filters()

    after, _ = canvas.composite_region(region, 2, 1.0)
    ref = canvas._flatten(canvas.layers, region, scale=1.0)
    check("edited active layer is recomposited",
          ImageChops.difference(after.convert("RGBA"),
                                ref.convert("RGBA")).getbbox(), None)
    check("edited active layer changed the output",
          ImageChops.difference(before.convert("RGBA"),
                                after.convert("RGBA")).getbbox() is not None,
          True)


def test_moving_active_layer_does_not_shift_cached_layers():
    """Moving the active layer is recomposited live, but if the move changes the
    content-bounds origin the cached backdrop must be rebuilt: reusing a buffer
    anchored at the old origin would crop it at the new one and shift every
    cached layer (the "other layers move too" bug)."""
    canvas = build_canvas()
    canvas.active = 2
    # Prime, then take a second render at a different region so the content-sized
    # backdrop is actually built (a one-off render stays on the cheap path).
    canvas.composite_region((0, 0, 200, 150), 2, 1.0)
    canvas.composite_region((20, 10, 160, 140), 2, 1.0)

    # Drag the active layer so it becomes the extremal content: content bounds
    # move from (0, 0) to a negative origin, as a layer dragged past the canvas
    # edge does.
    active = canvas.layers[2]
    active.offset = (-40, -30)

    got, _ = canvas.composite_region((0, 0, 200, 150), 2, 1.0)
    ref = reference(canvas, (0, 0, 200, 150), 1.0)
    check(
        "moving the active layer keeps the cached layers aligned",
        ImageChops.difference(got.convert("RGBA"),
                              ref.convert("RGBA")).getbbox(),
        None,
    )


def test_fallback_matches_cached_at_zoom_out():
    """The one-off (uncached) frame must render the same as the cached path at
    ``scale < 1``. They used to differ - per-layer downscale vs composite-at-
    document-resolution-then-resize - so a drag that promoted to the cache
    mid-gesture flickered the whole frame when zoomed out."""
    region = (0, 0, 200, 150)
    primed = build_canvas()
    primed.active = 2
    primed.composite_region(region, 2, 0.5)
    primed.composite_region((20, 10, 160, 140), 2, 0.5)  # build the cache
    cached, _ = primed.composite_region(region, 2, 0.5)

    fresh = build_canvas()
    fresh.active = 2
    fallback = fresh._flatten_cached(region, 2, 0.5)  # uncached one-off frame
    check(
        "uncached frame matches the cached frame when zoomed out",
        ImageChops.difference(cached.convert("RGBA"),
                              fallback.convert("RGBA")).getbbox(),
        None,
    )


def test_clipped_stack_falls_back_to_whole_flatten():
    """Clipping (which the cache path does not model) still composites
    correctly, because the stack falls back to a whole-stack flatten."""
    canvas = build_canvas()
    canvas.active = 2
    canvas.layers[2].clip = True
    region = (0, 0, 200, 150)
    canvas.composite_region(region, 2, 1.0)
    got, _ = canvas.composite_region(region, 2, 1.0)
    ref = canvas._flatten(canvas.layers, region, scale=1.0)
    check("clipped stack matches flatten",
          ImageChops.difference(got.convert("RGBA"),
                                ref.convert("RGBA")).getbbox(), None)


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall composite-cache tests passed")


if __name__ == "__main__":
    main()
