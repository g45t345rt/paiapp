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

from paintapp.canvasarea import CanvasView  # noqa: E402
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
    """The cached path's reference: a whole-stack ``_flatten`` clipped to the
    canvas. At ``scale < 1`` the stack is flattened per layer at that scale (the
    same path the live stroke patch uses), so the frame and a dirty-rect patch
    line up exactly; at ``scale == 1`` it is the document-resolution flatten the
    below/above cache reproduces."""
    rl = max(0, region[0])
    rt = max(0, region[1])
    rr = min(canvas.width, region[2])
    rb = min(canvas.height, region[3])
    return canvas._flatten(canvas.layers, (rl, rt, rr, rb), scale=scale)


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


def test_anchor_patch_matches_full_flatten():
    """The live stroke's dirty-rect patch must be pixel-identical to a full
    re-flatten. The frame is flattened at ``scale``, the active layer is edited,
    and only the edited rectangle is recomposited with the frame's anchor and
    pasted in; that has to equal flattening the whole region again. The live
    patch only runs at ``scale == 1`` (see ``CanvasView._patch_supported``);
    layers are canvas-sized, as painting layers are."""
    region = (0, 0, 200, 150)
    dirty = (20, 40, 150, 130)
    for scale in (1.0,):
        canvas = Canvas("t", (200, 150))
        canvas.layers = []
        for i, fill in enumerate(((255, 0, 0, 180), (0, 0, 255, 120))):
            layer = canvas.add_layer(name="L")
            img = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
            ImageDraw.Draw(img).rectangle((0, 0, 200, 150), fill=fill)
            layer.set_image(img)
        canvas.active = 1
        frame, origin = canvas.composite_region(region, 1, scale, draft=True)

        active = canvas.layers[1]
        draw = ImageDraw.Draw(active.begin_write())
        draw.rectangle((40, 40, 140, 110), fill=(255, 0, 255, 255))
        active.invalidate_filters()

        sub, sub_origin = canvas.composite_region(
            dirty, 1, scale, draft=True, anchor=origin
        )
        patched = frame.copy()
        patched.paste(
            sub,
            (int(round((sub_origin[0] - origin[0]) * scale)),
             int(round((sub_origin[1] - origin[1]) * scale))),
        )
        ref, _ = canvas.composite_region(region, 1, scale, draft=True)
        check(
            f"anchor patch matches full flatten at scale={scale}",
            ImageChops.difference(patched.convert("RGBA"),
                                  ref.convert("RGBA")).getbbox(),
            None,
        )


def live_view(doc):
    """A CanvasView wired up enough to run the live stroke patch without Tk."""
    view = CanvasView.__new__(CanvasView)
    view.doc = doc
    view._layout = None
    view._frame_size = (doc.width, doc.height)
    view._live_draft = None
    view._source = None
    view._region_provider = (
        lambda region, scale=1.0, draft=False, anchor=None:
        doc.composite_region(region, doc.active, scale, draft=draft, anchor=anchor)
    )
    return view


def test_live_composite_patches_dirty_rect():
    """The live stroke frame must equal a full re-flatten once the brush's dirty
    rectangle has been patched in."""
    region = (0, 0, 200, 150)
    dirty = (20, 40, 150, 130)
    render_size = (200, 150)
    for scale in (1.0, 0.5):
        canvas = Canvas("t", (200, 150))
        canvas.layers = []
        for fill in ((255, 0, 0, 180), (0, 0, 255, 120)):
            layer = canvas.add_layer(name="L")
            img = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
            ImageDraw.Draw(img).rectangle((0, 0, 200, 150), fill=fill)
            layer.set_image(img)
        canvas.active = 1
        view = live_view(canvas)
        view._live_composite(region, scale, None, render_size)  # first frame

        active = canvas.layers[1]
        draw = ImageDraw.Draw(active.begin_write())
        draw.rectangle((40, 40, 140, 110), fill=(255, 0, 255, 255))
        active.invalidate_filters()

        got, _ = view._live_composite(region, scale, dirty, render_size)
        ref, _ = canvas.composite_region(region, 1, scale, draft=True)
        diff = ImageChops.difference(got.convert("RGBA"), ref.convert("RGBA"))
        if scale == 1.0:
            check("live patch matches a full flatten at 100%",
                  diff.getbbox(), None)
        else:
            # A zoomed-out patch is a draft; the pixels the brush did not touch
            # (left of the dirty rectangle here) must be left alone.
            check("zoom-out patch leaves untouched columns",
                  diff.crop((0, 0, dirty[0], ref.height)).getbbox(), None)


def test_overview_live_patch_matches_full_render():
    """The live stroke patch must stay correct with the off-canvas preview on.

    ``PaintApp.render`` used to disable the live patch whenever the off-canvas
    preview was shown (the default), so a stroke on a large canvas re-flattened
    the whole viewport every drag. The overview provider applies the off-canvas
    veil to the patched rectangle too, so the patched frame still equals a full
    overview render - both for a dab inside the canvas and one whose dirty
    rectangle crosses the canvas edge (where the veil actually applies).
    """
    canvas = Canvas("t", (200, 150))
    canvas.layers = []
    base = canvas.add_layer(name="base")
    base.set_image(Image.new("RGBA", (200, 150), (255, 0, 0, 255)))
    top = canvas.add_layer(name="top")
    top.set_image(Image.new("RGBA", (200, 150), (0, 0, 0, 0)))
    top.offset = (-40, -30)  # part of the active layer lies off the canvas
    canvas.active = 1

    def provider(region, scale=1.0, draft=False, anchor=None):
        return canvas.composite_overview(
            0.35, region, scale, canvas.active, draft=draft, anchor=anchor
        )

    region = (0, 0, 200, 150)
    render_size = (200, 150)

    def patched_frame(dirty):
        view = CanvasView.__new__(CanvasView)
        view.doc = canvas
        view._layout = None
        view._frame_size = canvas.size
        view._live_draft = None
        view._source = None
        view._region_provider = provider
        view._live_composite(region, 1.0, None, render_size)
        image, _ = view._live_composite(region, 1.0, dirty, render_size)
        return image

    draw = ImageDraw.Draw(canvas.layers[1].begin_write())
    draw.rectangle((60, 40, 140, 120), fill=(0, 200, 255, 255))
    canvas.layers[1].invalidate_filters()

    inside = patched_frame((50, 30, 150, 130))
    ref, _ = canvas.composite_overview(0.35, region, 1.0, canvas.active, draft=True)
    check("overview in-canvas patch matches a full render",
          ImageChops.difference(inside.convert("RGBA"),
                                ref.convert("RGBA")).getbbox(), None)

    draw = ImageDraw.Draw(canvas.layers[1].begin_write())
    draw.rectangle((0, 0, 60, 60), fill=(0, 255, 0, 255))
    canvas.layers[1].invalidate_filters()
    edge = patched_frame((-50, -40, 40, 40))
    ref, _ = canvas.composite_overview(0.35, region, 1.0, canvas.active, draft=True)
    check("overview edge-crossing patch matches a full render",
          ImageChops.difference(edge.convert("RGBA"),
                                ref.convert("RGBA")).getbbox(), None)


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


def test_blank_layer_overlap_is_skipped_without_changing_output():
    """A fully transparent layer overlap is skipped instead of downscaled, which
    must not change the composite (the default new canvas is one blank layer).

    Covers the two ways a layer can be "blank" (all-zero RGBA, and invisible
    pixels that keep stale RGB under alpha 0) and checks that a genuinely opaque
    black layer - whose alpha is fully set - is *not* mistaken for blank."""
    region = (0, 0, 120, 90)

    def stack(extra_blank, opaque_black):
        canvas = Canvas("t", (200, 150))
        canvas.layers = []
        base = canvas.add_layer(name="base")
        base.set_image(Image.new("RGBA", (200, 150), (0, 0, 0, 0)))
        draw = ImageDraw.Draw(base.begin_write())
        draw.rectangle((10, 10, 110, 80), fill=(20, 160, 220, 255))
        base.invalidate_filters()
        if extra_blank is not None:
            layer = canvas.add_layer(name="blank")
            image = Image.new("RGBA", (200, 150), (0, 0, 0, 0))
            if extra_blank == "stale-rgb":
                draw = ImageDraw.Draw(image)
                draw.rectangle((0, 0, 200, 150), fill=(255, 0, 0, 0))
            layer.set_image(image)
        if opaque_black:
            layer = canvas.add_layer(name="black")
            layer.set_image(Image.new("RGBA", (200, 150), (0, 0, 0, 255)))
        canvas.active = len(canvas.layers) - 1
        return canvas

    reference, _ = stack(None, False).composite_region(region, 0, 1.0)
    for kind in (None, "zeros", "stale-rgb"):
        canvas = stack(kind, False)
        got, _ = canvas.composite_region(region, canvas.active, 1.0)
        check(f"blank layer ({kind}) leaves the composite unchanged",
              ImageChops.difference(got.convert("RGBA"),
                                    reference.convert("RGBA")).getbbox(), None)

    black = stack(None, True)
    got, _ = black.composite_region(region, black.active, 1.0)
    check("opaque black layer is not treated as blank",
          got.convert("RGB").getpixel((5, 5)), (0, 0, 0))


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
