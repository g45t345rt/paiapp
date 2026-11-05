"""Stress test for Pai: find where painting/rendering time goes.

Run it with the same Python the app uses (see ``run.bat``), for example::

    .venv\\Scripts\\python.exe tests\\stress_test.py
    .venv\\Scripts\\python.exe tests\\stress_test.py --canvases 8 --layers 5 --size 1600x1200
    .venv\\Scripts\\python.exe tests\\stress_test.py --points 240 --repeats 3

The window is created off-screen (so widgets get real sizes), then the test
drives the *real* ``PaintApp`` so the numbers reflect the shipping pipeline. It
reports timings for the operations that make a many-canvas / many-layer session
feel laggy:

    build scene        creating N canvases x L painted layers
    composite          Canvas.composite_transparent() (full-canvas flatten)
    render             app.render() (composite + viewport raster + PhotoImage)
    render zoomed out  app.render() while zoomed out (must stay viewport-bound)
    brush stroke       press + N drags + release through DrawingController
    brush stroke mask  same, but on a masked layer (region, not full-layer, work)
    eraser stroke      same, but the render-per-motion path
    large render       app.render() on a canvas far larger than the viewport
    large render zoom  ... and the same zoomed out (must not grow as 1/zoom^2)
    large masked stroke  a stroke on a large masked layer
    undo push          app._push_undo() (references pixels; copy-on-write)
    activate canvas    switching the active tab
    cycle canvases     activating every canvas once
    memory             process RSS before/after the scene, and per stroke

Exit code is non-zero when a step exceeds its budget, so it can gate a CI run.
"""

import argparse
import gc
import os
import sys
import time
import tkinter as tk

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402


# Per-step budgets in milliseconds, computed from the scene so the same limits
# make sense at different sizes. They are set at roughly 2x the expected cost,
# so a step that regresses by ~2x (or worse) is flagged and fails the run.
#
# ``render`` is deliberately a flat limit: viewport-only rendering must not grow
# with the canvas size, so a flat budget is itself the regression guard.
def budgets(canvases, layers, size, points):
    megapixels = size[0] * size[1] / 1e6
    return {
        # Full-canvas flatten (gallery/export path), per active canvas.
        "composite": 20.0 + 12.0 * layers * megapixels,
        "render": 30.0,
        # A zoomed-out repaint composites the visible document rectangle at
        # display resolution, so its cost must track the viewport, not the
        # document area (which grows as 1/zoom^2). Flat limits guard that.
        "render zoomed out": 30.0,
        "large render": 40.0,
        "large render zoomed out": 40.0,
        # A large masked layer must composite the dab's region each drag, not
        # copy and re-multiply the whole layer.
        "large masked stroke": 40.0 + 5.0 * points,
        # Live stroke: per-drag commit (+ a repaint per drag for the eraser).
        "brush stroke": 25.0 + 1.6 * points,
        "brush stroke masked": 25.0 + 1.6 * points,
        "eraser stroke": 80.0 + 18.0 * points,
        # History is per canvas, so a push snapshots only the active canvas's
        # layers. Snapshots reference pixels (copy-on-write), so a push costs
        # metadata per layer, not a pixel copy per layer; a full-copy or
        # whole-scene regression would be orders of magnitude over this.
        "undo push": 2.0 + 0.3 * layers,
        "activate canvas": 45.0,
    }


class _Event:
    """A minimal stand-in for a Tk event, matching the attributes the
    DrawingController reads (``widget``, ``x``, ``y``, ``state``, ``delta``)."""

    __slots__ = ("widget", "x", "y", "state", "delta")

    def __init__(self, widget, x, y, state=0, delta=0):
        self.widget = widget
        self.x = x
        self.y = y
        self.state = state
        self.delta = delta


def parse_size(text):
    text = text.lower().replace(" ", "")
    if "x" in text:
        width, height = text.split("x", 1)
        return (int(width), int(height))
    value = int(text)
    return (value, value)


def rss_mb():
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def paint_sample(layer, seed):
    """Fill a layer with cheap but non-trivial content (gradient + shapes)."""
    width, height = layer.image.size
    draw = ImageDraw.Draw(layer.begin_write())
    step = max(8, width // 24)
    for i in range(0, width, step):
        shade = (i * 255 // max(1, width)) % 256
        draw.rectangle(
            (i, 0, i + step, height),
            fill=(shade, (shade * 3) % 256, (shade * 7) % 256, 160),
        )
    for i in range(6):
        x = (seed * 37 + i * 53) % max(1, width - 40)
        y = (seed * 71 + i * 29) % max(1, height - 40)
        draw.ellipse(
            (x, y, x + 120, y + 120),
            fill=((seed * 40) % 256, (i * 40) % 256, (seed * 90) % 256, 200),
        )
    layer.invalidate_filters()


def build_scene(app, canvases, layers, size):
    app.canvases = []
    app.canvas_counter = 0
    app.active_canvas = 0
    for c in range(canvases):
        doc = app._new_canvas(f"Stress {c + 1}", size)
        doc.layers = []
        for l in range(layers):
            layer = doc.add_layer(name=f"Layer {l + 1}")
            paint_sample(layer, c * 100 + l)
        doc.active = len(doc.layers) - 1
    app.active_canvas = 0
    app.area.rebuild(app.canvases)
    app.area.select(0)
    app._refresh_layer_list()
    app.render()


def stroke(app, view, points, mode, size, span):
    """Drive one full stroke (press, drags, release) and return elapsed seconds.

    ``mode`` is ``"brush"`` or ``"eraser"``; coordinates walk a diagonal across
    the viewport so the stroke covers real pixels.
    """
    tools = app.tools
    tools.select(mode)
    tools.brush_size = size
    tools.size_var.set(size)
    app.drawing._stroke = None
    app.drawing._stroke_open = False
    app.drawing.last_x = app.drawing.last_y = None

    start = time.perf_counter()
    app.drawing.on_press(_Event(view, 4, 4))
    for i in range(1, points + 1):
        t = i / points
        x = int(4 + t * span)
        y = int(4 + t * span)
        app.drawing.on_drag(_Event(view, x, y, state=0x1))
    app.drawing.on_release(_Event(view, 4 + span, 4 + span))
    elapsed = time.perf_counter() - start
    tools.select("brush")
    return elapsed


def timeit(fn, repeats):
    """Run ``fn`` ``repeats`` times; return (per-op seconds, total seconds)."""
    gc.collect()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    total = time.perf_counter() - start
    return total / repeats, total


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pai painting stress test")
    parser.add_argument("--canvases", type=int, default=6)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--size", type=parse_size, default=(1600, 1200))
    parser.add_argument("--points", type=int, default=120,
                        help="drag events per simulated stroke")
    parser.add_argument("--brush", type=int, default=80,
                        help="brush size used for the strokes")
    parser.add_argument("--repeats", type=int, default=1,
                        help="repeat count for the timed single-op steps")
    parser.add_argument("--viewport", type=parse_size, default=(1000, 700))
    parser.add_argument("--zoom", type=float, default=0.2,
                        help="zoom used for the zoomed-out render steps")
    parser.add_argument("--large", type=parse_size, default=(4000, 4000),
                        help="canvas size for the large-image steps")
    args = parser.parse_args(argv)

    root = tk.Tk()
    # Keep the window off-screen rather than withdrawn: a withdrawn toplevel
    # reports 1x1 sizes, which would skip viewport rasterization entirely and
    # hide the very cost this test is meant to find.
    root.geometry(
        f"{args.viewport[0]}x{args.viewport[1]}+-4000+-4000"
    )
    app = PaintApp(root)
    root.geometry(
        f"{args.viewport[0]}x{args.viewport[1]}+-4000+-4000"
    )
    root.update_idletasks()
    root.update()

    before = rss_mb()
    build_start = time.perf_counter()
    build_scene(app, args.canvases, args.layers, args.size)
    root.update()
    build_time = time.perf_counter() - build_start
    after = rss_mb()

    view = app.area.current_view()
    doc = app.current
    span = max(20, min(view.winfo_width(), view.winfo_height()) - 20)

    results = []

    limits = budgets(args.canvases, args.layers, args.size, args.points)

    def record(name, per_op, count=1, total=None):
        if total is None:
            total = per_op * count
        budget = limits.get(name)
        status = ""
        if budget is not None and per_op * 1000.0 > budget:
            status = f"  <== SLOW (budget {budget:.0f} ms)"
        results.append((name, per_op * 1000.0, total * 1000.0, status))

    record("build scene", build_time, count=1, total=build_time)

    per, total = timeit(doc.composite_transparent, args.repeats)
    record("composite", per, args.repeats, total)

    per, total = timeit(app.render, args.repeats)
    record("render", per, args.repeats, total)

    # Zoomed-out view: the visible document rectangle is larger than the
    # viewport, so this is where a large canvas used to stall. The flatten must
    # stay bounded by the viewport's pixels (see CanvasView.render).
    saved_zoom = view.zoom
    view.set_zoom(args.zoom)
    root.update()
    per, total = timeit(app.render, args.repeats)
    record("render zoomed out", per, args.repeats, total)
    view.set_zoom(saved_zoom)
    root.update()

    per, total = timeit(
        lambda: stroke(app, view, args.points, "brush", args.brush, span),
        args.repeats,
    )
    record("brush stroke", per, args.repeats, total)

    # Masked layer: the mask must be folded into the dab's region only, not
    # copied and re-multiplied over the whole layer on every drag.
    masked = doc.active_layer()
    masked.mask = Image.new("L", doc.size, 255)
    per, total = timeit(
        lambda: stroke(app, view, args.points, "brush", args.brush, span),
        args.repeats,
    )
    record("brush stroke masked", per, args.repeats, total)
    masked.mask = None
    masked.invalidate_filters()
    app.render()

    per, total = timeit(
        lambda: stroke(app, view, args.points, "eraser", args.brush, span),
        args.repeats,
    )
    record("eraser stroke", per, args.repeats, total)

    per, total = timeit(lambda: app._push_undo("Stress"), args.repeats)
    record("undo push", per, args.repeats, total)

    per, total = timeit(lambda: app._activate_canvas(0), args.repeats)
    record("activate canvas", per, args.repeats, total)

    def cycle():
        for index in range(len(app.canvases)):
            app._activate_canvas(index)
            # A freshly packed tab defers its render until Tk reports a size;
            # pump the loop so the switch includes the real repaint work.
            root.update()

    per, total = timeit(cycle, args.repeats)
    record("cycle canvases", per, args.repeats, total)

    # History memory: each stroke must add about one edited layer (copy-on-write
    # forks only the layer it changes), never a whole-scene copy.
    layer_mb = doc.size[0] * doc.size[1] * 4 / (1024 * 1024)
    mem_before = rss_mb()
    mem_strokes = 10
    if mem_before is not None:
        view = app.area.current_view()
        for _ in range(mem_strokes):
            stroke(app, view, 8, "brush", args.brush, span)
        root.update()
        mem_after = rss_mb()
        per_stroke_mb = (mem_after - mem_before) / mem_strokes
    else:
        per_stroke_mb = None

    # Large image: a canvas far bigger than the viewport. Rendering must stay
    # bound to the viewport's pixels (so the flat limits hold) and a masked
    # stroke must not touch the whole layer per drag.
    app._push_undo("Large")
    large = app._new_canvas("Large", args.large)
    paint_sample(large.layers[0], 7)
    app.area.add(large)
    app._activate_canvas(len(app.canvases) - 1)
    root.update()
    lview = app.area.current_view()
    lspan = max(20, min(lview.winfo_width(), lview.winfo_height()) - 20)

    per, total = timeit(app.render, args.repeats)
    record("large render", per, args.repeats, total)

    saved_zoom = lview.zoom
    lview.set_zoom(args.zoom)
    root.update()
    per, total = timeit(app.render, args.repeats)
    record("large render zoomed out", per, args.repeats, total)
    lview.set_zoom(saved_zoom)
    root.update()

    large.active_layer().mask = Image.new("L", large.size, 255)
    per, total = timeit(
        lambda: stroke(app, lview, args.points, "brush", args.brush, lspan),
        args.repeats,
    )
    record("large masked stroke", per, args.repeats, total)

    # ---------------------------------------------------------------- report
    width = max(len(name) for name, *_ in results)
    print()
    print("=" * (width + 40))
    print("PAI STRESS TEST")
    print("=" * (width + 40))
    print(f"scene      : {args.canvases} canvases x {args.layers} layers "
          f"@ {args.size[0]}x{args.size[1]}")
    print(f"viewport   : {view.winfo_width()}x{view.winfo_height()}  "
          f"stroke: {args.points} drags, brush {args.brush}px")
    if before is not None and after is not None:
        print(f"memory     : {before:.0f} MB -> {after:.0f} MB "
              f"(+{after - before:.0f} MB)")
    if per_stroke_mb is not None:
        print(f"history    : +{per_stroke_mb:.1f} MB/stroke "
              f"(one layer ~{layer_mb:.1f} MB; whole scene ~{layer_mb * args.canvases * args.layers:.0f} MB)")
    print("-" * (width + 40))
    print(f"{'operation':<{width}}  {'per-op ms':>10}  {'total ms':>10}")
    print("-" * (width + 40))
    slow = False
    for name, per_ms, total_ms, status in results:
        print(f"{name:<{width}}  {per_ms:>10.1f}  {total_ms:>10.1f}{status}")
        if status:
            slow = True
    print("=" * (width + 40))

    try:
        root.destroy()
    except tk.TclError:
        pass
    return 1 if slow else 0


if __name__ == "__main__":
    sys.exit(main())
