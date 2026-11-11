"""Tool-interaction benchmark on a large (A4 @ 300 dpi) layered project.

Builds a realistic document - raster layers with filters, a group, a multiply
blend layer, text, path and vector layers - saves it as ``.pai`` (and reloads
it), then drives the real ``PaintApp`` through the interactions that feel laggy:
moving layers, dragging text, editing path anchors, tweaking filters and
reordering layers. Each interaction reports the mean and worst per-event time.

    .venv\\Scripts\\python.exe tests\\tool_bench.py
    .venv\\Scripts\\python.exe tests\\tool_bench.py --events 40 --zoom fit --out bench.pai
"""

import argparse
import gc
import os
import statistics
import sys
import tempfile
import time
import tkinter as tk

from PIL import ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import project  # noqa: E402
from paintapp import filters, path as pathspec, textrender, vector  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402

A4 = (2480, 3508)


class _Event:
    __slots__ = ("widget", "x", "y", "state", "delta")

    def __init__(self, widget, x, y, state=0, delta=0):
        self.widget, self.x, self.y, self.state, self.delta = widget, x, y, state, delta


def paint(layer, seed):
    w, h = layer.image.size
    draw = ImageDraw.Draw(layer.begin_write())
    for i in range(40):
        x = (seed * 131 + i * 197) % (w - 300)
        y = (seed * 71 + i * 263) % (h - 300)
        draw.ellipse((x, y, x + 280, y + 220),
                     fill=((seed * 50 + i * 9) % 256, (i * 31) % 256, (seed * 90) % 256, 190))
    layer.invalidate_filters()


def build(app, size):
    app.canvases = []
    app.canvas_counter = 0
    app.active_canvas = 0
    doc = app._new_canvas("Bench A4", size)
    w, h = size
    bg = doc.layers[0]
    paint(bg, 1)
    for i in range(1, 5):
        layer = doc.add_layer(name=f"Paint {i}")
        paint(layer, i + 1)
        if i == 2:
            layer.filters = [filters.new_spec("gaussian_blur"), filters.new_spec("hue")]
            layer.invalidate_filters()
        if i == 3:
            layer.blend = "multiply"
    group = doc.add_group("Group")
    for i in range(2):
        layer = doc.add_layer(name=f"Grouped {i}")
        paint(layer, 20 + i)
        layer.group = group
    text = doc.add_layer(name="Text")
    for i in range(8):
        text.texts.append(textrender.new_spec(
            f"Headline number {i}\nsecond line", 200 + i * 40, 300 + i * 380, size=96,
            color="#203080"))
    text.invalidate_texts()
    paths = doc.add_layer(name="Paths")
    for i in range(12):
        x, y = 300 + (i % 4) * 500, 400 + (i // 4) * 900
        anchors = [(x, y), (x + 400, y + 80), (x + 300, y + 600), (x - 60, y + 450)]
        paths.paths.append(pathspec.new_spec(
            anchors, list(anchors), list(anchors), closed=True, mode="both",
            fill_color="#e05030", stroke_color="#101010", width=8))
    paths.invalidate_paths()
    vec = doc.add_layer(name="Vectors")
    for i in range(30):
        pts = [(100 + i * 70, 200), (300 + i * 60, h // 2), (150 + i * 75, h - 200)]
        vec.strokes.append(vector.new_stroke(pts, 12, "#10a060"))
    vec.invalidate_vectors()
    doc.active = len(doc.layers) - 1
    app.area.rebuild(app.canvases)
    app.area.select(0)
    app._refresh_layer_list()
    app.render()
    return doc


def select_layer(app, doc, name):
    index = next(i for i, l in enumerate(doc.layers) if l.name == name)
    doc.active = index
    app._refresh_layer_list()
    app.render()
    return doc.layers[index]


def drive_move(app, view, root, events):
    app.tools.select("move")
    times = []
    x0, y0 = view._doc_to_view(1200, 1700)
    app.drawing.on_press(_Event(view, int(x0), int(y0)))
    for i in range(1, events + 1):
        t0 = time.perf_counter()
        app.drawing.on_drag(_Event(view, int(x0 + i * 3), int(y0 + i * 2), state=0x100))
        root.update()
        times.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    app.drawing.on_release(_Event(view, int(x0 + events * 3), int(y0 + events * 2)))
    root.update()
    return times, time.perf_counter() - t0


def drive_edit(app, root, events, mutate):
    times = []
    for i in range(1, events + 1):
        t0 = time.perf_counter()
        mutate(i)
        app.render()
        root.update()
        times.append(time.perf_counter() - t0)
    return times


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=int, default=30)
    p.add_argument("--viewport", default="1400x900")
    p.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "pai_bench_a4.pai"))
    args = p.parse_args(argv)
    vw, vh = (int(v) for v in args.viewport.split("x"))

    root = tk.Tk()
    root.geometry(f"{vw}x{vh}+-4000+-4000")
    app = PaintApp(root)
    root.geometry(f"{vw}x{vh}+-4000+-4000")
    root.update()

    t0 = time.perf_counter()
    doc = build(app, A4)
    root.update()
    results = [("build scene", [time.perf_counter() - t0], None)]

    t0 = time.perf_counter()
    project.save(app, args.out)
    results.append(("save .pai", [time.perf_counter() - t0], None))
    t0 = time.perf_counter()
    project.load(app, args.out)
    root.update()
    results.append(("load .pai", [time.perf_counter() - t0], None))
    doc = app.current

    view = app.area.current_view()
    for label, zoom in (("fit", min(vw / A4[0], vh / A4[1]) * 0.95), ("100%", 1.0)):
        view.set_zoom(zoom)
        root.update()
        gc.collect()
        results.append((f"[{label}] render", drive_edit(app, root, 5, lambda i: None), None))
        for name in ("Paint 1", "Text", "Paths", "Vectors", "Grouped 0"):
            select_layer(app, doc, name)
            times, release = drive_move(app, view, root, args.events)
            results.append((f"[{label}] move {name}", times, release))

        layer = select_layer(app, doc, "Text")

        def text_drag(i, layer=layer):
            layer.texts[0]["x"] += 3
            layer.invalidate_texts()
        results.append((f"[{label}] text drag", drive_edit(app, root, args.events, text_drag), None))

        layer = select_layer(app, doc, "Paths")

        def path_edit(i, layer=layer):
            spec = layer.paths[0]
            ax, ay = spec["anchors"][1]
            spec["anchors"][1] = (ax + 3, ay + 2)
            layer.invalidate_paths()
        results.append((f"[{label}] path anchor drag", drive_edit(app, root, args.events, path_edit), None))

        layer = select_layer(app, doc, "Paint 2")

        def filter_tweak(i, layer=layer):
            layer.filters[1]["params"]["hue"] = (i * 5) % 180 if "hue" in layer.filters[1]["params"] else i
            layer.invalidate_filters()
        results.append((f"[{label}] filter tweak", drive_edit(app, root, 5, filter_tweak), None))

        def reorder(i):
            a = doc.active
            b = a + 1 if a + 1 < len(doc.layers) and doc.layers[a + 1].group is None else a - 1
            doc.layers[a], doc.layers[b] = doc.layers[b], doc.layers[a]
            doc.active = b
            app._refresh_layer_list()
        select_layer(app, doc, "Paint 1")
        results.append((f"[{label}] reorder layer", drive_edit(app, root, 6, reorder), None))

    width = max(len(n) for n, *_ in results)
    print(f"\nA4 {A4[0]}x{A4[1]}, {len(doc.layers)} layers, viewport {vw}x{vh}, saved {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")
    print(f"{'operation':<{width}}  {'mean ms':>8}  {'max ms':>8}  {'release ms':>10}")
    for name, times, release in results:
        print(f"{name:<{width}}  {statistics.mean(times) * 1000:>8.1f}  "
              f"{max(times) * 1000:>8.1f}  {'' if release is None else f'{release * 1000:.1f}':>10}")
    root.destroy()


if __name__ == "__main__":
    main()
