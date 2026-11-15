"""Unit tests for canvas links (run: .venv\\Scripts\\python.exe tests\\test_canvas_links.py).

A canvas link is a layer that renders another canvas's visible layers, with no
master layer required. It behaves like a normal link layer: per-instance
offset, transform and mask, read-only content, save/load, unlink and undo.
"""

import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image  # noqa: E402

import project  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import AssetLibrary, Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def make_doc(name="Canvas", size=(4, 4), color=None, point=(1, 1)):
    doc = Canvas(name, size)
    layer = doc.add_layer(name="Art")
    if color is not None:
        layer.draw.point(point, fill=color)
    return doc


def link_to_canvas(doc, source, offset=(0, 0)):
    layer = doc.add_layer(name="Canvas link")
    layer.link_canvas_id = source.canvas_id
    layer.canvas_source = source.link_source()
    layer.offset = offset
    return layer


def fake_app(doc, canvases=None):
    app = SimpleNamespace(
        current=doc,
        canvases=canvases if canvases is not None else [doc],
        active=doc.active,
        root=None,
        _push_undo=lambda *_a, **_k: None,
        _refresh_layer_list=lambda *_a, **_k: None,
        _multi_layer_selection=lambda: None,
        active_layer=doc.active_layer,
        set_result=lambda *_a, **_k: None,
        render=lambda *_a, **_k: None,
        _transform_mode_active=lambda *_a, **_k: False,
        drawing=SimpleNamespace(sync_transform=lambda *_a, **_k: None),
    )
    app._canvas_link_reaches = lambda canvas, target, seen=None: (
        PaintApp._canvas_link_reaches(app, canvas, target, seen)
    )
    app._unlink_canvas_links_to = lambda canvas: (
        PaintApp._unlink_canvas_links_to(app, canvas)
    )
    return app


def make_project_app():
    app = SimpleNamespace(
        canvases=[],
        active_canvas=0,
        canvas_counter=0,
        master_counter=0,
        color="#000000",
        brush_size=8,
        tools=SimpleNamespace(
            export_settings=lambda: {}, apply_settings=lambda _s: None
        ),
        palette=["#000000"],
        selected_swatch=0,
        _text_style=None,
        _loading_project=False,
        guides_shared=False,
        shared_guides=[],
        set_color=lambda *_a, **_k: None,
    )
    library = AssetLibrary()

    def new_canvas(name=None, size=(4, 4)):
        doc = Canvas(name or f"Canvas {len(app.canvases) + 1}", size)
        doc.library = library
        app.canvases.append(doc)
        return doc

    app._new_canvas = new_canvas
    app._relink_masters = lambda: PaintApp._relink_masters(app)
    return app


def test_canvas_link_renders_and_follows_source():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    link = link_to_canvas(target, source)

    flat = target.composite_transparent()
    check("canvas link shows source pixel", flat.getpixel((1, 1))[:3], (255, 0, 0))
    check("canvas link is a link", link.is_link, True)
    check("canvas link is read-only", link.is_locked(), True)

    source.layers[0].draw.point((1, 1), fill=(0, 0, 255, 255))
    source.layers[0].invalidate_filters()
    check(
        "canvas link follows source edit",
        target.composite_transparent().getpixel((1, 1))[:3],
        (0, 0, 255),
    )


def test_canvas_link_offset_transform_and_mask():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    link = link_to_canvas(target, source, offset=(2, 2))

    flat = target.composite_transparent()
    check("offset moves canvas content", flat.getpixel((3, 3))[3], 255)
    check("offset leaves origin empty", flat.getpixel((1, 1))[3], 0)

    link.set_transform((2.0, 0.0, 0.0, 2.0, 0.0, 0.0))
    check(
        "per-instance transform scales canvas content",
        target.composite_transparent().getpixel((2, 2))[3] > 0,
        True,
    )

    link.mask = Image.new("L", link.size, 0)
    check(
        "per-instance mask hides the linked canvas",
        target.composite_transparent().getpixel((2, 2))[3],
        0,
    )


def test_canvas_link_action_uses_selected_layer():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    target.active = 0
    layer = target.layers[0]
    app = fake_app(target, [source, target])

    PaintApp.link_layer_to_canvas(app, 0)

    check("layer becomes a canvas link", layer.link_canvas_id, source.canvas_id)
    check("source resolved", layer.canvas_source.canvas is source, True)
    check(
        "target renders the canvas",
        target.composite_transparent().getpixel((1, 1))[:3],
        (255, 0, 0),
    )


def test_canvas_link_refuses_self_and_cycles():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    target.active = 0
    app = fake_app(target, [source, target])

    messages = []
    app.set_result = messages.append

    PaintApp.link_layer_to_canvas(app, 1)
    check("self link refused", target.layers[0].link_canvas_id, None)

    link_to_canvas(source, target)
    PaintApp.link_layer_to_canvas(app, 0)
    check("cycle refused", target.layers[0].link_canvas_id, None)
    check(
        "cycle reports a reason",
        any("circular" in message for message in messages),
        True,
    )


def test_canvas_link_unlink_makes_an_independent_copy():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    link = link_to_canvas(target, source)
    target.active = target.layers.index(link)

    PaintApp.unlink_layer(fake_app(target))

    check("canvas link id cleared", link.link_canvas_id, None)
    check("canvas source cleared", link.canvas_source, None)
    check("layer owns pixels", link.image.getpixel((1, 1))[:3], (255, 0, 0))

    source.layers[0].draw.point((1, 1), fill=(0, 0, 255, 255))
    check("unlinked layer ignores source", link.image.getpixel((1, 1))[:3], (255, 0, 0))


def test_canvas_link_save_load_round_trip():
    app = make_project_app()
    source = app._new_canvas("Source", (4, 4))
    source.layers = []
    art = source.add_layer(name="Art")
    art.draw.point((1, 1), fill=(255, 0, 0, 255))
    target = app._new_canvas("Target", (4, 4))
    target.layers = []
    link = target.add_layer(name="Canvas link")
    link.link_canvas_id = source.canvas_id
    link.canvas_source = source.link_source()
    link.offset = (1, 0)
    app.active_canvas = 1

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "canvas-links.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
        reloaded = make_project_app()
        project.apply(reloaded, payload)

    entry = manifest["canvases"][1]["layers"][0]
    check(
        "canvas id persisted",
        manifest["canvases"][0]["canvas_id"],
        source.canvas_id,
    )
    check("canvas link id persisted", entry["link_canvas_id"], source.canvas_id)

    restored_source = reloaded.canvases[0]
    restored_link = reloaded.canvases[1].layers[0]
    check(
        "canvas link restored",
        restored_link.link_canvas_id,
        restored_source.canvas_id,
    )
    check(
        "canvas source re-resolved",
        restored_link.canvas_source.canvas is restored_source,
        True,
    )
    check("link offset persisted", restored_link.offset, (1, 0))
    check(
        "reloaded canvas renders source",
        reloaded.canvases[1].composite_transparent().getpixel((2, 1))[:3],
        (255, 0, 0),
    )


def test_canvas_link_relinks_after_restore():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    link_to_canvas(target, source)

    target.restore(target.snapshot())
    restored = target.layers[-1]
    check("canvas source unresolved after restore", restored.canvas_source, None)

    PaintApp._relink_masters(SimpleNamespace(canvases=[source, target]))
    check("canvas source re-resolved", restored.canvas_source.canvas is source, True)
    check(
        "restored canvas link renders",
        target.composite_transparent().getpixel((1, 1))[:3],
        (255, 0, 0),
    )


def test_deleting_source_canvas_unlinks_its_links():
    source = make_doc("Source", color=(255, 0, 0, 255))
    target = make_doc("Target")
    link = link_to_canvas(target, source)
    app = fake_app(target, [source, target])

    PaintApp._unlink_canvas_links_to(app, source)

    check("link id cleared", link.link_canvas_id, None)
    check("link source cleared", link.canvas_source, None)
    check("link keeps the source pixels", link.image.getpixel((1, 1))[:3], (255, 0, 0))


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