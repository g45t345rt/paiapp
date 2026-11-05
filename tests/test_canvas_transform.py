"""Checks the canvas-level rotate/flip transform.

Run: .venv\\Scripts\\python.exe tests\\test_canvas_transform.py

``PaintApp._transform_canvas`` is called unbound with a minimal stand-in
coordinator, so the whole-document transform is exercised without a Tk root:
pixels, masks, layer offsets, guides, and editable path/vector/text geometry all
move together, and a 90 degree turn swaps the canvas size.
"""

import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import path, textrender, vector  # noqa: E402
from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_app(doc, guides_shared=False):
    app = SimpleNamespace(
        current=doc,
        guides_shared=guides_shared,
        area=SimpleNamespace(current_view=lambda: None),
        drawing=SimpleNamespace(
            clear_crop=lambda: None,
            discard_warp=lambda: None,
            discard_puppet=lambda: None,
        ),
        _push_undo=lambda *_args, **_kwargs: None,
        _sync_gen_size=lambda: None,
        _update_title=lambda: None,
        render=lambda: None,
        set_result=lambda *_args, **_kwargs: None,
    )
    app._transform_canvas = (
        lambda kind, label: PaintApp._transform_canvas(app, kind, label)
    )
    return app


def run(method, doc, **kwargs):
    getattr(PaintApp, method)(fake_app(doc, **kwargs))


def test_flip_horizontal_moves_pixels_and_offset():
    doc = Canvas("c", (5, 3))
    doc.add_layer(name="paint")
    layer = doc.active_layer()
    layer.draw.point((1, 0), fill=(255, 0, 0, 255))
    layer.offset = (1, 1)

    run("flip_canvas_horizontal", doc)

    check("size unchanged", doc.size, (5, 3))
    check("pixel mirrored", layer.image.getpixel((3, 0)), (255, 0, 0, 255))
    check("old spot cleared", layer.image.getpixel((1, 0))[3], 0)
    # New offset keeps the layer's right edge at the same document column.
    check("offset mirrored", layer.offset, (5 - 1 - 5, 1))


def test_rotate_cw_swaps_size_and_pixels():
    doc = Canvas("c", (4, 2))
    doc.add_layer(name="paint")
    layer = doc.active_layer()
    layer.draw.point((1, 0), fill=(0, 255, 0, 255))

    run("rotate_canvas_cw", doc)

    check("size swapped", doc.size, (2, 4))
    check("layer transposed", layer.image.size, (2, 4))
    # ROTATE_270 maps pixel (x, y) -> (H - 1 - y, x); (1, 0) -> (1, 1).
    check("pixel rotated clockwise", layer.image.getpixel((1, 1)), (0, 255, 0, 255))
    check("offset remapped", layer.offset, (0, 0))


def test_mask_and_inpaint_mask_transpose():
    doc = Canvas("c", (4, 2))
    doc.add_layer(name="paint")
    layer = doc.active_layer()
    layer.mask = layer.image.getchannel("A").copy()
    layer.mask.putpixel((3, 0), 255)
    doc.inpaint_mask = layer.image.getchannel("A").copy()
    doc.inpaint_mask.putpixel((0, 1), 255)

    run("rotate_canvas_ccw", doc)

    check("mask transposed", layer.mask.size, (2, 4))
    # CCW 90 maps (x, y) -> (y, W - 1 - x); (3, 0) -> (0, 0).
    check("mask pixel moved", layer.mask.getpixel((0, 0)), 255)
    check("inpaint mask transposed", doc.inpaint_mask.size, (2, 4))
    check("canvas size swapped", doc.size, (2, 4))


def test_guides_follow_the_transform():
    doc = Canvas("c", (5, 3))
    doc.add_layer(name="paint")
    doc.guides = [("v", 1), ("h", 1)]

    run("flip_canvas_horizontal", doc)
    check("flip_h mirrors vertical guide", doc.guides, [("v", 4), ("h", 1)])

    doc2 = Canvas("c", (5, 3))
    doc2.add_layer(name="paint")
    doc2.guides = [("v", 1), ("h", 1)]
    run("rotate_canvas_cw", doc2)
    # v(1) -> h(1); h(1) -> v(height - 1) = v(2)
    check("rotate_cw turns guides", doc2.guides, [("h", 1), ("v", 2)])

    doc3 = Canvas("c", (5, 3))
    doc3.add_layer(name="paint")
    doc3.guides = [("v", 1), ("h", 1)]
    run("flip_canvas_horizontal", doc3, guides_shared=True)
    check("shared guides untouched", doc3.guides, [("v", 1), ("h", 1)])


def test_editable_geometry_remaps_on_rotate():
    doc = Canvas("c", (4, 2))
    doc.add_layer(name="art")
    layer = doc.active_layer()
    layer.paths = [
        path.new_spec([[1, 0], [3, 0]], [[1, 0], [3, 0]], [[1, 0], [3, 0]])
    ]
    layer.strokes = [vector.new_stroke([[1, 0], [2, 1]], 3, "#000000")]
    layer.texts = [textrender.new_spec("hi", x=1, y=0, rotation=0.0)]

    run("rotate_canvas_cw", doc)

    check("path anchors remapped",
          layer.paths[0]["anchors"], [[2, 1], [2, 3]])
    check("stroke points remapped",
          layer.strokes[0]["points"], [[2, 1], [1, 2]])
    check("text anchor remapped",
          (layer.texts[0]["x"], layer.texts[0]["y"]), (2.0, 1.0))
    check("text rotation turned", layer.texts[0]["rotation"], 90.0)


def test_flip_bakes_text_but_remaps_paths():
    doc = Canvas("c", (4, 2))
    doc.add_layer(name="art")
    layer = doc.active_layer()
    layer.paths = [
        path.new_spec([[1, 0], [3, 0]], [[1, 0], [3, 0]], [[1, 0], [3, 0]])
    ]
    layer.texts = [textrender.new_spec("hi", x=0, y=0)]

    run("flip_canvas_horizontal", doc)

    check("text rasterised away", layer.texts, [])
    check("path anchor mirrored",
          layer.paths[0]["anchors"], [[3, 0], [1, 0]])


if __name__ == "__main__":
    test_flip_horizontal_moves_pixels_and_offset()
    test_rotate_cw_swaps_size_and_pixels()
    test_mask_and_inpaint_mask_transpose()
    test_guides_follow_the_transform()
    test_editable_geometry_remaps_on_rotate()
    test_flip_bakes_text_but_remaps_paths()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
