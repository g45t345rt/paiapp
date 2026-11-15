"""Unit tests for inserting an asset onto a layer.

Run: .venv\\Scripts\\python.exe tests\\test_insert_asset.py

An asset keeps its own pixel size when inserted: the layer buffer is the
asset's image and ``layer.offset`` centres it on the canvas, so an asset larger
than the canvas is not cropped.
"""

import os
import sys
from types import SimpleNamespace

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import Canvas  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_app(doc, asset):
    app = SimpleNamespace(
        current=doc,
        active=0,
        layers=doc.layers,
        _asset_by_id=lambda _asset_id: asset,
        _push_undo=lambda _label: None,
        _add_layer=lambda name=None, index=None: doc.add_layer(
            name=name, index=index),
        _refresh_layer_list=lambda: None,
        render=lambda: None,
        set_result=lambda *_args, **_kwargs: None,
    )
    app._centered_asset_placement = lambda a, fit=False: (
        PaintApp._centered_asset_placement(app, a, fit=fit)
    )
    return app


def make_asset(doc, size=(128, 64)):
    image = Image.new("RGBA", size, (255, 0, 0, 255))
    image.putpixel((size[0] - 1, 0), (0, 255, 0, 255))
    return doc.add_asset(name="big", image=image)


def test_new_layer_keeps_asset_larger_than_canvas():
    doc = Canvas("c", (64, 64))
    doc.add_layer(name="base")
    asset = make_asset(doc, (128, 64))
    app = fake_app(doc, asset)

    PaintApp._insert_asset_above(app, asset.id, 1)

    layer = doc.layers[-1]
    check("buffer keeps asset size", layer.image.size, (128, 64))
    check("buffer centred", layer.offset, (-32, 0))
    check("right edge kept", layer.image.getpixel((127, 0)), (0, 255, 0, 255))


def test_new_layer_centres_smaller_asset():
    doc = Canvas("c", (64, 64))
    doc.add_layer(name="base")
    asset = make_asset(doc, (32, 32))
    app = fake_app(doc, asset)

    PaintApp._insert_asset_above(app, asset.id, 1)

    layer = doc.layers[-1]
    check("buffer keeps asset size", layer.image.size, (32, 32))
    check("buffer centred", layer.offset, (16, 16))


def test_inserted_layer_does_not_share_asset_pixels():
    doc = Canvas("c", (64, 64))
    doc.add_layer(name="base")
    asset = make_asset(doc, (128, 64))
    app = fake_app(doc, asset)

    PaintApp._insert_asset_above(app, asset.id, 1)
    layer = doc.layers[-1]
    layer.image.putpixel((0, 0), (0, 0, 255, 255))

    check("asset pixels untouched", asset.image.getpixel((0, 0)), (255, 0, 0, 255))


def test_in_layer_keeps_asset_larger_than_canvas():
    doc = Canvas("c", (64, 64))
    base = doc.add_layer(name="base")
    asset = make_asset(doc, (128, 64))
    app = fake_app(doc, asset)
    app._preview = {"kind": "asset", "doc": doc, "asset": asset}
    app._clear_generation_preview = lambda: None
    app._show_doc = lambda _doc: None

    PaintApp.accept_asset_preview_in_layer(app)

    check("layer keeps asset size", base.image.size, (128, 64))
    check("layer buffer centred", base.offset, (-32, 0))
    check("right edge kept", base.image.getpixel((127, 0)), (0, 255, 0, 255))


def test_fit_scales_asset_to_canvas():
    doc = Canvas("c", (64, 64))
    doc.add_layer(name="base")
    asset = make_asset(doc, (128, 64))
    app = fake_app(doc, asset)

    image, origin = PaintApp._centered_asset_placement(app, asset, fit=True)

    check("fit buffer is canvas sized", image.size, (64, 64))
    check("fit buffer offset", origin, (0, 0))
    check("fit keeps aspect", image.getbbox(), (0, 16, 64, 48))


if __name__ == "__main__":
    test_new_layer_keeps_asset_larger_than_canvas()
    test_new_layer_centres_smaller_asset()
    test_inserted_layer_does_not_share_asset_pixels()
    test_in_layer_keeps_asset_larger_than_canvas()
    test_fit_scales_asset_to_canvas()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
