"""Unit tests for live gradient objects (run: .venv\\Scripts\\python.exe tests\\test_gradient.py)."""

import os
import sys
import json
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from PIL import Image, ImageDraw  # noqa: E402

import project  # noqa: E402
from paintapp import gradient  # noqa: E402
from paintapp.document import AssetLibrary, Canvas, Layer  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _spec(**overrides):
    values = {
        "start": (0, 0),
        "end": (31, 0),
        "mode": "linear",
        "stops": [[0.0, "#ff0000", 255], [1.0, "#0000ff", 255]],
        "spread": "clamp",
        "opacity": 100,
        "blend": "normal",
    }
    values.update(overrides)
    return gradient.normalize(values)


def test_normalize_defaults_and_rejects_bad():
    check("bad spec dropped", gradient.normalize(None), None)
    spec = gradient.normalize({
        "start": [0, 0], "end": [10, 0], "mode": "bogus",
        "spread": "bogus", "opacity": 500, "blend": "Normal",
    })
    check("mode falls back", spec["mode"], "linear")
    check("spread falls back", spec["spread"], "clamp")
    check("opacity clamps", spec["opacity"], 100)
    check("blend normalises", spec["blend"], "normal")
    check("two default stops", len(spec["stops"]), 2)


def test_clone_is_independent():
    spec = _spec(mask=Image.new("L", (32, 8), 128))
    copy = gradient.clone(spec)
    copy["stops"][0][1] = "#00ff00"
    copy["mask"].putpixel((0, 0), 0)
    check("original stops untouched", spec["stops"][0][1], "#ff0000")
    check("original mask untouched", spec["mask"].getpixel((0, 0)), 128)


def test_paste_paints_gradient_axis():
    spec = _spec()
    base = Image.new("RGBA", (32, 8), (0, 0, 0, 0))
    result = gradient.paste(base, spec)
    left = result.getpixel((1, 4))
    right = result.getpixel((30, 4))
    check("left is red", left[0] > left[2], True)
    check("right is blue", right[2] > right[0], True)
    check("painted alpha", left[3] > 0 and right[3] > 0, True)


def test_layer_composite_and_rasterize():
    layer = Layer("Gradient", (32, 8))
    layer.gradients = [_spec()]
    composited = layer.composited_image()
    check("composite paints gradient", composited.getpixel((1, 4))[0] > 0, True)
    check("rasterize reports change", layer.rasterize_gradients(), True)
    check("gradient list cleared", layer.gradients, [])
    check("pixels kept", layer.image.getpixel((1, 4))[0] > 0, True)


def test_hidden_gradient_is_skipped():
    layer = Layer("Gradient", (32, 8))
    layer.gradients = [_spec(hidden=True)]
    check("hidden leaves layer clear", layer.composited_image().getpixel((1, 4))[3], 0)


def test_mask_limits_gradient():
    mask = Image.new("L", (32, 8), 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 15, 7), fill=255)
    layer = Layer("Gradient", (32, 8))
    layer.gradients = [_spec(mask=mask)]
    image = layer.composited_image()
    check("inside mask painted", image.getpixel((1, 4))[3] > 0, True)
    check("outside mask clear", image.getpixel((30, 4))[3], 0)


def test_json_round_trips_mask():
    mask = Image.new("L", (32, 8), 0)
    ImageDraw.Draw(mask).rectangle((0, 0, 15, 7), fill=255)
    spec = _spec(mask=mask)
    data = gradient.to_json(spec)
    json.dumps(data)
    rebuilt = gradient.from_json(data)
    check("mask survives json", rebuilt["mask"].getpixel((1, 4)), 255)
    check("mask blank survives json", rebuilt["mask"].getpixel((30, 4)), 0)


def test_zoomed_out_cache_invalidates_for_gradient():
    canvas = Canvas("Zoom", (64, 64))
    layer = canvas.add_layer(name="Gradient")
    region = (0, 0, 64, 64)
    blank = canvas._flatten([layer], region, scale=0.5)
    check("blank before gradient", blank.getchannel("A").getbbox(), None)
    layer.gradients = [_spec(start=(0, 0), end=(0, 63))]
    layer.invalidate_gradients()
    drawn = canvas._flatten([layer], region, scale=0.5)
    check("zoomed-out render sees gradient",
          drawn.getchannel("A").getbbox() is not None, True)


def _fake_app(canvas):
    library = AssetLibrary()
    return SimpleNamespace(
        asset_library=library,
        active_canvas=0,
        canvas_counter=1,
        color="#000000",
        brush_size=8,
        tools=SimpleNamespace(export_settings=lambda: {}),
        palette=["#000000"],
        selected_swatch=0,
        canvases=[canvas],
    )


def test_project_round_trips_gradient_objects():
    canvas = Canvas("Canvas 1", (32, 8))
    layer = canvas.add_layer(name="Gradient")
    layer.gradients = [_spec()]
    app = _fake_app(canvas)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "gradient.pai")
        manifest = project.save(app, path)
        payload = project.read(SimpleNamespace(canvases=[]), path)
    check(
        "manifest records gradient",
        len(manifest["canvases"][0]["layers"][0]["gradients"]), 1,
    )
    specs = payload["canvases"][0].layers[0].gradients
    check("loaded one gradient", len(specs), 1)
    check("axis survived", specs[0]["start"], (0.0, 0.0))
    check("stops survived", specs[0]["stops"][1][1], "#0000ff")


def main():
    tests = [
        test_normalize_defaults_and_rejects_bad,
        test_clone_is_independent,
        test_paste_paints_gradient_axis,
        test_layer_composite_and_rasterize,
        test_hidden_gradient_is_skipped,
        test_mask_limits_gradient,
        test_json_round_trips_mask,
        test_zoomed_out_cache_invalidates_for_gradient,
        test_project_round_trips_gradient_objects,
    ]
    for test in tests:
        test()
    if FAILED:
        print("\nFAILED:")
        for failure in FAILED:
            print("  " + failure)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
