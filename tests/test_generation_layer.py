"""Unit tests for carrying a run's prompts onto a generated layer.

Run: .venv\\Scripts\\python.exe tests\\test_generation_layer.py

``PaintApp._apply_generated`` is called unbound with a minimal stand-in
coordinator, so the layer-building path is exercised without a Tk root.
"""

import os
import sys
from types import SimpleNamespace

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp.app import PaintApp  # noqa: E402
from paintapp.document import Canvas  # noqa: E402
from paintapp.generation import OUTPUT_ASSETS, OUTPUT_LAYER, OUTPUT_NEW_LAYER  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def fake_app(doc):
    return SimpleNamespace(
        current=doc,
        active=0,
        statusbar=SimpleNamespace(set_status=lambda _status: None),
        set_result=lambda *_args, **_kwargs: None,
        _set_modified=lambda _flag: None,
        _refresh_history=lambda: None,
        _show_doc=lambda _doc: None,
        _select_asset=lambda _asset: None,
    )


def run_generated(output=OUTPUT_NEW_LAYER, settings=None, size=(64, 64),
                  canvas=(64, 64)):
    doc = Canvas("c", canvas)
    doc.add_layer(name="base")
    app = fake_app(doc)
    result = Image.new("RGB", size, "red")
    PaintApp._apply_generated(
        app, doc, result, None, False, "green hat", output,
        doc.active_layer(), settings=settings,
    )
    return doc


def test_new_layer_carries_prompt_and_negative():
    settings = {
        "model": "m", "negative": "blurry", "steps": 20,
        "positive_prompt": "green hat",
        "width": 64, "height": 64,
    }
    doc = run_generated(settings=settings)
    check("one new layer", len(doc.layers), 2)
    layer = doc.layers[-1]
    check("positive prompt carried", layer.positive_prompt, "green hat")
    check("negative prompt carried", layer.gen_settings.get("negative"), "blurry")
    check("sampler field carried", layer.gen_settings.get("steps"), 20)
    check("positive prompt not stored in settings", "positive_prompt" in layer.gen_settings, False)


def test_new_layer_without_settings_still_carries_prompt():
    doc = run_generated(settings=None)
    layer = doc.layers[-1]
    check("positive prompt carried", layer.positive_prompt, "green hat")
    check("no settings stored", layer.gen_settings, {})


def test_in_layer_target_is_untouched():
    doc = Canvas("c", (64, 64))
    base = doc.add_layer(name="base")
    base.positive_prompt = "keep me"
    app = fake_app(doc)
    PaintApp._apply_generated(
        app, doc, Image.new("RGB", (64, 64), "blue"), None, False,
        "green hat", OUTPUT_LAYER, base,
        settings={"negative": "blurry", "positive_prompt": "green hat"},
    )
    check("still one layer", len(doc.layers), 1)
    check("existing positive prompt untouched", base.positive_prompt, "keep me")
    check("existing layer settings untouched", base.gen_settings, {})


def test_assets_path_unchanged():
    doc = run_generated(output=OUTPUT_ASSETS, settings={"negative": "blurry"})
    check("no new layer", len(doc.layers), 1)


def test_new_layer_centres_smaller_run():
    doc = run_generated(size=(32, 32), canvas=(64, 64))
    layer = doc.layers[-1]
    check("smaller run centred", layer.offset, (16, 16))


def test_in_layer_centres_smaller_run():
    doc = Canvas("c", (64, 64))
    base = doc.add_layer(name="base")
    app = fake_app(doc)
    PaintApp._apply_generated(
        app, doc, Image.new("RGB", (32, 32), "blue"), None, False,
        "green hat", OUTPUT_LAYER, base,
    )
    check("in-layer run centred", base.offset, (16, 16))


if __name__ == "__main__":
    test_new_layer_carries_prompt_and_negative()
    test_new_layer_without_settings_still_carries_prompt()
    test_in_layer_target_is_untouched()
    test_assets_path_unchanged()
    test_new_layer_centres_smaller_run()
    test_in_layer_centres_smaller_run()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")
