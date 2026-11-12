"""Unit tests for the background-removal ONNX family detection and preprocessing.

Run: .venv\\Scripts\\python.exe tests\\test_background.py

No model file or ``onnxruntime`` is needed: ``_spec_for`` only reads a session's
declared inputs/outputs, so a tiny stub stands in for a real graph.
"""

import os
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import background  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _IO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


class _Session:
    def __init__(self, inputs, outputs):
        self._inputs = inputs
        self._outputs = outputs

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def _session(input_name, input_shape, outputs):
    return _Session([_IO(input_name, input_shape)], [_IO(n, s) for n, s in outputs])


def test_birefnet_pixel_values():
    session = _session(
        "pixel_values", [1, 3, 1024, 1024], [("alphas", [1, 1, 1024, 1024])]
    )
    spec = background._spec_for(session)
    check("birefnet size", spec.size, 1024)
    check("birefnet activated", spec.activated, False)
    check("birefnet max_norm", spec.max_norm, True)
    check("birefnet dynamic", spec.dynamic, False)


def test_birefnet_generic_input():
    session = _session(
        "input", [1, 3, 1024, 1024], [("output", [1, 1, 1024, 1024])]
    )
    spec = background._spec_for(session)
    check("generic birefnet size", spec.size, 1024)
    check("generic birefnet activated", spec.activated, False)


def test_u2net_fixed_single_output():
    session = _session(
        "input.1", [1, 3, 320, 320], [("output", [1, 1, 320, 320])]
    )
    spec = background._spec_for(session)
    check("u2net size", spec.size, 320)
    check("u2net activated", spec.activated, True)
    check("u2net max_norm", spec.max_norm, False)
    check("u2net dynamic", spec.dynamic, False)


def test_u2net_side_outputs_pick_largest():
    outputs = [
        (f"d{i}", [1, 1, 320 >> i, 320 >> i]) for i in range(3)
    ]
    session = _session("input.1", [1, 3, 320, 320], outputs)
    spec = background._spec_for(session)
    check("u2net side output index", spec.output, 0)
    check("u2net side output size", spec.size, 320)
    check("u2net side output activated", spec.activated, True)


def test_modnet_dynamic():
    session = _session(
        "input", [1, 3, "height", "width"], [("output", [1, 1, "height", "width"])]
    )
    spec = background._spec_for(session)
    check("modnet dynamic", spec.dynamic, True)
    check("modnet size", spec.size, 512)
    check("modnet mean", spec.mean, background._MATTING_MEAN)
    check("modnet activated", spec.activated, True)


def test_spatial_ignores_symbolic_dims():
    check("spatial symbolic", background._spatial([1, 3, "height", "width"]), None)
    check("spatial negative", background._spatial([1, 3, -1, -1]), None)
    check("spatial fixed", background._spatial([1, 3, 512, 512]), (512, 512))


def test_preprocess_fixed_square():
    spec = background._Spec(320, background._IMAGENET_MEAN, background._IMAGENET_STD, True)
    array = background._preprocess(Image.new("RGB", (100, 200)), spec)
    check("preprocess fixed shape", array.shape, (1, 3, 320, 320))


def test_preprocess_dynamic_aspect():
    spec = background._Spec(
        512, background._MATTING_MEAN, background._MATTING_STD, True, dynamic=True
    )
    array = background._preprocess(Image.new("RGB", (1000, 500)), spec)
    check("preprocess dynamic shape", array.shape, (1, 3, 256, 512))


def test_supported_suffixes():
    check("onnx supported", background._is_supported("m.onnx"), True)
    check("safetensors supported", background._is_supported("m.safetensors"), True)
    check("unknown unsupported", background._is_supported("m.txt"), False)
    check("is_onnx", background._is_onnx("m.ONNX"), True)


def test_modnet_in_catalog():
    check("modnet catalogued", "MODNet (portrait)" in background.MODELS, True)
    check("u2net catalogued", "U2Net" in background.MODELS, True)


def test_restrict_to_mask():
    original = Image.new("RGBA", (4, 1), (10, 20, 30, 255))
    cutout = original.copy()
    cutout.putalpha(Image.new("L", (4, 1), 0))
    mask = Image.new("L", (4, 1), 0)
    mask.putpixel((0, 0), 255)
    mask.putpixel((1, 0), 255)

    result = background.restrict_to_mask(cutout, original, mask)
    alpha = result.getchannel("A")
    check("selection alpha removed", alpha.getpixel((0, 0)), 0)
    check("unselected alpha kept", alpha.getpixel((3, 0)), 255)
    check("unselected rgb kept", result.getpixel((3, 0)), (10, 20, 30, 255))

    unresized = background.restrict_to_mask(cutout, original, None)
    check("no mask keeps cutout", unresized.getchannel("A").getpixel((0, 0)), 0)


def test_restrict_to_mask_soft_edge():
    original = Image.new("RGBA", (2, 1), (0, 0, 0, 255))
    cutout = original.copy()
    cutout.putalpha(Image.new("L", (2, 1), 0))
    mask = Image.new("L", (2, 1), 128)
    alpha = background.restrict_to_mask(
        cutout, original, mask
    ).getchannel("A")
    check("soft mask blends alpha", alpha.getpixel((0, 0)) in (127, 128), True)


def test_restrict_to_mask_resizes():
    original = Image.new("RGBA", (8, 8), (5, 5, 5, 255))
    cutout = original.copy()
    cutout.putalpha(Image.new("L", (8, 8), 0))
    mask = Image.new("L", (4, 4), 0)
    for x in range(2):
        for y in range(4):
            mask.putpixel((x, y), 255)

    result = background.restrict_to_mask(cutout, original, mask)
    check("mask resized to source", result.size, (8, 8))
    check("left half removed", result.getchannel("A").getpixel((0, 0)), 0)
    check("right half kept", result.getchannel("A").getpixel((7, 0)), 255)


def test_layer_selection_mask():
    from paintapp.app import PaintApp

    class _Layer:
        offset = (2, 3)
        image = Image.new("RGBA", (4, 4))

    document = Image.new("L", (10, 10), 0)
    document.paste(255, (2, 3, 5, 5))
    local = PaintApp._layer_selection_mask(None, _Layer(), document)
    check("local mask size", local.size, (4, 4))
    check("local mask offset hit", local.getpixel((0, 0)), 255)
    check("local mask outside", local.getpixel((3, 3)), 0)

    class _Shifted:
        offset = (-1, -1)
        image = Image.new("RGBA", (4, 4))

    shifted_doc = Image.new("L", (10, 10), 0)
    shifted_doc.paste(255, (0, 0, 2, 2))
    shifted = PaintApp._layer_selection_mask(None, _Shifted(), shifted_doc)
    check("negative offset hit", shifted.getpixel((2, 2)), 255)
    check("negative offset miss", shifted.getpixel((0, 0)), 0)



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
