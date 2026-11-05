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
