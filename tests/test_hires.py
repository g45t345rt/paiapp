"""Unit tests for the hires-fix (latent upscale) second pass.

Run: .venv\\Scripts\\python.exe tests\\test_hires.py

Exercises the pure planning/resize helpers and drives :func:`_hires_fix` with a
trivial identity apply-model, so the sampler plumbing and the latent/pixel
branches are covered without any real DiT or VAE.
"""

import os
import sys

import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import paths  # noqa: E402
from engine import pipeline  # noqa: E402
from engine.sampler import ModelSamplingFlow  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


class _IdentityModel:
    """A stand-in apply-model: its ``x0`` prediction is the input itself."""

    def apply_model(self, x, sigma, context):
        return x


def _cfg(**overrides):
    cfg = {
        "upscale_type": "latent_bilinear",
        "upscale_by": 2.0,
        "upscale_denoise": 0.5,
        "steps": 4,
        "cfg": 1.0,
        "scheduler": "simple",
        "seed": 7,
    }
    cfg.update(overrides)
    return cfg


def test_upscale_spec():
    check("spec off for none", pipeline._upscale_spec({"upscale_type": "none"}), None)
    check("spec off when missing", pipeline._upscale_spec({}), None)
    check("spec off for unknown", pipeline._upscale_spec({"upscale_type": "bogus"}), None)
    check("spec off factor<=1", pipeline._upscale_spec(
        {"upscale_type": "pixel", "upscale_by": 1.0}), None)
    check("spec off denoise<=0", pipeline._upscale_spec(
        {"upscale_type": "pixel", "upscale_by": 2.0, "upscale_denoise": 0.0}), None)
    check("spec method/factor/denoise/steps", pipeline._upscale_spec(
        {"upscale_type": "pixel", "upscale_by": 1.5, "upscale_denoise": 0.4, "steps": 12}),
        ("pixel", 1.5, 0.4, 12))
    check("spec defaults steps from steps", pipeline._upscale_spec(
        {"upscale_type": "latent_nearest", "upscale_by": 2.0, "upscale_denoise": 0.5}),
        ("latent_nearest", 2.0, 0.5, 10))


def test_resize_latent():
    x = torch.rand(1, 4, 8, 8)
    up = pipeline._resize_latent(x, "latent_nearest", 16, 16)
    check("4d nearest shape", tuple(up.shape), (1, 4, 16, 16))
    # Nearest doubles each pixel, so the top-left 2x2 block is the source pixel.
    check("4d nearest value", float(up[0, 0, 0, 0]), float(x[0, 0, 0, 0]))
    check("4d nearest value 2", float(up[0, 0, 1, 1]), float(x[0, 0, 0, 0]))

    y = torch.rand(1, 16, 1, 8, 8)
    upy = pipeline._resize_latent(y, "latent_bicubic", 20, 12)
    check("5d bicubic shape", tuple(upy.shape), (1, 16, 1, 20, 12))

    odd = pipeline._resize_latent(torch.rand(1, 4, 7, 5), "latent_bilinear", 14, 10)
    check("non-square shape", tuple(odd.shape), (1, 4, 14, 10))


def test_hires_fix_latent():
    samples = torch.randn(1, 4, 8, 8)
    out = pipeline._hires_fix(
        _cfg(), samples, ModelSamplingFlow(), _IdentityModel(), None, None,
        "euler", 7, None, None,
        decode=lambda lat: (_ for _ in ()).throw(AssertionError("pixel ignored")),
        encode=lambda pil, h, w: (_ for _ in ()).throw(AssertionError("pixel ignored")),
    )
    check("latent out shape", tuple(out.shape), (1, 4, 16, 16))
    check("latent out finite", bool(torch.isfinite(out).all()), True)


def test_hires_fix_pixel():
    seen = {}

    def decode(lat):
        h, w = int(lat.shape[-2]) * 8, int(lat.shape[-1]) * 8
        return Image.new("RGB", (w, h), (0, 0, 0))

    def encode(pil, h, w):
        seen["size"] = (h, w)
        seen["pil"] = pil.size
        return torch.zeros(1, 4, h, w)

    samples = torch.randn(1, 4, 8, 8)
    out = pipeline._hires_fix(
        _cfg(upscale_type="pixel", upscale_by=2.0), samples, ModelSamplingFlow(),
        _IdentityModel(), None, None, "euler", 7, None, None, decode, encode,
    )
    check("pixel out shape", tuple(out.shape), (1, 4, 16, 16))
    check("pixel encode grid", seen["size"], (16, 16))
    check("pixel resized image", seen["pil"], (128, 128))


def test_types_constant():
    check("types include none", "none" in paths.UPSCALE_TYPES, True)
    for kind in ("latent_nearest", "latent_bilinear", "latent_bicubic",
                 "latent_bislerp", "pixel"):
        check(f"types include {kind}", kind in paths.UPSCALE_TYPES, True)
    check("default off", paths.DEFAULTS.get("upscale_type"), "none")


def main():
    for fn in (test_upscale_spec, test_resize_latent, test_hires_fix_latent,
               test_hires_fix_pixel, test_types_constant):
        fn()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print(" -", line)
        sys.exit(1)
    print("\nall hires tests passed")


if __name__ == "__main__":
    main()
