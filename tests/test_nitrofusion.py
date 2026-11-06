"""Unit tests for NitroFusion2 support (a Z-Image-Turbo distillation).

Run: .venv\\Scripts\\python.exe tests\\test_nitrofusion.py

NitroFusion2 reuses the Z-Image family wholesale: the same NextDiT, Qwen3-4B
text encoder and Flux VAE. What is added here is the 1-2 step recipe, the linear
sigma ramp, and the per-step de-distillation LoRA schedule. Nothing loads the
real weights.
"""

import os
import sys

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import lora, paths, sampler  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_kind_detection():
    check("nf2 gguf", paths.z_image_kind("models/nf2-Q4_K_S.gguf"), "nitrofusion")
    check("nitrofusion name", paths.z_image_kind("nitrofusion2_1step.safetensors"), "nitrofusion")
    check("nf2 underscore", paths.z_image_kind("foo_nf2_bf16.safetensors"), "nitrofusion")
    check("plain z-image", paths.z_image_kind("z_image_turbo_bf16.safetensors"), "z_image")
    check("no false positive", paths.z_image_kind("conf2.safetensors"), "z_image")
    check("empty", paths.z_image_kind(""), "z_image")


def test_classify_model():
    check("nf2 routes to z_image", paths.classify_model("nf2-Q8_0.gguf"), ("Z-Image", "z_image"))
    check("nitrofusion routes", paths.classify_model("ChenDY/NitroFusion2"), ("Z-Image", "z_image"))


def test_family_defaults():
    nf2 = paths.family_defaults("z_image", "nf2-Q4_K_S.gguf")
    check("nf2 steps", nf2["steps"], 2)
    check("nf2 cfg", nf2["cfg"], 1.0)
    check("nf2 sampler", nf2["sampler"], "euler")
    check("nf2 size", (nf2["width"], nf2["height"]), (1024, 1024))

    turbo = paths.family_defaults("z_image", "z_image_turbo_bf16.safetensors")
    check("turbo steps", turbo["steps"], 8)
    check("turbo sampler", turbo["sampler"], "res_multistep")


def test_linear_sigmas():
    check("two steps", sampler.linear_sigmas(2).tolist(), [1.0, 0.5, 0.0])
    check("one step", sampler.linear_sigmas(1).tolist(), [1.0, 0.0])
    check("four steps", sampler.linear_sigmas(4).tolist(), [1.0, 0.75, 0.5, 0.25, 0.0])


def test_is_dedistill():
    check("dedistill lora", lora.is_dedistill("nf2-dedistill-lora.safetensors"), True)
    check("other lora", lora.is_dedistill("style-lora.safetensors"), False)
    check("empty", lora.is_dedistill(""), False)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3, bias=False)


def _lora_sd():
    return {
        "proj.lora_A.weight": torch.randn(2, 4),
        "proj.lora_B.weight": torch.randn(3, 2),
    }


def test_set_scale_toggles_lora():
    torch.manual_seed(0)
    model = _Model()
    with torch.no_grad():
        model.proj.weight.normal_()
    applied = lora.apply_lora(model, _lora_sd(), strength=1.0, source="dedistill.safetensors")
    check("lora applied", applied, 1)

    x = torch.randn(1, 4)
    plain = torch.nn.functional.linear(x, model.proj.weight)
    on = model.proj(x)
    check("lora changes output", bool((on - plain).abs().max() > 1e-6), True)

    lora.set_scale(model, "dedistill.safetensors", 0.0)
    off = model.proj(x)
    check("factor 0 == plain", bool(torch.allclose(off, plain, atol=1e-6)), True)

    lora.set_scale(model, "dedistill.safetensors", 1.0)
    back = model.proj(x)
    check("factor 1 restores", bool(torch.allclose(back, on, atol=1e-6)), True)


def test_apply_model_schedules_by_step():
    torch.manual_seed(0)
    dit = _Model()
    with torch.no_grad():
        dit.proj.weight.normal_()
    lora.apply_lora(dit, _lora_sd(), strength=1.0, source="dedistill.safetensors")
    model = sampler.ZImageApplyModel(
        dit, lora_schedule=[("dedistill.safetensors", (1.0, 0.0))]
    )

    model.set_step(0, 2)
    check("first step factor", dit.proj._pai_lora[0][5], 1.0)
    model.set_step(1, 2)
    check("second step factor", dit.proj._pai_lora[0][5], 0.0)


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
