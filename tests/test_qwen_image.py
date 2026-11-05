"""Unit tests for the Qwen-Image DiT weight loader (engine.qwen_image).

Covers the safetensors source: plain bf16 passthrough and ComfyUI
``int8_tensorwise`` + ``convrot`` dequantisation, with and without the
``model.diffusion_model.`` container prefix. No full model file is needed.

Run: .venv\\Scripts\\python.exe tests\\test_qwen_image.py
"""

import json
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import moss_text  # noqa: E402
from engine import qwen_image  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_safetensors_int8_convrot():
    from safetensors.torch import save_file

    torch.manual_seed(0)
    weight = torch.randint(-127, 128, (4, 256), dtype=torch.int8)
    scale = torch.rand(4, 1) * 0.01 + 0.001
    conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
    state = {
        "transformer_blocks.0.attn.to_q.weight": weight,
        "transformer_blocks.0.attn.to_q.weight_scale": scale,
        "transformer_blocks.0.attn.to_q.comfy_quant": torch.tensor(list(conf), dtype=torch.uint8),
        "transformer_blocks.0.attn.norm_q.weight": torch.ones(256),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dit.safetensors")
        save_file(state, path)
        names, get, raw, bytes_of, close = qwen_image._model_source(path, torch.float32)
        try:
            got = get("transformer_blocks.0.attn.to_q.weight")
            want = moss_text._unrotate(weight.to(torch.float32) * scale, 256)
            check("int8 convrot un-rotates", bool(torch.allclose(got, want, atol=1e-5)), True)
            check("int8 convrot dtype is float", got.dtype, torch.float32)
            check("bf16 passthrough", bool(torch.equal(get("transformer_blocks.0.attn.norm_q.weight"),
                                                        torch.ones(256))), True)
            check("bytes_of uses shape", bytes_of("transformer_blocks.0.attn.to_q.weight"), 4 * 256 * 2)
            check("scale key not a model name",
                  "transformer_blocks.0.attn.to_q.weight_scale" in names, False)
        finally:
            close()


def test_safetensors_prefix_stripped():
    import tempfile

    from safetensors.torch import save_file

    state = {"model.diffusion_model.img_in.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3)}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dit.safetensors")
        save_file(state, path)
        names, get, raw, bytes_of, close = qwen_image._model_source(path, torch.float32)
        try:
            check("prefix stripped from name", "img_in.weight" in names, True)
            check("get returns tensor", bool(torch.equal(get("img_in.weight"),
                                                         state["model.diffusion_model.img_in.weight"])), True)
        finally:
            close()


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
