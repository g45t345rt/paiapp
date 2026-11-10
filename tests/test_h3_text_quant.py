"""Unit tests for H3 text encoder quantization (int8 + NVFP4/AWQ).

Run: .venv\\Scripts\\python.exe tests\\test_h3_text_quant.py

Covers the pure-torch NVFP4 dequant and AWQ pre_quant_scale path added for the
Comfy-Org ``qwen3vl_32b_minimax_h3_nvfp4_awq`` checkpoint. No model weights are
needed.
"""

import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.h3_text import (  # noqa: E402
    Qwen3VLTextStreamEncoder, _embed_rows, _load_layer, _mlp,
)
from engine.quant_formats import (  # noqa: E402
    _FP4_E2M1, _copy_weight, _nvfp4_scales_to_linear,
)
from engine.safetensors_io import open_safetensors  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _swizzle_scales(scales):
    rows, cols = scales.shape
    rows_padded, cols_padded = _round_up(rows, 128), _round_up(cols, 4)
    padded = torch.zeros((rows_padded, cols_padded), dtype=scales.dtype)
    padded[:rows, :cols] = scales
    reshaped = padded.reshape(1, rows_padded // 128, 4, 32, cols_padded // 4, 4)
    return reshaped.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(rows_padded, cols_padded)


def test_scale_swizzle_roundtrip():
    torch.manual_seed(0)
    scales = torch.randn(128, 8, dtype=torch.float32)
    stored = _swizzle_scales(scales)
    check("scale roundtrip",
          bool(torch.allclose(_nvfp4_scales_to_linear(stored), scales)), True)


def test_nvfp4_copy_weight():
    from safetensors.torch import save_file

    out_dim, in_dim = 128, 64
    codes = torch.arange(out_dim * in_dim, dtype=torch.int64).reshape(out_dim, in_dim) % 16
    packed = ((codes[:, 0::2] << 4) | codes[:, 1::2]).to(torch.uint8)
    scales = torch.ones(out_dim, in_dim // 16, dtype=torch.float8_e4m3fn)
    path = os.path.join(tempfile.mkdtemp(), "nvfp4.safetensors")
    save_file({
        "w.weight": packed,
        "w.weight_scale": scales,
        "w.weight_scale_2": torch.tensor([1.0], dtype=torch.float32),
    }, path)
    expected = torch.tensor(_FP4_E2M1, dtype=torch.float32)[codes]
    with open_safetensors(path) as handle:
        names = set(handle.keys())
        dst = torch.empty(out_dim, in_dim, dtype=torch.bfloat16)
        _copy_weight(handle, names, "w.weight", dst, torch.bfloat16)
    check("nvfp4 dequant shape", tuple(dst.shape), (out_dim, in_dim))
    check("nvfp4 dequant values",
          bool(torch.allclose(dst.float(), expected, atol=0.0)), True)


def test_embed_rows_int8():
    from safetensors.torch import save_file

    embed = torch.tensor([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ], dtype=torch.int8)
    scale = torch.tensor([[0.5], [2.0], [1.0], [0.25]], dtype=torch.float32)
    path = os.path.join(tempfile.mkdtemp(), "embed.safetensors")
    save_file({"model.embed_tokens.weight": embed,
               "model.embed_tokens.weight_scale": scale}, path)
    with open_safetensors(path) as handle:
        names = set(handle.keys())
        rows = _embed_rows(handle, [1, 3], torch.bfloat16, names)
    expected = torch.stack([embed[1].float() * 2.0, embed[3].float() * 0.25])
    check("int8 embedding dequant", bool(torch.allclose(rows.float(), expected)), True)


def test_pre_quant_scale_applied():
    x = torch.randn(1, 3, 8)
    gate = torch.randn(16, 8)
    up = torch.randn(16, 8)
    down = torch.randn(8, 16)
    mlp_pre = torch.rand(16) + 0.5
    got = _mlp(x, {"gate": gate, "up": up, "down": down, "down_pre": mlp_pre})
    h = torch.nn.functional.silu(x @ gate.T) * (x @ up.T)
    want = (h * mlp_pre) @ down.T
    check("down_proj pre_quant_scale", bool(torch.allclose(got, want, atol=1e-5)), True)


def _nvfp4_layer_rows(hidden, intermediate):
    rows = {
        "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
        "self_attn.q_norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "self_attn.k_norm.weight": torch.ones(8, dtype=torch.bfloat16),
    }

    def quantized(prefix, out_dim, in_dim, pre_quant=False):
        rows[prefix + ".weight"] = torch.zeros(out_dim, in_dim // 2, dtype=torch.uint8)
        rows[prefix + ".weight_scale"] = torch.ones(
            out_dim, in_dim // 16, dtype=torch.float8_e4m3fn)
        rows[prefix + ".weight_scale_2"] = torch.tensor([1.0], dtype=torch.float32)
        if pre_quant:
            rows[prefix + ".pre_quant_scale"] = (
                torch.rand(in_dim, dtype=torch.bfloat16) + 0.5)

    quantized("self_attn.q_proj", hidden, hidden)
    quantized("self_attn.k_proj", hidden, hidden)
    quantized("self_attn.v_proj", hidden, hidden)
    quantized("self_attn.o_proj", hidden, hidden, pre_quant=True)
    quantized("mlp.gate_proj", intermediate, hidden)
    quantized("mlp.up_proj", intermediate, hidden)
    quantized("mlp.down_proj", hidden, intermediate, pre_quant=True)
    return rows


def test_load_layer_nvfp4():
    from safetensors.torch import save_file

    hidden, intermediate = 128, 256
    rows = _nvfp4_layer_rows(hidden, intermediate)
    path = os.path.join(tempfile.mkdtemp(), "layer.safetensors")
    save_file({f"model.layers.0.{key}": value for key, value in rows.items()}, path)
    dims = {
        "hidden": hidden, "intermediate": intermediate,
        "q_dim": hidden, "kv_dim": hidden, "head_dim": 8,
    }
    with open_safetensors(path) as handle:
        scratch = _load_layer(handle, set(handle.keys()), 0,
                              torch.bfloat16, dims, {})
    check("layer q_proj", tuple(scratch["q_proj"].shape), (hidden, hidden))
    check("layer down", tuple(scratch["down"].shape), (hidden, intermediate))
    check("layer o_pre", tuple(scratch["o_pre"].shape), (hidden,))
    check("layer down_pre", tuple(scratch["down_pre"].shape), (intermediate,))
    check("layer down value", float(scratch["down"].float().abs().max()), 0.0)


def test_stream_forward_nvfp4():
    from safetensors.torch import save_file

    hidden, intermediate = 128, 256
    rows = {f"model.layers.0.{key}": value
            for key, value in _nvfp4_layer_rows(hidden, intermediate).items()}
    embed = torch.randint(-4, 5, (4, hidden), dtype=torch.int8)
    rows["model.embed_tokens.weight"] = embed
    rows["model.embed_tokens.weight_scale"] = torch.ones(4, 1, dtype=torch.float32)
    path = os.path.join(tempfile.mkdtemp(), "tiny_nvfp4.safetensors")
    save_file(rows, path)
    encoder = Qwen3VLTextStreamEncoder(
        path, dtype=torch.float32,
        config={"vocab": 4, "hidden": hidden, "intermediate": intermediate,
                "num_layers": 1, "heads": 16, "kv_heads": 16, "head_dim": 8,
                "eps": 1e-6, "rope_theta": 5_000_000.0},
    )
    out = encoder._forward([0, 1, 2])
    check("nvfp4 forward shape", tuple(out.shape), (1, 3, hidden))
    check("nvfp4 forward finite", bool(torch.isfinite(out).all()), True)


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