"""Unit tests for the MOSS text encoder's Comfy W4A8 (``asym_w4a8_int8``) loader.

Run: .venv\\Scripts\\python.exe tests\\test_moss_text_quant.py

The real Qwen3.5-9B W4A8 file is ~6 GB, so this covers the packed-int4 decode,
codebook/group/channel scales, ConvRot un-rotation, and the ``comfy_quant``
routing with synthetic tensors.
"""

import json
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.quant_formats import _copy_weight, _unrotate  # noqa: E402
from engine.safetensors_io import open_safetensors  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _reference(codes, s_rel, s_channel, codebook, group_size, convrot):
    n, k = codes.shape
    groups = k // group_size
    values = codebook[codes].view(n, groups, group_size)
    value = values * s_rel.unsqueeze(-1)
    value = value.round().clamp_(-127, 127)
    value = value * s_channel.view(-1, 1, 1)
    value = value.reshape(n, k)
    return _unrotate(value, 256) if convrot else value


def _write_w4a8(path, convrot):
    n, k, group_size = 4, 256, 16
    groups = k // group_size
    codes = torch.arange(n * k, dtype=torch.int64).reshape(n, k) % 16
    packed = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8)
    codebook = torch.linspace(-0.98, 0.98, 16, dtype=torch.float32)
    s_rel = (torch.rand(n, groups, dtype=torch.float32) + 0.5).to(torch.float8_e4m3fn)
    s_channel = torch.rand(n, dtype=torch.float32) + 0.5
    conf = {"format": "asym_w4a8_int8", "group_size": group_size,
            "convrot": convrot, "convrot_groupsize": 256}
    from safetensors.torch import save_file
    save_file({
        "w.weight": packed,
        "w.weight_s_rel": s_rel,
        "w.weight_s_channel": s_channel,
        "w.weight_codebook": codebook,
        "w.comfy_quant": torch.tensor(
            list(json.dumps(conf).encode("utf-8")), dtype=torch.uint8),
    }, path)
    return codes, s_rel.float(), s_channel, codebook


def _load(path):
    with open_safetensors(path) as handle:
        dst = torch.empty(4, 256, dtype=torch.float32)
        _copy_weight(handle, set(handle.keys()), "w.weight", dst, torch.float32)
    return dst


def test_w4a8_no_convrot():
    path = os.path.join(tempfile.mkdtemp(), "w4a8_plain.safetensors")
    codes, s_rel, s_channel, codebook = _write_w4a8(path, convrot=False)
    expected = _reference(codes, s_rel, s_channel, codebook, 16, False)
    check("w4a8 decode", bool(torch.allclose(_load(path), expected, atol=1e-5)), True)


def test_w4a8_convrot():
    path = os.path.join(tempfile.mkdtemp(), "w4a8_rot.safetensors")
    codes, s_rel, s_channel, codebook = _write_w4a8(path, convrot=True)
    expected = _reference(codes, s_rel, s_channel, codebook, 16, True)
    check("w4a8 convrot decode",
          bool(torch.allclose(_load(path), expected, atol=1e-5)), True)


def test_w4a8_without_codebook():
    path = os.path.join(tempfile.mkdtemp(), "w4a8_uniform.safetensors")
    codes, s_rel, s_channel, _ = _write_w4a8(path, convrot=False)
    # Drop the codebook: the loader must fall back to uniform levels c - 8.
    from safetensors.torch import load_file, save_file
    tensors = load_file(path)
    tensors.pop("w.weight_codebook")
    save_file(tensors, path)
    expected = _reference(
        codes, s_rel, s_channel, torch.arange(-8, 8, dtype=torch.float32), 16, False)
    check("w4a8 uniform decode",
          bool(torch.allclose(_load(path), expected, atol=1e-5)), True)


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