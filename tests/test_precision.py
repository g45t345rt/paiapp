"""Unit tests for runtime placement policy and tiled VAE decode.

Run: .venv\\Scripts\\python.exe tests\\test_precision.py
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import precision  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def close(name, got, want, tol=1e-4):
    if got.shape != want.shape or not torch.allclose(got, want, atol=tol, rtol=tol):
        FAILED.append(f"{name}: tensors differ (max {float((got - want).abs().max()):.2e})")
    else:
        print(f"ok  {name}")


def restore():
    precision.configure(
        fp8_mode="auto", vram_reserve=precision.DEFAULT_RESERVE,
        stream_reserve=precision.STREAM_RESERVE, keep_resident=True,
        compile_vae=True, tiled_decode=False, tile_size=512, tile_overlap=64,
        text_encoder_device="cpu", disk_stream=False, ram_budget=0, prefetch_depth=0,
    )


def test_configure_updates_known_keys_only():
    restore()
    got = precision.configure(bogus_key=1, vram_reserve=123)
    check("vram updated", got["vram_reserve"], 123)
    check("unknown ignored", "bogus_key" in got, False)
    restore()


def test_want_fp8_policy():
    restore()
    check("explicit on", precision._want_fp8("on"), True)
    check("explicit off", precision._want_fp8("off"), False)
    check("auto default", precision._want_fp8("auto"), None)
    precision.configure(fp8_mode="fp8")
    check("policy fp8", precision._want_fp8("auto"), True)
    precision.configure(fp8_mode="full")
    check("policy full", precision._want_fp8("auto"), False)
    restore()


class _HeadModel:
    """Minimal stand-in exposing the placement interface ``place_dit`` uses."""

    def __init__(self, blocks=()):
        self.blocks = list(blocks)
        self.placed = None
        self.resident = None

    def place_head(self, device):
        self.placed = device

    def enable_block_offload(self, storage, compute):
        self.storage, self.compute = storage, compute

    def set_resident_blocks(self, count, device):
        self.resident = count

    def to(self, device):
        self.moved = device
        return self


def test_place_dit_streaming_model_is_left_alone():
    restore()
    model = _HeadModel()
    model._streaming_source = object()
    model._streaming_label = "disk-streamed(3 blocks)"
    label = precision.place_dit(model, "cpu")
    check("streaming label", label, "disk-streamed(3 blocks)")
    check("head untouched", model.placed, None)


def test_place_dit_cpu_calls_place_head():
    restore()
    model = _HeadModel()
    check("cpu label", precision.place_dit(model, "cpu"), "cpu")
    check("place_head called", model.placed, "cpu")


def test_keep_resident_off_pins_none():
    restore()
    precision.configure(keep_resident=False)
    model = _HeadModel(blocks=[nn.Linear(2, 2) for _ in range(3)])
    model.set_resident_blocks = lambda count, device: setattr(model, "resident", count)
    count = precision._resident_block_count(model, 0)
    check("no resident when disabled", count, 0)
    restore()


def _tiny_qwen_dit(layers=3):
    from engine.qwen_image import QwenImage21Transformer2DModel

    return QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=layers, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2, axes_dims_rope=(4, 2, 2),
    )


def test_qwen_dit_exposes_placement_interface():
    model = _tiny_qwen_dit()
    check("blocks aliases transformer_blocks", model.blocks is model.transformer_blocks, True)
    model.enable_block_offload("cpu", "cpu")
    model.place_head("cpu")
    model.set_resident_blocks(2, "cpu")
    check("resident count set", model._resident_count, 2)
    model.set_resident_blocks(99, "cpu")
    check("resident count clamped to layer count", model._resident_count, 3)
    model.set_resident_blocks(-5, "cpu")
    check("resident count clamped at zero", model._resident_count, 0)


def test_qwen_dit_streams_blocks_keeping_resident_on_gpu():
    if not torch.cuda.is_available():
        print("skip  qwen block streaming (no cuda)")
        return
    torch.manual_seed(0)
    model = _tiny_qwen_dit().eval()
    # Same order place_dit uses: offload, place the head, then pin the resident blocks.
    model.enable_block_offload("cpu", "cuda")
    model.place_head("cuda")
    model.set_resident_blocks(1, "cuda")
    x = torch.randn(1, 8, 4, 4, device="cuda")
    ctx = torch.randn(1, 4, 32, device="cuda")
    ts = torch.tensor([0.5], device="cuda")
    with torch.no_grad():
        out = model(x, ts, ctx)
    check("streamed forward finite", bool(torch.isfinite(out).all()), True)
    check("resident block stays on gpu", str(next(model.blocks[0].parameters()).device), "cuda:0")
    check("streamed block returns to cpu", str(next(model.blocks[2].parameters()).device), "cpu")


def test_decode_tiled_matches_full_for_local_decoder():
    torch.manual_seed(0)
    latent = torch.randn(1, 3, 40, 40)

    def upsample(tile):
        return F.interpolate(tile, scale_factor=8, mode="nearest")

    full = upsample(latent)
    tiled = precision.decode_tiled(upsample, latent, tile_size=16, overlap=4)
    close("tiled nearest upsample equals full", tiled, full)


def test_decode_tiled_noop_when_small():
    calls = {"n": 0}

    def decode(tile):
        calls["n"] += 1
        return tile * 2

    latent = torch.randn(1, 2, 8, 8)
    out = precision.decode_tiled(decode, latent, tile_size=64, overlap=8)
    check("single call for small latent", calls["n"], 1)
    close("passthrough decoder", out, latent * 2)


def test_decode_tiled_noop_non_4d():
    calls = {"n": 0}

    def decode(tile):
        calls["n"] += 1
        return tile

    latent = torch.randn(1, 2, 3, 40, 40)
    precision.decode_tiled(decode, latent, tile_size=16, overlap=4)
    check("non-4d passthrough", calls["n"], 1)


def test_to_fp8_on_meta_module_stays_storage_free():
    """A meta-built module must swap to empty fp8, not fault copying no-data.

    ``h3.load_dit`` builds its DiT on ``meta`` and then calls ``to_fp8`` before
    filling the fp8 weights, so this is the no-copy branch.
    """
    try:
        from accelerate import init_empty_weights  # noqa: F401
    except Exception:
        print("skip  meta to_fp8 (no accelerate)")
        return

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 4)

    with precision.build_on_meta(torch.bfloat16, _M) as model:
        pass
    check("meta weight", model.proj.weight.is_meta, True)
    precision.to_fp8(model)
    check("swapped to fp8", isinstance(model.proj, precision.FP8Linear), True)
    check("fp8 weight is real", model.proj.weight.is_meta, False)
    check("fp8 weight dtype", model.proj.weight.dtype, torch.float8_e4m3fn)
    check("fp8 weight shape", tuple(model.proj.weight.shape), (4, 8))


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