"""Unit tests for the SeedVR2 upscaler port.

Run: .venv\\Scripts\\python.exe tests\\test_seedvr2.py

Covers the checkpoint detection, key prefix stripping, the ``NaDiT`` forward
plumbing and VAE round-trip (with tiny random weights, no real checkpoint), and
the end-to-end :class:`SeedVR2Upscaler`.
"""

import os
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import seedvr2  # noqa: E402
from engine.seedvr2 import _compat as C  # noqa: E402
from engine.seedvr2.model import NaDiT  # noqa: E402
from engine.seedvr2.vae import VideoAutoencoderKL  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


_TINY_CFG = dict(
    image_model="seedvr2", vid_dim=64, heads=2, head_dim=32, num_layers=2,
    mm_layers=1, norm_eps=1e-5, mlp_type="swiglu", vid_out_norm=True,
    txt_in_dim=5120, expand_ratio=2, patch_size=(1, 2, 2), rope_dim=32,
    rope_type="mmrope3d",
)


def _init(mod):
    for p in mod.parameters():
        nn.init.normal_(p, std=0.02)
    for b in mod.buffers():
        if torch.is_floating_point(b):
            nn.init.normal_(b, std=0.02)
    return mod


def _tiny_dit():
    torch.manual_seed(0)
    return _init(NaDiT(operations=C.ops, device="cpu", dtype=torch.float32, **_TINY_CFG)).eval()


def _tiny_vae():
    torch.manual_seed(0)
    vae = VideoAutoencoderKL()
    _init(vae)
    vae.eval()
    vae.device = torch.device("cpu")
    return vae


def test_is_seedvr2_file():
    check("dit file", seedvr2.is_seedvr2_file("seedvr2_3b_fp16.safetensors"), True)
    check("dit 7b", seedvr2.is_seedvr2_file("C:/m/seedvr2_7b_int8_convrot.safetensors"), True)
    check("vae excluded", seedvr2.is_seedvr2_file("seedvr2_ema_vae_fp16.safetensors"), False)
    check("other upscaler", seedvr2.is_seedvr2_file("RealESRGAN_x4plus.pth"), False)
    check("other safetensors", seedvr2.is_seedvr2_file("4x-UltraSharp.safetensors"), False)
    check("empty", seedvr2.is_seedvr2_file(""), False)


def test_dit_config_detection():
    cfg = seedvr2.dit_config_from_keys(["blocks.31.mlp.all.proj_in_gate.weight"])
    check("3b vid_dim", cfg["vid_dim"], 2560)
    check("3b layers", cfg["num_layers"], 32)
    check("3b vid_out_norm", cfg["vid_out_norm"], True)
    cfg = seedvr2.dit_config_from_keys(["blocks.35.mlp.all.proj_in_gate.weight"])
    check("7b shared vid_dim", cfg["vid_dim"], 3072)
    check("7b shared mm_layers", cfg["mm_layers"], 10)
    check("7b shared mlp", cfg["mlp_type"], "swiglu")
    cfg = seedvr2.dit_config_from_keys(["blocks.35.mlp.vid.proj_out.weight"])
    check("7b separate mm_layers", cfg["mm_layers"], 36)
    check("7b separate mlp", cfg["mlp_type"], "normal")
    check("unknown", seedvr2.dit_config_from_keys(["blocks.0.attn.q.weight"]), None)


def test_strip_prefix():
    check("diffusion prefix",
          seedvr2._strip("model.diffusion_model.blocks.0.attn.q.weight"),
          "blocks.0.attn.q.weight")
    check("bare", seedvr2._strip("blocks.0.attn.q.weight"), "blocks.0.attn.q.weight")
    check("model prefix", seedvr2._strip("model.blocks.0.w"), "blocks.0.w")


def test_nadit_forward():
    dit = _tiny_dit()
    x = torch.randn(1, 16, 1, 8, 8)
    cond = torch.randn(1, 17, 1, 8, 8)
    with torch.no_grad():
        out = dit(x, torch.tensor([1000.0]), None, condition=cond,
                  transformer_options={"cond_or_uncond": [0]})
    check("nadit shape", tuple(out.shape), (1, 16, 1, 8, 8))
    check("nadit finite", bool(torch.isfinite(out).all()), True)


def test_vae_roundtrip():
    vae = _tiny_vae()
    img = torch.randn(1, 3, 1, 32, 32)
    with torch.no_grad():
        raw = vae.encode(img)
        dec = vae.decode_(raw)
    check("vae latent channels", raw.shape[1], 16)
    check("vae latent spatial", tuple(raw.shape[-2:]), (4, 4))
    check("vae decode shape", tuple(dec.shape), (1, 3, 1, 32, 32))
    check("vae decode finite", bool(torch.isfinite(dec).all()), True)


def test_upscaler_upscale():
    dit = _tiny_dit()
    vae = _tiny_vae()
    up = seedvr2.SeedVR2Upscaler(dit, vae, "cpu", {"vid_dim": 64})
    check("label 3b by default", up.label, "SeedVR2 3B")
    arr = np.random.rand(24, 32, 3).astype("float32")
    out = up.upscale(arr, scale=2.0, seed=1, color="lab")
    check("upscale ndim", out.ndim, 3)
    check("upscale channels", out.shape[2], 3)
    check("upscale even dims", (out.shape[0] % 2, out.shape[1] % 2), (0, 0))
    check("upscale range", (float(out.min()) >= 0.0, float(out.max()) <= 1.0), (True, True))
    out = up.upscale(arr, scale=2.0, seed=1, color="none")
    check("upscale none ok", out.shape[2], 3)


def test_load_roundtrip():
    from safetensors.torch import save_file
    dit = _tiny_dit()
    vae = _tiny_vae()
    with tempfile.TemporaryDirectory() as tmp:
        dit_path = os.path.join(tmp, "seedvr2_3b_tiny.safetensors")
        vae_path = os.path.join(tmp, "seedvr2_ema_vae_tiny.safetensors")
        save_file({f"model.diffusion_model.{k}": v for k, v in dit.state_dict().items()},
                  dit_path)
        save_file({f"first_stage_model.{k}": v for k, v in vae.state_dict().items()},
                  vae_path)

        loaded, cfg, info = seedvr2.load_dit(dit_path, device="cpu", dtype=torch.float32,
                                             config=_TINY_CFG)
        check("load_dit missing", list(info.missing_keys), [])
        check("load_dit unexpected", list(info.unexpected_keys), [])
        with torch.no_grad():
            out = loaded(torch.randn(1, 16, 1, 8, 8), torch.tensor([1000.0]), None,
                         condition=torch.randn(1, 17, 1, 8, 8),
                         transformer_options={"cond_or_uncond": [0]})
        check("loaded dit shapes", tuple(out.shape), (1, 16, 1, 8, 8))

        loaded_vae = seedvr2.load_vae(vae_path, device="cpu", dtype=torch.float32)
        with torch.no_grad():
            dec = loaded_vae.decode_(loaded_vae.encode(torch.randn(1, 3, 1, 32, 32)))
        check("loaded vae shape", tuple(dec.shape), (1, 3, 1, 32, 32))


def test_cache_unload():
    up = seedvr2.SeedVR2Upscaler(_tiny_dit(), _tiny_vae(), "cpu", {"vid_dim": 64})
    key = os.path.abspath(os.path.join(tempfile.gettempdir(), "seedvr2_tiny.safetensors"))
    seedvr2._models[key] = up
    try:
        records = seedvr2.loaded_models()
        check("cache record count", len(records), 1)
        check("cache family", records[0]["family"].startswith("Upscale ("), True)
        check("cache components", sorted(c["name"] for c in records[0]["components"]),
              ["DiT", "VAE"])
        check("cache key", records[0]["key"], f"seedvr2:{key}")
        check("unload one", seedvr2.unload(f"seedvr2:{key}"), 1)
        check("unload empty", seedvr2.unload(), 0)
    finally:
        seedvr2._models.clear()


def main():
    for fn in (test_is_seedvr2_file, test_dit_config_detection, test_strip_prefix,
               test_nadit_forward, test_vae_roundtrip, test_upscaler_upscale,
               test_load_roundtrip, test_cache_unload):
        fn()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print(" -", line)
        sys.exit(1)
    print("\nall seedvr2 tests passed")


if __name__ == "__main__":
    main()
