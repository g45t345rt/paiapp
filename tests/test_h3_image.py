"""Unit tests for the MiniMax-H3-Image family wiring.

Run: .venv\\Scripts\\python.exe tests\\test_h3_image.py

The real checkpoint is ~20 GB, so these tests only cover the registry, the
packed layout, the PDD head selection, and tiny synthetic model/VAE forwards.
"""

import json
import os
import sys
import tempfile

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import component_kinds, paths  # noqa: E402
from engine import pipeline  # noqa: E402
from engine.h3 import (  # noqa: E402
    H3_IMAGE_PDD_SIGMAS, MiniMaxH3ImageDiT, _pdd_head, image_config,
)
from engine.h3_layout import build_image_packed_sequence  # noqa: E402
from engine.h3_vae import MiniMaxH3ImageVAE  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_registry():
    check("model group registered",
          ("MiniMax H3 Image", "h3_image") in paths.MODEL_GROUPS, True)
    check("base slot is dit", paths._FAMILY_BASE_SLOT.get("h3_image"), "dit")
    check("defaults present", set(paths.H3_IMAGE_DEFAULTS), {"dit", "text", "vae"})
    check("kind registered", component_kinds.KINDS.get("h3_image"),
          ("MiniMax H3 Image", component_kinds.MODEL))
    check("vae kind registered", component_kinds.KINDS.get("h3_image_vae"),
          ("MiniMax H3 Image VAE", component_kinds.VAE))
    check("kind offered", "h3_image" in component_kinds.CATEGORY_KINDS[component_kinds.MODEL], True)
    check("vae kind offered",
          "h3_image_vae" in component_kinds.CATEGORY_KINDS[component_kinds.VAE], True)
    check("text slot", component_kinds.SLOT_KIND[("h3_image", "text")], "qwen3vl_32b")
    check("vae slot", component_kinds.SLOT_KIND[("h3_image", "vae")], "h3_image_vae")


def test_classification():
    check("dit by name",
          paths.classify_model("minimax_h3_image_pruned_int8_convrot.safetensors"),
          ("MiniMax H3 Image", "h3_image"))
    check("vae is not a model",
          paths.classify_model("minimax_h3_image_vae_fp16.safetensors"),
          ("Anima", "anima"))
    check("video h3 unchanged",
          paths.classify_model("minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
          ("MiniMax H3", "h3"))
    check("vae token", component_kinds._vae_token("minimax_h3_image_vae_fp16.safetensors"),
          "h3_image_vae")
    check("video vae token unchanged",
          component_kinds._vae_token("minimax_h3_video_vae_int8_convrot.safetensors"),
          "h3_video_vae")
    check("dit by shapes", component_kinds._model_from_shapes({
        "video_patch_proj.weight": (5376, 96),
        "blocks.0.attn.qkv_proj.weight": (1, 1),
        "final_layer.video_out.weight": (576, 5376),
    }, None), "h3_image")
    check("video h3 by shapes", component_kinds._model_from_shapes({
        "video_patch_proj.weight": (5376, 96),
        "blocks.0.attn.qkv_proj.weight": (1, 1),
        "audio_patch_proj.weight": (5376, 32),
    }, None), "h3")


def test_family_defaults():
    defaults = paths.family_defaults("h3_image")
    check("steps", defaults["steps"], 6)
    check("cfg", defaults["cfg"], 1.0)
    check("size", (defaults["width"], defaults["height"]), (2048, 2048))
    check("recipe built", paths.DEFAULT_RECIPE_NAME in paths.family_recipes("h3_image"), True)


def test_checkpoint_metadata():
    try:
        from safetensors.torch import save_file
    except Exception:
        print("skip checkpoint metadata (safetensors unavailable)")
        return
    config = {"transformer": {
        "pdd_steps": 6, "pdd_sigmas": list(H3_IMAGE_PDD_SIGMAS),
    }}
    path = os.path.join(tempfile.mkdtemp(), "minimax_h3_image_test.safetensors")
    save_file({"a": torch.zeros(1)}, path,
              metadata={"config": json.dumps(config)})
    check("metadata parsed", image_config(path), (6, list(H3_IMAGE_PDD_SIGMAS)))
    check("missing metadata fallback",
          image_config(os.path.join(tempfile.mkdtemp(), "absent.safetensors")),
          (6, list(H3_IMAGE_PDD_SIGMAS)))


def test_layout():
    position_ids, tags, video, text, refs = build_image_packed_sequence(
        4, 2, 4, ref_shapes=[(2, 4)])
    check("sequence length", tuple(position_ids.shape), (8, 3))
    check("video indices", [int(i) for i in video], [6, 7])
    check("text indices", [int(i) for i in text], [0, 1, 2, 3])
    check("ref indices", [[int(i) for i in refs[0]]], [[4, 5]])
    check("tags", [int(v) for v in tags], [1, 1, 1, 1, 0, 0, 0, 0])
    check("text time", [float(v) for v in position_ids[:4, 0]], [0.0, 1.0, 2.0, 3.0])
    check("ref time", [float(v) for v in position_ids[4:6, 0]], [4.0, 4.0])
    check("target time", [float(v) for v in position_ids[6:, 0]], [5.0, 5.0])


def test_pdd_head_selection():
    head = torch.nn.Linear(4, 4, bias=True)
    rows = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    bias = torch.tensor([0.0, 1.0, 2.0, 3.0])
    with torch.no_grad():
        head.weight.copy_(rows)
        head.bias.copy_(bias)
    h = torch.randn(3, 4)
    out = _pdd_head(head, h, [1.0, 0.5, 0.0], 1.0, torch.tensor([1.0, 0.5, 0.0]), 2)
    want = torch.nn.functional.linear(
        h, rows.reshape(2, 2, 4)[0], bias.reshape(2, 2)[0])
    check("first interval uses the full head", bool(torch.allclose(out, want)), True)
    check("output shape", tuple(out.shape), (3, 2))


def test_dit_forward():
    model = MiniMaxH3ImageDiT(
        hidden_size=128, num_layers=1, token_refiner_num_layers=1,
        num_attention_heads=2, attention_head_dim=128, ffn_hidden_size=256,
        latents_dim=24, patch_size=(1, 2, 2), text_dim=32, time_embed_dim=8,
        adaln_curve_size=8, rope_inv_freq_len=16, pdd_steps=6,
    )
    model.adaln_t_table.normal_()
    model.eval()
    with torch.no_grad():
        context = torch.randn(5, 32)
        latent = torch.randn(1, 24, 1, 4, 4)
        layout = build_image_packed_sequence(5, 4, 4, ref_shapes=[(4, 4)])
        layout = (layout[0], layout[1], layout[2], layout[3], list(layout[4]))
        ref = torch.randn(1, 24, 1, 4, 4)
        grid = list(H3_IMAGE_PDD_SIGMAS)
        out = model.forward_image(
            latent, context, layout, 1.0, torch.tensor(grid), grid,
            ref_latents=[ref],
        )
    check("dit output shape", tuple(out.shape), (1, 24, 1, 4, 4))
    check("no audio head", model.final_layer.audio_out, None)
    check("pdd head bank", model.final_layer.pdd_steps, 6)


def test_image_vae_roundtrip():
    vae = MiniMaxH3ImageVAE(
        latent_channels=4,
        block_out_channels=(8,),
        layers_per_block=1,
        spatial_downsample_factors=(2,),
        temporal_downsample_factors=(1,),
        norm_num_groups=8,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        latents_mean=(0.0,) * 4,
        latents_std=(1.0,) * 4,
    )
    vae.eval()
    with torch.no_grad():
        latent = vae.encode_image(torch.randn(1, 3, 1, 8, 8))
        pixels = vae.decode(latent)
    check("encode shape", tuple(latent.shape), (1, 4, 1, 4, 4))
    check("decode shape", tuple(pixels.shape), (1, 3, 1, 8, 8))
    check("image decoder", vae.image_decoder, True)
    check("2d post-quant", type(vae.post_quant_conv).__name__, "Conv2d")


def test_pipeline_slot():
    check("cache slot", pipeline._SLOTS.get("h3_image"),
          ("_h3_image", "_h3_image_key", "_h3_image_error"))
    check("generator wired", callable(pipeline._generate_h3_image), True)
    check("loader wired", callable(pipeline._load_h3_image), True)


def test_dispatch_routes_to_image():
    seen = {}

    def fake(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
             regions_list=None, ref_images=None, preview=None):
        seen["family"] = cfg.get("family")
        seen["refs"] = ref_images
        return "sentinel"

    original = pipeline._generate_h3_image
    pipeline._generate_h3_image = fake
    try:
        marker = object()
        out = pipeline.generate("a caption", {"family": "h3_image"}, ref_images=[marker])
    finally:
        pipeline._generate_h3_image = original
    check("routed output", out, "sentinel")
    check("family in config", seen.get("family"), "h3_image")
    check("references forwarded", seen.get("refs"), [marker])


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