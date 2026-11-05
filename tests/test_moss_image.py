"""Unit tests for the MOSS Image family (DiT, latent packing, detection).

Run: .venv\\Scripts\\python.exe tests\\test_moss_image.py

Only the pure pieces run without the 5B weights: the DiT's forward on a tiny
config, the FLUX.2 latent patch/BN pack-unpack round trip (against a stub VAE, so
``diffusers>=0.38`` is not required) and the component-kind / family detection.
"""

import os
import sys
from types import SimpleNamespace

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import component_kinds  # noqa: E402
from engine import moss_vae, paths  # noqa: E402
from engine.moss_image import MOSSImage, _detect_config  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _tiny(**kwargs):
    return MOSSImage(dim=256, ffn_dim=256, num_heads=2, num_layers=3, text_dim=16,
                     freq_dim=8, latent_channels=4, **kwargs)


def test_forward_shape():
    model = _tiny().eval()
    x, ctx = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 16)
    with torch.no_grad():
        out = model(x, torch.tensor([0.7]), ctx)
    check("dit forward shape", tuple(out.shape), (1, 4, 8, 8))


def test_rope_matches_reference_layout():
    from engine.moss_image import _RoPE, _rotate

    height, width, text_len, head_dim = 6, 6, 4, 128
    num_bands = head_dim // 8
    bands = 1.0 / (10000.0 ** (torch.arange(num_bands, dtype=torch.float32) / num_bands))

    h, w = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image = torch.stack((torch.zeros_like(h), torch.full_like(h, text_len), h, w), -1).flatten(0, 1).float()
    image[:, 2] -= (height - 1) / 2
    image[:, 3] -= (width - 1) / 2
    text = torch.zeros(text_len, 4)
    text[:, 1] = torch.arange(text_len)
    ids = torch.cat((image, text), 0)
    # Reference: sin/cos of the (r, f, h, w) phases concatenated as [sin, cos].
    phase = ids.unsqueeze(-1) * bands
    ref_rope = torch.cat((phase.sin(), phase.cos()), -2).flatten(-2)

    rope_full = _RoPE(head_dim)(height, width, text_len, torch.device("cpu"))
    check("rope layout matches reference", bool(torch.allclose(rope_full[0], ref_rope, atol=1e-6)), True)

    x = torch.randn(1, ids.shape[0], 2, head_dim)
    sin, cos = ref_rope.chunk(2, -1)
    sin = torch.cat((sin, sin), -1)[None, :, None]
    cos = torch.cat((cos, cos), -1)[None, :, None]
    a, b = x.chunk(2, -1)
    ref = x * cos + torch.cat((-b, a), -1) * sin
    check("rotate applies half-split rotation", bool(torch.allclose(_rotate(x, rope_full), ref, atol=1e-6)), True)


def test_moss_context_skip_routing():
    from engine.sampler import _moss_context

    embeds = torch.randn(1, 5, 16)
    _, mask, skip = _moss_context(embeds)
    check("bare positive skips no block", skip, ())
    check("bare positive has no mask", mask, None)
    _, _, skip = _moss_context((embeds, None))
    check("pair positive skips no block", skip, ())
    _, _, skip = _moss_context((embeds, None, (22,)))
    check("negative triple carries its skip", skip, (22,))


def test_timestep_embedding_matches_reference():
    from engine import moss_image
    from engine.flux import timestep_embedding as ref_timestep_embedding

    t = torch.tensor([0.0, 0.25, 0.5, 1.0])
    got = moss_image._timestep_embedding(t, 256)
    want = ref_timestep_embedding(t, 256)
    check("timestep embedding matches reference (time_factor=1000)",
          bool(torch.allclose(got, want, atol=5e-4)), True)


def test_skip_layers_change_output():
    model = _tiny().eval()
    x, ctx = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 16)
    with torch.no_grad():
        full = model(x, torch.tensor([0.7]), ctx)
        skipped = model(x, torch.tensor([0.7]), ctx, skip_layers=(1,))
    check("skip_layers changes output", bool(torch.allclose(full, skipped)), False)


def test_region_mask_accepted():
    model = _tiny().eval()
    x, ctx = torch.randn(1, 4, 8, 8), torch.randn(1, 5, 16)
    mask = torch.zeros(1, 64 + 5, 64 + 5)
    with torch.no_grad():
        out = model(x, torch.tensor([0.7]), ctx, cross_attn_mask=mask)
    check("region mask forward shape", tuple(out.shape), (1, 4, 8, 8))


def test_fused_qkv_variant():
    model = _tiny(fused_qkv=True)
    keys = set(model.state_dict())
    check("fused qkv key present", "layers.0.proj.qkv.weight" in keys, True)
    check("split q key absent", "layers.0.proj.q.weight" in keys, False)


def test_detect_config():
    model = _tiny()
    cfg = _detect_config(dict(model.state_dict()))
    check("detect dim", cfg["dim"], 256)
    check("detect ffn_dim", cfg["ffn_dim"], 256)
    check("detect num_heads", cfg["num_heads"], 2)
    check("detect num_layers", cfg["num_layers"], 3)
    check("detect text_dim", cfg["text_dim"], 16)
    check("detect latent_channels", cfg["latent_channels"], 4)
    check("detect fused_qkv", cfg["fused_qkv"], False)


def test_latent_pack_round_trip():
    bn = torch.nn.BatchNorm2d(128)
    with torch.no_grad():
        bn.running_mean.normal_(0, 0.5)
        bn.running_var.uniform_(0.5, 2.0)
    vae = SimpleNamespace(bn=bn, config=SimpleNamespace(batch_norm_eps=1e-4))
    z32 = torch.randn(1, 32, 16, 16)
    packed = moss_vae.pack_latents(vae, z32)
    check("pack channels", packed.shape[1], 128)
    check("pack halves spatial", packed.shape[-1], 8)
    back = moss_vae.unpack_latents(vae, packed)
    check("unpack restores shape", tuple(back.shape), (1, 32, 16, 16))
    check("pack/unpack inverse", bool(torch.allclose(back, z32, atol=1e-5)), True)


def test_component_detection():
    sd = {
        "model.diffusion_model.video_embedder.weight": (3072, 128),
        "model.diffusion_model.shared_block_modulation.weight": (6 * 3072, 3072),
    }
    check("model kind is moss", component_kinds._model_from_shapes(sd, None), "moss_image")
    shapes = {"model.embed_tokens.weight": (248320, 4096),
              "model.layers.0.linear_attn.in_proj.weight": (10, 10)}
    check("text kind is qwen3.5", component_kinds._qwen_kind_from_shapes(shapes, False), "qwen3_5_9b")
    check("text token qwen3.5",
          component_kinds._text_encoder_token("qwen3.5_9b_int8_convrot.safetensors"), "qwen3_5_9b")
    check("vae kind is flux2",
          component_kinds._vae_kind("x.safetensors", {"bn.running_mean": (128,)}), "flux2_vae")
    check("vae token flux2", component_kinds._vae_token("flux2_vae.safetensors"), "flux2_vae")


def test_classify_and_slots():
    check("classify moss name",
          paths.classify_model("moss_image_sft_bf16.safetensors"), ("MOSS Image", "moss_image"))
    check("moss text slot", component_kinds.SLOT_KIND[("moss_image", "text")], "qwen3_5_9b")
    check("moss vae slot", component_kinds.SLOT_KIND[("moss_image", "vae")], "flux2_vae")


def test_convrot_hadamard():
    from engine import moss_text

    h = moss_text._hadamard(256)
    eye = torch.eye(256)
    check("hadamard symmetric", bool(torch.allclose(h, h.T, atol=1e-6)), True)
    check("hadamard orthogonal", bool(torch.allclose(h @ h.T, eye, atol=1e-5)), True)
    check("hadamard involutory", bool(torch.allclose(h @ h, eye, atol=1e-5)), True)


def test_convrot_fold_identity():
    import torch.nn.functional as F

    from engine import moss_text

    h = torch.block_diag(moss_text._hadamard(256), moss_text._hadamard(256))
    x = torch.randn(3, 512)
    w_rot = torch.randn(8, 512)
    # ComfyUI computes (x H) @ W_rot^T; folding H into the weight must match.
    check("convrot fold identity",
          bool(torch.allclose((x @ h) @ w_rot.T, F.linear(x, moss_text._unrotate(w_rot, 256)), atol=1e-4)),
          True)


def test_dit_int8_convrot_dequant_metadata():
    import json
    import tempfile

    from safetensors.torch import save_file

    from engine import moss_text
    from engine.moss_image import _checkpoint_source

    torch.manual_seed(0)
    weight = torch.randint(-127, 128, (4, 256), dtype=torch.int8)
    scale = torch.rand(4, 1) * 0.01 + 0.001
    layers = {"layers.0.proj.q": {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}}
    metadata = {"_quantization_metadata": json.dumps({"format_version": "1.0", "layers": layers})}
    state = {
        "layers.0.proj.q.weight": weight,
        "layers.0.proj.q.weight_scale": scale,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dit_meta.safetensors")
        save_file(state, path, metadata=metadata)
        shapes, fetch, close = _checkpoint_source(path, torch.float32)
        try:
            got = fetch("layers.0.proj.q.weight")
            want = moss_text._unrotate(weight.to(torch.float32) * scale, 256)
            check("dit metadata convrot un-rotates", bool(torch.allclose(got, want, atol=1e-5)), True)
        finally:
            close()


def test_dit_int8_convrot_dequant():
    import json
    import tempfile

    from safetensors.torch import save_file

    from engine import moss_text
    from engine.moss_image import _checkpoint_source

    torch.manual_seed(0)
    weight = torch.randint(-127, 128, (4, 256), dtype=torch.int8)
    scale = torch.rand(4, 1) * 0.01 + 0.001
    conf = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}).encode()
    state = {
        "model.diffusion_model.layers.0.proj.q.weight": weight,
        "model.diffusion_model.layers.0.proj.q.weight_scale": scale,
        "model.diffusion_model.layers.0.proj.q.comfy_quant": torch.tensor(list(conf), dtype=torch.uint8),
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "dit.safetensors")
        save_file(state, path)
        shapes, fetch, close = _checkpoint_source(path, torch.float32)
        try:
            got = fetch("layers.0.proj.q.weight")
            want = moss_text._unrotate(weight.to(torch.float32) * scale, 256)
            check("dit dequant strips prefix and un-rotates", bool(torch.allclose(got, want, atol=1e-5)), True)
            check("dit dequant dtype is float", got.dtype, torch.float32)
        finally:
            close()


def test_vae_config_from_state():
    state = {
        "encoder.conv_out.weight": torch.zeros(64, 512, 3, 3),
        "encoder.down.0.block.0.conv1.weight": torch.zeros(128, 128, 3, 3),
        "encoder.down.1.block.0.conv1.weight": torch.zeros(256, 128, 3, 3),
        "encoder.down.2.block.0.conv1.weight": torch.zeros(512, 256, 3, 3),
        "encoder.down.3.block.0.conv1.weight": torch.zeros(512, 512, 3, 3),
        "decoder.up.0.block.0.conv1.weight": torch.zeros(96, 192, 3, 3),
        "decoder.up.1.block.0.conv1.weight": torch.zeros(192, 384, 3, 3),
        "decoder.up.2.block.0.conv1.weight": torch.zeros(384, 384, 3, 3),
        "decoder.up.3.block.0.conv1.weight": torch.zeros(384, 384, 3, 3),
        "encoder.down.0.block.1.conv1.weight": torch.zeros(128, 128, 3, 3),
    }
    cfg = moss_vae._vae_config_from_state(state)
    check("derived encoder channels", cfg["block_out_channels"], (128, 256, 512, 512))
    check("derived decoder channels", cfg["decoder_block_out_channels"], (96, 192, 384, 384))
    check("derived layers_per_block", cfg["layers_per_block"], 2)
    check("derived latent channels", cfg["latent_channels"], 32)


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
