"""Checks for Qwen-Image 2.1 reference-image conditioning (multi-image edit).

Weights-free: the DiT's joint-sequence math (the rotary layout, the block-causal
attention segments, the substitution of condition latents into the vision slots)
is exercised with a tiny randomly-seeded model on the CPU, and the geometry
helpers are checked against hand-computed values. The vision tower (Qwen3-VL)
needs the real encoder weights, so it is left to the model test.

Run: .venv\\Scripts\\python.exe tests\\test_qwen_ref.py
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.qwen_image import (  # noqa: E402
    IMG_TOKENS_PER_SLOT,
    QwenImage21Transformer2DModel,
    QwenRef,
    _block_causal_segments,
    _build_token_metadata,
    _rope_ids,
)

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_true(name, cond):
    check(name, bool(cond), True)


def check_raises(name, fn):
    try:
        fn()
    except Exception:
        print(f"ok  {name}")
    else:
        FAILED.append(f"{name}: did not raise")


def _seed_params(model, seed=0):
    generator = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.dtype.is_floating_point:
            p.data.normal_(0.0, 0.1, generator=generator)
    return model.eval()


def _tiny_model():
    return _seed_params(QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 2, 2),
    ))


def _expanded_mask(vlm_img, target_tokens):
    """Repeat an over-context image mask (plus the target slots) to token length."""
    target_slots = target_tokens // IMG_TOKENS_PER_SLOT
    mask = torch.cat([vlm_img, vlm_img.new_ones(target_slots)])
    repeats = torch.where(mask, IMG_TOKENS_PER_SLOT, 1)
    return torch.repeat_interleave(mask, repeats)


# --------------------------------------------------------------- geometry
def test_rope_ids():
    # Vision slots first (one condition image, 2x2 latents), then text, then the
    # target image (4x4 latents). The expansion makes the mask already at token
    # length: 4 condition + 5 text + 16 target.
    is_img = torch.tensor(
        [True] * 4 + [False] * 5 + [True] * 16, dtype=torch.bool,
    )
    ids = _rope_ids(is_img, [(1, 2, 2), (1, 4, 4)], "cpu")
    frame = ids[:, 0].tolist()
    check("rope cond frame", frame[:4], [0.0, 0.0, 0.0, 0.0])
    check("rope text frame", frame[4:9], [2.0, 3.0, 4.0, 5.0, 6.0])
    check("rope target frame", frame[9:25], [7.0] * 16)
    check("rope cond height", ids[:4, 1].tolist(), [-1.0, -1.0, 0.0, 0.0])
    check("rope cond width", ids[:4, 2].tolist(), [-1.0, 0.0, -1.0, 0.0])
    check("rope text height==frame", ids[4:9, 1].tolist(), [2.0, 3.0, 4.0, 5.0, 6.0])
    check("rope target height", ids[9:13, 1].tolist(), [-2.0, -2.0, -2.0, -2.0])


def test_token_metadata():
    is_img = torch.tensor(
        [True] * 4 + [False] * 5 + [True] * 16, dtype=torch.bool,
    )
    image_ids, target_mask = _build_token_metadata(is_img, [(1, 2, 2), (1, 4, 4)])
    check("metadata cond id", image_ids[:4].tolist(), [0, 0, 0, 0])
    check("metadata text id", image_ids[4:9].tolist(), [-1, -1, -1, -1, -1])
    check("metadata target id", image_ids[9:25].tolist(), [1] * 16)
    check("metadata target mask text", int(target_mask[:9].sum()), 0)
    check("metadata target mask count", int(target_mask.sum()), 16)


def test_token_metadata_rejects_mismatch():
    is_img = torch.tensor([True] * 3 + [False] * 2, dtype=torch.bool)
    check_raises(
        "metadata mismatch raises",
        lambda: _build_token_metadata(is_img, [(1, 2, 2)]),
    )


def test_segments():
    image_ids, target_mask = _build_token_metadata(
        torch.tensor([True] * 4 + [False] * 5 + [True] * 16, dtype=torch.bool),
        [(1, 2, 2), (1, 4, 4)],
    )
    prefix_len = int((~target_mask).sum())
    segments = _block_causal_segments(image_ids, prefix_len, 25, "cpu")
    # condition image block, text run, target block.
    check("segment count", len(segments), 3)
    check("segment ranges", [(s, e) for s, e, _ in segments], [(0, 4), (4, 9), (9, 25)])
    check_true("condition block unmasked", segments[0][2] is None)
    text_mask = segments[1][2]
    check("text mask shape", tuple(text_mask.shape), (5, 9))
    check_true("text sees all earlier", bool(text_mask[:, :4].all()))
    check_true("text is causal", bool(text_mask[0, 4]) and not bool(text_mask[0, 5]))
    check_true("target unmasked", segments[2][2] is None)


# --------------------------------------------------------------- the model
def _ref_case():
    model = _tiny_model()
    # Target 4x4 latents -> 4 target slots. One condition image of 2x2 latents
    # (one slot) placed at context position 0; the rest is text.
    vlm_img = torch.zeros(6, dtype=torch.bool)
    vlm_img[0] = True
    cond = torch.randn(1, 4, 8)
    ref = QwenRef(cond, [(1, 2, 2)], vlm_img.unsqueeze(0))
    x = torch.randn(1, 8, 4, 4)
    ctx = torch.randn(1, 6, 32)
    return model, x, ctx, ref


def test_substitution_places_latents_in_slots():
    model, x, ctx, ref = _ref_case()
    seq, _pe, _ = model.build_sequence(x, ctx, ref=ref)

    txt = model.txt_in(ctx)
    imgs = torch.cat(
        [model.img_in(ref.latents), model.img_in(x.flatten(2).transpose(1, 2))],
        dim=1,
    )
    mask = _expanded_mask(ref.image_pad_mask[0], x.shape[-2] * x.shape[-1])
    check("joint length", seq.shape[1], int(mask.numel()))
    check_true(
        "condition+target latents land in the image slots",
        torch.allclose(seq[0][mask], imgs[0], atol=1e-5),
    )
    # Text positions keep the (projected) text embeddings; the vision-slot rows of
    # the context are the ones replaced by the condition latents.
    text_positions = ~mask
    kept_text = txt[0][~ref.image_pad_mask[0]]
    check_true(
        "text stays in the text positions",
        torch.allclose(seq[0][text_positions], kept_text, atol=1e-5),
    )


def test_forward_shapes_and_effect():
    model, x, ctx, ref = _ref_case()
    ts = torch.tensor([0.5])
    plain = model(x, ts, ctx)
    conditioned = model(x, ts, ctx, ref=ref)
    check("forward shape with ref", tuple(conditioned.shape), tuple(x.shape))
    check_true("forward is finite", bool(torch.isfinite(conditioned).all()))
    check_true(
        "reference conditioning changes the output",
        not torch.allclose(plain, conditioned, atol=1e-4),
    )


def test_multi_reference():
    model = _tiny_model()
    # Two condition images, both 2x2 latents (one vision slot each).
    vlm_img = torch.zeros(6, dtype=torch.bool)
    vlm_img[0] = True
    vlm_img[1] = True
    cond = torch.randn(1, 8, 8)  # two 2x2 images, 4 tokens each
    ref = QwenRef(cond, [(1, 2, 2), (1, 2, 2)], vlm_img.unsqueeze(0))
    x = torch.randn(1, 8, 4, 4)
    ctx = torch.randn(1, 6, 32)
    out = model(x, torch.tensor([0.5]), ctx, ref=ref)
    check("multi-reference forward shape", tuple(out.shape), tuple(x.shape))


def test_odd_target_size():
    # A target whose latent token count is not a multiple of four still packs.
    model = _tiny_model()
    vlm_img = torch.zeros(6, dtype=torch.bool)
    vlm_img[0] = True
    ref = QwenRef(torch.randn(1, 4, 8), [(1, 2, 2)], vlm_img.unsqueeze(0))
    x = torch.randn(1, 8, 3, 3)  # 9 target tokens
    out = model(x, torch.tensor([0.5]), torch.randn(1, 6, 32), ref=ref)
    check("odd target shape", tuple(out.shape), tuple(x.shape))


def test_bad_ref_raises():
    model, x, ctx, ref = _ref_case()
    # Claim a 4x4 condition image (16 tokens) but only supply 4.
    bad = QwenRef(torch.randn(1, 4, 8), [(1, 4, 4)], ref.image_pad_mask)
    check_raises(
        "mismatched reference latents raise",
        lambda: model.build_sequence(x, ctx, ref=bad),
    )


def test_regional_still_works():
    # A regional mask must still reach the target segment (unchanged t2i path).
    model = _tiny_model()
    x = torch.randn(1, 8, 4, 4)
    ctx = torch.randn(1, 4, 32)
    mask = torch.zeros(1, 1, 20, 20)
    seq, _pe, segments = model.build_sequence(x, ctx, cross_attn_mask=mask)
    check("regional joint length", seq.shape[1], 4 + 16)
    check_true(
        "regional mask applied to target",
        torch.equal(segments[-1][2], mask[0, 0, 4:, :20]),
    )


# --------------------------------------------------------------- encoding side
def test_vision_slot_arithmetic():
    """One vision slot is four latent tokens; the tokenizer expands to match."""
    from PIL import Image
    from transformers import Qwen2VLImageProcessor, Qwen2Tokenizer

    from engine.qwen_text import _IMAGE_TEMPLATE, _MERGE, _PATCH, IMAGE_PAD

    try:
        processor = Qwen2VLImageProcessor(
            patch_size=_PATCH, temporal_patch_size=2, merge_size=_MERGE,
        )
        tokenizer = Qwen2Tokenizer.from_pretrained(
            os.path.join(ROOT, "engine", "tokenizer_data")
        )
    except Exception as exc:  # transformers/image deps missing
        print(f"skip vision slot arithmetic: {exc}")
        return

    image = Image.new("RGB", (320, 256), (10, 20, 30))
    out = processor(images=[image], return_tensors="pt")
    t, h, w = out["image_grid_thw"].tolist()[0]
    slots = (t * h * w) // (_MERGE * _MERGE)
    latent_tokens = (320 // _PATCH) * (256 // _PATCH)

    check("slots per image", latent_tokens, slots * IMG_TOKENS_PER_SLOT)
    text = _IMAGE_TEMPLATE.format(i=1).replace("<|image_pad|>", "<|image_pad|>" * slots)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    check("tokenizer expands the slot", ids.count(IMAGE_PAD), slots)


def test_weight_name_mapping():
    from engine.qwen_text import _map_name, _map_vision_name

    check("text embed", _map_name("token_embd.weight"), "embed_tokens.weight")
    check("text layer", _map_name("blk.7.attn_q.weight"), "layers.7.self_attn.q_proj.weight")
    check("text qk norm", _map_name("blk.7.attn_k_norm.weight"), "layers.7.self_attn.k_norm.weight")
    check("vision block", _map_vision_name("v.blk.0.attn_qkv.weight"),
          "model.visual.blocks.0.attn.qkv.weight")
    check("vision patch", _map_vision_name("v.patch_embd.weight"),
          "model.visual.patch_embed.proj.weight")
    check("vision hf passthrough", _map_vision_name("model.visual.merger.linear_fc2.weight"),
          "model.visual.merger.linear_fc2.weight")
    check("unknown is ignored", _map_vision_name("v.mystery.weight"), None)


def test_safetensors_key_mapping():
    from engine.qwen_text import _safetensors_keys

    check(
        "text layer key",
        _safetensors_keys("model.language_model.layers.7.self_attn.q_proj.weight"),
        (
            "model.layers.7.self_attn.q_proj.weight",
            "layers.7.self_attn.q_proj.weight",
            "language_model.layers.7.self_attn.q_proj.weight",
        ),
    )
    check(
        "embed key",
        _safetensors_keys("model.language_model.embed_tokens.weight")[0],
        "model.embed_tokens.weight",
    )
    check(
        "visual key passthrough",
        _safetensors_keys("model.visual.blocks.0.attn.qkv.weight"),
        ("model.visual.blocks.0.attn.qkv.weight",),
    )
    check("unknown param", _safetensors_keys("something.else"), ())



def main():
    tests = [
        test_rope_ids,
        test_token_metadata,
        test_token_metadata_rejects_mismatch,
        test_segments,
        test_substitution_places_latents_in_slots,
        test_forward_shapes_and_effect,
        test_multi_reference,
        test_odd_target_size,
        test_bad_ref_raises,
        test_regional_still_works,
        test_vision_slot_arithmetic,
        test_weight_name_mapping,
        test_safetensors_key_mapping,
    ]
    for test in tests:
        try:
            test()
        except Exception as exc:  # keep going so one failure is not fatal
            import traceback

            FAILED.append(f"{test.__name__} raised: {exc}\n{traceback.format_exc()}")

    if FAILED:
        print("\nFAILURES:")
        for line in FAILED:
            print(" -", line)
        return 1
    print("\nall qwen reference tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
