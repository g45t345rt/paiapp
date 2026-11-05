"""Checks for cross-family regional prompting.

Two halves:

* :func:`engine.regions.joint_attn_mask` -- the joint-attention mask used by the
  single-stream families (Flux, Z-Image, Qwen-Image, Krea 2). Pure, always
  runs.
* the mask plumbing in each of those transformers, built at a tiny size with
  random weights on the CPU: an all-allow mask must match unmasked attention, and
  a region-restricting mask must change the image output (proving the mask is
  threaded to the attention).

Run: .venv\\Scripts\\python.exe tests\\test_region_joint.py
"""

import math
import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import regions  # noqa: E402

FAILED = []
SKIPPED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_close(name, a, b, tol=2e-3):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape or not np.allclose(a, b, atol=tol, rtol=1e-3):
        FAILED.append(f"{name}: arrays differ (max {np.abs(a - b).max():.3e})")
    else:
        print(f"ok  {name}")


def check_differs(name, a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if a.shape != b.shape or np.allclose(a, b, atol=1e-4, rtol=1e-4):
        FAILED.append(f"{name}: masked and unmasked output are identical")
    else:
        print(f"ok  {name}")


def rectangle(size, box):
    mask = Image.new("L", size, 0)
    mask.paste(255, box)
    return mask


# A regional mask is additive: 0.0 allows a key, -inf blocks it.
def _allowed(part):
    return part == 0.0


def _blocked(part):
    return part == float("-inf")


# ---------------------------------------------------------------------------
# The joint-attention mask helper.
# ---------------------------------------------------------------------------

def test_region_biases():
    entries = [
        {},                        # default strength 1.0 -> no-op
        {"strength": 0.0},         # blocks the region's block
        {"strength": 2.0},         # amplifies it
        {"strength": None},        # malformed -> no-op
    ]
    biases = regions.region_biases(entries)
    check("default bias", biases[0], 0.0)
    check("zero blocks", biases[1], float("-inf"))
    check_close("double amplifies", biases[2], math.log(2.0))
    check("malformed is a no-op", biases[3], 0.0)


def test_joint_mask_layout():
    left = {"mask": rectangle((4, 4), (0, 0, 2, 4))}   # columns 0..2
    right = {"mask": rectangle((4, 4), (2, 0, 4, 4))}  # columns 2..4
    table = regions.token_regions([left, right], 4, 4)  # 16 tokens, row-major
    lengths = [3, 2, 2]  # base + two regions -> text total 7
    mask = regions.joint_attn_mask(table, lengths, 16, torch.device("cpu"))
    m = mask[0, 0]
    t = sum(lengths)

    check("mask shape", tuple(mask.shape), (1, 1, t + 16, t + 16))
    check("mask dtype float", mask.dtype, torch.float32)
    check("text rows see everything", bool(_allowed(m[:t]).all()), True)
    check("image rows see the base", bool(_allowed(m[t:, :3]).all()), True)
    check("image rows see all image", bool(_allowed(m[t:, t:]).all()), True)

    # token 0 (row 0, col 0) is in the left region only.
    check("left token sees region 0", bool(_allowed(m[t + 0, 3:5]).all()), True)
    check("left token blocked from region 1", bool(_blocked(m[t + 0, 5:7]).all()), True)
    # token 2 (row 0, col 2) is in the right region only.
    check("right token sees region 1", bool(_allowed(m[t + 2, 5:7]).all()), True)
    check("right token blocked from region 0", bool(_blocked(m[t + 2, 3:5]).all()), True)


def test_joint_mask_strength():
    region = {"mask": rectangle((4, 4), (0, 0, 2, 4)), "strength": 2.0}
    table = regions.token_regions([region], 4, 4)
    mask = regions.joint_attn_mask(
        table, [3, 2], 16, torch.device("cpu"),
        biases=regions.region_biases([region]),
    )
    m = mask[0, 0]
    t = 5  # base 3 + region 2
    check_close("strength scales the region block", m[t + 0, 3:5].numpy(), [math.log(2.0)] * 2)
    check("base block stays neutral", bool(_allowed(m[t + 0, :3]).all()), True)

    zero = {"mask": rectangle((4, 4), (0, 0, 2, 4)), "strength": 0.0}
    mask = regions.joint_attn_mask(
        table, [3, 2], 16, torch.device("cpu"),
        biases=regions.region_biases([zero]),
    )
    m = mask[0, 0]
    check("strength 0 blocks the region block", bool(_blocked(m[t + 0, 3:5]).all()), True)
    check("strength 0 keeps the base", bool(_allowed(m[t + 0, :3]).all()), True)


def test_cross_attn_mask_strength():
    region = {"mask": rectangle((4, 4), (0, 0, 2, 4)), "strength": 0.0}
    table = regions.token_regions([region], 4, 4)
    mask = regions.cross_attn_mask(
        table, [3, 2], torch.device("cpu"), biases=regions.region_biases([region]),
    )
    m = mask[0, 0]
    check("cross mask shape", tuple(mask.shape), (1, 1, 16, 5))
    check("cross base neutral", bool(_allowed(m[:, :3]).all()), True)
    check("cross strength 0 blocks covered tokens", bool(_blocked(m[0, 3:5]).all()), True)
    check("cross blocks uncovered tokens", bool(_blocked(m[15, 3:5]).all()), True)


def test_joint_mask_handles_text_padding_and_image_padding():
    region = {"mask": rectangle((4, 4), (0, 0, 2, 4))}
    table = regions.token_regions([region], 4, 4)
    # text_keep zeroes the third base token; image_len 20 pads 4 image tokens.
    mask = regions.joint_attn_mask(
        table, [3, 2], 20, torch.device("cpu"),
        text_keep=[True, True, False],
    )
    m = mask[0, 0]
    t = 5  # base 3 + region 2
    check("shape with padded rows", tuple(mask.shape), (1, 1, t + 20, t + 20))
    check("padded text column blocked", bool(_blocked(m[:, 2]).all()), True)
    # row t+16 is a padded image token: base (its real tokens) + image only.
    check("padded image token sees base", bool(_allowed(m[t + 16, :2]).all()), True)
    check("padded image token has no region", bool(_blocked(m[t + 16, 3:5]).all()), True)
    check("padded image token sees image", bool(_allowed(m[t + 16, 5:]).all()), True)


def test_pad_length():
    check("pad_length rounds up", regions.pad_length(5, 8), 8)
    check("pad_length exact", regions.pad_length(16, 8), 16)
    check("pad_length none", regions.pad_length(7, None), 7)


# ---------------------------------------------------------------------------
# Per-family mask plumbing (tiny random models on the CPU).
# ---------------------------------------------------------------------------

def _seed_params(model):
    torch.manual_seed(0)
    with torch.no_grad():
        for param in model.parameters():
            if param.dtype.is_floating_point:
                torch.nn.init.normal_(param, mean=0.0, std=0.1)
    return model.eval()


def _region_mask(block_lengths, image_len, text_len=None):
    """A joint mask over one left-half region (grid 4x4), matching a ctx layout."""
    region = {"mask": rectangle((4, 4), (0, 0, 2, 4))}
    table = regions.token_regions([region], 4, 4)
    return regions.joint_attn_mask(
        table, block_lengths, image_len, torch.device("cpu"), text_len=text_len,
    )


def _all_mask(length):
    return torch.ones(1, 1, length, length, dtype=torch.bool, device=torch.device("cpu"))


class _Case:
    """A tiny model plus the mask shapes its conditioning implies."""

    def __init__(self, name, model, forward, text_total, image_len,
                 block_lengths, padded_text=None):
        self.name = name
        self.model = model
        self.forward = forward
        self.text_total = text_total
        self.image_len = image_len
        self.block_lengths = block_lengths
        self.padded_text = padded_text

    @property
    def length(self):
        return self.text_total + self.image_len

    def run(self, mask):
        return self.forward(self.model, mask)


def _flux_case():
    from engine.flux import Flux

    model = _seed_params(Flux(
        in_channels=16, out_channels=16, vec_in_dim=None, context_in_dim=32,
        hidden_size=32, mlp_ratio=2.0, num_heads=2, depth=2, depth_single_blocks=2,
        axes_dim=(2, 8, 6), theta=10000, patch_size=2, guidance_embed=False,
    ))
    x = torch.randn(1, 16, 8, 8)
    ctx = torch.randn(1, 4, 32)  # base(2) + region(2)
    ts = torch.tensor([500.0])
    forward = lambda m, mask: m(x, ts, ctx, y=None, guidance=None, cross_attn_mask=mask)
    return _Case("flux", model, forward, 4, 16, [2, 2])


def _z_case():
    from engine.z_image import NextDiT

    model = _seed_params(NextDiT(
        patch_size=2, in_channels=16, dim=64, n_layers=1, n_refiner_layers=1,
        n_heads=2, n_kv_heads=2, multiple_of=16, ffn_dim_multiplier=2.0,
        norm_eps=1e-5, qk_norm=True, cap_feat_dim=32, axes_dims=(8, 12, 12),
        axes_lens=(64, 64, 64), rope_theta=256.0, z_image_modulation=False,
        time_scale=1.0, pad_tokens_multiple=8,
    ))
    x = torch.randn(1, 16, 8, 8)
    ctx = torch.randn(1, 5, 32)  # base(3) + region(2), padded to a multiple of 8
    ts = torch.tensor([0.5])
    forward = lambda m, mask: m(x, ts, ctx, cross_attn_mask=mask)
    return _Case("z_image", model, forward, 8, 16, [3, 2], padded_text=8)


def _qwen_case():
    from engine.qwen_image import QwenImage21Transformer2DModel

    model = _seed_params(QwenImage21Transformer2DModel(
        in_channels=8, out_channels=8, num_layers=1, attention_head_dim=8,
        num_attention_heads=2, context_in_dim=32, mlp_ratio=2,
        axes_dims_rope=(4, 2, 2),
    ))
    x = torch.randn(1, 8, 4, 4)
    ctx = torch.randn(1, 4, 32)
    ts = torch.tensor([0.5])
    forward = lambda m, mask: m(x, ts, ctx, cross_attn_mask=mask)
    return _Case("qwen_image", model, forward, 4, 16, [2, 2])


def _krea2_case():
    from engine.krea2 import Krea2Transformer2DModel

    model = _seed_params(Krea2Transformer2DModel(
        features=32, tdim=8, txtdim=8, heads=2, kvheads=1, multiplier=2,
        layers=1, patch=2, channels=4, theta=1000.0, txtlayers=2, txtheads=2,
        txtkvheads=2,
    ))
    x = torch.randn(1, 4, 8, 8)
    ctx = torch.randn(1, 4, 16)  # txtlayers * txtdim
    ts = torch.tensor([0.5])
    forward = lambda m, mask: m(x, ts, ctx, attention_mask=None, cross_attn_mask=mask)
    return _Case("krea2", model, forward, 4, 16, [2, 2])


def test_family_plumbing():
    for case in (_flux_case, _z_case, _qwen_case, _krea2_case):
        try:
            item = case()
        except Exception as exc:  # keep the other families running
            SKIPPED.append(f"{case.__name__} skipped: {exc}")
            continue
        try:
            torch.manual_seed(1)
            with torch.no_grad():
                out_none = item.run(None).float().numpy()
                out_all = item.run(_all_mask(item.length)).float().numpy()
                region_mask = _region_mask(
                    item.block_lengths, item.image_len, item.padded_text
                )
                out_reg = item.run(region_mask).float().numpy()
            check_close(f"{item.name}: all-allow == unmasked", out_none, out_all)
            check_differs(f"{item.name}: region mask changes output", out_none, out_reg)
        except Exception as exc:
            FAILED.append(f"{item.name}: {exc}")


def test_apply_models_accept_regional_context():
    """Every apply-model must accept a regional ``(..., mask)`` context.

    The regional wrap adds one element to the family's conditioning, so an
    arity-1 family (Qwen-Image, Z-Image) carries ``(context, mask)`` and an
    arity-2 family (Flux, Krea 2) carries ``(a, b, mask)``. The apply-model
    must hand its DiT a plain tensor either way.
    """
    from engine import sampler

    class _Spy:
        def __init__(self):
            self.contexts = []

        def __call__(self, x, t, ctx, **kwargs):
            self.contexts.append((ctx, kwargs))
            return torch.zeros_like(x)

    x = torch.zeros(1, 8, 4, 4)
    sigma = torch.tensor([0.5])
    ctx = torch.zeros(1, 4, 8)
    pooled = torch.zeros(1, 8)
    keep = torch.ones(1, 4, dtype=torch.bool)
    mask = _all_mask(4)

    cases = [
        ("qwen_image", lambda d: sampler.QwenImageApplyModel(d), (ctx, mask), ctx),
        ("z_image", lambda d: sampler.ZImageApplyModel(d), (ctx, mask), ctx),
        ("flux", lambda d: sampler.FluxApplyModel(d), (ctx, pooled, mask), (ctx, pooled)),
        ("krea2", lambda d: sampler.Krea2ApplyModel(d), (ctx, keep, mask), (ctx, keep)),
    ]
    for name, build, regional, plain in cases:
        spy = _Spy()
        model = build(spy)
        try:
            model.apply_model(x, sigma, regional)
            model.apply_model(x, sigma, plain)
        except Exception as exc:
            FAILED.append(f"{name}: apply-model rejected a context: {exc!r}")
            continue
        got, _ = spy.contexts[0]
        if not torch.is_tensor(got):
            FAILED.append(f"{name}: regional context reached the DiT as {type(got).__name__}")
            continue
        print(f"ok  {name}: apply-model accepts regional + plain context")


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    for note in SKIPPED:
        print(f"skip {note}")
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
