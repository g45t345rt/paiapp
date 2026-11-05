"""Checks for the Anima reference adapter (offset-RoPE reference frame + LoRA).

Weights-free: the reference forward runs a tiny randomly-seeded ``MiniTrainDIT``
on the CPU to check the two-frame geometry (frame 0 out, ref resized to the
target grid), the gated adapter LoRA is exercised with a hand-built state dict,
and the guider math (separate reference guidance plus APG/rescale) is checked
against hand-computed values with a stub model.

Run: .venv\\Scripts\\python.exe tests\\test_anima_ref.py
"""

import os
import sys
import tempfile
import json

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.anima_ref import (  # noqa: E402
    AnimaRefConditioning,
    RefAdapter,
    activate,
    restore,
)
from engine.cosmos import MiniTrainDIT, _frame_positions, _rope_from_positions  # noqa: E402
from engine.sampler import anima_ref_guider  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_true(name, cond):
    check(name, bool(cond), True)


def check_close(name, got, want, tol=1e-5):
    if abs(float(got) - float(want)) > tol:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _tiny_dit(seed=0):
    torch.manual_seed(seed)
    dit = MiniTrainDIT(
        in_channels=16, out_channels=16, model_channels=64, num_blocks=2,
        num_heads=4, crossattn_emb_channels=32, adaln_lora_dim=32,
        max_img_h=64, max_img_w=64, rope_h_extrapolation_ratio=1.0,
        rope_w_extrapolation_ratio=1.0,
    )
    return dit.eval()


# --------------------------------------------------------------------- forward
def test_forward_with_ref_shape():
    dit = _tiny_dit()
    x = torch.randn(1, 16, 1, 4, 4)
    ctx = torch.randn(1, 8, 32)
    t = torch.tensor([500.0])
    ref = torch.randn(1, 16, 1, 4, 4)
    with torch.no_grad():
        plain = dit(x, t, ctx)
        out = dit.forward_with_ref(x, t, ctx, ref, offset_ratio=1.0)
    check("plain shape", tuple(plain.shape), (1, 16, 1, 4, 4))
    check("ref shape", tuple(out.shape), (1, 16, 1, 4, 4))
    check_true("ref differs from plain", not torch.allclose(plain, out))


def test_forward_with_ref_resizes_ref():
    dit = _tiny_dit()
    x = torch.randn(1, 16, 1, 4, 4)
    ctx = torch.randn(1, 8, 32)
    t = torch.tensor([500.0])
    # A 4-D, half-size reference latent takes the resize path.
    ref = torch.randn(1, 16, 2, 2)
    with torch.no_grad():
        out = dit.forward_with_ref(x, t, ctx, ref, offset_ratio=1.0)
    check("resized ref shape", tuple(out.shape), (1, 16, 1, 4, 4))


def test_rope_offset_shifts_rows():
    dit = _tiny_dit()
    pe = dit.pos_embedder
    t, h, w = _frame_positions(1, 4, 4, 0, 0, torch.device("cpu"))
    base = _rope_from_positions(pe, t, h, w)
    t2, h2, w2 = _frame_positions(1, 4, 4, 4, 0, torch.device("cpu"))
    shifted = _rope_from_positions(pe, t2, h2, w2)
    check("rope shape", tuple(base.shape), tuple(shifted.shape))
    check_true("offset shifts rope", not torch.allclose(base, shifted))


# ------------------------------------------------------------------- adapter
def _save_adapter(path, metadata=None):
    from safetensors.torch import save_file

    down = torch.randn(4, 64)  # [rank, in]
    up = torch.randn(64, 4)    # [out, rank]
    state = {
        "diffusion_model.blocks.0.self_attn.q_proj.lora_down.weight": down,
        "diffusion_model.blocks.0.self_attn.q_proj.lora_up.weight": up,
        "diffusion_model.blocks.0.self_attn.q_proj.alpha": torch.tensor(4.0),
    }
    save_file(state, path, metadata=metadata or {})
    return path


def test_ref_adapter_gating():
    dit = _tiny_dit()
    path = _save_adapter(os.path.join(tempfile.mkdtemp(), "ref.safetensors"))

    adapter = RefAdapter(path, 1.0)
    check("pilot adapter is not asym", adapter.asym, False)
    check("adapter layers", adapter.install(dit), 1)

    x = torch.randn(1, 16, 1, 4, 4)
    ctx = torch.randn(1, 8, 32)
    t = torch.tensor([500.0])
    with torch.no_grad():
        off = dit(x, t, ctx)
        previous = activate(adapter)
        on = dit(x, t, ctx)
        restore(previous)
        off_again = dit(x, t, ctx)
    check_true("lora changes output when active", not torch.allclose(off, on))
    check_true("lora off again matches", torch.allclose(off, off_again))


def test_asym_metadata():
    meta = {"anima_ref_adapter": json.dumps({
        "arch": "asym",
        "slots": {"character": {"t": 1, "offset_h": 1.0, "offset_w": 0}},
    })}
    path = _save_adapter(os.path.join(tempfile.mkdtemp(), "v2.safetensors"), meta)
    adapter = RefAdapter(path, 1.0)
    check("asym detected", adapter.asym, True)
    check("slots read", adapter.slots["character"]["t"], 1)
    check("ref_lora default", adapter.ref_lora, True)


# ---------------------------------------------------------------- asym forward
def test_asym_forward_shape():
    dit = _tiny_dit()
    x = torch.randn(1, 16, 1, 4, 4)
    ctx = torch.randn(1, 8, 32)
    t = torch.tensor([500.0])
    # A reference at a *different* size than the target: the v2 path reads any size.
    ref = torch.randn(1, 16, 1, 2, 2)
    with torch.no_grad():
        ref_kv = dit.encode_ref_asym(ref, (2, 0))
        out = dit.forward_asym(x, t, ctx, ref_kv)
        plain = dit(x, t, ctx)
    check("asym out shape", tuple(out.shape), (1, 16, 1, 4, 4))
    check("asym kv per block", len(ref_kv), len(dit.blocks))
    check_true("asym differs from plain", not torch.allclose(plain, out))


def test_asym_ref_kv_cache():
    meta = {"anima_ref_adapter": json.dumps({"arch": "asym"})}
    path = _save_adapter(os.path.join(tempfile.mkdtemp(), "v2.safetensors"), meta)
    adapter = RefAdapter(path, 1.0)
    dit = _tiny_dit()
    x = torch.randn(1, 16, 1, 4, 4)
    ref = torch.randn(1, 16, 1, 2, 2)
    with torch.no_grad():
        first = adapter.ref_kv(dit, ref, x, 1.0)
        again = adapter.ref_kv(dit, ref, x, 1.0)
        other = adapter.ref_kv(dit, ref, x, 2.0)
    check_true("same key reuses cache", first is again)
    check_true("new offset replaces cache", other is not first)
    check("one cache entry", len(adapter.ref_cache), 1)


def test_apply_model_asym():
    from engine.sampler import AnimaRefApplyModel

    meta = {"anima_ref_adapter": json.dumps({"arch": "asym"})}
    path = _save_adapter(os.path.join(tempfile.mkdtemp(), "v2.safetensors"), meta)
    adapter = RefAdapter(path, 1.0)
    dit = _tiny_dit()
    adapter.install(dit)
    model = AnimaRefApplyModel(dit, torch.float32, adapter=adapter, offset=1.0)
    x = torch.randn(1, 16, 1, 4, 4)
    ctx = torch.randn(1, 8, 32)
    ref = torch.randn(1, 16, 1, 2, 2)
    sigma = torch.tensor([0.5])
    with torch.no_grad():
        out = model.apply_model(x, sigma, AnimaRefConditioning(ctx, ref))
    check("apply_model asym shape", tuple(out.shape), (1, 16, 1, 4, 4))


def test_forward_with_refs_multi_slot():
    dit = _tiny_dit()
    x = torch.randn(1, 16, 1, 4, 4)
    ctx = torch.randn(1, 8, 32)
    t = torch.tensor([500.0])
    char = torch.randn(1, 16, 1, 4, 4)
    style = torch.randn(1, 16, 1, 4, 4)
    with torch.no_grad():
        one = dit.forward_with_refs(x, t, ctx, [(char, 1, 2, 0)])
        two = dit.forward_with_refs(
            x, t, ctx, [(char, 1, 2, 0), (style, 2, 1, 1)]
        )
    check("multi-slot shape", tuple(two.shape), (1, 16, 1, 4, 4))
    check_true("a second slot changes the output", not torch.allclose(one, two))


def test_conditioning_slots():
    char = torch.zeros(1)
    style = torch.ones(1)
    ctx = AnimaRefConditioning(torch.zeros(1, 2), {"character": char, "style": style})
    check_true("character is default ref", ctx.ref is char)
    check_true("style slot kept", ctx.refs["style"] is style)
    check("without_ref empties", ctx.without_ref().refs, {})
    single = AnimaRefConditioning(torch.zeros(1, 2), char)
    check_true("bare latent is the character", single.refs["character"] is char)


# -------------------------------------------------------------------- guider
class _StubModel:
    """Returns ``ctx.text`` with a reference frame, else a fixed ``base`` value.

    This mirrors the real guider: the ``negative_base`` branch (no reference
    frame) predicts a different denoising target than the reference-attached
    branches.
    """

    def __init__(self, base=None):
        self.base = base

    def apply_model(self, x, sigma, ctx):
        if ctx.ref is None:
            return self.base
        return ctx.text


def test_guider_math():
    pos = torch.tensor([[1.0, 2.0, 3.0]])
    neg = torch.tensor([[0.5, 0.5, 0.5]])
    base = torch.tensor([[0.1, 0.1, 0.1]])
    model = _StubModel(base)
    ref = torch.zeros(1)
    ctx_pos = AnimaRefConditioning(pos, ref)
    ctx_neg = AnimaRefConditioning(neg, ref)
    ctx_base = ctx_neg.without_ref()
    check_true("without_ref drops ref", ctx_base.ref is None)

    cfg, rg = 4.5, 4.5
    with torch.no_grad():
        guided = anima_ref_guider(
            model, None, None, ctx_pos, ctx_neg, cfg, ref_guidance=rg,
            correction="none",
        )
    want = base + rg * (neg - base) + cfg * (pos - neg)
    check_true("guider formula", torch.allclose(guided, want))


def test_guider_apg_orthogonal():
    pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    neg = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    base = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    model = _StubModel(base)
    ctx_pos = AnimaRefConditioning(pos, torch.zeros(1))
    ctx_neg = AnimaRefConditioning(neg, torch.zeros(1))
    with torch.no_grad():
        guided = anima_ref_guider(
            model, None, None, ctx_pos, ctx_neg, 4.5, ref_guidance=4.5,
            correction="apg",
        )
    update = (guided - pos)
    unit = torch.nn.functional.normalize(pos.double().flatten(1), dim=1)
    parallel = (update.double().flatten(1) * unit).sum(dim=1)
    check_close("apg removes parallel component", parallel.item(), 0.0, 1e-5)


def test_guider_rescale_matches_std():
    pos = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    neg = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
    base = torch.tensor([[0.1, 0.1, 0.1, 0.1]])
    model = _StubModel(base)
    ctx_pos = AnimaRefConditioning(pos, torch.zeros(1))
    ctx_neg = AnimaRefConditioning(neg, torch.zeros(1))
    phi = 0.5
    with torch.no_grad():
        guided = anima_ref_guider(
            model, None, None, ctx_pos, ctx_neg, 4.5, ref_guidance=4.5,
            correction="rescale", rescale_phi=phi,
        )
    raw = base + 4.5 * (neg - base) + 4.5 * (pos - neg)
    ratio = pos.std() / raw.std()
    want = phi * raw * ratio + (1 - phi) * raw
    check_true("rescale formula", torch.allclose(guided, want, atol=1e-6))


def test_component_wiring():
    from engine import component_kinds as kinds
    from engine import paths

    check("adapter kind label", kinds.kind_label("anima_ref"), "Anima Ref Adapter")
    check("adapter category", kinds.category_of("anima_ref"), kinds.ADAPTER)
    check("adapters pool registered", "adapters" in paths.component_pools(), True)
    check("adapter pool category",
          kinds.POOL_CATEGORY.get("adapters"), kinds.ADAPTER)
    check("adapter detected by name",
          kinds.detect("C:/x/anima_ref_adapter_pilot_r32.safetensors", "adapter"),
          "anima_ref")


def main():
    test_forward_with_ref_shape()
    test_forward_with_ref_resizes_ref()
    test_rope_offset_shifts_rows()
    test_ref_adapter_gating()
    test_asym_metadata()
    test_asym_forward_shape()
    test_asym_ref_kv_cache()
    test_apply_model_asym()
    test_forward_with_refs_multi_slot()
    test_conditioning_slots()
    test_guider_math()
    test_guider_apg_orthogonal()
    test_guider_rescale_matches_std()
    test_component_wiring()
    if FAILED:
        print("\nFAILED:")
        for item in FAILED:
            print("  " + item)
        sys.exit(1)
    print("\nall anima ref checks passed")


if __name__ == "__main__":
    main()
