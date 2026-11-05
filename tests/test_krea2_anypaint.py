"""Checks for Krea 2 AnyPaint (arbitrary-mask inpainting/outpainting helper).

Weights-free: the input-preparation helper, the latent-grid packing, the
destination-relative reference coordinates and the per-step known-latent
constraint are checked against hand-computed values, and a tiny randomly-seeded
Krea 2 DiT (2 layers) exercises the cached reference-attention forward path.

Run: .venv\\Scripts\\python.exe tests\\test_krea2_anypaint.py
"""

import os
import sys

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import krea2_anypaint as ap  # noqa: E402
from engine.krea2 import Krea2Transformer2DModel  # noqa: E402
from engine import component_kinds  # noqa: E402

PASSED = 0
FAILED = []


def check(name, got, want):
    global PASSED
    if got == want:
        PASSED += 1
        print(f"ok  {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name}: got {got!r} want {want!r}")


def _source(size=128, seed=0):
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size, 3) * 255).astype("uint8"))


def test_prepare_inpaint():
    src = _source()
    mask = Image.new("L", (128, 128), 0)
    mask.paste(255, (40, 40, 80, 80))
    prep = ap.prepare_anypaint(src, mask, (128, 128))
    check("canvas kept", prep.canvas_size, (128, 128))
    check("bbox full", prep.source_bbox, (0, 0, 128, 128))
    check("known is canvas-sized", prep.known_image.size, (128, 128))
    check("keep is canvas-sized", prep.keep_mask.size, (128, 128))
    # 40x40 generated, dilated by 32 each way (up to the 128 canvas).
    gen = np.asarray(prep.generated_mask) > 0
    check("generated area", int(gen.sum()), 40 * 40)
    kept = np.asarray(prep.keep_mask) > 127
    check("keep area", int(kept.sum()), 128 * 128 - 104 * 104)
    # The condition is scaled to the 384 max edge only when larger than that.
    check("condition max edge", max(prep.condition.size) <= 384, True)
    check("condition multiple of 16", prep.condition.width % 16 == 0
          and prep.condition.height % 16 == 0, True)


def test_prepare_outpaint():
    # A 64x64 source placed in the top-left of a 192x128 canvas.
    src = _source(64, seed=1)
    mask = Image.new("L", (64, 64), 0)  # nothing extra generated inside the source
    prep = ap.prepare_anypaint(src, mask, (192, 128), source_bbox=(0, 0, 64, 64))
    gen = np.asarray(prep.generated_mask) > 0
    # Everything outside the placed source is generated.
    expected = 192 * 128 - 64 * 64
    check("outpaint generated area", int(gen.sum()), expected)
    check("placement normalized", prep.reference_placement["bbox_normalized"],
          [0.0, 0.0, 1.0, 1.0])


def test_prepare_rejects_bad_canvas():
    src = _source(64)
    mask = Image.new("L", (64, 64), 0)
    raised = False
    try:
        ap.prepare_anypaint(src, mask, (100, 64))
    except ValueError:
        raised = True
    check("non-multiple-of-16 canvas rejected", raised, True)


def test_reference_positions():
    pos = ap.reference_positions(8, 8, (16, 16), bbox=(0.0, 0.0, 1.0, 1.0), frame=1)
    # 8 latent / 2 patch = 4 -> a 4x4 reference token grid.
    check("reference token count", tuple(pos.shape), (16, 3))
    check("frame axis", float(pos[:, 0].max()), 1.0)
    # A full-bbox reference spans the destination grid: first cell ~0, last ~15.
    ys = pos[:, 1]
    check("spans grid start", round(float(ys.min()), 3), round(0.5 * 16 / 4 - 0.5, 3))
    check("spans grid end", round(float(ys.max()), 3),
          round((3 + 0.5) * (16 / 4) - 0.5, 3))


def test_latent_packing_roundtrip():
    latent = torch.arange(1 * 16 * 1 * 8 * 8, dtype=torch.float32).reshape(1, 16, 1, 8, 8)
    toks = ap.pack_latents(latent)
    check("packed shape", tuple(toks.shape), (1, 16, 64))
    # Reverse the packing to confirm the patch layout is the model's _process_img.
    b, c, _, h, w = latent.shape
    p = ap.PATCH
    ref = latent[:, :, 0].reshape(b, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    ref = ref.reshape(b, (h // p) * (w // p), c * p * p)
    check("packing matches _process_img", bool(torch.equal(toks, ref)), True)


def test_known_mask_and_restore():
    keep_mask = Image.new("L", (64, 64), 255)
    keep_mask.paste(0, (0, 0, 16, 16))  # a 16x16 generated block -> 8x8 latent cell
    token = ap.known_token_mask(keep_mask, 8, 8)  # latent grid 8x8, patch 2 -> 4x4 tok
    check("token mask shape", tuple(token.shape), (1, 16, 1))
    # A 16x16 pixel block of a 64px image is 2x2 latent pixels = one token.
    check("kept token count", int(token.sum()), 15)
    spatial = ap.expand_token_mask(token, 8, 8)
    check("expanded shape", tuple(spatial.shape), (1, 1, 8, 8))
    check("expanded kept count", int(spatial.sum()), 15 * 4)

    x = torch.full((1, 4, 1, 8, 8), 0.5)
    known = torch.zeros_like(x)
    noise = torch.ones_like(x)
    out = ap.restore_known(x, known, spatial.bool(), noise, 0.25)
    # Preserved tokens -> 0.25 (known + sigma * (noise - known)); generated unchanged.
    check("preserved value", round(float(out[0, 0, 0, 6, 6]), 3), 0.25)
    check("generated value", round(float(out[0, 0, 0, 0, 0]), 3), 0.5)


def test_adapter_detection():
    check("kind label", component_kinds.KINDS["anypaint"][0], "Krea 2 AnyPaint")
    check("adapter category", component_kinds.KINDS["anypaint"][1],
          component_kinds.ADAPTER)
    check("pool kind", "anypaint" in component_kinds.CATEGORY_KINDS[component_kinds.ADAPTER],
          True)


def _tiny_dit():
    return Krea2Transformer2DModel(
        features=64, tdim=16, txtdim=8, heads=4, kvheads=4, multiplier=1,
        layers=2, patch=2, channels=4, theta=1000.0, txtlayers=2, txtheads=2,
        txtkvheads=2,
    )


def test_reference_forward():
    torch.manual_seed(0)
    dit = _tiny_dit().eval()
    x = torch.randn(1, 4, 8, 8)
    ctx = torch.randn(1, 5, 16)
    t = torch.tensor([0.5])
    attn_mask = torch.ones(1, 5, dtype=torch.bool)
    attn_mask[0, 3:] = False
    ref_tokens = torch.randn(1, 9, 16)
    ref_pos = torch.zeros(9, 3)
    ref_pos[:, 0] = 1
    with torch.no_grad():
        plain = dit(x, t, ctx, attention_mask=attn_mask)
        kv = dit.precompute_ref_kv(ref_tokens, ref_pos)
        with_ref = dit(x, t, ctx, attention_mask=attn_mask, ref_kv_cache=kv)
    check("plain shape", tuple(plain.shape), (1, 4, 8, 8))
    check("ref forward shape", tuple(with_ref.shape), (1, 4, 8, 8))
    check("one kv per block", len(kv), 2)
    # k is (B, kvheads, ref_seq, headdim); headdim = 64/4 = 16.
    check("kv shape", tuple(kv[0][0].shape), (1, 4, 9, 16))
    # The reference changes the output (the cache is actually attended to).
    check("reference changes output", bool(torch.allclose(plain, with_ref)), False)


def main():
    for fn in (test_prepare_inpaint, test_prepare_outpaint,
               test_prepare_rejects_bad_canvas, test_reference_positions,
               test_latent_packing_roundtrip, test_known_mask_and_restore,
               test_adapter_detection, test_reference_forward):
        fn()
    if FAILED:
        print("\nFAILED:")
        for line in FAILED:
            print(" -", line)
        sys.exit(1)
    print("\nall krea2 anypaint tests passed")


if __name__ == "__main__":
    main()
