"""Qwen3-VL text streamer: numerical equivalence to transformers.

Builds a tiny ``Qwen3VLTextModel`` in float32, saves it in the ComfyUI
``model.`` key layout the streamer reads, and checks that the streamed forward
matches ``transformers`` layer-for-layer. This pins the op ordering, RoPE table
and the "no final norm" (layer-50 output) semantics without needing the 32B
weights.
"""

import os
import sys
import tempfile

import torch
from safetensors.torch import save_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.h3_text import Qwen3VLTextStreamEncoder  # noqa: E402


def _tiny_config():
    from transformers import Qwen3VLTextConfig

    config = Qwen3VLTextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        hidden_act="silu",
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=5_000_000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 4, 4]},
        attention_bias=False,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return config


def _streamer_config():
    return {
        "vocab": 128,
        "hidden": 64,
        "intermediate": 128,
        "num_layers": 3,
        "heads": 4,
        "kv_heads": 2,
        "head_dim": 16,
        "eps": 1e-6,
        "rope_theta": 5_000_000.0,
    }


def test_matches_transformers():
    from transformers import Qwen3VLTextModel

    torch.manual_seed(0)
    model = Qwen3VLTextModel(_tiny_config()).eval()
    ids = [5, 11, 2, 42, 7, 99, 1, 64]

    # The H3 conditioning is the *unnormalized* last-layer output (ComfyUI
    # truncates at layer 50, so there is no final norm). transformers'
    # ``hidden_states[-1]`` is the post-norm output, so capture the real target
    # from the last decoder layer instead.
    captured = {}
    model.layers[-1].register_forward_hook(
        lambda m, inp, out: captured.__setitem__("x", out.detach())
    )
    with torch.no_grad():
        model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    reference = captured["x"]

    state = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()
             if not k.endswith("rotary_emb.inv_freq")}
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "tiny.safetensors")
        save_file(state, path)
        encoder = Qwen3VLTextStreamEncoder(path, dtype=torch.float32,
                                           config=_streamer_config())
        with torch.no_grad():
            got = encoder._forward(ids)

    assert got.shape == reference.shape, (got.shape, reference.shape)
    error = (got - reference).abs().max().item()
    print(f"streamed vs transformers max abs error: {error:.3e}  shape {tuple(got.shape)}")
    assert torch.allclose(got, reference, atol=1e-3, rtol=1e-3), error


if __name__ == "__main__":
    test_matches_transformers()
    print("all passed")
