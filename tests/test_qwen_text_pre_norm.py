"""Qwen-Image 2.1 text encoder: conditioning must be the *pre*-norm states.

Qwen-Image 2.1 conditions its DiT on the last Qwen3-VL decoder layer's output
before the final RMSNorm. From transformers 5.0 the encoder's captured
``hidden_states[-1]`` is tied to ``last_hidden_state`` and so comes back
*post*-norm; ``Qwen3VLTextEncoder._pre_norm`` neutralizes that norm with a
forward hook. This pins the fix on a tiny local model (no real weights needed).

Run: .venv\\Scripts\\python.exe tests\\test_qwen_text_pre_norm.py
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine.qwen_text import Qwen3VLTextEncoder  # noqa: E402

QWEN_DATA = os.path.join(ROOT, "engine", "tokenizer_data")

FAILED = []


def check(name, cond):
    if cond:
        print(f"ok  {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name}")


def tiny_qwen3vl(layers=3):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel

    torch.manual_seed(0)
    cfg = Qwen3VLTextConfig(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu",
        max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=5000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        attention_bias=False, tie_word_embeddings=False,
    )
    return Qwen3VLTextModel(cfg).eval()


def make_encoder():
    from transformers import Qwen2Tokenizer

    enc = object.__new__(Qwen3VLTextEncoder)
    enc.device = "cpu"
    enc.dtype = torch.float32
    enc.tokenizer = Qwen2Tokenizer.from_pretrained(QWEN_DATA)
    enc.model = tiny_qwen3vl()
    return enc


def test_forward_is_pre_norm():
    enc = make_encoder()
    model = enc.model
    ids = [5, 11, 2, 42, 7, 99, 1, 64]

    # The DiT's target is the last decoder layer's output (== the input to the
    # final RMSNorm); capture it directly, and the post-norm state too.
    captured = {}
    model.layers[-1].register_forward_hook(
        lambda m, inp, out: captured.__setitem__("pre", out.detach())
    )
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), output_hidden_states=True)
    pre = captured["pre"]
    post = out.hidden_states[-1]

    check("transformers hidden_states[-1] is post-norm",
          torch.allclose(pre, post, atol=1e-5) is False)

    got = enc._forward(ids)
    check("encoder forward matches pre-norm layer output",
          torch.allclose(got, pre, atol=1e-5))
    check("encoder forward is not the post-norm state",
          torch.allclose(got, post, atol=1e-5) is False)
    check("hook is removed after the call", len(model.norm._forward_hooks) == 0)


def main():
    test_forward_is_pre_norm()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
