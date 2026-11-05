"""Integration smoke tests: run each text encoder's weighted path on tiny local
models (no real weights needed). Run: .venv\\Scripts\\python.exe tests\\test_prompt_weights_integration.py
"""

import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import checkpoint as ckpt_mod  # noqa: E402
from engine import prompt_weights as pw  # noqa: E402
from engine.flux_text import T5Encoder  # noqa: E402
from engine.krea2_text import Krea2TextEncoder, PREFIX as KREA_PREFIX, SUFFIX as KREA_SUFFIX  # noqa: E402
from engine.qwen_text import Qwen3VLTextEncoder  # noqa: E402
from engine.text_encoder import TextEncoder  # noqa: E402
from engine.z_image_text import ZImageTextEncoder  # noqa: E402

QWEN_DATA = os.path.join(ROOT, "engine", "tokenizer_data")
T5_DATA = os.path.join(ROOT, "engine", "t5_tokenizer_data")
CLIP_REPO = "openai/clip-vit-large-patch14"

FAILED = []
PLAIN = "a photo of a red car"
MARKED = "a photo of a (red:2) car"


def check(name, cond):
    if cond:
        print(f"ok  {name}")
    else:
        FAILED.append(name)
        print(f"FAIL {name}")


def qwen_tokenizer():
    from transformers import Qwen2Tokenizer
    return Qwen2Tokenizer.from_pretrained(QWEN_DATA)


def tiny_clip(with_projection=False):
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection
    cfg = CLIPTextConfig(
        vocab_size=49408, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, max_position_embeddings=77, hidden_act="quick_gelu",
        layer_norm_eps=1e-5, projection_dim=32,
    )
    cls = CLIPTextModelWithProjection if with_projection else CLIPTextModel
    return cls(cfg).eval()


def tiny_t5_model():
    from transformers import T5Config, T5EncoderModel
    cfg = T5Config(
        vocab_size=32128, d_model=32, d_kv=8, d_ff=64, num_layers=2,
        num_decoder_layers=2, num_heads=4, relative_attention_num_buckets=32,
        relative_attention_max_distance=128, dropout_rate=0.0,
        layer_norm_epsilon=1e-6, feed_forward_proj="gated-gelu",
        is_encoder_decoder=True, use_cache=False, tie_word_embeddings=False,
    )
    return T5EncoderModel(cfg).eval()


def tiny_t5_encoder():
    from transformers import T5TokenizerFast
    enc = object.__new__(T5Encoder)
    enc.device = "cpu"
    enc.tokenizer = T5TokenizerFast.from_pretrained(T5_DATA)
    enc.model = tiny_t5_model()
    return enc


def tiny_qwen3(layers=2):
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=40960, rms_norm_eps=1e-6, rope_theta=1000000.0,
        attention_bias=False, tie_word_embeddings=False,
    )
    return Qwen3Model(cfg).eval()


def tiny_qwen3vl(layers=2):
    from transformers import Qwen3VLTextConfig, Qwen3VLTextModel
    cfg = Qwen3VLTextConfig(
        vocab_size=151936, hidden_size=32, intermediate_size=64, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, hidden_act="silu",
        max_position_embeddings=262144, rms_norm_eps=1e-6, rope_theta=5000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
        attention_bias=False, tie_word_embeddings=False,
    )
    return Qwen3VLTextModel(cfg).eval()


def differs(a, b):
    if a.shape != b.shape:
        return True
    return not torch.allclose(a.float(), b.float())


def finite(t):
    return torch.isfinite(t.float()).all().item()


def run_anima():
    enc = object.__new__(TextEncoder)
    enc.device = "cpu"
    enc.dtype = torch.float32
    enc.qwen_tok = qwen_tokenizer()
    from transformers import T5TokenizerFast
    enc.t5_tok = T5TokenizerFast.from_pretrained(T5_DATA)
    enc.model = tiny_qwen3()

    h_plain, _, w_plain = enc.encode(PLAIN)
    h_one, _, w_one = enc.encode("a photo of a (red:1.0) car")
    h_marked, t5, w_marked = enc.encode(MARKED)
    # Emphasis rides on the T5 tokens, so the Qwen hidden states are unweighted
    # and identical whether or not the prompt carries weights.
    check("anima qwen unweighted", torch.allclose(h_plain, h_marked))
    check("anima weight-1 no-weights", w_plain is None and w_one is None)
    check("anima weights present", w_marked is not None)
    check("anima weights differ", w_marked != w_plain)
    check("anima weights len", len(w_marked) == t5.shape[1])
    check("anima finite", finite(h_marked) and all(map(lambda x: x == x, w_marked)))
    check("anima t5 shape", t5.dim() == 2)


def run_t5():
    enc = tiny_t5_encoder()
    o_plain = enc.encode(PLAIN)
    o_one = enc.encode("a photo of a (red:1.0) car")
    o_marked = enc.encode(MARKED)
    check("t5 weight-1 no-op", torch.allclose(o_plain, o_one))
    check("t5 weighted differs", differs(o_plain, o_marked))
    check("t5 finite", finite(o_marked))


def run_classic():
    from transformers import CLIPTokenizer
    ck = object.__new__(ckpt_mod.ClassicCheckpoint)
    ck.family = "sd15"
    ck.device = "cpu"
    ck.tokenizer = CLIPTokenizer.from_pretrained(CLIP_REPO)
    ck.text_encoders = [tiny_clip()]
    a = ck.encode(PLAIN)["prompt_embeds"]
    b = ck.encode("a photo of a (red:1.0) car")["prompt_embeds"]
    c = ck.encode(MARKED)["prompt_embeds"]
    check("classic weight-1 no-op", torch.allclose(a, b))
    check("classic weighted differs", differs(a, c))
    check("classic finite", finite(c))


def run_z():
    enc = object.__new__(ZImageTextEncoder)
    enc.device = "cpu"
    enc.tokenizer = qwen_tokenizer()
    enc.model = tiny_qwen3()
    a = enc.encode(PLAIN)
    b = enc.encode("a photo of a (red:1.0) car")
    c = enc.encode(MARKED)
    # ComfyUI's Z-Image encoder runs with disable_weights=True: the prompt is
    # tokenized verbatim, so emphasis markup is ordinary text, not a no-op.
    check("z markup literal", differs(a, b))
    check("z verbatim differs", differs(a, c))
    check("z finite", finite(a) and finite(c))


def run_qwen():
    enc = object.__new__(Qwen3VLTextEncoder)
    enc.device = "cpu"
    enc.tokenizer = qwen_tokenizer()
    enc.model = tiny_qwen3vl()
    a = enc.encode(PLAIN)
    b = enc.encode("a photo of a (red:1.0) car")
    c = enc.encode(MARKED)
    # Qwen-Image: disable_weights=True, prompt tokenized verbatim.
    check("qwen markup literal", differs(a, b))
    check("qwen verbatim differs", differs(a, c))
    check("qwen finite", finite(a) and finite(c))


def run_krea2():
    enc = object.__new__(Krea2TextEncoder)
    enc.device = "cpu"
    enc.tokenizer = qwen_tokenizer()
    enc.model = tiny_qwen3vl(layers=36)
    enc._prefix_len = len(enc._encode_ids(KREA_PREFIX))
    enc._suffix_width = len(enc._encode_ids(KREA_SUFFIX))
    a, _ = enc.encode(PLAIN)
    b, _ = enc.encode("a photo of a (red:1.0) car")
    c, _ = enc.encode(MARKED)
    # Krea 2: disable_weights=True, prompt tokenized verbatim.
    check("krea markup literal", differs(a, b))
    check("krea verbatim differs", differs(a, c))
    check("krea finite", finite(a) and finite(c))


if __name__ == "__main__":
    for name, fn in [
        ("anima", run_anima), ("flux t5", run_t5), ("classic", run_classic),
        ("z", run_z), ("qwen", run_qwen), ("krea2", run_krea2),
    ]:
        print(f"-- {name} --")
        fn()
    if FAILED:
        print("\nFAILURES:\n  " + "\n  ".join(FAILED))
        sys.exit(1)
    print("\nall integration smoke tests passed")
