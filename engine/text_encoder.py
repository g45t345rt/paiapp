"""Anima text conditioning.

Anima uses two tokenizers on the raw prompt text (no chat template):
* a Qwen2 BPE tokenizer feeding a Qwen3-0.6B encoder that produces the hidden
  states (the *source* sequence), and
* a T5 tokenizer whose token ids are the *target* queries of the LLM adapter.

Both are tokenized from the exact prompt string the user provides.

``(text:weight)`` emphasis rides on the **T5** tokens, not the Qwen hidden
states: matching ComfyUI's Anima, the Qwen3 token weights are fixed at 1.0 and
the caller scales the LLM-adapter output with the returned per-token weights
(see :func:`engine.anima.preprocess_text_embeds`).
"""

import os

import torch
from safetensors import safe_open
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model, T5TokenizerFast

from . import prompt_weights
from .precision import build_on_meta

_HERE = os.path.dirname(os.path.abspath(__file__))
_QWEN_DATA = os.path.join(_HERE, "tokenizer_data")
_T5_DATA = os.path.join(_HERE, "t5_tokenizer_data")


def _qwen3_config():
    return Qwen3Config(
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=32768,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )


class TextEncoder:
    """Qwen3-0.6B hidden states + T5 token ids for Anima."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        self.device = device
        self.dtype = dtype

        self.qwen_tok = Qwen2Tokenizer.from_pretrained(_QWEN_DATA)
        self.t5_tok = T5TokenizerFast.from_pretrained(_T5_DATA)

        # Build on the ``meta`` device: no parameter storage and no random
        # initialiser over the Qwen3-0.6B tower. The loaded tensors are *assigned*
        # in place (``assign=True``).
        with build_on_meta(dtype, lambda: Qwen3Model(_qwen3_config())) as model:
            pass
        params = dict(model.named_parameters())
        state = {}
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [k[len("model."):] for k in handle.keys() if k.startswith("model.")]
            total = max(1, len(keys))
            if tick is not None:
                tick(0.0)
            for position, name in enumerate(keys):
                dst = params.get(name)
                if dst is not None:
                    state[name] = handle.get_tensor("model." + name).reshape(dst.shape).to(dtype)
                if tick is not None and position % 8 == 0:
                    tick((position + 1) / total)
        if tick is not None:
            tick(1.0)
        model.load_state_dict(state, strict=False, assign=True)
        model.tie_weights()
        # Materialise the loaded parameters in the target dtype/device; params
        # the file did not carry stay on ``meta`` and are skipped.
        for p in model.parameters():
            if p.device.type != "meta" and (p.dtype != dtype or str(p.device) != str(device)):
                p.data = p.to(device=device, dtype=dtype)
        model.eval()
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text):
        """Return ``(qwen_hidden [1, N, 1024], t5_ids [1, M], t5_weights|None)``.

        ``(text:weight)`` emphasis weights the **T5** tokens: the Qwen hidden
        states are encoded unweighted (ComfyUI fixes the Qwen3 token weights at
        1.0) and the per-token weights are returned for the caller to scale the
        LLM-adapter output. ``t5_weights`` is ``None`` when the prompt carries no
        emphasis, so an unweighted prompt skips the scaling entirely.
        """
        pairs = prompt_weights.parse(text or "")
        clean = prompt_weights.plain(pairs)
        qwen_ids = self.qwen_tok(clean)["input_ids"]
        t5_ids = self.t5_tok(clean)["input_ids"]
        if len(qwen_ids) == 0:
            qwen_ids = [self.qwen_tok.pad_token_id or 151643]
        if len(t5_ids) == 0:
            t5_ids = [0]

        input_ids = torch.tensor([qwen_ids], device=self.device, dtype=torch.long)
        hidden = self.model(input_ids=input_ids).last_hidden_state

        weights = None
        if prompt_weights.has_weights(pairs):
            weights = prompt_weights.token_weights(
                self.t5_tok, clean, pairs, target_ids=t5_ids,
            )

        t5 = torch.tensor([t5_ids], device=self.device, dtype=torch.long)
        return hidden, t5, weights
