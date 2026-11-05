"""Z-Image text conditioning: Qwen3-4B (penultimate layer).

Z-Image conditions its DiT on the **penultimate** hidden states of a **Qwen3-4B**
text model (``cap_feat_dim = 2560``), tokenised with the Qwen2 BPE tokenizer and
the Qwen3 chat template
``<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n`` (ComfyUI
``comfy/text_encoders/z_image.py``). The Qwen3 control tokens matter: the DiT was
trained on templated hidden states, and dropping them yields exactly the waxy
skin/detail artifacts a raw ``user\\n...\\nassistant\\n`` string produces. Only
``transformers`` is used.
"""

import torch
from safetensors import safe_open
from transformers import Qwen2Tokenizer, Qwen3Config, Qwen3Model

from .text_encoder import _QWEN_DATA

_PREFIX = "<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"


def _qwen3_4b_config():
    return Qwen3Config(
        vocab_size=151936,
        hidden_size=2560,
        intermediate_size=9728,
        num_hidden_layers=36,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        attention_bias=False,
        tie_word_embeddings=True,
    )


class ZImageTextEncoder:
    """Qwen3-4B penultimate hidden states for Z-Image."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        self.device = device
        self.tokenizer = Qwen2Tokenizer.from_pretrained(_QWEN_DATA)

        model = Qwen3Model(_qwen3_4b_config())
        params = dict(model.named_parameters())
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = [k[len("model."):] for k in handle.keys() if k.startswith("model.")]
            total = max(1, len(keys))
            if tick is not None:
                tick(0.0)
            for position, name in enumerate(keys):
                dst = params.get(name)
                if dst is not None:
                    with torch.no_grad():
                        src = handle.get_tensor("model." + name)
                        dst.copy_(src.reshape(dst.shape).to(dst.dtype))
                if tick is not None and position % 8 == 0:
                    tick((position + 1) / total)
        if tick is not None:
            tick(1.0)
        self.model = model.to(device=device, dtype=dtype).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, text):
        """Return the penultimate Qwen3-4B hidden states ``[1, L, 2560]``.

        The prompt is tokenized verbatim: like ComfyUI's Z-Image text encoder
        (``disable_weights=True``), ``(text:weight)`` emphasis is not interpreted.
        """
        return self._forward(self._ids(text or ""))

    def _ids(self, text):
        prompt = f"{_PREFIX}{text}{_SUFFIX}"
        ids = self.tokenizer(prompt)["input_ids"]
        if not ids:
            ids = [self.tokenizer.pad_token_id or 151643]
        return list(ids)

    def _forward(self, ids):
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        out = self.model(input_ids=input_ids, output_hidden_states=True)
        return out.hidden_states[-2]
