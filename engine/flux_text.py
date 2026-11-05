"""Flux text conditioning: CLIP-L (pooled) + T5-XXL (context).

Flux conditions its DiT on the **T5-XXL** last hidden states (4096-dim, the
``txt_in`` context) and the **CLIP-L** pooled output (768-dim, the ``vector_in``
vector). Matches ComfyUI's ``comfy/text_encoders/flux.py``.

CLIP-L loads from a ComfyUI ``clip_l.safetensors`` (its keys already match HF
``CLIPTextModel``); T5-XXL loads from a llama.cpp ``t5-v1_1-xxl`` GGUF via
:mod:`engine.gguf`. Only ``transformers`` is used for the two towers.
"""

import torch

from . import gguf as gguf_reader
from . import prompt_weights
from .precision import build_on_meta

# llama.cpp ``enc.blk.N.<tail>`` -> transformers ``T5EncoderModel`` suffix.
_T5_BLK_MAP = {
    "attn_q.weight": "layer.0.SelfAttention.q.weight",
    "attn_k.weight": "layer.0.SelfAttention.k.weight",
    "attn_v.weight": "layer.0.SelfAttention.v.weight",
    "attn_o.weight": "layer.0.SelfAttention.o.weight",
    "attn_rel_b.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
    "attn_norm.weight": "layer.0.layer_norm.weight",
    "ffn_norm.weight": "layer.1.layer_norm.weight",
    "ffn_gate.weight": "layer.1.DenseReluDense.wi_0.weight",
    "ffn_up.weight": "layer.1.DenseReluDense.wi_1.weight",
    "ffn_down.weight": "layer.1.DenseReluDense.wo.weight",
}


def _map_t5_name(key):
    if key == "token_embd.weight":
        return "shared.weight"
    if key == "enc.output_norm.weight":
        return "encoder.final_layer_norm.weight"
    if key.startswith("enc.blk."):
        idx, _, tail = key[len("enc.blk."):].partition(".")
        mapped = _T5_BLK_MAP.get(tail)
        return f"encoder.block.{idx}.{mapped}" if mapped else None
    return None


def _t5_config():
    from transformers import T5Config

    return T5Config(
        vocab_size=32128,
        d_model=4096,
        d_kv=64,
        d_ff=10240,
        num_layers=24,
        num_decoder_layers=24,
        num_heads=64,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.0,
        layer_norm_epsilon=1e-6,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=False,
        tie_word_embeddings=False,
    )


class CLIPLEncoder:
    """CLIP-L pooled-output encoder for Flux."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        from safetensors import safe_open
        from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

        from .checkpoint import _CLIP_REPO, _TEXT_L

        self.device = device
        self.dtype = dtype
        self.tokenizer = CLIPTokenizer.from_pretrained(_CLIP_REPO)

        # Build on the ``meta`` device: no parameter storage is allocated and no
        # random initialiser runs (the checkpoint overwrites every weight). The
        # loaded tensors are *assigned* in place (``assign=True``).
        with build_on_meta(
            dtype, lambda: CLIPTextModel(CLIPTextConfig(**_TEXT_L))
        ) as model:
            pass
        params = dict(model.named_parameters())
        state = {}
        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            total = max(1, len(keys))
            if tick is not None:
                tick(0.0)
            for position, key in enumerate(keys):
                # The ComfyUI file stores the tower under ``text_model.`` while
                # the built ``CLIPTextModel``'s parameters are unprefixed; accept
                # either so the weights actually load.
                name = key[len("text_model."):] if key.startswith("text_model.") else key
                dst = params.get(name)
                if dst is not None:
                    state[name] = handle.get_tensor(key).reshape(dst.shape).to(dtype)
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
        """Return the pooled CLIP-L vector ``[1, 768]`` for ``text``."""
        tokens = self.tokenizer(
            text or "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        out = self.model(tokens.input_ids.to(self.device))
        return out.pooler_output


class T5Encoder:
    """T5-XXL last-hidden-state encoder for Flux (GGUF weights)."""

    def __init__(self, weights_path, device="cpu", dtype=torch.bfloat16, tick=None):
        from transformers import T5EncoderModel, T5Tokenizer

        self.device = device
        self.dtype = dtype
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")

        # Build on the ``meta`` device: no ~9.5 GB bf16 parameter allocation and
        # no random initialiser runs (they cost minutes over T5-XXL's ~4.8B
        # parameters). Every weight is read from the GGUF and *assigned* in place
        # (``assign=True``), so the peak stays near one copy of the weights.
        with build_on_meta(dtype, lambda: T5EncoderModel(_t5_config())) as model:
            pass
        params = dict(model.named_parameters())
        reader = gguf_reader.gguf.GGUFReader(weights_path)
        tensors = list(reader.tensors)
        total = max(1, len(tensors))
        state = {}
        if tick is not None:
            tick(0.0)
        for position, tensor in enumerate(tensors):
            name = _map_t5_name(tensor.name)
            if name is None or name not in params:
                continue
            state[name] = gguf_reader.dequantize(tensor, dtype=dtype).reshape(params[name].shape)
            if tick is not None and position % 8 == 0:
                tick((position + 1) / total)
        if tick is not None:
            tick(1.0)

        model.load_state_dict(state, strict=False, assign=True)
        model.tie_weights()
        loaded = set(state)
        self.missing = [n for n in params if n not in loaded]

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
    def encode(self, text, length=256):
        """Return the T5 context ``[1, T, 4096]`` for ``text``.

        ``length`` is the minimum padded length (Flux pads to 256);
        longer prompts are not truncated. ``(text:weight)`` emphasis weights the
        token hidden states (see :mod:`engine.prompt_weights`).
        """
        pairs = prompt_weights.parse(text or "")
        clean = prompt_weights.plain(pairs)
        ids = self._tokenize(clean, length)
        out = self._forward(ids)
        if prompt_weights.has_weights(pairs):
            empty = self._forward(self._tokenize("", max(length, len(ids))))
            weights = prompt_weights.token_weights(
                self.tokenizer, clean, pairs, target_ids=ids,
            )
            out = prompt_weights.interpolate(out, empty, weights)
        return out

    def _tokenize(self, text, length):
        tokens = self.tokenizer(
            text or "",
            padding="max_length",
            max_length=length,
            truncation=False,
            return_tensors="pt",
        )
        return tokens.input_ids[0].tolist()

    def _forward(self, ids):
        input_ids = torch.tensor([ids], device=self.device, dtype=torch.long)
        return self.model(input_ids=input_ids).last_hidden_state
