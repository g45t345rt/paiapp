"""Anima's LLM adapter.

Anima conditions its Cosmos-Predict2 DiT on the output of a small adapter that
takes the Qwen3 text hidden states as a *source* sequence and a **T5**
token-id sequence as its *target* queries, then cross-attends the two. The
result is the cross-attention context of the DiT. This matches the ``Anima``
model in ComfyUI (``comfy/ldm/anima/model.py``).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import attention
from .cosmos import MiniTrainDIT, RMSNorm


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta=10000):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class AdapterAttention(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, head_dim):
        super().__init__()
        inner = head_dim * n_heads
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(query_dim, inner, bias=False)
        self.q_norm = RMSNorm(head_dim, eps=1e-6)
        self.k_proj = nn.Linear(context_dim, inner, bias=False)
        self.k_norm = RMSNorm(head_dim, eps=1e-6)
        self.v_proj = nn.Linear(context_dim, inner, bias=False)
        self.o_proj = nn.Linear(inner, query_dim, bias=False)

    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        context = x if context is None else context
        in_shape = x.shape[:-1]
        ctx_shape = context.shape[:-1]
        q = self.q_norm(self.q_proj(x).view(*in_shape, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(context).view(*ctx_shape, self.n_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(context).view(*ctx_shape, self.n_heads, self.head_dim).transpose(1, 2)
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q = _apply_rope(q, cos, sin)
            cos, sin = position_embeddings_context
            k = _apply_rope(k, cos, sin)
        out = attention(q, k, v, attn_mask=mask)
        out = out.transpose(1, 2).reshape(*in_shape, -1).contiguous()
        return self.o_proj(out)


class AdapterBlock(nn.Module):
    def __init__(self, source_dim, model_dim, num_heads=16, mlp_ratio=4.0, use_self_attn=True, layer_norm=False):
        super().__init__()
        self.use_self_attn = use_self_attn
        norm = nn.LayerNorm if layer_norm else (lambda d: RMSNorm(d, eps=1e-6))
        if use_self_attn:
            self.norm_self_attn = norm(model_dim)
            self.self_attn = AdapterAttention(model_dim, model_dim, num_heads, model_dim // num_heads)
        self.norm_cross_attn = norm(model_dim)
        self.cross_attn = AdapterAttention(model_dim, source_dim, num_heads, model_dim // num_heads)
        self.norm_mlp = norm(model_dim)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, int(model_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(model_dim * mlp_ratio), model_dim),
        )

    def forward(self, x, context, target_attention_mask=None, source_attention_mask=None,
                position_embeddings=None, position_embeddings_context=None):
        if self.use_self_attn:
            x = x + self.self_attn(self.norm_self_attn(x), mask=target_attention_mask,
                                   position_embeddings=position_embeddings,
                                   position_embeddings_context=position_embeddings)
        x = x + self.cross_attn(self.norm_cross_attn(x), mask=source_attention_mask, context=context,
                                position_embeddings=position_embeddings,
                                position_embeddings_context=position_embeddings_context)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class LLMAdapter(nn.Module):
    def __init__(self, source_dim=1024, target_dim=1024, model_dim=1024, num_layers=6,
                 num_heads=16, use_self_attn=True, layer_norm=False):
        super().__init__()
        self.embed = nn.Embedding(32128, target_dim)
        self.in_proj = nn.Identity() if model_dim == target_dim else nn.Linear(target_dim, model_dim)
        self.rotary_emb = RotaryEmbedding(model_dim // num_heads)
        self.blocks = nn.ModuleList([
            AdapterBlock(source_dim, model_dim, num_heads, 4.0, use_self_attn, layer_norm)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(model_dim, target_dim)
        self.norm = RMSNorm(target_dim, eps=1e-6)

    def forward(self, source_hidden_states, target_input_ids, target_attention_mask=None, source_attention_mask=None):
        context = source_hidden_states
        x = self.in_proj(self.embed(target_input_ids).to(context.dtype))
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        pos_ctx = torch.arange(context.shape[1], device=x.device).unsqueeze(0)
        pe = self.rotary_emb(x, pos)
        pe_ctx = self.rotary_emb(x, pos_ctx)
        for block in self.blocks:
            x = block(x, context, target_attention_mask, source_attention_mask, pe, pe_ctx)
        return self.norm(self.out_proj(x))


def preprocess_text_embeds(adapter, text_embeds, text_ids, t5xxl_weights=None):
    """Compute the DiT cross-attention context (padded to at least 512 tokens).

    ``t5xxl_weights`` is an optional per-token emphasis weight (a list/tensor
    aligned to ``text_ids``); it scales the adapter output, matching ComfyUI's
    Anima where the prompt's ``(text:weight)`` emphasis rides on the T5 tokens.
    """
    out = adapter(text_embeds, text_ids)
    if t5xxl_weights is not None:
        weights = torch.as_tensor(t5xxl_weights, device=out.device, dtype=out.dtype)
        out = out * weights.reshape(1, -1, 1)
    if out.shape[1] < 512:
        out = F.pad(out, (0, 0, 0, 512 - out.shape[1]))
    return out


def load_anima(path):
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    prefix = "model.diffusion_model."
    dit_sd = {}
    adapter_sd = {}
    for k, v in state.items():
        if not k.startswith(prefix):
            continue
        k2 = k[len(prefix):]
        if k2.startswith("llm_adapter."):
            adapter_sd[k2[len("llm_adapter."):]] = v
        else:
            dit_sd[k2] = v

    dit = MiniTrainDIT()
    m1, u1 = dit.load_state_dict(dit_sd, strict=False)
    adapter = LLMAdapter()
    m2, u2 = adapter.load_state_dict(adapter_sd, strict=False)
    return dit, adapter, (m1, u1, m2, u2)
