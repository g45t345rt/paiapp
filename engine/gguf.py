"""GGUF (llama.cpp) reader and weight dequantization.

The Qwen-Image 2.1 GGUF builds store the DiT (and optionally the Qwen3-VL text
encoder) as ``.gguf`` files: a mix of plain ``F32``/``F16``/``BF16`` tensors and
K-quantized blocks (``Q4_K`` / ``Q6_K`` for the names you have). This module
reads such a file and turns every tensor back into a dense ``torch`` tensor,
with no ComfyUI import.

The block dequantizers are ported from ``ComfyUI-GGUF`` by City96
(https://github.com/city96/ComfyUI-GGUF, Apache-2.0) so the bit layouts match
llama.cpp exactly. Only ``gguf``, ``numpy`` and ``torch`` are used.

Besides the K-quants this app shipped with (``Q4_K`` / ``Q5_K`` / ``Q6_K``), the
legacy (``Q4_0`` / ``Q4_1`` / ``Q5_0`` / ``Q5_1``), the ``Q2_K`` / ``Q3_K`` and
the i-quants (``IQ4_NL`` / ``IQ4_XS``) are handled too, so the widely-published
GGUF releases (Q4_0, Q3_K, IQ4_XS, ...) load instead of failing at load time.
The port matches WeeLLM's ``io/ggufs/gguf_dequant.py`` (MIT).
"""

import warnings

import numpy as np
import torch

import gguf

QK_K = 256
K_SCALE_SIZE = 12

_TORCH_NATIVE = (
    gguf.GGMLQuantizationType.F32,
    gguf.GGMLQuantizationType.F16,
    gguf.GGMLQuantizationType.BF16,
)

# Integer storage types carried verbatim (e.g. Gemma4 ``layer_scalar`` /
# ``pos_embedding`` rows); they are not quantized, just reinterpreted and cast.
_INT_TYPES = {}
for _name, _dtype in (("I8", torch.int8), ("I16", torch.int16), ("I32", torch.int32)):
    _enum = getattr(gguf.GGMLQuantizationType, _name, None)
    if _enum is not None:
        _INT_TYPES[_enum] = _dtype

# The 16-centroid table the i-quant ("IQ4") types index into.
_KVALUES = torch.tensor(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=torch.int8,
)


def is_quantized(tensor_type):
    return tensor_type not in _TORCH_NATIVE


def _split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def _to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


# ---------------------------------------------------------------------------
# Block dequantizers (llama.cpp layouts).
# ---------------------------------------------------------------------------

def _deq_bf16(blocks, block_size, type_size, dtype=None):
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


def _deq_q8_0(blocks, block_size, type_size, dtype=None):
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _deq_q4_k(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    shift = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> shift
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _deq_q5_k(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    shift = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> shift
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.arange(8, device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _deq_q6_k(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> shift
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _deq_q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return (d * qs) + m


def _deq_q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _deq_q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n_blocks, -1)
    return (d * qs) + m


def _deq_q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1)).to(torch.int8) - 8
    return d * qs


def _deq_q3_k(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> torch.arange(8, device=d.device, dtype=torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))


def _deq_q2_k(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def _deq_iq4_nl(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 1)).to(torch.int64)
    kvalues = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs).reshape((n_blocks, -1))
    return d * qs


def _deq_iq4_xs(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = _to_uint16(scales_h)
    shift_a = torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape((1, 1, 2))
    shift_b = torch.tensor([2 * i for i in range(QK_K // 32)], device=d.device, dtype=torch.uint8).reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1, 1)) >> shift_a.reshape((1, 1, 2))
    scales_h = scales_h.reshape((n_blocks, -1, 1)) >> shift_b.reshape((1, -1, 1))
    scales_l = scales_l.reshape((n_blocks, -1)) & 0x0F
    scales_h = scales_h.reshape((n_blocks, -1)).to(torch.uint8) & 0x03
    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n_blocks, -1, 32, 1)) & 0x0F
    kvalues = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    qs = torch.gather(kvalues, dim=-1, index=qs.to(torch.int64)).reshape((n_blocks, -1, 32))
    return (dl * qs).reshape((n_blocks, -1))


def _register(name, fn):
    enum = getattr(gguf.GGMLQuantizationType, name, None)
    if enum is not None:
        _DEQUANT[enum] = fn


_DEQUANT = {
    gguf.GGMLQuantizationType.BF16: _deq_bf16,
    gguf.GGMLQuantizationType.Q8_0: _deq_q8_0,
    gguf.GGMLQuantizationType.Q4_K: _deq_q4_k,
    gguf.GGMLQuantizationType.Q5_K: _deq_q5_k,
    gguf.GGMLQuantizationType.Q6_K: _deq_q6_k,
}
for _name, _fn in (
    ("Q5_1", _deq_q5_1), ("Q5_0", _deq_q5_0), ("Q4_1", _deq_q4_1), ("Q4_0", _deq_q4_0),
    ("Q3_K", _deq_q3_k), ("Q2_K", _deq_q2_k),
    ("IQ4_NL", _deq_iq4_nl), ("IQ4_XS", _deq_iq4_xs),
):
    _register(_name, _fn)
del _name, _fn


def raw_tensor(tensor):
    """A CPU ``uint8`` view of a GGUF tensor's raw bytes."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
        t = torch.from_numpy(np.asarray(tensor.data))
    return t.view(torch.uint8).flatten()


def dequantize(tensor, dtype=torch.bfloat16):
    """Return ``tensor`` as a dense tensor in ``dtype`` with its logical shape.

    Huge tensors are dequantized in chunks so the bit-manipulation islands a
    kernel needs (e.g. a 2.9 GB intermediate for a 1.45 GB bf16 tensor) never
    spike peak RAM -- which matters precisely in the low-RAM streaming path this
    feeds.
    """
    ttype = tensor.tensor_type
    shape = torch.Size(tuple(int(v) for v in reversed(tensor.shape)))
    data = raw_tensor(tensor)
    if ttype in _TORCH_NATIVE:
        if ttype == gguf.GGMLQuantizationType.BF16:
            flat = _deq_bf16(data.reshape(1, -1), 0, 0).flatten()[: int(np.prod(shape))]
        elif ttype == gguf.GGMLQuantizationType.F32:
            flat = data.view(torch.float32).flatten()[: int(np.prod(shape))]
        else:  # F16
            flat = data.view(torch.float16).flatten()[: int(np.prod(shape))]
        return flat.reshape(shape).to(dtype)

    if ttype in _INT_TYPES:
        return data.view(_INT_TYPES[ttype]).flatten()[: int(np.prod(shape))].reshape(shape).to(dtype)

    block_size, type_size = gguf.GGML_QUANT_SIZES[ttype]
    fn = _DEQUANT.get(ttype)
    if fn is None:
        new = gguf.quants.dequantize(data.numpy(), ttype)
        return torch.from_numpy(new).reshape(shape).to(dtype)
    n_blocks = data.numel() // type_size
    blocks = data.reshape((n_blocks, type_size))

    # ~32 MB of raw input per chunk keeps the intermediate small on big tensors.
    chunk_blocks = (1024 * 1024 * 32) // type_size
    if chunk_blocks > 0 and n_blocks > chunk_blocks:
        out = torch.empty(n_blocks * block_size, dtype=dtype)
        for start in range(0, n_blocks, chunk_blocks):
            end = min(start + chunk_blocks, n_blocks)
            piece = fn(blocks[start:end], block_size, type_size, dtype)
            out[start * block_size: end * block_size] = piece.flatten()
        return out.reshape(shape).to(dtype)

    out = fn(blocks, block_size, type_size, dtype)
    return out.reshape(shape).to(dtype)


def load_state_dict(path, dtype=torch.bfloat16, prefix=None):
    """Load a whole ``.gguf`` into ``{name: dense_tensor}`` (all on CPU).

    ``prefix`` strips a leading key prefix (e.g. ``"model.diffusion_model."``).
    """
    reader = gguf.GGUFReader(path)
    out = {}
    for tensor in reader.tensors:
        name = tensor.name
        if prefix is not None:
            if not name.startswith(prefix):
                continue
            name = name[len(prefix):]
        out[name] = dequantize(tensor, dtype=dtype)
    return out


def architecture(path):
    reader = gguf.GGUFReader(path)
    field = reader.get_field("general.architecture")
    if field is None:
        return None
    return str(field.parts[field.data[-1]], encoding="utf-8")


def tensor_summary(path):
    """Quantization type histogram plus total param/byte counts."""
    reader = gguf.GGUFReader(path)
    counts = {}
    params = 0
    for tensor in reader.tensors:
        name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
        counts[name] = counts.get(name, 0) + 1
        n = 1
        for d in tensor.shape:
            n *= int(d)
        params += n
    return {"types": counts, "params": params}
