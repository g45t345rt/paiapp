"""ComfyUI weight formats shared by the text, DiT and VAE loaders.

The same ``.safetensors`` conventions show up across the engine:

* plain bf16/fp16/fp32 tensors;
* ``int8_tensorwise`` -- ``X.weight`` int8 plus a per-output-row
  ``X.weight_scale``, optionally ``convrot`` (a normalized Hadamard rotation of
  256-wide groups along the input dim);
* ``nvfp4`` -- packed E2M1 nibbles, fp8 block scales in a 128x4 swizzle, and a
  per-tensor ``weight_scale_2``;
* ``asym_w4a8_int8`` -- ConvRot-rotated packed int4 with a Lloyd-Max codebook,
  fp8 per-group scale (``weight_s_rel``) and fp32 per-channel scale
  (``weight_s_channel``).

Loaders should call :func:`_copy_weight` (streamed into a destination
parameter) or :func:`_dequant` (whole tensor) instead of reimplementing any of
this. AWQ activation smoothing (``pre_quant_scale``) is intentionally left to
the caller: it belongs to the model forward, not to the weight codec.
"""

import json
import math

import torch

# Row-chunk budget for loading weights; bounds the float32 transient so a 9B
# model loads within RAM.
_CHUNK_BYTES = 128 << 20

_DTYPE_BYTES = {
    "F64": 8, "I64": 8, "F32": 4, "I32": 4, "F16": 2, "BF16": 2, "I16": 2,
    "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}

# NVFP4 E2M1 codebook (Comfy/ModelOpt layout).
_FP4_E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _itemsize(dtype_name):
    return _DTYPE_BYTES.get(dtype_name, 4)


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


# ------------------------------------------------------------------ NVFP4
def _nvfp4_scales_to_linear(scales):
    """Undo FlashInfer/CUTLASS's 128x4 swizzle on Comfy NVFP4 block scales."""
    ndim = scales.ndim
    if ndim == 2:
        scales = scales.unsqueeze(0)
    batch, rows, cols = scales.shape
    rows_padded = _round_up(rows, 128)
    cols_padded = _round_up(cols, 4)
    if rows != rows_padded or cols != cols_padded:
        padded = torch.zeros((batch, rows_padded, cols_padded),
                             dtype=scales.dtype, device=scales.device)
        padded[:, :rows, :cols] = scales
        scales = padded
    linear = scales.reshape(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    linear = linear.permute(0, 1, 4, 3, 2, 5).contiguous()
    linear = linear.reshape(batch, rows_padded, cols_padded)[:, :rows, :cols]
    return linear.squeeze(0) if ndim == 2 else linear


def _copy_nvfp4_weight(handle, names, key, dst, dtype):
    """Dequantize packed NVFP4 (Comfy ``nvfp4``) rows into ``dst``.

    Layout: ``weight`` uint8 ``[out, in/2]`` (high nibble first), block scales
    ``weight_scale`` fp8-e4m3 ``[out, in/16]`` in the 128x4 swizzle, and a
    per-tensor ``weight_scale_2``.
    """
    scale_key = key + "_scale"
    scale2_key = key + "_scale_2"
    scale = _nvfp4_scales_to_linear(handle.get_tensor(scale_key)).to(torch.float32)
    scale2 = handle.get_tensor(scale2_key).to(torch.float32).reshape(())
    lut = torch.tensor(_FP4_E2M1, dtype=torch.float32)
    sl = handle.get_slice(key)
    shape = sl.get_shape()
    rows, packed_cols = int(shape[0]), int(shape[1])
    cols = packed_cols * 2
    step = max(1, _CHUNK_BYTES // max(1, packed_cols + cols * 4))
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        packed = sl[start:stop].to(torch.int64)
        low = packed & 0xF
        high = (packed >> 4) & 0xF
        value = torch.empty(stop - start, cols, dtype=torch.float32)
        value[:, 0::2] = lut[high]
        value[:, 1::2] = lut[low]
        blocks = scale[start:stop].repeat_interleave(16, dim=-1) * scale2
        dst[start:stop].copy_((value * blocks).to(dtype))


# -------------------------------------------------------------- int8/convrot
def _hadamard(size):
    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32,
    )
    h = h4
    while h.shape[0] < size:
        h = torch.kron(h, h4)
    return h / math.sqrt(size)


def _unrotate(weight, group_size):
    """Un-rotate the ``convrot`` rotation: ``W @ H_block`` along the last dim."""
    if group_size < 4 or weight.shape[-1] % group_size:
        return weight
    h = _hadamard(group_size).to(weight.dtype)
    rows, cols = weight.shape
    return (weight.reshape(rows, cols // group_size, group_size) @ h).reshape(rows, cols)


# ------------------------------------------------------------- W4A8/W6A8
def _copy_w4a8_weight(handle, names, key, conf, dst, dtype):
    """Dequantize Comfy ``asym_w4a8_int8`` rows into ``dst``.

    Storage: packed int4 nibbles in an int8 row (even col = low nibble), an fp8
    ``weight_s_rel`` group scale, an fp32 ``weight_s_channel`` per-row scale and
    an optional 16-entry Lloyd-Max ``weight_codebook``. The checkpoint is ConvRot
    rotated, so the result is un-rotated before it reaches the model.
    """
    s_rel_key = key + "_s_rel"
    s_channel_key = key + "_s_channel"
    if s_rel_key not in names or s_channel_key not in names:
        raise ValueError(f"asym_w4a8_int8 missing scales for {key}")
    s_rel = handle.get_tensor(s_rel_key).to(torch.float32)
    s_channel = handle.get_tensor(s_channel_key).to(torch.float32)
    groups = int(s_rel.shape[1])
    group_size = int(conf.get("group_size", 16))
    k = groups * group_size
    sl = handle.get_slice(key)
    if sl.get_dtype() != "I8":
        raise ValueError(f"asym_w4a8_int8 weight must be int8, got {sl.get_dtype()}")
    rows, packed_cols = [int(value) for value in sl.get_shape()]
    bits = packed_cols * 8 // max(1, k)
    if bits != 4:
        raise ValueError(
            f"asym_w4a8_int8 with {bits}-bit codes is not supported (expected 4-bit)")
    codebook_key = key + "_codebook"
    codebook = (
        handle.get_tensor(codebook_key).to(torch.float32)
        if codebook_key in names else None
    )
    correction_key = key + "_correction"
    correction = (
        handle.get_tensor(correction_key).to(torch.float32)
        if correction_key in names else None
    )
    convrot_groupsize = int(conf.get("convrot_groupsize", 256))
    convrot = bool(conf.get("convrot", False))
    step = max(1, _CHUNK_BYTES // max(1, packed_cols + k * 8))
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        packed = sl[start:stop].view(torch.uint8).to(torch.int32)
        codes = torch.empty(stop - start, k, dtype=torch.int64)
        codes[:, 0::2] = packed & 0xF
        codes[:, 1::2] = (packed >> 4) & 0xF
        if codebook is not None:
            values = codebook[codes]
        else:
            values = codes.to(torch.float32) - 8.0
        value = values.view(stop - start, groups, group_size)
        value = value * s_rel[start:stop].unsqueeze(-1)
        value = value.round().clamp_(-127, 127)
        value = value * s_channel[start:stop].view(-1, 1, 1)
        if correction is not None:
            value = value + correction[:, start:stop].t().unsqueeze(-1)
        value = value.reshape(stop - start, k)
        if convrot:
            value = _unrotate(value, convrot_groupsize)
        dst[start:stop].copy_(value.to(dtype))


# ------------------------------------------------------------------ config
def _strip_component_prefix(name):
    """Drop a leading module prefix so a tensor name matches metadata layer keys."""
    for prefix in (
        "model.diffusion_model.", "diffusion_model.", "model.language_model.",
        "language_model.", "model.",
    ):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _metadata_quant_layers(handle):
    """The ``_quantization_metadata`` layer map from the file's ``__metadata__``.

    Some int8 checkpoints (the MOSS DiT ``*_int8_convrot`` files) do not carry a
    sibling ``X.comfy_quant`` tensor at all; instead the whole quant config lives
    in the safetensors ``__metadata__`` as a JSON string keyed by module path
    (``layers.0.proj.q``). Returns ``{}`` when the file has no such metadata.
    """
    try:
        meta = handle.metadata() or {}
    except Exception:
        return {}
    raw = meta.get("_quantization_metadata")
    if not raw:
        return {}
    try:
        return json.loads(raw).get("layers", {}) or {}
    except (ValueError, TypeError):
        return {}


def _comfy_conf(handle, names, key):
    """Parse the sibling ``X.comfy_quant`` JSON tensor, if the file carries one."""
    if not key.endswith(".weight"):
        return {}
    conf_key = key[: -len(".weight")] + ".comfy_quant"
    if conf_key not in names:
        return {}
    try:
        raw = handle.get_tensor(conf_key).to(torch.uint8).numpy().tobytes()
        return json.loads(raw.decode())
    except (ValueError, TypeError):
        return {}


def _quant_conf(handle, names, key):
    """``(group_size, convrot)`` for a quantized weight (defaults when absent).

    A sibling ``X.comfy_quant`` tensor wins when present; otherwise the
    file-level ``_quantization_metadata`` is consulted (the MOSS DiT layout,
    which keys the config by module path rather than by tensor name).
    """
    conf = _comfy_conf(handle, names, key)
    if conf:
        return int(conf.get("convrot_groupsize", 256)), bool(conf.get("convrot"))
    name = _strip_component_prefix(key)
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    conf = _metadata_quant_layers(handle).get(name)
    if conf:
        return int(conf.get("convrot_groupsize", 256)), bool(conf.get("convrot"))
    return 0, False


# --------------------------------------------------------------- dequant
def _dequant(handle, names, key):
    """Reconstruct one tensor, dequantizing ComfyUI int8 (+ convrot) weights."""
    tensor = handle.get_tensor(key)
    scale_key = key + "_scale"
    if tensor.dtype != torch.int8 or scale_key not in names:
        return tensor
    value = tensor.to(torch.float32) * handle.get_tensor(scale_key).to(torch.float32)
    group_size, convrot = _quant_conf(handle, names, key)
    return _unrotate(value, group_size) if convrot else value


def _copy_weight(handle, names, key, dst, dtype):
    """Dequantize ``key`` into the ``dst`` parameter in bounded row chunks.

    Full-tensor materialisation would need ~4 GB for a 248320x4096 embedding
    alone; chunking keeps the float32 transient at ``_CHUNK_BYTES`` regardless of
    the tensor's size.
    """
    scale_key = key + "_scale"
    sl = handle.get_slice(key)
    shape = sl.get_shape()
    rows = int(shape[0])
    cols = int(shape[1]) if len(shape) > 1 else 1
    if (key + "_s_rel") in names and (key + "_s_channel") in names:
        conf = _comfy_conf(handle, names, key)
        if conf.get("format") == "asym_w4a8_int8":
            _copy_w4a8_weight(handle, names, key, conf, dst, dtype)
            return
    if sl.get_dtype() == "U8" and scale_key in names and (key + "_scale_2") in names:
        _copy_nvfp4_weight(handle, names, key, dst, dtype)
        return
    if sl.get_dtype() != "I8" or scale_key not in names:
        step = max(1, _CHUNK_BYTES // max(1, cols * _itemsize(sl.get_dtype())))
        for start in range(0, rows, step):
            dst[start:start + step].copy_(sl[start:start + step].to(dtype))
        return
    group_size, convrot = _quant_conf(handle, names, key)
    scale = handle.get_tensor(scale_key).to(torch.float32)
    per_row = scale.dim() >= 2
    scale_sl = handle.get_slice(scale_key) if per_row else None
    step = max(1, _CHUNK_BYTES // max(1, cols * 4))
    for start in range(0, rows, step):
        stop = start + step
        chunk = sl[start:stop].to(torch.float32)
        factor = scale_sl[start:stop].to(torch.float32) if per_row else scale
        value = chunk * factor
        if convrot:
            value = _unrotate(value, group_size)
        dst[start:stop].copy_(value.to(dtype))