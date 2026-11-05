"""Low-RAM disk streaming for GGUF / safetensors diffusion transformers.

The normal loaders in this app dequantize every tensor into a dense ``bf16``
module that then sits in system RAM for the whole session. That is the friendly
path when RAM is plentiful, but a 20B model costs ~40 GB of RAM that way.

This module implements the alternative the "Stream weights from disk" setting
selects: the model is built on the ``meta`` device (no storage), only its small
non-block modules ("resident" keys -- embedders, modulation, final layer) are
loaded up front, and each transformer block is read from the checkpoint on
demand and evicted again after its forward pass. Peak RAM no longer scales with
the model.

It is a port of the relevant parts of WeeLLM (MIT) --
``io/ggufs/gguf_seek.py`` and ``models/transformers/base_transformer_streamer.py``
-- reduced to what this engine needs:

* a small **reader** protocol (``has`` / ``tensor`` / ``tensor_bytes``) so a
  family can expose its own key layout (a container prefix, a fused SwiGLU key
  split into two parameters, or a safetensors ``fetch`` closure);
* :class:`GgufSource` for the common memory-mapped GGUF case and
  :class:`CallableSource` for loaders that already read one tensor at a time;
* :func:`shard_order`, which derives the streamed block list from the module
  tree so any ``nn.ModuleList`` of blocks works with no per-family subclass;
* :class:`DiskStreamer` -- the pre/post hooks, the adaptive prefetch depth from
  the RAM budget, and greedy VRAM pinning calibrated on the first pass.

Everything degrades safely: if ``accelerate`` (meta init / ``set_module_tensor_
to_device``) is unavailable the caller falls back to the dense loader, and a
block whose checkpoint key is missing is simply left as-is rather than raising.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from . import gguf as gguf_reader

# Reserve for the OS, the interpreter and activations; not usable for staging.
_RAM_SAFETY_BYTES = 2 * 1024 * 1024 * 1024
# Beyond this many blocks read ahead the benefit flattens (matches WeeLLM).
_MAX_PREFETCH_DEPTH = 6
# Headroom kept free while pinning blocks into VRAM.
_VRAM_SAFETY_BYTES = 512 * 1024 * 1024

_SHARD_NAME = "_pai_stream_shard"


def accelerate_available() -> bool:
    try:
        from accelerate.utils import set_module_tensor_to_device  # noqa: F401
        from accelerate import init_empty_weights  # noqa: F401
    except Exception:
        return False
    return True


def can_stream(path, loras=()) -> bool:
    """Whether ``path`` can be disk-streamed.

    LoRA patching needs the block weights present, so a run with LoRAs keeps the
    dense loader; and the loader has to be one of the ones wired for streaming
    (Flux, Z-Image, Qwen-Image, Krea 2, MOSS Image). Callers pass their own
    loader's suitability; this gate covers the two rules that hold everywhere.
    """
    if loras:
        return False
    if not str(path).lower().endswith((".gguf", ".safetensors")):
        return False
    return accelerate_available()


class GgufSource:
    """Memory-mapped access to a GGUF's tensors, keyed by *model parameter* name.

    ``name_map`` turns an on-disk key into one model name (``"stripped.name"``)
    or a list of ``(model_name, slicer)`` pairs for a fused tensor that has to
    feed several parameters (e.g. Qwen's fused ``img_mlp.gate_up.weight``). The
    default is the identity, which is what the Flux and Z-Image GGUFs use.
    """

    def __init__(self, path, name_map: Optional[Callable] = None):
        self.path = str(path)
        self.reader = gguf_reader.gguf.GGUFReader(self.path)
        self._tensors = {t.name: t for t in self.reader.tensors}
        self._by_model: Dict[str, Tuple[str, Optional[Callable]]] = {}
        for source_name in self._tensors:
            mapped = source_name if name_map is None else name_map(source_name)
            if mapped is None:
                continue
            entries = [(mapped, None)] if isinstance(mapped, str) else mapped
            for model_name, slicer in entries:
                self._by_model[model_name] = (source_name, slicer)
        self.names = set(self._by_model)

    def has(self, model_name: str) -> bool:
        return model_name in self._by_model

    def source_for(self, model_name: str) -> Optional[str]:
        """The on-disk key behind ``model_name`` (compatibility shim)."""
        entry = self._by_model.get(model_name)
        return entry[0] if entry else None

    def tensor(self, model_name: str, dtype=torch.bfloat16) -> Optional[torch.Tensor]:
        entry = self._by_model.get(model_name)
        if entry is None:
            return None
        source_name, slicer = entry
        full = gguf_reader.dequantize(self._tensors[source_name], dtype=dtype)
        return slicer(full) if slicer is not None else full

    def get_many(self, model_names: Iterable[str], dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
        out = {}
        for name in model_names:
            value = self.tensor(name, dtype)
            if value is not None:
                out[name] = value
        return out

    # Kept for callers/tests written against the earlier interface.
    def get_tensors(self, model_names, dtype=torch.bfloat16):
        return self.get_many(model_names, dtype)

    def tensor_bytes(self, model_name: str) -> int:
        entry = self._by_model.get(model_name)
        if entry is None:
            return 0
        tensor = self._tensors.get(entry[0])
        data = getattr(tensor, "data", None)
        size = getattr(data, "nbytes", None)
        if size:
            return int(size)
        try:
            block_size, type_size = gguf_reader.gguf.GGML_QUANT_SIZES[tensor.tensor_type]
            count = 1
            for dim in tensor.shape:
                count *= int(dim)
            return int(count / block_size * type_size)
        except Exception:
            return 0

    def block_bytes(self, model_names) -> int:
        return sum(self.tensor_bytes(name) for name in model_names)

    def close(self):
        pass


class CallableSource:
    """A reader over a loader's own one-tensor-at-a-time ``fetch(name)`` closure.

    Used by Krea 2 and MOSS Image, whose loaders already read a single tensor at
    a time (GGUF or memory-mapped safetensors) but still build the whole model in
    RAM. ``names`` is the set of model parameter names that exist; ``bytes_of``
    is an optional size estimate for prefetch budgeting.
    """

    def __init__(self, names: Iterable[str], fetch: Callable, bytes_of: Optional[Callable] = None,
                 close: Optional[Callable] = None):
        self.names = set(names)
        self._fetch = fetch
        self._bytes_of = bytes_of
        self._close = close

    def has(self, model_name: str) -> bool:
        return model_name in self.names

    def tensor(self, model_name: str, dtype=torch.bfloat16) -> Optional[torch.Tensor]:
        tensor = self._fetch(model_name)
        if tensor is None:
            return None
        return tensor.to(dtype)

    def get_many(self, model_names: Iterable[str], dtype=torch.bfloat16) -> Dict[str, torch.Tensor]:
        out = {}
        for name in model_names:
            value = self.tensor(name, dtype)
            if value is not None:
                out[name] = value
        return out

    def tensor_bytes(self, model_name: str) -> int:
        if self._bytes_of is None:
            return 0
        try:
            return int(self._bytes_of(model_name) or 0)
        except Exception:
            return 0

    def block_bytes(self, model_names) -> int:
        return sum(self.tensor_bytes(name) for name in model_names)

    def close(self):
        if self._close is not None:
            try:
                self._close()
            except Exception:
                pass


def shard_order(model) -> List[Tuple[str, nn.Module]]:
    """Return ``[(prefix, block_module), ...]`` for every streamed block.

    A "block" is a child of any ``nn.ModuleList`` in the model tree that is not
    itself a ``ModuleList`` -- the numbered transformer layers every family builds
    (``double_blocks``/``single_blocks``, ``transformer_blocks``, ``blocks``,
    ``noise_refiner``/``layers``...). Deriving the list from the tree keeps this
    family-agnostic.
    """
    order: List[Tuple[str, nn.Module]] = []
    chosen: List[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or len(module) == 0:
            continue
        if any(isinstance(child, nn.ModuleList) for child in module):
            continue
        if any(module_name == parent or module_name.startswith(parent + ".")
               for parent in chosen):
            continue
        chosen.append(module_name)
        for index, child in enumerate(module):
            name = f"{module_name}.{index}" if module_name else str(index)
            order.append((name, child))
    return order


class DiskStreamer:
    """Stream a meta-built transformer's blocks from a checkpoint reader."""

    def __init__(self, model, source, order: List[Tuple[str, nn.Module]],
                 device="cuda", dtype=torch.bfloat16, ram_budget=0,
                 prefetch_depth=0, pin_blocks=True):
        self.model = model
        self.source = source
        self.order = order
        self.device = torch.device(device)
        self.dtype = dtype
        self.pin_blocks = pin_blocks
        self._prefixes = tuple(name for name, _ in order)
        self._pos = {name: index for index, (name, _) in enumerate(order)}

        # The model parameter names of every streamed block, filtered to those the
        # checkpoint actually carries. Fused keys feed several names but the
        # source resolves each name to its own slice.
        self._block_names: Dict[str, List[str]] = {}
        for name, module in order:
            keys = [f"{name}.{param_name}"
                    for param_name, _param in module.named_parameters(recurse=True)]
            self._block_names[name] = [key for key in keys if source.has(key)]

        self._resident_names = [
            name for name in _all_param_names(model)
            if not self._is_streamed(name) and source.has(name)
        ]

        self._prefetch_depth = self._compute_prefetch_depth(ram_budget, prefetch_depth)
        self._disk_executor = ThreadPoolExecutor(
            max_workers=max(1, self._prefetch_depth), thread_name_prefix="pai_stream",
        )
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()
        self._pinned: set = set()
        self._cache_budget = 0
        self._calibrated = False
        self._resident_loaded = False
        self._vram_reserve = _VRAM_SAFETY_BYTES

        self._install_hooks()
        model.register_forward_pre_hook(self._ensure_resident)

    # -- helpers ----------------------------------------------------------
    def _is_streamed(self, model_name: str) -> bool:
        return any(model_name.startswith(prefix + ".") for prefix in self._prefixes)

    def _compute_prefetch_depth(self, ram_budget, requested) -> int:
        if requested and int(requested) > 0:
            return max(1, min(_MAX_PREFETCH_DEPTH, int(requested)))
        try:
            if ram_budget and int(ram_budget) > 0:
                budget = int(ram_budget)
            else:
                import psutil
                budget = psutil.virtual_memory().available
            largest = 1
            for name, _module in self.order:
                largest = max(largest, self.source.block_bytes(self._block_names[name]) or 1)
            usable = max(0, budget - _RAM_SAFETY_BYTES)
            return max(1, min(_MAX_PREFETCH_DEPTH, int(usable // largest) or 1))
        except Exception:
            return 2

    # -- residency --------------------------------------------------------
    def _ensure_resident(self, module, args):
        self.load_resident()
        return args

    def load_resident(self):
        """Materialise the non-block weights and every real buffer on the device."""
        if self._resident_loaded:
            return
        from accelerate.utils.modeling import set_module_tensor_to_device

        state = self.source.get_many(self._resident_names, self.dtype)
        for name, tensor in state.items():
            set_module_tensor_to_device(self.model, name, self.device,
                                        value=tensor, dtype=self.dtype)
        for buffer_name, buffer in self.model.named_buffers():
            if buffer is not None and buffer.device.type != "meta":
                set_module_tensor_to_device(self.model, buffer_name, self.device, value=buffer)
        self._resident_loaded = True

    def release_resident(self):
        """Evict the resident weights while another pipeline component runs."""
        if not self._resident_loaded:
            return 0
        from accelerate.utils.modeling import set_module_tensor_to_device

        released = 0
        for name in self._resident_names:
            try:
                set_module_tensor_to_device(self.model, name, "meta")
                released += 1
            except Exception:
                pass
        self._resident_loaded = False
        _empty_cache()
        return released

    # -- hooks ------------------------------------------------------------
    def _install_hooks(self):
        for name, module in self.order:
            setattr(module, _SHARD_NAME, name)
            module.register_forward_pre_hook(self._pre_hook)
            module.register_forward_hook(self._post_hook)

    def _materialize(self, name: str) -> None:
        from accelerate.utils.modeling import set_module_tensor_to_device

        state = self._take(name)
        if state is None:
            state = self.source.get_many(self._block_names[name], self.dtype)
        for full, tensor in state.items():
            set_module_tensor_to_device(self.model, full, self.device,
                                        value=tensor, dtype=self.dtype)

    def _take(self, name: str) -> Optional[Dict[str, torch.Tensor]]:
        with self._lock:
            future = self._futures.pop(name, None)
        if future is None:
            return None
        try:
            return future.result()
        except Exception:
            return None

    def _prefetch(self, position: int) -> None:
        if position >= len(self.order):
            return
        name = self.order[position][0]
        if name in self._pinned:
            return
        with self._lock:
            if name in self._futures:
                return
            names = list(self._block_names[name])
            self._futures[name] = self._disk_executor.submit(
                self.source.get_many, names, self.dtype
            )

    def _pre_hook(self, module, args):
        name = getattr(module, _SHARD_NAME)
        if name in self._pinned:
            return args
        self._materialize(name)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.current_stream().synchronize()
        position = self._pos[name]
        for ahead in range(1, self._prefetch_depth + 1):
            self._prefetch(position + ahead)
        return args

    def _post_hook(self, module, args, output):
        name = getattr(module, _SHARD_NAME)
        if name in self._pinned:
            return output
        if self._maybe_pin(name, module):
            return output
        self._evict(module)
        return output

    def _maybe_pin(self, name, module) -> bool:
        if not self.pin_blocks or not torch.cuda.is_available():
            return False
        if name in self._pinned:
            return True
        if not self._calibrated:
            self._calibrate()
        if self._cache_budget <= 0:
            return False
        size = _module_bytes(module)
        if size <= 0 or size > self._cache_budget:
            return False
        self._cache_budget -= size
        self._pinned.add(name)
        return True

    def _calibrate(self):
        """Work out the spare VRAM left after one full pass, for pinning blocks."""
        if self._calibrated:
            return
        self._calibrated = True
        try:
            peak = torch.cuda.max_memory_reserved(self.device)
            _free, total = torch.cuda.mem_get_info(self.device)
            self._cache_budget = max(0, total - peak - self._vram_reserve)
        except Exception:
            self._cache_budget = 0

    def _evict(self, module) -> None:
        from accelerate.utils.modeling import set_module_tensor_to_device

        for param_name, param in list(module.named_parameters(recurse=True)):
            if param.device.type != "meta":
                try:
                    set_module_tensor_to_device(module, param_name, "meta")
                except Exception:
                    pass

    def release_cached_blocks(self) -> int:
        """Drop every pinned block and future (used after a CUDA OOM; retries)."""
        with self._lock:
            futures = list(self._futures.values())
            self._futures.clear()
            self._pinned.clear()
            self._cache_budget = 0
        for future in futures:
            future.cancel()
        released = 0
        for _name, module in self.order:
            released += _evict_module(module)
        _empty_cache()
        return released

    def close(self):
        try:
            self._disk_executor.shutdown(wait=False)
        except Exception:
            pass
        try:
            self.source.close()
        except Exception:
            pass

    def __del__(self):
        self.close()


def _all_param_names(model) -> List[str]:
    return [name for name, _param in model.named_parameters()]


def _module_bytes(module) -> int:
    return sum(p.numel() * p.element_size()
               for p in module.parameters() if p.device.type != "meta")


def _evict_module(module) -> int:
    from accelerate.utils.modeling import set_module_tensor_to_device

    count = 0
    for name, param in module.named_parameters(recurse=True):
        if param.device.type == "meta":
            continue
        try:
            set_module_tensor_to_device(module, name, "meta")
            count += 1
        except Exception:
            pass
    return count


def _empty_cache():
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def build_streaming_model(factory, source, device="cuda", dtype=torch.bfloat16,
                          ram_budget=0, prefetch_depth=0, pin_blocks=True):
    """Build ``factory`` on meta and attach a :class:`DiskStreamer`.

    Returns ``(model, streamer)``. The caller must keep ``streamer`` (and the
    model) alive for the lifetime of the model; the streamer owns the prefetch
    executor and the (memory-mapped) source.
    """
    from .precision import build_on_meta

    with build_on_meta(dtype, factory):
        model = factory()
    model.eval()
    order = shard_order(model)
    streamer = DiskStreamer(model, source, order, device=device, dtype=dtype,
                            ram_budget=ram_budget, prefetch_depth=prefetch_depth,
                            pin_blocks=pin_blocks)
    # Arm place_dit/loaded-models reporting and keep the executor from being GC'd.
    model._streaming_source = source
    model._streaming_object = streamer
    model._streaming_label = f"disk-streamed({len(order)} blocks)"
    for param in model.parameters():
        param.requires_grad_(False)
    return model, streamer