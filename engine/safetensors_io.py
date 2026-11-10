"""A minimal safetensors reader that copies tensor bytes from the file.

Why this exists
---------------
``safetensors.safe_open`` hands out tensors backed by an internal buffer it owns
in Rust. Under memory pressure that buffer is sometimes released before the
tensor is read, so the read faults with a Windows access violation
(``0xC0000005``) -- reproduced with the MiniMax H3 workflow, where a preloaded
~21 GB DiT is resident while the streamed Qwen3-VL text encoder reads its own
27 GB file (``engine.h3_text``). Reading the same bytes through Python's
``torch.frombuffer`` over a plain file read is 100% reliable, so this reader
``seek``/``read``s exactly the requested byte range into owned memory and builds
the tensor from that, sharing no lifetime with any loader.

It is a drop-in for the subset of the ``safe_open`` interface the engine's
loaders use (``keys`` / ``get_tensor`` / ``get_slice`` with ``get_shape`` /
``get_dtype`` / row slicing / ``metadata``), so it can be passed to
:func:`engine.quant_formats._copy_weight` and :func:`engine.quant_formats._dequant`.
"""

import json
import struct

import torch

_DTYPES = {
    "BOOL": torch.bool,
    "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "U16": torch.uint16,
    "I32": torch.int32, "U32": torch.uint32,
    "I64": torch.int64, "U64": torch.uint64,
    "F16": torch.float16, "BF16": torch.bfloat16,
    "F32": torch.float32, "F64": torch.float64,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E5M2": torch.float8_e5m2,
}


class _Slice:
    """Row-sliceable view description of one tensor (no data held)."""

    def __init__(self, reader, info):
        self._reader = reader
        self._info = info
        self._shape = list(info["shape"])
        self._dtype = info["dtype"]
        start, stop = info["data_offsets"]
        self._start = reader._data_start + int(start)
        self._nbytes = int(stop) - int(start)

    def get_shape(self):
        return list(self._shape)

    def get_dtype(self):
        return self._dtype

    def __getitem__(self, item):
        rows = int(self._shape[0]) if self._shape else 1
        per_row = self._nbytes // max(1, rows)
        if isinstance(item, slice):
            start, stop, step = item.indices(rows)
            if step != 1:
                full = self._reader._read(self._start, self._nbytes, self._dtype, self._shape)
                return full[item]
            tail = self._shape[1:]
        elif isinstance(item, int):
            start = item + rows if item < 0 else item
            stop = start + 1
            tail = self._shape[1:]
        else:
            raise TypeError(f"unsupported index {item!r}")
        data = self._reader._read(self._start + start * per_row,
                                  (stop - start) * per_row, self._dtype, None)
        return data.reshape([stop - start] + tail)


class SafetensorsFile:
    """Read-only safetensors accessor over owned file reads (see module docstring)."""

    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, "rb")
        (header_len,) = struct.unpack("<Q", self._f.read(8))
        self._header = json.loads(self._f.read(header_len).decode("utf-8"))
        self._data_start = 8 + header_len
        self._meta = self._header.get("__metadata__")
        self._names = [key for key in self._header if key != "__metadata__"]

    def _read(self, offset, nbytes, dtype_name, shape):
        self._f.seek(offset)
        buf = self._f.read(nbytes)
        if len(buf) != nbytes:
            raise ValueError(f"short read ({len(buf)}/{nbytes}) in {self.path}")
        tensor = torch.frombuffer(bytearray(buf), dtype=_DTYPES[dtype_name])
        return tensor.reshape(shape) if shape is not None else tensor

    def keys(self):
        return list(self._names)

    def __contains__(self, name):
        return name in self._header and name != "__metadata__"

    def metadata(self):
        return self._meta

    def get_tensor(self, name):
        info = self._header[name]
        start, stop = info["data_offsets"]
        return self._read(self._data_start + int(start), int(stop) - int(start),
                          info["dtype"], info["shape"])

    def get_slice(self, name):
        return _Slice(self, self._header[name])

    def close(self):
        try:
            self._f.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def open_safetensors(path):
    """Open ``path`` with :class:`SafetensorsFile`."""
    return SafetensorsFile(path)
