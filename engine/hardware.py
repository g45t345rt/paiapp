"""Detect the GPU and memory a model recommendation should be based on.

The download wizard uses this to pick the largest quantization a machine can
actually run. It is deliberately independent of the UI: only ``torch`` (already
the engine's runtime) is consulted, and it is imported lazily inside
:func:`probe`, so importing this module -- and showing the wizard before a model
is ever loaded -- stays cheap.

The UI layers a DXGI/vendor-CLI fallback on top (see ``paintapp.monitor``) for a
machine where torch is not installed or reports no CUDA device. When nothing can
be detected the fields stay ``None`` and the wizard shows balanced defaults
instead of failing.
"""

import ctypes
import os


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _system_ram_gb():
    """Total RAM without psutil: Windows API or ``sysconf`` on POSIX."""
    if os.name == "nt":
        try:
            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(_MemoryStatusEx)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return status.ullTotalPhys / 1e9
        except Exception:
            return None
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return pages * size / 1e9
    except (ValueError, OSError, AttributeError):
        return None


def _ram_gb():
    """Total system RAM in GB, or ``None`` when it cannot be read."""
    try:
        import psutil
    except Exception:
        return _system_ram_gb()
    try:
        return psutil.virtual_memory().total / 1e9
    except Exception:
        return _system_ram_gb()


def _torch_device():
    """``(name, vram_gb, backend)`` for the largest CUDA/ROCm device, or None."""
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        best = None
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total = float(props.total_memory)
            if best is None or total > best[1]:
                best = (props.name, total / 1e9)
        if best is None:
            return None
        backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        return best[0], best[1], backend
    except Exception:
        return None


def probe():
    """A hardware snapshot for model recommendations.

    Returns ``{"gpu", "gpu_name", "vram_gb", "ram_gb", "backend", "source"}``.
    Every value is best-effort: a field is ``None`` (or ``""``) when it could not
    be read, and the caller must treat that as "unknown", not as zero.
    """
    result = {
        "gpu": False,
        "gpu_name": "",
        "vram_gb": None,
        "ram_gb": _ram_gb(),
        "backend": "none",
        "source": "",
    }
    device = _torch_device()
    if device is not None:
        name, vram, backend = device
        result.update({
            "gpu": True, "gpu_name": name, "vram_gb": vram,
            "backend": backend, "source": "torch",
        })
    return result
