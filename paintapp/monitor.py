"""Background CPU/GPU/RAM usage sampling for the status bar.

CPU usage (percent) and RAM usage (used/total GB) come from :mod:`psutil` when
it is installed. GPU memory (used/total GB) is read through the first source
that answers:

* Windows: the DXGI adapter list gives every GPU's dedicated VRAM and the PDH
  ``GPU Adapter Memory`` counters give its usage, so NVIDIA, AMD and Intel
  cards all report without any vendor tool.
* Elsewhere: ``nvidia-smi`` (NVIDIA), then ``rocm-smi``/``amd-smi`` (AMD).

All sources are optional; a machine with none of them reports ``None`` and the
status bar shows a placeholder.
"""

import ctypes
import json
import os
import shutil
import subprocess
import threading

try:
    import psutil
except ImportError:  # pragma: no cover - psutil is normally present
    psutil = None

_GIB = 1024 ** 3

# DXGI: the IID of IDXGIFactory1, and the software-adapter flag inside
# DXGI_ADAPTER_DESC1.Flags (basic render drivers must not be counted).
_DXGI_FACTORY1_IID = "{770AAE78-F26F-4DBA-A829-253C83D1B387}"
_DXGI_ADAPTER_FLAG_SOFTWARE = 0x2

# PDH: the counter whose instances carry each adapter's dedicated VRAM usage,
# formatted as a double.
_PDH_FMT_DOUBLE = 0x00000200
_PDH_MORE_DATA = 0x800007D2
_GPU_MEMORY_COUNTER = r"\GPU Adapter Memory(*)\Dedicated Usage"


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", ctypes.c_ulong), ("Data2", ctypes.c_ushort),
                ("Data3", ctypes.c_ushort), ("Data4", ctypes.c_ubyte * 8)]


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_ulong), ("HighPart", ctypes.c_long)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [
        ("Description", ctypes.c_wchar * 128),
        ("VendorId", ctypes.c_uint), ("DeviceId", ctypes.c_uint),
        ("SubSysId", ctypes.c_uint), ("Revision", ctypes.c_uint),
        ("DedicatedVideoMemory", ctypes.c_size_t),
        ("DedicatedSystemMemory", ctypes.c_size_t),
        ("SharedSystemMemory", ctypes.c_size_t),
        ("AdapterLuid", _LUID), ("Flags", ctypes.c_uint),
    ]


class _PDH_FMT_VALUE(ctypes.Union):
    _fields_ = [("longValue", ctypes.c_long), ("doubleValue", ctypes.c_double),
                ("largeValue", ctypes.c_longlong),
                ("AnsiStringValue", ctypes.c_char_p),
                ("WideStringValue", ctypes.c_wchar_p)]


class _PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", ctypes.c_ulong), ("u", _PDH_FMT_VALUE)]


class _PDH_FMT_COUNTERVALUE_ITEM_W(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p),
                ("FmtValue", _PDH_FMT_COUNTERVALUE)]


def _probes():
    """The GPU memory sources to try, in order, for this platform."""
    if os.name == "nt":
        return (_windows_gpu_memory, _nvidia_smi_memory)
    return (_nvidia_smi_memory, _rocm_smi_memory, _amd_smi_memory)


def gpu_memory():
    """Return GPU memory ``(used_gb, total_gb)``, or ``None``."""
    for probe in _probes():
        try:
            result = probe()
        except Exception:
            result = None
        if result:
            return result
    return None


def _run_command(cmd):
    """Run a helper tool and return its stdout, or ``None`` on any failure."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


# --------------------------------------------------------------- Windows (native)

def _com_method(instance, index, restype, argtypes):
    """Bind COM vtable slot ``index`` of ``instance`` to a callable."""
    vtable = ctypes.cast(instance, ctypes.POINTER(ctypes.c_void_p))[0]
    function = ctypes.cast(vtable, ctypes.POINTER(ctypes.c_void_p))[index]
    return ctypes.WINFUNCTYPE(restype, *argtypes)(function)


def _dxgi_adapters():
    """Hardware adapters as ``{luid: (name, dedicated_bytes)}``, or ``{}``."""
    try:
        dxgi = ctypes.WinDLL("dxgi.dll")
    except OSError:
        return {}
    iid = _GUID()
    ctypes.oledll.ole32.CLSIDFromString(_DXGI_FACTORY1_IID, ctypes.byref(iid))
    dxgi.CreateDXGIFactory1.restype = ctypes.c_long
    dxgi.CreateDXGIFactory1.argtypes = [ctypes.POINTER(_GUID),
                                        ctypes.POINTER(ctypes.c_void_p)]
    factory = ctypes.c_void_p()
    if dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) & 0xffffffff:
        return {}
    # IDXGIFactory1 slot 12 is EnumAdapters1; IDXGIAdapter1 slot 10 is GetDesc1.
    enum = _com_method(factory, 12, ctypes.c_long,
                       [ctypes.c_void_p, ctypes.c_uint,
                        ctypes.POINTER(ctypes.c_void_p)])
    get_desc = None
    adapters = {}
    index = 0
    while True:
        adapter = ctypes.c_void_p()
        if enum(factory, index, ctypes.byref(adapter)) & 0xffffffff:
            break
        if get_desc is None:
            get_desc = _com_method(adapter, 10, ctypes.c_long,
                                   [ctypes.c_void_p,
                                    ctypes.POINTER(_DXGI_ADAPTER_DESC1)])
        desc = _DXGI_ADAPTER_DESC1()
        if not get_desc(adapter, ctypes.byref(desc)) & 0xffffffff:
            if not desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE:
                luid = (desc.AdapterLuid.HighPart & 0xffffffff,
                        desc.AdapterLuid.LowPart & 0xffffffff)
                adapters[luid] = (desc.Description,
                                  int(desc.DedicatedVideoMemory))
        index += 1
    return adapters


def _pdh_instance_luid(name):
    """Parse ``luid_0xHIGH_0xLOW_...`` into a ``(high, low)`` key, or ``None``."""
    if not name or "luid_" not in name:
        return None
    parts = name.split("luid_", 1)[1].split("_")
    if len(parts) < 2:
        return None
    try:
        return int(parts[0], 16) & 0xffffffff, int(parts[1], 16) & 0xffffffff
    except ValueError:
        return None


def _pdh_dedicated_usage(adapters):
    """Sum dedicated VRAM usage (bytes) over ``adapters``, or ``None``."""
    try:
        pdh = ctypes.WinDLL("pdh.dll")
    except OSError:
        return None
    pdh.PdhOpenQueryW.restype = ctypes.c_ulong
    pdh.PdhOpenQueryW.argtypes = [ctypes.c_wchar_p, ctypes.c_size_t,
                                  ctypes.POINTER(ctypes.c_void_p)]
    pdh.PdhAddEnglishCounterW.restype = ctypes.c_ulong
    pdh.PdhAddEnglishCounterW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p,
                                          ctypes.c_size_t,
                                          ctypes.POINTER(ctypes.c_void_p)]
    pdh.PdhCollectQueryData.restype = ctypes.c_ulong
    pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
    pdh.PdhGetFormattedCounterArrayW.restype = ctypes.c_ulong
    pdh.PdhGetFormattedCounterArrayW.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
    pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]

    query = ctypes.c_void_p()
    if pdh.PdhOpenQueryW(None, 0, ctypes.byref(query)) & 0xffffffff:
        return None
    try:
        counter = ctypes.c_void_p()
        if pdh.PdhAddEnglishCounterW(query, _GPU_MEMORY_COUNTER, 0,
                                     ctypes.byref(counter)) & 0xffffffff:
            return None
        if pdh.PdhCollectQueryData(query) & 0xffffffff:
            return None
        size = ctypes.c_ulong(0)
        count = ctypes.c_ulong(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count),
            None) & 0xffffffff
        if status not in (0, _PDH_MORE_DATA):
            return None
        buffer = ctypes.create_string_buffer(size.value)
        if pdh.PdhGetFormattedCounterArrayW(
                counter, _PDH_FMT_DOUBLE, ctypes.byref(size),
                ctypes.byref(count), buffer) & 0xffffffff:
            return None
        items = ctypes.cast(buffer, ctypes.POINTER(_PDH_FMT_COUNTERVALUE_ITEM_W))
        used = 0.0
        for i in range(count.value):
            if _pdh_instance_luid(items[i].szName) in adapters:
                used += items[i].FmtValue.u.doubleValue
        return used
    finally:
        pdh.PdhCloseQuery(query)


def _windows_gpu_memory():
    """Dedicated VRAM used/total across hardware adapters, via DXGI and PDH."""
    adapters = _dxgi_adapters()
    if not adapters:
        return None
    total = sum(dedicated for _, dedicated in adapters.values())
    if not total:
        return None
    used = _pdh_dedicated_usage(adapters)
    if used is None:
        return None
    return used / _GIB, total / _GIB


# --------------------------------------------------------- vendor CLIs (POSIX)

def _nvidia_smi_memory():
    """NVIDIA GPU memory (used/total GB) from ``nvidia-smi``, or ``None``."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    out = _run_command([exe, "--query-gpu=memory.used,memory.total",
                        "--format=csv,noheader,nounits"])
    if not out:
        return None
    used = total = 0.0
    for line in out.splitlines():
        if not line.strip():
            continue
        try:
            used_mb, total_mb = (float(part) for part in line.split(","))
        except ValueError:
            return None
        used += used_mb
        total += total_mb
    if not total:
        return None
    return used / 1024, total / 1024


def _number(value):
    """Coerce a JSON number, numeric string, or ``{"value": ...}`` to float."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return _number(value.get("value"))
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _first_matching(mapping, needle):
    """The first value under a key containing ``needle``, as a float."""
    for key, value in mapping.items():
        if isinstance(key, str) and needle in key:
            number = _number(value)
            if number is not None:
                return number
    return None


def _rocm_smi_memory():
    """AMD GPU memory (used/total GB) from ``rocm-smi``, or ``None``."""
    exe = shutil.which("rocm-smi")
    if not exe:
        return None
    out = _run_command([exe, "--showmeminfo", "vram", "--json"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None
    used = total = 0.0
    found = False
    for entry in data.values():
        if not isinstance(entry, dict):
            continue
        used_b = _first_matching(entry, "Used Memory (B)")
        total_b = _first_matching(entry, "Total Memory (B)")
        if used_b is None or total_b is None:
            continue
        found = True
        used += used_b
        total += total_b
    if not found or not total:
        return None
    return used / _GIB, total / _GIB


def _vram_entries(node):
    """Yield every dict under ``node`` that holds ``used_vram``/``total_vram``."""
    if isinstance(node, dict):
        if "used_vram" in node and "total_vram" in node:
            yield node
        for value in node.values():
            yield from _vram_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from _vram_entries(value)


def _vram_gib(value):
    """Convert an ``amd-smi`` vram reading to GiB, honouring its ``unit``."""
    number = _number(value)
    if number is None:
        return None
    unit = (value.get("unit") if isinstance(value, dict) else None) or "MB"
    unit = str(unit).upper()
    if unit in ("B", "BYTE", "BYTES"):
        return number / _GIB
    if unit in ("KB", "KIB"):
        return number / 1024 ** 2
    if unit in ("MB", "MIB"):
        return number / 1024
    if unit in ("GB", "GIB"):
        return number
    return number / 1024


def _amd_smi_memory():
    """AMD GPU memory (used/total GB) from ``amd-smi``, or ``None``."""
    exe = shutil.which("amd-smi")
    if not exe:
        return None
    out = _run_command([exe, "metric", "--vram", "--json"])
    if not out:
        return None
    try:
        data = json.loads(out)
    except ValueError:
        return None
    used = total = 0.0
    found = False
    for entry in _vram_entries(data):
        used_gib = _vram_gib(entry.get("used_vram"))
        total_gib = _vram_gib(entry.get("total_vram"))
        if used_gib is None or total_gib is None:
            continue
        found = True
        used += used_gib
        total += total_gib
    if not found or not total:
        return None
    return used, total


def ram_usage():
    """Return system memory ``(used_gb, total_gb)``, or ``None``."""
    if psutil is None:
        return None
    try:
        mem = psutil.virtual_memory()
    except Exception:
        return None
    gib = 1024 ** 3
    return mem.used / gib, mem.total / gib


class SystemMonitor:
    """Poll CPU/GPU/RAM usage on a background thread and report each sample."""

    def __init__(self, on_update, interval=2.0):
        self.on_update = on_update
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        if psutil is None:
            self.on_update(None, None, None)
        else:
            psutil.cpu_percent(interval=None)
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.wait(self.interval):
            cpu = psutil.cpu_percent(interval=None)
            self.on_update(cpu, gpu_memory(), ram_usage())
