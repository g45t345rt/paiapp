"""First-run bootstrap for Pai.

Creates a project-local ``.venv`` from a base Python 3.12+ (3.12-3.14), installs
the pinned dependencies on the first run (and again whenever a requirements file
changes), then launches ``paint.py`` with the venv's interpreter.

This replaces the old ``run.bat`` search for a ComfyUI virtualenv: Pai now owns
its own environment and never imports anything from another install.

Usage:
    run.bat [--core-only] [--reinstall] [--setup-only] [--rocm] [--] [app args]

Environment:
    PAI_TORCH_INDEX   pip index for the NVIDIA/CPU torch wheels. Defaults to the
                      CUDA 13.0 build used during development; set to ``cpu``
                      for a CPU-only install or to a full URL for another CUDA
                      build. Set to ``none`` to take torch straight from PyPI.
    PAI_BACKEND       force the compute backend: ``cuda``, ``cpu`` or ``rocm``.
                      ``--rocm`` is shorthand for ``PAI_BACKEND=rocm``. When
                      unset the backend is auto-detected from the installed GPU
                      (NVIDIA -> cuda, AMD -> rocm, macOS/no GPU -> cpu, falling
                      back to cuda for an unknown GPU).
    PAI_ROCM_INDEX    pip index for the AMD ROCm torch wheels. Defaults to AMD's
                      stable multi-arch index (Windows and Linux wheels).
    PAI_ROCM_GFX      the AMD GPU target for the ROCm wheel extra, e.g.
                      ``gfx1100``; auto-detected when unset from rocminfo, the
                      Linux KFD topology, the OpenCL runtime (which exists with
                      just the graphics driver), ``rocm-bootstrap-detect`` or
                      ``clinfo``. When the target cannot be detected, an
                      auto-detected AMD GPU falls back to CPU; set this (or pass
                      ``--rocm``) to install the ``all`` kernels instead.
    PAI_VENV          where the private environment lives. Defaults to ``.venv``
                      beside the app; a system package sets it to a user-writable
                      folder because its install tree (e.g. ``/usr/lib/pai``) is
                      read-only.
"""

import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
# A launcher can move the environment out of a read-only install directory (a
# system package under /usr/lib/pai) by setting PAI_VENV; by default it stays
# in .venv beside the app, so source checkouts are unchanged.
_venv_override = os.environ.get("PAI_VENV")
VENV = (
    os.path.abspath(os.path.expanduser(_venv_override))
    if _venv_override
    else os.path.join(ROOT, ".venv")
)
MARKER = os.path.join(VENV, ".pai-setup.json")
CORE_REQ = os.path.join(ROOT, "requirements.txt")
AI_REQ = os.path.join(ROOT, "requirements-ai.txt")
ROCM_REQ = os.path.join(ROOT, "requirements-rocm.txt")
APP = os.path.join(ROOT, "paint.py")

DEFAULT_TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
# AMD's stable multi-arch index carries torch/torchvision for both Windows and
# Linux; the ROCm runtime is pulled in automatically by the torch wheel.
ROCM_TORCH_INDEX = "https://stable.repo.amd.com/rocm/whl-next/"
PYPI_INDEX = "https://pypi.org/simple"
# The GFX targets the AMD index carries a ``device-gfxNNNN`` extra for (see the
# wheel's Provides-Extra metadata). A detected target outside this set has no
# narrow wheel, so the ``all`` extra is used instead. Kept in sync with the
# index listing at ROCM_TORCH_INDEX.
SUPPORTED_GFX = frozenset({
    "gfx908", "gfx90a", "gfx942", "gfx950",
    "gfx1010", "gfx1011", "gfx1012",
    "gfx1030", "gfx1031", "gfx1032", "gfx1033", "gfx1034", "gfx1035",
    "gfx1036",
    "gfx1100", "gfx1101", "gfx1102", "gfx1103",
    "gfx1150", "gfx1151", "gfx1152", "gfx1153",
    "gfx1200", "gfx1201", "gfx1250",
})
# The torch version every backend shares. 2.13.0 has wheels on the CUDA 13.0
# index (Windows + Linux) and on AMD's ROCm index, so one pin works for NVIDIA
# and AMD installs.
TORCH_VERSION = "2.13.0"
TORCHVISION_VERSION = "0.28.0"
# The interpreter range Pai is tested against. The lower bound is inclusive and
# the upper bound is exclusive: torch/torchvision publish wheels for 3.12-3.14
# only, so a newer interpreter (e.g. 3.15) must be rejected up front with a
# clear message instead of failing later inside pip. Keep in sync with README.md
# and the version gate in the launchers (run.bat, run.sh, installer/*).
MIN_PYTHON = (3, 12)
MAX_PYTHON = (3, 15)

KNOWN_FLAGS = ("--core-only", "--reinstall", "--setup-only", "--rocm")


def _supported_python(info):
    """True when a version tuple is in the tested ``[MIN_PYTHON, MAX_PYTHON)``.

    Only the major/minor pair matters, so ``info`` may be a ``sys.version_info``
    or a plain ``(major, minor)`` tuple.
    """
    return MIN_PYTHON <= (info[0], info[1]) < MAX_PYTHON


def venv_python():
    """Path to the interpreter inside the project venv."""
    if os.name == "nt":
        return os.path.join(VENV, "Scripts", "python.exe")
    return os.path.join(VENV, "bin", "python")


def _requirements_hash(include_ai, backend="cuda"):
    digest = hashlib.sha256()
    digest.update(b"ai" if include_ai else b"core")
    digest.update(backend.encode())
    paths = [CORE_REQ, AI_REQ]
    if backend == "rocm":
        # The ROCm torch pin lives in this module rather than a requirements
        # file, so fold the version and the detected GPU target into the hash to
        # make a version bump or a GPU change re-run the install.
        digest.update(TORCH_VERSION.encode())
        digest.update(_rocm_gfx().encode())
        paths.append(ROCM_REQ)
    else:
        # The CUDA/CPU wheel index is part of the install: changing it (e.g.
        # cu130 -> cpu) must re-run the install.
        digest.update(str(_torch_index()).encode())
    for path in paths:
        try:
            with open(path, "rb") as handle:
                digest.update(handle.read())
        except OSError:
            pass
    return digest.hexdigest()


def _read_marker():
    try:
        with open(MARKER, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_marker(payload):
    try:
        with open(MARKER, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
    except OSError:
        pass


def _run(cmd):
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


def _parse(argv):
    """Split our own flags from arguments meant for the app."""
    core_only = "--core-only" in argv
    reinstall = "--reinstall" in argv
    setup_only = "--setup-only" in argv
    rocm = "--rocm" in argv
    passthrough = []
    after = False
    for arg in argv:
        if after:
            passthrough.append(arg)
        elif arg == "--":
            after = True
        elif arg not in KNOWN_FLAGS:
            passthrough.append(arg)
    return core_only, reinstall, setup_only, rocm, passthrough


def _backend(rocm_flag):
    """The compute backend to install for: ``cuda``, ``cpu`` or ``rocm``.

    ``--rocm`` and an explicit ``PAI_BACKEND`` win; otherwise the backend is
    auto-detected from the GPU (see :func:`_auto_backend`), falling back to
    CUDA -- whose wheels still run on a CPU -- when no GPU is recognised. An
    auto-detected AMD GPU falls back to CPU when its target is unsupported or
    cannot be detected (see :func:`_rocm_supported`); only an explicit ROCm
    request installs the ``all`` kernels for an undetected target.
    """
    value = os.environ.get("PAI_BACKEND", "").strip().lower()
    if value in ("cuda", "cpu"):
        return value
    explicit_rocm = rocm_flag or value == "rocm"
    detected = None if explicit_rocm else _auto_backend()
    if explicit_rocm or detected == "rocm":
        if _rocm_supported():
            return "rocm"
        _, unsupported = _gfx_detection()
        if unsupported:
            print("AMD GPU target(s) {0} are not supported by the current ROCm "
                  "wheels; falling back to the CPU backend. Set PAI_ROCM_GFX to "
                  "force a target.".format(", ".join(unsupported)))
            return "cpu"
        if explicit_rocm:
            print("Could not detect the AMD GPU target; installing kernels for "
                  "all supported GPUs. Set PAI_ROCM_GFX (e.g. "
                  "PAI_ROCM_GFX=gfx1100) to narrow this.")
            return "rocm"
        print("Could not detect the AMD GPU target; falling back to the CPU "
              "backend. Set PAI_ROCM_GFX (e.g. PAI_ROCM_GFX=gfx1100) to use "
              "ROCm.")
        return "cpu"
    return detected or "cuda"


def _vendors_in_text(text):
    """Map a driver description to the GPU vendors it names."""
    text = (text or "").lower()
    vendors = set()
    if "nvidia" in text:
        vendors.add("nvidia")
    if "amd" in text or "advanced micro devices" in text or "radeon" in text:
        vendors.add("amd")
    if "intel" in text:
        vendors.add("intel")
    return vendors


def _sysfs_vendors():
    """GPU vendors from Linux's ``/sys/class/drm`` PCI vendor IDs."""
    vendors = set()
    base = "/sys/class/drm"
    try:
        names = os.listdir(base)
    except OSError:
        return vendors
    for name in names:
        # Connector entries look like ``card0-DP-1``; only the card nodes carry
        # a ``device/vendor`` PCI id.
        if not name.startswith("card") or "-" in name:
            continue
        try:
            with open(os.path.join(base, name, "device", "vendor"),
                      encoding="ascii") as handle:
                vendor_id = handle.read().strip().lower()
        except OSError:
            continue
        if vendor_id == "0x10de":
            vendors.add("nvidia")
        elif vendor_id == "0x1002":
            vendors.add("amd")
        elif vendor_id == "0x8086":
            vendors.add("intel")
    return vendors


def _windows_registry_vendors():
    """GPU vendors from the Windows display-class registry keys."""
    if os.name != "nt":
        return set()
    try:
        import winreg
    except ImportError:  # pragma: no cover - winreg exists on Windows
        return set()
    vendors = set()
    class_key = (r"SYSTEM\CurrentControlSet\Control\Class"
                 r"\{4d36e968-e325-11ce-bfc1-08002be10318}")
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, class_key)
    except OSError:
        return vendors
    try:
        index = 0
        while True:
            try:
                name = winreg.EnumKey(root, index)
            except OSError:
                break
            index += 1
            try:
                sub = winreg.OpenKey(root, name)
            except OSError:
                continue
            try:
                for value in ("ProviderName", "DriverDesc"):
                    try:
                        text = winreg.QueryValueEx(sub, value)[0]
                    except OSError:
                        continue
                    vendors |= _vendors_in_text(str(text))
            finally:
                winreg.CloseKey(sub)
    finally:
        winreg.CloseKey(root)
    return vendors


def _detected_vendors():
    """The GPU vendors present, from OS-level probes (torch is not installed).

    Vendor CLIs are the strongest signal and exist on Windows and Linux; the
    sysfs and registry scans catch cards whose tools are not on PATH.
    """
    vendors = set()
    if shutil.which("nvidia-smi"):
        vendors.add("nvidia")
    if shutil.which("rocm-smi") or shutil.which("amd-smi"):
        vendors.add("amd")
    if os.path.exists("/dev/kfd"):  # the ROCm kernel driver
        vendors.add("amd")
    vendors |= _sysfs_vendors()
    vendors |= _windows_registry_vendors()
    return vendors


def _auto_backend():
    """Pick the backend from the detected hardware, or ``None`` when unknown.

    macOS has no CUDA or ROCm, so it is always CPU. When both vendors are
    present (an AMD integrated GPU beside an NVIDIA card) the CUDA path wins,
    matching the previous default.
    """
    if sys.platform == "darwin":
        return "cpu"
    vendors = _detected_vendors()
    if "nvidia" in vendors:
        return "cuda"
    if "amd" in vendors:
        return "rocm"
    return None


def _rocm_index():
    """The pip index that serves the AMD ROCm torch wheels."""
    return os.environ.get("PAI_ROCM_INDEX", ROCM_TORCH_INDEX).strip() or ROCM_TORCH_INDEX


_gfx_cache = None

# Matches a GFX target token, e.g. ``gfx1100`` or ``gfx90a``, anywhere in text.
_GFX_TOKEN = re.compile(r"gfx[0-9][0-9a-z]*", re.IGNORECASE)
# The Linux KFD topology directory; a module constant so tests can redirect it.
KFD_TOPOLOGY = "/sys/class/kfd/kfd/topology/nodes"


def _gfx_tokens(text):
    """Yield lowercased ``gfxNNNN`` tokens found in ``text``."""
    for match in _GFX_TOKEN.findall(text or ""):
        yield match.lower()


def _gfx_from_version(version):
    """Decode a KFD ``gfx_target_version`` int into a target name.

    The sysfs encoding is ``major * 10000 + minor * 100 + stepping`` and the
    stepping is the last name character (``gfx90a`` has stepping 10 -> ``a``).
    """
    if not version:
        return None
    major, minor, stepping = version // 10000, (version % 10000) // 100, version % 100
    digit = "0123456789abcdef"[stepping] if stepping < 16 else str(stepping)
    return "gfx{0}{1}{2}".format(major, minor, digit)


def _gfx_from_kfd():
    """AMD targets from Linux's KFD topology (needs only the amdgpu driver)."""
    try:
        entries = os.listdir(KFD_TOPOLOGY)
    except OSError:
        return
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(KFD_TOPOLOGY, name, "properties"),
                      encoding="utf-8") as handle:
                text = handle.read()
        except OSError:
            continue
        props = {}
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                props[parts[0]] = parts[1]
        # ``simd_count`` is 0 for CPU-only KFD nodes.
        if props.get("simd_count", "0") == "0":
            continue
        try:
            version = int(props.get("gfx_target_version", "0"))
        except ValueError:
            continue
        target = _gfx_from_version(version)
        if target:
            yield target


def _gfx_from_command(exe, args):
    """Yield GFX tokens from a helper tool's stdout, or nothing on failure."""
    path = shutil.which(exe)
    if not path:
        return
    try:
        done = subprocess.run([path] + args, capture_output=True, text=True,
                              timeout=60)
    except Exception:
        return
    for token in _gfx_tokens(done.stdout):
        yield token


def _opencl_library():
    """Load the platform's OpenCL runtime, or return ``None``.

    The AMD graphics driver installs an OpenCL runtime and, on Windows, reports
    the GPU target as the device name (e.g. ``gfx1036``). That lets us detect
    the target without ROCm or ``clinfo`` being installed first.
    """
    if os.name == "nt":
        loader, names = ctypes.WinDLL, ("OpenCL.dll",)
    elif sys.platform == "darwin":
        loader, names = ctypes.CDLL, (
            "/System/Library/Frameworks/OpenCL.framework/OpenCL",)
    else:
        loader, names = ctypes.CDLL, (
            "libOpenCL.so.1", "libOpenCL.so", "OpenCL")
    for name in names:
        try:
            return loader(name)
        except OSError:
            continue
    return None


def _opencl_info(cl, device, param):
    """A string-valued OpenCL device attribute, or ``""`` on failure."""
    size = ctypes.c_size_t()
    if cl.clGetDeviceInfo(device, param, 0, None, ctypes.byref(size)) != 0 \
            or not size.value:
        return ""
    buffer = ctypes.create_string_buffer(size.value)
    if cl.clGetDeviceInfo(device, param, size.value, buffer,
                          ctypes.byref(size)) != 0:
        return ""
    return buffer.raw[:size.value].rstrip(b"\0").decode("utf-8", "replace")


def _opencl_device_names():
    """Yield ``(name, vendor)`` for every OpenCL device, or nothing.

    Loading the runtime or enumerating devices is best-effort: a missing or
    broken OpenCL install must not stop the bootstrapper.
    """
    cl = _opencl_library()
    if cl is None:
        return
    cl.clGetPlatformIDs.argtypes = [
        ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_uint)]
    cl.clGetDeviceIDs.argtypes = [
        ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_uint,
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint)]
    cl.clGetDeviceInfo.argtypes = [
        ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t)]

    count = ctypes.c_uint()
    if cl.clGetPlatformIDs(0, None, ctypes.byref(count)) != 0 or not count.value:
        return
    platforms = (ctypes.c_void_p * count.value)()
    if cl.clGetPlatformIDs(count.value, platforms, None) != 0:
        return
    for platform in platforms:
        devices = ctypes.c_uint()
        if cl.clGetDeviceIDs(platform, 0xFFFFFFFF, 0, None,
                             ctypes.byref(devices)) != 0 or not devices.value:
            continue
        handles = (ctypes.c_void_p * devices.value)()
        if cl.clGetDeviceIDs(platform, 0xFFFFFFFF, devices.value, handles,
                             None) != 0:
            continue
        for handle in handles:
            # CL_DEVICE_NAME and CL_DEVICE_VENDOR.
            yield (_opencl_info(cl, handle, 0x102B),
                   _opencl_info(cl, handle, 0x102C))


def _gfx_from_opencl():
    """Yield GFX targets from the driver-installed OpenCL runtime."""
    try:
        for name, _ in _opencl_device_names():
            for token in _gfx_tokens(name):
                yield token
    except Exception:
        # A broken or partial OpenCL install is not fatal.
        return


def _gfx_from_clinfo():
    """Yield GFX targets from ``clinfo`` (AMD's own Windows fallback).

    ``clinfo`` prints one ``Device Type:`` line per device followed by its
    ``Name:``; AMD's OpenCL driver reports the Name as the GFX target.
    """
    path = shutil.which("clinfo")
    if not path:
        return
    try:
        done = subprocess.run([path], capture_output=True, text=True, timeout=60)
    except Exception:
        return
    text = done.stdout or ""
    if not text:
        return
    current_is_gpu = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Device Type:"):
            current_is_gpu = "CL_DEVICE_TYPE_GPU" in stripped
        elif stripped.startswith("Name:") and current_is_gpu:
            for token in _gfx_tokens(stripped.split(":", 1)[1]):
                yield token
            current_is_gpu = False


def _gfx_candidates():
    """Yield detected AMD GFX targets, most reliable source first."""
    # rocminfo / rocm_agent_enumerator ship with a ROCm install.
    yield from _gfx_from_command("rocm_agent_enumerator", [])
    yield from _gfx_from_command("rocminfo", [])
    # The KFD topology works on Linux with only the amdgpu kernel driver.
    yield from _gfx_from_kfd()
    # AMD's graphics driver ships an OpenCL runtime that names the target on
    # Windows, so this works before ROCm is installed.
    yield from _gfx_from_opencl()
    # AMD's own helper and the optional clinfo CLI.
    yield from _gfx_from_command("rocm-bootstrap-detect", [])
    yield from _gfx_from_clinfo()


# Classified GPU targets: ``(supported, unsupported)`` as returned by
# :func:`_gfx_detection`. Cached because the probes shell out to driver tools.
_gfx_result = None


def _gfx_detection():
    """Classify the AMD GFX targets the driver reports.

    Returns ``(supported, unsupported)``: ``supported`` is the first detected
    target present in :data:`SUPPORTED_GFX` (or ``None``), and ``unsupported``
    is the ordered list of detected targets the current ROCm wheels carry no
    kernels for. Both derive once from :func:`_gfx_candidates` and are cached.
    """
    global _gfx_result
    if _gfx_result is not None:
        return _gfx_result
    supported = None
    unsupported = []
    for token in _gfx_candidates():
        if token in SUPPORTED_GFX:
            if supported is None:
                supported = token
        elif token not in unsupported:
            unsupported.append(token)
    _gfx_result = (supported, unsupported)
    return _gfx_result


def _rocm_supported():
    """True when the ROCm backend can target this machine's AMD GPU.

    A ``PAI_ROCM_GFX`` override always counts, since the user has named the
    target. Otherwise a detected supported target means yes; a card whose
    targets are all unsupported, or a machine where the driver reports nothing
    (so the target is unknown), means no.
    """
    if os.environ.get("PAI_ROCM_GFX", "").strip():
        return True
    supported, _ = _gfx_detection()
    return supported is not None


def _rocm_gfx():
    """The GPU target for the ROCm wheel extra, e.g. ``gfx1100``.

    ``PAI_ROCM_GFX`` wins; otherwise the target is detected from the installed
    driver via rocminfo, the Linux KFD topology, the OpenCL runtime,
    ``rocm-bootstrap-detect`` or ``clinfo`` (see :func:`_gfx_candidates`). When
    the driver reports only targets the current ROCm wheels do not support, or
    reports nothing usable, the ``all`` extra is chosen -- installing kernels
    for every supported GPU at the cost of a larger download. Only an explicit
    ROCm request reaches that fallback; auto-detection falls back to CPU
    instead.
    """
    global _gfx_cache
    if _gfx_cache is not None:
        return _gfx_cache

    override = os.environ.get("PAI_ROCM_GFX", "").strip().lower()
    if override:
        _gfx_cache = override
        return override

    supported, _ = _gfx_detection()
    if supported:
        _gfx_cache = supported
        return supported

    _gfx_cache = "all"
    return "all"


def _torch_index():
    value = os.environ.get("PAI_TORCH_INDEX", DEFAULT_TORCH_INDEX).strip()
    if not value or value.lower() == "none":
        return None
    if value.lower() == "cpu":
        return CPU_TORCH_INDEX
    return value


def _torch_source(backend):
    """A string identifying the torch wheels a backend installs.

    Stored in the marker so a change here (e.g. an AMD -> NVIDIA hardware swap
    moving ``rocm`` -> ``cuda``, or a different torch index) is detected and the
    old build is force-replaced rather than left in place.
    """
    if backend == "rocm":
        return "rocm:" + _rocm_index() + ":" + _rocm_gfx()
    return "torch:" + str(_torch_index())


def _ensure_venv():
    python = venv_python()
    if os.path.exists(python):
        return python
    if os.path.normcase(sys.executable) == os.path.normcase(python):
        return python
    print("Creating virtual environment in .venv ...")
    if _run([sys.executable, "-m", "venv", VENV]) != 0 \
            or not os.path.exists(python):
        sys.exit("Could not create the virtual environment in .venv.")
    return python


def _install_rocm(python):
    """Install the AMD ROCm AI stack (returns pip's exit code).

    The non-torch AI dependencies come from PyPI, then torch/torchvision are
    installed from AMD's index with the ``[device-<gfx>]`` extra, which selects
    the GPU-specific kernels and pulls in the matching ROCm runtime packages.
    """
    if _run([python, "-m", "pip", "install", "-r", ROCM_REQ]) != 0:
        return 1
    gfx = _rocm_gfx()
    index = _rocm_index()
    extra = "device-" + gfx
    print("Using AMD ROCm index: " + index)
    print("AMD GPU target: " + gfx)
    # AMD's index is primary so its torch (a local ``+rocm`` version, which
    # sorts above the plain PyPI wheel) is chosen; PyPI is kept as an extra so
    # torch's pure-Python dependencies still resolve.
    return _run([
        python, "-m", "pip", "install",
        "--index-url", index,
        "--extra-index-url", PYPI_INDEX,
        "torch[{0}]=={1}".format(extra, TORCH_VERSION),
        "torchvision[{0}]=={1}".format(extra, TORCHVISION_VERSION),
    ])


def _install(python, core_only, reinstall, backend):
    want = _requirements_hash(not core_only, backend)
    source = _torch_source(backend)
    marker = _read_marker()
    if not reinstall and marker.get("hash") == want:
        print("Environment already set up.")
        return

    # A pinned ``torch==2.13.0`` is satisfied by both the ``+cu130`` and the
    # ``+rocm`` local builds, so pip leaves an old-backend torch in place. When
    # the torch source changed, remove it first so the correct wheel is fetched.
    previous = marker.get("torch_source")
    if previous and previous != source:
        print("Compute backend changed; replacing torch ({0} -> {1}).".format(
            previous, source))
        _run([python, "-m", "pip", "uninstall", "-y",
              "torch", "torchvision", "torchaudio"])

    if not core_only and backend == "rocm" and not os.path.exists(ROCM_REQ):
        print("No requirements-rocm.txt found; installing core dependencies only.")
        core_only = True
    if not core_only and backend != "rocm" and not os.path.exists(AI_REQ):
        print("No requirements-ai.txt found; installing core dependencies only.")
        core_only = True

    _run([python, "-m", "pip", "install", "--upgrade", "pip"])
    if _run([python, "-m", "pip", "install", "-r", CORE_REQ]) != 0:
        sys.exit("Installing Pai's core dependencies failed.")

    if core_only:
        print("Core-only install: generation, upscale and background "
              "removal stay disabled.")
    elif backend == "rocm":
        if _install_rocm(python) != 0:
            sys.exit("Installing Pai's AMD ROCm dependencies failed.")
    else:
        cmd = [python, "-m", "pip", "install", "-r", AI_REQ]
        index = _torch_index()
        if index:
            print("Using torch index: " + index)
            cmd += ["--extra-index-url", index]
        if _run(cmd) != 0:
            sys.exit("Installing Pai's AI dependencies failed.")

    _write_marker({
        "hash": want,
        "core_only": core_only,
        "backend": backend,
        "torch_source": source,
        "python": sys.version.split()[0],
    })
    print("Environment ready.")


def main(argv):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):
        pass
    if not _supported_python(sys.version_info):
        if sys.version_info < MIN_PYTHON:
            sys.exit(
                "Pai needs Python {0}.{1} or newer, but this interpreter is "
                "{2}.{3}.".format(MIN_PYTHON[0], MIN_PYTHON[1],
                                  sys.version_info[0], sys.version_info[1])
            )
        sys.exit(
            "Pai supports Python up to {0}.{1}, but this interpreter is "
            "{2}.{3}. Torch wheels for it are not published yet; install "
            "Python {0}.{1} and run this again.".format(
                MAX_PYTHON[0], MAX_PYTHON[1] - 1,
                sys.version_info[0], sys.version_info[1])
        )

    core_only, reinstall, setup_only, rocm, passthrough = _parse(argv)
    backend = _backend(rocm)
    explicit = rocm or bool(os.environ.get("PAI_BACKEND", "").strip())
    if backend == "rocm":
        if explicit:
            print("Backend: AMD ROCm")
        else:
            print("Detected an AMD GPU; using the ROCm backend.")
    elif explicit:
        # An explicit request; ``_backend`` already printed the reason for any
        # ROCm -> CPU fallback.
        print("Backend: " + ("CPU" if backend == "cpu" else "CUDA"))
    else:
        detected = _auto_backend()
        if detected == "cuda":
            print("Detected an NVIDIA GPU; using the CUDA backend.")
        elif backend == "cpu":
            print("Using the CPU backend.")
        else:
            print("No supported GPU detected; using the CUDA backend "
                  "(which also runs on CPU).")
    if reinstall and os.path.isdir(VENV):
        # ``--reinstall`` means rebuild from scratch: dropping the whole venv is
        # the only way to be sure a changed backend or torch index takes effect.
        print("Removing the existing environment in " + VENV + " ...")
        shutil.rmtree(VENV, ignore_errors=True)
    python = _ensure_venv()
    _install(python, core_only, reinstall, backend)

    if setup_only:
        return 0
    return subprocess.call([python, APP] + passthrough)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
