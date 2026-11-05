"""Unit tests for backend auto-detection (run: .venv\\Scripts\\python.exe tests\\test_setup_env.py)."""

import contextlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import setup_env  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


@contextlib.contextmanager
def patched(**attrs):
    """Temporarily replace module-level attributes and restore them."""
    saved = {name: getattr(setup_env, name) for name in attrs}
    for name, value in attrs.items():
        setattr(setup_env, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(setup_env, name, value)


@contextlib.contextmanager
def backend_env(value):
    saved = os.environ.get("PAI_BACKEND")
    if value is None:
        os.environ.pop("PAI_BACKEND", None)
    else:
        os.environ["PAI_BACKEND"] = value
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("PAI_BACKEND", None)
        else:
            os.environ["PAI_BACKEND"] = saved


def test_vendors_in_text():
    check("nvidia", setup_env._vendors_in_text("NVIDIA GeForce RTX 4090"),
          {"nvidia"})
    check("amd", setup_env._vendors_in_text("Advanced Micro Devices, Inc."),
          {"amd"})
    check("radeon", setup_env._vendors_in_text("AMD Radeon RX 7900 XTX"),
          {"amd"})
    check("intel", setup_env._vendors_in_text("Intel(R) UHD Graphics"),
          {"intel"})
    check("empty", setup_env._vendors_in_text(""), set())


def test_detected_vendors():
    # Drive the seams directly instead of faking whole modules.
    with patched(_sysfs_vendors=lambda: set(),
                 _windows_registry_vendors=lambda: set()):
        saved_which = setup_env.shutil.which
        saved_exists = setup_env.os.path.exists
        setup_env.shutil.which = lambda name: "/x/nvidia-smi" if name == "nvidia-smi" else None
        setup_env.os.path.exists = lambda path: False
        try:
            check("nvidia cli", setup_env._detected_vendors(), {"nvidia"})
            setup_env.shutil.which = lambda name: "/x/rocm-smi" if name == "rocm-smi" else None
            check("amd cli", setup_env._detected_vendors(), {"amd"})
            setup_env.shutil.which = lambda name: None
            setup_env.os.path.exists = lambda path: path == "/dev/kfd"
            check("amd kernel driver", setup_env._detected_vendors(), {"amd"})
            setup_env.os.path.exists = lambda path: False
            check("no gpu", setup_env._detected_vendors(), set())
        finally:
            setup_env.shutil.which = saved_which
            setup_env.os.path.exists = saved_exists


def test_auto_backend():
    saved_platform = sys.platform
    try:
        sys.platform = "darwin"
        check("macos is cpu", setup_env._auto_backend(), "cpu")
        sys.platform = "linux"
        with patched(_detected_vendors=lambda: {"nvidia"}):
            check("nvidia -> cuda", setup_env._auto_backend(), "cuda")
        with patched(_detected_vendors=lambda: {"amd"}):
            check("amd -> rocm", setup_env._auto_backend(), "rocm")
        with patched(_detected_vendors=lambda: {"nvidia", "amd"}):
            check("both -> cuda", setup_env._auto_backend(), "cuda")
        with patched(_detected_vendors=lambda: set()):
            check("unknown -> None", setup_env._auto_backend(), None)
    finally:
        sys.platform = saved_platform


def test_backend_precedence():
    with backend_env(None):
        with patched(_detected_vendors=lambda: {"amd"}):
            check("detect amd", setup_env._backend(False), "rocm")
        with patched(_detected_vendors=lambda: {"nvidia"}):
            check("detect nvidia", setup_env._backend(False), "cuda")
        with patched(_detected_vendors=lambda: set()):
            check("fallback cuda", setup_env._backend(False), "cuda")
        with patched(_detected_vendors=lambda: {"nvidia"}):
            check("--rocm wins", setup_env._backend(True), "rocm")
    with backend_env("cpu"):
        with patched(_detected_vendors=lambda: {"nvidia"}):
            check("PAI_BACKEND=cpu wins", setup_env._backend(False), "cpu")
    with backend_env("rocm"):
        with patched(_detected_vendors=lambda: {"nvidia"}):
            check("PAI_BACKEND=rocm wins", setup_env._backend(False), "rocm")
    with backend_env("cuda"):
        with patched(_detected_vendors=lambda: {"amd"}):
            check("PAI_BACKEND=cuda wins", setup_env._backend(False), "cuda")


def test_supported_python():
    check("3.11 rejected", setup_env._supported_python((3, 11)), False)
    check("3.12 accepted", setup_env._supported_python((3, 12)), True)
    check("3.13 accepted", setup_env._supported_python((3, 13)), True)
    check("3.14 accepted", setup_env._supported_python((3, 14)), True)
    check("3.15 rejected", setup_env._supported_python((3, 15)), False)
    check("3.16 rejected", setup_env._supported_python((3, 16)), False)
    check("micro ignored", setup_env._supported_python((3, 14, 2)), True)


class _Done:
    def __init__(self, stdout):
        self.stdout = stdout


@contextlib.contextmanager
def fake_run(which, run):
    saved_which = setup_env.shutil.which
    saved_run = setup_env.subprocess.run
    setup_env.shutil.which = which
    setup_env.subprocess.run = run
    try:
        yield
    finally:
        setup_env.shutil.which = saved_which
        setup_env.subprocess.run = saved_run


def test_gfx_from_version():
    check("gfx1100", setup_env._gfx_from_version(110000), "gfx1100")
    check("gfx942", setup_env._gfx_from_version(90402), "gfx942")
    check("gfx90a", setup_env._gfx_from_version(90010), "gfx90a")
    check("gfx1151", setup_env._gfx_from_version(110501), "gfx1151")
    check("gfx1101", setup_env._gfx_from_version(110001), "gfx1101")
    check("gfx1201", setup_env._gfx_from_version(120001), "gfx1201")
    check("zero -> None", setup_env._gfx_from_version(0), None)


def test_gfx_tokens():
    check("tokens", list(setup_env._gfx_tokens("Name: gfx1100 (RDNA3)")),
          ["gfx1100"])
    check("multiple", list(setup_env._gfx_tokens("gfx90a and GFX942")),
          ["gfx90a", "gfx942"])
    check("none", list(setup_env._gfx_tokens("no targets here")), [])


def test_gfx_from_kfd():
    with tempfile.TemporaryDirectory() as root:
        os.mkdir(os.path.join(root, "0"))
        with open(os.path.join(root, "0", "properties"), "w") as fh:
            fh.write("simd_count 0\ngfx_target_version 110000\n")
        os.mkdir(os.path.join(root, "1"))
        with open(os.path.join(root, "1", "properties"), "w") as fh:
            fh.write("simd_count 24\ngfx_target_version 110000\n")
        os.mkdir(os.path.join(root, "2"))
        with open(os.path.join(root, "2", "properties"), "w") as fh:
            fh.write("simd_count 24\ngfx_target_version 90402\n")
        with patched(KFD_TOPOLOGY=root):
            check("kfd skips cpu, decodes gpu",
                  list(setup_env._gfx_from_kfd()), ["gfx1100", "gfx942"])
    with patched(KFD_TOPOLOGY="/no/such/dir"):
        check("kfd absent", list(setup_env._gfx_from_kfd()), [])


def test_gfx_from_clinfo():
    output = (
        "  Device Type:      CL_DEVICE_TYPE_GPU\n"
        "  Name:             gfx1100\n"
        "  Device Type:      CL_DEVICE_TYPE_CPU\n"
        "  Name:             gfx9999\n"
        "  Device Type:      CL_DEVICE_TYPE_GPU\n"
        "  Name:             gfx90a\n"
    )
    which = lambda name: "/usr/bin/clinfo" if name == "clinfo" else None  # noqa: E731
    with fake_run(which, lambda cmd, **kw: _Done(output)):
        check("clinfo gpu names", list(setup_env._gfx_from_clinfo()),
              ["gfx1100", "gfx90a"])
    with fake_run(lambda name: None, lambda cmd, **kw: _Done(output)):
        check("clinfo absent", list(setup_env._gfx_from_clinfo()), [])


def test_torch_source():
    with patched(_torch_index=lambda: "https://cu130",
                 _rocm_index=lambda: "https://amd",
                 _rocm_gfx=lambda: "gfx1100"):
        cuda = setup_env._torch_source("cuda")
        rocm = setup_env._torch_source("rocm")
        check("cuda vs rocm differ", cuda != rocm, True)
        check("cuda source", cuda, "torch:https://cu130")
        check("rocm source", rocm, "rocm:https://amd:gfx1100")
    with patched(_torch_index=lambda: "https://cpu",
                 _rocm_index=lambda: "https://amd",
                 _rocm_gfx=lambda: "gfx1100"):
        check("index change differs", setup_env._torch_source("cuda") != cuda,
              True)
    with patched(_torch_index=lambda: "https://cu130",
                 _rocm_index=lambda: "https://amd",
                 _rocm_gfx=lambda: "gfx942"):
        check("gfx change differs", setup_env._torch_source("rocm") != rocm,
              True)


def test_hash_tracks_torch_index():
    with patched(_torch_index=lambda: "https://cu130"):
        a = setup_env._requirements_hash(True, "cuda")
    with patched(_torch_index=lambda: "https://cpu"):
        b = setup_env._requirements_hash(True, "cuda")
    check("hash changes with index", a != b, True)


def test_install_replaces_torch_on_backend_change():
    commands = []
    saved_marker = setup_env._read_marker
    saved_write = setup_env._write_marker
    saved_run = setup_env._run
    setup_env._read_marker = lambda: {
        "hash": "old-hash",
        "backend": "rocm",
        "torch_source": "rocm:https://amd:gfx1100",
    }
    setup_env._write_marker = lambda payload: None
    setup_env._run = lambda cmd: commands.append(cmd) or 0
    try:
        setup_env._install("python", False, False, "cuda")
    finally:
        setup_env._read_marker = saved_marker
        setup_env._write_marker = saved_write
        setup_env._run = saved_run
    uninstalled = [c for c in commands if "uninstall" in c]
    check("backend swap uninstalls torch", len(uninstalled), 1)
    check("uninstalls torch and vision",
          uninstalled[0][-3:], ["torch", "torchvision", "torchaudio"])


def test_install_skips_uninstall_when_unchanged():
    commands = []
    saved_read = setup_env._read_marker
    saved_write = setup_env._write_marker
    saved_run = setup_env._run
    saved_source = setup_env._torch_source
    setup_env._read_marker = lambda: {
        "hash": "old-hash",
        "backend": "cuda",
        "torch_source": "torch:same",
    }
    setup_env._write_marker = lambda payload: None
    setup_env._run = lambda cmd: commands.append(cmd) or 0
    setup_env._torch_source = lambda backend: "torch:same"
    try:
        setup_env._install("python", False, False, "cuda")
    finally:
        setup_env._read_marker = saved_read
        setup_env._write_marker = saved_write
        setup_env._run = saved_run
        setup_env._torch_source = saved_source
    check("same source keeps torch",
          any("uninstall" in c for c in commands), False)


def test_rocm_gfx_selection():
    saved = setup_env._gfx_cache
    try:
        setup_env._gfx_cache = None
        with backend_env(None):
            with patched(_gfx_candidates=lambda: iter(["gfx9999", "gfx1100"])):
                check("skips unsupported, picks known",
                      setup_env._rocm_gfx(), "gfx1100")
        setup_env._gfx_cache = None
        with patched(_gfx_candidates=lambda: iter(["gfx900"])):
            check("unknown target -> all", setup_env._rocm_gfx(), "all")
        setup_env._gfx_cache = None
        with backend_env(None):
            os.environ["PAI_ROCM_GFX"] = "gfx1234"
            try:
                with patched(_gfx_candidates=lambda: iter(["gfx1100"])):
                    check("PAI_ROCM_GFX override", setup_env._rocm_gfx(),
                          "gfx1234")
            finally:
                os.environ.pop("PAI_ROCM_GFX", None)
    finally:
        setup_env._gfx_cache = saved


def main():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    if FAILED:
        print(f"\n{len(FAILED)} FAILED")
        for line in FAILED:
            print("  " + line)
        sys.exit(1)
    print("\nall passed")


if __name__ == "__main__":
    main()
