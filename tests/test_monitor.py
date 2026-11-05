"""Unit tests for GPU memory sampling (run: .venv\\Scripts\\python.exe tests\\test_monitor.py)."""

import contextlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from paintapp import monitor  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


@contextlib.contextmanager
def fake_tools(run, which=None):
    """Patch the subprocess seam so the CLI probes can be driven from strings."""
    if which is None:
        which = lambda name: "/usr/bin/" + name  # noqa: E731
    saved_which, saved_run = monitor.shutil.which, monitor._run_command
    monitor.shutil.which, monitor._run_command = which, run
    try:
        yield
    finally:
        monitor.shutil.which, monitor._run_command = saved_which, saved_run


def test_pdh_instance_luid():
    check("luid basic", monitor._pdh_instance_luid(
        "luid_0x00000000_0x0001131F_phys_0"), (0, 0x1131F))
    check("luid high part", monitor._pdh_instance_luid(
        "luid_0x0000000A_0x0001279F_phys_1_part_0"), (0x0A, 0x1279F))
    check("luid junk", monitor._pdh_instance_luid("not a counter"), None)
    check("luid empty", monitor._pdh_instance_luid(""), None)
    check("luid truncated", monitor._pdh_instance_luid("luid_0x0001"), None)


def test_nvidia_smi_memory():
    run = lambda cmd: "1024, 12288\n512, 12288\n"  # noqa: E731
    with fake_tools(run):
        check("nvidia sums cards", monitor._nvidia_smi_memory(), (1.5, 24.0))
    with fake_tools(run, which=lambda name: None):
        check("nvidia absent", monitor._nvidia_smi_memory(), None)
    with fake_tools(lambda cmd: "garbage"):
        check("nvidia unparseable", monitor._nvidia_smi_memory(), None)
    with fake_tools(lambda cmd: ""):
        check("nvidia empty", monitor._nvidia_smi_memory(), None)


def test_rocm_smi_memory():
    data = {
        "card0": {"VRAM Total Memory (B)": "8589934592",
                  "VRAM Total Used Memory (B)": "1073741824"},
        "card1": {"VRAM Total Memory (B)": "2147483648",
                  "VRAM Total Used Memory (B)": "536870912"},
    }
    with fake_tools(lambda cmd: json.dumps(data)):
        check("rocm sums cards", monitor._rocm_smi_memory(), (1.5, 10.0))
    with fake_tools(lambda cmd: json.dumps({"card0": {"other": "1"}})):
        check("rocm no vram", monitor._rocm_smi_memory(), None)
    with fake_tools(lambda cmd: "not json"):
        check("rocm bad json", monitor._rocm_smi_memory(), None)
    with fake_tools(lambda cmd: json.dumps(data), which=lambda name: None):
        check("rocm absent", monitor._rocm_smi_memory(), None)


def test_amd_smi_memory():
    data = {"gpu_data": [{"mem_usage": {
        "total_vram": {"value": 24576, "unit": "MB"},
        "used_vram": {"value": 1024, "unit": "MB"}}}]}
    with fake_tools(lambda cmd: json.dumps(data)):
        check("amd-smi MB", monitor._amd_smi_memory(), (1.0, 24.0))

    nested = {"gpu_data": [{"mem_usage": {
        "total_vram": {"value": 8, "unit": "GB"},
        "used_vram": {"value": 512, "unit": "MB"}}}]}
    with fake_tools(lambda cmd: json.dumps(nested)):
        check("amd-smi mixed units", monitor._amd_smi_memory(), (0.5, 8.0))

    plain = {"gpu_data": [{"used_vram": 2048, "total_vram": 8192}]}
    with fake_tools(lambda cmd: json.dumps(plain)):
        check("amd-smi plain MB", monitor._amd_smi_memory(), (2.0, 8.0))

    with fake_tools(lambda cmd: json.dumps({"gpu_data": []})):
        check("amd-smi empty", monitor._amd_smi_memory(), None)


def test_chain_falls_through():
    def raising():
        raise RuntimeError("boom")

    saved = monitor._probes
    monitor._probes = lambda: (raising, lambda: None, lambda: (1.0, 2.0))
    try:
        check("first working probe wins", monitor.gpu_memory(), (1.0, 2.0))
    finally:
        monitor._probes = saved


def test_chain_all_fail():
    saved = monitor._probes
    monitor._probes = lambda: (lambda: None, lambda: None)
    try:
        check("no source -> None", monitor.gpu_memory(), None)
    finally:
        monitor._probes = saved


def test_probe_order():
    probes = monitor._probes()
    if os.name == "nt":
        check("windows leads with native", probes[0].__name__,
              "_windows_gpu_memory")
        check("windows falls back to nvidia-smi", probes[1].__name__,
              "_nvidia_smi_memory")
    else:
        check("posix leads with nvidia-smi", probes[0].__name__,
              "_nvidia_smi_memory")
        check("posix includes rocm-smi", probes[1].__name__, "_rocm_smi_memory")


def test_gpu_memory_live():
    value = monitor.gpu_memory()
    ok = value is None or (
        isinstance(value, tuple) and len(value) == 2
        and all(isinstance(part, float) for part in value)
        and value[0] >= 0 and value[1] > 0)
    check("live gpu_memory shape", ok, True)
    print(f"    live reading: {value}")


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
