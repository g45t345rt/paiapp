"""Unit tests for the generation-engine tuning settings.

Run: .venv\\Scripts\\python.exe tests\\test_tuning.py
"""

import contextlib
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import settings as engine_settings  # noqa: E402
from engine import tuning  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


@contextlib.contextmanager
def temp_settings():
    """Point the settings file at a throwaway directory for the test."""
    saved = engine_settings.app_dir
    with tempfile.TemporaryDirectory() as tmp:
        engine_settings.app_dir = lambda: os.path.join(tmp, "Pai")
        try:
            yield tmp
        finally:
            engine_settings.app_dir = saved


def test_defaults_when_unset():
    with temp_settings():
        check("defaults equal", tuning.read(), dict(tuning.DEFAULTS))


def test_round_trip():
    with temp_settings():
        values = dict(tuning.DEFAULTS)
        values["precision"] = "fp8"
        values["vram_reserve_gb"] = 1.5
        values["auto_unload_minutes"] = 30
        values["tiled_decode"] = True
        values["disk_stream"] = True
        tuning.write(values)
        got = tuning.read()
        check("precision kept", got["precision"], "fp8")
        check("vram kept", got["vram_reserve_gb"], 1.5)
        check("minutes kept", got["auto_unload_minutes"], 30)
        check("tiled kept", got["tiled_decode"], True)
        check("disk kept", got["disk_stream"], True)


def test_defaults_drop_section():
    with temp_settings():
        tuning.write(dict(tuning.DEFAULTS))
        check("section removed", engine_settings.section(tuning.SECTION), {})


def test_normalize_clamps_and_coerces():
    got = tuning.normalize({
        "vram_reserve_gb": 999,
        "auto_unload_minutes": 0,
        "prefetch_depth": 99,
        "precision": "bogus",
        "keep_resident": "false",
        "ram_budget_gb": "-5",
        "unknown": 1,
    })
    check("clamp high", got["vram_reserve_gb"], 32.0)
    check("clamp low", got["auto_unload_minutes"], 1)
    check("prefetch clamp", got["prefetch_depth"], 6)
    check("bad choice falls back", got["precision"], "auto")
    check("bool from string", got["keep_resident"], False)
    check("negative float clamped", got["ram_budget_gb"], 0.0)
    check("unknown dropped", "unknown" in got, False)


def test_normalize_rejects_junk_type():
    check("non-dict", tuning.normalize(["nope"]), dict(tuning.DEFAULTS))
    got = tuning.normalize({"tile_size": "abc"})
    check("bad int falls back", got["tile_size"], tuning.DEFAULTS["tile_size"])


def test_auto_unload_seconds():
    with temp_settings():
        values = dict(tuning.DEFAULTS)
        values["auto_unload_minutes"] = 5
        tuning.write(values)
        check("minutes to seconds", tuning.auto_unload_seconds(), 300)
        values["auto_unload"] = False
        tuning.write(values)
        check("disabled is zero", tuning.auto_unload_seconds(), 0)


def test_apply_pushes_into_precision():
    from engine import precision

    with temp_settings():
        values = dict(tuning.DEFAULTS)
        values["precision"] = "full"
        values["vram_reserve_gb"] = 1.0
        values["stream_reserve_gb"] = 3.0
        values["keep_resident"] = False
        values["compile_vae"] = False
        values["disk_stream"] = True
        tuning.write(values)
        tuning.apply()
        cfg = precision.config()
        check("fp8 mode pushed", cfg["fp8_mode"], "full")
        check("vram reserve pushed", cfg["vram_reserve"], int(1.0 * 1e9))
        check("stream reserve pushed", cfg["stream_reserve"], int(3.0 * 1e9))
        check("keep_resident pushed", cfg["keep_resident"], False)
        check("compile pushed", cfg["compile_vae"], False)
        check("disk pushed", cfg["disk_stream"], True)
    # Restore the shipped defaults so later tests see the normal policy.
    precision.configure(fp8_mode="auto", vram_reserve=precision.DEFAULT_RESERVE,
                        stream_reserve=precision.STREAM_RESERVE, keep_resident=True,
                        compile_vae=True, disk_stream=False)


def test_human_size():
    check("bytes", tuning.human_size(512), "512 B")
    check("gb", tuning.human_size(1.5 * 1024 ** 3), "1.50 GB")
    check("junk", tuning.human_size("x"), "")


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