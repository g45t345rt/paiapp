"""Checks for the on-disk component-kind cache.

Weights-free: tiny stub files are detected by filename, and the persisted
cache's warm/reload/invalidate/prune behaviour is exercised against a temp
cache path (the real per-user folder is never touched).

Run: .venv\\Scripts\\python.exe tests\\test_component_kinds_cache.py
"""

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import component_kinds as kinds  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def check_true(name, cond):
    check(name, bool(cond), True)


def _reset():
    """Simulate a fresh process: empty in-memory cache, nothing loaded yet."""
    kinds._CACHE.clear()
    kinds._CACHE_LOADED = False


def _write_file(path, data=b"\x00" * 16):
    with open(path, "wb") as handle:
        handle.write(data)
    return path


def test_save_then_reload_is_warm():
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = os.path.join(tmp, "kinds.json")
        model = _write_file(os.path.join(tmp, "clip_l.safetensors"))
        original_path = kinds.cache_path
        kinds.cache_path = lambda: cache_file
        try:
            _reset()
            check("detected by name", kinds.detect(model, kinds.TEXT_ENCODER), "clip_l")
            kinds.save_cache()
            check_true("cache file written", os.path.exists(cache_file))

            _reset()
            kinds._ensure_loaded()
            check("reloaded from disk is warm",
                  kinds.cached(model, kinds.TEXT_ENCODER), "clip_l")

            # A disk hit must not fall through to a header read: make the
            # detector explode and confirm detect() still answers from cache.
            original_detect = kinds._detect_uncached

            def boom(*_args, **_kwargs):
                raise AssertionError("header read on a warm cache")

            kinds._detect_uncached = boom
            try:
                check("no header read on warm disk hit",
                      kinds.detect(model, kinds.TEXT_ENCODER), "clip_l")
            finally:
                kinds._detect_uncached = original_detect
        finally:
            kinds.cache_path = original_path
            _reset()


def test_stamp_change_invalidates():
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = os.path.join(tmp, "kinds.json")
        model = _write_file(os.path.join(tmp, "clip_l.safetensors"))
        original_path = kinds.cache_path
        kinds.cache_path = lambda: cache_file
        try:
            _reset()
            kinds.detect(model, kinds.TEXT_ENCODER)
            kinds.save_cache()

            status = os.stat(model)
            os.utime(model, (status.st_atime, int(status.st_mtime) + 10))

            _reset()
            kinds._ensure_loaded()
            check("changed stamp is not warm",
                  kinds.cached(model, kinds.TEXT_ENCODER), None)
        finally:
            kinds.cache_path = original_path
            _reset()


def test_save_prunes_missing_files():
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = os.path.join(tmp, "kinds.json")
        model = _write_file(os.path.join(tmp, "clip_l.safetensors"))
        original_path = kinds.cache_path
        kinds.cache_path = lambda: cache_file
        try:
            _reset()
            kinds.detect(model, kinds.TEXT_ENCODER)
            kinds.save_cache()

            os.remove(model)
            _reset()
            kinds._ensure_loaded()
            kinds.save_cache()  # must drop the now-missing file
            with open(cache_file, encoding="utf-8") as handle:
                check("missing file pruned", json.load(handle), [])
        finally:
            kinds.cache_path = original_path
            _reset()


def test_undetectable_is_remembered():
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = os.path.join(tmp, "kinds.json")
        model = _write_file(os.path.join(tmp, "mystery.bin"))
        original_path = kinds.cache_path
        kinds.cache_path = lambda: cache_file
        try:
            _reset()
            check("unknown kind", kinds.detect(model, kinds.TEXT_ENCODER), None)
            kinds.save_cache()

            _reset()
            kinds._ensure_loaded()
            stamp = kinds._stamp(model)
            check_true("null kind stored",
                       (model, kinds.TEXT_ENCODER, stamp) in kinds._CACHE)
        finally:
            kinds.cache_path = original_path
            _reset()


def test_corrupt_cache_is_ignored():
    with tempfile.TemporaryDirectory() as tmp:
        cache_file = os.path.join(tmp, "kinds.json")
        model = _write_file(os.path.join(tmp, "clip_l.safetensors"))
        original_path = kinds.cache_path
        kinds.cache_path = lambda: cache_file
        try:
            with open(cache_file, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            _reset()
            kinds._ensure_loaded()
            check("corrupt cache loads empty",
                  kinds.cached(model, kinds.TEXT_ENCODER), None)
        finally:
            kinds.cache_path = original_path
            _reset()


def main():
    test_save_then_reload_is_warm()
    test_stamp_change_invalidates()
    test_save_prunes_missing_files()
    test_undetectable_is_remembered()
    test_corrupt_cache_is_ignored()
    if FAILED:
        print("\nFAILED:")
        for item in FAILED:
            print("  " + item)
        sys.exit(1)
    print("\nall component-kinds cache checks passed")


if __name__ == "__main__":
    main()