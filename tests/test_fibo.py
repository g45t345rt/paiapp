"""Unit tests for the Fibo 2.0 family wiring.

Run: .venv\\Scripts\\python.exe tests\\test_fibo.py

The family wraps ``diffusers``' ``BriaFibo2Pipeline``, so nothing here loads the
gated 8B weights: only the loader's helpers, the family registry entries and the
``engine.pipeline.generate`` dispatch (with the generator stubbed) are checked.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import component_kinds, fibo, paths  # noqa: E402
from engine import pipeline  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_registry():
    check("model group registered", ("Fibo 2.0", "fibo") in paths.MODEL_GROUPS, True)
    check("base slot is model", paths._FAMILY_BASE_SLOT.get("fibo"), "model")
    check("fibo defaults present", "model" in paths.FIBO_DEFAULTS, True)
    check("kind registered", component_kinds.KINDS.get("fibo"), ("Fibo 2.0", "model"))
    check("kind offered", "fibo" in component_kinds.CATEGORY_KINDS[component_kinds.MODEL], True)
    check("classified by name", paths.classify_model("Fibo-2.0-dev"), ("Fibo 2.0", "fibo"))
    check("detected by name", component_kinds.detect("briaai/Fibo-2.0-dev"), "fibo")


def test_family_defaults():
    defaults = paths.family_defaults("fibo")
    check("steps", defaults["steps"], 50)
    check("cfg", defaults["cfg"], 5.0)
    check("size", (defaults["width"], defaults["height"]), (1024, 1024))
    check("recipe built", paths.DEFAULT_RECIPE_NAME in paths.family_recipes("fibo"), True)


def test_pipeline_slot():
    check("cache slot", pipeline._SLOTS.get("fibo"),
          ("_fibo", "_fibo_key", "_fibo_error"))
    check("generator wired", callable(pipeline._generate_fibo), True)
    check("loader wired", callable(pipeline._load_fibo), True)


def test_repo_id_detection():
    check("repo id", fibo.is_repo_id("briaai/Fibo-2.0-dev"), True)
    check("absolute path is not a repo", fibo.is_repo_id("C:/models/fibo"), False)
    check("blank is not a repo", fibo.is_repo_id(""), False)
    check("missing dir-like is not a repo", fibo.is_repo_id(str(ROOT)), False)


def test_pipeline_class_available():
    # diffusers>=0.42 is installed for this environment, so the class resolves.
    cls = fibo._pipeline_class()
    check("pipeline class named", getattr(cls, "__name__", ""), "BriaFibo2Pipeline")


def test_dispatch_routes_to_fibo():
    seen = {}

    def fake(prompt, cfg, image=None, progress=None, cancel=None, mask=None,
             regions_list=None, ref_images=None, preview=None):
        seen["family"] = cfg.get("family")
        seen["refs"] = ref_images
        return "sentinel"

    original = pipeline._generate_fibo
    pipeline._generate_fibo = fake
    try:
        marker = object()
        out = pipeline.generate("a caption", {"family": "fibo"}, ref_images=[marker])
    finally:
        pipeline._generate_fibo = original
    check("routed output", out, "sentinel")
    check("family in config", seen.get("family"), "fibo")
    check("references forwarded", seen.get("refs"), [marker])


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
