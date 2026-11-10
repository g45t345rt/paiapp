"""Regression test for the Model Settings component pool folder fallback.

Run: .venv\\Scripts\\python.exe tests\\test_modelsettings_pools.py

A file downloaded into a pool's engine folder must be selectable in a model
row's component slot even when it was never curated into the pool (e.g. a Hub
download whose pool additions were not saved). Covers the VAE slot of the
MiniMax H3 Image family.
"""

import os
import sys
import tempfile
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import paths  # noqa: E402
from paintapp import modelsettings as ms  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _dialog(pools):
    dialog = object.__new__(ms.ModelSettingsDialog)
    dialog.pools = {name: [] for _title, name in ms._POOL_CATEGORIES}
    dialog.pools.update(pools)
    dialog.entries = []
    dialog._kinds_ready = True
    return dialog


def test_folder_vae_is_listed_in_slot():
    with tempfile.TemporaryDirectory() as folder:
        vae = os.path.join(folder, "minimax_h3_image_vae_fp16.safetensors")
        open(vae, "wb").close()
        with mock.patch.object(
            paths, "component_path", return_value=folder,
        ):
            dialog = _dialog({})
            items = dialog._pool_items("vaes")
            check("folder file listed", [item["path"] for item in items], [vae])

            kind = ms.engine_kinds.SLOT_KIND[("h3_image", "vae")]
            choices, mapping = dialog._slot_file_choices("vaes", kind, None)
            check("slot lists file",
                  mapping.get("minimax_h3_image_vae_fp16  (MiniMax H3 Image VAE)"),
                  vae)
            check("default is first", choices[0], "(default)")


def test_curated_entry_wins_over_folder_file():
    with tempfile.TemporaryDirectory() as folder:
        vae = os.path.join(folder, "minimax_h3_image_vae_fp16.safetensors")
        open(vae, "wb").close()
        curated = {"label": "Curated", "path": vae}
        with mock.patch.object(
            paths, "component_path", return_value=folder,
        ):
            dialog = _dialog({"vaes": [curated]})
            items = dialog._pool_items("vaes")
            check("one entry per path", len(items), 1)
            check("curated label kept", items[0]["label"], "Curated")


def test_detection_pass_includes_folder_file():
    with tempfile.TemporaryDirectory() as folder:
        vae = os.path.join(folder, "minimax_h3_image_vae_fp16.safetensors")
        open(vae, "wb").close()
        with mock.patch.object(
            paths, "component_path", return_value=folder,
        ):
            dialog = _dialog({})
            pending = dialog._pending_detections()
            check("folder file detected",
                  (vae, ms.engine_kinds.VAE) in pending, True)


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