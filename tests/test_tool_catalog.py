"""Offline tests for the tool-model download catalog.

Run: .venv\\Scripts\\python.exe tests\\test_tool_catalog.py

No network or GPU is used: the catalog's consistency checks, job destinations
and already-present detection are exercised directly.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import paths  # noqa: E402
from engine import tool_catalog  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def test_catalog_is_consistent():
    check("catalog issues", tool_catalog.validate(), [])


def test_bundles_have_descriptions():
    for bundle in tool_catalog.bundles():
        description = bundle.get("description") or {}
        check(f"{bundle['id']} purpose", bool(description.get("purpose")), True)
        check(f"{bundle['id']} strengths", bool(description.get("strengths")), True)
        check(f"{bundle['id']} consider", bool(description.get("consider")), True)


def test_jobs_land_in_the_tool_folders():
    birefnet = tool_catalog.bundle("birefnet-general")
    jobs = tool_catalog.jobs(birefnet)
    check("birefnet one file", len(jobs), 1)
    check("birefnet background folder",
          jobs[0]["dest_dir"], paths.background_dir())
    check("birefnet pool", jobs[0]["pool"], "background")

    esrgan = tool_catalog.bundle("realesrgan-x4")
    job = tool_catalog.jobs(esrgan)[0]
    check("esrgan upscale folder", job["dest_dir"], paths.upscale_dir())
    check("esrgan pool", job["pool"], "upscale")


def test_seedvr2_bundle_pairs_model_and_vae():
    bundle = tool_catalog.bundle("seedvr2-3b")
    jobs = {job["slot"]: job for job in tool_catalog.jobs(bundle)}
    check("seedvr2 slots", set(jobs), {"model", "vae"})
    check("seedvr2 model pool", jobs["model"]["pool"], "upscale")
    check("seedvr2 vae folder", jobs["vae"]["dest_dir"], paths.vae_dir())
    check("seedvr2 vae name looks right",
          "seedvr2" in jobs["vae"]["name"] and "vae" in jobs["vae"]["name"],
          True)
    check("seedvr2 vae pool", jobs["vae"]["pool"], "vaes")


def test_existing_target_is_reused():
    with tempfile.TemporaryDirectory() as folder:
        job = dict(tool_catalog.jobs(tool_catalog.bundle("realesrgan-x4"))[0],
                   dest_dir=folder)
        check("missing target", tool_catalog.find_existing(job), None)
        path = tool_catalog.target_path(job)
        open(path, "wb").close()
        info = tool_catalog.find_existing(job)
        check("existing target path", info["path"], path)
        check("existing target reason", info["reason"], "exact")


def test_registered_seedvr2_vae_is_reused():
    with tempfile.TemporaryDirectory() as folder, \
            tempfile.TemporaryDirectory() as other:
        vae = os.path.join(other, "seedvr2_ema_vae_fp16.safetensors")
        open(vae, "wb").close()
        original = tool_catalog.paths.component_pools
        tool_catalog.paths.component_pools = (
            lambda: {"vaes": [{"label": "SeedVR2 VAE", "path": vae}]}
        )
        try:
            job = dict(
                tool_catalog.jobs(tool_catalog.bundle("seedvr2-3b"))[1],
                dest_dir=folder,
            )
            info = tool_catalog.find_existing(job)
        finally:
            tool_catalog.paths.component_pools = original
        check("seedvr2 vae found", info["path"], vae)
        check("seedvr2 vae reason", info["reason"], "variant")


def test_unrelated_vae_is_not_reused():
    with tempfile.TemporaryDirectory() as folder, \
            tempfile.TemporaryDirectory() as other:
        vae = os.path.join(other, "qwen_image_vae.safetensors")
        open(vae, "wb").close()
        original = tool_catalog.paths.component_pools
        tool_catalog.paths.component_pools = (
            lambda: {"vaes": [{"label": "Qwen VAE", "path": vae}]}
        )
        try:
            job = dict(
                tool_catalog.jobs(tool_catalog.bundle("seedvr2-3b"))[1],
                dest_dir=folder,
            )
            info = tool_catalog.find_existing(job)
        finally:
            tool_catalog.paths.component_pools = original
        check("unrelated vae ignored", info, None)


def test_segmentation_jobs_land_in_the_folder_and_archive_is_reused():
    original = tool_catalog.DESTINATIONS["segmentation"]
    with tempfile.TemporaryDirectory() as folder:
        tool_catalog.DESTINATIONS["segmentation"] = lambda: folder
        try:
            job = tool_catalog.jobs(tool_catalog.bundle("mobile-sam"))[0]
            check("sam segmentation folder", job["dest_dir"], folder)
            check("sam pool", job["pool"], "segmentation")
            extracted = os.path.join(folder, "mobile_sam_20230629_quant")
            os.makedirs(extracted)
            open(os.path.join(extracted, "mobile_sam.encoder.quant.onnx"), "wb").close()
            info = tool_catalog.find_existing(job)
            check("sam archive reused", info["reason"], "variant")
        finally:
            tool_catalog.DESTINATIONS["segmentation"] = original


def test_expected_models_are_present():
    ids = {bundle["id"] for bundle in tool_catalog.bundles()}
    for bundle_id in (
        "birefnet-general", "u2net", "u2net-portrait",
        "modnet-portrait", "birefnet-lite", "birefnet-portrait",
        "mobile-sam", "sam-vit-b", "sam-vit-h",
        "realesrgan-x2", "realesrgan-x4",
        "realesrgan-x8", "seedvr2-3b", "seedvr2-7b",
    ):
        check(f"bundle {bundle_id}", bundle_id in ids, True)


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
