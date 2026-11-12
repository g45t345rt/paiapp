"""Offline tests for the model download catalog and hardware recommendation.

Run: .venv\\Scripts\\python.exe tests\\test_model_catalog.py

No network or GPU is used: hardware snapshots are plain dicts, and the catalog's
own consistency checks are exercised directly.
"""

import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import component_kinds  # noqa: E402
from engine import hardware  # noqa: E402
from engine import model_catalog  # noqa: E402

FAILED = []


def check(name, got, want):
    if got != want:
        FAILED.append(f"{name}: got {got!r} want {want!r}")
    else:
        print(f"ok  {name}")


def _hw(gpu=True, vram=12, ram=32):
    return {"gpu": gpu, "vram_gb": vram if gpu else None, "ram_gb": ram}


def test_catalog_is_consistent():
    check("catalog issues", model_catalog.validate(), [])


def test_bundles_match_families():
    families = {family for _group, family in model_catalog.paths.MODEL_GROUPS}
    for bundle in model_catalog.bundles():
        check(f"{bundle['id']} family known", bundle["family"] in families, True)
        check(f"{bundle['id']} has profiles", bool(bundle["profiles"]), True)
        required = {
            slot for slot, _title, _pool in
            component_kinds.FAMILY_SLOTS.get(bundle["family"], ())
        }
        for profile in bundle["profiles"]:
            check(
                f"{bundle['id']}/{profile['id']} slots",
                set(profile["files"]), required | {"model"},
            )


def test_z_image_recommendation():
    bundle = model_catalog.bundle("z-image-turbo")
    rec = model_catalog.recommend(bundle, _hw(vram=12, ram=32))
    check("z-image 12GB tier", rec["tier"], "fp8")
    check("z-image 12GB profile", rec["profile"]["id"], "high")
    rec = model_catalog.recommend(bundle, _hw(vram=8, ram=16))
    check("z-image 8GB tier", rec["tier"], "resident")
    check("z-image 8GB profile", rec["profile"]["id"], "small")


def test_qwen_fp8_tier():
    bundle = model_catalog.bundle("qwen-image-2.1")
    # 32 GB RAM cannot hold the bf16 encoder + bf16 model file set, so Max is
    # rejected even though its fp8 VRAM figure fits; High is the right profile.
    rec = model_catalog.recommend(bundle, _hw(vram=12, ram=32))
    check("qwen 12GB tier", rec["tier"], "fp8")
    check("qwen 12GB profile", rec["profile"]["id"], "high")
    # With enough RAM, the highest-quality profile is viable again.
    rec = model_catalog.recommend(bundle, _hw(vram=12, ram=64))
    check("qwen 12GB/64GB profile", rec["profile"]["id"], "max")
    rec = model_catalog.recommend(bundle, _hw(vram=24, ram=64))
    check("qwen 24GB resident", rec["tier"], "resident")


def test_classify_profile_is_per_profile():
    hw = _hw(vram=12, ram=32)
    z = {
        profile["id"]: profile
        for profile in model_catalog.bundle("z-image-turbo")["profiles"]
    }
    check("z-image high fp8",
          model_catalog.classify_profile(z["high"], hw)["tier"], "fp8")
    check("z-image balanced resident",
          model_catalog.classify_profile(z["balanced"], hw)["tier"],
          "resident")
    qwen = {
        profile["id"]: profile
        for profile in model_catalog.bundle("qwen-image-2.1")["profiles"]
    }
    check("qwen max too-large",
          model_catalog.classify_profile(qwen["max"], hw)["tier"],
          "too-large")
    check("qwen high fp8",
          model_catalog.classify_profile(qwen["high"], hw)["tier"], "fp8")
    check("qwen balanced resident",
          model_catalog.classify_profile(qwen["balanced"], hw)["tier"],
          "resident")
    sdxl = model_catalog.ordered_profiles(
        model_catalog.bundle("sdxl-base"))[0]
    check("sdxl no gpu cpu",
          model_catalog.classify_profile(
              sdxl, _hw(gpu=False, ram=16))["tier"], "cpu")


def test_no_gpu_uses_cpu():
    bundle = model_catalog.bundle("z-image-turbo")
    rec = model_catalog.recommend(bundle, _hw(gpu=False, ram=32))
    check("no gpu tier", rec["tier"], "cpu")
    check("no gpu smallest profile", rec["profile"]["id"], "small")


def test_too_large():
    bundle = model_catalog.bundle("minimax-h3")
    rec = model_catalog.recommend(bundle, _hw(vram=4, ram=8))
    check("h3 too large", rec["tier"], "too-large")


def test_text_encoder_on_gpu_changes_vram_need():
    bundle = model_catalog.bundle("z-image-turbo")
    profile = model_catalog.ordered_profiles(bundle)[0]
    cpu_need = model_catalog.footprint(profile, False)["vram"]
    gpu_need = model_catalog.footprint(profile, True)["vram"]
    check("text on GPU needs more VRAM", gpu_need > cpu_need, True)


def test_jobs_and_destinations():
    bundle = model_catalog.bundle("z-image-turbo")
    profile = model_catalog.ordered_profiles(bundle)[-1]
    jobs = model_catalog.jobs(profile)
    check("one job per file", len(jobs), len(profile["files"]))
    check("model target named", os.path.basename(
        model_catalog.target_path(next(j for j in jobs if j["slot"] == "model")),
    ), "z_image_turbo_nvfp4.safetensors")
    check("vae target named", os.path.basename(
        model_catalog.target_path(next(j for j in jobs if j["slot"] == "vae")),
    ), "ae.safetensors")
    check("every job has a folder", all(j["dest_dir"] for j in jobs), True)


def test_existing_file_and_verify():
    bundle = model_catalog.bundle("z-image-turbo")
    profile = model_catalog.ordered_profiles(bundle)[-1]
    job = next(j for j in model_catalog.jobs(profile) if j["slot"] == "model")
    with tempfile.TemporaryDirectory() as folder:
        job = dict(job, dest_dir=folder)
        check("missing target", model_catalog.existing_path(job), "")
        path = model_catalog.target_path(job)
        open(path, "wb").close()
        check("existing target", model_catalog.existing_path(job), path)
        # An unidentifiable file must not be rejected as the wrong kind.
        unknown = os.path.join(folder, "unknown.safetensors")
        open(unknown, "wb").close()
        detected, ok = model_catalog.verify(unknown, {"kind": "qwen3_06b"})
        check("unknown kind accepted", detected is None, True)
        check("unknown kind passes", ok, True)


def test_turbo_and_base_variants_present():
    ids = {bundle["id"] for bundle in model_catalog.bundles()}
    for bundle_id in (
        "qwen-image-2.1", "qwen-image-2.1-turbo",
        "krea2-turbo", "krea2-raw",
        "moss-image", "moss-image-base",
        "z-image-turbo", "z-image-base",
        "minimax-h3", "minimax-h3-ref2va",
        "sdxl-base", "sdxl-turbo",
        "fibo-2.0-dev", "fibo-2.0-edit-turbo",
    ):
        check(f"bundle {bundle_id}", bundle_id in ids, True)
    check("aesthetic removed", "anima-aesthetic" in ids, False)


def test_every_bundle_has_a_description():
    for bundle in model_catalog.bundles():
        description = bundle.get("description") or {}
        check(f"{bundle['id']} purpose", bool(description.get("purpose")), True)
        check(f"{bundle['id']} strengths", bool(description.get("strengths")), True)
        check(f"{bundle['id']} consider", bool(description.get("consider")), True)


def test_stable_diffusion_is_a_single_file_bundle():
    bundle = model_catalog.bundle("stable-diffusion-1.5")
    check("sd bundle exists", bundle is not None, True)
    for profile in bundle["profiles"]:
        check("sd only has a model", set(profile["files"]), {"model"})
    rec = model_catalog.recommend(bundle, _hw(gpu=False, ram=8))
    check("sd runs on cpu", rec["tier"], "cpu")


def test_turbo_and_non_turbo_differ():
    turbo = model_catalog.bundle("krea2-turbo")
    raw = model_catalog.bundle("krea2-raw")
    turbo_model = model_catalog.ordered_profiles(turbo)[0]["files"]["model"]
    raw_model = model_catalog.ordered_profiles(raw)[0]["files"]["model"]
    check("turbo/raw are different files",
          turbo_model["filename"] != raw_model["filename"], True)


def test_fibo_is_a_remote_repo_reference():
    for bundle_id in ("fibo-2.0-dev", "fibo-2.0-edit-turbo"):
        bundle = model_catalog.bundle(bundle_id)
        profile = model_catalog.ordered_profiles(bundle)[0]
        check(f"{bundle_id} is remote",
              model_catalog.is_remote(profile), True)
        jobs = model_catalog.jobs(profile)
        check(f"{bundle_id} one job", len(jobs), 1)
        check(f"{bundle_id} repo id", jobs[0]["name"],
              "briaai/" + bundle_id.replace("fibo-2.0-", "Fibo-2.0-"))
        check(f"{bundle_id} no target",
              model_catalog.target_path(jobs[0]), "")
        check(f"{bundle_id} not existing",
              model_catalog.existing_path(jobs[0]), "")


def test_stable_diffusion_variants():
    ids = {bundle["id"] for bundle in model_catalog.bundles()}
    for bundle_id in (
        "stable-diffusion-1.5", "stable-diffusion-2.1",
        "sdxl-base", "sdxl-turbo",
    ):
        check(f"sd bundle {bundle_id}", bundle_id in ids, True)
    sdxl = model_catalog.bundle("sdxl-base")
    spec = model_catalog.ordered_profiles(sdxl)[0]["files"]["model"]
    check("sdxl kind", spec["kind"], "sdxl")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "sd_xl_base_1.0.safetensors")
        open(path, "wb").close()
        detected, ok = model_catalog.verify(path, spec)
        check("sdxl verify accepts sd kinds", ok, True)


def test_find_existing_matches_the_same_component():
    """A same-kind file is not reusable unless it is the same component."""
    with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as other:
        seedvr = os.path.join(folder, "seedvr2_ema_vae_fp16.safetensors")
        open(seedvr, "wb").close()
        ae = os.path.join(other, "ae.safetensors")
        open(ae, "wb").close()
        job = {
            "slot": "vae", "kind": "flux_vae", "dest_dir": folder,
            "name": "ae.safetensors", "remote": False,
        }
        original = model_catalog.paths.component_pools
        try:
            model_catalog.paths.component_pools = lambda: {}
            check("unrelated same-kind vae ignored",
                  model_catalog.find_existing(job), None)
            model_catalog.paths.component_pools = (
                lambda: {"vaes": [{"label": "ae", "path": ae}]}
            )
            info = model_catalog.find_existing(job)
        finally:
            model_catalog.paths.component_pools = original
        check("registered vae found", info["path"], ae)
        check("registered vae reason", info["reason"], "variant")
        check("registered vae ok", info["ok"], True)

        exact = dict(job, name="seedvr2_ema_vae_fp16.safetensors")
        info = model_catalog.find_existing(exact)
        check("exact reuse reason", info["reason"], "exact")

        wrong_target = os.path.join(folder, "qwen_image_vae.safetensors")
        open(wrong_target, "wb").close()
        wrong = dict(job, name="qwen_image_vae.safetensors")
        info = model_catalog.find_existing(wrong)
        check("wrong kind not ok", info["ok"], False)


def test_find_existing_matches_quantization_variants():
    with tempfile.TemporaryDirectory() as folder:
        original_entries = model_catalog.paths.model_entries
        original_pools = model_catalog.paths.component_pools
        model_catalog.paths.model_entries = lambda: []
        model_catalog.paths.component_pools = lambda: {}
        try:
            bf16 = os.path.join(folder, "z_image_turbo_bf16.safetensors")
            open(bf16, "wb").close()
            nvfp4_job = {
                "slot": "model", "kind": "z_image", "dest_dir": folder,
                "name": "z_image_turbo_nvfp4.safetensors", "remote": False,
            }
            # A different quantization is a different file: reusing it would
            # make the chosen profile's VRAM figure wrong.
            check("model other quantization ignored",
                  model_catalog.find_existing(nvfp4_job), None)
            same_quant = dict(nvfp4_job, name="z_image_turbo_bf16.safetensors")
            info = model_catalog.find_existing(same_quant)
            check("model same quantization found", info["path"], bf16)

            encoder = os.path.join(folder, "qwen_3_4b.safetensors")
            open(encoder, "wb").close()
            text_job = {
                "slot": "text", "kind": "qwen3_4b", "dest_dir": folder,
                "name": "qwen_3_4b_fp4_mixed.safetensors", "remote": False,
            }
            info = model_catalog.find_existing(text_job)
            check("encoder variant found", info["path"], encoder)
            check("encoder variant reason", info["reason"], "variant")
        finally:
            model_catalog.paths.model_entries = original_entries
            model_catalog.paths.component_pools = original_pools


def test_find_existing_accepts_component_aliases():
    with tempfile.TemporaryDirectory() as folder:
        alias = os.path.join(folder, "flux_vae.safetensors")
        open(alias, "wb").close()
        job = {
            "slot": "vae", "kind": "flux_vae", "dest_dir": folder,
            "name": "ae.safetensors", "remote": False,
        }
        original = model_catalog.paths.component_pools
        model_catalog.paths.component_pools = lambda: {}
        try:
            info = model_catalog.find_existing(job)
        finally:
            model_catalog.paths.component_pools = original
        check("flux vae alias found", info["path"], alias)
        check("flux vae alias reason", info["reason"], "kind")


def test_installed_profile_is_preferred():
    bundle = model_catalog.bundle("z-image-turbo")
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "z_image_turbo_bf16.safetensors")
        open(path, "wb").close()
        original_entries = model_catalog.paths.model_entries
        model_catalog.paths.model_entries = lambda: [{"path": path}]
        try:
            rec = model_catalog.installed_profile(bundle, _hw(vram=12, ram=32))
        finally:
            model_catalog.paths.model_entries = original_entries
        check("installed model profile", rec["profile"]["id"], "high")


def test_hardware_probe_shape():
    result = hardware.probe()
    for key in ("gpu", "gpu_name", "vram_gb", "ram_gb", "backend", "source"):
        check(f"probe has {key}", key in result, True)


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
