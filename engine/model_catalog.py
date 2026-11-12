"""Curated download bundles for complete, compatible model installs.

A generation model is rarely one file: the engine needs the diffusion weights
plus the VAE and text encoder that share its latent space. :data:`BUNDLES` pairs
each family with the exact companion files it needs, so the Model Download
Wizard can fetch a whole working set instead of leaving the user to guess which
VAE or text encoder belongs with a checkpoint.

Each bundle holds one or more *profiles* -- the same model at different
quantizations. Hardware only ever changes which profile is recommended: the VAE
and text-encoder architecture stay fixed, because compatibility is decided by
the family, not by how much VRAM the machine has.

The file sizes are the Hub's LFS sizes (kept here so a hardware recommendation
works offline); the wizard refreshes them from the Hub when it can. Repos are
public and ungated, so the download needs no token.
"""

import os
import re

from . import component_kinds
from . import paths

#: Display title for a bundle file's role.
SLOT_TITLES = {
    "model": "Diffusion model",
    "clip": "CLIP-L",
    "t5": "T5-XXL",
    "text": "Text encoder",
    "vae": "VAE",
}

#: Destination key -> the folder function that resolves it at run time.
DESTINATIONS = {
    "checkpoints": paths.checkpoints_dir,
    "diffusion": paths.diffusion_dir,
    "unet": paths.unet_dir,
    "text_encoders": paths.text_encoders_dir,
    "vae": paths.vae_dir,
    "loras": paths.loras_dir,
    "adapters": paths.adapters_dir,
    "controlnet": paths.controlnet_dir,
}


def destination(name):
    """The folder a destination key names (the models root when unknown)."""
    fn = DESTINATIONS.get(name)
    return fn() if fn is not None else paths.models_root()


def _f(repo, filename, save_as, slot, kind, dest, size, fp8=False):
    """One downloadable file in a bundle profile."""
    return {
        "repo": repo,
        "filename": filename,
        "save_as": save_as,
        "slot": slot,
        "kind": kind,
        "dest": dest,
        "bytes": int(size),
        # True when the engine can additionally store this model in fp8 at load
        # time (a bf16 safetensors DiT). Already-quantized files are not halved
        # again.
        "fp8": bool(fp8),
    }


def _profile(profile_id, label, quality, files):
    return {
        "id": profile_id,
        "label": label,
        "quality": int(quality),
        "files": {spec["slot"]: spec for spec in files},
    }


def _remote(repo, kind, size):
    """A model that is referenced by Hub repo id instead of downloaded.

    Fibo 2.0 is loaded by ``diffusers`` as a whole repository, so there is no
    single file to save: the row keeps the repo id and the pipeline fetches it on
    first use. This also avoids a duplicate copy in the models folder.
    """
    return {
        "repo": repo,
        "filename": "",
        "save_as": "",
        "slot": "model",
        "kind": kind,
        "dest": "diffusion",
        "bytes": int(size),
        "fp8": False,
        "remote": True,
    }


# ------------------------------------------------------------------ file specs

_ANIMA_CLIP = _f(
    "circlestone-labs/Anima",
    "split_files/text_encoders/qwen_3_06b_base.safetensors",
    "qwen_3_06b_base.safetensors",
    "clip", "qwen3_06b", "text_encoders", 1192135096,
)
_ANIMA_VAE = _f(
    "circlestone-labs/Anima",
    "split_files/vae/qwen_image_vae.safetensors",
    "qwen_image_vae.safetensors",
    "vae", "qwen_image_vae", "vae", 253806246,
)
_ANIMA_BASE = _f(
    "circlestone-labs/Anima",
    "split_files/diffusion_models/anima-base-v1.0.safetensors",
    "anima-base-v1.0.safetensors",
    "model", "anima", "diffusion", 4182218328, fp8=True,
)
_ANIMA_TURBO = _f(
    "circlestone-labs/Anima",
    "split_files/diffusion_models/anima-turbo-v1.1.safetensors",
    "anima-turbo-v1.1.safetensors",
    "model", "anima", "diffusion", 4182230656, fp8=True,
)

_QWEN_REPO = "Comfy-Org/Qwen-Image-2.1"
_QWEN_MODEL_BF16 = _f(
    _QWEN_REPO, "diffusion_models/qwen_image_2.1_bf16.safetensors",
    "qwen_image_2.1_bf16.safetensors", "model", "qwen_image", "diffusion",
    14230280616, fp8=True,
)
_QWEN_MODEL_INT8 = _f(
    _QWEN_REPO, "diffusion_models/qwen_image_2.1_int8_convrot.safetensors",
    "qwen_image_2.1_int8_convrot.safetensors", "model", "qwen_image",
    "diffusion", 7256783064,
)
_QWEN_TEXT_BF16 = _f(
    _QWEN_REPO, "text_encoders/qwen3vl_8b_bf16.safetensors",
    "qwen3vl_8b_bf16.safetensors", "text", "qwen3vl_8b", "text_encoders",
    17534334616,
)
_QWEN_TEXT_INT8 = _f(
    _QWEN_REPO, "text_encoders/qwen3vl_8b_int8_convrot.safetensors",
    "qwen3vl_8b_int8_convrot.safetensors", "text", "qwen3vl_8b",
    "text_encoders", 9350798360,
)
_QWEN_TEXT_W4A8 = _f(
    _QWEN_REPO, "text_encoders/qwen3vl_8b_w4a8.safetensors",
    "qwen3vl_8b_w4a8.safetensors", "text", "qwen3vl_8b", "text_encoders",
    6312105364,
)
_QWEN_VAE = _f(
    _QWEN_REPO, "vae/qwen_image_2.1_vae_bf16.safetensors",
    "qwen_image_2.1_vae_bf16.safetensors", "vae", "qwen_image_21_vae", "vae",
    675509688,
)

_Z_REPO = "Comfy-Org/Z-Image-Turbo"
_Z_MODEL_BF16 = _f(
    _Z_REPO, "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "z_image_turbo_bf16.safetensors", "model", "z_image", "diffusion",
    12309866400, fp8=True,
)
_Z_MODEL_INT8 = _f(
    _Z_REPO, "split_files/diffusion_models/z_image_turbo_int8_convrot.safetensors",
    "z_image_turbo_int8_convrot.safetensors", "model", "z_image", "diffusion",
    6201001296,
)
_Z_MODEL_NVFP4 = _f(
    _Z_REPO, "split_files/diffusion_models/z_image_turbo_nvfp4.safetensors",
    "z_image_turbo_nvfp4.safetensors", "model", "z_image", "diffusion",
    4509509600,
)
_Z_TEXT_BF16 = _f(
    _Z_REPO, "split_files/text_encoders/qwen_3_4b.safetensors",
    "qwen_3_4b.safetensors", "text", "qwen3_4b", "text_encoders", 8044982048,
)
_Z_TEXT_FP8 = _f(
    _Z_REPO, "split_files/text_encoders/qwen_3_4b_fp8_mixed.safetensors",
    "qwen_3_4b_fp8_mixed.safetensors", "text", "qwen3_4b", "text_encoders",
    5631994051,
)
_Z_TEXT_FP4 = _f(
    _Z_REPO, "split_files/text_encoders/qwen_3_4b_fp4_mixed.safetensors",
    "qwen_3_4b_fp4_mixed.safetensors", "text", "qwen3_4b", "text_encoders",
    3479416193,
)
_Z_VAE = _f(
    _Z_REPO, "split_files/vae/ae.safetensors", "ae.safetensors",
    "vae", "flux_vae", "vae", 335304388,
)

_KREA_REPO = "Comfy-Org/Krea-2"
_KREA_MODEL_BF16 = _f(
    _KREA_REPO, "diffusion_models/krea2_turbo_bf16.safetensors",
    "krea2_turbo_bf16.safetensors", "model", "krea2", "diffusion",
    26283332608, fp8=True,
)
_KREA_MODEL_FP8 = _f(
    _KREA_REPO, "diffusion_models/krea2_turbo_fp8_scaled.safetensors",
    "krea2_turbo_fp8_scaled.safetensors", "model", "krea2", "diffusion",
    13141730784,
)
_KREA_MODEL_INT8 = _f(
    _KREA_REPO, "diffusion_models/krea2_turbo_int8_convrot.safetensors",
    "krea2_turbo_int8_convrot.safetensors", "model", "krea2", "diffusion",
    13492686496,
)
_KREA_MODEL_NVFP4 = _f(
    _KREA_REPO, "diffusion_models/krea2_turbo_nvfp4.safetensors",
    "krea2_turbo_nvfp4.safetensors", "model", "krea2", "diffusion",
    7673668448,
)
_KREA_TEXT_BF16 = _f(
    _KREA_REPO, "text_encoders/qwen3vl_4b_bf16.safetensors",
    "qwen3vl_4b_bf16.safetensors", "text", "qwen3vl_4b", "text_encoders",
    8875719384,
)
_KREA_TEXT_FP8 = _f(
    _KREA_REPO, "text_encoders/qwen3vl_4b_fp8_scaled.safetensors",
    "qwen3vl_4b_fp8_scaled.safetensors", "text", "qwen3vl_4b",
    "text_encoders", 5242467968,
)
_KREA_TEXT_INT8 = _f(
    _KREA_REPO, "text_encoders/qwen3vl_4b_int8_convrot.safetensors",
    "qwen3vl_4b_int8_convrot.safetensors", "text", "qwen3vl_4b",
    "text_encoders", 4858559240,
)
_KREA_VAE = _f(
    _KREA_REPO, "vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors",
    "vae", "qwen_image_vae", "vae", 253806246,
)

_MOSS_REPO = "Comfy-Org/MOSS-Image"
_MOSS_MODEL_BF16 = _f(
    _MOSS_REPO, "diffusion_models/moss_image_sft_bf16.safetensors",
    "moss_image_sft_bf16.safetensors", "model", "moss_image", "diffusion",
    10164655808, fp8=True,
)
_MOSS_MODEL_INT8 = _f(
    _MOSS_REPO, "diffusion_models/moss_image_sft_int8_convrot.safetensors",
    "moss_image_sft_int8_convrot.safetensors", "model", "moss_image",
    "diffusion", 5413698958,
)
_MOSS_TEXT_BF16 = _f(
    _MOSS_REPO, "text_encoders/qwen3.5_9b_bf16.safetensors",
    "qwen3.5_9b_bf16.safetensors", "text", "qwen3_5_9b", "text_encoders",
    19306312328,
)
_MOSS_TEXT_INT8 = _f(
    _MOSS_REPO, "text_encoders/qwen3.5_9b_int8_convrot.safetensors",
    "qwen3.5_9b_int8_convrot.safetensors", "text", "qwen3_5_9b",
    "text_encoders", 9748121532,
)
_MOSS_TEXT_W4A8 = _f(
    _MOSS_REPO, "text_encoders/qwen3.5_9b_w4a8_gptq.safetensors",
    "qwen3.5_9b_w4a8_gptq.safetensors", "text", "qwen3_5_9b",
    "text_encoders", 7343591088,
)
_MOSS_VAE = _f(
    _MOSS_REPO, "vae/flux2-vae_small.safetensors",
    "flux2-vae_small.safetensors", "vae", "flux2_vae", "vae", 249521340,
)

_H3_REPO = "Comfy-Org/MiniMax-H3"
_H3_MODEL_BF16 = _f(
    _H3_REPO, "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors",
    "minimax_h3_fl2va_pruned_bf16.safetensors", "model", "h3", "diffusion",
    40225724176, fp8=True,
)
_H3_MODEL_FP8 = _f(
    _H3_REPO, "diffusion_models/minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
    "minimax_h3_fl2va_pruned_fp8_scaled.safetensors", "model", "h3",
    "diffusion", 20958205608,
)
_H3_MODEL_INT8 = _f(
    _H3_REPO, "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
    "minimax_h3_fl2va_pruned_int8_convrot.safetensors", "model", "h3",
    "diffusion", 20970379616,
)
_H3_TEXT_BF16 = _f(
    _H3_REPO, "text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors",
    "qwen3vl_32b_minimax_h3_bf16.safetensors", "text", "qwen3vl_32b",
    "text_encoders", 51506295256,
)
_H3_TEXT_INT8 = _f(
    _H3_REPO, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text", "qwen3vl_32b",
    "text_encoders", 27141342152,
)
_H3_TEXT_NVFP4 = _f(
    _H3_REPO, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text", "qwen3vl_32b",
    "text_encoders", 15687142551,
)
_H3_VAE_INT8 = _f(
    _H3_REPO, "vae/minimax_h3_video_vae_int8_convrot.safetensors",
    "minimax_h3_video_vae_int8_convrot.safetensors", "vae", "h3_video_vae",
    "vae", 2811065184,
)
_H3_VAE_FP16 = _f(
    _H3_REPO, "vae/minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_video_vae_fp16.safetensors", "vae", "h3_video_vae", "vae",
    5207808496,
)

_H3I_REPO = "Comfy-Org/MiniMax-H3-Image"
_H3I_MODEL_BF16 = _f(
    _H3I_REPO, "diffusion_models/minimax_h3_image_pruned_bf16.safetensors",
    "minimax_h3_image_pruned_bf16.safetensors", "model", "h3_image",
    "diffusion", 40234657162, fp8=True,
)
_H3I_MODEL_INT8 = _f(
    _H3I_REPO, "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
    "minimax_h3_image_pruned_int8_convrot.safetensors", "model", "h3_image",
    "diffusion", 20979314802,
)
_H3I_MODEL_W4A8 = _f(
    _H3I_REPO, "diffusion_models/minimax_h3_image_pruned_w4a8.safetensors",
    "minimax_h3_image_pruned_w4a8.safetensors", "model", "h3_image",
    "diffusion", 12549794530,
)
_H3I_TEXT_INT8 = _f(
    _H3I_REPO, "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
    "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "text", "qwen3vl_32b",
    "text_encoders", 27141342152,
)
_H3I_TEXT_NVFP4 = _f(
    _H3I_REPO, "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "text", "qwen3vl_32b",
    "text_encoders", 15687142551,
)
_H3I_VAE = _f(
    _H3I_REPO, "vae/minimax_h3_image_vae_fp16.safetensors",
    "minimax_h3_image_vae_fp16.safetensors", "vae", "h3_image_vae", "vae",
    1168505312,
)


# --- additional variants -----------------------------------------------------

_QWEN_MODEL_TURBO_BF16 = _f(
    _QWEN_REPO, "diffusion_models/qwen_image_2.1_turbo_bf16.safetensors",
    "qwen_image_2.1_turbo_bf16.safetensors", "model", "qwen_image",
    "diffusion", 14230280616, fp8=True,
)
_QWEN_MODEL_TURBO_INT8 = _f(
    _QWEN_REPO, "diffusion_models/qwen_image_2.1_turbo_int8_convrot.safetensors",
    "qwen_image_2.1_turbo_int8_convrot.safetensors", "model", "qwen_image",
    "diffusion", 7256787829,
)

_Z_BASE_BF16 = _f(
    "Comfy-Org/Z-Image", "split_files/diffusion_models/z_image_bf16.safetensors",
    "z_image_bf16.safetensors", "model", "z_image", "diffusion",
    12309866400, fp8=True,
)
_Z_BASE_INT8 = _f(
    "Comfy-Org/Z-Image",
    "split_files/diffusion_models/z_image_int8_convrot.safetensors",
    "z_image_int8_convrot.safetensors", "model", "z_image", "diffusion",
    6201001296,
)

_KREA_RAW_BF16 = _f(
    _KREA_REPO, "diffusion_models/krea2_raw_bf16.safetensors",
    "krea2_raw_bf16.safetensors", "model", "krea2", "diffusion",
    26283332608, fp8=True,
)
_KREA_RAW_FP8 = _f(
    _KREA_REPO, "diffusion_models/krea2_raw_fp8_scaled.safetensors",
    "krea2_raw_fp8_scaled.safetensors", "model", "krea2", "diffusion",
    13141730784,
)
_KREA_RAW_INT8 = _f(
    _KREA_REPO, "diffusion_models/krea2_raw_int8_convrot.safetensors",
    "krea2_raw_int8_convrot.safetensors", "model", "krea2", "diffusion",
    13492686496,
)

_MOSS_BASE_BF16 = _f(
    _MOSS_REPO, "diffusion_models/moss_image_base_bf16.safetensors",
    "moss_image_base_bf16.safetensors", "model", "moss_image", "diffusion",
    10164655808, fp8=True,
)
_MOSS_BASE_INT8 = _f(
    _MOSS_REPO, "diffusion_models/moss_image_base_int8_convrot.safetensors",
    "moss_image_base_int8_convrot.safetensors", "model", "moss_image",
    "diffusion", 5413698958,
)

_H3_REF_BF16 = _f(
    _H3_REPO, "diffusion_models/minimax_h3_ref2va_pruned_bf16.safetensors",
    "minimax_h3_ref2va_pruned_bf16.safetensors", "model", "h3", "diffusion",
    40225724176, fp8=True,
)
_H3_REF_FP8 = _f(
    _H3_REPO, "diffusion_models/minimax_h3_ref2va_pruned_fp8_scaled.safetensors",
    "minimax_h3_ref2va_pruned_fp8_scaled.safetensors", "model", "h3",
    "diffusion", 20958205608,
)
_H3_REF_INT8 = _f(
    _H3_REPO, "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors", "model", "h3",
    "diffusion", 20970379616,
)

_FIBO_DEV = _remote("briaai/Fibo-2.0-dev", "fibo", 25670000000)
_FIBO_EDIT_TURBO = _remote(
    "briaai/Fibo-2.0-edit-turbo", "fibo", 25670000000,
)

_SD15 = _f(
    "Comfy-Org/stable-diffusion-1.5", "v1-5-pruned-emaonly-fp16.safetensors",
    "v1-5-pruned-emaonly-fp16.safetensors", "model", "sd15", "checkpoints",
    2132696762,
)
_SD21 = _f(
    "Comfy-Org/stable-diffusion-2.1", "v2-1_768-ema-pruned.safetensors",
    "v2-1_768-ema-pruned.safetensors", "model", "stable_diffusion",
    "checkpoints", 5214604494,
)
_SDXL = _f(
    "stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors",
    "sd_xl_base_1.0.safetensors", "model", "sdxl", "checkpoints", 6938078334,
)
_SDXL_TURBO = _f(
    "stabilityai/sdxl-turbo", "sd_xl_turbo_1.0_fp16.safetensors",
    "sd_xl_turbo_1.0_fp16.safetensors", "model", "sdxl", "checkpoints",
    6938081905,
)


# -------------------------------------------------------------------- bundles

def _bundle(bundle_id, label, family, notes, profiles):
    group = next(
        (name for name, value in paths.MODEL_GROUPS if value == family), family,
    )
    return {
        "id": bundle_id,
        "label": label,
        "group": group,
        "family": family,
        "notes": notes,
        "profiles": tuple(profiles),
    }


BUNDLES = (
    # Anima ----------------------------------------------------------------
    _bundle(
        "anima-turbo", "Anima Turbo v1.1", "anima",
        "The few-step Anima checkpoint, tuned for fast iteration.",
        (
            _profile("standard", "Turbo v1.1 (bf16)", 3,
                     (_ANIMA_TURBO, _ANIMA_CLIP, _ANIMA_VAE)),
        ),
    ),
    _bundle(
        "anima-base", "Anima Base v1.0", "anima",
        "The base Anima checkpoint, better for custom styles and LoRAs.",
        (
            _profile("standard", "Base v1.0 (bf16)", 3,
                     (_ANIMA_BASE, _ANIMA_CLIP, _ANIMA_VAE)),
        ),
    ),
    # Qwen-Image -----------------------------------------------------------
    _bundle(
        "qwen-image-2.1", "Qwen-Image 2.1", "qwen_image",
        "The full Qwen-Image 2.1 model with the Qwen3-VL-8B encoder.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_QWEN_MODEL_BF16, _QWEN_TEXT_BF16, _QWEN_VAE)),
            _profile("high", "bf16 model + int8 encoder", 3,
                     (_QWEN_MODEL_BF16, _QWEN_TEXT_INT8, _QWEN_VAE)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_QWEN_MODEL_INT8, _QWEN_TEXT_INT8, _QWEN_VAE)),
            _profile("small", "int8 model + w4a8 encoder", 1,
                     (_QWEN_MODEL_INT8, _QWEN_TEXT_W4A8, _QWEN_VAE)),
        ),
    ),
    _bundle(
        "qwen-image-2.1-turbo", "Qwen-Image 2.1 Turbo", "qwen_image",
        "The distilled few-step Qwen-Image 2.1 for faster drafts.",
        (
            _profile("max", "bf16 turbo + bf16 encoder", 4,
                     (_QWEN_MODEL_TURBO_BF16, _QWEN_TEXT_BF16, _QWEN_VAE)),
            _profile("high", "bf16 turbo + int8 encoder", 3,
                     (_QWEN_MODEL_TURBO_BF16, _QWEN_TEXT_INT8, _QWEN_VAE)),
            _profile("balanced", "int8 turbo + int8 encoder", 2,
                     (_QWEN_MODEL_TURBO_INT8, _QWEN_TEXT_INT8, _QWEN_VAE)),
            _profile("small", "int8 turbo + w4a8 encoder", 1,
                     (_QWEN_MODEL_TURBO_INT8, _QWEN_TEXT_W4A8, _QWEN_VAE)),
        ),
    ),
    # Z-Image --------------------------------------------------------------
    _bundle(
        "z-image-turbo", "Z-Image Turbo", "z_image",
        "The compact few-step Z-Image model.",
        (
            _profile("high", "bf16 model + bf16 encoder", 3,
                     (_Z_MODEL_BF16, _Z_TEXT_BF16, _Z_VAE)),
            _profile("balanced", "int8 model + fp8 encoder", 2,
                     (_Z_MODEL_INT8, _Z_TEXT_FP8, _Z_VAE)),
            _profile("small", "nvfp4 model + fp4 encoder", 1,
                     (_Z_MODEL_NVFP4, _Z_TEXT_FP4, _Z_VAE)),
        ),
    ),
    _bundle(
        "z-image-base", "Z-Image Base", "z_image",
        "The non-distilled Z-Image model, tuned with CFG and more steps.",
        (
            _profile("high", "bf16 model + bf16 encoder", 3,
                     (_Z_BASE_BF16, _Z_TEXT_BF16, _Z_VAE)),
            _profile("balanced", "int8 model + fp8 encoder", 2,
                     (_Z_BASE_INT8, _Z_TEXT_FP8, _Z_VAE)),
        ),
    ),
    # Krea 2 ---------------------------------------------------------------
    _bundle(
        "krea2-turbo", "Krea 2 Turbo", "krea2",
        "The few-step Krea 2 model for design and illustration.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_KREA_MODEL_BF16, _KREA_TEXT_BF16, _KREA_VAE)),
            _profile("high", "fp8 model + fp8 encoder", 3,
                     (_KREA_MODEL_FP8, _KREA_TEXT_FP8, _KREA_VAE)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_KREA_MODEL_INT8, _KREA_TEXT_INT8, _KREA_VAE)),
            _profile("small", "nvfp4 model + int8 encoder", 1,
                     (_KREA_MODEL_NVFP4, _KREA_TEXT_INT8, _KREA_VAE)),
        ),
    ),
    _bundle(
        "krea2-raw", "Krea 2 Raw", "krea2",
        "The non-distilled Krea 2 base model with more style range.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_KREA_RAW_BF16, _KREA_TEXT_BF16, _KREA_VAE)),
            _profile("high", "fp8 model + fp8 encoder", 3,
                     (_KREA_RAW_FP8, _KREA_TEXT_FP8, _KREA_VAE)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_KREA_RAW_INT8, _KREA_TEXT_INT8, _KREA_VAE)),
        ),
    ),
    # MOSS -----------------------------------------------------------------
    _bundle(
        "moss-image", "MOSS Image SFT", "moss_image",
        "The instruction-tuned MOSS Image model.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_MOSS_MODEL_BF16, _MOSS_TEXT_BF16, _MOSS_VAE)),
            _profile("high", "bf16 model + int8 encoder", 3,
                     (_MOSS_MODEL_BF16, _MOSS_TEXT_INT8, _MOSS_VAE)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_MOSS_MODEL_INT8, _MOSS_TEXT_INT8, _MOSS_VAE)),
            _profile("small", "int8 model + w4a8 encoder", 1,
                     (_MOSS_MODEL_INT8, _MOSS_TEXT_W4A8, _MOSS_VAE)),
        ),
    ),
    _bundle(
        "moss-image-base", "MOSS Image Base", "moss_image",
        "The MOSS Image base weights, suited to finetuning.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_MOSS_BASE_BF16, _MOSS_TEXT_BF16, _MOSS_VAE)),
            _profile("high", "bf16 model + int8 encoder", 3,
                     (_MOSS_BASE_BF16, _MOSS_TEXT_INT8, _MOSS_VAE)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_MOSS_BASE_INT8, _MOSS_TEXT_INT8, _MOSS_VAE)),
        ),
    ),
    # MiniMax H3 -----------------------------------------------------------
    _bundle(
        "minimax-h3", "MiniMax H3 (FL2VA)", "h3",
        "First/last-frame to video with audio.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_H3_MODEL_BF16, _H3_TEXT_BF16, _H3_VAE_FP16)),
            _profile("high", "fp8 model + int8 encoder", 3,
                     (_H3_MODEL_FP8, _H3_TEXT_INT8, _H3_VAE_FP16)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_H3_MODEL_INT8, _H3_TEXT_INT8, _H3_VAE_INT8)),
            _profile("small", "int8 model + nvfp4 encoder", 1,
                     (_H3_MODEL_INT8, _H3_TEXT_NVFP4, _H3_VAE_INT8)),
        ),
    ),
    _bundle(
        "minimax-h3-ref2va", "MiniMax H3 (Ref2VA)", "h3",
        "Reference-image to video with audio.",
        (
            _profile("max", "bf16 model + bf16 encoder", 4,
                     (_H3_REF_BF16, _H3_TEXT_BF16, _H3_VAE_FP16)),
            _profile("high", "fp8 model + int8 encoder", 3,
                     (_H3_REF_FP8, _H3_TEXT_INT8, _H3_VAE_FP16)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_H3_REF_INT8, _H3_TEXT_INT8, _H3_VAE_INT8)),
            _profile("small", "int8 model + nvfp4 encoder", 1,
                     (_H3_REF_INT8, _H3_TEXT_NVFP4, _H3_VAE_INT8)),
        ),
    ),
    _bundle(
        "minimax-h3-image", "MiniMax H3 Image", "h3_image",
        "The 2D image model from the H3 family.",
        (
            _profile("high", "bf16 model + int8 encoder", 3,
                     (_H3I_MODEL_BF16, _H3I_TEXT_INT8, _H3I_VAE)),
            _profile("balanced", "int8 model + int8 encoder", 2,
                     (_H3I_MODEL_INT8, _H3I_TEXT_INT8, _H3I_VAE)),
            _profile("small", "w4a8 model + nvfp4 encoder", 1,
                     (_H3I_MODEL_W4A8, _H3I_TEXT_NVFP4, _H3I_VAE)),
        ),
    ),
    # Fibo 2.0 -------------------------------------------------------------
    _bundle(
        "fibo-2.0-dev", "Fibo 2.0 Dev", "fibo",
        "Bria's full multi-reference image model, loaded from its Hub repo.",
        (
            _profile("standard", "Diffusers repo (fetched on first use)", 3,
                     (_FIBO_DEV,)),
        ),
    ),
    _bundle(
        "fibo-2.0-edit-turbo", "Fibo 2.0 Edit Turbo", "fibo",
        "The few-step Fibo editing model, loaded from its Hub repo.",
        (
            _profile("standard", "Diffusers repo (fetched on first use)", 3,
                     (_FIBO_EDIT_TURBO,)),
        ),
    ),
    # Stable Diffusion -----------------------------------------------------
    _bundle(
        "stable-diffusion-1.5", "Stable Diffusion 1.5", "stable_diffusion",
        "The classic lightweight checkpoint with a huge ecosystem.",
        (
            _profile("standard", "FP16 pruned (2.1 GB)", 2, (_SD15,)),
        ),
    ),
    _bundle(
        "stable-diffusion-2.1", "Stable Diffusion 2.1", "stable_diffusion",
        "The 768px OpenCLIP checkpoint, between 1.5 and SDXL in quality.",
        (
            _profile("standard", "768 EMA pruned (5.2 GB)", 2, (_SD21,)),
        ),
    ),
    _bundle(
        "sdxl-base", "SDXL Base 1.0", "stable_diffusion",
        "The full-quality SDXL checkpoint with two text encoders.",
        (
            _profile("standard", "FP16 checkpoint (6.9 GB)", 3, (_SDXL,)),
        ),
    ),
    _bundle(
        "sdxl-turbo", "SDXL Turbo", "stable_diffusion",
        "The one-to-four-step SDXL variant for fast drafts.",
        (
            _profile("standard", "FP16 checkpoint (6.9 GB)", 3, (_SDXL_TURBO,)),
        ),
    ),
)


#: Purpose, strengths and trade-offs shown by the wizard for each bundle.
DESCRIPTIONS = {
    "anima-turbo": {
        "purpose": (
            "Anime and illustration generation tuned for few-step sampling."
        ),
        "strengths": (
            "Fast enough for rapid previews; clean anime line work and colors.",
            "Small footprint next to the large video models.",
        ),
        "consider": (
            "Less fine detail than the base model; prompt changes are subtle.",
            "Anime-first, so photoreal output needs heavy styling.",
        ),
    },
    "anima-base": {
        "purpose": (
            "The base Anima checkpoint for custom styles, LoRAs and refinement."
        ),
        "strengths": (
            "More flexible than Turbo when LoRAs or style prompts are used.",
            "Same text encoder and VAE, so switching to Turbo is cheap.",
        ),
        "consider": (
            "Needs more steps and CFG tuning than Turbo.",
            "Similar download size to Turbo.",
        ),
    },
    "qwen-image-2.1": {
        "purpose": (
            "High-resolution image generation with strong prompt following and "
            "legible text rendering."
        ),
        "strengths": (
            "Best-in-class text and layout; handles long, structured prompts.",
            "Strong bilingual (English/Chinese) understanding.",
        ),
        "consider": (
            "Large model and 8B text encoder; 12 GB VRAM is a tight fit.",
            "Slower per image than distilled variants.",
        ),
    },
    "qwen-image-2.1-turbo": {
        "purpose": (
            "The distilled Qwen-Image 2.1 for fast drafts and iteration."
        ),
        "strengths": (
            "Far fewer sampling steps for the same composition.",
            "Keeps the Qwen text-rendering strengths.",
        ),
        "consider": (
            "Lower fine detail and text fidelity than the full model.",
            "Same large text encoder and VAE download.",
        ),
    },
    "z-image-turbo": {
        "purpose": (
            "Compact, very fast text-to-image generation."
        ),
        "strengths": (
            "Small enough for modest GPUs; excellent speed.",
            "Good realism for its size.",
        ),
        "consider": (
            "Weaker text rendering and fine detail than larger families.",
            "Few-step settings are tuned for Turbo; avoid base-style CFG.",
        ),
    },
    "z-image-base": {
        "purpose": (
            "The non-distilled Z-Image model for more control and refinement."
        ),
        "strengths": (
            "Responds to CFG and extra steps for higher-quality output.",
            "Shares the Turbo text encoder and VAE.",
        ),
        "consider": (
            "Much slower than Turbo and needs sampler/CFG tuning.",
            "Only int8/bf16 profiles are published.",
        ),
    },
    "krea2-turbo": {
        "purpose": (
            "Design, illustration and stylized imagery with Krea's aesthetic."
        ),
        "strengths": (
            "Strong style control and graphic-design output.",
            "Few-step Turbo inference.",
        ),
        "consider": (
            "Very large in bf16; the 4B text encoder adds to memory use.",
            "Style is opinionated and can fight neutral prompts.",
        ),
    },
    "krea2-raw": {
        "purpose": (
            "The non-distilled Krea 2 base model for broader style range."
        ),
        "strengths": (
            "More flexibility and diversity than Turbo.",
            "Good starting point for LoRAs and finetunes.",
        ),
        "consider": (
            "Needs more steps and CFG; slower than Turbo.",
            "Large download in every profile.",
        ),
    },
    "moss-image": {
        "purpose": (
            "Instruction-following image generation with fine-grained prompt "
            "control."
        ),
        "strengths": (
            "Strong prompt following and detail for its class.",
            "The SFT weights are the polished, ready-to-use release.",
        ),
        "consider": (
            "Very large download: the Qwen3.5-9B encoder dominates.",
            "The small FLUX.2 VAE is required; the full VAE will not work.",
        ),
    },
    "moss-image-base": {
        "purpose": (
            "MOSS Image base weights for finetuning and custom LoRAs."
        ),
        "strengths": (
            "More flexible than SFT as a starting point.",
            "Shares the SFT text encoder and small VAE.",
        ),
        "consider": (
            "Less polished out of the box; expects tuning.",
            "Same large encoder download as SFT.",
        ),
    },
    "minimax-h3": {
        "purpose": (
            "Generate short video clips with synchronized audio from a first "
            "and last frame."
        ),
        "strengths": (
            "Joint video and audio generation.",
            "Strong motion and camera control from the endpoint frames.",
        ),
        "consider": (
            "Enormous: even the smallest profile streams from disk.",
            "High-end GPU and lots of RAM are effectively required.",
        ),
    },
    "minimax-h3-ref2va": {
        "purpose": (
            "Generate video with synchronized audio from a reference image."
        ),
        "strengths": (
            "Keeps a reference subject consistent across the clip.",
            "Same text encoder and VAE as the FL2VA model.",
        ),
        "consider": (
            "Just as large as FL2VA; disk streaming is likely.",
            "Reference conditioning can constrain creative motion.",
        ),
    },
    "minimax-h3-image": {
        "purpose": (
            "High-resolution 2D image generation from the H3 family."
        ),
        "strengths": (
            "Strong detail and composition; shares the H3 text encoder.",
            "Much smaller than the H3 video models.",
        ),
        "consider": (
            "The 32B text encoder still makes it a heavy download.",
            "Not a video model despite the H3 name.",
        ),
    },
    "stable-diffusion-1.5": {
        "purpose": (
            "Lightweight 512px generation and the widest LoRA/ControlNet "
            "ecosystem."
        ),
        "strengths": (
            "Runs on almost any GPU; fast and cheap.",
            "Huge library of community checkpoints, LoRAs and tools.",
        ),
        "consider": (
            "Dated image quality and weak text rendering.",
            "Best at 512px; larger sizes need upscaling or extensions.",
        ),
    },
    "stable-diffusion-2.1": {
        "purpose": (
            "768px generation with an OpenCLIP text encoder, better prompt "
            "understanding than 1.5."
        ),
        "strengths": (
            "Sharper and more accurate than SD 1.5.",
            "Still smaller and faster than SDXL.",
        ),
        "consider": (
            "The 2.1 ecosystem is much smaller than 1.5 or SDXL.",
            "Weak text rendering compared with modern models.",
        ),
    },
    "sdxl-base": {
        "purpose": (
            "1024px high-detail generation with the largest Stable Diffusion "
            "ecosystem."
        ),
        "strengths": (
            "Excellent detail, lighting and composition at 1024px.",
            "Two text encoders and a vast library of LoRAs/ControlNets.",
        ),
        "consider": (
            "About 7 GB per checkpoint.",
            "Slower than SD 1.5 and needs more VRAM.",
        ),
    },
    "sdxl-turbo": {
        "purpose": (
            "The distilled SDXL variant for one-to-four-step drafts."
        ),
        "strengths": (
            "Very fast, with SDXL proportions and detail.",
            "Great for exploring compositions before a full render.",
        ),
        "consider": (
            "Lower prompt fidelity and detail than SDXL Base.",
            "Uses fixed low guidance; it does not behave like a normal SDXL.",
        ),
    },
    "fibo-2.0-dev": {
        "purpose": (
            "Bria's full multi-reference image generation and editing model."
        ),
        "strengths": (
            "Strong structured prompts, multi-reference editing and inpainting.",
            "Loads through diffusers; no duplicate file copy in the models folder.",
        ),
        "consider": (
            "Fetched on first use from a gated repo: accept the license and set "
            "HF_TOKEN.",
            "About 26 GB of weights; needs a large GPU or heavy offload.",
        ),
    },
    "fibo-2.0-edit-turbo": {
        "purpose": (
            "The few-step Fibo editing model for fast reference-guided edits."
        ),
        "strengths": (
            "Much faster than the full Fibo dev weights.",
            "Same diffusers loading, so no duplicate download folder.",
        ),
        "consider": (
            "Gated repo: accept the license and set HF_TOKEN before first use.",
            "Distilled, so subtle detail and prompt nuance are reduced.",
        ),
    },
}

for _item in BUNDLES:
    _item["description"] = DESCRIPTIONS.get(_item["id"], {})


# ------------------------------------------------------------------ queries

def bundles():
    """Every curated bundle, in display order."""
    return list(BUNDLES)


def bundle(bundle_id):
    """The bundle with ``bundle_id``, or ``None``."""
    for item in BUNDLES:
        if item["id"] == bundle_id:
            return item
    return None


def for_family(family):
    """Every bundle for a picker family."""
    return [item for item in BUNDLES if item["family"] == family]


def ordered_profiles(bundle):
    """A bundle's profiles, best quality first."""
    return sorted(
        bundle["profiles"], key=lambda profile: profile["quality"], reverse=True,
    )


def profile(bundle, profile_id):
    """One profile by id, or the bundle's best profile."""
    for item in bundle["profiles"]:
        if item["id"] == profile_id:
            return item
    return ordered_profiles(bundle)[0]


def total_bytes(profile):
    """The on-disk size of every file in a profile."""
    return sum(file["bytes"] for file in profile["files"].values())


def footprint(profile, text_on_gpu=False):
    """Approximate load needs in bytes.

    ``vram`` counts the model, VAE and (when the text encoder is on GPU) the
    text encoder. ``fp8_vram`` is the same with a bf16 DiT stored in fp8.
    ``ram`` is the whole file set, which is what a streaming or CPU load reads
    through memory.
    """
    files = profile["files"]
    model = files["model"]
    reserved = sum(
        file["bytes"] for slot, file in files.items()
        if slot not in ("model", "clip", "t5", "text")
    )
    encoders = sum(
        file["bytes"] for slot, file in files.items()
        if slot in ("clip", "t5", "text")
    )
    vram = model["bytes"] + reserved + (encoders if text_on_gpu else 0)
    fp8_model = model["bytes"] // 2 if model.get("fp8") else model["bytes"]
    fp8_vram = fp8_model + reserved + (encoders if text_on_gpu else 0)
    return {
        "total": total_bytes(profile),
        "vram": vram,
        "fp8_vram": fp8_vram,
        "ram": total_bytes(profile),
    }


#: A recommendation tier, best to worst.
TIERS = ("resident", "fp8", "stream", "cpu", "too-large")

_TIER_LABELS = {
    "resident": "Fits your GPU",
    "fp8": "Tight (fp8)",
    "stream": "Streams from disk",
    "cpu": "CPU only",
    "too-large": "Too large for this PC",
}


def tier_label(tier):
    """A short badge for a recommendation tier."""
    return _TIER_LABELS.get(tier, "Unknown")


def tier_detail(tier, need=None):
    """A sentence explaining a recommendation tier."""
    need = need or {}
    if tier == "resident":
        return "The whole model fits in VRAM and runs resident."
    if tier == "fp8":
        return (
            "The model fits when its blocks are stored in fp8"
            + (f" (about {need.get('fp8_vram_gb', 0):.1f} GB)." if need else ".")
        )
    if tier == "stream":
        return (
            "The model is larger than VRAM; the engine streams blocks from "
            "disk, so generation is slower."
        )
    if tier == "cpu":
        return "No usable GPU was detected; the model runs on the CPU and is very slow."
    if tier == "too-large":
        return (
            "This profile is too large for the detected memory. Choose a "
            "smaller profile, a smaller family, or free some RAM."
        )
    return ""


def _ram_fits(need, ram, reserve):
    """Whether a profile's full file set leaves the RAM reserve intact.

    ``ram`` is ``None`` when it could not be detected, in which case the check
    is skipped rather than blocking every profile.
    """
    return ram is None or need["ram"] / 1e9 + reserve <= ram


def recommend(bundle_item, hardware, tuning=None):
    """The best profile for ``hardware`` and the tier it lands in.

    ``hardware`` is the dict from :func:`engine.hardware.probe` (or the UI
    fallback): ``gpu``, ``vram_gb`` and ``ram_gb``. ``tuning`` is the
    ``engine.tuning`` mapping, so the user's VRAM reserve and text-encoder
    placement are honoured. Hardware never changes the VAE or encoder family
    inside a profile -- only which quantization profile is chosen.
    """
    tuning = tuning or {}
    reserve = float(tuning.get("vram_reserve_gb", 2.5))
    text_on_gpu = tuning.get("text_encoder_device", "cpu") == "auto"
    vram = hardware.get("vram_gb") if hardware.get("gpu") else None
    ram = hardware.get("ram_gb")

    profiles = ordered_profiles(bundle_item)
    # Prefer the highest-quality profile that fits in VRAM (resident or fp8)
    # over a larger one that has to stream from disk: streaming is slower in
    # practice despite the higher-quality file. Only when nothing fits in VRAM
    # fall back to a streaming profile, then to the smallest one.
    for item in profiles:
        need = footprint(item, text_on_gpu)
        if vram is None:
            break
        if not _ram_fits(need, ram, reserve):
            # The GPU may hold the DiT in fp8, but the whole file set still has
            # to pass through RAM to load (and the text encoder stays there when
            # it is placed on the CPU). A profile that would exhaust RAM is not
            # a fit even when its VRAM figure looks comfortable.
            continue
        if need["vram"] / 1e9 + reserve <= vram:
            return _result(item, "resident", need, reserve)
        if need["fp8_vram"] / 1e9 + reserve <= vram:
            return _result(item, "fp8", need, reserve)

    if ram is not None and vram is not None:
        for item in profiles:
            need = footprint(item, text_on_gpu)
            if need["ram"] / 1e9 + reserve <= ram:
                return _result(item, "stream", need, reserve)

    smallest = profiles[-1]
    need = footprint(smallest, text_on_gpu)
    if vram is None and ram is not None and need["ram"] / 1e9 + reserve <= ram:
        return _result(smallest, "cpu", need, reserve)
    return _result(smallest, "too-large", need, reserve)


def classify_profile(profile_item, hardware, tuning=None):
    """The tier a specific profile lands in on this hardware.

    Unlike :func:`recommend`, this does not choose a profile: it answers whether
    the one the user selected runs resident, needs fp8, streams from disk, falls
    back to the CPU, or is too large.
    """
    tuning = tuning or {}
    reserve = float(tuning.get("vram_reserve_gb", 2.5))
    text_on_gpu = tuning.get("text_encoder_device", "cpu") == "auto"
    vram = hardware.get("vram_gb") if hardware.get("gpu") else None
    ram = hardware.get("ram_gb")
    need = footprint(profile_item, text_on_gpu)
    if vram is not None:
        if not _ram_fits(need, ram, reserve):
            return _result(profile_item, "too-large", need, reserve)
        if need["vram"] / 1e9 + reserve <= vram:
            return _result(profile_item, "resident", need, reserve)
        if need["fp8_vram"] / 1e9 + reserve <= vram:
            return _result(profile_item, "fp8", need, reserve)
        if ram is not None and need["ram"] / 1e9 + reserve <= ram:
            return _result(profile_item, "stream", need, reserve)
    elif ram is not None and need["ram"] / 1e9 + reserve <= ram:
        return _result(profile_item, "cpu", need, reserve)
    return _result(profile_item, "too-large", need, reserve)


def _result(profile_item, tier, need, reserve):
    return {
        "profile": profile_item,
        "tier": tier,
        "tier_label": tier_label(tier),
        "need_vram_gb": need["vram"] / 1e9,
        "need_fp8_vram_gb": need["fp8_vram"] / 1e9,
        "need_ram_gb": need["ram"] / 1e9,
        "total_gb": need["total"] / 1e9,
        "reserve_gb": reserve,
    }


# --------------------------------------------------------------- download plan

def is_remote(profile_item):
    """Whether a profile is a Hub repo reference instead of files to save."""
    return any(
        spec.get("remote") for spec in profile_item.get("files", {}).values()
    )


def remote_repo(profile_item):
    """The repo id a remote profile points at, or ``""``."""
    if not is_remote(profile_item):
        return ""
    return next(
        spec["repo"] for spec in profile_item["files"].values()
        if spec.get("remote")
    )


def jobs(profile_item):
    """One job per profile file, with its resolved destination.

    A remote profile yields one pseudo-job whose ``name`` is the repo id; the
    wizard adds it without downloading, and the engine fetches the repository on
    first use.
    """
    if is_remote(profile_item):
        spec = next(
            item for item in profile_item["files"].values() if item.get("remote")
        )
        return [{
            "role": "model",
            "label": SLOT_TITLES["model"],
            "repo": spec["repo"],
            "filename": spec["repo"],
            "name": spec["repo"],
            "dest_dir": "",
            "size": spec["bytes"],
            "kind": spec["kind"],
            "slot": "model",
            "remote": True,
        }]
    result = []
    for slot, spec in profile_item["files"].items():
        result.append({
            "role": slot,
            "label": SLOT_TITLES.get(slot, slot),
            "repo": spec["repo"],
            "filename": spec["filename"],
            "name": spec.get("save_as") or spec["filename"].rsplit("/", 1)[-1],
            "dest_dir": destination(spec["dest"]),
            "size": spec["bytes"],
            "kind": spec["kind"],
            "slot": slot,
        })
    return result


def target_path(job):
    """The local file a job writes (empty for a remote repo reference)."""
    if job.get("remote"):
        return ""
    return os.path.join(job["dest_dir"], job["name"])


def profile_model_job(profile_item):
    """The download job for a profile's diffusion model.

    Returns ``None`` for a remote profile, which has no file to check.
    """
    if is_remote(profile_item):
        return None
    for job in jobs(profile_item):
        if job["slot"] == "model":
            return job
    return None


#: Model kinds that all mean "a Stable Diffusion checkpoint".
_SD_MODEL_KINDS = frozenset(("sd15", "sdxl", "stable_diffusion"))

#: Extensions the wizard will inspect for an already-present component.
_CANDIDATE_SUFFIXES = (
    ".safetensors", ".gguf", ".ckpt", ".pt", ".pth", ".bin",
)

#: Filename tokens that mark a precision or storage variant rather than a
#: different file. Stems that differ only by these name the same component saved
#: at another quantization (``qwen_3_4b`` vs ``qwen_3_4b_fp4_mixed``).
_VARIANT_TOKENS = frozenset((
    "bf16", "f16", "fp16", "fp32", "f32", "fp8", "fp4", "int8", "int4",
    "w4a8", "w8a8", "nvfp4", "awq", "gptq", "scaled", "mixed", "convrot",
    "quantized",
))

#: A GGUF quantization token (``q4_k_m``, ``q8_0``) ends the descriptive part
#: of a filename; the letters after it are quant sub-options, not name words.
_QUANT_TOKEN = re.compile(r"^q\d+[a-z0-9]*$")

#: Kinds published under several unrelated filenames. A file is interchangeable
#: only when *both* stems are aliases of the kind, so another file of the same
#: architecture (SeedVR2's VAE is Flux-shaped) is never reused for it.
_KIND_ALIASES = {
    "flux_vae": frozenset(("ae", "fluxvae", "fluxae")),
    "qwen_image_vae": frozenset(("qwenimagevae", "qwenimage")),
    "qwen_image_21_vae": frozenset(("qwenimage21vae", "qwenimage21")),
    "flux2_vae": frozenset(("flux2vae", "flux2")),
}


def _identity(name):
    """A filename's identity with precision/variant tokens stripped.

    ``qwen_3_4b_fp4_mixed.safetensors`` and ``qwen_3_4b.safetensors`` both
    identify as ``qwen34b``; ``krea-2`` and ``krea2`` agree because separators
    are dropped. A GGUF quant suffix (``q4_k_m``) is ignored along with the
    letters that follow it.
    """
    stem = os.path.splitext(os.path.basename(name or "").lower())[0]
    tokens = []
    for token in re.split(r"[^a-z0-9]+", stem):
        if not token:
            continue
        if _QUANT_TOKEN.match(token):
            break
        if token in _VARIANT_TOKENS:
            continue
        tokens.append(token)
    return "".join(tokens)


def _precision(name):
    """The precision/quantization tokens in a filename (``{"fp8", "scaled"}``)."""
    stem = os.path.splitext(os.path.basename(name or "").lower())[0]
    tokens = set()
    for token in re.split(r"[^a-z0-9]+", stem):
        if not token:
            continue
        if _QUANT_TOKEN.match(token):
            tokens.add(token)
            break
        if token in _VARIANT_TOKENS:
            tokens.add(token)
    return frozenset(tokens)


def _folder_files(folder):
    """Candidate model/component files directly inside ``folder``."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    found = []
    for name in names:
        if not name.lower().endswith(_CANDIDATE_SUFFIXES):
            continue
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            found.append(path)
    return found


def _registered_files(job):
    """Files registered in Model Settings that could satisfy ``job``.

    Component jobs draw from the matching component pool; a model job draws from
    the curated model rows. Those paths may live in another models tree (a
    ComfyUI install), which the destination folder alone would never expose.
    """
    try:
        if job.get("slot") == "model":
            paths_list = [entry.get("path") for entry in paths.model_entries()]
        else:
            pool = component_kinds.SLOT_POOL.get(job.get("slot"))
            paths_list = [
                item.get("path")
                for item in paths.component_pools().get(pool, [])
            ]
    except Exception:  # noqa: BLE001 - settings unavailable, use folders only
        return []
    return [
        os.fspath(path) for path in paths_list
        if path and os.fspath(path).lower().endswith(_CANDIDATE_SUFFIXES)
    ]


def _candidate_files(job):
    """Every file that could satisfy ``job``: destination, then registered."""
    seen = set()
    result = []

    def add(path):
        if not path:
            return
        key = os.path.normcase(os.path.abspath(path))
        if key in seen:
            return
        if not path.lower().endswith(_CANDIDATE_SUFFIXES):
            return
        if not os.path.isfile(path):
            return
        seen.add(key)
        result.append(path)

    if job.get("dest_dir"):
        for path in _folder_files(job["dest_dir"]):
            add(path)
    for path in _registered_files(job):
        add(path)
    return result


def _kind_matches(job, detected):
    """Whether a detected kind can stand in for the job's expected kind."""
    return (
        job["kind"] in _SD_MODEL_KINDS
        or detected is None
        or detected == job["kind"]
    )


def existing_path(job):
    """The job's target when it already exists, else ``""``."""
    target = target_path(job)
    if not target:
        return ""
    return target if os.path.isfile(target) else ""


def find_existing(job):
    """An already-present file for ``job``, or ``None``.

    Checks the exact target first. It then looks for the same file elsewhere:
    the destination folder and the paths registered in Model Settings (the
    component pools and the curated model rows), which may point at another
    models tree such as a ComfyUI install. A file matches when it has the
    expected filename, or the same identity with only a precision/quantization
    suffix changed. ``qwen_3_4b`` satisfies ``qwen_3_4b_fp4_mixed``, while a
    diffusion model must also match the profile's quantization so the profile's
    VRAM figure stays true.

    Files that merely share the detected architecture are *not* reused:
    ``seedvr2_ema_vae`` is a Flux-shaped VAE but is not the Flux VAE that a Flux
    or Z-Image model needs. The only same-kind exception is a component
    published under several aliases (see :data:`_KIND_ALIASES`), such as the
    Flux VAE's ``ae`` and ``flux_vae`` names.

    Returns ``{"path", "reason", "detected", "ok"}`` where ``reason`` is
    ``"exact"`` (the job's own target), ``"variant"`` (the same file under
    another name or in another registered folder) or ``"kind"`` (an alias of
    the same component). ``ok`` is False when an exact-name file exists but its
    header says it is the wrong kind; the caller should download over it rather
    than reuse it.
    """
    if job.get("remote"):
        return None
    target = target_path(job)
    category = component_kinds.category_of(job["kind"])
    if target and os.path.isfile(target):
        detected = component_kinds.detect(target, category)
        ok = _kind_matches(job, detected)
        return {
            "path": target, "reason": "exact", "detected": detected, "ok": ok,
        }

    candidates = _candidate_files(job)
    wanted_name = os.path.basename(job.get("name") or "").lower()
    wanted = _identity(job.get("name"))

    # 1. The same filename in another registered folder.
    if wanted_name:
        for path in candidates:
            if os.path.basename(path).lower() != wanted_name:
                continue
            detected = component_kinds.detect(path, category)
            if _kind_matches(job, detected):
                return {
                    "path": path, "reason": "variant", "detected": detected,
                    "ok": True,
                }

    # 2. The same identity saved at another precision (bf16 vs fp8 vs int8).
    #    A diffusion model must also match the profile's quantization: reusing a
    #    bf16 file for an nvfp4 profile would make the profile's VRAM figure
    #    wrong. Encoders and VAEs may stand in at another precision.
    wanted_precision = _precision(job.get("name"))
    if wanted:
        for path in candidates:
            if _identity(os.path.basename(path)) != wanted:
                continue
            if (
                job.get("slot") == "model"
                and _precision(os.path.basename(path)) != wanted_precision
            ):
                continue
            detected = component_kinds.detect(path, category)
            if _kind_matches(job, detected):
                return {
                    "path": path, "reason": "variant", "detected": detected,
                    "ok": True,
                }

    # 3. A component published under more than one filename (the Flux VAE is
    #    both ``ae`` and ``flux_vae``).
    aliases = _KIND_ALIASES.get(job["kind"], frozenset())
    if wanted in aliases:
        for path in candidates:
            if _identity(os.path.basename(path)) not in aliases:
                continue
            detected = component_kinds.detect(path, category)
            if detected == job["kind"]:
                return {
                    "path": path, "reason": "kind", "detected": detected,
                    "ok": True,
                }
    return None


def installed_profile(bundle_item, hardware=None, tuning=None):
    """The best-quality profile whose model file is already present.

    Walks the bundle's profiles best-quality first and returns a recommendation
    dict (like :func:`recommend`) for the profile whose model file is on disk and
    is not too large for the machine, or ``None`` when none is. The wizard relies
    on this so a model the user already has is kept instead of offering a
    duplicate download. With no usable ``hardware`` the match is returned
    without a fit check.
    """
    known = bool(hardware) and bool(
        hardware.get("gpu") or hardware.get("ram_gb")
    )
    variant = None
    for item in ordered_profiles(bundle_item):
        model_job = profile_model_job(item)
        if model_job is None:
            continue
        found = find_existing(model_job)
        if found is None:
            continue
        if known:
            rec = classify_profile(item, hardware, tuning)
            if rec["tier"] == "too-large":
                continue
        else:
            rec = {
                "profile": item,
                "tier": "unknown",
                "tier_label": tier_label("unknown"),
                "total_gb": total_bytes(item) / 1e9,
            }
        expected = os.path.basename(model_job.get("name") or "").lower()
        if os.path.basename(found["path"]).lower() == expected:
            return rec
        if variant is None:
            variant = rec
    return variant


def verify(path, spec):
    """``(detected_kind, ok)`` for a downloaded file against its expected kind.

    A file whose header cannot be identified is not treated as a mismatch: the
    detector is best-effort and must not reject a valid download. Stable
    Diffusion kinds are interchangeable here because the family covers sd15,
    sdxl and 2.x checkpoints.
    """
    category = component_kinds.category_of(spec["kind"])
    detected = component_kinds.detect(path, category)
    if spec["kind"] in _SD_MODEL_KINDS:
        # The stable_diffusion family spans sd15/sdxl/2.x checkpoints and the
        # name-based fallback is unreliable for community filenames, so any
        # model detection is accepted here.
        return detected, True
    return detected, (detected is None or detected == spec["kind"])


def validate():
    """Consistency issues in the catalog (empty when it is well formed)."""
    issues = []
    for item in BUNDLES:
        family = item["family"]
        required = {slot for slot, _title, _pool in
                    component_kinds.FAMILY_SLOTS.get(family, ())}
        for profile_item in item["profiles"]:
            slots = set(profile_item["files"])
            if slots != required | {"model"}:
                issues.append(
                    f"{item['id']}/{profile_item['id']}: slots {sorted(slots)} "
                    f"do not match family slots {sorted(required | {'model'})}"
                )
            for slot, spec in profile_item["files"].items():
                if spec["slot"] != slot:
                    issues.append(
                        f"{item['id']}/{profile_item['id']}: slot key mismatch"
                    )
                if spec["bytes"] <= 0:
                    issues.append(f"{item['id']}/{profile_item['id']}: bad size")
                if spec.get("remote"):
                    if not spec.get("repo"):
                        issues.append(
                            f"{item['id']}/{profile_item['id']}: missing repo"
                        )
                    continue
                if spec["dest"] not in DESTINATIONS:
                    issues.append(f"{item['id']}/{profile_item['id']}: bad destination")
                if slot == "model":
                    # Stable Diffusion checkpoints are one family whose kind
                    # depends on the checkpoint header (sd15/sdxl), so there is
                    # no fixed expected model kind to compare against.
                    expected = None if family == "stable_diffusion" else family
                else:
                    expected = component_kinds.SLOT_KIND.get((family, slot))
                if expected and spec["kind"] != expected:
                    issues.append(
                        f"{item['id']}/{profile_item['id']}/{slot}: kind "
                        f"{spec['kind']!r} != {expected!r}"
                    )
    return issues
