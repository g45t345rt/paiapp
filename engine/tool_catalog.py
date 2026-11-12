"""Curated download bundles for the standalone tools.

Background removal, upscaling and segmentation run on their own model files,
separate from a generation model's diffusion/VAE/text-encoder set. This catalog
pairs each tool model with the Hub file(s) it needs, the folder it belongs in and
an offline size, so the Tool Download Wizard can install it without the user
working out which file goes where.

The file sizes are the Hub's LFS sizes (kept here so the wizard can show a size
before it contacts the Hub); the wizard still refreshes nothing at download
time. Repos are public and ungated unless the bundle says otherwise; a gated
bundle needs the user to accept the license and set ``HF_TOKEN``.
"""

import os

from . import paths

#: Display title for a tool file's role.
SLOT_TITLES = {
    "model": "Model",
    "vae": "VAE",
}

#: Destination key -> the folder function that resolves it at run time.
DESTINATIONS = {
    "background": paths.background_dir,
    "upscale": paths.upscale_dir,
    "segmentation": paths.segmentation_dir,
    "vae": paths.vae_dir,
}

#: Destination key -> the Model Settings component pool a file is curated into.
POOLS = {
    "background": "background",
    "upscale": "upscale",
    "segmentation": "segmentation",
    "vae": "vaes",
}

#: Tool key -> the group heading the wizard shows.
GROUPS = {
    "background": "Background removal",
    "upscale": "Upscaling",
    "segmentation": "Segmentation",
}


def destination(name):
    """The folder a destination key names (the models root when unknown)."""
    fn = DESTINATIONS.get(name)
    return fn() if fn is not None else paths.models_root()


def _file(repo, filename, save_as, slot, dest, size, gated=False):
    """One downloadable file in a tool bundle."""
    return {
        "repo": repo,
        "filename": filename,
        "save_as": save_as,
        "slot": slot,
        "dest": dest,
        "pool": POOLS.get(dest, dest),
        "bytes": int(size),
        "gated": bool(gated),
    }


def _bundle(bundle_id, label, tool, notes, files, gated=False):
    return {
        "id": bundle_id,
        "label": label,
        "tool": tool,
        "group": GROUPS.get(tool, tool),
        "notes": notes,
        "files": tuple(files),
        "gated": bool(gated),
    }


# ------------------------------------------------------------------ file specs

_BIREFNET_GENERAL = _file(
    "ZhengPeng7/BiRefNet",
    "model.safetensors",
    "birefnet-general.safetensors",
    "model", "background", 444473596,
)
_U2NET = _file(
    "jellybox/u2net",
    "u2net_320.onnx",
    "u2net.onnx",
    "model", "background", 175997641,
)
_U2NET_PORTRAIT = _file(
    "jellybox/u2net-human-seg",
    "u2net-human-seg_320.onnx",
    "u2net_human_seg.onnx",
    "model", "background", 175997641,
)
_MODNET = _file(
    "Xenova/modnet",
    "onnx/model.onnx",
    "modnet_photographic_portrait_matting.onnx",
    "model", "background", 25888640,
)
_BIREFNET_LITE = _file(
    "onnx-community/BiRefNet_lite-ONNX",
    "onnx/model_fp16.onnx",
    "birefnet-lite.onnx",
    "model", "background", 114538221,
)
_BIREFNET_PORTRAIT = _file(
    "onnx-community/BiRefNet-portrait-ONNX",
    "onnx/model_fp16.onnx",
    "birefnet-portrait.onnx",
    "model", "background", 489666272,
)

_MOBILE_SAM = _file(
    "vietanhdev/segment-anything-onnx-models",
    "mobile_sam_20230629_quant.zip",
    "mobile_sam_20230629_quant.zip",
    "model", "segmentation", 10971233,
)
_SAM_VIT_B = _file(
    "vietanhdev/segment-anything-onnx-models",
    "sam_vit_b_01ec64_quant.zip",
    "sam_vit_b_01ec64_quant.zip",
    "model", "segmentation", 75377417,
)
_SAM_VIT_H = _file(
    "vietanhdev/segment-anything-onnx-models",
    "sam_vit_h_4b8939_quant.zip",
    "sam_vit_h_4b8939_quant.zip",
    "model", "segmentation", 442519065,
)

_REALESRGAN_X2 = _file(
    "ai-forever/Real-ESRGAN",
    "RealESRGAN_x2.pth",
    "RealESRGAN_x2.pth",
    "model", "upscale", 67061725,
)
_REALESRGAN_X4 = _file(
    "ai-forever/Real-ESRGAN",
    "RealESRGAN_x4.pth",
    "RealESRGAN_x4.pth",
    "model", "upscale", 67040989,
)
_REALESRGAN_X8 = _file(
    "ai-forever/Real-ESRGAN",
    "RealESRGAN_x8.pth",
    "RealESRGAN_x8.pth",
    "model", "upscale", 67189359,
)

_SEEDVR2_VAE = _file(
    "numz/SeedVR2_comfyUI",
    "ema_vae_fp16.safetensors",
    "seedvr2_ema_vae_fp16.safetensors",
    "vae", "vae", 501324814,
)
_SEEDVR2_3B = _file(
    "numz/SeedVR2_comfyUI",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    "model", "upscale", 3391544696,
)
_SEEDVR2_7B = _file(
    "numz/SeedVR2_comfyUI",
    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
    "seedvr2_ema_7b_fp8_e4m3fn.safetensors",
    "model", "upscale", 8239729704,
)


# -------------------------------------------------------------------- bundles

TOOL_BUNDLES = (
    # Background removal ---------------------------------------------------
    _bundle(
        "birefnet-general", "BiRefNet (general)", "background",
        "A general-purpose remover with strong edge and hair detail.",
        (_BIREFNET_GENERAL,),
    ),
    _bundle(
        "u2net", "U2Net", "background",
        "The classic fast remover, small enough to run on almost any machine.",
        (_U2NET,),
    ),
    _bundle(
        "u2net-portrait", "U2Net (portrait)", "background",
        "The human-segmentation U2Net, tuned for people and portraits.",
        (_U2NET_PORTRAIT,),
    ),
    _bundle(
        "modnet-portrait", "MODNet (portrait)", "background",
        "A tiny portrait matting model with soft hair edges.",
        (_MODNET,),
    ),
    _bundle(
        "birefnet-lite", "BiRefNet lite", "background",
        "A small, fast general remover that still handles fine edges.",
        (_BIREFNET_LITE,),
    ),
    _bundle(
        "birefnet-portrait", "BiRefNet (portrait)", "background",
        "A high-quality remover tuned for people, hair and portraits.",
        (_BIREFNET_PORTRAIT,),
    ),
    # Segmentation ---------------------------------------------------------
    _bundle(
        "mobile-sam", "MobileSAM", "segmentation",
        "A tiny click-to-segment model for quick masks on any machine.",
        (_MOBILE_SAM,),
    ),
    _bundle(
        "sam-vit-b", "SAM ViT-B", "segmentation",
        "The balanced Segment Anything model: sharper masks, still compact.",
        (_SAM_VIT_B,),
    ),
    _bundle(
        "sam-vit-h", "SAM ViT-H", "segmentation",
        "The largest Segment Anything model, for the most precise masks.",
        (_SAM_VIT_H,),
    ),
    # Upscaling ------------------------------------------------------------
    _bundle(
        "realesrgan-x4", "Real-ESRGAN x4", "upscale",
        "The standard 4x photo upscaler, fast on CPU or GPU.",
        (_REALESRGAN_X4,),
    ),
    _bundle(
        "realesrgan-x2", "Real-ESRGAN x2", "upscale",
        "A gentler 2x upscale for already-large images.",
        (_REALESRGAN_X2,),
    ),
    _bundle(
        "realesrgan-x8", "Real-ESRGAN x8", "upscale",
        "An 8x upscale for small sources; slower and more inventive.",
        (_REALESRGAN_X8,),
    ),
    _bundle(
        "seedvr2-3b", "SeedVR2 3B", "upscale",
        "A one-step diffusion restorer for photos and AI images.",
        (_SEEDVR2_3B, _SEEDVR2_VAE),
    ),
    _bundle(
        "seedvr2-7b", "SeedVR2 7B", "upscale",
        "The larger SeedVR2 restorer, for more detail at a higher cost.",
        (_SEEDVR2_7B, _SEEDVR2_VAE),
    ),
)


#: Purpose, strengths and trade-offs shown by the wizard for each bundle.
DESCRIPTIONS = {
    "birefnet-general": {
        "purpose": (
            "Cut a foreground subject out of a photo with fine edge detail."
        ),
        "strengths": (
            "Handles hair, fur and semi-transparent edges better than U2Net.",
            "A good default when no dedicated portrait model fits.",
        ),
        "consider": (
            "Needs a GPU to feel responsive; 444 MB on disk.",
            "Slower per image than U2Net.",
        ),
    },
    "u2net": {
        "purpose": (
            "Fast, general saliency-based background removal."
        ),
        "strengths": (
            "Very fast and small (176 MB).",
            "Runs comfortably on CPU when no GPU is available.",
        ),
        "consider": (
            "Coarser edges around hair and thin shapes.",
            "Less accurate on cluttered backgrounds.",
        ),
    },
    "u2net-portrait": {
        "purpose": (
            "Fast human segmentation for portraits and full-body shots."
        ),
        "strengths": (
            "More reliable on people than the general U2Net.",
            "Same small, fast ONNX graph.",
        ),
        "consider": (
            "Trained for people: pets and objects are poorly segmented.",
            "Coarser edges than BiRefNet or MODNet.",
        ),
    },
    "modnet-portrait": {
        "purpose": (
            "Lightweight portrait matting with a soft alpha matte."
        ),
        "strengths": (
            "Tiny (26 MB) and quick; good hair edges on people.",
            "Its matte suits compositing better than a hard mask.",
        ),
        "consider": (
            "Portraits only; it is not a general object remover.",
            "Lower resolution than BiRefNet.",
        ),
    },
    "birefnet-lite": {
        "purpose": (
            "A compact general-purpose remover for everyday cutouts."
        ),
        "strengths": (
            "Only 109 MB and much faster than the full BiRefNet.",
            "Runs on CPU; good hair and fur edges for its size.",
        ),
        "consider": (
            "Less precise than the full or portrait BiRefNet on hard cases.",
            "A single ONNX file, so no separate preprocessor is needed.",
        ),
    },
    "birefnet-portrait": {
        "purpose": (
            "Cut people out of portraits and full-body shots with clean edges."
        ),
        "strengths": (
            "Better hair and semi-transparent edges than U2Net or MODNet.",
            "Handles full-body subjects, not only head-and-shoulders crops.",
        ),
        "consider": (
            "About 467 MB and slower than the lite model.",
            "Tuned for people; objects are better served by BiRefNet general.",
        ),
    },
    "mobile-sam": {
        "purpose": (
            "Click-to-segment masks for any subject in the image."
        ),
        "strengths": (
            "Tiny (11 MB) and fast, so it works on CPU-only machines.",
            "One click produces a usable mask; extra clicks refine it.",
        ),
        "consider": (
            "Coarser masks than the larger SAM models on thin detail.",
            "Installs as a folder with its encoder, decoder and config.",
        ),
    },
    "sam-vit-b": {
        "purpose": (
            "Balanced click-to-segment masks with sharper boundaries."
        ),
        "strengths": (
            "Noticeably more precise than MobileSAM at a moderate size.",
            "Good default for detailed objects and product cutouts.",
        ),
        "consider": (
            "About 75 MB; slower than MobileSAM, especially on CPU.",
            "Installs as a folder with its encoder, decoder and config.",
        ),
    },
    "sam-vit-h": {
        "purpose": (
            "The most precise click-to-segment masks SAM can produce."
        ),
        "strengths": (
            "Best boundaries on fine structures and cluttered scenes.",
            "Worth it when a mask feeds inpainting or compositing.",
        ),
        "consider": (
            "About 442 MB and the slowest of the three, particularly on CPU.",
            "Installs as a folder with its encoder, decoder and config.",
        ),
    },
    "realesrgan-x4": {
        "purpose": (
            "Enlarge an image 4x and restore light detail."
        ),
        "strengths": (
            "The usual first choice for photos and illustrations.",
            "Fast enough to iterate on CPU, much faster on a GPU.",
        ),
        "consider": (
            "Can sharpen noise and compression artifacts.",
            "Not a generative restorer: heavy blur stays soft.",
        ),
    },
    "realesrgan-x2": {
        "purpose": (
            "A gentler 2x enlargement that preserves the source character."
        ),
        "strengths": (
            "Less risk of over-sharpening than the x4 model.",
            "Good for already-large renders and screenshots.",
        ),
        "consider": (
            "Only doubles the size; use x4 for small sources.",
            "The same per-tile cost as the other Real-ESRGAN models.",
        ),
    },
    "realesrgan-x8": {
        "purpose": (
            "An 8x enlargement for very small sources."
        ),
        "strengths": (
            "Makes tiny icons or crops usable at print sizes.",
            "The fastest way to gain resolution without diffusion.",
        ),
        "consider": (
            "Invents detail at extreme scales.",
            "The slowest and least faithful of the x2/x4/x8 trio.",
        ),
    },
    "seedvr2-3b": {
        "purpose": (
            "Diffusion-based restoration and 4x upscaling for photos and AI "
            "images."
        ),
        "strengths": (
            "Recovers plausible detail instead of only sharpening.",
            "The 3B fp8 weights fit much smaller GPUs than the 7B model.",
        ),
        "consider": (
            "Downloads the model plus its VAE (about 3.9 GB).",
            "Slow compared with Real-ESRGAN; needs torch and a strong GPU.",
        ),
    },
    "seedvr2-7b": {
        "purpose": (
            "The larger SeedVR2 restorer, for the most detail it can recover."
        ),
        "strengths": (
            "Better texture and structure than the 3B variant on difficult "
            "images.",
            "Shares the same VAE, so switching from 3B is a single file.",
        ),
        "consider": (
            "About 8.7 GB with the VAE; heavy VRAM use.",
            "Slower than the 3B model with no benefit on clean inputs.",
        ),
    },
}

for _item in TOOL_BUNDLES:
    _item["description"] = DESCRIPTIONS.get(_item["id"], {})


# ------------------------------------------------------------------ queries

def bundles():
    """Every curated tool bundle, in display order."""
    return list(TOOL_BUNDLES)


def bundle(bundle_id):
    """The bundle with ``bundle_id``, or ``None``."""
    for item in TOOL_BUNDLES:
        if item["id"] == bundle_id:
            return item
    return None


def for_tool(tool):
    """Every bundle that installs a model for ``tool``."""
    return [item for item in TOOL_BUNDLES if item["tool"] == tool]


def total_bytes(bundle_item):
    """The on-disk size of every file in a bundle."""
    return sum(spec["bytes"] for spec in bundle_item["files"])


def jobs(bundle_item):
    """One download job per bundle file, with its resolved destination."""
    result = []
    for spec in bundle_item["files"]:
        result.append({
            "role": spec["slot"],
            "label": SLOT_TITLES.get(spec["slot"], spec["slot"]),
            "repo": spec["repo"],
            "filename": spec["filename"],
            "name": spec.get("save_as") or spec["filename"].rsplit("/", 1)[-1],
            "dest_dir": destination(spec["dest"]),
            "size": spec["bytes"],
            "slot": spec["slot"],
            "pool": spec["pool"],
            "gated": spec["gated"],
        })
    return result


def target_path(job):
    """The local file a job writes."""
    return os.path.join(job["dest_dir"], job["name"])


def _registered_files(job):
    """Files curated in the job's Model Settings pool that could satisfy it."""
    try:
        items = paths.component_pools().get(job.get("pool"), [])
    except Exception:  # noqa: BLE001 - settings unavailable, use the folder
        return []
    return [
        os.fspath(item["path"]) for item in items if item.get("path")
    ]


def _same_file(job, path):
    """Whether ``path`` can stand in for ``job``.

    An exact filename is the strong match. The SeedVR2 VAE is also matched by
    its role tokens, because the Hub publishes it as ``ema_vae_fp16`` while the
    engine looks for a name containing ``seedvr2`` and ``vae``.
    """
    name = os.path.basename(path).lower()
    if name == os.path.basename(job.get("name") or "").lower():
        return True
    if job.get("slot") == "vae" and job.get("pool") == "vaes":
        return "seedvr2" in name and "vae" in name
    return False


def _extracted_folder(job):
    """The folder an archive job was unpacked into, or ``None``.

    :func:`engine.hub.extract_archive` unpacks a zip into a folder named after
    the archive stem, so a SAM bundle that is already installed is detected
    without re-downloading the zip.
    """
    name = os.path.basename(job.get("name") or "")
    if not name.lower().endswith(".zip"):
        return None
    folder = os.path.join(job.get("dest_dir") or "", os.path.splitext(name)[0])
    return folder if os.path.isdir(folder) else None


def find_existing(job):
    """An already-present file for ``job``, or ``None``.

    Checks the exact target first, then an extracted-archive folder, then the
    destination folder and the files curated in the matching Model Settings pool
    (which may live in another models tree). Returns ``{"path", "reason"}`` where
    ``reason`` is ``"exact"`` or ``"variant"``.
    """
    target = target_path(job)
    if os.path.isfile(target):
        return {"path": target, "reason": "exact"}

    folder = _extracted_folder(job)
    if folder:
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            names = []
        for name in names:
            path = os.path.join(folder, name)
            if os.path.isfile(path) and path.lower().endswith(".onnx"):
                return {"path": path, "reason": "variant"}

    seen = set()
    candidates = []
    folders = [job.get("dest_dir")]
    for path in _registered_files(job):
        folders.append(os.path.dirname(path))
    for folder in folders:
        if not folder:
            continue
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            key = os.path.normcase(os.path.abspath(path))
            if key in seen or not os.path.isfile(path):
                continue
            seen.add(key)
            candidates.append(path)
    for path in candidates:
        if _same_file(job, path):
            return {"path": path, "reason": "variant"}
    return None


def validate():
    """Consistency issues in the catalog (empty when it is well formed)."""
    issues = []
    ids = set()
    for item in TOOL_BUNDLES:
        if item["id"] in ids:
            issues.append(f"{item['id']}: duplicate id")
        ids.add(item["id"])
        if item["tool"] not in ("background", "upscale", "segmentation"):
            issues.append(f"{item['id']}: unknown tool {item['tool']}")
        if not item["files"]:
            issues.append(f"{item['id']}: no files")
        for spec in item["files"]:
            if spec["dest"] not in DESTINATIONS:
                issues.append(f"{item['id']}/{spec['slot']}: unknown dest")
            if spec["pool"] not in ("background", "upscale", "segmentation", "vaes"):
                issues.append(f"{item['id']}/{spec['slot']}: unknown pool")
            if not spec["repo"] or not spec["filename"]:
                issues.append(f"{item['id']}/{spec['slot']}: missing repo/file")
            if spec["bytes"] <= 0:
                issues.append(f"{item['id']}/{spec['slot']}: bad size")
    return issues
