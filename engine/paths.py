"""Model locations, per-family defaults and per-model sampler presets.

Model files are resolved from the ``models`` section of ``settings.json`` (see
:mod:`engine.settings`). A family's default files come from the curated links
first -- the first curated model row of the family, its component assignment,
else the matching component pool -- so the models root and its per-directory and
per-component folders are only the backup and the download location. Nothing here
assumes ComfyUI is installed -- the ``PAI_MODELS`` environment variable can stand
in for the root, and with neither set the root falls back to a per-user folder
(see :func:`default_models_root`), so a fresh install has a working location to
download into.

The ``generation`` section holds the sampler settings last used with each model
(see :func:`model_preset` / :func:`save_model_preset`), so they layer over the
per-family :func:`family_defaults` when a model is picked. The same section also
holds named recipes: per-model ones the user saves plus the read-only per-family
ones :func:`family_recipes` offers, each capturing the sampler fields and the
output size.
"""

import json
import os
import struct
import uuid

from . import component_kinds
from . import settings

# Component files used when a family's file is not overridden in settings. The
# paths are relative to the models root and mirror the reference workflow's
# ComfyUI layout.
_ANIMA_COMPONENTS = {
    "model": ("diffusion_models", "anima_aestheticV11.safetensors"),
    "clip": ("text_encoders", "qwen_3_06b_base.safetensors"),
    "vae": ("vae", "qwen_image_vae.safetensors"),
}
_QWEN_COMPONENTS = {
    "dit": ("unet", "qwen-image-2.1-UC-Q4_K_M.gguf"),
    "text": ("text_encoders", "Qwen3VL-8B-Instruct-Q4_K_M.gguf"),
    "vae": ("vae", "qwen_image_2.1_vae_bf16.safetensors"),
}
_FLUX_COMPONENTS = {
    "dit": ("unet", "flux1-dev-Q4_K_S.gguf"),
    "clip": ("text_encoders", "clip_l.safetensors"),
    "t5": ("text_encoders", "t5-v1_1-xxl-encoder-Q5_K_S.gguf"),
    "vae": ("vae", "flux_vae.safetensors"),
}
_ZIMAGE_COMPONENTS = {
    "dit": ("diffusion_models", "z_image_turbo_bf16.safetensors"),
    "text": ("text_encoders", "qwen_3_4b.safetensors"),
    "vae": ("vae", "ae.safetensors"),
}
_KREA2_COMPONENTS = {
    "dit": ("diffusion_models", "krea2_turbo_bf16.safetensors"),
    "text": ("text_encoders", "qwen3vl_4b_bf16.safetensors"),
    "vae": ("vae", "qwen_image_vae.safetensors"),
}
_MOSS_COMPONENTS = {
    "dit": ("diffusion_models", "moss_image_sft_bf16.safetensors"),
    "text": ("text_encoders", "qwen3.5_9b_bf16.safetensors"),
    # MOSS Image uses the *small* FLUX.2 VAE (Comfy-Org ``flux2-vae_small``,
    # 62M params); the full FLUX.2 VAE (``flux2-vae``, 84M) is a different
    # decoder and does not match the DiT's latent space.
    "vae": ("vae", "flux2-vae_small.safetensors"),
}
_H3_COMPONENTS = {
    "dit": ("diffusion_models", "minimax_h3_fl2va_pruned_int8_convrot.safetensors"),
    "text": ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "vae": ("vae", "minimax_h3_video_vae_int8_convrot.safetensors"),
}
# MiniMax-H3-Image shares the H3 text encoder; its DiT and 2D-ViT image VAE are
# different files (the Comfy-Org single-file repackage).
_H3_IMAGE_COMPONENTS = {
    "dit": ("diffusion_models", "minimax_h3_image_pruned_int8_convrot.safetensors"),
    "text": ("text_encoders", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
    "vae": ("vae", "minimax_h3_image_vae_fp16.safetensors"),
}
# Fibo 2.0 is loaded as a whole diffusers-format directory (or a repo id), not as
# separate component files, so its single "model" slot points at a folder rather
# than a transformer file (see :mod:`engine.fibo`).
_FIBO_COMPONENTS = {
    "model": ("diffusion_models", "Fibo-2.0-dev"),
}

# Candidate pools a model's per-model component assignment can draw from (see
# :func:`component_pools`). "adapters" holds run-time adapter files (currently
# the Anima reference adapter LoRA, which is applied only while a reference is
# attached -- see :mod:`engine.anima_ref`), kept out of "loras" because an
# ordinary LoRA applies on every run.
_COMPONENT_POOLS = (
    "text_encoders", "vaes", "loras", "adapters", "controlnet", "background",
    "upscale",
)

# The picker groups a curated model row can belong to, in display order. The
# value is the engine family ``generate`` dispatches on. Model Settings uses the
# same table, so both its Models tab and the Generate picker order rows alike.
MODEL_GROUPS = (
    ("Anima", "anima"),
    ("Qwen-Image", "qwen_image"),
    ("Flux", "flux"),
    ("Z-Image", "z_image"),
    ("Krea 2", "krea2"),
    ("MOSS Image", "moss_image"),
    ("MiniMax H3", "h3"),
    ("MiniMax H3 Image", "h3_image"),
    ("Fibo 2.0", "fibo"),
    ("Stable Diffusion", "stable_diffusion"),
)



def anima_components():
    """Default anima component files, keyed by config slot (see ``DEFAULTS``)."""
    return {key: component_path(parts) for key, parts in _ANIMA_COMPONENTS.items()}


def qwen_components():
    """Default Qwen-Image 2.1 component files, keyed by ``QWEN_DEFAULTS`` slot."""
    return {key: component_path(parts) for key, parts in _QWEN_COMPONENTS.items()}


def flux_components():
    """Default Flux component files, keyed by ``FLUX_DEFAULTS`` slot."""
    return {key: component_path(parts) for key, parts in _FLUX_COMPONENTS.items()}


def zimage_components():
    """Default Z-Image component files, keyed by ``ZIMAGE_DEFAULTS`` slot."""
    return {key: component_path(parts) for key, parts in _ZIMAGE_COMPONENTS.items()}


def krea2_components():
    """Default Krea 2 component files, keyed by ``KREA2_DEFAULTS`` slot."""
    return {key: component_path(parts) for key, parts in _KREA2_COMPONENTS.items()}


def moss_components():
    """Default MOSS Image component files, keyed by ``MOSS_DEFAULTS`` slot."""
    return {key: component_path(parts) for key, parts in _MOSS_COMPONENTS.items()}


def h3_components():
    """Default MiniMax H3 component files, keyed by ``H3_DEFAULTS`` slot."""
    return {key: component_path(parts) for key, parts in _H3_COMPONENTS.items()}


def h3_image_components():
    """Default MiniMax H3 Image component files, keyed by ``H3_IMAGE_DEFAULTS``."""
    return {key: component_path(parts) for key, parts in _H3_IMAGE_COMPONENTS.items()}


def _model_settings():
    return settings.section("models")


def model_entries():
    """User-curated model rows from the ``models.entries`` setting.

    Returns ``[{"id", "label", "family", "group", "path", "components",
    "loras"}, ...]`` in stored order. Rows without a path are ignored; a blank
    label falls back to the file stem. ``id`` is the row's stable identity
    (``""`` when it has none yet), so two rows may share one file ``path`` --
    the same model with different labels and LoRA stacks -- without collapsing.
    ``components`` holds only the slots the user overrode (``clip``/``vae`` or
    ``text``/``vae``); an unset slot means "use the family default". ``loras``
    is the model's LoRA list (``[{"path", "strength"}]``), empty when the model
    has none.
    """
    value = _model_settings().get("entries")
    if not isinstance(value, list):
        return []
    entries = []
    for item in value:
        if not isinstance(item, dict) or not item.get("path"):
            continue
        path = os.fspath(item["path"])
        # A segmentation file is a support model, not a picker model: drop it so
        # a row an older build curated (e.g. a SAM ONNX bundle) never reaches the
        # picker. See :func:`is_segmentation_model`.
        if is_segmentation_model(path):
            continue
        components = item.get("components")
        entries.append({
            "id": str(item.get("id") or ""),
            "label": item.get("label") or _stem(path),
            "family": item.get("family") or "anima",
            "group": item.get("group") or "Anima",
            "path": path,
            "components": {
                str(slot): os.fspath(value)
                for slot, value in (components or {}).items()
                if value
            } if isinstance(components, dict) else {},
            "loras": _parse_loras(item.get("loras")),
        })
    return entries


def new_entry_id():
    """A fresh identity for a curated model row (unique within the settings)."""
    return uuid.uuid4().hex[:12]


def entry_key(entry):
    """The stable preset/recipe key for a model row.

    A curated row keys on its ``id`` so two rows sharing one file path keep
    their own settings; a scanned row has no identity, so its ``path`` is the
    key.
    """
    return entry.get("id") or os.fspath(entry["path"])


def _parse_loras(value):
    """A saved ``loras`` list as ``[{"path", "strength"}]`` (empty when unset)."""
    loras = []
    if not isinstance(value, list):
        return loras
    for item in value:
        if isinstance(item, dict) and item.get("path"):
            loras.append({
                "path": os.fspath(item["path"]),
                "strength": float(item.get("strength", 1.0) or 1.0),
            })
    return loras


def component_pools():
    """Curated component pools from the ``models.components`` setting.

    Returns ``{pool: [{"label", "path"}, ...]}`` for every known pool, with an
    empty list when unset. Pools are the candidate files a model's components
    can be assigned from.
    """
    section = _model_settings().get("components")
    section = section if isinstance(section, dict) else {}
    pools = {}
    for name in _COMPONENT_POOLS:
        items = section.get(name)
        pool = []
        if isinstance(items, list):
            for item in items:
                if not isinstance(item, dict) or not item.get("path"):
                    continue
                path = os.fspath(item["path"])
                pool.append({"label": item.get("label") or _stem(path), "path": path})
        pools[name] = pool
    return pools



def default_models_root():
    """Where models live when the user has not pointed the app anywhere.

    A per-user folder beside ``settings.json`` (``%APPDATA%\\Pai\\models`` on
    Windows, ``~/.config/Pai/models`` elsewhere), so a fresh install has a
    working root to download into before anything is configured. The folder is
    not created here; :func:`models_root` only resolves the path.
    """
    return os.path.join(settings.app_dir(), "models")


def models_root():
    """The folder relative model paths resolve under.

    Precedence: the ``models.root`` setting, then the ``PAI_MODELS`` environment
    variable, then :func:`default_models_root`, so the root is never empty and a
    new user always has somewhere to put a download.
    """
    configured = _model_settings().get("root")
    if configured:
        return os.fspath(configured)
    return os.environ.get("PAI_MODELS") or default_models_root()


def model(*parts):
    """Join ``parts`` under the configured models root."""
    return os.path.join(models_root(), *parts)


# How many trailing path segments the models UI shows when full paths are off.
# Enough to tell two files apart, few enough that the user's home folder never
# ends up on screen -- a full path in a screenshot or screen recording would
# otherwise leak it.
_TAIL_SEGMENTS = 2


def show_full_paths():
    """Whether the models UI may print whole paths.

    A full path puts the user's home folder (``C:\\Users\\...``) on screen, where
    it can end up in a screenshot or recording. Model Settings asks for the full
    form, stored as ``models.show_full_paths``; every other window follows it.
    """
    return bool(_model_settings().get("show_full_paths"))


def shorten_path(path, full=False):
    """A path as the models UI shows it: the whole thing, or just its tail.

    Unless full paths are asked for, the home folder is folded to ``~`` and only
    the last couple of segments are kept, prefixed with an ellipsis. A Fibo 2.0
    model can be a Hub repo id (``owner/name``) rather than a local path; that is
    kept whole, since normpath would rewrite its slash.
    """
    if full or not path:
        return path
    raw = os.fspath(path)
    if "/" in raw and "\\" not in raw and not os.path.isabs(raw) and not os.path.exists(raw):
        return raw
    text = os.path.normpath(raw)
    home = os.path.normpath(os.path.expanduser("~"))
    folded = text
    if os.path.normcase(text) == os.path.normcase(home):
        folded = "~"
    elif os.path.normcase(text).startswith(os.path.normcase(home + os.sep)):
        folded = "~" + text[len(home):]
    parts = [part for part in folded.replace("\\", "/").split("/") if part]
    if len(parts) > _TAIL_SEGMENTS:
        return "\u2026" + os.sep + os.sep.join(parts[-_TAIL_SEGMENTS:])
    return folded


def _dir(key, default_parts):
    override = _model_settings().get(key)
    if override:
        return os.fspath(override)
    return model(*default_parts)


def checkpoints_dir():
    return _dir("checkpoints_dir", ("checkpoints",))


def diffusion_dir():
    return _dir("diffusion_dir", ("diffusion_models",))


def unet_dir():
    return _dir("unet_dir", ("unet",))


def background_dir():
    return _dir("background_dir", ("background",))


def segmentation_dir():
    return _dir("segmentation_dir", ("segmentation",))


def upscale_dir():
    return _dir("upscale_dir", ("upscale_models",))


def text_encoders_dir():
    return _dir("text_encoders_dir", ("text_encoders",))


def vae_dir():
    return _dir("vae_dir", ("vae",))


def loras_dir():
    return _dir("loras_dir", ("loras",))


def adapters_dir():
    return _dir("adapters_dir", ("adapters",))


def controlnet_dir():
    return _dir("controlnet_dir", ("controlnet",))


# Default first path part -> the folder function that can move it, so a family's
# default component files honor the same per-folder overrides the Folders tab
# edits (a text encoder under a moved ``text_encoders_dir``, and so on).
_COMPONENT_DIRS = {
    "checkpoints": checkpoints_dir,
    "diffusion_models": diffusion_dir,
    "unet": unet_dir,
    "text_encoders": text_encoders_dir,
    "vae": vae_dir,
    "loras": loras_dir,
    "adapters": adapters_dir,
    "controlnet": controlnet_dir,
    "background": background_dir,
    "upscale_models": upscale_dir,
}


def component_path(parts):
    """Resolve a default component path, honoring a per-folder override.

    ``parts`` is a ``(folder, filename)`` pair; a bare filename has no folder to
    override. The folder's override wins when set, so pointing ``text_encoders``
    elsewhere moves every default text encoder with it.
    """
    if not parts:
        return models_root()
    fn = _COMPONENT_DIRS.get(parts[0])
    if fn is not None:
        return os.path.join(fn(), *parts[1:])
    return model(*parts)


def background_file(label, default_parts):
    """Resolve one background-removal model, honoring a per-label override."""
    overrides = _model_settings().get("background")
    if isinstance(overrides, dict) and overrides.get(label):
        return os.fspath(overrides[label])
    return os.path.join(background_dir(), *default_parts)


def upscale_file(label, default_parts):
    """Resolve one upscale model, honoring a per-label override."""
    overrides = _model_settings().get("upscale")
    if isinstance(overrides, dict) and overrides.get(label):
        return os.fspath(overrides[label])
    return os.path.join(upscale_dir(), *default_parts)


def _list_files(directory, suffixes):
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    return [
        os.path.join(directory, n)
        for n in sorted(names)
        if n.lower().endswith(suffixes)
    ]


def list_checkpoints():
    """Return the stable-diffusion checkpoints under :func:`checkpoints_dir`."""
    return _list_files(checkpoints_dir(), (".safetensors", ".ckpt"))


def list_controlnets():
    """Return curated control adapters, plus any others in the ControlNet folder.

    Holds both the classic ControlNets (Stable Diffusion, Z-Image) and the Anima
    ControlNet-LLLite adapters; the detected kind tells them apart (see
    :mod:`engine.component_kinds`). A folder scan in Model Settings can curate
    files outside the configured folder, so those entries must be listed too.
    """
    found = [
        item["path"] for item in component_pools().get("controlnet", [])
    ]
    known = {os.path.normcase(os.path.abspath(path)) for path in found}
    for path in _list_files(controlnet_dir(), (".safetensors", ".ckpt", ".pth", ".pt")):
        key = os.path.normcase(os.path.abspath(path))
        if key not in known:
            found.append(path)
            known.add(key)
    return found


def list_upscale_files():
    """Return the upscale models under :func:`upscale_dir`."""
    return _list_files(upscale_dir(), (".onnx", ".safetensors", ".pth", ".pt"))


def list_adapters():
    """Return the run-time adapter files under :func:`adapters_dir`.

    The Anima reference adapter LoRAs (see :mod:`engine.anima_ref`) and the Krea 2
    AnyPaint functional adapter (see :mod:`engine.krea2_anypaint`).
    """
    return _list_files(adapters_dir(), (".safetensors", ".pt", ".ckpt", ".bin"))


def read_safetensors_keys(path):
    """Return the tensor key names from a ``.safetensors`` header (no tensors)."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
    header.pop("__metadata__", None)
    return set(header)


def arch_from_keys(keys):
    return "sdxl" if any(k.startswith("conditioner.embedders") for k in keys) else "sd15"


def checkpoint_arch(path):
    """Return ``"sdxl"`` or ``"sd15"`` for a stable-diffusion checkpoint file."""
    try:
        keys = read_safetensors_keys(path)
    except (OSError, ValueError, struct.error):
        return "sd15"
    return arch_from_keys(keys)


# Distilled / few-step SD finetunes need very different step and CFG defaults
# from a normal checkpoint, so :func:`family_defaults` looks the kind up by
# filename. Longest tokens come first so ``dmd2`` wins over ``dmd``.
_FINETUNE_TOKENS = (
    ("lightning", "lightning"),
    ("turbo", "turbo"),
    ("hyper", "hyper"),
    ("lcm", "lcm"),
    ("dmd2", "dmd2"),
    ("dmd", "dmd"),
)


def checkpoint_kind(path):
    """Classify a checkpoint's finetune kind from its filename.

    Returns one of ``"turbo"``, ``"lightning"``, ``"lcm"``, ``"hyper"``,
    ``"dmd"``/``"dmd2"`` or ``"standard"``. The guess only changes the default
    steps/CFG/sampler; every field stays editable in the generate panel.
    """
    name = os.path.basename(path or "").lower()
    for token, kind in _FINETUNE_TOKENS:
        if token in name:
            return kind
    return "standard"


def _is_nitrofusion(path):
    """Whether a filename reads as NitroFusion2 (``nitrofusion`` or ``nf2``).

    NitroFusion2's files are named ``nitrofusion2_*`` and ``nf2-*`` (the DiT and
    GGUF tiers) or ``nf2-dedistill-lora.safetensors``. The ``nf2`` token is only
    matched at the start of the name or after a ``-``/``_`` so an unrelated name
    like ``conf2`` does not collide.
    """
    name = os.path.basename(path or "").lower()
    return (
        "nitrofusion" in name
        or name.startswith("nf2")
        or "nf2-" in name
        or "nf2_" in name
    )


def z_image_kind(path):
    """Classify a Z-Image model file: ``"nitrofusion"`` or ``"z_image"``.

    NitroFusion2 is a 1-2 step adversarial distillation of Z-Image-Turbo; the
    guess only changes the default steps/CFG and the engine's sampling schedule
    (a linear sigma ramp at a higher flow shift), and every field stays editable
    in the generate panel.
    """
    return "nitrofusion" if _is_nitrofusion(path) else "z_image"


def _stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def classify_model(path, in_checkpoints=False):
    """Guess ``(group, family)`` for a model file from its name and folder.

    Shared by the settings scan and the Hugging Face downloader so both agree on
    which picker group a file belongs to. A checkpoints-folder file (or a
    ``.ckpt``) is Stable Diffusion; ``.gguf`` or a qwen-named file is Qwen-Image;
    everything else is Anima. The guess is always editable in the dialog.
    """
    name = os.path.basename(path).lower()
    if in_checkpoints or name.endswith(".ckpt"):
        return "Stable Diffusion", "stable_diffusion"
    if "flux" in name or "chroma" in name:
        return "Flux", "flux"
    if "krea2" in name or "krea-2" in name or "krea_2" in name:
        return "Krea 2", "krea2"
    if "moss" in name:
        return "MOSS Image", "moss_image"
    # MiniMax-H3-Image's DiT is a single-file ComfyUI repackage too, but it is a
    # different architecture (no audio stream) from the H3 video DiT below.
    if ("minimax_h3_image" in name or "minimax-h3-image" in name) and not (
        "vae" in name or "text_encoder" in name or "text-encoder" in name):
        return "MiniMax H3 Image", "h3_image"
    # Only the H3 DiT (FL2VA / Ref2VA) is a generation model; the Qwen3-VL-32B
    # text encoder and the video VAE share the "minimax_h3" token and must not
    # be read as models, so require a checkpoint marker.
    if ("minimax_h3" in name or "minimax-h3" in name) and (
        "fl2va" in name or "ref2va" in name or "fl2v" in name or "ref2v" in name
    ):
        return "MiniMax H3", "h3"
    if "z_image" in name or "zimage" in name or "z-image" in name or _is_nitrofusion(path):
        return "Z-Image", "z_image"
    # Fibo 2.0's model unit is a diffusers directory (or Hub repo id) whose name
    # carries "fibo" (``Fibo-2.0-dev``), not a transformer file.
    if "fibo" in name:
        return "Fibo 2.0", "fibo"
    if name.endswith(".gguf") or "qwen" in name:
        return "Qwen-Image", "qwen_image"
    return "Anima", "anima"


def is_segmentation_model(path):
    """Whether ``path`` is a SAM segmentation model rather than a picker model.

    Segmentation exports live under :func:`segmentation_dir` -- a support folder
    that is not a generation pipeline -- and are listed by
    :mod:`engine.segmentation`, not by the picker. A Hub download or folder scan
    that lands there must therefore not be curated as a model row, and
    :func:`model_entries` also drops such a row so an earlier misclassification
    (a SAM ONNX bundle filed as an Anima model) heals on the next read.
    """
    directory = os.path.normcase(os.path.abspath(segmentation_dir()))
    target = os.path.normcase(os.path.abspath(os.fspath(path)))
    return target == directory or target.startswith(directory + os.sep)


def list_models():
    """Grouped model options for the generation panel's picker.

    Returns ``[(group, [(label, family, path, components, loras, key), ...]), ...]``
    in the order the curated rows were stored. Only the ``models.entries``
    setting is listed: a file that merely exists in a scan folder is not offered
    until it is added in Model Settings, so the picker and the Models tab show
    the same rows. ``family`` is the value the engine's ``generate`` selects a
    pipeline with; ``path`` is the model file that family loads (a checkpoint
    for ``stable_diffusion``, the diffusion transformer otherwise);
    ``components`` is the per-model component assignment and ``loras`` its LoRA
    list. ``key`` is the row's identity for presets and recipes (a curated row's
    ``id``, else its ``path``), so two rows over one file stay distinct.
    """
    by_group = {}
    for item in model_entries():
        by_group.setdefault(item["group"], []).append((
            item["label"], item["family"], item["path"], item["components"],
            item["loras"], entry_key(item),
        ))
    known = [group for group, _family in MODEL_GROUPS]
    order = [group for group in known if group in by_group]
    order += [group for group in by_group if group not in known]
    return [(group, by_group[group]) for group in order]


# Selectable samplers/schedulers. Every entry has an implementation in
# engine.sampler (SAMPLERS / SCHEDULERS); the reference set follows ComfyUI so
# community workflows map 1:1. Defaults stay on the reference recipe below.
SAMPLERS = [
    "euler",
    "euler_ancestral",
    "euler_cfg_pp",
    "euler_ancestral_cfg_pp",
    "heun",
    "heunpp2",
    "dpm_2",
    "dpm_2_ancestral",
    "lms",
    "dpmpp_2s_ancestral",
    "dpmpp_2s_ancestral_cfg_pp",
    "dpmpp_2m",
    "dpmpp_2m_cfg_pp",
    "dpmpp_sde",
    "dpmpp_sde_gpu",
    "dpmpp_2m_sde",
    "dpmpp_2m_sde_heun",
    "dpmpp_2m_sde_gpu",
    "dpmpp_3m_sde",
    "dpmpp_3m_sde_gpu",
    "ddpm",
    "ddim",
    "ipndm",
    "ipndm_v",
    "deis",
    "lcm",
    "res_multistep",
    "res_multistep_ancestral",
    "res_multistep_cfg_pp",
    "res_multistep_ancestral_cfg_pp",
    "gradient_estimation",
    "gradient_estimation_cfg_pp",
    "er_sde",
]
SCHEDULERS = [
    "simple",
    "normal",
    "karras",
    "exponential",
    "ddim_uniform",
    "beta",
    "sgm_uniform",
    "linear_quadratic",
    "kl_optimal",
    "polyexponential",
    "laplace",
]
# The hires-fix second pass: how the finished latent is enlarged before it is
# re-denoised at ``upscale_denoise``. ``none`` disables it; the ``latent_*``
# methods interpolate the latent directly (in the sampler's normalized space)
# and ``pixel`` decodes, resizes and re-encodes through the family's VAE.
UPSCALE_TYPES = [
    "none",
    "latent_nearest",
    "latent_bilinear",
    "latent_bicubic",
    "latent_bislerp",
    "pixel",
]


# The config slot each family's curated model row fills with its model file, so
# the base file (the diffusion transformer, or a checkpoint) can be read from the
# first curated row of that family.
_FAMILY_BASE_SLOT = {
    "anima": "model",
    "qwen_image": "dit",
    "flux": "dit",
    "z_image": "dit",
    "krea2": "dit",
    "moss_image": "dit",
    "h3": "dit",
    "h3_image": "dit",
    "fibo": "model",
    "stable_diffusion": "checkpoint",
}


def _first_family_entry(family):
    """The curated model row for ``family`` (folders are ignored).

    Prefers the first row whose file is on disk so a stale link cannot shadow a
    working one; when none exist on disk the first row is still returned, so the
    link is used -- and reported missing if it is -- rather than a download
    folder path.
    """
    entries = [entry for entry in model_entries() if entry["family"] == family]
    for entry in entries:
        if os.path.isfile(entry["path"]):
            return entry
    return entries[0] if entries else None


def _pool_component(family, slot):
    """A slot's default file from the curated component pools, folders ignored.

    Uses the pool item whose detected kind is the family's intended
    implementation for the slot, so one slot never grabs another's file (an Anima
    text-encoder slot does not pick up a stray CLIP-L). Returns ``None`` when no
    pool item fits, so the caller falls back to the download folder.
    """
    pool = component_kinds.SLOT_POOL.get(slot)
    wanted = component_kinds.SLOT_KIND.get((family, slot))
    if pool is None or wanted is None:
        return None
    category = component_kinds.POOL_CATEGORY.get(pool)
    for item in component_pools().get(pool, []):
        if component_kinds.detect(item["path"], category) == wanted:
            return item["path"]
    return None


def _family_files(section_key, components):
    """Resolve a family's default files from the curated links.

    Precedence per slot: an explicit ``models.<family>`` override, then the
    curated data -- the base file from the family's first curated model row, and a
    component from that row's assignment or else the matching component pool --
    and only then the models-root folder (the download location). Folders thus
    stay a download target, not the source of truth.
    """
    overrides = _model_settings().get(section_key)
    overrides = overrides if isinstance(overrides, dict) else {}
    entry = _first_family_entry(section_key)
    linked = entry.get("components", {}) if entry else {}
    base_slot = _FAMILY_BASE_SLOT.get(section_key)
    files = {}
    for key, parts in components.items():
        value = overrides.get(key)
        if not value and entry is not None and key == base_slot:
            value = entry["path"]
        if not value:
            value = linked.get(key)
        if not value:
            value = _pool_component(section_key, key)
        files[key] = os.fspath(value) if value else component_path(parts)
    return files


def _defaults():
    files = _family_files("anima", _ANIMA_COMPONENTS)
    return {
        "family": "anima",
        "model": files["model"],
        "checkpoint": None,
        "clip": files["clip"],
        "vae": files["vae"],
        "steps": 10,
        "cfg": 1.0,
        "sampler": "euler",
        "scheduler": "simple",
        "beta_alpha": 0.5,
        "beta_beta": 0.7,
        "karras_rho": 7.0,
        "poly_rho": 1.0,
        "laplace_mu": 0.0,
        "laplace_beta": 0.5,
        "lq_threshold": 0.025,
        "lq_linear_steps": 0,
        "denoise": 1.0,
        "width": 1024,
        "height": 1024,
        "negative": "",
        "seed": 0,
        "upscale_type": "none",
        "upscale_by": 2.0,
        "upscale_denoise": 0.5,
    }


def _qwen_defaults():
    files = _family_files("qwen_image", _QWEN_COMPONENTS)
    return {"dit": files["dit"], "text": files["text"], "vae": files["vae"]}


def _flux_defaults():
    files = _family_files("flux", _FLUX_COMPONENTS)
    return {
        "dit": files["dit"], "clip": files["clip"],
        "t5": files["t5"], "vae": files["vae"],
    }


def _zimage_defaults():
    files = _family_files("z_image", _ZIMAGE_COMPONENTS)
    return {"dit": files["dit"], "text": files["text"], "vae": files["vae"]}


def _krea2_defaults():
    files = _family_files("krea2", _KREA2_COMPONENTS)
    return {"dit": files["dit"], "text": files["text"], "vae": files["vae"]}


def _moss_defaults():
    files = _family_files("moss_image", _MOSS_COMPONENTS)
    return {"dit": files["dit"], "text": files["text"], "vae": files["vae"]}


def _h3_defaults():
    files = _family_files("h3", _H3_COMPONENTS)
    return {"dit": files["dit"], "text": files["text"], "vae": files["vae"]}


def _h3_image_defaults():
    files = _family_files("h3_image", _H3_IMAGE_COMPONENTS)
    return {"dit": files["dit"], "text": files["text"], "vae": files["vae"]}


def _fibo_defaults():
    files = _family_files("fibo", _FIBO_COMPONENTS)
    return {"model": files["model"]}


DEFAULTS = _defaults()
QWEN_DEFAULTS = _qwen_defaults()
FLUX_DEFAULTS = _flux_defaults()
ZIMAGE_DEFAULTS = _zimage_defaults()
KREA2_DEFAULTS = _krea2_defaults()
MOSS_DEFAULTS = _moss_defaults()
H3_DEFAULTS = _h3_defaults()
H3_IMAGE_DEFAULTS = _h3_image_defaults()
FIBO_DEFAULTS = _fibo_defaults()


def reload():
    """Rebuild the resolved model defaults after the model settings change."""
    DEFAULTS.clear()
    DEFAULTS.update(_defaults())
    QWEN_DEFAULTS.clear()
    QWEN_DEFAULTS.update(_qwen_defaults())
    FLUX_DEFAULTS.clear()
    FLUX_DEFAULTS.update(_flux_defaults())
    ZIMAGE_DEFAULTS.clear()
    ZIMAGE_DEFAULTS.update(_zimage_defaults())
    KREA2_DEFAULTS.clear()
    KREA2_DEFAULTS.update(_krea2_defaults())
    MOSS_DEFAULTS.clear()
    MOSS_DEFAULTS.update(_moss_defaults())
    H3_DEFAULTS.clear()
    H3_DEFAULTS.update(_h3_defaults())
    H3_IMAGE_DEFAULTS.clear()
    H3_IMAGE_DEFAULTS.update(_h3_image_defaults())
    FIBO_DEFAULTS.clear()
    FIBO_DEFAULTS.update(_fibo_defaults())

# Per-finetune defaults for the stable_diffusion family. A normal checkpoint
# keeps the reference 28-step / CFG 6 recipe; distilled models want far fewer
# steps and near-1.0 CFG, and LCM additionally needs the ``lcm`` sampler.
_SD_FINETUNE_DEFAULTS = {
    "standard": {"steps": 28, "cfg": 6.0, "sampler": "euler", "scheduler": "karras"},
    "turbo": {"steps": 6, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "lightning": {"steps": 8, "cfg": 1.5, "sampler": "euler", "scheduler": "sgm_uniform"},
    "hyper": {"steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "sgm_uniform"},
    "lcm": {"steps": 8, "cfg": 1.5, "sampler": "lcm", "scheduler": "sgm_uniform"},
    "dmd2": {"steps": 6, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
    "dmd": {"steps": 6, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"},
}

# The built-in recipe name every family offers.
DEFAULT_RECIPE_NAME = "Default"


def family_defaults(family, path=None):
    """Per-family generation defaults (steps/cfg/sampler/scheduler/size).

    For ``stable_diffusion``, ``path`` selects the finetune-specific recipe
    (Turbo/Lightning/LCM/Hyper/DMD) via :func:`checkpoint_kind`; without a path
    the standard checkpoint recipe is returned.
    """
    if family == "anima":
        return {
            "steps": 10, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "qwen_image":
        return {
            "steps": 20, "cfg": 3.0, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "flux":
        # Flux dev bakes guidance into the model (guidance_scale 3.5), so CFG
        # stays at 1.0 and only the step count matters.
        return {
            "steps": 20, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "z_image":
        if z_image_kind(path) == "nitrofusion":
            # NitroFusion2: a 1-2 step adversarial distillation of Z-Image-Turbo,
            # guidance-free, sampled on a linear flow schedule (see
            # :func:`engine.sampler.linear_sigmas`) at a higher shift.
            return {
                "steps": 2, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
                "width": 1024, "height": 1024,
            }
        # Z-Image-Turbo: 8 steps, guidance-free (CFG 1.0), res_multistep.
        return {
            "steps": 8, "cfg": 1.0, "sampler": "res_multistep", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "krea2":
        # Turbo is few-step and guidance-free (guidance_scale 0 -> CFG 1.0); Raw
        # is the base checkpoint (28 steps, guidance_scale 4.5 -> CFG 5.5).
        turbo = "turbo" in os.path.basename(path or "").lower()
        if turbo or not path:
            return {
                "steps": 8, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
                "width": 1024, "height": 1024,
            }
        return {
            "steps": 28, "cfg": 5.5, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "moss_image":
        # Reference recipe: 50 steps, CFG 3, flow-matching Euler at shift 5.
        return {
            "steps": 50, "cfg": 3.0, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "h3":
        # MiniMax H3 single-frame still (provisional: the engine port is not
        # wired yet). The no-Turbo reference is 50 steps; H3-Base is
        # CFG-distilled, so guidance stays at 1.0.
        return {
            "steps": 50, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    if family == "h3_image":
        # MiniMax-H3-Image is distilled to a fixed 6-step PDD grid with no CFG.
        # 2048x2048 is the released model's default minimum generation area
        # (~4 MP); the effective working range goes up to ~16 MP.
        return {
            "steps": 6, "cfg": 1.0, "sampler": "euler", "scheduler": "simple",
            "width": 2048, "height": 2048,
        }
    if family == "fibo":
        # Fibo 2.0 Dev is the full, non-distilled base model: the reference
        # recipe is 50 steps at guidance 5.0 (Edit Turbo, a separate checkpoint,
        # is the few-step guidance-free variant). The pipeline runs its own
        # sampler/scheduler, so those fields are inert for this family.
        return {
            "steps": 50, "cfg": 5.0, "sampler": "euler", "scheduler": "simple",
            "width": 1024, "height": 1024,
        }
    # stable_diffusion (SD 1.x / SDXL single-file checkpoints).
    defaults = _SD_FINETUNE_DEFAULTS.get(
        checkpoint_kind(path) if path else "standard",
        _SD_FINETUNE_DEFAULTS["standard"],
    )
    return {**defaults, "width": 1024, "height": 1024}


def family_recipes(family, path=None):
    """The built-in recipe for ``family``: ``{DEFAULT_RECIPE_NAME: {field: value}}``.

    Each family offers exactly one read-only ``Default`` recipe, taken from
    :func:`family_defaults` -- which for ``stable_diffusion`` follows the
    checkpoint's finetune -- so the Generate panel's recipe dropdown gains one
    built-in entry per family. Picking it applies its fields (including width and
    height) and it then becomes the model's sticky preset. A saved recipe of the
    same name shadows it.
    """
    defaults = family_defaults(family, path)
    recipe = {
        field: defaults[field]
        for field in _RECIPE_FIELDS
        if defaults.get(field) is not None
    }
    recipe.setdefault("denoise", 1.0)
    return {DEFAULT_RECIPE_NAME: recipe}


# Sampler fields that stick to a model. Switching back to a model restores the
# values last used with it, so a tuned Turbo checkpoint does not silently take
# the family defaults again.
_PRESET_FIELDS = (
    "steps", "cfg", "sampler", "scheduler", "denoise",
    "beta_alpha", "beta_beta",
    "karras_rho", "poly_rho", "laplace_mu", "laplace_beta",
    "lq_threshold", "lq_linear_steps",
)

# A named recipe captures the sampler fields plus the output size, so recalling a
# recipe restores both how a render is sampled and how big it is. Size stays out
# of the sticky preset: the generation size otherwise follows the active canvas,
# and a model switch should not silently resize the render.
_RECIPE_FIELDS = _PRESET_FIELDS + ("width", "height")


def _clean_fields(values, fields=_PRESET_FIELDS):
    """Keep only the known ``fields`` from a stored dict."""
    if not isinstance(values, dict):
        return {}
    return {field: values[field] for field in fields if values.get(field) is not None}


def _apply_fields(entry, values):
    """Merge ``values`` into ``entry``; ``None``/blank values clear a field."""
    for field in _PRESET_FIELDS:
        if field not in values:
            continue
        value = values[field]
        if value is None or value == "":
            entry.pop(field, None)
        else:
            entry[field] = value


def _load_generation():
    """Return ``(models, recipes)`` from the ``generation`` setting, cleaned.

    ``models`` maps a model key (a curated row's identity, else its path) to its
    sticky preset; ``recipes`` maps a key to its ``{name: preset}`` named
    recipes.
    """
    section = settings.section("generation")
    raw_models = section.get("models")
    models = {}
    if isinstance(raw_models, dict):
        for path, values in raw_models.items():
            preset = _clean_fields(values)
            if preset:
                models[os.fspath(path)] = preset
    raw_recipes = section.get("recipes")
    recipes = {}
    if isinstance(raw_recipes, dict):
        for path, by_name in raw_recipes.items():
            if not isinstance(by_name, dict):
                continue
            cleaned = {}
            for name, values in by_name.items():
                preset = _clean_fields(values, _RECIPE_FIELDS)
                if preset:
                    cleaned[name] = preset
            if cleaned:
                recipes[os.fspath(path)] = cleaned
    return models, recipes


def _write_generation(models, recipes):
    """Persist both generation maps, dropping the section when both are empty."""
    section = {}
    if models:
        section["models"] = models
    if recipes:
        section["recipes"] = recipes
    settings.set_section("generation", section or None)


def model_presets():
    """Per-model sticky presets from the ``generation`` setting.

    Returns ``{key: {field: value}}`` carrying only the known
    :data:`_PRESET_FIELDS`; malformed entries are dropped.
    """
    return _load_generation()[0]


def model_preset(key, path=None):
    """The stored ``{field: value}`` preset for ``key`` (``{}`` when none).

    ``path`` is the model file, consulted as a fallback key so a preset saved
    before the row carried an identity is still found.
    """
    if not key:
        return {}
    models = _load_generation()[0]
    if key in models:
        return dict(models[key])
    if path and path != key and path in models:
        return dict(models[path])
    return {}


def save_model_preset(key, values):
    """Persist a model's sampler preset, dropping the entry when it is empty.

    ``values`` is the full current set of :data:`_PRESET_FIELDS`; a blank value
    clears that field. The sibling recipes map is preserved.
    """
    if not key:
        return
    models, recipes = _load_generation()
    key = os.fspath(key)
    entry = models.get(key, {})
    _apply_fields(entry, values)
    if entry:
        models[key] = entry
    else:
        models.pop(key, None)
    _write_generation(models, recipes)


def model_recipes(key, path=None):
    """Named recipes stored for ``key``: ``{name: {field: value}}``.

    ``path`` is the model file, consulted as a fallback key so a recipe saved
    before the row carried an identity is still found.
    """
    if not key:
        return {}
    recipes = _load_generation()[1]
    if key in recipes:
        return dict(recipes[key])
    if path and path != key and path in recipes:
        return dict(recipes[path])
    return {}


def save_model_recipe(key, name, values):
    """Store (or overwrite) a model's named recipe from the given field set."""
    if not key or not name:
        return
    models, recipes = _load_generation()
    key = os.fspath(key)
    entry = recipes.get(key, {})
    preset = _clean_fields(values, _RECIPE_FIELDS)
    if preset:
        entry[name] = preset
    else:
        entry.pop(name, None)
    if entry:
        recipes[key] = entry
    else:
        recipes.pop(key, None)
    _write_generation(models, recipes)


def delete_model_recipe(key, name):
    """Remove one named recipe from a model, dropping empty maps."""
    if not key or not name:
        return
    models, recipes = _load_generation()
    key = os.fspath(key)
    entry = recipes.get(key, {})
    entry.pop(name, None)
    if entry:
        recipes[key] = entry
    else:
        recipes.pop(key, None)
    _write_generation(models, recipes)


def all_model_recipes():
    """Every stored recipe, as ``{path: {name: {field: value}}}``."""
    return _load_generation()[1]


def set_model_recipes(recipes):
    """Replace the whole recipe map (used by the settings editor).

    The sibling sticky-preset map is preserved; malformed entries are dropped.
    """
    models, _existing = _load_generation()
    cleaned = {}
    if isinstance(recipes, dict):
        for path, by_name in recipes.items():
            if not isinstance(by_name, dict):
                continue
            names = {}
            for name, values in by_name.items():
                preset = _clean_fields(values, _RECIPE_FIELDS)
                if preset:
                    names[str(name)] = preset
            if names:
                cleaned[os.fspath(path)] = names
    _write_generation(models, cleaned)
