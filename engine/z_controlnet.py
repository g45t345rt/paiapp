"""Z-Image ControlNet-Union (Alibaba-PAI "Fun") loader.

Z-Image-Turbo's control model is a DiT ControlNet, not a UNet one: the control
image is VAE-encoded to the same 16-channel latent, patch-embedded, refined by a
couple of control noise-refiner blocks, then run through a handful of control
blocks. Each control block emits a skip that the matching main DiT layer adds
back, scaled by the conditioning strength -- the branch lives inside
:class:`engine.z_image.NextDiT` so the DiT's own block offload covers it.

The single-file checkpoints ship no config, so everything is read from the
tensor keys: the number of control blocks gives the spaced main-layer
``places`` (``i * n_layers // count``), the patch embedder's width gives
``control_in_dim``, and the refiner block count gives ``n_refiner_layers``. The
weights use diffusers names (``attention.to_q``/``to_out.0``) while the DiT uses
ComfyUI/Lumina names (a fused ``attention.qkv``), so the attention weights are
remapped on load.

Only the union (``control_in_dim == 16``) layout is supported; the two-stage
variants (e.g. the SAM control model, ``control_in_dim == 33``) are rejected with
a clear error.
"""

import json
import os
import struct

import torch

_PATCH = 2
_MAX_HEADER = 100_000_000

# The main-path tensor groups a control branch owns (used to scope the load).
_CONTROL_PREFIXES = ("control_x_embedder.", "control_noise_refiner.", "control_layers.")


def _shapes(path):
    """``{tensor name: shape tuple}`` from a ``.safetensors`` header (no data)."""
    try:
        with open(path, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size <= 0 or size > _MAX_HEADER:
                return {}
            header = json.loads(handle.read(size))
    except (OSError, ValueError, struct.error):
        return {}
    header.pop("__metadata__", None)
    return {
        key: tuple(value["shape"])
        for key, value in header.items()
        if isinstance(value, dict) and isinstance(value.get("shape"), list)
    }


def _indices(shapes, prefix):
    found = set()
    for key in shapes:
        if key.startswith(prefix):
            head = key[len(prefix):].split(".", 1)[0]
            if head.isdigit():
                found.add(int(head))
    return found


def config_from_shapes(shapes, n_layers):
    """Resolve the control config from a checkpoint's tensor shapes."""
    layer_ids = _indices(shapes, "control_layers.")
    if not layer_ids:
        raise RuntimeError("Not a Z-Image ControlNet (no control_layers tensors).")
    embed = shapes.get("control_all_x_embedder.2-1.weight")
    if embed is None:
        raise RuntimeError("Not a Z-Image ControlNet (no control_all_x_embedder).")
    count = len(layer_ids)
    step = max(1, n_layers // count)
    return {
        "places": [index * step for index in range(count)],
        "control_in_dim": int(embed[1]) // (_PATCH * _PATCH),
        "n_refiner_layers": len(_indices(shapes, "control_noise_refiner.")),
        "two_stage": any(
            key.startswith("control_noise_refiner.0.before_proj") for key in shapes
        ),
    }


def _to_app_keys(state):
    """Rename the file's diffusers-style keys to the DiT's fused-attention names."""
    renamed = {}
    qkv = {}
    for key, tensor in state.items():
        if key == "control_all_x_embedder.2-1.weight":
            renamed["control_x_embedder.weight"] = tensor
        elif key == "control_all_x_embedder.2-1.bias":
            renamed["control_x_embedder.bias"] = tensor
        elif key.endswith(".attention.to_q.weight"):
            qkv.setdefault(key[: -len("to_q.weight")], {})["q"] = tensor
        elif key.endswith(".attention.to_k.weight"):
            qkv.setdefault(key[: -len("to_k.weight")], {})["k"] = tensor
        elif key.endswith(".attention.to_v.weight"):
            qkv.setdefault(key[: -len("to_v.weight")], {})["v"] = tensor
        elif key.endswith(".attention.to_out.0.weight"):
            renamed[key[: -len("to_out.0.weight")] + "out.weight"] = tensor
        elif key.endswith(".attention.norm_q.weight"):
            renamed[key[: -len("norm_q.weight")] + "q_norm.weight"] = tensor
        elif key.endswith(".attention.norm_k.weight"):
            renamed[key[: -len("norm_k.weight")] + "k_norm.weight"] = tensor
        else:
            renamed[key] = tensor
    for prefix, parts in qkv.items():
        renamed[prefix + "qkv.weight"] = torch.cat(
            [parts["q"], parts["k"], parts["v"]], dim=0
        )
    return renamed


def load(dit, path, device="cpu", dtype=torch.bfloat16):
    """Build the control branch on ``dit`` and load its weights from ``path``.

    Returns ``(missing, unexpected)`` key lists scoped to the control branch. The
    branch is placed on the DiT's head/storage devices so block offload covers it.
    """
    from safetensors.torch import load_file

    state = load_file(path, device="cpu")
    config = config_from_shapes({k: v.shape for k, v in state.items()}, len(dit.layers))
    if config["two_stage"]:
        raise RuntimeError(
            f"{os.path.basename(path)} is a two-stage Z-Image ControlNet "
            "(control_in_dim > 16); only the ControlNet-Union layout is supported."
        )

    rebuild = (
        dit.control_places is None
        or dit.control_in_dim != config["control_in_dim"]
        or len(dit.control_layers) != len(config["places"])
    )
    if rebuild:
        dit.build_control(
            config["places"], config["control_in_dim"],
            n_refiner_layers=config["n_refiner_layers"], dtype=dtype,
        )

    renamed = _to_app_keys(state)
    expected = {
        name for name, _ in dit.named_parameters()
        if name.startswith(_CONTROL_PREFIXES)
    }
    provided = {name for name in renamed if name.startswith(_CONTROL_PREFIXES)}
    missing = sorted(expected - provided)
    unexpected = sorted(provided - expected)
    dit.load_state_dict({k: v for k, v in renamed.items() if k in expected}, strict=False)

    dit.place_control()
    branch = dit.control_modules()
    branch.eval()
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return missing, unexpected
