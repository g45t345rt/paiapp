"""Fibo 2.0 (Bria) family.

Fibo 2.0 is an 8B single-stream flow-matching DiT operating in the FLUX.2 VAE
latent space, prompted by a Qwen3-VL-4B encoder through a two-layer Perceiver
"gist" resampler. Unlike the other families in :mod:`engine.pipeline`, whose
architectures are hand-ported so they can load ComfyUI GGUF / int8 / single-file
weights with low-RAM disk streaming, this family wraps the maintained
``diffusers`` implementation, :class:`diffusers.BriaFibo2Pipeline`, which ships
the transformer, the Qwen3-VL text encoder, the FLUX.2 VAE and the shifted
sigmoid scheduler together.

The model unit is therefore a *diffusers-format directory*: the local folder of a
``briaai/Fibo-2.0-dev`` download, or a Hugging Face repo id that is fetched on
demand. One path loads the whole pipeline, so the family has no per-component
slots. ``diffusers`` >= 0.42 is required for ``BriaFibo2Pipeline``.

Only ``diffusers``, ``torch`` and ``Pillow`` are used here.
"""

import os

import torch

from .cancel import CancelToken


def _pipeline_class():
    """The installed ``BriaFibo2Pipeline``, or raise a clear load error.

    ``BriaFibo2Pipeline`` only exists in ``diffusers`` >= 0.42; on older versions
    it resolves to a stub that raises on use, so the instantiation is what the
    caller should see fail.
    """
    import diffusers

    cls = getattr(diffusers, "BriaFibo2Pipeline", None)
    if cls is None:
        raise RuntimeError(
            "Fibo 2.0 needs diffusers>=0.42 (no BriaFibo2Pipeline found)."
        )
    return cls


def is_repo_id(source):
    """Whether ``source`` looks like a Hub repo id (``owner/name``), not a path.

    A source that exists on disk is always treated as a local diffusers
    directory; anything else containing a ``/`` is assumed to be a repo id for
    ``from_pretrained`` to fetch (public repos need no credentials, gated ones
    read ``HF_TOKEN`` from the environment, which diffusers does itself).
    """
    text = os.fspath(source or "").strip()
    if not text:
        return False
    if os.path.exists(text):
        return False
    return "/" in text and not os.path.isabs(text)


def load_pipeline(source, device="cpu", dtype=torch.bfloat16, progress=None,
                  cancel=None):
    """Build and place a :class:`BriaFibo2Pipeline` from ``source`` (a dir or repo).

    ``progress(frac, label)`` is called at each phase boundary, and a
    :class:`~engine.cancel.CancelToken` ``cancel`` is honoured between phases
    (``from_pretrained`` itself is not interruptible). On CUDA the pipeline is
    offloaded to the GPU one component at a time
    (``enable_model_cpu_offload``), so the 8B DiT and the ~4B VLM text encoder can
    share a card; on CPU it is left in place.
    """
    if not source:
        raise RuntimeError("No Fibo 2.0 model selected.")
    if cancel is None:
        cancel = CancelToken()
    cls = _pipeline_class()

    cancel.throw_if_cancelled()
    if progress is not None:
        progress(0.05, "loading pipeline")
    pipe = cls.from_pretrained(os.fspath(source), dtype=dtype)

    cancel.throw_if_cancelled()
    if progress is not None:
        progress(0.9, "placing pipeline")
    if str(device).startswith("cuda") and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    if progress is not None:
        progress(1.0, "ready")
    return pipe
