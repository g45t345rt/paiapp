"""Self-contained inference engine for the Paint app.

No ComfyUI code is imported. Generation uses only ``torch``, ``transformers``,
``diffusers``, ``safetensors``, ``numpy`` and ``Pillow``. Background removal
(:mod:`engine.background`) additionally uses ``onnxruntime`` for an exported
ONNX graph, or ``torch`` for a PyTorch checkpoint: the RMBG-1.4 ISNet network in
:mod:`engine.rmbg14` or the BiRefNet network in :mod:`engine.birefnet`.
:mod:`engine.hub` is the only network module (the Hugging Face model browser)
and uses just the standard library.
"""

from . import paths  # noqa: F401

__all__ = ["paths"]
