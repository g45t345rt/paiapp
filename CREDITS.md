# Credits and Third-Party Licenses

Copyright (C) 2026  g45t345rt

This app is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version. See the `LICENSE` file for the full text.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE.

---

## Why this project is GPL-3.0

Parts of `engine/` are ports of source code from
[ComfyUI](https://github.com/comfyanonymous/ComfyUI), which is distributed under
the GNU General Public License v3.0. A port of GPL-3.0 code is a derivative
work, so this application as a whole is released under GPL-3.0.

ComfyUI itself vendors some files adapted from permissively licensed upstreams
(Apache-2.0, MIT); those individual ports remain available under their original
terms and are listed separately below. Apache-2.0 and MIT are compatible with
GPL-3.0, so they can be combined in this work.

---

## Ported from ComfyUI (GPL-3.0)

These modules reimplement or port logic from ComfyUI files that carry no
separate per-file license and are therefore covered by ComfyUI's GPL-3.0:

| This project | Ported from |
| --- | --- |
| `engine/prompt_weights.py` | `comfy/sd1_clip.py` (`token_weights`, `parse_parentheses`) |
| `engine/diffusion_samplers.py` | `comfy/samplers.py`, `comfy/k_diffusion` (samplers, schedulers) |
| `engine/preview.py` | ComfyUI `latent_formats` (latent2rgb coefficient tables) |
| `engine/sampler.py` | ComfyUI `ModelSamplingDiscreteFlow` sampling |
| `engine/anima.py` | `comfy/ldm/anima/model.py` |
| `engine/h3.py` | `comfy/ldm/minimax/model.py` |
| `engine/cosmos.py` | Cosmos-Predict2 DiT as used by ComfyUI (`comfy/ldm/cosmos`) |
| `engine/anima_ref.py` | ComfyUI "Anima Ref Adapter" (`ElectronGirl/ComfyUI-Anima-Ref-Adapter`) |
| `engine/seedvr2/` | `comfy/ldm/seedvr` (NaDiT transformer, causal video VAE, colour fix) |

ComfyUI is Copyright (C) Comfy Org and contributors, licensed GPL-3.0.
Upstream: https://github.com/comfyanonymous/ComfyUI

---

## Ported from permissively licensed upstreams

These modules are ports of files that ComfyUI and others adopted from
Apache-2.0 / MIT projects. Keep the original notices when redistributing.

| This project | Origin | License |
| --- | --- | --- |
| `engine/flux.py` | Black Forest Labs Flux via `comfy/ldm/flux` | Apache-2.0 |
| `engine/z_image.py` | Lumina-2 / Z-Image via `comfy/ldm/lumina` (Tongyi) | Apache-2.0 |
| `engine/krea2.py` | Krea 2 / HF Diffusers via `comfy/ldm/krea2` | Apache-2.0 |
| `engine/qwen_image.py` | Qwen-Image via ComfyUI / HF Diffusers | Apache-2.0 |
| `engine/qwen_vae21.py` | Alibaba Wan Team VAE via `comfy/ldm/wan/vae2_2` | Apache-2.0 |
| `engine/h3.py`, `engine/h3_layout.py`, `engine/h3_sampler.py`, `engine/h3_vae.py` | MiniMax H3 / H3-Image via ComfyUI + Diffusers | Apache-2.0 |
| `engine/moss_image.py`, `engine/moss_text.py`, `engine/qwen_text.py` | HF Diffusers reference implementations | Apache-2.0 |
| `engine/gguf.py` | `city96/ComfyUI-GGUF` | Apache-2.0 |
| `engine/gguf.py` (legacy/quants) | `WeeLLM` `io/ggufs/gguf_dequant.py` | MIT |
| `engine/diffusion_samplers.py` (Heun++2) | `sd-webui-samplers-scheduler` | MIT |

---

## Bundled assets

- `paintapp/assets/models/Superhero_*.gltf` + `.bin` — "Universal Base Characters"
  by Quaternius (https://quaternius.com), licensed CC0 1.0 (public domain).

---

## Runtime dependencies

These are installed separately and are not part of this source distribution.
Consult each project for its license:

| Package | License |
| --- | --- |
| `Pillow` | HPND / MIT-like |
| `numpy`, `scipy` | BSD-3-Clause |
| `moderngl` | MIT |
| `onnxruntime` | MIT |
| `gguf` | MIT |
| `torch`, `torchvision` | BSD-3-Clause |
| `diffusers`, `transformers`, `accelerate`, `safetensors`, `huggingface_hub`, `sentencepiece` | Apache-2.0 |
| `psutil` | BSD-3-Clause |
| `xformers` | BSD-3-Clause |
| `tkinter` | PSF (Python standard library) |

---

## AI model weights

This application does not ship AI model weights. Weights are downloaded at
runtime by the user and are governed by their own licenses (for example,
RMBG-2.0 / RMBG-1.4 and FLUX.2-dev are non-commercial). Those licenses apply to
the weights and their outputs, not to this application's source code.
