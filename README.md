# <img src="pai_icon.png" alt="Pai" width="48" height="48"> Pai

[![Release](https://github.com/g45t345rt/paiapp/actions/workflows/release.yml/badge.svg)](https://github.com/g45t345rt/paiapp/actions/workflows/release.yml)
[![Version](https://img.shields.io/github/v/release/g45t345rt/paiapp)](https://github.com/g45t345rt/paiapp/releases)
[![License: GPL-3.0](https://img.shields.io/github/license/g45t345rt/paiapp)](LICENSE)

A desktop painting and image-editing app with a built-in AI image generator. It has a layered canvas, brushes, selections, text, vector tools and poses, plus generation, upscaling and background removal that run locally.

Built with Python and Tkinter.

![Pai](demo_header.png)

## Features

**Canvas and layers**
- Layered canvas with groups, layer masks, blend modes, link layers, move-to-canvas/group, and rasterize
- Per-canvas assets and a gallery, plus a full undo/redo history panel
- Grid, off-canvas opacity, customizable sidebar panels, and keyboard shortcuts

**Painting tools**
- Brush, Erase, Fill, Pick, Clone, Smudge, Dodge, and Burn
- Liquify, Warp, and Puppet Warp for freeform deformation
- Symmetry modes for mirrored strokes

**Selection and masks**
- Rectangle, Lasso, Magic Wand, and Segment AI (click-to-segment) selection modes
- Mask editing tools for inpaint and mask layers

**Vector, text, and shapes**
- Path and Vector tools, Shape tool with fill/stroke, and a Text tool
- Gradient tool with linear, reflected, radial, square, diamond, and conical types

**Transform**
- Move, Crop, and Transform, plus canvas-level transforms

**3D poses**
- A built-in 3D pose tool for sketching figures and props

**Filters**
- Adjustments, Color, Blur & Sharpen, Stylize, Pixel & Color, and Manga categories

**AI tools**
- Image generation across nine model families (see [Supported models](#supported-models))
- Reference images, ControlNet, and Region Prompts for guided generation
- Remove Background, Upscale, and a job queue for batch work
- Running and cached models are shown in the Loaded Models panel

## Requirements

- Python 3.12–3.14 on your PATH
- Optional NVIDIA or AMD GPU for the AI features (CPU works too)

If Python 3.12–3.14 is not found, the launcher offers to download and install it for you.

## Run

On Windows:

```
run.bat
```

On Linux/macOS (or Git Bash/WSL):

```
sh run.sh
```

The first run creates a private `.venv` next to the app and installs the dependencies; later runs reuse it. No other Python environment is needed.

## Launcher options

```
run.bat [--core-only] [--reinstall] [--setup-only] [--rocm] [--] [app args]
```

- `--core-only` – install the paint UI only, skip the AI engine
- `--reinstall` – rebuild the environment from scratch
- `--setup-only` – set up the environment, then exit
- `--rocm` – force the AMD ROCm build instead of auto-detecting

The compute backend is auto-detected on first run: an NVIDIA GPU installs the CUDA build, an AMD GPU installs the ROCm build, macOS uses the CPU build, and an unrecognised GPU falls back to CUDA (which also runs on CPU). An AMD GPU whose target is unsupported or cannot be detected uses the CPU build instead; set `PAI_BACKEND` to `cuda`, `cpu` or `rocm` to override the detection.

## Supported models

Generation runs through the model family you pick in the Generate panel:

| Family | Notes |
| --- | --- |
| Anima | Cosmos-Predict2 DiT with a T5 adapter, aesthetic-tuned |
| Qwen-Image 2.1 | Qwen3-VL text encoder, GGUF quantized DiT |
| Flux | FLUX.1 dev |
| Z-Image | Z-Image Turbo |
| Krea 2 | Turbo DiT, includes AnyPaint masked editing |
| MOSS Image | Instruction-tuned image model with JSON layout prompts |
| MiniMax H3 | Qwen3-VL-32B text encoder with video VAE |
| Fibo 2.0 | Bria Fibo 2.0 |
| Stable Diffusion | SD 1.x / 2.x / SDXL, including Pony/Illustrious-style finetunes |

Each family loads its components (DiT, text encoder, VAE) from separate files, so you can mix and match quantizations. Model Settings lets you point any slot at a custom file.

The AI tools use additional local models:

- **Upscaling** – Real-ESRGAN (RRDBNet), RCAN, SeedVR2 diffusion, and ONNX exports
- **Background removal** – RMBG-2.0 (BiRefNet, full or quantized), BiRefNet general, U2Net (full/portrait/lite), and MODNet portrait matting
- **Segmentation** – SAM ONNX encoder/decoder pairs for click-to-segment
- **ControlNet** – ControlNet for SD 1.x / SDXL, Z-Image ControlNet, and Anima LLLite
- **Adapters/LoRA** – standard LoRA plus the Anima reference adapter and Krea 2 AnyPaint

## Project files

Projects are saved as `.pai` files: a zip archive with a small JSON manifest and one PNG per layer, so your work stays portable and easy to inspect.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to set up a development environment and run the tests.

## Community

Join the Discord server to ask questions, share your work and get help.

<a href='https://discord.gg/bU6RzjwxFX' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://img.shields.io/badge/Discord-Join%20the%20server-5865F2?logo=discord&logoColor=white' alt='Join the Discord server' /></a>

## Support

Pai is free and open source. If you find it useful, you can support development with a donation on Ko-fi.

<a href='https://ko-fi.com/O5I829R7BD' target='_blank'><img height='100' style='border:0px;height:100px;' src='https://storage.ko-fi.com/cdn/brandasset/v2/support_me_on_kofi_badge_dark.png' border='0' alt='Support me on Ko-fi' /></a>

## Disclaimer

This project was built primarily with AI coding tools. It is provided, without warranty of any kind.

## License

GPL-3.0. See [LICENSE](LICENSE) and [CREDITS.md](CREDITS.md).

Copyright (C) 2026 g45t345rt
