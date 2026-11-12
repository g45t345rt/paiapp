# Model Download Guide

## Quick start

1. Open **Model Settings** and set the models root folder.
2. Use **Download model wizard...** to pick a model and fetch the matching VAE
   and text encoders in one step, or **Download tools...** to install a
   background-removal, upscale or segmentation model. You can also download the
   files for your chosen family into the matching subfolders listed below.
3. Optional: In Model Settings, use **Scan folder...** to point the models root at your ComfyUI `models` folder
4. In Generate, select the model row and run.

The wizard reads your GPU and RAM and preselects the largest quantization that
fits. It never changes which VAE or text-encoder architecture a model needs --
only how much of the model to download. If hardware cannot be detected it picks
a balanced profile and lets you choose another one.

No model is required for the paint tools. AI generation, upscaling, and background removal are optional.

## Download suggestions

### Anima

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `anima-base-v1.0.safetensors` | `diffusion_models/` |
| Text encoder | `qwen_3_06b_base.safetensors` | `text_encoders/` |
| VAE | `qwen_image_vae.safetensors` | `vae/` |

### Qwen-Image 2.1

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `qwen-image-2.1-UC-Q4_K_M.gguf` | `unet/` |
| Text encoder | `Qwen3VL-8B-Instruct-Q4_K_M.gguf` | `text_encoders/` |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | `vae/` |

### Flux.1 dev

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `flux1-dev-Q4_K_S.gguf` | `unet/` |
| Text encoder | `clip_l.safetensors` + `t5-v1_1-xxl-encoder-Q5_K_S.gguf` | `text_encoders/` |
| VAE | `flux_vae.safetensors` | `vae/` |

### Z-Image Turbo

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `z_image_turbo_bf16.safetensors` | `diffusion_models/` |
| Text encoder | `qwen_3_4b.safetensors` | `text_encoders/` |
| VAE | `ae.safetensors` | `vae/` |

### Krea 2

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `krea2_turbo_bf16.safetensors` | `diffusion_models/` |
| Text encoder | `qwen3vl_4b_bf16.safetensors` | `text_encoders/` |
| VAE | `qwen_image_vae.safetensors` | `vae/` |

### MOSS Image

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `moss_image_sft_bf16.safetensors` | `diffusion_models/` |
| Text encoder | `qwen3.5_9b_bf16.safetensors`, `qwen3.5_9b_int8_convrot.safetensors`, or `qwen3.5_9b_w4a8_gptq.safetensors` | `text_encoders/` |
| VAE | `flux2-vae_small.safetensors` | `vae/` |

### MiniMax H3

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `diffusion_models/` |
| Text encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (or `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`) | `text_encoders/` |
| VAE | `minimax_h3_video_vae_int8_convrot.safetensors` | `vae/` |

Both 32B text-encoder quantizations stream through the same loader; install
either one and select it in Model Settings if the default int8 file is not the
one you have.

### MiniMax H3 Image

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | `minimax_h3_image_pruned_int8_convrot.safetensors` | `diffusion_models/` |
| Text encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` or `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (same as MiniMax H3) | `text_encoders/` |
| VAE | `minimax_h3_image_vae_fp16.safetensors` | `vae/` |

The image model runs on a fixed 6-step PDD schedule at guidance 1. Put up to 16
reference images in the References panel, or use the Generate panel's Source
image, for editing; an empty reference list is plain text-to-image. The native
working range is roughly 4–16 MP (2048×2048 is the default).

### Fibo 2.0

| Component | File | Folder |
| --- | --- | --- |
| Diffusion model | Whole `Fibo-2.0-dev` diffusers folder (or Hub repo id) | `diffusion_models/` |
| Text encoder | Included in the folder | — |
| VAE | Included in the folder | — |

### Stable Diffusion

| Component | File | Folder |
| --- | --- | --- |
| Checkpoint | Any single-file `.safetensors` or `.ckpt` (SD 1.5 or SDXL) | `checkpoints/` |
| Text encoder | Included in the checkpoint | — |
| VAE | Included in the checkpoint | — |

### Upscale

| Component | File | Folder |
| --- | --- | --- |
| Upscale model | Any Real-ESRGAN `.safetensors`, `.pth`, or `.onnx` | `upscale_models/` |
| SeedVR2 model | `seedvr2_ema_3b_fp8_e4m3fn.safetensors` | `upscale_models/` |
| SeedVR2 VAE | `seedvr2_ema_vae_fp16.safetensors` | `vae/` |

**Download tools...** offers Real-ESRGAN x2/x4/x8 and the SeedVR2 3B/7B
restorers, and files them into the same pools.

### Background removal

| Component | File | Folder |
| --- | --- | --- |
| Background removal model | `birefnet-general.onnx`, `birefnet-general.safetensors`, `birefnet-lite.onnx`, `birefnet-portrait.onnx`, `rmbg-2.0.safetensors`, `u2net.onnx`, or `modnet_photographic_portrait_matting.onnx` | `background/` |

**Download tools...** offers BiRefNet general, BiRefNet lite, BiRefNet portrait,
U2Net, U2Net portrait, and MODNet, and files them into the background pool.

### Segmentation

| Component | File | Folder |
| --- | --- | --- |
| Click-to-segment model | A SAM ONNX encoder/decoder pair: `<label>_encoder.onnx` + `<label>_decoder.onnx`, or an extracted archive folder `<label>/encoder.onnx` + `<label>/decoder.onnx` | `segmentation/` |

**Download tools...** offers MobileSAM, SAM ViT-B and SAM ViT-H (quantized ONNX
archives), unpacks them into `segmentation/`, and lists them in the Segment
tool.
