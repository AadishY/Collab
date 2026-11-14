# 🎨 Qwen-Image-2.1 Studio for Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AadishY/Collab/blob/main/Qwen_Image_2_1.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/AadishY/Collab?style=social)](https://github.com/AadishY/Collab)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A comprehensive, production-ready **Qwen-Image-2.1** generative studio optimized for **Google Colab (Free T4 GPU compatible)**. Powered by a headless ComfyUI inference engine with dynamic WebSocket streaming, native DiT patch resolution alignment, self-healing memory management, and multi-LoRA chaining.

---

## 🚀 Quick Launch in Google Colab

Click the badge below to directly open the notebook in Google Colab:

<div align="center">

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AadishY/Collab/blob/main/Qwen_Image_2_1.ipynb)

</div>

> **Tip:** In Google Colab, go to `Runtime` ➔ `Change runtime type` ➔ Select `T4 GPU` (Standard Free Tier).

---

## 🆕 Qwen-Image-2.1 Studio v2

[![Open v2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AadishY/Collab/blob/main/Qwen_Image_2_1_v2.ipynb)

The v2 notebook is an **image-editing and head-swap** workflow; it no longer includes text-to-image generation. The rest of this README describes the original notebook unless marked v2.

1. Run **1. Environment Setup, Drive Options & Model Synchronization** on a Colab GPU runtime. Set the Drive options before running it. Optionally run **2. Preset LoRAs** to pre-download adapters.
2. Run **5. Multi-Image Editing** to edit one or more independent images, with no face swap or reference image. Select multiple files or a ZIP in the upload dialog, or enter comma-separated image paths/URLs, a folder, or a local ZIP path. For multiple upload dialogs, enable `upload_in_multiple_dialogs`; cancel the next dialog when finished.
3. Set one prompt, negative prompt, LoRAs, dimensions and sampler settings for the batch. Images are processed **one at a time**, with individual output dimensions calculated from each input by default (subject to 32-pixel alignment and size limits). Fast mode overrides the selected dimension mode. A fixed seed is reused for all images; `-1` draws a different random seed for each image. The seed is shown with each result so it can be reused.
4. Results are saved to `/content/output/` (or to the linked Drive output folder when enabled). Enable `create_batch_zip_download` to create a ZIP there; the summary reports failed inputs while continuing with the remaining images. If every edit fails, the cell raises an error. Re-running creates new output filenames.
5. **6. Dedicated Head Swap** and **6B. Batch Head Swap** remain separate, reference-image-based workflows. In 6B, `-1` gives each target a different random seed; a fixed seed increments by one for each target.

### Four LoRA slots and presets (v2)

All three v2 inference cells offer **primary, secondary, tertiary, and quaternary** slots, each with a custom filename/URL and strength. Set a slot strength to `-1` (the default) to use the preset-specific recommendation, `0` to disable it, or a positive number to override it. Select `None` to leave a slot empty. Duplicate adapters and combinations of Lightning with Viggle Turbo are rejected. The Viggle merged model occupies a slot and replaces the UNET at full strength, rather than acting as a blendable LoRA. Four adapters can increase memory use; results from stacking are not guaranteed. Cell 6B can inherit the additional slots and trigger setting from Cell 6; turn inheritance off to use only the settings shown in Cell 6B.

| Preset | Source / usage | Automatic trigger / default settings |
| --- | --- | --- |
| `pytorch_lora_weights_v1_5.safetensors` | [Qwen-Image-2.1 2-step DMD2 LoRA V1.5](https://huggingface.co/linoyts/Qwen-Image-2.1-DMD2-2steps); attention-only 2-step distillation adapter | Defaults to `2` steps, fixed sigmas `[1.0, 0.5]` (`use_dynamic_shifting=False`, `shift=1.0`), `CFG=1.0`, `euler` sampler, strength `1.0` |
| `elusarcas-qwen2-1-detailer-v1.safetensors` | [Elusarca's Qwen 2.1 Detail Enhancer LoRA](https://huggingface.co/reverentelusarca/elusarcas-qwen-2.1-detail-enhancer-lora); high-resolution detail enhancement & microdetails | Default prompt: `Enhance this image to high resolution with rich fine details. Sharpen textures and surfaces, add natural microdetails, and improve clarity and definition across the image while preserving the original composition, lighting, atmosphere, and visual style; preserve the original style.` |
| `civit_anything2real_google_drive.safetensors` | [Anything2Real from the supplied Google Drive link](https://drive.google.com/file/d/1RA3DwpCE7ztEYl2cd4XAYJ2lm6-D_LSI/view?usp=sharing), separate from the Hugging Face preset | `photorealistic`, `transform the image to realistic photograph` |
| `qwen-image-2.1-restore.safetensors` | [Restore / skin](https://huggingface.co/ausboss/Qwen-Image-2.1-Restore-LoRA); the author recommends 1.0 for in-focus images and 0.5 for blurry photos | None required |
| `Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors` | [Natural Exposure](https://huggingface.co/prithivMLmods/Qwen-Image-2.1-Natural-Exposure-LoRA), an experimental model | `Transform the image with balanced neutral exposure` |
| `qwen-image-2.1-consistency.safetensors` | [Consistency](https://huggingface.co/ausboss/Qwen-Image-2.1-Consistency-LoRA); may resist pose or head movement | None required |
| `bfs_head_v1.1_qwen_2.1.safetensors` | [Head swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap/blob/main/docs/qwen-image-2.1.md), with target and reference images in cells 6/6B | `head_swap:` (already in the head-swap prompt) |

`auto_add_lora_triggers` is enabled by default. It adds only missing documented triggers for nonzero-strength selected presets; turn it off to use the prompt exactly as typed. When `elusarcas-qwen2-1-detailer-v1.safetensors` is selected in Cell 5 without an Anything2Real preset and the prompt is left at the default or empty, it automatically switches to the Detail Enhancer default prompt. Restore and Consistency have no required trigger according to their model cards. Triggers for custom LoRAs must be added manually. In cell 2, the download checkboxes are optional; missing selected presets also download when needed. The Drive-sourced Anything2Real uses a new filename so an older Civitai cache is not reused. The link must remain accessible to anyone with the link; Google Drive may also apply download or quota limits.

**Automatic strengths:** DMD2 2-Step `1.0`; Detail Enhancer `1.0`; Lightning `0.7`; Hugging Face and Drive Anything2Real `0.8`; Consistency `0.8`; Natural Exposure `0.8`; Restore `0.7`. BFS and unspecified/custom models default to `1.0`. You can override each slot individually; the merged Viggle UNET is not strength-blended.

### Google Drive and space requirements (v2)

In cell 1, enable `mount_google_drive` to access images using paths such as `/content/drive/MyDrive/photos/image.png`. Set `image_source` to `paths_or_urls_or_folder` in batch editing and enter image paths, a folder, or a ZIP. The head-swap cells also accept Drive image paths. Mounting alone does not change where models or results are stored.

- Enable `use_drive_for_models` to keep and load base models and LoRAs directly from `MyDrive/Collab_Qwen_Studio/models/`. Enable `save_outputs_to_drive` to save results and ZIPs to `MyDrive/Collab_Qwen_Studio/output/`. Either setting mounts Drive automatically. Change `drive_studio_folder` to another folder **relative to MyDrive** if desired.
- Models already in that Drive folder can be reused without copying: use the required subfolder (`diffusion_models`, `text_encoders`, `vae`, or `loras`) and exact preset filename, for example `models/loras/civit_anything2real_google_drive.safetensors`. The file must pass validation. The supplied shared link otherwise downloads through `gdown` into the chosen model storage; simply mounting MyDrive does not automatically make a shared file part of it.
- Choose storage options in a fresh runtime before downloading models. To protect existing data, setup refuses to replace populated runtime folders or change an existing Drive link. Start a fresh runtime to switch storage; no model or output files are moved or deleted. Initial downloads and loading from Drive can be slower than local disk.
- The three required base files total about **17.3 GB** (7.26 GB diffusion model, 9.35 GB text encoder, 0.676 GB VAE; decimal sizes). Allow **at least 20 GB free** for them and small LoRAs. The optional merged Viggle model adds about **7.3 GB**: allow **about 30 GB free** with it and LoRAs, plus extra room for results and future models. A standard 15 GB Google Drive quota is insufficient for all base weights. Colab also needs local disk for ComfyUI, packages, intermediate images and downloads; leave roughly **10 GB or more** local free when models are on Drive, or roughly **30–40 GB** local free when using local model storage. These are planning estimates, not quota guarantees.

For local, GPU-free checks, install Pillow, requests, and websocket-client, then run `python -m unittest discover -s tests -v`. These tests mock the downloader and inference; they do not replace a full Colab GPU run.

---

## ✨ Studio Capabilities

| Studio Cell | Task | Primary Model / LoRA | Description |
| :--- | :--- | :--- | :--- |
| **Cell 4** | **Text-to-Image** | Qwen-Image-2.1 Base (INT8) | Text prompt generation up to 2K (2048×2048) with aspect-ratio presets. |
| **Cell 5** | **Single-Image Edit** | `Qwen2.1_Anything2RealCharacters` | Transform drawings, anime, or 3D art into photorealistic humans, or execute instruction edits. |
| **Cell 6** | **Dedicated Head Swap** | `bfs_head_v1.1_qwen_2.1` | Replace head and facial identity while strictly preserving target scene lighting, pose, and background. |
| **Cell 7** | **Dedicated Body Swap** | `bfs_body_swap_v1.0_qwen_2.1` | Transfer full body, outfit, and proportions from a reference person while preserving the target scene background. |

---

## 🌟 Key Features

### 1. 🎛️ Interactive LoRA Dropdown Selectors
No need to manually type model paths. Every studio cell includes interactive form dropdowns for both **Primary** and **Secondary** LoRAs:
- `None`
- `bfs_head_v1.1_qwen_2.1.safetensors` *(Face & Head Swap)*
- `bfs_body_swap_v1.0_qwen_2.1.safetensors` *(Full Body & Clothing Transfer)*
- `Qwen2.1_Anything2RealCharacters.safetensors` *(Style to Photorealism)*
- `p_qwen_image_2.1_8step_v0.1.safetensors` *(8-Step Turbo Acceleration)*
- `Custom (Enter filename below)` *(For user-downloaded models from Civitai/HuggingFace/Drive)*

### 2. ⚡ Speed & VRAM Saver (Aspect-Preserving Fast Mode)
A single toggle that cuts rendering time by **2.5× to 3×** while strictly preserving the target image's exact aspect ratio:
- `lower_dimensions_for_speed = True`
- `speed_target_resolution = "0.6 Megapixels (Fastest / Low VRAM)"`
- Automatically calculates 32-pixel DiT aligned dimensions matching the native aspect ratio of your target scene.

### 3. 🔗 Multi-LoRA Chaining & Turbo Auto-Optimizer
- Run identity or style LoRAs simultaneously with the **8-step Turbo LoRA** (`p_qwen_image_2.1_8step_v0.1.safetensors`).
- **Auto-Optimize Turbo Steps:** Automatically sets sampling steps to `8` when the turbo LoRA is active, reducing render times from ~45s to **~8-12 seconds** on Colab T4.

### 4. 🛡️ Self-Healing Engine & Out-of-Order Execution
- Engine utilities are shared across cells via `qwen_studio_utils.py`.
- **Run in any order:** Run **Cell 1** once, and you can jump straight to **Cell 5 (Edit)**, **Cell 6 (Head Swap)**, or **Cell 7 (Body Swap)** without having to run previous cells.
- The engine automatically verifies and launches the backend server in the background if it is offline.
- Automatic VRAM cache purging (`purge_backend_memory()`) prevents CUDA Out-Of-Memory errors.

### 5. 🎚️ Sampler & Scheduler Dropdown Controls
Every generation cell now includes interactive dropdowns for sampling algorithms and noise schedules:
- **Default Sampler:** `euler` (standard, high stability)
- **Advanced Sampler:** `er_sde` (extended stochastic SDE solver for enhanced textural micro-detail), along with `euler_ancestral`, `dpmpp_2m`, `dpmpp_sde`, `dpmpp_2m_sde`, `heun`, `ddim`, `uni_pc`.
- **Default Scheduler:** `simple` (standard, stable)
- **Alternative Schedulers:** `beta` (smooth noise schedule recommended for modern diffusion models), `normal`, `karras`, `exponential`, `sgm_uniform`, `ddim_uniform`.

### 6. 📸 Live Visual Image Previews
- Before generation begins, input images are displayed in clean side-by-side comparison cards with source dimensions and aspect ratio tags.
- Output images are rendered at full resolution and saved locally to `/content/output/`.

---

## 📖 Step-by-Step Usage Guide

### Step 1: Environment Setup & Model Sync
Run **Cell 1 (`1. Environment Setup & Model Synchronization`)**:
- Installs `uv`, `aria2`, PyTorch with CUDA 12.4, and ComfyUI.
- Downloads official INT8 weights (`qwen_image_2.1_int8_convrot.safetensors`, `qwen3vl_8b_int8_convrot.safetensors`, and BF16 VAE).
- Initializes `/content/output/` and shared engine utilities.

### Step 2: (Optional) LoRA Pre-Downloader
Run **Cell 2 (`2. Preset LoRAs & Custom LoRA Downloader`)**:
- Toggle checkboxes to download preset LoRAs or enter custom Civitai / Hugging Face links.
- *Note:* If you skip this cell, the generation cells will automatically download missing preset LoRAs on the fly when selected!

### Step 3: Choose Your Studio & Generate!

#### 🎨 Text-to-Image Generation (Cell 4)
- Enter your prompt and optional negative prompt.
- Select aspect ratio (`5:4`, `16:9`, `9:16`, `1:1`, `21:9`, etc.) or choose `2K High-Res (2048x2048)`.
- Enable LoRAs from the dropdown and click Run!

#### 🖼️ Single-Image Editing & Style-to-Real (Cell 5)
- Choose `upload` (prompts file picker) or `path_or_url`.
- Use the default `Anything2RealCharacters` LoRA to turn anime, 3D avatars, or sketches into photorealistic humans.
- Reference `<image1>` directly in your prompt (e.g. `transform <image1> into a real human, 8k uhd`).

#### 👤 Dedicated Head Swap (Cell 6)
- **Picture 1 (`<image1>`) [TARGET BASE SCENE]:** The person whose body, pose, clothing, lighting, camera angle, and background are strictly kept.
- **Picture 2 (`<image2>`) [REFERENCE HEAD]:** The reference face whose identity and hair are transferred into Picture 1.
- Output strictly inherits Picture 1's aspect ratio.

#### 🧍 Dedicated Body Swap (Cell 7)
- **Image 1 (`<image1>`) [TARGET SCENE]:** The scene whose pose, framing, lighting, and background are kept.
- **Image 2 (`<image2>`) [REFERENCE PERSON]:** The reference person whose clothing, body shape, and proportions are transferred.
- Includes automatic 0.59 MP token-bleed protection to guarantee the target scene's background is preserved.

---

## 📦 Preset LoRA Registry

| LoRA Filename | Hugging Face Source | Recommended Use |
| :--- | :--- | :--- |
| `bfs_head_v1.1_qwen_2.1.safetensors` | [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) | Dedicated Head & Face Swap (Cell 6) |
| `bfs_body_swap_v1.0_qwen_2.1.safetensors` | [Alissonerdx/BFS-Best-Face-Swap](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap) | Dedicated Body & Clothing Swap (Cell 7) |
| `Qwen2.1_Anything2RealCharacters.safetensors` | [WarmBloodAban/Qwen-Image-2.1-LoRAs](https://huggingface.co/WarmBloodAban/Qwen-Image-2.1-LoRAs) | Style-to-Real Characters & Image Edit (Cell 5) |
| `p_qwen_image_2.1_8step_v0.1.safetensors` | [PrunaAI/Pruna-Qwen-Image-2.1](https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1) | 8-Step Turbo Acceleration across all cells |

---

## 🛠️ Hardware Requirements

- **GPU:** Minimum 12 GB VRAM (Google Colab free **T4 GPU with 15 GB VRAM** is supported).
- **RAM:** Standard 12.7 GB system RAM.
- **Disk:** ~25 GB free disk space in Colab environment.

---

## 📄 License & Credits

- Based on the [Qwen-Image-2.1](https://github.com/QwenLM) architecture by the Qwen Team / Alibaba Cloud.
- Built on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous.
- Special thanks to [Alissonerdx](https://huggingface.co/Alissonerdx) for the BFS models and [AI With Chucky](https://youtube.com/@AIWithChucky).
