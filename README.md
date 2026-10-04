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

### 5. 📸 Live Visual Image Previews
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
