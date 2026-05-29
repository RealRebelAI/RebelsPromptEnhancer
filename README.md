# Rebels Prompt Enhancer Nodes for ComfyUI

**WORK IN PROGRESS - More features coming soon!**

VERSION 2 Now LIVE!

<img width="1660" height="804" alt="Screenshot (142)" src="https://github.com/user-attachments/assets/f445bb62-fe1e-4e25-8478-2e139c592c4d" />



Four ultra-lightweight, local-only nodes for ComfyUI:

- 🚀 **Rebels Prompt Enhancer** — curated Qwen3.5-4B text rewriter with a layered style system
- 🧪 **Rebels Prompt Enhancer (Custom GGUF)** — bring your own model, full control over every parameter
- 👁️ **Rebels Image to Prompt** — vision-capable node that turns a reference image into a prompt
- 🔒 **Rebels Prompt Locker** — workflow gate that displays the passed-through prompt on its node face

No API keys, no external calls, 100% private. Aggressive VRAM cleanup so your diffusion model gets the GPU back immediately after the prompt is generated.

---

## Features

### Universal across all nodes
- **Local-Only.** Everything runs on your machine via `llama-cpp-python`. Nothing leaves the box.
- **Aggressive VRAM Cleanup.** Models fully unload after each call.
- **Prompt Lock + Cache.** Toggle LIVE to iterate freely, then LOCKED to freeze the result you like. Locked = no model load on subsequent runs.
- **Full Diagnostic Output.** `thought_process` output shows the assembled system prompt, raw model output, what was stripped, and the final result.

### Layered Style System (text enhancers)

Three independent dropdowns compose into the system prompt at runtime:

- **Purpose** — Image / Video / Edit. Controls subject framing.
- **Model Format** — How the prompt should be structured for the model you're feeding.
- **Aesthetic** — The visual style/vibe to apply.

This means hundreds of useful combinations from short dropdowns, all powered by the same 4B model.

### Model Format Support

| Format | Use For |
|---|---|
| Flux / Chroma | Natural-language Flux and Chroma models |
| Z-Image / Lumina-2 | LLM-text-encoder models |
| HiDream | HiDream multi-encoder pipeline |
| SDXL | Tag + weight syntax for SDXL |
| SD 1.5 | Tag + weight syntax for SD 1.5 |
| Pony / Illustrious | Booru tags with score and rating tags |
| LTX Video | Motion-focused prose for LTX |
| Hunyuan / Wan Video | Cinematic motion prose |
| Universal Natural Language | Generic, works with most models |

### Aesthetic Library (22 styles)

Photorealistic · Cinematic Film · Anime/Manga · Studio Ghibli · Pixar/3D Animation · Comic Book · Concept Art · Oil Painting · Watercolor · Pencil Sketch · Cyberpunk · Steampunk · Fantasy · Sci-Fi · Horror/Dark · Vintage/Retro Film · Film Noir · Glamour/Editorial · Minimalist · Surreal/Dreamy · 3D Render/CGI · None (skip aesthetic injection)

### Vision Support

The Image to Prompt node accepts any vision-capable GGUF + its paired `mmproj` projector. Supports multiple architectures via `llama-cpp-python` chat handlers:

- LLaVA 1.5 / 1.6
- Moondream
- MiniCPM-V 2.6
- NanoLLaVA
- Qwen2.5-VL

Auto-detect picks the right handler from the filename, or override manually.

---

## The Nodes

### 🚀 Rebels Prompt Enhancer (curated)

Uses Qwen3.5-4B locally with `/no_think` baked in to suppress reasoning mode. Output arrives clean.

| Input | Purpose |
|---|---|
| `raw_prompt` | Your input idea |
| `purpose` | Image / Video / Edit |
| `model_format` | Output structure for your target model |
| `aesthetic` | Visual style to apply |
| `precision` | Efficiency (UD-IQ2, ~2GB) or Quality (UD-Q8, ~6.5GB) |
| `seed` | Randomize freely; ignored when locked |
| `lock_in` | LIVE or LOCKED |

Outputs `enhanced_prompt` (clean string for the sampler) and `thought_process` (full diagnostic).

### 🧪 Rebels Prompt Enhancer (Custom GGUF)

Same layered style system as the curated enhancer, but scans the folder for any `.gguf` and lets you pick. Adds full sampling controls.

Additional inputs vs the curated node:

| Input | Purpose |
|---|---|
| `model_file` | Pick any GGUF found in the node folder |
| `extra_instructions` | Appended to the layered system prompt |
| `system_prompt_override` | If non-empty, replaces the layered prompt entirely (full manual mode) |
| `append_no_think` | Toggle the Qwen3 directive (only useful for Qwen3 family models) |
| `n_gpu_layers` | `-1` = all on GPU, `0` = CPU only, `N` = partial offload |
| `n_ctx`, `max_tokens`, `temperature`, `top_p`, `repeat_penalty` | Sampling controls |

### 👁️ Rebels Image to Prompt

Vision-capable node. Takes a reference image and outputs a prompt suitable for downstream generation.

| Input | Purpose |
|---|---|
| `image` | ComfyUI IMAGE input |
| `model_file` | Vision-capable GGUF from the folder |
| `mmproj_file` | Paired mmproj projector file |
| `chat_handler` | Auto-detect, or pick (LLaVA 1.5/1.6, Moondream, MiniCPM-V 2.6, NanoLLaVA, Qwen2.5-VL) |
| `vision_task` | Caption / Caption + Format / SD Tags / Pose & Anatomy / Custom |
| `model_format` | Applied when vision_task = "Caption + Format" |
| `aesthetic` | Applied when vision_task = "Caption + Format" |
| `custom_instruction` | Used when vision_task = "Custom Instruction" |
| Standard sampling + GPU controls + lock | Same as Custom GGUF enhancer |

Outputs `image_prompt` (clean) and `thought_process` (full diagnostic).

### 🔒 Rebels Prompt Locker

Workflow gate that also displays the prompt on its node face.

| Input | Purpose |
|---|---|
| `text_input` | String to gate (typically `enhanced_prompt` or `image_prompt`) |
| `lock_in_prompt` | PAUSED halts with ValueError; LOCKED IN passes through |

The on-node text display requires the bundled `web/js/rebels_locker_display.js` extension to be present.

---

## Installation

1. Navigate to your ComfyUI `custom_nodes` folder:
   ```
   cd ComfyUI/custom_nodes
   ```

2. Clone this repository:
   ```
   git clone https://github.com/RealRebelAI/RebelsPromptEnhancer_-minicpm5-1b-ggufs-.git
   ```

3. Install dependencies.

   **ComfyUI Portable** (open cmd inside the cloned folder):
   ```
   ../../python_embeded/python.exe -m pip install -r requirements.txt
   ```

   **ComfyUI Desktop:**
   ```
   pip install -r requirements.txt
   ```

   If you hit build errors, upgrade your build chain first:
   ```
   ..\..\python_embeded\python.exe -m pip install --upgrade pip setuptools wheel scikit-build-core
   ```

4. Download model file(s) from the Models section below and drop them directly in this node pack's folder.

5. Restart ComfyUI.

---

## Models

All models are GGUF format and live directly in the node pack folder. Download only what you'll use.

### Curated text model (for 🚀 Rebels Prompt Enhancer)

Repo: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF

| Precision | File | Size |
|---|---|---|
| Efficiency (UD-IQ2) | `Qwen3.5-4B-UD-IQ2_XXS.gguf` (or any `UD-IQ2_*`) | 1.5–1.8 GB |
| Quality (UD-Q8) | `Qwen3.5-4B-UD-Q8_K_XL.gguf` | 5.95 GB |

The node searches by filename substring (`qwen3.5-4b` + `ud-iq2` or `ud-q8`), so renames work as long as the key tokens are intact.

### Custom text models (for 🧪 Custom GGUF Enhancer)

Drop any GGUF text model in the folder and it'll show up in the dropdown. Examples:

- **Standard models:** Any architecture llama.cpp supports — Llama, Mistral, Phi, Gemma, Qwen, etc.
- **Uncensored fine-tunes** (for NSFW prompt rewriting where standard models refuse or sanitize):
  - https://huggingface.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive — Q4_K_M at 2.71 GB, also includes mmproj so it doubles as a vision model
  - Look for `abliterated` versions from `mlabonne` or `huihui-ai` on HuggingFace
- **Larger models** for users with more VRAM: use `n_gpu_layers` to control GPU offload at any model size

### Vision models (for 👁️ Image to Prompt)

Vision setups need **two files**: the main GGUF and a paired `mmproj-*.gguf` (vision projector). Both go in the node folder.

Recommended starting picks:

- **Qwen3.5-4B-Uncensored** (HauhauCS link above) — includes mmproj, works as both text AND vision model. Most efficient option if you're already using it for text.
- **Moondream2** — ~1.6B, tiny and fast, solid general captioning
- **Qwen2.5-VL-3B** — modern vision model, fits 8GB VRAM cleanly
- **MiniCPM-V 2.6** — 8B, SOTA for small VLMs, best descriptive quality

---

## Hardware Notes

The enhancers aggressively unload the model after each run, so VRAM only needs to fit the LLM during the rewrite — not alongside your diffusion model.

### Approximate VRAM usage (model + 4k context)

| Model | VRAM |
|---|---|
| Qwen3.5-4B UD-IQ2 | ~2 GB |
| Qwen3.5-4B UD-Q8 | ~6.5 GB |
| Qwen3.5-4B-Uncensored Q4_K_M | ~3 GB |
| Moondream2 | ~1.5 GB |
| Qwen2.5-VL-3B Q4 | ~2.5 GB |
| MiniCPM-V 2.6 Q4 | ~5–6 GB |

### For bigger models

The Custom Enhancer's `n_gpu_layers` setting controls GPU offload:

- `-1` (default) — all layers on GPU (requires the model fits VRAM)
- `0` — pure CPU (slow but works for any model size, no VRAM needed)
- `N` — partial offload, N layers on GPU and the rest on system RAM

This means users with bigger cards can drop in 30B+ models and run them natively, and anyone can experiment with partial offload for models that exceed their VRAM.

---

## Usage

Find all four nodes under **"Rebel AI"** in the node menu.

### Basic text wiring

```
[Your Prompt Text] → raw_prompt → 🚀 Enhancer → enhanced_prompt → 🔒 Locker → KSampler
                                              → thought_process → Preview-As-Text (optional)
```

### Image-to-prompt wiring

```
[Load Image] → image → 👁️ Image to Prompt → image_prompt → 🔒 Locker → KSampler
                                            → thought_process → Preview-As-Text (optional)
```

### Chained (image → faithful caption → styled prompt)

```
[Load Image] → 👁️ Image to Prompt (Caption) → 🚀 Enhancer (apply Format + Aesthetic) → 🔒 Locker → KSampler
```

This chain captures the reference faithfully with the vision model, then runs that description through Qwen3.5-4B for the final styling — which often beats letting the vision model do both jobs at once, because text models are better wordsmiths than vision models.

### The Lock workflow

1. Start the enhancer in **LIVE** with seed on `randomize`.
2. Queue runs and iterate — each run produces a different prompt because the seed changes.
3. When you find a prompt you love, toggle the enhancer to **LOCKED**. From then on it skips model loading and returns the cached prompt every run. Seed changes are ignored.
4. Independently, the Locker gates downstream execution — flip it to `LOCKED IN` to let the prompt flow to the sampler.

The two locks are intentionally separate:

- **Enhancer lock** freezes the prompt (cache).
- **Locker** halts the workflow (gate).

The cache lives in memory for the ComfyUI session and clears on restart. Changing any non-seed input while locked invalidates the cache and triggers a fresh generation under the new key.

---

## Folder Structure

```
RebelsPromptEnhancer/
├── __init__.py
├── requirements.txt
├── README.md
├── *.gguf                          ← drop text model files here
├── mmproj-*.gguf                   ← vision projector files (paired with vision models)
└── web/
    └── js/
        └── rebels_locker_display.js
```

The `web/js/` extension is what makes the Locker display its text on the node face. Without it, the Locker still functions but won't render the text visually.

---

*Built by Rebel AI.*
