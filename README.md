# Rebels Prompt Enhancer V3 for ComfyUI

**Current Version: 3.2.7**

A local prompt enhancement toolkit for ComfyUI built around lightweight GGUF language models.

Rebels Prompt Enhancer can expand simple ideas into detailed prompts for image generation, image editing, and video generation while staying entirely local.

No API keys. No cloud inference. No prompt data sent to external services.

Designed with **LOW VRAM users** in mind.

---

## What's New in V3

V3 greatly expands the original Rebels Prompt Enhancer while preserving compatibility with older workflows.

### Major V3 Features

- Three curated Qwen3.5-4B quality tiers
- Built-in GGUF model downloader
- Resumable model downloads
- Structured JSON generation
- Automatic Qwen3 `/no_think`
- Multiple prompt variants
- Optional negative prompt generation
- Detail-strength controls
- Prompt-length controls
- Strict user-detail preservation
- Persistent prompt cache
- VRAM Saver mode
- Fast Iteration mode
- Automatic CPU fallback for VRAM failures
- Automatic compatibility fallback when installed `llama-cpp-python` cannot run the model
- Private portable llama.cpp runtime on Windows x64
- Custom GGUF support
- Image-to-Prompt node
- General LLM Console
- Prompt Locker
- Existing V2 workflows remain compatible

---

# Included Nodes

All nodes are available under:

`Rebel AI`

### 🚀 Rebels Prompt Enhancer V3

The main curated prompt enhancer.

Uses Qwen3.5-4B GGUF models with presets designed around prompt rewriting rather than general conversation.

Outputs:

- `enhanced_prompt`
- `thought_process`
- `negative_prompt`
- `variants_json`

### 🧪 Rebels Prompt Enhancer V3 — Custom GGUF

Use your own GGUF language model.

Provides manual control over:

- GPU layers
- Context size
- Maximum tokens
- Temperature
- Top-P
- Repeat penalty
- System prompt
- Extra instructions
- `/no_think`
- Model loading behavior

### 📦 Rebels Prompt Enhancer Model Helper

Checks for or downloads the curated Qwen3.5-4B models.

The downloader supports resumable `.part` downloads so interrupted large downloads do not need to restart from zero.

### 👁️ Rebels Image to Prompt

Experimental vision node for converting an image into a descriptive generation prompt.

Requires:

- vision-capable GGUF
- matching `mmproj` GGUF
- supported `llama-cpp-python` vision handler

### 🧠 Rebels LLM Console

General-purpose local GGUF text console inside ComfyUI.

Useful for testing models, asking questions, or experimenting with sampling settings.

### 🔒 Rebels Prompt Locker

Workflow gate for freezing and passing prompts downstream.

The included frontend extension displays the locked prompt directly on the node.

---

# Curated Models

Curated models come from:

`unsloth/Qwen3.5-4B-GGUF`

You do **not** need every model.

Choose the tier that matches your hardware and desired quality.

| Preset | File | Approx. Size | Use Case |
| --- | --- | ---: | --- |
| Efficiency (UD-IQ2) | `Qwen3.5-4B-UD-IQ2_M.gguf` | ~1.66 GB | V2 compatibility / low VRAM |
| Ultra Low VRAM (UD-IQ2) | `Qwen3.5-4B-UD-IQ2_M.gguf` | ~1.66 GB | Lowest memory use |
| Balanced (Q4) | `Qwen3.5-4B-Q4_K_M.gguf` | ~2.54 GiB | Recommended balance |
| High Quality (UD-Q8) | `Qwen3.5-4B-UD-Q8_K_XL.gguf` | ~5.95 GB | Highest curated quality |

`Efficiency (UD-IQ2)` and `Ultra Low VRAM (UD-IQ2)` intentionally use the same model.

The Efficiency label remains so workflows created with older versions continue loading correctly.

---

# Automatic Model Downloader

You can download curated models directly from ComfyUI.

Add:

`📦 Rebels Prompt Enhancer Model Helper`

Select the desired precision and change:

`CHECK ONLY`

to:

`DOWNLOAD MODEL`

The model downloads directly into:

```text
ComfyUI/
└── custom_nodes/
    └── RebelsPromptEnhancer/
        └── Qwen3.5-4B-*.gguf
```

The downloader uses paths relative to the node itself.

There are **no hardcoded user folders, drive letters, usernames, or personal installation paths**.

This means an installation on:

```text
C:
D:
E:
Linux
another Windows PC
```

does not depend on the developer's local directory layout.

The node determines its own directory using its installed location.

---

# Resumable Downloads

Large model downloads use temporary `.part` files.

Example:

```text
Qwen3.5-4B-UD-Q8_K_XL.gguf.part
```

If the download is interrupted, the next attempt will resume when supported rather than automatically starting from zero.

After completion the file becomes:

```text
Qwen3.5-4B-UD-Q8_K_XL.gguf
```

---

# Automatic Missing-Model Download

The main curated enhancer also includes:

`auto_download_missing`

When enabled, the selected model is downloaded automatically if it is not already installed.

If disabled, the node will instead report which model file is missing.

---

# Prompt Controls

## Purpose

### Image

Optimized for still-image generation.

Focuses on:

- subject
- appearance
- pose
- environment
- composition
- camera
- lens
- lighting
- materials
- atmosphere
- spatial relationships

### Video

Adds motion-aware prompt construction.

Focuses on:

- subject movement
- camera movement
- motion speed
- temporal order
- environmental motion
- continuity
- pacing
- final visible state

### Edit — Inpainting / I2V

Designed for editing existing content.

The enhancer emphasizes preserving anything the user did not explicitly ask to change.

---

# Model Format Presets

V3 includes prompt formatting for:

- Qwen Image / Qwen Image Edit
- Flux / Chroma
- Z-Image / Lumina
- HiDream
- SDXL
- SD 1.5
- Pony / Illustrious
- Wan / Hunyuan Video
- LTX Video
- FastH3 / MiniMax Video
- Universal Natural Language

Different models respond better to different prompt structures.

The Model Format selector changes how the prompt is constructed without changing the user's core idea.

---

# Detail Strength

### Preserve

Minimal expansion.

Useful when the original prompt is already detailed and should remain close to the source.

### Expand

Adds useful visual information while remaining tightly faithful to the original idea.

### Highly Detailed

Adds richer information across:

- appearance
- environment
- lighting
- camera
- materials
- composition
- atmosphere

### Creative

Allows the enhancer to creatively fill unspecified details while preserving everything the user explicitly requested.

---

# Prompt Length

Available targets:

- Short
- Medium
- Long
- Max Detail

Approximate targets:

| Setting | Target |
| --- | --- |
| Short | ~40–90 words |
| Medium | ~90–180 words |
| Long | ~180–320 words |
| Max Detail | As much useful detail as needed |

---

# Strict Detail Preservation

When:

`preserve_user_details = true`

the enhancer is instructed not to remove, contradict, or reinterpret explicit user information.

This includes things such as:

- names
- identities
- number of subjects
- clothing
- colors
- written text
- pose
- camera angle
- environment
- time of day
- layout
- explicit constraints

Preservation does **not** mean simply copying the prompt.

V3 includes a quality check that detects obvious no-op rewrites and retries the enhancement when necessary.

For example:

```text
Input:
a woman
```

should not simply return:

```text
a woman
```

The model is instructed to preserve the idea while adding useful compatible visual detail.

---

# Prompt Variants

The enhancer can generate:

- 1 variant
- 2 variants
- 3 variants
- 4 variants

The primary result is returned through:

`enhanced_prompt`

All generated versions are also returned through:

`variants_json`

---

# Negative Prompts

Set:

`negative_prompt`

to:

`Generate when useful`

to allow the LLM to produce a matching negative prompt.

Otherwise select:

`Off`

---

# Structured Output

V3 uses structured JSON generation internally.

Expected output resembles:

```json
{
  "final_prompt": "enhanced prompt here",
  "variants": [
    "enhanced prompt here"
  ],
  "negative_prompt": ""
}
```

The node extracts the clean prompt before passing it downstream.

This greatly reduces:

- conversational responses
- explanations
- markdown
- reasoning text
- preambles
- unwanted assistant commentary

---

# Qwen `/no_think`

Qwen3-family models are automatically detected.

The curated enhancer enables:

`/no_think`

to prevent reasoning text from leaking into prompt output.

The Custom GGUF node also allows:

- Auto
- Force ON
- Force OFF

---

# Persistent Prompt Cache

The enhancer includes a persistent disk cache.

When:

`lock_in = true`

a previously generated prompt can be reused without loading the model again.

Cache location:

```text
ComfyUI/
└── custom_nodes/
    └── RebelsPromptEnhancer/
        └── .cache/
            └── prompt_cache.json
```

The cache is stored relative to the node folder.

It is not tied to any specific user's installation path.

The cache is automatically bounded to prevent unlimited growth.

---

# Load Modes

## VRAM Saver

`VRAM Saver (unload after run)`

The model is released after prompt generation.

Recommended when the same GPU also needs to run large image or video models.

## Fast Iteration

`Fast Iteration (keep model loaded)`

Keeps one text model available between prompt generations when the active in-process backend supports it.

Useful when repeatedly experimenting with prompts.

---

# Automatic CPU Fallback

When the installed `llama-cpp-python` backend encounters an out-of-memory condition while attempting GPU inference, the enhancer can automatically retry on CPU.

Controlled by:

`auto_cpu_fallback`

This avoids requiring users to manually change GPU-layer settings for many memory-related failures.

---

# Portable llama.cpp Compatibility Fallback

Some combinations of:

- Windows
- Python
- llama-cpp-python
- llama.cpp / GGML builds
- CPU instruction support

can cause the installed native backend to terminate with:

```text
STATUS_ILLEGAL_INSTRUCTION
0xC000001D
```

Rebels Prompt Enhancer V3 includes an automatic compatibility fallback for this situation.

When this exact backend failure is detected:

1. The node stops retrying the incompatible installed backend.
2. The user's existing `llama-cpp-python` installation is left completely untouched.
3. A private official Windows x64 llama.cpp runtime is downloaded.
4. The runtime is stored only inside RebelsPromptEnhancer.
5. Prompt generation runs in a separate process.
6. The node remembers the failure for the rest of the current ComfyUI session and sends future text-enhancer runs directly to the portable backend.

Runtime location:

```text
ComfyUI/
└── custom_nodes/
    └── RebelsPromptEnhancer/
        └── .runtime/
            └── llama_cpp_portable/
```

### Important

The compatibility runtime:

- does **not** replace `llama-cpp-python`
- does **not** modify ComfyUI Python packages
- does **not** modify another custom node
- does **not** install system-wide software
- does **not** redownload your GGUF
- stays contained inside RebelsPromptEnhancer

The current automatic portable fallback is intended for **Windows x64**.

The bundled compatibility path currently uses the CPU llama.cpp runtime for maximum compatibility.

---

# Native Backend Session Memory

Once the node detects that the installed `llama-cpp-python` backend produces `0xC000001D`, it disables further attempts through that broken backend for the rest of the current ComfyUI session.

This prevents repeated Windows fatal-exception stack traces every time another quant is selected.

Restarting ComfyUI resets this detection.

---

# Installation

## Expected Folder Layout

The important part is that the final installation looks like this:

```text
ComfyUI/
└── custom_nodes/
    └── RebelsPromptEnhancer/
        ├── __init__.py
        ├── requirements.txt
        ├── VERSION
        ├── README.md
        └── web/
```

Do **not** accidentally install it as:

```text
ComfyUI/
└── custom_nodes/
    └── RebelsPromptEnhancer/
        └── RebelsPromptEnhancer/
            └── __init__.py
```

`__init__.py` should be directly inside:

```text
ComfyUI/custom_nodes/RebelsPromptEnhancer/
```

---

## Manual Installation

Download the repository and copy the actual `RebelsPromptEnhancer` node folder into:

```text
ComfyUI/custom_nodes/
```

Then install the requirements.

### ComfyUI Portable

From the root of your portable ComfyUI installation:

```bat
python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\RebelsPromptEnhancer\requirements.txt
```

### Standard / Desktop Python Environment

Open a terminal inside:

```text
ComfyUI/custom_nodes/RebelsPromptEnhancer/
```

and run:

```bash
python -m pip install -r requirements.txt
```

Restart ComfyUI afterward.

---

# Requirements

The node uses:

```text
llama-cpp-python
numpy
Pillow
```

ComfyUI itself already provides PyTorch.

### Important

You should **not need to downgrade or replace your current `llama-cpp-python` installation just for Rebels Prompt Enhancer**.

If the installed backend works, the node uses it.

If the supported Windows compatibility failure is detected, the curated/custom prompt rewriting path can use the private llama.cpp fallback instead.

---

# Basic Workflow

```text
Text
  ↓
🚀 Rebels Prompt Enhancer V3
  ↓
enhanced_prompt
  ↓
🔒 Rebels Prompt Locker
  ↓
Text Encoder / Generation Workflow
```

Optional outputs:

```text
thought_process
negative_prompt
variants_json
```

---

# Suggested LOW VRAM Setup

For smaller GPUs:

```text
Precision:
Ultra Low VRAM (UD-IQ2)

Load Mode:
VRAM Saver

Auto CPU Fallback:
ON
```

For a stronger balance between memory and prompt quality:

```text
Precision:
Balanced (Q4)

Load Mode:
VRAM Saver

Auto CPU Fallback:
ON
```

For systems with enough RAM/VRAM and users wanting the strongest curated quant:

```text
Precision:
High Quality (UD-Q8)
```

---

# Custom GGUF Models

The Custom GGUF enhancer scans the RebelsPromptEnhancer folder for `.gguf` files.

Place compatible text GGUF models directly inside:

```text
ComfyUI/custom_nodes/RebelsPromptEnhancer/
```

They will appear in the model dropdown after ComfyUI is restarted.

The node attempts to detect common families including:

- Qwen
- Llama
- Mistral / Mixtral
- Gemma
- Phi

Actual compatibility depends on the capabilities of the llama.cpp backend being used.

---

# Image to Prompt

The Image-to-Prompt node requires:

```text
model.gguf
mmproj-model.gguf
```

Both files should be placed directly inside:

```text
ComfyUI/custom_nodes/RebelsPromptEnhancer/
```

Supported handler types are detected on a best-effort basis from the installed `llama-cpp-python`.

Available handler families can include:

- LLaVA 1.5
- LLaVA 1.6
- Moondream
- MiniCPM-V 2.6
- NanoLLaVA
- Qwen2.5-VL

The vision node currently relies on the installed `llama-cpp-python` vision stack rather than the portable text fallback.

---

# LLM Console

The LLM Console allows any compatible text GGUF in the node folder to be used as a general local language model.

Controls include:

- system prompt
- context size
- maximum tokens
- GPU layers
- temperature
- Top-P
- repeat penalty
- seed
- `/no_think`
- load mode
- CPU fallback

This node is useful for directly testing whether a GGUF behaves correctly before using it for prompt enhancement.

---

# Prompt Locker

The Prompt Locker accepts a string input and either pauses or allows the workflow to continue.

### PAUSED

Stops execution.

### LOCKED IN

Passes the text downstream.

The included frontend extension:

```text
web/js/rebels_locker_display.js
```

allows the prompt to be displayed directly on the node.

---

# Folder Structure

A populated installation may look like:

```text
RebelsPromptEnhancer/
├── __init__.py
├── requirements.txt
├── VERSION
├── README.md
│
├── Qwen3.5-4B-UD-IQ2_M.gguf
├── Qwen3.5-4B-Q4_K_M.gguf
├── Qwen3.5-4B-UD-Q8_K_XL.gguf
│
├── .cache/
│   └── prompt_cache.json
│
├── .runtime/
│   └── llama_cpp_portable/
│
└── web/
    └── js/
        └── rebels_locker_display.js
```

The GGUF models, cache, and compatibility runtime are all stored relative to the custom node.

No developer-specific drive path is required.

---

# Troubleshooting

## Missing Curated Model

Use:

`📦 Rebels Prompt Enhancer Model Helper`

and toggle:

`DOWNLOAD MODEL`

or enable:

`auto_download_missing`

inside the curated enhancer.

---

## Download Was Interrupted

Run the downloader again.

If a valid partial `.part` file exists, Rebels Prompt Enhancer will attempt to resume the download.

---

## STATUS_ILLEGAL_INSTRUCTION / 0xC000001D

On supported Windows x64 systems, the curated/custom text-enhancement path should automatically switch to the private portable llama.cpp compatibility runtime.

You do not need to uninstall or downgrade your existing `llama-cpp-python` just because this error occurs.

---

## Model Loads but Prompt Is Not Enhanced

V3 includes a no-op quality detector.

Very sparse prompts that are returned essentially unchanged are automatically retried with stronger enhancement instructions.

Try increasing:

`detail_strength`

from:

`Preserve`

to:

`Expand`

or:

`Highly Detailed`

for larger expansions.

---

## VRAM Problems

Use:

```text
Load Mode:
VRAM Saver
```

and enable:

```text
Auto CPU Fallback:
ON
```

You can also select a smaller quant such as:

```text
Ultra Low VRAM (UD-IQ2)
```

---

# Privacy

Prompt enhancement is performed locally.

The enhancer does not require:

- OpenAI API
- Anthropic API
- Google API
- OpenRouter
- remote inference services

Network access is only needed when you explicitly download a model or when the Windows compatibility runtime needs to be downloaded for the first time.

After those files are present, prompt inference itself runs locally.

---

# Credits

Built by **Rebel AI**.

Powered by:

- ComfyUI
- llama.cpp
- llama-cpp-python
- Qwen3.5
- Unsloth GGUF quantizations

Built for creators who want powerful prompt enhancement without needing a cloud API or a massive GPU.
