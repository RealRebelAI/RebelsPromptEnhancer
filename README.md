# Rebels Prompt Enhancer Nodes for ComfyUI

**WORK IN PROGRESS**

Two ultra-lightweight, local-only nodes for ComfyUI built around **Qwen3.5-4B**:

- 🚀 **Rebels Prompt Enhancer** — runs Qwen3.5-4B locally as a GGUF to expand or rewrite your prompt into a detailed image / video / edit description, then completely flushes VRAM so your diffusion model has the GPU to itself.
- 🔒 **Rebels Prompt Locker** — workflow gate that halts the run until you toggle it, and displays the passed-through prompt directly on the node face.

No API keys, no external calls, 100% private.

---

## Features

- **Local-Only:** Everything runs on your machine via `llama-cpp-python`. Nothing leaves the box.
- **Aggressive VRAM Cleanup:** The LLM is fully unloaded after each generation. Zero residual VRAM for the diffusion stage.
- **Qwen3.5-4B with `/no_think`:** Qwen3's built-in reasoning mode is disabled by default, so output arrives clean — no chain-of-thought, just the prompt.
- **Two Quant Tiers:** **Efficiency (UD-IQ2)** for fast and tiny, or **Quality (UD-Q8)** for the best output the model can produce.
- **Three Purpose Modes:** Photorealistic Image, Cinematic Video, or Inpainting/I2V editing.
- **Prompt Lock + Cache:** Toggle the enhancer to **LOCKED** and it skips model loading entirely, returning the cached output. Iterate freely until you find a prompt you love, then lock to freeze it — seed becomes irrelevant.
- **On-Node Text Display:** The Locker shows the passed-through prompt directly on its node face. No separate Preview-As-Text node needed.

---

## The Nodes

### 🚀 Rebels Prompt Enhancer

Inputs:
- `raw_prompt` — your input idea
- `purpose` — Image Prompt (Photorealistic) / Video Prompt (Cinematic) / Editing (Inpainting/I2V)
- `precision` — Efficiency (UD-IQ2) or Quality (UD-Q8)
- `seed` — randomize freely; ignored when locked
- `lock_in` — `🔄 LIVE` = generate fresh each run, `🔒 LOCKED` = return cached output, no model load

Outputs:
- `enhanced_prompt` — clean prompt string ready for your sampler
- `thought_process` — diagnostic dump (model file, char counts, raw model output pre-clean, final prompt). Wire to a Preview-As-Text node if you want to see what the model said.

### 🔒 Rebels Prompt Locker

Inputs:
- `text_input` — string to gate (typically the enhancer's `enhanced_prompt`)
- `lock_in_prompt` — `PAUSED` halts the workflow with a ValueError; `LOCKED IN` lets the string through

Output:
- `text_output` — passes through `text_input` when LOCKED IN
- Also displays the passed-through text on the node face (requires the bundled JS file).

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

3. Install dependencies (open cmd inside the cloned folder):
   ```
   ../../python_embeded/python.exe -m pip install -r requirements.txt
   ```

4. **Download a model file** (see Models below) and place it directly inside this node pack's folder.

5. Restart ComfyUI.

---

## Models

Repo: https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main

Place the GGUF file directly inside the node pack folder. The enhancer searches by filename substring (`qwen3.5-4b` + either `ud-iq2` or `ud-q8`), so renames are fine as long as the key tokens are intact.

| Precision | File | Approx Size |
|---|---|---|
| Efficiency (UD-IQ2) | `Qwen3.5-4B-UD-IQ2_XXS.gguf` (or any `UD-IQ2_*` variant) | 1.5–1.8 GB |
| Quality (UD-Q8) | `Qwen3.5-4B-UD-Q8_K_XL.gguf` | 5.95 GB |

You only need to download the variant(s) you plan to use. The node throws a clear FileNotFoundError if you select a precision whose file isn't present.

---

## Hardware Notes

The enhancer aggressively unloads the model after each run, so VRAM only needs to fit the LLM during the prompt rewrite phase — not alongside your diffusion model.

| Precision | Approx VRAM (model + 4k ctx) |
|---|---|
| Efficiency (UD-IQ2) | ~2 GB |
| Quality (UD-Q8) | ~6.5 GB |

Both fit on an 8 GB card. UD-Q8_K_XL is tight — if you hit OOM, drop down to a smaller `UD-Q*` variant (e.g. `UD-Q6_K_XL` at ~4.5 GB) and edit `PRECISION_SEARCH` in `__init__.py` to point Quality at it.

---

## Usage

Find both nodes under **"Rebel AI"** in the node menu.

Typical wiring:

```
[Your Prompt Text] → raw_prompt → 🚀 Enhancer → enhanced_prompt → 🔒 Locker → KSampler
                                              → thought_process → Preview-As-Text (optional)
```

### The Lock workflow

1. Start with the enhancer in **LIVE** mode and seed on `randomize`.
2. Queue runs and iterate — each run produces a new prompt because the seed changes.
3. When you find a prompt you love, toggle the enhancer to **LOCKED**. From then on the enhancer skips model loading and returns the cached prompt every run. Seed changes are ignored while locked.
4. Independently, the Locker gates downstream execution — flip it to `LOCKED IN` to allow the prompt to flow to the sampler.

The two locks are intentionally separate:
- **Enhancer lock** freezes the prompt (cache).
- **Locker** halts the workflow (gate).

The cache lives in memory for the ComfyUI session and clears on restart. Changing `raw_prompt`, `purpose`, or `precision` while locked invalidates the cache and triggers a fresh generation under the new key.

---

## Folder Structure

```
RebelsPromptEnhancer/
├── __init__.py
├── requirements.txt
├── README.md
├── Qwen3.5-4B-UD-*.gguf            ← drop your model file(s) here
└── web/
    └── js/
        └── rebels_locker_display.js
```

The `web/js/` extension is what makes the Locker display its text on the node face. Without it, the Locker still functions but won't render the text visually.

---

*Built by Rebel AI.*
