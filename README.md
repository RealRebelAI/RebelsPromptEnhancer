# Rebels Prompt Enhancer V3 for ComfyUI


### V3.3.5 — Long/Max no-fail hotfix

- Fixes V3.3.4 causing ComfyUI execution errors when Qwen remained below the Long/Max target after repair attempts.
- Long/Max still receive up to five in-server expansion attempts.
- The best clean generated prompt is now always returned if the model remains under target.
- Under-length status is reported as a quality warning instead of crashing the workflow.
- Portable CUDA llama-server behavior and model loading are unchanged.



### V3.3.5 — strict Long / Max Detail length repair

- Long now requires at least **180 useful words / 900 characters** before it can leave the node.
- Max Detail requires at least **260 useful words / 1350 characters**.
- Under-length Long/Max drafts are expanded from the best existing draft instead of being replaced by shorter summaries.
- Long/Max get up to five in-server repair passes without reloading the GGUF.
- A candidate that satisfies the selected Long/Max floor always outranks an under-length candidate.
- The persistent portable CUDA `llama-server` architecture from V3.3.1+ is unchanged.



**Current Version: 3.3.5**

## V3.3.5 Prompt Quality + Max Detail Fixes

- Long now targets a real 180+ useful-word quality floor; Max Detail targets roughly 300-500 useful words for natural-language formats.
- Single-variant structured generation no longer writes the same long prompt twice. `variants` is derived internally from `final_prompt`, reducing output pressure and improving Long/Max reliability.
- Multiple-variant generation likewise avoids duplicating variant 1 into a separate `final_prompt` field.
- Quality repair attempts can no longer overwrite a good draft with an empty, shorter, contaminated, or otherwise worse retry.
- Under-length and repetition checks trigger targeted repairs, but a clean usable prompt is returned instead of throwing an execution error just because a model misses the target after all repairs.
- Obvious repeated visual phrases are detected and repaired.
- Long/Max output-token budgets were increased while keeping the reusable llama-server performance architecture unchanged.
- Quality retries may run up to three times and all reuse the already-loaded portable server.

## V3.3.1 Performance Fixes

- Portable fallback now uses a reusable private `llama-server` process instead of launching a fresh `llama-completion.exe` for every quality pass.
- Quality retries reuse the already-loaded GGUF, eliminating repeated Q8 reloads inside a single enhancement.
- `VRAM Saver` keeps the portable model loaded only for the current enhancer execution, then shuts the server down before downstream image/video nodes continue.
- `Fast Iteration` now works with the portable fallback too: the server stays alive between enhancements and reuses the loaded model until the model/context changes or ComfyUI exits.
- CUDA remains preferred on NVIDIA Windows systems; the private CPU server remains the fallback if CUDA cannot run.

## V3.3.0 Reliability Fixes

- Persists portable llama.cpp compatibility mode across ComfyUI restarts after a Windows `0xC000001D` native-backend failure.
- Existing V3.2.x portable-runtime installs migrate automatically and skip the known-bad native backend on the next restart.
- Long / Max Detail length is validated after generation instead of forcing huge JSON-string grammar lengths.
- Rejects/salvages prompt values contaminated with JSON keys, placeholders, `FINAL ANSWER`, `USER INPUT`, templates, or repeated JSON objects.
- Keeps aesthetic enforcement and automatic quality retries without allowing retry/schema text into the final prompt.

## V3.2.9 prompt-control enforcement

V3.2.9 introduced stronger prompt controls; V3.3.0 keeps those controls but moves length enforcement out of the JSON grammar for cleaner generation:

- Long and Max Detail use post-generation quality floors instead of oversized JSON-string grammar constraints.
- empty or contaminated structured output is treated as a failed enhancement and automatically repaired instead of being surfaced as raw JSON.
- output-token budget now scales with the selected length and number of variants.
- context size expands automatically when larger outputs need more room.
- selected aesthetics are treated as hard requirements and validated after generation.
- `Pixar / 3D Animation` now explicitly injects a Pixar-inspired 3D animation treatment.
- failed length/aesthetic/no-op checks automatically retry with targeted repair instructions while preserving the best clean draft.
- cache revision was bumped so older short/empty cached results are not reused.

The portable llama.cpp compatibility path still leaves the user's installed `llama-cpp-python` untouched. If Windows raises `STATUS_ILLEGAL_INSTRUCTION (0xC000001D)`, RebelsPromptEnhancer switches to its private portable llama.cpp runtime under the node's own `.runtime` folder and now remembers that compatibility choice across ComfyUI restarts.

A local prompt rewriting toolkit for ComfyUI built around GGUF language models through `llama-cpp-python`.

V3 keeps the original low-VRAM workflow but makes the enhancer much more reliable and configurable: structured JSON output, strict detail preservation, three curated quant tiers, Qwen Image/Edit formatting, stronger video prompting, persistent caching, optional negative prompts, multi-variant generation, automatic model-family behavior, CPU fallback, and an optional keep-loaded iteration mode.

No API key is required for prompt generation. Prompt rewriting runs locally after the GGUF is installed.

---

## Important folder layout

The custom node must be installed **directly** like this:

```text
ComfyUI/
└── custom_nodes/
    └── RebelsPromptEnhancer/
        ├── __init__.py
        ├── requirements.txt
        ├── README.md
        ├── *.gguf
        ├── mmproj-*.gguf
        ├── .cache/                 # created automatically
        ├── .runtime/               # created only if native compatibility fallback is needed
        └── web/
            └── js/
                └── rebels_locker_display.js
```

Do **not** install it as:

```text
custom_nodes/RebelsPromptEnhancer/RebelsPromptEnhancer/__init__.py
```

ComfyUI expects `__init__.py` in the custom-node package root.

The code never contains a machine-specific drive path. All files are resolved relative to this node's own folder with `Path(__file__).resolve().parent`.

---

# Nodes

## 🚀 Rebels Prompt Enhancer V3

Curated Qwen3.5-4B prompt rewriter.

### Inputs

| Input | Purpose |
|---|---|
| `raw_prompt` | Original prompt to rewrite |
| `purpose` | Image / Video / Edit |
| `model_format` | Target prompt structure |
| `aesthetic` | Optional aesthetic injection |
| `precision` | Ultra Low VRAM / Balanced / High Quality |
| `detail_strength` | Preserve / Expand / Highly Detailed / Creative |
| `length_target` | Short / Medium / Long / Max Detail |
| `preserve_user_details` | Prevents the rewriter from changing explicit details |
| `variant_count` | Generate 1-4 alternate rewrites |
| `negative_prompt` | Optional negative-prompt generation |
| `load_mode` | Unload after every run or keep model loaded for iteration |
| `auto_cpu_fallback` | Retry on CPU if GPU loading/inference runs out of memory |
| `auto_download_missing` | Explicit opt-in automatic download of the selected curated GGUF |
| `seed` | Generation seed |
| `lock_in` | Use persistent cached result instead of re-running the model |

### Outputs

The first two outputs remain in the same order as V2 for workflow compatibility:

1. `enhanced_prompt`
2. `thought_process`
3. `negative_prompt`
4. `variants_json`

`variants_json` is a JSON array containing the requested A/B variants. `enhanced_prompt` is the primary/default variant.

---

## 🧪 Rebels Prompt Enhancer V3 (Custom GGUF)

Uses any compatible text GGUF placed directly in the node folder.

It includes everything in the curated enhancer plus:

- manual GGUF selection
- extra system instructions
- full system-prompt override
- automatic model-family detection
- automatic Qwen3 `/no_think` behavior
- manual GPU-layer control
- context size
- max tokens
- temperature
- top-p
- repeat penalty

### Model-family behavior

The custom node currently recognizes filenames containing common family names such as:

- Qwen3 / Qwen3.5
- Qwen2 / Qwen2.5
- Llama
- Mistral / Mixtral
- Gemma
- Phi

`no_think = Auto` enables `/no_think` for Qwen3-family filenames and leaves it off for other families. `llama-cpp-python` still uses the GGUF's embedded chat template when available.

---

## 📦 Rebels Prompt Enhancer Model Helper

Checks whether the selected curated model exists.

With `CHECK ONLY`, it reports the expected filename and install location.

With `DOWNLOAD MODEL`, it explicitly downloads the selected curated model into the current custom-node folder. Downloads are never started unless the user enables the download toggle or enables `auto_download_missing` on the curated enhancer.

---

## 👁️ Rebels Image to Prompt

Vision GGUF + paired `mmproj` image captioning node.

Supported handler choices depend on the installed `llama-cpp-python` build and may include:

- LLaVA 1.5 / 1.6
- Moondream
- MiniCPM-V 2.6
- NanoLLaVA
- Qwen2.5-VL

The node looks only inside its own custom-node folder for the GGUF and mmproj files.

---

## 🧠 Rebels LLM Console

General local GGUF chat/testing node with:

- automatic `/no_think` behavior
- GPU layer selection
- context size
- sampling controls
- VRAM Saver / Fast Iteration loading
- CPU fallback on memory errors

---

## 🔒 Rebels Prompt Locker

Workflow gate. When paused it raises an intentional error to stop downstream execution. When locked in, the text passes through.

The bundled frontend extension displays the passed text directly on the node.

---

# V3 prompt system

## Strict detail preservation

When `preserve_user_details = true`, the system prompt explicitly instructs the LLM not to replace or contradict concrete user-provided information such as:

- identity or names
- number of people/objects
- face/body traits
- clothing
- colors
- exact text on signs/screens
- pose
- camera angle
- environment
- layout
- explicit constraints

The enhancer may elaborate only where the original prompt leaves details unspecified.

This is especially useful for image editing and reference-preservation workflows.

---

## Detail strength

### Preserve
Minimal useful expansion. Stays close to the source prompt.

### Expand
Adds concrete scene and visual information while remaining conservative.

### Highly Detailed
Expands subject, scene, lighting, camera, materials, atmosphere and composition.

### Creative
May invent tasteful unspecified details while still preserving every explicit user requirement.

---

## Prompt length

`length_target` changes the requested verbosity:

- Short: roughly 40-90 words
- Medium: roughly 90-180 words
- Long: roughly 180-320 words
- Max Detail: roughly 300-500 useful words for natural-language formats, while avoiding filler and repetition

The curated enhancer also adjusts its completion token budget based on this setting.

---

# Target model formats

V3 includes dedicated prompt structures for:

- Qwen Image / Qwen Image Edit
- Flux / Chroma
- Krea
- Z-Image / Lumina-2
- HiDream
- SDXL
- SD 1.5
- Pony / Illustrious
- Wan / Hunyuan Video
- LTX Video
- FastH3 / MiniMax Video
- Universal Natural Language

## Qwen Image / Qwen Image Edit

Uses dense natural-language prose and places the subject/composition early. It then adds environment, lighting, camera/lens, material/texture, color, mood and spatial information.

For editing, it emphasizes both the requested changes and the details that must remain unchanged.

## Video modes

Video formats now explicitly prioritize:

- temporal order
- subject motion
- camera movement
- movement speed
- environmental motion
- continuity
- stable elements
- pacing

This reduces prompts that describe a pretty frame but fail to describe actual motion.

---

# Structured output

V3 requests constrained JSON from `llama-cpp-python` using `response_format` and JSON Schema.

The model is asked to return:

```json
{
  "final_prompt": "...",
  "variants": ["..."],
  "negative_prompt": "..."
}
```

The negative field is only required when negative generation is enabled.

This is significantly more reliable than trying to remove reasoning, headings and drafts using regex after free-form generation.

A legacy cleanup fallback remains in the code so a model that fails structured output does not automatically make the workflow unusable.

---

# Persistent cache / lock

V2 cached only in Python memory. V3 stores locked rewrite results in:

```text
custom_nodes/RebelsPromptEnhancer/.cache/prompt_cache.json
```

This means a locked result can survive a ComfyUI restart.

The cache key includes the prompt and rewrite settings that affect the output. Seeds are intentionally irrelevant once a result is locked.

The cache is bounded so it does not grow forever.

---

# VRAM behavior

## VRAM Saver

```text
VRAM Saver (unload after run)
```

The GGUF is released after the enhancer execution finishes, garbage collection runs, and the CUDA cache is emptied. If a Long / Max Detail quality repair is needed, the portable llama-server is reused during that same execution instead of reloading the GGUF for every retry. This is the recommended mode when the same GPU also needs to load the image/video model.

## Fast Iteration

```text
Fast Iteration (keep model loaded)
```

Keeps one GGUF resident for repeated prompt iteration. This now applies to both the in-process `llama-cpp-python` path and the private portable llama-server fallback. Switching to another GGUF/context restarts the hot model/server.

Use this when you are testing many prompt rewrites before starting a diffusion generation. On low-VRAM systems, switch back to VRAM Saver before running a large image/video model if necessary.

---

# Automatic CPU fallback

When enabled, a GPU memory-allocation failure while loading or running the GGUF causes one retry with:

```text
n_gpu_layers = 0
```

The node does not silently retry arbitrary errors. Fallback is limited to errors that look like memory/CUDA allocation failures.

CPU generation will be much slower but can save a workflow that otherwise fails due to VRAM pressure.

---

# Curated models

Curated GGUF source:

```text
https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/tree/main
```

Download only the tier you want and put the GGUF directly beside `__init__.py`.

| Tier | Preferred file | Approx size |
|---|---|---:|
| Ultra Low VRAM | `Qwen3.5-4B-UD-IQ2_M.gguf` | 1.76 GB |
| Balanced | `Qwen3.5-4B-Q4_K_M.gguf` | 2.74 GB |
| High Quality | `Qwen3.5-4B-UD-Q8_K_XL.gguf` | 5.95 GB |

The Balanced selector intentionally resolves only `Qwen3.5-4B-Q4_K_M.gguf`; `UD-Q4_K_XL` is not selected by this preset.

The curated node searches filenames case-insensitively, so the important quant/model tokens must remain in the filename.

---

# Installation

## 1. Open ComfyUI's custom nodes folder

From the ComfyUI folder:

```bat
cd custom_nodes
```

## 2. Clone

```bat
git clone https://github.com/RealRebelAI/RebelsPromptEnhancer.git
```

After cloning, verify this exact file exists:

```text
custom_nodes/RebelsPromptEnhancer/__init__.py
```

If the repository is packaged with another `RebelsPromptEnhancer` folder inside it, move the package contents up one level. The release/package should be kept flat so this step is normally unnecessary.

## 3. Install dependencies

### ComfyUI Portable

Run from inside:

```text
custom_nodes/RebelsPromptEnhancer
```

Then:

```bat
..\..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

### Standard Python / Desktop

```bat
pip install -r requirements.txt
```

## 4. Install a GGUF

Either manually download one of the curated files above or use the Model Helper node.

Place it here:

```text
custom_nodes/RebelsPromptEnhancer/<model>.gguf
```

## 5. Restart ComfyUI

The nodes appear under:

```text
Rebel AI
```

---

# Basic workflow

```text
Prompt text
    │
    ▼
🚀 Rebels Prompt Enhancer V3
    │ enhanced_prompt
    ▼
Positive prompt / text encode path
```

Optional gate:

```text
Prompt
  │
  ▼
Enhancer
  │
  ▼
Prompt Locker
  │
  ▼
Positive prompt path
```

For A/B testing, connect `variants_json` to a text preview or parse/select one of the alternatives downstream.

---

# Recommended settings

## RTX-class 8 GB card / low VRAM

```text
precision: Ultra Low VRAM (UD-IQ2) or Balanced (Q4)
load_mode: VRAM Saver
length_target: Medium
preserve_user_details: true
auto_cpu_fallback: true
variant_count: 1
```

## Prompt-writing session before generation

```text
precision: Balanced (Q4)
load_mode: Fast Iteration
variant_count: 2-4
```

When finished iterating, switch back to VRAM Saver before loading a large diffusion/video model if necessary.

---

# Upgrading from V2

Back up your existing folder first.

Replace:

```text
__init__.py
requirements.txt
web/js/rebels_locker_display.js
README.md
```

Your GGUF files can remain beside `__init__.py`.

The original class identifiers remain:

```text
RebelsPromptEnhancer
RebelsPromptEnhancerCustom
RebelsImageToPrompt
RebelsLLMConsole
RebelsPromptLocker
```

so ComfyUI can still identify the node types. V3 appends new outputs to the enhancer nodes while leaving `enhanced_prompt` and `thought_process` as outputs 0 and 1.

A new node identifier is added:

```text
RebelsPromptEnhancerModelHelper
```

Because V3 adds inputs, existing saved enhancer nodes may show new widgets with defaults after the update. If ComfyUI displays a stale/broken node after an upgrade, recreate that node once so the current input schema is saved into the workflow.

---

# Troubleshooting

## `No GGUF found`

Verify:

```text
custom_nodes/RebelsPromptEnhancer/*.gguf
```

The GGUF must be beside `__init__.py`, not inside another nested package directory.

## ComfyUI says `__init__.py` is missing

Your repository is nested incorrectly. You probably have:

```text
custom_nodes/RebelsPromptEnhancer/RebelsPromptEnhancer/__init__.py
```

Move the inner package contents up so you have:

```text
custom_nodes/RebelsPromptEnhancer/__init__.py
```

## Curated tier says model missing

Use the Model Helper node or make sure the filename contains the expected model/quant tokens.

## Qwen outputs thinking/reasoning

The curated Qwen3.5 node always appends `/no_think`.

For Custom GGUF, keep:

```text
no_think: Auto
```

or force it on manually.

## GPU OOM

Try, in order:

1. `VRAM Saver`
2. Ultra Low VRAM quant
3. reduce custom `n_gpu_layers`
4. enable `auto_cpu_fallback`
5. use `n_gpu_layers = 0` manually in the Custom node

## Structured JSON fails

V3 automatically falls back to legacy text cleanup. The diagnostic output reports whether the result used:

```text
structured
```

or:

```text
legacy-fallback
```

Some unusual/custom GGUF chat templates may be less reliable in constrained JSON mode.

---

# Notes

- Prompt generation itself is local.
- The optional model downloader accesses Hugging Face only when explicitly enabled. The private compatibility runtime accesses the official `ggml-org/llama.cpp` GitHub release only after an actual `0xC000001D` native-backend failure.
- Vision support depends on the chat handlers available in the installed `llama-cpp-python` version.
- Fast Iteration intentionally keeps one LLM/server resident between enhancements; VRAM Saver unloads the portable server when the enhancer execution finishes.
- Persistent cache files contain generated prompt text. Delete `.cache/` if you want to clear them.

---

Built by Rebel AI.
