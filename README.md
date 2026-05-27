# Rebels Prompt Enhancer for ComfyUI

An ultra-lightweight, local-only prompt enhancement node for ComfyUI. Built for high-performance workflows using the **MiniCPM5-1B** model. 

Designed to operate as a "disposable" node—it loads the LLM to process your prompt and **completely flushes VRAM** immediately after, ensuring zero interference with your generation models.

## Features
- **Local-Only:** No API keys, no external calls, 100% private.
- **VRAM Conscious:** Automatic memory management flushes the model after every use.
- **Multi-Purpose:** Dedicated modes for Photorealistic Images, Cinematic Video, and Inpainting/I2V editing.
- **Model Flexibility:** Choose between `Q4_K_M` (Efficiency) or `F16` (High Fidelity) precision.

## Installation
1. Navigate to your ComfyUI `custom_nodes` folder:
   `cd ComfyUI/custom_nodes`
2. Clone this repository:
   `git clone [YOUR_REPO_URL]`
3. Install dependencies:
   `../python_embeded/python.exe -m pip install llama-cpp-python`
4. **Download Models:** Place your `minicpm5-1b.Q4_K_M.gguf` and `minicpm5-1b.F16.gguf` files directly inside the `RebelsPromptEnhancer` folder.
5. Restart ComfyUI.

## Usage
Find the node under **"Rebel AI" > 🚀 Rebels Prompt Enhancer**.
- Connect your raw prompt input.
- Select your operational mode (Image, Video, or Editing).
- Select your preferred precision.
- Connect the output string to your KSampler/Video node.

---
*Built by Rebel AI.*
