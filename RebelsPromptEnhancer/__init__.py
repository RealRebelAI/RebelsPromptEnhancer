import torch
import gc
import os
from llama_cpp import Llama


class RebelsPromptEnhancer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_prompt": ("STRING", {"multiline": True}),
                "purpose": ([
                    "Image Prompt (Photorealistic)",
                    "Video Prompt (Cinematic)",
                    "Editing (Inpainting/I2V)",
                ],),
                "precision": (["Efficiency (Q4_K_M)", "High Fidelity (F16)"],),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "thought_process")
    FUNCTION = "enhance"
    CATEGORY = "Rebel AI"

    # Aggressive, structured system prompts. 1B models respond to ALL-CAPS
    # directives and numbered rules far better than polite instructions.
    SYSTEM_INSTRUCTIONS = {
        "Image Prompt (Photorealistic)": (
            "You are a TEXT-ONLY prompt expansion machine for image generation.\n"
            "RULES:\n"
            "1. Output ONE single descriptive paragraph for a photorealistic 8k scene.\n"
            "2. Include lighting, composition, lens type, mood, and fine surface detail.\n"
            "3. NEVER repeat the user's input verbatim.\n"
            "4. NEVER write 'Here is', 'Sure', 'Certainly', 'Okay', or any conversational filler.\n"
            "5. NEVER explain what you are doing. NEVER add commentary or notes.\n"
            "6. Start the response directly with visual description words.\n"
            "OUTPUT THE PROMPT ONLY. NOTHING ELSE."
        ),
        "Video Prompt (Cinematic)": (
            "You are a TEXT-ONLY prompt expansion machine for video generation.\n"
            "RULES:\n"
            "1. Output ONE single paragraph describing a cinematic video shot.\n"
            "2. Include camera movement, subject motion, pacing, lighting, and mood.\n"
            "3. NEVER repeat the user's input verbatim.\n"
            "4. NEVER write 'Here is', 'Sure', 'Certainly', 'Okay', or any conversational filler.\n"
            "5. NEVER explain what you are doing. NEVER add commentary or notes.\n"
            "6. Start the response directly with the cinematic description.\n"
            "OUTPUT THE PROMPT ONLY. NOTHING ELSE."
        ),
        "Editing (Inpainting/I2V)": (
            "You are a TEXT-ONLY prompt expansion machine for image editing.\n"
            "RULES:\n"
            "1. Output ONE single paragraph describing the FINAL transformed scene as if it already exists.\n"
            "2. Do not describe an action or process — describe the result.\n"
            "3. NEVER repeat the user's input verbatim.\n"
            "4. NEVER write 'Here is', 'Sure', 'Certainly', 'Okay', or any conversational filler.\n"
            "5. NEVER explain what you are doing. NEVER add commentary or notes.\n"
            "6. Start the response directly with the description of the final scene.\n"
            "OUTPUT THE PROMPT ONLY. NOTHING ELSE."
        ),
    }

    # Common chatbot preambles to strip from start of output.
    PREAMBLES = (
        "here's", "here is", "sure,", "sure!", "sure.",
        "certainly,", "certainly!", "certainly.",
        "of course,", "of course!", "okay,", "okay.", "alright,",
        "i'll", "i will", "let me",
        "enhanced prompt:", "expanded prompt:", "prompt:", "output:",
    )

    def _clean_output(self, text, raw_prompt):
        text = text.strip()

        # Iteratively strip preambles — sometimes the model nests them
        # (e.g. "Sure! Here's the prompt: ...").
        for _ in range(3):
            lowered = text.lower().lstrip()
            stripped = False
            for p in self.PREAMBLES:
                if lowered.startswith(p):
                    newline_idx = text.lower().find("\n")
                    colon_idx = text.find(":")
                    # If there's a colon early on, cut after it. Otherwise,
                    # if there's a newline, cut after that. Otherwise drop
                    # just the preamble token.
                    if 0 < colon_idx < 60:
                        text = text[colon_idx + 1:].strip()
                    elif 0 < newline_idx < 80:
                        text = text[newline_idx + 1:].strip()
                    else:
                        text = text[len(p):].strip(" ,.:-")
                    stripped = True
                    break
            if not stripped:
                break

        # Strip exact input repetition at start (case-insensitive).
        raw_stripped = raw_prompt.strip()
        if raw_stripped and text.lower().startswith(raw_stripped.lower()):
            text = text[len(raw_stripped):].lstrip(" ,.:-\"'\n")

        # Strip wrapping quotes the model often adds.
        text = text.strip().strip('"\'')

        return text

    def enhance(self, raw_prompt, purpose, precision, seed):
        node_dir = os.path.dirname(os.path.abspath(__file__))
        search_terms = {"Efficiency (Q4_K_M)": "q4_k_m", "High Fidelity (F16)": "f16"}
        term = search_terms.get(precision)

        files = [
            f for f in os.listdir(node_dir)
            if f.lower().endswith(".gguf") and term in f.lower()
        ]
        if not files:
            raise FileNotFoundError(
                f"Rebels Prompt Enhancer Error: Could not find a '.gguf' file "
                f"containing '{term}' in {node_dir}"
            )

        model_file = files[0]
        model_path = os.path.join(node_dir, model_file)

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            verbose=False,
            n_ctx=2048,
            seed=seed,
        )

        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.SYSTEM_INSTRUCTIONS[purpose]},
                    {"role": "user", "content": raw_prompt},
                ],
                max_tokens=512,           # 512 is plenty; 2048 invites rambling
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.15,      # kills loops on small models
                stop=[
                    "\n\nUser:", "\n\nAssistant:",
                    "\n\nNote:", "\n\nExplanation:",
                    "</prompt>", "Human:",
                ],
            )
            raw_output = output["choices"][0]["message"]["content"].strip()
        finally:
            # Memory cleanup, even on error.
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_prompt = self._clean_output(raw_output, raw_prompt)

        # Fallback: if the cleaner ate everything, return the raw output
        # rather than an empty string.
        if not final_prompt.strip():
            final_prompt = raw_output

        thought = (
            f"Model: {model_file}\n"
            f"Purpose: {purpose}\n"
            f"Precision: {precision}\n"
            f"Seed: {seed}\n"
            f"Input chars: {len(raw_prompt)}\n"
            f"Raw output chars: {len(raw_output)}\n"
            f"Cleaned output chars: {len(final_prompt)}"
        )

        return (final_prompt, thought)


class RebelsPromptLocker:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
                "lock_in_prompt": ("BOOLEAN", {
                    "default": False,
                    "label_on": "LOCKED IN",
                    "label_off": "PAUSED",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "execute"
    CATEGORY = "Rebel AI"
    OUTPUT_NODE = True

    def execute(self, text_input, lock_in_prompt):
        if not lock_in_prompt:
            raise ValueError(
                "🛑 WORKFLOW PAUSED: Toggle 'lock_in_prompt' to LOCKED IN to pass the text through."
            )
        return {"ui": {"text": [text_input]}, "result": (text_input,)}


NODE_CLASS_MAPPINGS = {
    "RebelsPromptEnhancer": RebelsPromptEnhancer,
    "RebelsPromptLocker": RebelsPromptLocker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsPromptEnhancer": "🚀 Rebels Prompt Enhancer",
    "RebelsPromptLocker": "🔒 Rebels Prompt Locker",
}

# Tells ComfyUI where to find the JS extension that renders text on the locker node face.
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
