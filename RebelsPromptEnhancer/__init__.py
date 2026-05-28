import torch
import gc
import os
import re
from llama_cpp import Llama


class RebelsPromptEnhancer:
    def __init__(self):
        pass

    # Class-level cache survives across executions within a ComfyUI session.
    # Cleared on ComfyUI restart. Keyed by (raw_prompt, purpose, model, precision)
    # — deliberately excludes seed so locking truly freezes the output.
    _cache = {}

    MODEL_OPTIONS = ["MiniCPM5-1B", "Qwen3.5-4B"]
    PRECISION_OPTIONS = ["Efficiency (Smallest)", "Quality (Largest)"]

    MODEL_SEARCH = {
        ("MiniCPM5-1B", "Efficiency (Smallest)"): ["minicpm5", "q4_k_m"],
        ("MiniCPM5-1B", "Quality (Largest)"):     ["minicpm5", "f16"],
        ("Qwen3.5-4B", "Efficiency (Smallest)"):  ["qwen3.5-4b", "ud-iq2"],
        ("Qwen3.5-4B", "Quality (Largest)"):      ["qwen3.5-4b", "ud-q8"],
    }

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
                "model": (s.MODEL_OPTIONS,),
                "precision": (s.PRECISION_OPTIONS,),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lock_in": ("BOOLEAN", {
                    "default": False,
                    "label_on": "🔒 LOCKED (cached)",
                    "label_off": "🔄 LIVE (generating)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "thought_process")
    FUNCTION = "enhance"
    CATEGORY = "Rebel AI"

    @classmethod
    def IS_CHANGED(cls, raw_prompt, purpose, model, precision, seed, lock_in):
        """When locked, return a stable hash so ComfyUI uses its cached output
        and doesn't even call enhance(). When live, return NaN so it always
        re-executes (NaN != NaN)."""
        if lock_in:
            return f"LOCKED|{raw_prompt}|{purpose}|{model}|{precision}"
        return float("nan")

    SYSTEM_INSTRUCTIONS = {
        "Image Prompt (Photorealistic)": (
            "Rewrite the user's input as one descriptive paragraph for a photorealistic "
            "8k image with cinematic lighting. Output only the paragraph."
        ),
        "Video Prompt (Cinematic)": (
            "Rewrite the user's input as one descriptive paragraph for a cinematic "
            "video shot with camera movement and mood. Output only the paragraph."
        ),
        "Editing (Inpainting/I2V)": (
            "Rewrite the user's edit instruction as one descriptive paragraph describing "
            "the final transformed scene as it appears. Output only the paragraph."
        ),
    }

    REASONING_MARKERS = (
        "let me", "i'll ", "i will ", "i need", "i must", "i should",
        "the prompt is", "the user", "key elements", "brainstorm",
        "as per the rules", "according to the rules", "the rules say",
        "let's", "okay so", "first,", "second,", "third,",
        "i can say", "i might", "since this", "so i",
    )

    PREAMBLES = (
        "here's", "here is", "sure,", "sure!", "certainly,", "of course,",
        "okay,", "okay.", "alright,",
        "enhanced prompt:", "expanded prompt:", "prompt:",
        "output:", "answer:", "final prompt:", "final:", "example:",
    )

    def _get_system_prompt(self, model, purpose):
        base = self.SYSTEM_INSTRUCTIONS[purpose]
        if model.startswith("Qwen3"):
            return base + " /no_think"
        return base

    def _find_model_files(self, node_dir, terms):
        terms = [t.lower() for t in terms]
        out = []
        for f in os.listdir(node_dir):
            fl = f.lower()
            if fl.endswith(".gguf") and all(t in fl for t in terms):
                out.append(f)
        return out

    def _strip_thinking_tags(self, text):
        for pat in (
            r"<think(?:ing)?>.*?</think(?:ing)?>",
            r"<\|thinking\|>.*?<\|/thinking\|>",
            r"\[THINK(?:ING)?\].*?\[/THINK(?:ING)?\]",
        ):
            text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
        return text

    def _extract_final_paragraph(self, text):
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paragraphs) < 2:
            return text

        full_lower = text.lower()
        if sum(1 for m in self.REASONING_MARKERS if m in full_lower) < 2:
            return text

        for p in reversed(paragraphs):
            lower = p.lower().lstrip()
            if any(lower.startswith(m) for m in self.REASONING_MARKERS):
                continue
            lines = p.split("\n")
            bullets = sum(
                1 for l in lines
                if l.strip().startswith(("-", "*", "•", "1.", "2.", "3."))
            )
            if bullets > 0 and bullets >= len(lines) / 2:
                continue
            if len(p) < 60:
                continue
            if sum(1 for m in self.REASONING_MARKERS if m in lower) >= 2:
                continue
            return p
        return text

    def _strip_preambles(self, text):
        for _ in range(3):
            lowered = text.lower().lstrip()
            stripped = False
            for p in self.PREAMBLES:
                if lowered.startswith(p):
                    colon = text.find(":")
                    newline = text.find("\n")
                    if 0 < colon < 60:
                        text = text[colon + 1:].strip()
                    elif 0 < newline < 80:
                        text = text[newline + 1:].strip()
                    else:
                        text = text[len(p):].strip(" ,.:-")
                    stripped = True
                    break
            if not stripped:
                break
        return text

    def _clean_output(self, text, raw_prompt):
        text = text.strip()
        text = self._strip_thinking_tags(text)
        text = self._extract_final_paragraph(text)
        text = self._strip_preambles(text)
        raw = raw_prompt.strip()
        if raw and text.lower().startswith(raw.lower()):
            text = text[len(raw):].lstrip(" ,.:-\"'\n")
        return text.strip().strip('"\'')

    def enhance(self, raw_prompt, purpose, model, precision, seed, lock_in):
        # Cache key intentionally excludes seed — locking freezes regardless of seed changes.
        cache_key = (raw_prompt, purpose, model, precision)

        # --- LOCKED PATH: return cached output if we have one ---
        if lock_in and cache_key in self._cache:
            cached = self._cache[cache_key]
            cached_prompt = cached["final_prompt"]
            cached_raw = cached["raw_output"]
            cached_meta = cached["meta"]

            thought = (
                f"=== 🔒 LOCKED — Returning Cached Output ===\n"
                f"No model load. No VRAM used. Seed ignored.\n"
                f"\n"
                f"=== Original Generation Info ===\n"
                f"{cached_meta}"
                f"\n"
                f"=== Cached Raw Output ===\n"
                f"{cached_raw}\n"
                f"\n"
                f"=== Cached Final Prompt ===\n"
                f"{cached_prompt}"
            )
            return (cached_prompt, thought)

        # --- LIVE PATH: generate fresh ---
        node_dir = os.path.dirname(os.path.abspath(__file__))

        terms = self.MODEL_SEARCH.get((model, precision))
        if not terms:
            raise ValueError(
                f"Rebels Prompt Enhancer: unknown combination {model} / {precision}"
            )

        files = self._find_model_files(node_dir, terms)
        if not files:
            raise FileNotFoundError(
                f"Rebels Prompt Enhancer: no .gguf in {node_dir} matching all of {terms}.\n"
                f"Expected a filename containing: {' AND '.join(terms)}"
            )

        files.sort(key=lambda x: (len(x), x))
        model_file = files[0]
        model_path = os.path.join(node_dir, model_file)

        n_ctx = 4096 if model.startswith("Qwen3") else 2048

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            verbose=False,
            n_ctx=n_ctx,
            seed=seed,
        )

        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self._get_system_prompt(model, purpose)},
                    {"role": "user", "content": raw_prompt},
                ],
                max_tokens=400,
                temperature=0.7,
                top_p=0.9,
                repeat_penalty=1.15,
                stop=[
                    "\n\nUser:", "\n\nAssistant:", "Human:",
                    "</think>", "</thinking>",
                ],
            )
            raw_output = output["choices"][0]["message"]["content"].strip()
        finally:
            del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_prompt = self._clean_output(raw_output, raw_prompt)
        if not final_prompt.strip() or len(final_prompt) < 30:
            final_prompt = raw_output

        meta_block = (
            f"File:       {model_file}\n"
            f"Family:     {model}\n"
            f"Precision:  {precision}\n"
            f"Purpose:    {purpose}\n"
            f"Context:    {n_ctx}\n"
            f"Seed:       {seed}\n"
            f"Raw chars:  {len(raw_output)}\n"
            f"Clean chars:{len(final_prompt)}\n"
            f"Stripped:   {len(raw_output) - len(final_prompt)} chars\n"
        )

        thought = (
            f"=== 🔄 LIVE — Fresh Generation ===\n"
            f"\n"
            f"=== Model Info ===\n"
            f"{meta_block}"
            f"\n"
            f"=== Raw Model Output (pre-clean) ===\n"
            f"{raw_output}\n"
            f"\n"
            f"=== Final Prompt (sent to Locker) ===\n"
            f"{final_prompt}"
        )

        # Cache this run so a later lock_in toggle has something to freeze on.
        self._cache[cache_key] = {
            "final_prompt": final_prompt,
            "raw_output": raw_output,
            "meta": meta_block,
        }

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

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
