import torch
import gc
import os
import re
import base64
from io import BytesIO

import numpy as np
from PIL import Image
from llama_cpp import Llama

# Vision chat handlers — best-effort import.
_HANDLER_CLASSES = {}
for _label, _modname in [
    ("LLaVA 1.5", "Llava15ChatHandler"),
    ("LLaVA 1.6", "Llava16ChatHandler"),
    ("Moondream", "MoondreamChatHandler"),
    ("MiniCPM-V 2.6", "MiniCPMv26ChatHandler"),
    ("NanoLLaVA", "NanoLlavaChatHandler"),
    ("Qwen2.5-VL", "Qwen25VLChatHandler"),
]:
    try:
        _mod = __import__("llama_cpp.llama_chat_format", fromlist=[_modname])
        _HANDLER_CLASSES[_label] = getattr(_mod, _modname)
    except Exception:
        pass


# =========================================================================
# Shared cleaning helpers
# =========================================================================

REASONING_MARKERS = (
    "let me", "i'll ", "i will ", "i need", "i must", "i should",
    "the prompt is", "the user", "key elements", "brainstorm",
    "as per the rules", "according to the rules", "the rules say",
    "let's", "okay so", "first,", "second,", "third,",
    "i can say", "i might", "since this", "so i",
    "thinking process", "analyze the request", "deconstruct",
    "step-by-step", "step by step", "drafting",
)

PREAMBLES = (
    "here's", "here is", "sure,", "sure!", "certainly,", "of course,",
    "okay,", "okay.", "alright,",
    "enhanced prompt:", "expanded prompt:", "prompt:",
    "output:", "answer:", "final prompt:", "final:", "example:",
    "the image shows", "the image depicts", "this image shows",
    "in the image", "i can see", "i see",
)

# Stop sequences to cut off verbose preamble patterns before they eat the token budget.
VERBOSE_STOPS = [
    "\n\nUser:", "\n\nAssistant:", "Human:",
    "</think>", "</thinking>",
    "Thinking Process:", "**Thinking Process",
    "## Thinking", "### Thinking",
    "Analysis:", "**Analysis",
]


def _strip_thinking_tags(text):
    for pat in (
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        r"<\|thinking\|>.*?<\|/thinking\|>",
        r"\[THINK(?:ING)?\].*?\[/THINK(?:ING)?\]",
    ):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _extract_drafted_content(text):
    """For verbose models that show reasoning then draft a prompt, find the
    draft. Looks for section markers like 'Drafting', 'Final Prompt', 'Output'
    and returns content after the last occurrence."""
    marker_patterns = [
        r"(?:^|\n)\s*\d+\.\s*\*{0,2}\s*Draft(?:ing)?[^\n]*",
        r"(?:^|\n)#{0,3}\s*\*{0,2}\s*Draft(?:ing)?[^\n]*",
        r"(?:^|\n)#{0,3}\s*\*{0,2}\s*(?:Final\s+(?:Prompt|Output|Answer)|Final|Prompt|Output|Result)\s*:?\s*\*{0,2}",
    ]
    last_match_end = -1
    for pat in marker_patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            if m.end() > last_match_end:
                last_match_end = m.end()

    if last_match_end > 0:
        candidate = text[last_match_end:].strip()
        candidate = candidate.lstrip(":*-• \t\n")
        candidate = re.sub(
            r"\*{1,2}\s*(?:Subject|Setting|Lighting|Composition|Mood|Style|Environment|Atmosphere|Scene|Background|Foreground|Pose|Camera)\s*:?\s*\*{0,2}\s*",
            "", candidate, flags=re.IGNORECASE,
        )
        candidate = re.sub(r"(?:^|\n)\s*[-*•]\s+", " ", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate) > 40:
            return candidate
    return text


def _extract_final_paragraph(text):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 2:
        return text
    full_lower = text.lower()
    if sum(1 for m in REASONING_MARKERS if m in full_lower) < 2:
        return text
    for p in reversed(paragraphs):
        lower = p.lower().lstrip()
        if any(lower.startswith(m) for m in REASONING_MARKERS):
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
        if sum(1 for m in REASONING_MARKERS if m in lower) >= 2:
            continue
        return p
    return text


def _strip_preambles(text):
    for _ in range(3):
        lowered = text.lower().lstrip()
        stripped = False
        for p in PREAMBLES:
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


def _clean_output(text, original_input=""):
    """Aggressive cleaning for prompt rewriter outputs."""
    text = text.strip()
    text = _strip_thinking_tags(text)
    text = _extract_drafted_content(text)
    text = _extract_final_paragraph(text)
    text = _strip_preambles(text)
    if original_input:
        raw = original_input.strip()
        if raw and text.lower().startswith(raw.lower()):
            text = text[len(raw):].lstrip(" ,.:-\"'\n")
    return text.strip().strip('"\'')


def _clean_chat_output(text):
    """Light cleaning for general chat. Keeps conversational structure intact."""
    text = text.strip()
    text = _strip_thinking_tags(text)
    return text.strip()


def _free_llm(llm):
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _list_ggufs(exclude_mmproj=True):
    """List .gguf files in the node folder. Excludes any file with 'mmproj'
    anywhere in the name (those are vision projectors, not standalone models)."""
    try:
        node_dir = os.path.dirname(os.path.abspath(__file__))
        files = []
        for f in os.listdir(node_dir):
            if not f.lower().endswith(".gguf"):
                continue
            if exclude_mmproj and "mmproj" in f.lower():
                continue
            files.append(f)
        files.sort()
        return files if files else ["NO_GGUF_FILES_IN_FOLDER"]
    except Exception:
        return ["NO_GGUF_FILES_IN_FOLDER"]


def _list_mmproj():
    """List mmproj projector files. Matches 'mmproj' ANYWHERE in the filename,
    so it catches mmproj-F16.gguf, moondream2-mmproj-f16.gguf, etc."""
    try:
        node_dir = os.path.dirname(os.path.abspath(__file__))
        files = sorted(
            f for f in os.listdir(node_dir)
            if f.lower().endswith(".gguf") and "mmproj" in f.lower()
        )
        return files if files else ["NO_MMPROJ_FILE_FOUND"]
    except Exception:
        return ["NO_MMPROJ_FILE_FOUND"]


# =========================================================================
# Style system — Purpose + Model Format + Aesthetic
# =========================================================================

PURPOSE_OPTIONS = ["Image", "Video", "Edit (Inpainting/I2V)"]

PURPOSE_FRAMING = {
    "Image": (
        "Rewrite the user's input as a detailed prompt for a static image. "
        "Cover subject, environment, lighting, composition, and mood."
    ),
    "Video": (
        "Rewrite the user's input as a detailed prompt for a video shot. "
        "Cover subject motion, camera movement, pacing, lighting, and mood."
    ),
    "Edit (Inpainting/I2V)": (
        "Rewrite the user's edit instruction as a description of the final transformed "
        "scene as it appears after the edit. Do not describe the editing process."
    ),
}

MODEL_FORMAT_OPTIONS = [
    "Flux / Chroma (natural language)",
    "Z-Image / Lumina-2 (LLM text encoder)",
    "HiDream (hybrid prose + descriptors)",
    "SDXL (tags + weights)",
    "SD 1.5 (tags + weights)",
    "Pony / Illustrious (booru tags + score)",
    "LTX Video (motion-focused prose)",
    "Hunyuan / Wan Video (cinematic motion prose)",
    "Universal Natural Language",
]

MODEL_FORMAT_INSTRUCTIONS = {
    "Flux / Chroma (natural language)": (
        "Format the output as flowing natural language in long descriptive sentences "
        "suitable for Flux or Chroma. No tag syntax. No parenthesis weights. "
        "No 'masterpiece' or quality boosters. Describe the scene as prose."
    ),
    "Z-Image / Lumina-2 (LLM text encoder)": (
        "Format the output as long, richly descriptive natural-language prose suitable "
        "for LLM-based text encoders (Z-Image, Lumina-2). Use complex sentences and "
        "vivid concrete detail. No tag syntax, no weights."
    ),
    "HiDream (hybrid prose + descriptors)": (
        "Format the output as flowing natural language with concrete photographic and "
        "material descriptors woven in (lighting type, lens feel, surface texture). "
        "Suitable for HiDream's multi-encoder pipeline. No weight syntax."
    ),
    "SDXL (tags + weights)": (
        "Format the output as a comma-separated list of descriptive tags for SDXL. "
        "Start with quality tags (masterpiece, best quality, highly detailed). "
        "Use parenthesis weight syntax for emphasis like (cinematic lighting:1.2). "
        "Order: subject, action, environment, lighting, camera, style, quality."
    ),
    "SD 1.5 (tags + weights)": (
        "Format the output as a comma-separated list of descriptive tags for SD 1.5. "
        "Lead with quality boosters (masterpiece, best quality, ultra-detailed). "
        "Use parenthesis weight syntax for emphasis. Keep tags compact and direct. "
        "Order: subject, action, environment, lighting, style, quality."
    ),
    "Pony / Illustrious (booru tags + score)": (
        "Format the output as comma-separated booru-style tags for Pony / Illustrious. "
        "Lead with score tags: score_9, score_8_up, score_7_up. Include an appropriate "
        "rating tag (rating_safe, rating_questionable, rating_explicit). Use underscores "
        "for multi-word tags (long_hair, blue_eyes). "
        "Order: score, rating, subject, character traits, action, setting, style."
    ),
    "LTX Video (motion-focused prose)": (
        "Format the output as a video shot description for LTX Video. "
        "Lead with a clear shot description, then describe subject motion explicitly "
        "(what moves and how), camera movement (pan, dolly, zoom, tracking), pacing, "
        "and atmosphere. Natural language sentences. No tag syntax."
    ),
    "Hunyuan / Wan Video (cinematic motion prose)": (
        "Format the output as a cinematic video description with detailed motion. "
        "Cover camera angle, camera movement, subject action, environmental motion, "
        "atmosphere, and pacing. Natural language sentences. No tag syntax."
    ),
    "Universal Natural Language": (
        "Format the output as a flowing natural language paragraph describing the scene "
        "in concrete visual detail. No tags, no weights."
    ),
}

AESTHETIC_OPTIONS = [
    "None (no aesthetic injection)",
    "Photorealistic", "Cinematic Film", "Anime / Manga", "Studio Ghibli",
    "Pixar / 3D Animation", "Comic Book / Graphic Novel", "Concept Art",
    "Oil Painting", "Watercolor", "Pencil Sketch",
    "Cyberpunk", "Steampunk", "Fantasy", "Sci-Fi", "Horror / Dark",
    "Vintage / Retro Film", "Film Noir", "Glamour / Editorial",
    "Minimalist", "Surreal / Dreamy", "3D Render / CGI",
]

AESTHETIC_DESCRIPTORS = {
    "Photorealistic": (
        "Visual style: photorealistic, sharp focus, accurate textures, lifelike skin and "
        "materials, realistic lighting and shadows, shallow depth of field, 8k camera detail."
    ),
    "Cinematic Film": (
        "Visual style: cinematic film aesthetic, dramatic lighting with strong key and rim, "
        "filmic color grade, anamorphic framing, atmospheric haze, shallow depth of field."
    ),
    "Anime / Manga": (
        "Visual style: anime/manga aesthetic, cel-shaded coloring, stylized features, "
        "expressive large eyes, dynamic poses, vibrant saturated color palette, clean lineart."
    ),
    "Studio Ghibli": (
        "Visual style: Studio Ghibli aesthetic, hand-painted watercolor backgrounds, "
        "soft natural lighting, warm pastoral atmosphere, gentle character designs, "
        "nostalgic and serene mood."
    ),
    "Pixar / 3D Animation": (
        "Visual style: 3D animation aesthetic in the spirit of Pixar/Disney, expressive "
        "stylized character proportions, polished CG surfaces, warm cinematic lighting, "
        "vibrant family-friendly color palette."
    ),
    "Comic Book / Graphic Novel": (
        "Visual style: comic book aesthetic, bold ink linework, halftone or hatching shading, "
        "dynamic poses, exaggerated proportions, saturated panel colors."
    ),
    "Concept Art": (
        "Visual style: digital concept art, painterly brushwork, atmospheric perspective, "
        "value-driven composition, loose suggestion of detail over full rendering, "
        "professional production-art feel."
    ),
    "Oil Painting": (
        "Visual style: oil painting aesthetic, visible impasto brushwork, rich color depth, "
        "painterly textures, classical composition, warm gallery lighting."
    ),
    "Watercolor": (
        "Visual style: watercolor painting aesthetic, soft translucent washes, paper texture "
        "visible, flowing pigment bleeds, gentle edges, limited palette."
    ),
    "Pencil Sketch": (
        "Visual style: pencil sketch aesthetic, graphite linework, crosshatching shadows, "
        "monochrome or restrained color accents, loose unfinished sketchbook feel."
    ),
    "Cyberpunk": (
        "Visual style: cyberpunk aesthetic, neon signage, rain-slicked streets, holographic "
        "interfaces, cybernetic implants, dystopian high-tech low-life atmosphere, "
        "magenta-cyan color palette, deep shadows with glowing accents."
    ),
    "Steampunk": (
        "Visual style: steampunk aesthetic, brass and copper machinery, Victorian-era styling, "
        "exposed gears and clockwork, gas-lamp lighting, sepia and bronze color palette, "
        "steam and soot atmosphere."
    ),
    "Fantasy": (
        "Visual style: high fantasy aesthetic, medieval or magical setting, ethereal lighting, "
        "ornate detail, mythological elements, lush detailed environments, painterly atmosphere."
    ),
    "Sci-Fi": (
        "Visual style: science fiction aesthetic, advanced technology, sleek futuristic surfaces, "
        "panel-screen lighting, industrial design, blue and white accent palette."
    ),
    "Horror / Dark": (
        "Visual style: horror aesthetic, low-key dramatic lighting, deep shadows, "
        "unsettling atmosphere, desaturated muted palette with occasional blood-red accents, "
        "dread-filled mood."
    ),
    "Vintage / Retro Film": (
        "Visual style: vintage film aesthetic, 35mm grain, faded warm color cast, soft contrast, "
        "period-appropriate styling, light leaks and lens artifacts."
    ),
    "Film Noir": (
        "Visual style: film noir aesthetic, high-contrast black-and-white or near-monochrome, "
        "dramatic chiaroscuro lighting, venetian-blind shadows, urban night atmosphere, "
        "cigarette smoke and rain."
    ),
    "Glamour / Editorial": (
        "Visual style: high-fashion editorial aesthetic, polished beauty lighting, "
        "magazine-shoot composition, soft skin rendering, dramatic backdrop, "
        "professional studio styling."
    ),
    "Minimalist": (
        "Visual style: minimalist aesthetic, clean composition, generous negative space, "
        "limited color palette, simple geometric forms, calm uncluttered framing."
    ),
    "Surreal / Dreamy": (
        "Visual style: surreal dreamlike aesthetic, soft hazy lighting, impossible compositions, "
        "dream-logic juxtapositions, ethereal color shifts, painterly atmosphere."
    ),
    "3D Render / CGI": (
        "Visual style: 3D render aesthetic, clean CGI surfaces, ray-traced lighting, "
        "sharp reflections, subsurface scattering, Octane/Blender render quality."
    ),
}

PROMPT_CLOSER = (
    "Output ONLY the final prompt as a single block of text. "
    "Do NOT show thinking, reasoning, analysis, drafts, or markdown sections. "
    "Do NOT write 'Thinking Process', 'Analysis', 'Drafting', 'Step-by-Step', "
    "or any section headers. Start your response immediately with the prompt text itself."
)


def build_layered_system_prompt(purpose, model_format, aesthetic,
                                 extra_instructions="", append_no_think=False):
    parts = [PURPOSE_FRAMING[purpose]]
    if aesthetic in AESTHETIC_DESCRIPTORS:
        parts.append(AESTHETIC_DESCRIPTORS[aesthetic])
    parts.append(MODEL_FORMAT_INSTRUCTIONS[model_format])
    if extra_instructions.strip():
        parts.append(extra_instructions.strip())
    parts.append(PROMPT_CLOSER)
    text = " ".join(parts)
    if append_no_think:
        text = text.rstrip() + " /no_think"
    return text


# =========================================================================
# Node 1: Curated Qwen3.5-4B enhancer
# =========================================================================

class RebelsPromptEnhancer:
    def __init__(self):
        pass

    _cache = {}

    PRECISION_OPTIONS = ["Efficiency (UD-IQ2)", "Quality (UD-Q8)"]
    PRECISION_SEARCH = {
        "Efficiency (UD-IQ2)": ["qwen3.5-4b", "ud-iq2"],
        "Quality (UD-Q8)":     ["qwen3.5-4b", "ud-q8"],
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_prompt": ("STRING", {"multiline": True}),
                "purpose": (PURPOSE_OPTIONS,),
                "model_format": (MODEL_FORMAT_OPTIONS,),
                "aesthetic": (AESTHETIC_OPTIONS,),
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
    def IS_CHANGED(cls, raw_prompt, purpose, model_format, aesthetic, precision, seed, lock_in):
        if lock_in:
            return f"LOCKED|{raw_prompt}|{purpose}|{model_format}|{aesthetic}|{precision}"
        return float("nan")

    def _find_files(self, node_dir, terms):
        terms = [t.lower() for t in terms]
        return [
            f for f in os.listdir(node_dir)
            if f.lower().endswith(".gguf") and all(t in f.lower() for t in terms)
        ]

    def enhance(self, raw_prompt, purpose, model_format, aesthetic, precision, seed, lock_in):
        cache_key = (raw_prompt, purpose, model_format, aesthetic, precision)

        if lock_in and cache_key in self._cache:
            c = self._cache[cache_key]
            thought = (
                f"=== 🔒 LOCKED — Returning Cached Output ===\n"
                f"No model load. No VRAM used. Seed ignored.\n\n"
                f"=== Original Generation Info ===\n{c['meta']}\n"
                f"=== Cached System Prompt ===\n{c['sys_prompt']}\n\n"
                f"=== Cached Raw Output ===\n{c['raw_output']}\n\n"
                f"=== Cached Final Prompt ===\n{c['final_prompt']}"
            )
            return (c['final_prompt'], thought)

        node_dir = os.path.dirname(os.path.abspath(__file__))
        terms = self.PRECISION_SEARCH[precision]
        files = self._find_files(node_dir, terms)
        if not files:
            raise FileNotFoundError(
                f"No .gguf in {node_dir} matching {terms}.\n"
                f"Expected a filename containing: {' AND '.join(terms)}"
            )
        files.sort(key=lambda x: (len(x), x))
        model_file = files[0]
        model_path = os.path.join(node_dir, model_file)
        n_ctx = 4096

        sys_prompt = build_layered_system_prompt(
            purpose, model_format, aesthetic, append_no_think=True
        )

        llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False,
                    n_ctx=n_ctx, seed=seed)
        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": raw_prompt},
                ],
                max_tokens=600, temperature=0.7, top_p=0.9, repeat_penalty=1.15,
                stop=VERBOSE_STOPS,
            )
            raw_output = output["choices"][0]["message"]["content"].strip()
        finally:
            _free_llm(llm)

        final_prompt = _clean_output(raw_output, raw_prompt)
        if not final_prompt.strip() or len(final_prompt) < 30:
            final_prompt = raw_output

        meta_block = (
            f"File:         {model_file}\n"
            f"Precision:    {precision}\n"
            f"Purpose:      {purpose}\n"
            f"Model Format: {model_format}\n"
            f"Aesthetic:    {aesthetic}\n"
            f"Context:      {n_ctx}\n"
            f"Seed:         {seed}\n"
            f"Raw chars:    {len(raw_output)}\n"
            f"Clean chars:  {len(final_prompt)}\n"
            f"Stripped:     {len(raw_output) - len(final_prompt)} chars\n"
        )
        thought = (
            f"=== 🔄 LIVE — Fresh Generation ===\n\n"
            f"=== Run Info ===\n{meta_block}\n"
            f"=== Assembled System Prompt ===\n{sys_prompt}\n\n"
            f"=== Raw Model Output ===\n{raw_output}\n\n"
            f"=== Final Prompt ===\n{final_prompt}"
        )
        self._cache[cache_key] = {
            "final_prompt": final_prompt, "raw_output": raw_output,
            "meta": meta_block, "sys_prompt": sys_prompt,
        }
        return (final_prompt, thought)


# =========================================================================
# Node 2: Custom GGUF enhancer
# =========================================================================

class RebelsPromptEnhancerCustom:
    def __init__(self):
        pass

    _cache = {}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_prompt": ("STRING", {"multiline": True}),
                "model_file": (_list_ggufs(),),
                "purpose": (PURPOSE_OPTIONS,),
                "model_format": (MODEL_FORMAT_OPTIONS,),
                "aesthetic": (AESTHETIC_OPTIONS,),
                "extra_instructions": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Optional extra instructions appended to the system prompt.",
                }),
                "system_prompt_override": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "If non-empty, REPLACES the entire layered system prompt.",
                }),
                "append_no_think": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Append /no_think",
                    "label_off": "Don't append",
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 800, "min": 50, "max": 4096, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
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
    def IS_CHANGED(cls, raw_prompt, model_file, purpose, model_format, aesthetic,
                   extra_instructions, system_prompt_override, append_no_think,
                   n_gpu_layers, n_ctx, max_tokens, temperature, top_p,
                   repeat_penalty, seed, lock_in):
        if lock_in:
            return (
                f"LOCKED|{raw_prompt}|{model_file}|{purpose}|{model_format}|{aesthetic}|"
                f"{extra_instructions}|{system_prompt_override}|{append_no_think}|"
                f"{temperature}|{top_p}|{repeat_penalty}"
            )
        return float("nan")

    def _build_sys_prompt(self, purpose, model_format, aesthetic,
                          extra_instructions, override, append_no_think):
        if override.strip():
            base = override.strip()
            if append_no_think:
                base = base.rstrip() + " /no_think"
            return base
        return build_layered_system_prompt(
            purpose, model_format, aesthetic,
            extra_instructions=extra_instructions,
            append_no_think=append_no_think,
        )

    def enhance(self, raw_prompt, model_file, purpose, model_format, aesthetic,
                extra_instructions, system_prompt_override, append_no_think,
                n_gpu_layers, n_ctx, max_tokens, temperature, top_p,
                repeat_penalty, seed, lock_in):
        cache_key = (
            raw_prompt, model_file, purpose, model_format, aesthetic,
            extra_instructions, system_prompt_override, append_no_think,
            temperature, top_p, repeat_penalty,
        )

        if lock_in and cache_key in self._cache:
            c = self._cache[cache_key]
            thought = (
                f"=== 🔒 LOCKED — Returning Cached Output ===\n"
                f"No model load. No VRAM used. Seed ignored.\n\n"
                f"=== Original Generation Info ===\n{c['meta']}\n"
                f"=== Cached System Prompt ===\n{c['sys_prompt']}\n\n"
                f"=== Cached Raw Output ===\n{c['raw_output']}\n\n"
                f"=== Cached Final Prompt ===\n{c['final_prompt']}"
            )
            return (c['final_prompt'], thought)

        if model_file == "NO_GGUF_FILES_IN_FOLDER":
            raise FileNotFoundError("Drop a .gguf in the node folder and restart ComfyUI.")
        node_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(node_dir, model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Selected file not found: {model_path}. Restart ComfyUI to refresh.")

        sys_prompt = self._build_sys_prompt(
            purpose, model_format, aesthetic,
            extra_instructions, system_prompt_override, append_no_think,
        )

        llm = Llama(model_path=model_path, n_gpu_layers=n_gpu_layers, verbose=False,
                    n_ctx=n_ctx, seed=seed)
        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": raw_prompt},
                ],
                max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=VERBOSE_STOPS,
            )
            raw_output = output["choices"][0]["message"]["content"].strip()
        finally:
            _free_llm(llm)

        final_prompt = _clean_output(raw_output, raw_prompt)
        if not final_prompt.strip() or len(final_prompt) < 30:
            final_prompt = raw_output

        gpu_label = "all" if n_gpu_layers < 0 else ("CPU only" if n_gpu_layers == 0 else f"{n_gpu_layers} layers")
        override_status = "ACTIVE (custom override used)" if system_prompt_override.strip() else "inactive"
        meta_block = (
            f"File:           {model_file}\n"
            f"Purpose:        {purpose}\n"
            f"Model Format:   {model_format}\n"
            f"Aesthetic:      {aesthetic}\n"
            f"Override:       {override_status}\n"
            f"GPU layers:     {gpu_label}\n"
            f"Context:        {n_ctx}\n"
            f"Max tokens:     {max_tokens}\n"
            f"Temperature:    {temperature}\n"
            f"top_p:          {top_p}\n"
            f"repeat_penalty: {repeat_penalty}\n"
            f"/no_think:      {'yes' if append_no_think else 'no'}\n"
            f"Seed:           {seed}\n"
            f"Raw chars:      {len(raw_output)}\n"
            f"Clean chars:    {len(final_prompt)}\n"
            f"Stripped:       {len(raw_output) - len(final_prompt)} chars\n"
        )
        thought = (
            f"=== 🔄 LIVE — Fresh Generation (Custom) ===\n\n"
            f"=== Run Settings ===\n{meta_block}\n"
            f"=== Assembled System Prompt ===\n{sys_prompt}\n\n"
            f"=== Raw Model Output ===\n{raw_output}\n\n"
            f"=== Final Prompt ===\n{final_prompt}"
        )
        self._cache[cache_key] = {
            "final_prompt": final_prompt, "raw_output": raw_output,
            "meta": meta_block, "sys_prompt": sys_prompt,
        }
        return (final_prompt, thought)


# =========================================================================
# Node 3: Image-to-Prompt vision node
# =========================================================================

class RebelsImageToPrompt:
    def __init__(self):
        pass

    _cache = {}

    VISION_TASK_OPTIONS = [
        "Caption (plain description)",
        "Caption + Format (apply model_format below)",
        "SD/Booru Tags",
        "Pose & Anatomy Focus",
        "Custom Instruction",
    ]

    VISION_TASK_INSTRUCTIONS = {
        "Caption (plain description)": (
            "Describe this image in one detailed paragraph covering subject, composition, "
            "lighting, colors, mood, and notable details. Output only the description."
        ),
        "SD/Booru Tags": (
            "Generate a comma-separated list of descriptive tags for this image. "
            "Include subject, action, setting, lighting, mood, and style tags. "
            "Output only the tags, comma-separated."
        ),
        "Pose & Anatomy Focus": (
            "Describe the subject's pose, body position, expression, framing, and what's "
            "visible in detail. Be precise about positioning. Output only the description."
        ),
    }

    @classmethod
    def INPUT_TYPES(s):
        handler_options = ["Auto-detect"] + list(_HANDLER_CLASSES.keys())
        if not _HANDLER_CLASSES:
            handler_options = ["NO_VISION_HANDLERS_AVAILABLE"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (_list_ggufs(),),
                "mmproj_file": (_list_mmproj(),),
                "chat_handler": (handler_options,),
                "vision_task": (s.VISION_TASK_OPTIONS,),
                "model_format": (MODEL_FORMAT_OPTIONS,),
                "aesthetic": (AESTHETIC_OPTIONS,),
                "custom_instruction": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Used when vision_task is 'Custom Instruction'.",
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 400, "min": 50, "max": 2048, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lock_in": ("BOOLEAN", {
                    "default": False,
                    "label_on": "🔒 LOCKED (cached)",
                    "label_off": "🔄 LIVE (analyzing)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "thought_process")
    FUNCTION = "describe"
    CATEGORY = "Rebel AI"

    @classmethod
    def IS_CHANGED(cls, image, model_file, mmproj_file, chat_handler, vision_task,
                   model_format, aesthetic, custom_instruction, n_gpu_layers,
                   n_ctx, max_tokens, temperature, seed, lock_in):
        if lock_in:
            return (f"LOCKED|{model_file}|{mmproj_file}|{chat_handler}|{vision_task}|"
                    f"{model_format}|{aesthetic}|{custom_instruction}")
        try:
            img_hash = hash(image.detach().cpu().numpy().tobytes())
        except Exception:
            img_hash = "noimg"
        return f"LIVE|{img_hash}|{seed}|{float('nan')}"

    def _resolve_handler(self, chat_handler_choice, model_file):
        if chat_handler_choice == "Auto-detect":
            mf = model_file.lower()
            if "moondream" in mf and "Moondream" in _HANDLER_CLASSES:
                return _HANDLER_CLASSES["Moondream"], "Moondream"
            if "qwen" in mf and "Qwen2.5-VL" in _HANDLER_CLASSES:
                return _HANDLER_CLASSES["Qwen2.5-VL"], "Qwen2.5-VL"
            if "minicpm" in mf and "MiniCPM-V 2.6" in _HANDLER_CLASSES:
                return _HANDLER_CLASSES["MiniCPM-V 2.6"], "MiniCPM-V 2.6"
            if "nano" in mf and "llava" in mf and "NanoLLaVA" in _HANDLER_CLASSES:
                return _HANDLER_CLASSES["NanoLLaVA"], "NanoLLaVA"
            if "llava" in mf:
                if "LLaVA 1.6" in _HANDLER_CLASSES:
                    return _HANDLER_CLASSES["LLaVA 1.6"], "LLaVA 1.6"
                if "LLaVA 1.5" in _HANDLER_CLASSES:
                    return _HANDLER_CLASSES["LLaVA 1.5"], "LLaVA 1.5"
            if "LLaVA 1.5" in _HANDLER_CLASSES:
                return _HANDLER_CLASSES["LLaVA 1.5"], "LLaVA 1.5 (fallback)"
            raise RuntimeError("No vision chat handler available.")
        if chat_handler_choice not in _HANDLER_CLASSES:
            raise RuntimeError(
                f"Chat handler '{chat_handler_choice}' isn't available. "
                f"Available: {list(_HANDLER_CLASSES.keys())}"
            )
        return _HANDLER_CLASSES[chat_handler_choice], chat_handler_choice

    def _tensor_to_data_uri(self, image_tensor):
        if image_tensor.dim() == 4:
            img = image_tensor[0]
        else:
            img = image_tensor
        img_np = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(img_np)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"

    def _build_instruction(self, vision_task, model_format, aesthetic, custom_instruction):
        if vision_task == "Custom Instruction":
            return custom_instruction.strip() or "Describe this image."
        if vision_task == "Caption + Format (apply model_format below)":
            base = (
                "Describe this image faithfully, then format the description according "
                "to the rules below."
            )
            base += " " + MODEL_FORMAT_INSTRUCTIONS[model_format]
            if aesthetic in AESTHETIC_DESCRIPTORS:
                base += " " + AESTHETIC_DESCRIPTORS[aesthetic]
            base += " " + PROMPT_CLOSER
            return base
        base = self.VISION_TASK_INSTRUCTIONS[vision_task]
        if aesthetic in AESTHETIC_DESCRIPTORS:
            base += " " + AESTHETIC_DESCRIPTORS[aesthetic]
        return base

    def describe(self, image, model_file, mmproj_file, chat_handler, vision_task,
                 model_format, aesthetic, custom_instruction, n_gpu_layers,
                 n_ctx, max_tokens, temperature, seed, lock_in):
        cache_key = (model_file, mmproj_file, chat_handler, vision_task,
                     model_format, aesthetic, custom_instruction, temperature)

        if lock_in and cache_key in self._cache:
            c = self._cache[cache_key]
            thought = (
                f"=== 🔒 LOCKED — Returning Cached Caption ===\n"
                f"No model load. New image ignored.\n\n"
                f"=== Original Run Info ===\n{c['meta']}\n"
                f"=== Cached Instruction ===\n{c['instruction']}\n\n"
                f"=== Cached Raw Output ===\n{c['raw_output']}\n\n"
                f"=== Cached Image Prompt ===\n{c['final_prompt']}"
            )
            return (c['final_prompt'], thought)

        if model_file == "NO_GGUF_FILES_IN_FOLDER":
            raise FileNotFoundError("Drop a vision-capable .gguf in the node folder and restart ComfyUI.")
        if mmproj_file == "NO_MMPROJ_FILE_FOUND":
            raise FileNotFoundError("No mmproj file found. Vision models need a paired mmproj-*.gguf.")
        if not _HANDLER_CLASSES:
            raise RuntimeError("No vision chat handlers available. Update llama-cpp-python.")

        node_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(node_dir, model_file)
        mmproj_path = os.path.join(node_dir, mmproj_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.isfile(mmproj_path):
            raise FileNotFoundError(f"mmproj not found: {mmproj_path}")

        handler_cls, handler_label = self._resolve_handler(chat_handler, model_file)
        instruction = self._build_instruction(vision_task, model_format, aesthetic, custom_instruction)
        img_uri = self._tensor_to_data_uri(image)

        chat_handler_instance = handler_cls(clip_model_path=mmproj_path, verbose=False)
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler_instance,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_ctx=n_ctx,
            seed=seed,
            logits_all=True,
        )
        try:
            output = llm.create_chat_completion(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_uri}},
                            {"type": "text", "text": instruction},
                        ],
                    },
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
            )
            raw_output = output["choices"][0]["message"]["content"].strip()
        finally:
            _free_llm(llm)
            try:
                del chat_handler_instance
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_prompt = _clean_output(raw_output)
        if not final_prompt.strip() or len(final_prompt) < 20:
            final_prompt = raw_output

        gpu_label = "all" if n_gpu_layers < 0 else ("CPU only" if n_gpu_layers == 0 else f"{n_gpu_layers} layers")
        meta_block = (
            f"Model file:    {model_file}\n"
            f"mmproj file:   {mmproj_file}\n"
            f"Chat handler:  {handler_label}\n"
            f"Vision task:   {vision_task}\n"
            f"Model Format:  {model_format}\n"
            f"Aesthetic:     {aesthetic}\n"
            f"GPU layers:    {gpu_label}\n"
            f"Context:       {n_ctx}\n"
            f"Max tokens:    {max_tokens}\n"
            f"Temperature:   {temperature}\n"
            f"Seed:          {seed}\n"
            f"Raw chars:     {len(raw_output)}\n"
            f"Clean chars:   {len(final_prompt)}\n"
        )
        thought = (
            f"=== 🔄 LIVE — Image Analysis ===\n\n"
            f"=== Run Settings ===\n{meta_block}\n"
            f"=== Instruction Sent ===\n{instruction}\n\n"
            f"=== Raw Model Output ===\n{raw_output}\n\n"
            f"=== Image Prompt (downstream) ===\n{final_prompt}"
        )
        self._cache[cache_key] = {
            "final_prompt": final_prompt, "raw_output": raw_output,
            "meta": meta_block, "instruction": instruction,
        }
        return (final_prompt, thought)


# =========================================================================
# Node 4: General LLM Console
# =========================================================================

class RebelsLLMConsole:
    def __init__(self):
        pass

    _cache = {}

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, knowledgeable assistant. Answer directly and concisely "
        "without unnecessary preamble. If asked a technical question, be precise. "
        "If asked an open-ended question, be thoughtful."
    )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "question": ("STRING", {
                    "multiline": True,
                    "placeholder": "Ask anything…",
                }),
                "model_file": (_list_ggufs(),),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": s.DEFAULT_SYSTEM_PROMPT,
                }),
                "append_no_think": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Append /no_think",
                    "label_off": "Don't append",
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 800, "min": 50, "max": 4096, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.15, "min": 1.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "lock_in": ("BOOLEAN", {
                    "default": False,
                    "label_on": "🔒 LOCKED (cached)",
                    "label_off": "🔄 LIVE (asking)",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "thought_process")
    FUNCTION = "chat"
    CATEGORY = "Rebel AI"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, question, model_file, system_prompt, append_no_think,
                   n_gpu_layers, n_ctx, max_tokens, temperature, top_p,
                   repeat_penalty, seed, lock_in):
        if lock_in:
            return (f"LOCKED|{question}|{model_file}|{system_prompt}|"
                    f"{append_no_think}|{temperature}|{top_p}|{repeat_penalty}")
        return float("nan")

    def chat(self, question, model_file, system_prompt, append_no_think,
             n_gpu_layers, n_ctx, max_tokens, temperature, top_p,
             repeat_penalty, seed, lock_in):

        cache_key = (question, model_file, system_prompt, append_no_think,
                     temperature, top_p, repeat_penalty)

        if lock_in and cache_key in self._cache:
            c = self._cache[cache_key]
            response = c["response"]
            thought = (
                f"=== 🔒 LOCKED — Returning Cached Response ===\n"
                f"No model load.\n\n"
                f"=== Original Run Info ===\n{c['meta']}\n"
                f"=== System Prompt ===\n{c['sys_prompt']}\n\n"
                f"=== Cached Response ===\n{response}"
            )
            return {"ui": {"text": [response]}, "result": (response, thought)}

        if model_file == "NO_GGUF_FILES_IN_FOLDER":
            raise FileNotFoundError("Drop a .gguf in the node folder and restart ComfyUI.")
        node_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(node_dir, model_file)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Selected file not found: {model_path}. Restart ComfyUI to refresh.")

        sys_prompt = system_prompt.strip() or self.DEFAULT_SYSTEM_PROMPT
        if append_no_think:
            sys_prompt = sys_prompt.rstrip() + " /no_think"

        llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_ctx=n_ctx,
            seed=seed,
        )
        try:
            output = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": question},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop=["\n\nUser:", "\n\nHuman:", "</think>", "</thinking>"],
            )
            raw_output = output["choices"][0]["message"]["content"].strip()
        finally:
            _free_llm(llm)

        response = _clean_chat_output(raw_output)
        if not response.strip():
            response = raw_output

        gpu_label = "all" if n_gpu_layers < 0 else ("CPU only" if n_gpu_layers == 0 else f"{n_gpu_layers} layers")
        meta_block = (
            f"File:           {model_file}\n"
            f"GPU layers:     {gpu_label}\n"
            f"Context:        {n_ctx}\n"
            f"Max tokens:     {max_tokens}\n"
            f"Temperature:    {temperature}\n"
            f"top_p:          {top_p}\n"
            f"repeat_penalty: {repeat_penalty}\n"
            f"/no_think:      {'yes' if append_no_think else 'no'}\n"
            f"Seed:           {seed}\n"
            f"Question chars: {len(question)}\n"
            f"Response chars: {len(response)}\n"
        )
        thought = (
            f"=== 🔄 LIVE — Fresh Response ===\n\n"
            f"=== Run Settings ===\n{meta_block}\n"
            f"=== System Prompt ===\n{sys_prompt}\n\n"
            f"=== Question ===\n{question}\n\n"
            f"=== Response ===\n{response}"
        )

        self._cache[cache_key] = {
            "response": response,
            "meta": meta_block,
            "sys_prompt": sys_prompt,
        }

        return {"ui": {"text": [response]}, "result": (response, thought)}


# =========================================================================
# Node 5: Locker
# =========================================================================

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
    "RebelsPromptEnhancerCustom": RebelsPromptEnhancerCustom,
    "RebelsImageToPrompt": RebelsImageToPrompt,
    "RebelsLLMConsole": RebelsLLMConsole,
    "RebelsPromptLocker": RebelsPromptLocker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsPromptEnhancer": "🚀 Rebels Prompt Enhancer",
    "RebelsPromptEnhancerCustom": "🧪 Rebels Prompt Enhancer (Custom GGUF)",
    "RebelsImageToPrompt": "👁️ Rebels Image to Prompt",
    "RebelsLLMConsole": "🧠 Rebels LLM Console",
    "RebelsPromptLocker": "🔒 Rebels Prompt Locker",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
