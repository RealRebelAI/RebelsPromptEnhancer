import base64
import gc
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from functools import wraps
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from llama_cpp import Llama


# ================================================================
# Paths and package state
# ================================================================

NODE_DIR = Path(__file__).resolve().parent
CACHE_DIR = NODE_DIR / ".cache"
CACHE_FILE = CACHE_DIR / "prompt_cache.json"
CACHE_LOCK = threading.Lock()
RUNTIME_DIR = NODE_DIR / ".runtime" / "llama_cpp_portable"
CUDA_RUNTIME_DIR = NODE_DIR / ".runtime" / "llama_cpp_portable_cuda12"
RUNTIME_LOCK = threading.Lock()
NATIVE_BACKEND_STATE_FILE = NODE_DIR / ".runtime" / "native_backend_disabled.json"

# Keep-at-most-one hot model for Fast Iteration mode.
_HOT_MODEL = {"key": None, "llm": None}

# Portable llama-server state. Keeping one server alive lets quality retries reuse
# the already-loaded GGUF. Fast Iteration can also reuse it across node executions.
_PORTABLE_SERVER_LOCK = threading.RLock()
_PORTABLE_SERVER = {
    "key": None, "proc": None, "base_url": None, "log_handle": None,
    "log_path": None, "runtime": None,
}


def _portable_runtime_already_present():
    """Lightweight startup check that does not import/run the portable backend."""
    for root in (CUDA_RUNTIME_DIR, RUNTIME_DIR):
        if not root.exists():
            continue
        try:
            if any(p.is_file() and p.name.lower() == "llama-completion.exe" for p in root.rglob("*.exe")):
                return True
        except Exception:
            pass
    return False


def _persist_native_backend_disabled(reason="STATUS_ILLEGAL_INSTRUCTION"):
    try:
        NATIVE_BACKEND_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = NATIVE_BACKEND_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({
                "disabled": True,
                "reason": str(reason),
                "portable_backend": True,
            }, indent=2),
            encoding="utf-8",
        )
        tmp.replace(NATIVE_BACKEND_STATE_FILE)
    except Exception:
        pass


def _load_native_backend_disabled():
    # Normal persistent state.
    try:
        if NATIVE_BACKEND_STATE_FILE.is_file():
            data = json.loads(NATIVE_BACKEND_STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("disabled") is True:
                return True
    except Exception:
        pass

    # Migration from V3.2.x: a portable runtime only existed after this node had
    # already encountered the native Windows illegal-instruction failure. Seed the
    # persistent state automatically so upgrading does not cause one more crash.
    if os.name == "nt" and _portable_runtime_already_present():
        _persist_native_backend_disabled("migrated-existing-portable-runtime")
        return True
    return False


# Once the native backend proves incompatible, the choice survives ComfyUI restarts.
_NATIVE_BACKEND_DISABLED = _load_native_backend_disabled()
if _NATIVE_BACKEND_DISABLED:
    print(
        "[RebelsPromptEnhancer] Portable llama.cpp compatibility mode is persistent for this install; "
        "skipping the incompatible in-process llama-cpp-python backend."
    )


# ================================================================
# Vision chat handlers — best-effort import
# ================================================================

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


# ================================================================
# Shared helpers
# ================================================================

VERBOSE_STOPS = [
    "\n\nUser:", "\n\nAssistant:", "\n\nHuman:",
    "</think>", "</thinking>",
    "Thinking Process:", "**Thinking Process",
    "## Thinking", "### Thinking",
    "Analysis:", "**Analysis",
]

REASONING_MARKERS = (
    "let me", "i'll ", "i will ", "i need", "i must", "i should",
    "the prompt is", "the user", "key elements", "brainstorm",
    "as per the rules", "according to the rules", "the rules say",
    "let's", "okay so", "first,", "second,", "third,",
    "thinking process", "analyze the request", "deconstruct",
    "step-by-step", "step by step", "drafting",
)

PREAMBLES = (
    "here's", "here is", "sure,", "sure!", "certainly,", "of course,",
    "okay,", "okay.", "alright,",
    "enhanced prompt:", "expanded prompt:", "prompt:",
    "output:", "answer:", "final prompt:", "final:", "example:",
)


def _strip_thinking_tags(text):
    for pat in (
        r"<think(?:ing)?>.*?</think(?:ing)?>",
        r"<\|thinking\|>.*?<\|/thinking\|>",
        r"\[THINK(?:ING)?\].*?\[/THINK(?:ING)?\]",
    ):
        text = re.sub(pat, "", text, flags=re.DOTALL | re.IGNORECASE)
    return text


def _strip_preambles(text):
    for _ in range(3):
        lowered = text.lower().lstrip()
        matched = False
        for preamble in PREAMBLES:
            if lowered.startswith(preamble):
                colon = text.find(":")
                newline = text.find("\n")
                if 0 < colon < 60:
                    text = text[colon + 1:].strip()
                elif 0 < newline < 80:
                    text = text[newline + 1:].strip()
                else:
                    text = text[len(preamble):].strip(" ,.:-")
                matched = True
                break
        if not matched:
            break
    return text


def _legacy_clean_output(text, original_input=""):
    """Fallback cleaning only. Structured JSON is preferred in V3."""
    text = _strip_thinking_tags((text or "").strip())
    text = _strip_preambles(text)

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    lower_all = text.lower()
    if len(paragraphs) > 1 and sum(m in lower_all for m in REASONING_MARKERS) >= 2:
        for paragraph in reversed(paragraphs):
            low = paragraph.lower().lstrip()
            if len(paragraph) >= 60 and not any(low.startswith(m) for m in REASONING_MARKERS):
                text = paragraph
                break

    if original_input:
        raw = original_input.strip()
        if raw and text.lower().startswith(raw.lower()) and len(text) > len(raw) + 20:
            text = text[len(raw):].lstrip(" ,.:-\"'\n")

    return text.strip().strip('"\'')


def _clean_chat_output(text):
    return _strip_thinking_tags((text or "").strip()).strip()


def _is_oom_error(exc):
    msg = str(exc).lower()
    needles = (
        "out of memory", "cuda error", "cuda_malloc", "cudamalloc",
        "failed to allocate", "insufficient memory", "not enough memory",
        "memory allocation", "ggml_cuda", "cuda host", "vram",
    )
    return any(n in msg for n in needles)


def _is_illegal_instruction_error(exc):
    msg = str(exc).lower()
    return (
        "0xc000001d" in msg
        or "-1073741795" in msg
        or "illegal instruction" in msg
    )


def _free_llm(llm):
    try:
        if hasattr(llm, "close"):
            llm.close()
    except Exception:
        pass
    try:
        del llm
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _release_hot_model():
    with CACHE_LOCK:
        llm = _HOT_MODEL.get("llm")
        _HOT_MODEL["llm"] = None
        _HOT_MODEL["key"] = None
    if llm is not None:
        _free_llm(llm)


def _model_key(model_path, n_gpu_layers, n_ctx):
    return (str(Path(model_path).resolve()), int(n_gpu_layers), int(n_ctx))


def _get_llm(model_path, n_gpu_layers, n_ctx, seed, load_mode):
    key = _model_key(model_path, n_gpu_layers, n_ctx)
    if load_mode == "Fast Iteration (keep model loaded)":
        with CACHE_LOCK:
            if _HOT_MODEL["key"] == key and _HOT_MODEL["llm"] is not None:
                return _HOT_MODEL["llm"], False
        _release_hot_model()
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            seed=seed,
            verbose=False,
        )
        with CACHE_LOCK:
            _HOT_MODEL["key"] = key
            _HOT_MODEL["llm"] = llm
        return llm, False

    llm = Llama(
        model_path=str(model_path),
        n_gpu_layers=n_gpu_layers,
        n_ctx=n_ctx,
        seed=seed,
        verbose=False,
    )
    return llm, True


def _list_ggufs(exclude_mmproj=True):
    try:
        files = []
        for path in NODE_DIR.iterdir():
            if not path.is_file() or path.suffix.lower() != ".gguf":
                continue
            if exclude_mmproj and "mmproj" in path.name.lower():
                continue
            files.append(path.name)
        return sorted(files) if files else ["NO_GGUF_FILES_IN_FOLDER"]
    except Exception:
        return ["NO_GGUF_FILES_IN_FOLDER"]


def _list_mmproj():
    try:
        files = sorted(
            p.name for p in NODE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() == ".gguf" and "mmproj" in p.name.lower()
        )
        return files if files else ["NO_MMPROJ_FILE_FOUND"]
    except Exception:
        return ["NO_MMPROJ_FILE_FOUND"]


def _relative_model_hint():
    return "custom_nodes/RebelsPromptEnhancer/"


def _load_disk_cache():
    try:
        if not CACHE_FILE.exists():
            return {}
        data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_disk_cache(cache):
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(CACHE_FILE)
    except Exception:
        pass


def _cache_hash(payload):
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_get(payload):
    key = _cache_hash(payload)
    with CACHE_LOCK:
        return _load_disk_cache().get(key)


def _cache_set(payload, value):
    key = _cache_hash(payload)
    with CACHE_LOCK:
        cache = _load_disk_cache()
        cache[key] = value
        # Bound persistent cache growth.
        if len(cache) > 300:
            for old_key in list(cache.keys())[: len(cache) - 300]:
                cache.pop(old_key, None)
        _save_disk_cache(cache)


def _detect_model_family(filename):
    f = (filename or "").lower()
    if "qwen3.5" in f or "qwen3" in f:
        return "Qwen3"
    if "qwen2.5" in f or "qwen2" in f:
        return "Qwen2"
    if "gemma" in f:
        return "Gemma"
    if "mistral" in f or "mixtral" in f:
        return "Mistral"
    if "phi" in f:
        return "Phi"
    if "llama" in f:
        return "Llama"
    return "Auto"


def _auto_no_think(filename, requested):
    if requested == "Force ON":
        return True
    if requested == "Force OFF":
        return False
    return _detect_model_family(filename) == "Qwen3"


# ================================================================
# Prompt system
# ================================================================

PURPOSE_OPTIONS = ["Image", "Video", "Edit (Inpainting/I2V)"]
DETAIL_OPTIONS = ["Preserve", "Expand", "Highly Detailed", "Creative"]
LENGTH_OPTIONS = ["Short", "Medium", "Long", "Max Detail"]
VARIANT_OPTIONS = ["1", "2", "3", "4"]
NEGATIVE_OPTIONS = ["Off", "Generate when useful"]
LOAD_MODE_OPTIONS = ["VRAM Saver (unload after run)", "Fast Iteration (keep model loaded)"]
NO_THINK_OPTIONS = ["Auto", "Force ON", "Force OFF"]

PURPOSE_FRAMING = {
    "Image": (
        "Rewrite for a single static image. Prioritize subject identity and attributes, pose/action, "
        "environment, composition, camera/lens, lighting, materials/textures, mood, and spatial relationships."
    ),
    "Video": (
        "Rewrite for a video shot. Prioritize temporal order, subject motion, camera movement, motion speed, "
        "environmental movement, continuity, lighting changes, pacing, and the final visible state."
    ),
    "Edit (Inpainting/I2V)": (
        "Rewrite the instruction as a precise description of the desired final result. Explicitly preserve all "
        "unchanged identity, scene, framing, lighting, geometry, and background details unless the user asks to alter them."
    ),
}

DETAIL_INSTRUCTIONS = {
    "Preserve": (
        "Make the minimum useful expansion. Preserve the user's wording and intent closely. Add only missing visual clarity."
    ),
    "Expand": (
        "Expand the prompt with concrete visual details while staying tightly faithful to the user's stated concept."
    ),
    "Highly Detailed": (
        "Add rich, concrete detail across subject, scene, lighting, camera, materials, atmosphere, and composition. "
        "Do not invent changes that contradict the user's explicit details."
    ),
    "Creative": (
        "Add tasteful creative visual choices where the user left details unspecified, while never changing explicitly "
        "stated facts, identities, text, counts, colors, poses, or required composition."
    ),
}

LENGTH_INSTRUCTIONS = {
    "Short": "Target roughly 40-90 words or an equivalently compact tag list.",
    "Medium": "Target roughly 90-180 words or an equivalently detailed tag list.",
    "Long": (
        "Target roughly 180-320 useful words or an equivalently rich tag list. Develop the scene fully and do not stop early; "
        "cover subject, action/pose, environment, composition, camera/lens, lighting, materials/textures, color, atmosphere, and spatial relationships where relevant."
    ),
    "Max Detail": (
        "Target roughly 300-500 useful words for natural-language formats, using as much concrete scene information as the concept supports. "
        "Fully develop subject, action/pose, environment, foreground/background, composition, camera/lens, lighting, materials/textures, color palette, atmosphere, and spatial relationships. "
        "Do not pad with filler, repeat phrases, or duplicate quality buzzwords."
    ),
}

PRESERVE_RULE = (
    "STRICT PRESERVATION RULE: Never remove, replace, contradict, or reinterpret concrete user-provided details. "
    "Preserve names, identities, subject count, body/face traits, clothing, colors, exact written text, camera angle, "
    "pose, environment, time of day, layout, and explicit negatives/constraints. Only elaborate unspecified details. "
    "Preservation is NOT copying: you are expected to add useful detail wherever the user left details unspecified."
)

MODEL_FORMAT_OPTIONS = [
    "Qwen Image / Qwen Image Edit",
    "Flux / Chroma (natural language)",
    "Krea (photographic natural language)",
    "Z-Image / Lumina-2 (LLM text encoder)",
    "HiDream (hybrid prose + descriptors)",
    "SDXL (tags + weights)",
    "SD 1.5 (tags + weights)",
    "Pony / Illustrious (booru tags + score)",
    "Wan / Hunyuan Video (cinematic motion prose)",
    "LTX Video (motion-focused prose)",
    "FastH3 / MiniMax Video",
    "Universal Natural Language",
]

MODEL_FORMAT_INSTRUCTIONS = {
    "Qwen Image / Qwen Image Edit": (
        "Use dense natural-language visual prose optimized for an LLM-style image text encoder. State the main subject and "
        "composition early, then layer concrete environment, lighting, camera/lens, texture/material, color, mood, and spatial details. "
        "For edits, clearly describe what changes and what must remain unchanged. Avoid tag soup and weighting syntax."
    ),
    "Flux / Chroma (natural language)": (
        "Use flowing natural-language sentences. No tag syntax or parenthesis weights. Prefer concrete visual descriptions over quality buzzwords."
    ),
    "Krea (photographic natural language)": (
        "Use concise photographic prose emphasizing subject, camera framing, lens feel, natural materials, realistic lighting, texture, and composition."
    ),
    "Z-Image / Lumina-2 (LLM text encoder)": (
        "Use long, richly descriptive natural-language prose with vivid concrete details. No tag syntax or weighting."
    ),
    "HiDream (hybrid prose + descriptors)": (
        "Use natural language with photographic/material descriptors woven in: lighting type, lens feel, surface texture, composition, and atmosphere."
    ),
    "SDXL (tags + weights)": (
        "Use a comma-separated SDXL tag list. Order subject, action, environment, lighting, camera, style, quality. Use weighting only sparingly."
    ),
    "SD 1.5 (tags + weights)": (
        "Use a compact comma-separated SD1.5 tag list with selective weighting. Keep concepts direct and avoid prose."
    ),
    "Pony / Illustrious (booru tags + score)": (
        "Use booru-style tags, underscore multi-word tags, and appropriate score/rating tags. Preserve literal user details exactly."
    ),
    "Wan / Hunyuan Video (cinematic motion prose)": (
        "Use cinematic motion prose: shot size/angle, camera movement, subject action, environmental motion, timing, atmosphere, and continuity."
    ),
    "LTX Video (motion-focused prose)": (
        "Lead with the shot description, then explicit subject motion, camera motion, timing/pacing, environment motion, and atmosphere."
    ),
    "FastH3 / MiniMax Video": (
        "Write a temporally ordered cinematic video prompt. Be explicit about who moves, how limbs/objects move, camera movement, speed, "
        "continuity between moments, background motion, and what remains stable. Avoid contradictory simultaneous actions."
    ),
    "Universal Natural Language": (
        "Use a clear natural-language paragraph with concrete visual detail. No tag syntax or weights."
    ),
}

AESTHETIC_OPTIONS = [
    "None (no aesthetic injection)",
    "Photorealistic", "Cinematic Film", "Anime / Manga", "Studio Ghibli",
    "Pixar / 3D Animation", "Comic Book / Graphic Novel", "Concept Art",
    "Oil Painting", "Watercolor", "Pencil Sketch", "Cyberpunk", "Steampunk",
    "Fantasy", "Sci-Fi", "Horror / Dark", "Vintage / Retro Film", "Film Noir",
    "Glamour / Editorial", "Minimalist", "Surreal / Dreamy", "3D Render / CGI",
]

AESTHETIC_DESCRIPTORS = {
    "Photorealistic": "Photorealistic visual treatment with believable materials, skin/surfaces, natural optics, realistic light falloff and shadows.",
    "Cinematic Film": "Cinematic film aesthetic with motivated key/rim lighting, filmic contrast, atmospheric depth and deliberate composition.",
    "Anime / Manga": "Anime/manga aesthetic with clean stylized linework, cel-shaded forms, expressive design and intentional color blocking.",
    "Studio Ghibli": "Hand-painted animation aesthetic with watercolor-like backgrounds, soft natural lighting, warm pastoral detail and gentle character design.",
    "Pixar / 3D Animation": "Pixar-inspired polished stylized 3D animation with expressive character design, appealing proportions, clean cinematic CG materials, soft global illumination, and warm feature-animation lighting.",
    "Comic Book / Graphic Novel": "Graphic-novel look with bold ink linework, hatching/halftone texture, dramatic composition and controlled color.",
    "Concept Art": "Professional digital concept-art treatment with strong value design, atmospheric perspective and selective painterly detail.",
    "Oil Painting": "Traditional oil-painting look with visible brushwork, rich pigment depth and painterly edges.",
    "Watercolor": "Watercolor look with translucent washes, paper texture, pigment variation and softened edges.",
    "Pencil Sketch": "Graphite sketch aesthetic with expressive linework, crosshatching and paper texture.",
    "Cyberpunk": "Cyberpunk aesthetic with dense futuristic infrastructure, neon-emissive accents, reflective surfaces and deep urban shadows.",
    "Steampunk": "Steampunk aesthetic with brass/copper machinery, Victorian styling, exposed mechanics, steam and warm industrial light.",
    "Fantasy": "High-fantasy visual language with ornate worldbuilding, magical atmosphere and richly detailed environments.",
    "Sci-Fi": "Science-fiction aesthetic with advanced technology, engineered surfaces, purposeful industrial design and futuristic lighting.",
    "Horror / Dark": "Dark horror aesthetic with low-key lighting, deep shadow, unsettling negative space and restrained ominous detail.",
    "Vintage / Retro Film": "Vintage film look with period styling, grain, softer contrast and analog lens/color characteristics.",
    "Film Noir": "Film-noir aesthetic with stark chiaroscuro, dramatic shadow geometry and moody urban atmosphere.",
    "Glamour / Editorial": "High-fashion editorial treatment with polished beauty lighting, controlled posing and magazine-style composition.",
    "Minimalist": "Minimalist composition with strong negative space, clean forms and restrained visual clutter.",
    "Surreal / Dreamy": "Surreal dreamlike aesthetic with soft atmospheric transitions, impossible but coherent juxtapositions and ethereal mood.",
    "3D Render / CGI": "High-end CGI treatment with physically plausible materials, ray-traced light behavior and crisp modeled detail.",
}

# Quality floors for post-generation validation. Long now matches the requested
# lower bound; Max Detail keeps a slightly forgiving floor to avoid filler while still
# strongly preferring the 300-500 word target.
LENGTH_MIN_CHARS = {"Short": 80, "Medium": 260, "Long": 900, "Max Detail": 1350}
LENGTH_MIN_WORDS = {"Short": 24, "Medium": 60, "Long": 180, "Max Detail": 260}

AESTHETIC_ANCHORS = {
    "Photorealistic": ("photoreal", "realistic"),
    "Cinematic Film": ("cinematic", "filmic"),
    "Anime / Manga": ("anime", "manga"),
    "Studio Ghibli": ("ghibli", "hand-painted animation", "hand painted animation"),
    "Pixar / 3D Animation": ("pixar", "3d animation", "3d animated", "feature-animation", "feature animation"),
    "Comic Book / Graphic Novel": ("comic", "graphic novel"),
    "Concept Art": ("concept art",),
    "Oil Painting": ("oil paint", "oil-paint"),
    "Watercolor": ("watercolor", "watercolour"),
    "Pencil Sketch": ("pencil", "graphite"),
    "Cyberpunk": ("cyberpunk",),
    "Steampunk": ("steampunk",),
    "Fantasy": ("fantasy",),
    "Sci-Fi": ("sci-fi", "science fiction", "science-fiction"),
    "Horror / Dark": ("horror", "dark aesthetic", "ominous"),
    "Vintage / Retro Film": ("vintage", "retro film", "analog film", "analogue film"),
    "Film Noir": ("film noir", "noir"),
    "Glamour / Editorial": ("glamour", "editorial"),
    "Minimalist": ("minimalist", "minimalism"),
    "Surreal / Dreamy": ("surreal", "dreamlike", "dreamy"),
    "3D Render / CGI": ("3d render", "cgi", "cg render"),
}


# ================================================================
# Curated models / download helper
# ================================================================

CURATED_MODELS = {
    # V2 compatibility alias. Keep this exact label so existing workflows load cleanly.
    "Efficiency (UD-IQ2)": {
        "match_groups": [["qwen3.5-4b", "ud-iq2_m"]],
        "preferred": "Qwen3.5-4B-UD-IQ2_M.gguf",
        # Pin the older revision that the original release used successfully.
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/b0bd07786f94b700012140fd9263b2de83d4f97b/Qwen3.5-4B-UD-IQ2_M.gguf?download=true",
        "approx": "~1.66 GB",
    },
    "Ultra Low VRAM (UD-IQ2)": {
        "match_groups": [["qwen3.5-4b", "ud-iq2_m"]],
        "preferred": "Qwen3.5-4B-UD-IQ2_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/b0bd07786f94b700012140fd9263b2de83d4f97b/Qwen3.5-4B-UD-IQ2_M.gguf?download=true",
        "approx": "~1.66 GB",
    },
    "Balanced (Q4)": {
        # Do not accidentally select UD-Q4_K_XL when the preset says Q4_K_M.
        "match_groups": [["qwen3.5-4b", "q4_k_m"]],
        "preferred": "Qwen3.5-4B-Q4_K_M.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/b0bd07786f94b700012140fd9263b2de83d4f97b/Qwen3.5-4B-Q4_K_M.gguf?download=true",
        "approx": "~2.54 GiB",
    },
    "High Quality (UD-Q8)": {
        "match_groups": [["qwen3.5-4b", "ud-q8_k_xl"]],
        "preferred": "Qwen3.5-4B-UD-Q8_K_XL.gguf",
        "url": "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/b0bd07786f94b700012140fd9263b2de83d4f97b/Qwen3.5-4B-UD-Q8_K_XL.gguf?download=true",
        "approx": "~5.95 GB",
    },
}



def _find_curated_model(precision):
    cfg = CURATED_MODELS[precision]
    candidates = []
    for path in NODE_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() != ".gguf" or "mmproj" in path.name.lower():
            continue
        low = path.name.lower()
        for group_index, terms in enumerate(cfg["match_groups"]):
            if all(term.lower() in low for term in terms):
                candidates.append((group_index, len(path.name), path.name, path))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1], x[2].lower()))
    return candidates[0][3]


def _download_with_resume(url, target, label="file"):
    """Resumable stdlib downloader; avoids replacing the user's Python environment."""
    target = Path(target)
    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "RebelsPromptEnhancer/3.2.2"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    req = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(req, timeout=300)
        status = getattr(response, "status", None) or response.getcode()
        # If the server ignored Range, restart cleanly instead of appending duplicate bytes.
        mode = "ab" if existing and status == 206 else "wb"
        if mode == "wb":
            existing = 0
        total_hdr = response.headers.get("Content-Length")
        total = (int(total_hdr) + existing) if total_hdr and total_hdr.isdigit() else None
        downloaded = existing
        print(f"[RebelsPromptEnhancer] Downloading {label} -> {target.name}")
        with response, part.open(mode) as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded * 100.0 / total
                    print(f"[RebelsPromptEnhancer] {label}: {pct:.1f}%")
        part.replace(target)
        return target
    except Exception:
        # Keep a non-empty .part so the next attempt can resume.
        if part.exists() and part.stat().st_size == 0:
            part.unlink(missing_ok=True)
        raise


def _download_curated_model(precision):
    cfg = CURATED_MODELS[precision]
    target = NODE_DIR / cfg["preferred"]
    if target.exists():
        return target
    return _download_with_resume(cfg["url"], target, f"{precision} model")


def _portable_runtime_manifest_path(runtime_dir=RUNTIME_DIR):
    return Path(runtime_dir) / "runtime.json"


def _find_portable_llama_binary(binary_name, runtime_dir=RUNTIME_DIR):
    runtime_dir = Path(runtime_dir)
    if not runtime_dir.exists():
        return None
    exe_name = binary_name + (".exe" if os.name == "nt" else "")
    direct = runtime_dir / exe_name
    if direct.is_file():
        return direct
    for path in runtime_dir.rglob(exe_name):
        if path.is_file():
            return path
    return None


def _find_portable_llama_cli(runtime_dir=RUNTIME_DIR):
    return _find_portable_llama_binary("llama-cli", runtime_dir)


def _find_portable_llama_completion(runtime_dir=RUNTIME_DIR):
    return _find_portable_llama_binary("llama-completion", runtime_dir)


def _find_portable_llama_server(runtime_dir=RUNTIME_DIR):
    return _find_portable_llama_binary("llama-server", runtime_dir)


# Pinned official llama.cpp build used by the private fallback. The CPU package is
# retained as the universal Windows safety net. NVIDIA systems can additionally use
# the matching CUDA 12.4 package + CUDART package for GPU offload.
PORTABLE_LLAMA_TAG = "b10964"
PORTABLE_LLAMA_CPU_ASSET = "llama-b10964-bin-win-cpu-x64.zip"
PORTABLE_LLAMA_CPU_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{PORTABLE_LLAMA_TAG}/{PORTABLE_LLAMA_CPU_ASSET}"
)
PORTABLE_LLAMA_CUDA_ASSET = "llama-b10964-bin-win-cuda-12.4-x64.zip"
PORTABLE_LLAMA_CUDA_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{PORTABLE_LLAMA_TAG}/{PORTABLE_LLAMA_CUDA_ASSET}"
)
PORTABLE_LLAMA_CUDART_ASSET = "cudart-llama-bin-win-cuda-12.4-x64.zip"
PORTABLE_LLAMA_CUDART_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{PORTABLE_LLAMA_TAG}/{PORTABLE_LLAMA_CUDART_ASSET}"
)


def _extract_runtime_archive(archive, runtime_dir):
    runtime_dir = Path(runtime_dir)
    extract_dir = runtime_dir / "extracting"
    shutil.rmtree(extract_dir, ignore_errors=True)
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(extract_dir)
    # Keep each official package together because llama.cpp executables depend on
    # sibling DLLs. Overlaying CUDART onto the CUDA package is intentional.
    for child in list(extract_dir.iterdir()):
        dest = runtime_dir / child.name
        if dest.exists():
            if child.is_dir() and dest.is_dir():
                # Merge nested package directories without deleting files from the
                # first archive (important when overlaying CUDART).
                for nested in child.rglob("*"):
                    if nested.is_dir():
                        continue
                    rel = nested.relative_to(child)
                    out = dest / rel
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(nested, out)
            else:
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink(missing_ok=True)
                shutil.move(str(child), str(dest))
        else:
            shutil.move(str(child), str(dest))
    shutil.rmtree(extract_dir, ignore_errors=True)


def _ensure_portable_cpu_runtime():
    existing = _find_portable_llama_completion(RUNTIME_DIR)
    server = _find_portable_llama_server(RUNTIME_DIR)
    if existing is not None and server is not None:
        return existing
    if os.name != "nt":
        raise RuntimeError(
            "Portable llama.cpp auto-fallback currently supports Windows x64 only. "
            "The installed llama-cpp-python backend failed before inference."
        )

    with RUNTIME_LOCK:
        existing = _find_portable_llama_completion(RUNTIME_DIR)
        server = _find_portable_llama_server(RUNTIME_DIR)
        if existing is not None and server is not None:
            return existing
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        archive = RUNTIME_DIR / PORTABLE_LLAMA_CPU_ASSET
        _download_with_resume(
            PORTABLE_LLAMA_CPU_URL, archive,
            f"portable llama.cpp CPU {PORTABLE_LLAMA_TAG}",
        )
        _extract_runtime_archive(archive, RUNTIME_DIR)
        archive.unlink(missing_ok=True)
        _portable_runtime_manifest_path(RUNTIME_DIR).write_text(
            json.dumps({
                "tag": PORTABLE_LLAMA_TAG,
                "asset": PORTABLE_LLAMA_CPU_ASSET,
                "backend": "cpu",
            }, indent=2),
            encoding="utf-8",
        )
        exe = _find_portable_llama_completion(RUNTIME_DIR)
        if exe is None:
            raise RuntimeError(
                "Portable llama.cpp CPU runtime downloaded, but llama-completion.exe was not found."
            )
        return exe


def _ensure_portable_cuda_runtime():
    """Install the pinned official CUDA 12.4 runtime privately for NVIDIA systems."""
    if os.name != "nt" or not torch.cuda.is_available():
        return None
    existing = _find_portable_llama_completion(CUDA_RUNTIME_DIR)
    server = _find_portable_llama_server(CUDA_RUNTIME_DIR)
    if existing is not None and server is not None:
        return existing

    with RUNTIME_LOCK:
        existing = _find_portable_llama_completion(CUDA_RUNTIME_DIR)
        server = _find_portable_llama_server(CUDA_RUNTIME_DIR)
        if existing is not None and server is not None:
            return existing
        CUDA_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        packages = [
            (PORTABLE_LLAMA_CUDA_ASSET, PORTABLE_LLAMA_CUDA_URL, "CUDA runtime"),
            (PORTABLE_LLAMA_CUDART_ASSET, PORTABLE_LLAMA_CUDART_URL, "CUDA support DLLs"),
        ]
        for asset_name, asset_url, label in packages:
            archive = CUDA_RUNTIME_DIR / asset_name
            _download_with_resume(
                asset_url, archive,
                f"portable llama.cpp {label} {PORTABLE_LLAMA_TAG}",
            )
            _extract_runtime_archive(archive, CUDA_RUNTIME_DIR)
            archive.unlink(missing_ok=True)

        _portable_runtime_manifest_path(CUDA_RUNTIME_DIR).write_text(
            json.dumps({
                "tag": PORTABLE_LLAMA_TAG,
                "asset": PORTABLE_LLAMA_CUDA_ASSET,
                "cudart": PORTABLE_LLAMA_CUDART_ASSET,
                "backend": "cuda12.4",
            }, indent=2),
            encoding="utf-8",
        )
        exe = _find_portable_llama_completion(CUDA_RUNTIME_DIR)
        if exe is None:
            raise RuntimeError(
                "Portable llama.cpp CUDA runtime downloaded, but llama-completion.exe was not found."
            )
        return exe


def _portable_server_log_tail(max_chars=3000):
    path = _PORTABLE_SERVER.get("log_path")
    if not path:
        return ""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
        return text[-max_chars:]
    except Exception:
        return ""


def _stop_portable_server():
    with _PORTABLE_SERVER_LOCK:
        proc = _PORTABLE_SERVER.get("proc")
        if proc is not None:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
                        proc.wait(timeout=5)
            except Exception:
                pass
        log_handle = _PORTABLE_SERVER.get("log_handle")
        if log_handle is not None:
            try:
                log_handle.close()
            except Exception:
                pass
        _PORTABLE_SERVER.update({
            "key": None, "proc": None, "base_url": None, "log_handle": None,
            "log_path": None, "runtime": None,
        })


def _free_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _portable_health_ready(base_url, proc):
    if proc.poll() is not None:
        return False, "exited"
    try:
        with urllib.request.urlopen(base_url + "/health", timeout=1.5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8", errors="replace"))
                if isinstance(data, dict) and data.get("status") == "ok":
                    return True, "ready"
    except Exception as exc:
        # 503 while loading is expected; connection errors are expected during startup.
        return False, str(exc)
    return False, "loading"


def _ensure_portable_server(runtime_dir, model_path, n_ctx, gpu_layers, runtime_name, startup_timeout):
    server_exe = _find_portable_llama_server(runtime_dir)
    if server_exe is None:
        raise RuntimeError(f"Portable {runtime_name} runtime does not contain llama-server.exe")

    key = (
        str(Path(server_exe).resolve()), str(Path(model_path).resolve()),
        int(n_ctx), int(gpu_layers), runtime_name,
    )
    with _PORTABLE_SERVER_LOCK:
        proc = _PORTABLE_SERVER.get("proc")
        if (
            _PORTABLE_SERVER.get("key") == key
            and proc is not None
            and proc.poll() is None
            and _PORTABLE_SERVER.get("base_url")
        ):
            return _PORTABLE_SERVER["base_url"]

        _stop_portable_server()
        port = _free_local_port()
        base_url = f"http://127.0.0.1:{port}"
        log_path = Path(runtime_dir) / f"rebels_llama_server_{runtime_name}.log"
        log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        cmd = [
            str(server_exe),
            "-m", str(Path(model_path).resolve()),
            "-c", str(int(n_ctx)),
            "-ngl", str(int(gpu_layers)),
            "--no-repack",
            "--no-warmup",
            "--host", "127.0.0.1",
            "--port", str(port),
        ]
        env = os.environ.copy()
        env.pop("LLAMA_ARG_REPACK", None)
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(server_exe).parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _PORTABLE_SERVER.update({
            "key": key, "proc": proc, "base_url": base_url,
            "log_handle": log_handle, "log_path": str(log_path), "runtime": runtime_name,
        })

    print(
        f"[RebelsPromptEnhancer] Loading portable {runtime_name} llama-server once "
        f"for {Path(model_path).name} (ctx={int(n_ctx)}, ngl={int(gpu_layers)})."
    )
    deadline = time.monotonic() + float(startup_timeout)
    last_status = "starting"
    while time.monotonic() < deadline:
        with _PORTABLE_SERVER_LOCK:
            proc = _PORTABLE_SERVER.get("proc")
            current_url = _PORTABLE_SERVER.get("base_url")
        if proc is None or current_url != base_url:
            raise RuntimeError("Portable llama-server was replaced while starting.")
        ready, last_status = _portable_health_ready(base_url, proc)
        if ready:
            print(f"[RebelsPromptEnhancer] Portable {runtime_name} llama-server ready.")
            return base_url
        if proc.poll() is not None:
            tail = _portable_server_log_tail()
            _stop_portable_server()
            raise RuntimeError(
                f"Portable {runtime_name} llama-server exited during startup. {tail[-1800:]}"
            )
        time.sleep(0.35)

    tail = _portable_server_log_tail()
    _stop_portable_server()
    raise RuntimeError(
        f"Portable {runtime_name} llama-server timed out while loading the model. "
        f"Last status: {last_status}. {tail[-1800:]}"
    )


def _portable_server_complete(
    base_url, prompt, max_tokens, temperature, top_p, repeat_penalty, seed, schema, timeout_seconds
):
    payload = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "repeat_penalty": float(repeat_penalty),
        "seed": int(seed) & 0xFFFFFFFF,
        "json_schema": schema,
        "stream": False,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        base_url + "/completion",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError(f"Portable llama-server completion request failed: {exc}") from exc
    content = data.get("content") if isinstance(data, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"Portable llama-server returned no completion text: {data}")
    return content.strip()


def _portable_server_scope(func):
    """Honor load_mode for the portable backend without reloading between repair passes."""
    @wraps(func)
    def wrapped(*args, **kwargs):
        load_mode = kwargs.get("load_mode")
        try:
            return func(*args, **kwargs)
        finally:
            if load_mode == "VRAM Saver (unload after run)":
                _stop_portable_server()
    return wrapped


def _portable_completion_cmd(
    exe,
    model_path,
    prompt,
    n_ctx,
    max_tokens,
    temperature,
    top_p,
    repeat_penalty,
    seed,
    schema_json,
    gpu_layers,
):
    return [
        str(exe),
        "-m", str(Path(model_path).resolve()),
        "-p", prompt,
        "-c", str(int(n_ctx)),
        "-n", str(int(max_tokens)),
        "--temp", str(float(temperature)),
        "--top-p", str(float(top_p)),
        "--repeat-penalty", str(float(repeat_penalty)),
        "--seed", str(int(seed) & 0xFFFFFFFF),
        "-ngl", str(int(gpu_layers)),
        "--no-repack",
        "--no-warmup",
        "--no-conversation",
        "--no-display-prompt",
        "-j", schema_json,
    ]


def _run_portable_process(cmd, exe, timeout_seconds):
    env = os.environ.copy()
    env.pop("LLAMA_ARG_REPACK", None)
    try:
        return subprocess.run(
            cmd,
            cwd=str(Path(exe).parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Portable llama.cpp process timed out after {timeout_seconds // 60} minutes."
        ) from exc


def _portable_failure_detail(proc):
    stderr = (proc.stderr or "").strip()
    stdout = (proc.stdout or "").strip()
    detail = stderr[-3000:] if stderr else f"exit code {proc.returncode}"
    return stdout, detail


def _run_portable_rewrite(
    model_path,
    raw_prompt,
    sys_prompt,
    n_ctx,
    max_tokens,
    temperature,
    top_p,
    repeat_penalty,
    seed,
    schema,
    load_mode="VRAM Saver (unload after run)",
):
    """Out-of-process fallback using a reusable local llama-server.

    The server keeps the GGUF loaded across quality retries. With Fast Iteration it
    also remains loaded across node executions. VRAM Saver unloads it when the node
    execution finishes via _portable_server_scope.
    """
    prompt = (
        sys_prompt.strip()
        + "\n\nUSER REQUEST:\n"
        + raw_prompt.strip()
        + "\n\nReturn ONLY the requested JSON object. Do not use markdown fences or commentary."
    )

    cuda_error = None
    if os.name == "nt" and torch.cuda.is_available():
        try:
            _ensure_portable_cuda_runtime()
            base_url = _ensure_portable_server(
                CUDA_RUNTIME_DIR, model_path, n_ctx, 999, "CUDA", startup_timeout=300
            )
            return _portable_server_complete(
                base_url, prompt, max_tokens, temperature, top_p,
                repeat_penalty, seed, schema, timeout_seconds=300,
            )
        except Exception as exc:
            cuda_error = str(exc)
            print(
                "[RebelsPromptEnhancer] Portable CUDA llama-server failed; "
                "falling back to portable CPU server. " + cuda_error[-1000:]
            )
            _stop_portable_server()

    try:
        _ensure_portable_cpu_runtime()
        base_url = _ensure_portable_server(
            RUNTIME_DIR, model_path, n_ctx, 0, "CPU", startup_timeout=900
        )
        return _portable_server_complete(
            base_url, prompt, max_tokens, temperature, top_p,
            repeat_penalty, seed, schema, timeout_seconds=900,
        )
    except Exception as exc:
        _stop_portable_server()
        if cuda_error:
            raise RuntimeError(
                "Portable llama.cpp CUDA server failed, then CPU server also failed: "
                f"CUDA: {cuda_error[-1200:]} | CPU: {str(exc)}"
            ) from exc
        raise


# ================================================================
# Structured rewrite engine
# ================================================================


def _schema_for_rewrite(variant_count, generate_negative, min_prompt_chars=1):
    # Keep grammar constraints structural only. Do NOT encode Long/Max length
    # floors into JSON-string minLength: small quants may satisfy giant grammar
    # constraints by stuffing template/schema text into the prompt.
    #
    # Also avoid duplicating long text. For one requested variant we generate only
    # final_prompt and derive variants=[final_prompt] in Python. For multiple
    # variants we generate only the variants array and use variants[0] as the
    # primary final_prompt. This substantially reduces Max Detail output pressure.
    prompt_schema = {"type": "string", "minLength": 1}
    props = {}
    required = []
    if int(variant_count) <= 1:
        props["final_prompt"] = prompt_schema
        required.append("final_prompt")
    else:
        props["variants"] = {
            "type": "array",
            "items": prompt_schema,
            "minItems": int(variant_count),
            "maxItems": int(variant_count),
        }
        required.append("variants")
    if generate_negative:
        props["negative_prompt"] = {"type": "string"}
        required.append("negative_prompt")
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def _build_system_prompt(
    purpose,
    model_format,
    aesthetic,
    detail_strength,
    length_target,
    preserve_user_details,
    variant_count,
    generate_negative,
    extra_instructions="",
    append_no_think=False,
):
    parts = [
        "You are a prompt rewriting engine for generative media. Rewrite the user's input, do not answer it conversationally.",
        PURPOSE_FRAMING[purpose],
        MODEL_FORMAT_INSTRUCTIONS[model_format],
        DETAIL_INSTRUCTIONS[detail_strength],
        LENGTH_INSTRUCTIONS[length_target],
    ]
    if preserve_user_details:
        parts.append(PRESERVE_RULE)
    if aesthetic in AESTHETIC_DESCRIPTORS:
        parts.append(
            f"MANDATORY AESTHETIC: The selected aesthetic is '{aesthetic}'. This is a hard requirement, not a suggestion. "
            "Every returned positive prompt must explicitly name or unmistakably describe this aesthetic so it survives downstream. "
            + AESTHETIC_DESCRIPTORS[aesthetic]
        )
    if variant_count > 1:
        parts.append(
            f"Return exactly {variant_count} meaningfully different prompts in the variants array. Variant 1 is the primary/best balanced default; "
            "other variants may shift composition, camera, or atmosphere only where the user left those unspecified. "
            "Do not duplicate variant 1 into a separate final_prompt field; the caller derives the primary prompt from variants[0]."
        )
    else:
        parts.append(
            "Return exactly one rewritten prompt in final_prompt. Do not duplicate the same long prompt into a variants array; "
            "the caller will derive the single-item variants output automatically."
        )
    if generate_negative:
        parts.append(
            "Also return a concise negative_prompt only when negatives are useful for the selected target model; otherwise return an empty string. "
            "Never put positive scene content into the negative prompt."
        )
    if extra_instructions.strip():
        parts.append("Additional user-configured rewrite instruction: " + extra_instructions.strip())
    parts.append(
        "ENHANCEMENT REQUIREMENT: A rewrite must materially improve the input, not merely repeat it. "
        "Never return the user's prompt unchanged unless it already fully satisfies the selected detail and length target. "
        "For sparse prompts, actively fill in unspecified visual details that are compatible with the user's concept. "
        "Preserving explicit details does not mean staying vague. Obey the requested target length; for example, a two-word "
        "subject prompt cannot satisfy a 40-90 word Short target by being returned unchanged."
    )
    parts.append(
        "STRUCTURED OUTPUT RULE: JSON syntax belongs only to the outer response object. Inside final_prompt, variants, and "
        "negative_prompt values, write ONLY the actual generation prompt text. Never place JSON objects, key names such as "
        "final_prompt/variants, placeholders such as [INSERT PROMPT HERE], FINAL ANSWER labels, USER INPUT labels, schema examples, "
        "word-count notes, character-count notes, or output templates inside a prompt string."
    )
    parts.append("Output exactly one valid JSON object matching the requested schema. Do not include markdown, reasoning, examples, or a second JSON object.")
    text = " ".join(parts)
    if append_no_think:
        text += " /no_think"
    return text


_OUTPUT_SCAFFOLD_PATTERNS = (
    r"\[\s*insert\s+prompt\s+here\s*\]",
    r"\bfinal\s+answer\s*:",
    r"\buser\s+input\s*:",
    r"\(\s*(?:min|max)\s+\d+\s*(?:chars?|characters?|words?)\s*\)",
    r"```(?:json)?",
)


def _contains_output_scaffolding(text):
    value = (text or "").strip()
    if not value:
        return False
    low = value.lower()
    if any(re.search(pat, value, flags=re.IGNORECASE) for pat in _OUTPUT_SCAFFOLD_PATTERNS):
        return True
    # JSON/schema keys inside the generated prompt value are almost always leakage.
    if '{"final_prompt"' in low or "{'final_prompt'" in low:
        return True
    if low.count('"final_prompt"') >= 1 or low.count('"variants"') >= 1:
        return True
    if "[end of text]" in low:
        return True
    return False


def _iter_json_dicts(text):
    """Yield JSON objects found anywhere in a string, ignoring trailing text."""
    value = (text or "").strip()
    if not value:
        return
    decoder = json.JSONDecoder()
    seen = set()
    try:
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            seen.add(json.dumps(parsed, sort_keys=True, ensure_ascii=False))
            yield parsed
    except Exception:
        pass
    for match in re.finditer(r"\{", value):
        try:
            parsed, _ = decoder.raw_decode(value[match.start():])
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue
        fingerprint = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        yield parsed


def _prompt_value_is_clean(text):
    value = (text or "").strip()
    return bool(value) and not _contains_output_scaffolding(value)


def _salvage_embedded_prompt(text):
    """Recover the best clean prompt if a model stuffed one or more JSON objects inside a string."""
    candidates = []
    for obj in _iter_json_dicts(text):
        fp = obj.get("final_prompt")
        if isinstance(fp, str) and _prompt_value_is_clean(fp):
            candidates.append(fp.strip())
        variants = obj.get("variants")
        if isinstance(variants, list):
            for item in variants:
                if isinstance(item, str) and _prompt_value_is_clean(item):
                    candidates.append(item.strip())
    if not candidates:
        return ""
    # Prefer the richest complete candidate; truncated retry fragments tend to be shorter.
    return max(candidates, key=lambda x: (len(x.split()), len(x)))


def _clean_structured_value(text, allow_salvage=True):
    value = (text or "").strip()
    value = re.sub(r"\s*\[end of text\]\s*$", "", value, flags=re.IGNORECASE).strip()
    if not value:
        return "", False
    if _contains_output_scaffolding(value):
        if allow_salvage:
            salvaged = _salvage_embedded_prompt(value)
            if salvaged:
                return salvaged, True
        return "", False
    return value, False


def _decode_json_object_from_output(raw_output):
    """Decode the best outer JSON object even when llama.cpp appends status text."""
    objects = list(_iter_json_dicts(raw_output))
    if not objects:
        raise ValueError("no JSON object found in model output")
    # Prefer an object with the expected key. Normally this is the first/outer object.
    for obj in objects:
        if "final_prompt" in obj or "variants" in obj:
            return obj
    return objects[0]


def _extract_json_payload(raw_output, raw_prompt, variant_count, generate_negative):
    try:
        payload = _decode_json_object_from_output(raw_output)
    except Exception:
        cleaned = _legacy_clean_output(raw_output, raw_prompt)
        cleaned, salvaged = _clean_structured_value(cleaned, allow_salvage=True)
        return cleaned, "", [cleaned] if cleaned else [], "legacy-salvaged" if salvaged else "legacy-fallback"

    parse_mode = "structured"
    final_prompt, final_salvaged = _clean_structured_value(payload.get("final_prompt", ""), allow_salvage=True)
    if final_salvaged:
        parse_mode = "structured-salvaged"

    raw_variants = payload.get("variants", [])
    if not isinstance(raw_variants, list):
        raw_variants = []
    variants = []
    for item in raw_variants:
        cleaned, salvaged = _clean_structured_value(item, allow_salvage=True)
        if cleaned:
            variants.append(cleaned)
            if salvaged:
                parse_mode = "structured-salvaged"

    negative = ""
    if generate_negative:
        negative, _ = _clean_structured_value(payload.get("negative_prompt", ""), allow_salvage=False)

    if not final_prompt and variants:
        final_prompt = variants[0]
    if final_prompt and variant_count == 1:
        variants = [final_prompt]
    elif final_prompt and not variants:
        variants = [final_prompt]
    variants = variants[:variant_count]

    if not final_prompt:
        return "", negative, [], "structured-empty"
    return final_prompt, negative, variants, parse_mode


def _normalize_rewrite_text(text):
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _rewrite_is_too_literal(raw_prompt, final_prompt, detail_strength):
    """Catch obvious no-op rewrites without imposing rigid stylistic length rules."""
    raw_norm = _normalize_rewrite_text(raw_prompt)
    final_norm = _normalize_rewrite_text(final_prompt)
    if not raw_norm or not final_norm:
        return False
    if raw_norm == final_norm:
        return True

    raw_words = raw_norm.split()
    final_words = final_norm.split()
    if detail_strength in {"Expand", "Highly Detailed", "Creative"} and len(raw_words) <= 12:
        if len(final_words) <= max(len(raw_words) + 5, 12):
            return True
    return False


def _aesthetic_is_present(aesthetic, text):
    if aesthetic not in AESTHETIC_DESCRIPTORS:
        return True
    low = (text or "").lower()
    anchors = AESTHETIC_ANCHORS.get(aesthetic, ())
    return any(anchor in low for anchor in anchors) if anchors else True


def _has_repeated_visual_phrases(text):
    """Detect obvious exact phrase/chunk repetition without penalizing normal prose."""
    value = (text or "").strip()
    if not value:
        return False
    chunks = re.split(r"[,.!?;:\n]+", value)
    seen = set()
    for chunk in chunks:
        norm = _normalize_rewrite_text(chunk)
        words = norm.split()
        if len(words) < 3 or len(norm) < 18:
            continue
        if norm in seen:
            return True
        seen.add(norm)
    return False


def _quality_issue_penalty(issues):
    """Severity score used to keep the best draft when a repair attempt regresses."""
    total = 0
    for issue in issues or []:
        low = issue.lower()
        if "empty structured prompt" in low:
            total += 1000
        elif "structured-output/template" in low:
            total += 700
        elif "expected" in low and "variant" in low:
            total += 500
        elif "duplicated" in low:
            total += 350
        elif "too close" in low:
            total += 300
        elif "aesthetic" in low:
            total += 250
        elif "repetitive" in low:
            total += 120
        elif "under-length" in low:
            total += 60
        else:
            total += 100
    return total


def _candidate_rank(raw_prompt, final_prompt, variants, detail_strength, length_target, aesthetic, variant_count=1):
    """Higher is better. Never let an empty/worse retry replace a usable draft."""
    issues = _rewrite_quality_issues(
        raw_prompt, final_prompt, variants, detail_strength, length_target, aesthetic, variant_count
    )
    if not (final_prompt or "").strip():
        return (-10_000, -10_000, -10_000), issues
    words = len(_normalize_rewrite_text(final_prompt).split())
    chars = len((final_prompt or "").strip())
    penalty = _quality_issue_penalty(issues)
    target_words = LENGTH_MIN_WORDS[length_target]
    useful_words = min(words, target_words + 120)
    return (-penalty, useful_words, chars), issues


def _rewrite_quality_issues(raw_prompt, final_prompt, variants, detail_strength, length_target, aesthetic, variant_count=1):
    if not (final_prompt or "").strip():
        return ["empty structured prompt"]

    issues = []
    prompts = [final_prompt] + [v for v in variants if v and v != final_prompt]
    if any(_contains_output_scaffolding(p) for p in prompts):
        issues.append("structured-output/template text leaked into a prompt value")

    if len(variants) < int(variant_count):
        issues.append(f"expected {variant_count} clean variant(s), received {len(variants)}")
    elif int(variant_count) > 1:
        unique = {_normalize_rewrite_text(v) for v in variants if v}
        if len(unique) < int(variant_count):
            issues.append("requested variants are duplicated")

    min_words = LENGTH_MIN_WORDS[length_target]
    min_chars = LENGTH_MIN_CHARS[length_target]
    clean_prompts = [p for p in prompts if p]
    if clean_prompts:
        shortest_words = min(len(_normalize_rewrite_text(p).split()) for p in clean_prompts)
        shortest_chars = min(len(p.strip()) for p in clean_prompts)
        if shortest_words < min_words or shortest_chars < min_chars:
            issues.append(
                f"under-length for {length_target} (quality floor {min_words} words / {min_chars} characters per prompt)"
            )
    if _rewrite_is_too_literal(raw_prompt, final_prompt, detail_strength):
        issues.append("rewrite is too close to the source prompt")
    if aesthetic in AESTHETIC_DESCRIPTORS and any(not _aesthetic_is_present(aesthetic, p) for p in clean_prompts):
        issues.append(f"selected aesthetic '{aesthetic}' is missing or not explicit")
    if any(_has_repeated_visual_phrases(p) for p in clean_prompts):
        issues.append("obvious repetitive visual phrases")
    return issues


def _run_rewrite(
    model_path,
    raw_prompt,
    sys_prompt,
    n_gpu_layers,
    n_ctx,
    max_tokens,
    temperature,
    top_p,
    repeat_penalty,
    seed,
    variant_count,
    generate_negative,
    load_mode,
    auto_cpu_fallback,
    force_portable=False,
    min_prompt_chars=1,
):
    global _NATIVE_BACKEND_DISABLED
    schema = _schema_for_rewrite(variant_count, generate_negative, min_prompt_chars=1)

    if force_portable or _NATIVE_BACKEND_DISABLED:
        raw = _run_portable_rewrite(
            model_path=model_path,
            raw_prompt=raw_prompt,
            sys_prompt=sys_prompt,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            schema=schema,
            load_mode=load_mode,
        )
        return raw, 0, "portable llama.cpp fallback"

    def _once(gpu_layers, structured=True):
        llm, release_after = _get_llm(model_path, gpu_layers, n_ctx, seed, load_mode)
        try:
            kwargs = dict(
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": raw_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                seed=seed,
                stop=VERBOSE_STOPS,
            )
            if structured:
                kwargs["response_format"] = {"type": "json_object", "schema": schema}
            output = llm.create_chat_completion(**kwargs)
            return output["choices"][0]["message"]["content"].strip(), gpu_layers, "llama-cpp-python"
        finally:
            if release_after:
                _free_llm(llm)

    def _attempt(gpu_layers):
        try:
            return _once(gpu_layers, structured=True)
        except Exception as exc:
            msg = str(exc).lower()
            structured_error = any(
                token in msg for token in (
                    "response_format", "json schema", "json_schema",
                    "grammar", "gbnf", "schema",
                )
            )
            if structured_error and not _is_oom_error(exc):
                return _once(gpu_layers, structured=False)
            raise

    primary_exc = None
    try:
        return _attempt(n_gpu_layers)
    except Exception as exc:
        primary_exc = exc

    # CPU retry is useful for CUDA/VRAM failures. An illegal-instruction failure is
    # a native backend incompatibility and retrying the same DLL on CPU only just
    # produces a second identical Windows fatal-exception trace.
    should_retry_cpu = (
        auto_cpu_fallback
        and n_gpu_layers != 0
        and _is_oom_error(primary_exc)
    )
    if should_retry_cpu:
        _release_hot_model()
        try:
            return _attempt(0)
        except Exception as cpu_exc:
            primary_exc = cpu_exc

    # V3.2: never tell the user to reinstall/rollback llama-cpp-python for this known native crash.
    # If the currently installed native DLL raises STATUS_ILLEGAL_INSTRUCTION, transparently
    # execute the same GGUF with an official private llama.cpp CLI stored under this node.
    if _is_illegal_instruction_error(primary_exc):
        _release_hot_model()
        _NATIVE_BACKEND_DISABLED = True
        _persist_native_backend_disabled("STATUS_ILLEGAL_INSTRUCTION / 0xC000001D")
        print(
            "[RebelsPromptEnhancer] Installed llama-cpp-python raised STATUS_ILLEGAL_INSTRUCTION. "
            "Portable llama.cpp compatibility mode has been persisted for this RebelsPromptEnhancer install; "
            "future ComfyUI restarts will skip the incompatible native backend. Global Python packages are unchanged."
        )
        raw = _run_portable_rewrite(
            model_path=model_path,
            raw_prompt=raw_prompt,
            sys_prompt=sys_prompt,
            n_ctx=n_ctx,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            schema=schema,
            load_mode=load_mode,
        )
        return raw, 0, "portable llama.cpp fallback"

    raise primary_exc



def _max_tokens_for_length(length_target, base=None, variant_count=1, generate_negative=False):
    # max_tokens covers the entire JSON object, including all variants.
    per_variant = {"Short": 260, "Medium": 520, "Long": 1100, "Max Detail": 1800}[length_target]
    if base is not None:
        # Custom node: keep the user's explicit max_tokens as the ceiling.
        return min(max(int(base), 64), 4096)
    value = per_variant * max(1, int(variant_count))
    if generate_negative:
        value += 160
    return min(max(value, 128), 4096)


def _context_for_rewrite(max_tokens, raw_prompt=""):
    # Reserve space for the system prompt, user prompt, and structured JSON framing.
    prompt_allowance = max(1536, min(3072, 1536 + len((raw_prompt or "")) // 4))
    needed = int(max_tokens) + prompt_allowance
    if needed <= 4096:
        return 4096
    if needed <= 8192:
        return 8192
    return 12288


# ================================================================
# Node 1: Curated enhancer V3
# ================================================================

class RebelsPromptEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        # IMPORTANT: The first 7 widgets preserve the exact V2 order so existing
        # workflows do not reinterpret seed/randomize values as new V3 controls.
        return {
            "required": {
                "raw_prompt": ("STRING", {"multiline": True}),
                "purpose": (PURPOSE_OPTIONS,),
                "model_format": (MODEL_FORMAT_OPTIONS,),
                "aesthetic": (AESTHETIC_OPTIONS,),
                "precision": (list(CURATED_MODELS.keys()),),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "lock_in": ("BOOLEAN", {
                    "default": False,
                    "label_on": "🔒 LOCKED (persistent cache)",
                    "label_off": "🔄 LIVE (generating)",
                }),
                # V3+ controls are appended after all legacy widgets.
                "detail_strength": (DETAIL_OPTIONS,),
                "length_target": (LENGTH_OPTIONS,),
                "preserve_user_details": ("BOOLEAN", {"default": True}),
                "variant_count": (VARIANT_OPTIONS,),
                "negative_prompt": (NEGATIVE_OPTIONS,),
                "load_mode": (LOAD_MODE_OPTIONS,),
                "auto_cpu_fallback": ("BOOLEAN", {"default": True}),
                "auto_download_missing": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "thought_process", "negative_prompt", "variants_json")
    FUNCTION = "enhance"
    CATEGORY = "Rebel AI"

    @classmethod
    def IS_CHANGED(cls, raw_prompt, purpose, model_format, aesthetic, precision,
                   detail_strength, length_target, preserve_user_details, variant_count,
                   negative_prompt, load_mode, auto_cpu_fallback, auto_download_missing,
                   seed, lock_in):
        if lock_in:
            payload = [raw_prompt, purpose, model_format, aesthetic, precision, detail_strength,
                       length_target, preserve_user_details, variant_count, negative_prompt]
            return "LOCKED|" + _cache_hash(payload)
        return float("nan")

    @_portable_server_scope
    def enhance(self, raw_prompt, purpose, model_format, aesthetic, precision,
                detail_strength, length_target, preserve_user_details, variant_count,
                negative_prompt, load_mode, auto_cpu_fallback, auto_download_missing,
                seed, lock_in):
        variant_count = int(variant_count)
        generate_negative = negative_prompt == "Generate when useful"
        cache_payload = {
            "engine_revision": "3.3.2",
            "node": "curated-v3", "raw_prompt": raw_prompt, "purpose": purpose,
            "model_format": model_format, "aesthetic": aesthetic, "precision": precision,
            "detail_strength": detail_strength, "length_target": length_target,
            "preserve_user_details": preserve_user_details, "variant_count": variant_count,
            "negative_prompt": negative_prompt,
        }

        if lock_in:
            cached = _cache_get(cache_payload)
            if cached:
                thought = "=== 🔒 PERSISTENT CACHE HIT ===\nNo model load.\n\n" + cached.get("meta", "")
                return (
                    cached["final_prompt"], thought,
                    cached.get("negative_prompt", ""),
                    json.dumps(cached.get("variants", [cached["final_prompt"]]), ensure_ascii=False, indent=2),
                )

        model_path = _find_curated_model(precision)
        if model_path is None and auto_download_missing:
            model_path = _download_curated_model(precision)
        if model_path is None:
            cfg = CURATED_MODELS[precision]
            raise FileNotFoundError(
                f"Missing curated model for '{precision}'.\n"
                f"Place {cfg['preferred']} (or a compatible matching quant) directly in {_relative_model_hint()}\n"
                f"Expected size: {cfg['approx']}. You may also enable auto_download_missing."
            )

        sys_prompt = _build_system_prompt(
            purpose, model_format, aesthetic, detail_strength, length_target,
            preserve_user_details, variant_count, generate_negative,
            append_no_think=True,
        )
        max_tokens = _max_tokens_for_length(
            length_target, variant_count=variant_count, generate_negative=generate_negative
        )
        effective_ctx = _context_for_rewrite(max_tokens, raw_prompt)
        min_prompt_chars = LENGTH_MIN_CHARS[length_target]
        raw_output, used_gpu_layers, backend_used = _run_rewrite(
            model_path=model_path,
            raw_prompt=raw_prompt,
            sys_prompt=sys_prompt,
            n_gpu_layers=-1,
            n_ctx=effective_ctx,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.12,
            seed=seed,
            variant_count=variant_count,
            generate_negative=generate_negative,
            load_mode=load_mode,
            auto_cpu_fallback=auto_cpu_fallback,
            min_prompt_chars=min_prompt_chars,
        )
        final_prompt, negative, variants, parse_mode = _extract_json_payload(
            raw_output, raw_prompt, variant_count, generate_negative
        )

        quality_retries = 0
        best_rank, quality_issues = _candidate_rank(
            raw_prompt, final_prompt, variants, detail_strength, length_target, aesthetic, variant_count
        )
        best_candidate = (final_prompt, negative, variants, parse_mode, raw_output, list(quality_issues))

        # Because llama-server remains loaded for this whole enhance() call, repair
        # passes are cheap. Crucially, a bad/empty retry can never overwrite a
        # better earlier draft.
        while quality_issues and quality_retries < 3:
            quality_retries += 1
            issue_text = "; ".join(quality_issues)
            repair_sys_prompt = sys_prompt + (
                f" QUALITY RETRY {quality_retries}: The best draft so far was rejected because: {issue_text}. "
                f"Write a genuinely {length_target.lower()} prompt with substantial, concrete visual detail. "
                f"For {length_target}, aim for at least about {LENGTH_MIN_WORDS[length_target]} useful words, but never pad with repetition. "
                "Use varied, specific details rather than repeating the same lighting, material, camera, or pose phrases. "
                "Do not mention word counts, character counts, schemas, templates, retries, JSON keys, or validation rules inside any prompt value. "
                "Keep every explicit user fact, materially expand unspecified visual details, and obey the selected aesthetic as a hard requirement. "
                "Return exactly one clean JSON object matching the supplied schema and nothing else."
            )
            retry_raw, retry_gpu_layers, retry_backend = _run_rewrite(
                model_path=model_path,
                raw_prompt=raw_prompt,
                sys_prompt=repair_sys_prompt,
                n_gpu_layers=-1,
                n_ctx=effective_ctx,
                max_tokens=max_tokens,
                temperature=min(0.84, 0.74 + 0.03 * quality_retries),
                top_p=0.92,
                repeat_penalty=1.10,
                seed=seed + quality_retries,
                variant_count=variant_count,
                generate_negative=generate_negative,
                load_mode=load_mode,
                auto_cpu_fallback=auto_cpu_fallback,
                force_portable=(backend_used == "portable llama.cpp fallback"),
                min_prompt_chars=min_prompt_chars,
            )
            retry_final, retry_negative, retry_variants, retry_parse_mode = _extract_json_payload(
                retry_raw, raw_prompt, variant_count, generate_negative
            )
            retry_rank, retry_issues = _candidate_rank(
                raw_prompt, retry_final, retry_variants, detail_strength, length_target, aesthetic, variant_count
            )
            if retry_rank > best_rank:
                best_rank = retry_rank
                best_candidate = (
                    retry_final, retry_negative, retry_variants, retry_parse_mode, retry_raw, list(retry_issues)
                )
                final_prompt, negative, variants, parse_mode, raw_output, quality_issues = best_candidate
                used_gpu_layers, backend_used = retry_gpu_layers, retry_backend
            else:
                # Keep repairing the best known draft; never replace it with an
                # empty, contaminated, shorter, or otherwise worse retry.
                final_prompt, negative, variants, parse_mode, raw_output, quality_issues = best_candidate

            if not quality_issues:
                break

        if not (final_prompt or "").strip():
            raise RuntimeError("Prompt enhancer returned an empty result after automatic repair attempts.")

        # Under-length or repetition are quality warnings, not fatal execution
        # errors. Returning the best clean draft is better than destroying a usable
        # result after a repair attempt. Structural contamination remains fatal.
        fatal_quality_issues = [
            issue for issue in quality_issues
            if (
                "empty structured prompt" in issue.lower()
                or "structured-output/template" in issue.lower()
                or ("expected" in issue.lower() and "variant" in issue.lower())
            )
        ]
        if fatal_quality_issues:
            raise RuntimeError(
                "Prompt enhancer could not produce a clean structured result after automatic repair attempts: "
                + "; ".join(fatal_quality_issues)
            )
        quality_warning = "; ".join(quality_issues) if quality_issues else ""

        meta = (
            f"Model: {model_path.name}\nPrecision: {precision}\nPurpose: {purpose}\n"
            f"Format: {model_format}\nAesthetic: {aesthetic}\nDetail: {detail_strength}\n"
            f"Length: {length_target}\nVariants: {variant_count}\nStructured parse: {parse_mode}\n"
            f"Quality retries: {quality_retries}\nQuality warning: {quality_warning or 'none'}\nContext: {effective_ctx}\nMax tokens: {max_tokens}\n"
            f"Backend: {backend_used}\nGPU layers used: {'all' if used_gpu_layers < 0 else used_gpu_layers}\nLoad mode: {load_mode}\n"
        )
        thought = (
            "=== 🔄 LIVE — Fresh Generation ===\n\n"
            f"=== Run Info ===\n{meta}\n"
            f"=== Assembled System Prompt ===\n{sys_prompt}\n\n"
            f"=== Raw Structured Output ===\n{raw_output}\n\n"
            f"=== Final Prompt ===\n{final_prompt}"
        )
        _cache_set(cache_payload, {
            "final_prompt": final_prompt,
            "negative_prompt": negative,
            "variants": variants,
            "meta": meta,
        })
        return final_prompt, thought, negative, json.dumps(variants, ensure_ascii=False, indent=2)


# ================================================================
# Node 2: Custom GGUF enhancer V3
# ================================================================

class RebelsPromptEnhancerCustom:
    @classmethod
    def INPUT_TYPES(cls):
        # IMPORTANT: Preserve V2 widget order/names first; append V3 controls later.
        return {
            "required": {
                "raw_prompt": ("STRING", {"multiline": True}),
                "model_file": (_list_ggufs(),),
                "purpose": (PURPOSE_OPTIONS,),
                "model_format": (MODEL_FORMAT_OPTIONS,),
                "aesthetic": (AESTHETIC_OPTIONS,),
                "extra_instructions": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "Optional extra rewrite instructions.",
                }),
                "system_prompt_override": ("STRING", {
                    "multiline": True, "default": "",
                    "placeholder": "If non-empty, replaces the layered system prompt.",
                }),
                "append_no_think": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Append /no_think",
                    "label_off": "Don't append",
                }),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 800, "min": 64, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.12, "min": 1.0, "max": 2.0, "step": 0.02}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "lock_in": ("BOOLEAN", {
                    "default": False,
                    "label_on": "🔒 LOCKED (persistent cache)",
                    "label_off": "🔄 LIVE (generating)",
                }),
                # V3+ controls appended only after legacy widgets.
                "detail_strength": (DETAIL_OPTIONS,),
                "length_target": (LENGTH_OPTIONS,),
                "preserve_user_details": ("BOOLEAN", {"default": True}),
                "variant_count": (VARIANT_OPTIONS,),
                "negative_prompt": (NEGATIVE_OPTIONS,),
                "no_think_mode": (NO_THINK_OPTIONS,),
                "load_mode": (LOAD_MODE_OPTIONS,),
                "auto_cpu_fallback": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "thought_process", "negative_prompt", "variants_json")
    FUNCTION = "enhance"
    CATEGORY = "Rebel AI"

    @classmethod
    def IS_CHANGED(cls, raw_prompt, model_file, purpose, model_format, aesthetic,
                   extra_instructions, system_prompt_override, append_no_think,
                   n_gpu_layers, n_ctx, max_tokens, temperature, top_p, repeat_penalty,
                   seed, lock_in, detail_strength, length_target, preserve_user_details,
                   variant_count, negative_prompt, no_think_mode, load_mode, auto_cpu_fallback):
        if lock_in:
            payload = [raw_prompt, model_file, purpose, model_format, aesthetic, detail_strength,
                       length_target, preserve_user_details, variant_count, negative_prompt,
                       extra_instructions, system_prompt_override, append_no_think, no_think_mode,
                       temperature, top_p, repeat_penalty]
            return "LOCKED|" + _cache_hash(payload)
        return float("nan")

    @_portable_server_scope
    def enhance(self, raw_prompt, model_file, purpose, model_format, aesthetic,
                extra_instructions, system_prompt_override, append_no_think,
                n_gpu_layers, n_ctx, max_tokens, temperature, top_p, repeat_penalty,
                seed, lock_in, detail_strength, length_target, preserve_user_details,
                variant_count, negative_prompt, no_think_mode, load_mode, auto_cpu_fallback):
        if model_file == "NO_GGUF_FILES_IN_FOLDER":
            raise FileNotFoundError(f"No GGUF found. Put a text GGUF directly in {_relative_model_hint()} and restart ComfyUI.")
        model_path = NODE_DIR / model_file
        if not model_path.is_file():
            raise FileNotFoundError(f"Selected GGUF no longer exists in {_relative_model_hint()}: {model_file}")

        variant_count = int(variant_count)
        generate_negative = negative_prompt == "Generate when useful"
        if append_no_think:
            effective_no_think_mode = "Force ON"
        else:
            effective_no_think_mode = no_think_mode
        append_no_think = _auto_no_think(model_file, effective_no_think_mode)
        detected_family = _detect_model_family(model_file)

        cache_payload = {
            "engine_revision": "3.3.2",
            "node": "custom-v3", "raw_prompt": raw_prompt, "model_file": model_file,
            "purpose": purpose, "model_format": model_format, "aesthetic": aesthetic,
            "detail_strength": detail_strength, "length_target": length_target,
            "preserve_user_details": preserve_user_details, "variant_count": variant_count,
            "negative_prompt": negative_prompt, "extra_instructions": extra_instructions,
            "system_prompt_override": system_prompt_override,
            "append_no_think": append_no_think, "no_think_mode": no_think_mode,
            "temperature": temperature, "top_p": top_p, "repeat_penalty": repeat_penalty,
        }
        if lock_in:
            cached = _cache_get(cache_payload)
            if cached:
                thought = "=== 🔒 PERSISTENT CACHE HIT ===\nNo model load.\n\n" + cached.get("meta", "")
                return (
                    cached["final_prompt"], thought,
                    cached.get("negative_prompt", ""),
                    json.dumps(cached.get("variants", [cached["final_prompt"]]), ensure_ascii=False, indent=2),
                )

        if system_prompt_override.strip():
            sys_prompt = system_prompt_override.strip()
            if append_no_think:
                sys_prompt += " /no_think"
            sys_prompt += " Output valid JSON only with final_prompt, variants, and optional negative_prompt fields."
        else:
            sys_prompt = _build_system_prompt(
                purpose, model_format, aesthetic, detail_strength, length_target,
                preserve_user_details, variant_count, generate_negative,
                extra_instructions=extra_instructions,
                append_no_think=append_no_think,
            )

        effective_max = _max_tokens_for_length(
            length_target, max_tokens, variant_count=variant_count, generate_negative=generate_negative
        )
        effective_ctx = max(int(n_ctx), _context_for_rewrite(effective_max, raw_prompt))
        min_prompt_chars = LENGTH_MIN_CHARS[length_target]
        raw_output, used_gpu_layers, backend_used = _run_rewrite(
            model_path=model_path,
            raw_prompt=raw_prompt,
            sys_prompt=sys_prompt,
            n_gpu_layers=n_gpu_layers,
            n_ctx=effective_ctx,
            max_tokens=effective_max,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            variant_count=variant_count,
            generate_negative=generate_negative,
            load_mode=load_mode,
            auto_cpu_fallback=auto_cpu_fallback,
            min_prompt_chars=min_prompt_chars,
        )
        final_prompt, negative, variants, parse_mode = _extract_json_payload(
            raw_output, raw_prompt, variant_count, generate_negative
        )

        quality_retries = 0
        quality_issues = _rewrite_quality_issues(
            raw_prompt, final_prompt, variants, detail_strength, length_target, aesthetic, variant_count
        )
        while quality_issues and quality_retries < 2:
            quality_retries += 1
            repair_sys_prompt = sys_prompt + (
                " QUALITY RETRY: " + "; ".join(quality_issues) + ". "
                f"Write a genuinely {length_target.lower()} clean prompt with substantial useful detail and obey the selected aesthetic. "
                "Do not mention schemas, templates, JSON keys, word/character limits, retries, or validation rules inside prompt values. "
                "Do not output empty strings or an unchanged source prompt. Return exactly one JSON object."
            )
            raw_output, used_gpu_layers, backend_used = _run_rewrite(
                model_path=model_path, raw_prompt=raw_prompt, sys_prompt=repair_sys_prompt,
                n_gpu_layers=n_gpu_layers, n_ctx=effective_ctx, max_tokens=effective_max,
                temperature=min(0.9, float(temperature) + 0.05 * quality_retries),
                top_p=top_p, repeat_penalty=max(1.0, float(repeat_penalty) - 0.04),
                seed=seed + quality_retries, variant_count=variant_count,
                generate_negative=generate_negative, load_mode=load_mode,
                auto_cpu_fallback=auto_cpu_fallback,
                force_portable=(backend_used == "portable llama.cpp fallback"),
                min_prompt_chars=min_prompt_chars,
            )
            final_prompt, negative, variants, parse_mode = _extract_json_payload(
                raw_output, raw_prompt, variant_count, generate_negative
            )
            quality_issues = _rewrite_quality_issues(
                raw_prompt, final_prompt, variants, detail_strength, length_target, aesthetic, variant_count
            )

        if not final_prompt:
            raise RuntimeError("Custom enhancer returned an empty result after automatic repair attempts.")
        if quality_issues:
            raise RuntimeError(
                "Custom enhancer could not satisfy the selected controls after automatic repair attempts: "
                + "; ".join(quality_issues)
            )

        gpu_label = "all" if used_gpu_layers < 0 else ("CPU only" if used_gpu_layers == 0 else str(used_gpu_layers))
        meta = (
            f"Model: {model_file}\nDetected family: {detected_family}\n/no_think: {append_no_think}\n"
            f"Backend: {backend_used}\nGPU layers used: {gpu_label}\nContext: {effective_ctx}\nMax tokens: {effective_max}\n"
            f"Quality retries: {quality_retries}\nTemperature: {temperature}\ntop_p: {top_p}\nrepeat_penalty: {repeat_penalty}\n"
            f"Structured parse: {parse_mode}\nLoad mode: {load_mode}\n"
        )
        thought = (
            "=== 🔄 LIVE — Fresh Custom GGUF Generation ===\n\n"
            f"=== Run Info ===\n{meta}\n"
            f"=== Assembled System Prompt ===\n{sys_prompt}\n\n"
            f"=== Raw Structured Output ===\n{raw_output}\n\n"
            f"=== Final Prompt ===\n{final_prompt}"
        )
        _cache_set(cache_payload, {
            "final_prompt": final_prompt,
            "negative_prompt": negative,
            "variants": variants,
            "meta": meta,
        })
        return final_prompt, thought, negative, json.dumps(variants, ensure_ascii=False, indent=2)


# ================================================================
# Node 3: Curated model helper/downloader
# ================================================================

class RebelsPromptEnhancerModelHelper:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "precision": (list(CURATED_MODELS.keys()),),
                "download": ("BOOLEAN", {
                    "default": False,
                    "label_on": "DOWNLOAD MODEL",
                    "label_off": "CHECK ONLY",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status",)
    FUNCTION = "run"
    CATEGORY = "Rebel AI"
    OUTPUT_NODE = True

    def run(self, precision, download):
        found = _find_curated_model(precision)
        cfg = CURATED_MODELS[precision]
        if found:
            status = f"READY: {found.name} is installed in {_relative_model_hint()}"
        elif not download:
            status = (
                f"MISSING: {precision}\nExpected: {cfg['preferred']} ({cfg['approx']})\n"
                f"Place it in {_relative_model_hint()} or toggle DOWNLOAD MODEL."
            )
        else:
            path = _download_curated_model(precision)
            status = f"DOWNLOADED: {path.name} to {_relative_model_hint()}"
        return {"ui": {"text": [status]}, "result": (status,)}


# ================================================================
# Node 4: Image-to-prompt vision node
# ================================================================

class RebelsImageToPrompt:
    VISION_TASK_OPTIONS = [
        "Caption (plain description)",
        "Caption + Format (apply model_format below)",
        "SD/Booru Tags",
        "Pose & Anatomy Focus",
        "Custom Instruction",
    ]

    @classmethod
    def INPUT_TYPES(cls):
        handler_options = ["Auto-detect"] + list(_HANDLER_CLASSES.keys())
        if not _HANDLER_CLASSES:
            handler_options = ["NO_VISION_HANDLERS_AVAILABLE"]
        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (_list_ggufs(),),
                "mmproj_file": (_list_mmproj(),),
                "chat_handler": (handler_options,),
                "vision_task": (cls.VISION_TASK_OPTIONS,),
                "model_format": (MODEL_FORMAT_OPTIONS,),
                "aesthetic": (AESTHETIC_OPTIONS,),
                "custom_instruction": ("STRING", {"multiline": True, "default": ""}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 500, "min": 50, "max": 2048, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 2.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("image_prompt", "thought_process")
    FUNCTION = "describe"
    CATEGORY = "Rebel AI"

    def _resolve_handler(self, choice, model_file):
        if choice != "Auto-detect":
            if choice not in _HANDLER_CLASSES:
                raise RuntimeError(f"Vision handler '{choice}' is unavailable. Update llama-cpp-python.")
            return _HANDLER_CLASSES[choice], choice
        mf = model_file.lower()
        checks = [
            ("moondream", "Moondream"),
            ("qwen", "Qwen2.5-VL"),
            ("minicpm", "MiniCPM-V 2.6"),
            ("nano", "NanoLLaVA"),
            ("llava", "LLaVA 1.6"),
            ("llava", "LLaVA 1.5"),
        ]
        for token, label in checks:
            if token in mf and label in _HANDLER_CLASSES:
                return _HANDLER_CLASSES[label], label
        if "LLaVA 1.5" in _HANDLER_CLASSES:
            return _HANDLER_CLASSES["LLaVA 1.5"], "LLaVA 1.5 (fallback)"
        raise RuntimeError("No compatible vision chat handler is available.")

    def _tensor_to_data_uri(self, image_tensor):
        img = image_tensor[0] if image_tensor.dim() == 4 else image_tensor
        arr = (img.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr)
        buf = BytesIO()
        pil.save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

    def _instruction(self, vision_task, model_format, aesthetic, custom_instruction):
        if vision_task == "Custom Instruction":
            return custom_instruction.strip() or "Describe this image faithfully in concrete visual detail."
        if vision_task == "Caption + Format (apply model_format below)":
            text = "Describe the image faithfully, then format the description using these rules: " + MODEL_FORMAT_INSTRUCTIONS[model_format]
            if aesthetic in AESTHETIC_DESCRIPTORS:
                text += " " + AESTHETIC_DESCRIPTORS[aesthetic]
            return text + " Output only the final description."
        if vision_task == "SD/Booru Tags":
            return "Return only a comma-separated descriptive tag list covering subject, action, setting, lighting, mood and style."
        if vision_task == "Pose & Anatomy Focus":
            return "Describe pose, body position, framing, expression, visible anatomy, hand placement and subject orientation precisely."
        return "Describe this image in one detailed paragraph covering subject, composition, lighting, colors, mood and notable visual details."

    def describe(self, image, model_file, mmproj_file, chat_handler, vision_task,
                 model_format, aesthetic, custom_instruction, n_gpu_layers,
                 n_ctx, max_tokens, temperature, seed):
        if model_file == "NO_GGUF_FILES_IN_FOLDER":
            raise FileNotFoundError(f"Put a vision-capable GGUF in {_relative_model_hint()} and restart ComfyUI.")
        if mmproj_file == "NO_MMPROJ_FILE_FOUND":
            raise FileNotFoundError(f"Put the paired mmproj GGUF in {_relative_model_hint()} and restart ComfyUI.")
        if not _HANDLER_CLASSES:
            raise RuntimeError("No vision chat handlers available. Update llama-cpp-python.")

        model_path = NODE_DIR / model_file
        mmproj_path = NODE_DIR / mmproj_file
        if not model_path.is_file() or not mmproj_path.is_file():
            raise FileNotFoundError("Selected vision model or mmproj is missing from the custom node folder.")

        handler_cls, handler_label = self._resolve_handler(chat_handler, model_file)
        instruction = self._instruction(vision_task, model_format, aesthetic, custom_instruction)
        img_uri = self._tensor_to_data_uri(image)
        handler = handler_cls(clip_model_path=str(mmproj_path), verbose=False)
        llm = Llama(
            model_path=str(model_path), chat_handler=handler,
            n_gpu_layers=n_gpu_layers, n_ctx=n_ctx, seed=seed,
            logits_all=True, verbose=False,
        )
        try:
            output = llm.create_chat_completion(
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_uri}},
                        {"type": "text", "text": instruction},
                    ],
                }],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                repeat_penalty=1.1,
                seed=seed,
            )
            raw = output["choices"][0]["message"]["content"].strip()
        finally:
            _free_llm(llm)
            try:
                del handler
            except Exception:
                pass

        final = _legacy_clean_output(raw)
        thought = (
            f"=== Image Analysis ===\nModel: {model_file}\nmmproj: {mmproj_file}\n"
            f"Handler: {handler_label}\nTask: {vision_task}\n\nInstruction:\n{instruction}\n\nRaw:\n{raw}\n\nFinal:\n{final}"
        )
        return final, thought


# ================================================================
# Node 5: General LLM console
# ================================================================

class RebelsLLMConsole:
    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful, knowledgeable assistant. Answer directly and concisely without unnecessary preamble."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "question": ("STRING", {"multiline": True}),
                "model_file": (_list_ggufs(),),
                "system_prompt": ("STRING", {"multiline": True, "default": cls.DEFAULT_SYSTEM_PROMPT}),
                "no_think": (NO_THINK_OPTIONS,),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 999, "step": 1}),
                "n_ctx": ("INT", {"default": 4096, "min": 512, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 800, "min": 50, "max": 4096, "step": 50}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.12, "min": 1.0, "max": 2.0, "step": 0.02}),
                "load_mode": (LOAD_MODE_OPTIONS,),
                "auto_cpu_fallback": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "thought_process")
    FUNCTION = "chat"
    CATEGORY = "Rebel AI"
    OUTPUT_NODE = True

    def chat(self, question, model_file, system_prompt, no_think, n_gpu_layers, n_ctx,
             max_tokens, temperature, top_p, repeat_penalty, load_mode, auto_cpu_fallback, seed):
        if model_file == "NO_GGUF_FILES_IN_FOLDER":
            raise FileNotFoundError(f"No GGUF found in {_relative_model_hint()}.")
        model_path = NODE_DIR / model_file
        sys_prompt = system_prompt.strip() or self.DEFAULT_SYSTEM_PROMPT
        if _auto_no_think(model_file, no_think):
            sys_prompt += " /no_think"

        def _once(gpu_layers):
            llm, release_after = _get_llm(model_path, gpu_layers, n_ctx, seed, load_mode)
            try:
                out = llm.create_chat_completion(
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": question},
                    ],
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    repeat_penalty=repeat_penalty, seed=seed, stop=VERBOSE_STOPS,
                )
                return out["choices"][0]["message"]["content"].strip(), gpu_layers
            finally:
                if release_after:
                    _free_llm(llm)

        try:
            raw, used = _once(n_gpu_layers)
        except Exception as exc:
            if auto_cpu_fallback and n_gpu_layers != 0 and _is_oom_error(exc):
                _release_hot_model()
                raw, used = _once(0)
            else:
                raise
        response = _clean_chat_output(raw)
        thought = (
            f"Model: {model_file}\nGPU layers used: {'all' if used < 0 else used}\n"
            f"Context: {n_ctx}\n\nSystem Prompt:\n{sys_prompt}\n\nResponse:\n{response}"
        )
        return {"ui": {"text": [response]}, "result": (response, thought)}


# ================================================================
# Node 6: Prompt Locker
# ================================================================

class RebelsPromptLocker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text_input": ("STRING", {"forceInput": True}),
                "lock_in_prompt": ("BOOLEAN", {
                    "default": False,
                    "label_on": "LOCKED IN",
                    "label_off": "PAUSED",
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text_output",)
    FUNCTION = "execute"
    CATEGORY = "Rebel AI"
    OUTPUT_NODE = True

    def execute(self, text_input, lock_in_prompt):
        if not lock_in_prompt:
            raise ValueError("🛑 WORKFLOW PAUSED: Toggle 'lock_in_prompt' to LOCKED IN to pass the text through.")
        return {"ui": {"text": [text_input]}, "result": (text_input,)}


NODE_CLASS_MAPPINGS = {
    "RebelsPromptEnhancer": RebelsPromptEnhancer,
    "RebelsPromptEnhancerCustom": RebelsPromptEnhancerCustom,
    "RebelsPromptEnhancerModelHelper": RebelsPromptEnhancerModelHelper,
    "RebelsImageToPrompt": RebelsImageToPrompt,
    "RebelsLLMConsole": RebelsLLMConsole,
    "RebelsPromptLocker": RebelsPromptLocker,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RebelsPromptEnhancer": "🚀 Rebels Prompt Enhancer V3",
    "RebelsPromptEnhancerCustom": "🧪 Rebels Prompt Enhancer V3 (Custom GGUF)",
    "RebelsPromptEnhancerModelHelper": "📦 Rebels Prompt Enhancer Model Helper",
    "RebelsImageToPrompt": "👁️ Rebels Image to Prompt",
    "RebelsLLMConsole": "🧠 Rebels LLM Console",
    "RebelsPromptLocker": "🔒 Rebels Prompt Locker",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
