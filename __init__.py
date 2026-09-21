import base64
import gc
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.request
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
RUNTIME_LOCK = threading.Lock()

# Keep-at-most-one hot model for Fast Iteration mode.
_HOT_MODEL = {"key": None, "llm": None}

# Once the installed in-process llama-cpp-python backend proves incompatible during
# this ComfyUI session, skip it for all subsequent prompt-enhancer runs and use
# the private portable runtime directly. This prevents repeated native crash traces.
_NATIVE_BACKEND_DISABLED = False


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
    "Long": "Target roughly 180-320 words or an equivalently rich tag list.",
    "Max Detail": "Use as much concrete detail as useful without redundancy; avoid filler and repeated quality buzzwords.",
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
    "Pixar / 3D Animation": "Polished stylized 3D animation with expressive proportions, clean CG materials and warm cinematic lighting.",
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


def _portable_runtime_manifest_path():
    return RUNTIME_DIR / "runtime.json"


def _find_portable_llama_binary(binary_name):
    if not RUNTIME_DIR.exists():
        return None
    exe_name = binary_name + (".exe" if os.name == "nt" else "")
    direct = RUNTIME_DIR / exe_name
    if direct.is_file():
        return direct
    for path in RUNTIME_DIR.rglob(exe_name):
        if path.is_file():
            return path
    return None


def _find_portable_llama_cli():
    return _find_portable_llama_binary("llama-cli")


def _find_portable_llama_completion():
    return _find_portable_llama_binary("llama-completion")


# Pin the private fallback runtime instead of discovering GitHub's latest release at runtime.
# This keeps RebelsPromptEnhancer reproducible and avoids breakage when GitHub release/API
# naming or "latest" semantics change. It does NOT replace llama-cpp-python.
PORTABLE_LLAMA_TAG = "b10964"
PORTABLE_LLAMA_ASSET = "llama-b10964-bin-win-cpu-x64.zip"
PORTABLE_LLAMA_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/"
    f"{PORTABLE_LLAMA_TAG}/{PORTABLE_LLAMA_ASSET}"
)


def _portable_windows_cpu_asset():
    """Return the pinned official llama.cpp Windows x64 CPU runtime."""
    return PORTABLE_LLAMA_TAG, PORTABLE_LLAMA_ASSET, PORTABLE_LLAMA_URL


def _ensure_portable_llama_cli():
    """Install a private official llama.cpp CLI under this node only.

    This does not modify ComfyUI's Python packages or the user's llama-cpp-python install.
    It is used only when the in-process native DLL raises STATUS_ILLEGAL_INSTRUCTION.
    """
    existing = _find_portable_llama_cli()
    if existing is not None:
        return existing
    if os.name != "nt":
        raise RuntimeError(
            "Portable llama.cpp auto-fallback currently supports Windows x64 only. "
            "The installed llama-cpp-python backend failed before inference."
        )

    with RUNTIME_LOCK:
        existing = _find_portable_llama_cli()
        if existing is not None:
            return existing
        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        tag, asset_name, asset_url = _portable_windows_cpu_asset()
        if not asset_url:
            raise RuntimeError("Could not resolve the official llama.cpp Windows download URL.")

        archive = RUNTIME_DIR / asset_name
        _download_with_resume(asset_url, archive, f"portable llama.cpp {tag}")
        extract_dir = RUNTIME_DIR / "extracting"
        shutil.rmtree(extract_dir, ignore_errors=True)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(extract_dir)

        # Keep the whole release together because llama.cpp executables depend on sibling DLLs.
        for child in list(extract_dir.iterdir()):
            dest = RUNTIME_DIR / child.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink(missing_ok=True)
            shutil.move(str(child), str(dest))
        shutil.rmtree(extract_dir, ignore_errors=True)
        archive.unlink(missing_ok=True)
        _portable_runtime_manifest_path().write_text(
            json.dumps({"tag": tag, "asset": asset_name}, indent=2), encoding="utf-8"
        )

        exe = _find_portable_llama_cli()
        if exe is None:
            raise RuntimeError("Portable llama.cpp downloaded, but llama-cli.exe was not found in the archive.")
        return exe


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
):
    """Safe out-of-process fallback for broken/incompatible in-process llama native DLLs."""
    # Ensure the official runtime is present, then use llama-completion for scripted
    # generation. llama-cli auto-enables conversation mode for chat-template models,
    # which can inject Qwen special tokens before a JSON grammar and break constrained
    # generation. llama-completion is the non-interactive completion frontend.
    _ensure_portable_llama_cli()
    exe = _find_portable_llama_completion()
    if exe is None:
        raise RuntimeError(
            "Portable llama.cpp runtime is missing llama-completion.exe; reinstall the node runtime."
        )
    prompt = (
        sys_prompt.strip()
        + "\n\nUSER REQUEST:\n"
        + raw_prompt.strip()
        + "\n\nReturn ONLY the requested JSON object. Do not use markdown fences or commentary."
    )
    schema_json = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    cmd = [
        str(exe),
        "-m", str(Path(model_path).resolve()),
        "-p", prompt,
        "-c", str(int(n_ctx)),
        "-n", str(int(max_tokens)),
        "--temp", str(float(temperature)),
        "--top-p", str(float(top_p)),
        "--repeat-penalty", str(float(repeat_penalty)),
        "--seed", str(int(seed) & 0xFFFFFFFF),
        "-ngl", "0",
        "--no-repack",
        "--no-warmup",
        "--no-conversation",
        "--no-display-prompt",
        "-j", schema_json,
    ]
    env = os.environ.copy()
    # The command line already forces --no-repack. Remove any inherited
    # LLAMA_ARG_REPACK setting so llama.cpp does not emit an override warning.
    # --no-conversation is important for Qwen GGUFs: llama.cpp otherwise auto-enables
    # conversation mode from the embedded chat template and waits for another turn.
    env.pop("LLAMA_ARG_REPACK", None)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(exe.parent),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=300,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Portable llama.cpp fallback timed out after 5 minutes.") from exc

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if proc.returncode != 0 or not stdout:
        detail = stderr[-2000:] if stderr else f"exit code {proc.returncode}"
        raise RuntimeError(f"Portable llama.cpp fallback failed: {detail}")
    return stdout



# ================================================================
# Structured rewrite engine
# ================================================================


def _schema_for_rewrite(variant_count, generate_negative):
    props = {
        "final_prompt": {"type": "string"},
        "variants": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": variant_count,
            "maxItems": variant_count,
        },
    }
    required = ["final_prompt", "variants"]
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
        parts.append(AESTHETIC_DESCRIPTORS[aesthetic])
    if variant_count > 1:
        parts.append(
            f"Return exactly {variant_count} meaningfully different prompt variants. Variant 1 should be the best balanced default; "
            "other variants may shift composition, camera, or atmosphere only where the user left those unspecified."
        )
    else:
        parts.append("Return one rewritten prompt and also place that same prompt as the sole item in the variants array.")
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
    parts.append("Output valid JSON only, matching the requested schema. Do not include markdown or reasoning.")
    text = " ".join(parts)
    if append_no_think:
        text += " /no_think"
    return text


def _decode_json_object_from_output(raw_output):
    """Decode the first JSON object even when llama.cpp appends markers like [end of text]."""
    text = (raw_output or "").strip()
    if not text:
        raise ValueError("empty model output")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    decoder = json.JSONDecoder()
    # llama-completion may append '[end of text]' or other status text after a valid object.
    # Walk candidate object starts and use raw_decode so trailing text does not poison parsing.
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    raise ValueError("no JSON object found in model output")


def _extract_json_payload(raw_output, raw_prompt, variant_count, generate_negative):
    try:
        payload = _decode_json_object_from_output(raw_output)
        final_prompt = str(payload.get("final_prompt", "")).strip()
        variants = payload.get("variants", [])
        if not isinstance(variants, list):
            variants = []
        variants = [str(v).strip() for v in variants if str(v).strip()]
        negative = str(payload.get("negative_prompt", "")).strip() if generate_negative else ""

        if not final_prompt and variants:
            final_prompt = variants[0]
        if final_prompt and not variants:
            variants = [final_prompt]
        while variants and len(variants) < variant_count:
            variants.append(variants[-1])
        variants = variants[:variant_count]
        if not final_prompt:
            raise ValueError("empty final_prompt")
        return final_prompt, negative, variants, "structured"
    except Exception:
        cleaned = _legacy_clean_output(raw_output, raw_prompt)
        return cleaned, "", [cleaned] if cleaned else [], "legacy-fallback"


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
    # Sparse prompts selected for expansion should not come back effectively unchanged.
    if detail_strength in {"Expand", "Highly Detailed", "Creative"} and len(raw_words) <= 12:
        if len(final_words) <= max(len(raw_words) + 5, 12):
            return True
    return False


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
):
    global _NATIVE_BACKEND_DISABLED
    schema = _schema_for_rewrite(variant_count, generate_negative)

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
        print(
            "[RebelsPromptEnhancer] Installed llama-cpp-python raised STATUS_ILLEGAL_INSTRUCTION. "
            "Portable llama.cpp is now locked in for RebelsPromptEnhancer for the rest of this ComfyUI session; "
            "global Python packages are unchanged."
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
        )
        return raw, 0, "portable llama.cpp fallback"

    raise primary_exc



def _max_tokens_for_length(length_target, base=None):
    defaults = {"Short": 240, "Medium": 480, "Long": 800, "Max Detail": 1200}
    value = defaults[length_target]
    if base is not None:
        value = min(max(int(base), 64), 4096)
    return value


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

    def enhance(self, raw_prompt, purpose, model_format, aesthetic, precision,
                detail_strength, length_target, preserve_user_details, variant_count,
                negative_prompt, load_mode, auto_cpu_fallback, auto_download_missing,
                seed, lock_in):
        variant_count = int(variant_count)
        generate_negative = negative_prompt == "Generate when useful"
        cache_payload = {
            "engine_revision": "3.2.6",
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
        max_tokens = _max_tokens_for_length(length_target)
        raw_output, used_gpu_layers, backend_used = _run_rewrite(
            model_path=model_path,
            raw_prompt=raw_prompt,
            sys_prompt=sys_prompt,
            n_gpu_layers=-1,
            n_ctx=4096,
            max_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.12,
            seed=seed,
            variant_count=variant_count,
            generate_negative=generate_negative,
            load_mode=load_mode,
            auto_cpu_fallback=auto_cpu_fallback,
        )
        final_prompt, negative, variants, parse_mode = _extract_json_payload(
            raw_output, raw_prompt, variant_count, generate_negative
        )

        quality_retry = False
        if _rewrite_is_too_literal(raw_prompt, final_prompt, detail_strength):
            quality_retry = True
            repair_sys_prompt = sys_prompt + (
                " QUALITY RETRY: The previous draft was rejected because it was essentially a copy of the source. "
                "Substantially enhance the prompt now. Keep every explicit user fact, but add concrete compatible details "
                "for subject appearance, pose/action, environment, composition, camera/lens, lighting, materials/textures, "
                "mood, and spatial relationships as appropriate. Do not return the source text unchanged."
            )
            raw_output, used_gpu_layers, backend_used = _run_rewrite(
                model_path=model_path,
                raw_prompt=raw_prompt,
                sys_prompt=repair_sys_prompt,
                n_gpu_layers=-1,
                n_ctx=4096,
                max_tokens=max_tokens,
                temperature=0.75,
                top_p=0.92,
                repeat_penalty=1.08,
                seed=seed,
                variant_count=variant_count,
                generate_negative=generate_negative,
                load_mode=load_mode,
                auto_cpu_fallback=auto_cpu_fallback,
                force_portable=(backend_used == "portable llama.cpp fallback"),
            )
            final_prompt, negative, variants, parse_mode = _extract_json_payload(
                raw_output, raw_prompt, variant_count, generate_negative
            )

        if not final_prompt:
            raise RuntimeError("Prompt enhancer returned an empty result.")

        meta = (
            f"Model: {model_path.name}\nPrecision: {precision}\nPurpose: {purpose}\n"
            f"Format: {model_format}\nAesthetic: {aesthetic}\nDetail: {detail_strength}\n"
            f"Length: {length_target}\nVariants: {variant_count}\nStructured parse: {parse_mode}\n"
            f"Quality retry: {'yes' if quality_retry else 'no'}\n"
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
            "engine_revision": "3.2.6",
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

        effective_max = _max_tokens_for_length(length_target, max_tokens)
        raw_output, used_gpu_layers, backend_used = _run_rewrite(
            model_path=model_path,
            raw_prompt=raw_prompt,
            sys_prompt=sys_prompt,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            max_tokens=effective_max,
            temperature=temperature,
            top_p=top_p,
            repeat_penalty=repeat_penalty,
            seed=seed,
            variant_count=variant_count,
            generate_negative=generate_negative,
            load_mode=load_mode,
            auto_cpu_fallback=auto_cpu_fallback,
        )
        final_prompt, negative, variants, parse_mode = _extract_json_payload(
            raw_output, raw_prompt, variant_count, generate_negative
        )
        if not final_prompt:
            raise RuntimeError("Custom enhancer returned an empty result.")

        gpu_label = "all" if used_gpu_layers < 0 else ("CPU only" if used_gpu_layers == 0 else str(used_gpu_layers))
        meta = (
            f"Model: {model_file}\nDetected family: {detected_family}\n/no_think: {append_no_think}\n"
            f"Backend: {backend_used}\nGPU layers used: {gpu_label}\nContext: {n_ctx}\nMax tokens: {effective_max}\n"
            f"Temperature: {temperature}\ntop_p: {top_p}\nrepeat_penalty: {repeat_penalty}\n"
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
