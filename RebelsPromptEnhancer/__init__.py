import torch
import gc
import os

class RebelsPromptEnhancer:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "raw_prompt": ("STRING", {"multiline": True}),
                "purpose": (["Image Prompt (Photorealistic)", "Video Prompt (Cinematic)", "Editing (Inpainting/I2V)"],),
                "precision": (["Efficiency (Q4_K_M)", "High Fidelity (F16)"],),
                # FIX 1: The Seed Widget forces ComfyUI to re-run the node every time, preventing caching.
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}), 
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "thought_process")
    FUNCTION = "enhance"
    CATEGORY = "Rebel AI"

    # Don't forget to add 'seed' to the function arguments here!
    def enhance(self, raw_prompt, purpose, precision, seed):
        
        # FIX 2: Step-by-Step procedural instructions. This stops 1B models from overthinking.
        system_instructions = {
            "Image Prompt (Photorealistic)": "You are a prompt expansion machine.\nSTEP 1: Write EXACTLY ONE SENTENCE of brainstorming. Do not overthink. Do not waste tokens.\nSTEP 2: Type the exact word '[REBEL_OUTPUT]' on a new line.\nSTEP 3: Write a highly detailed, cinematic, photorealistic 8k image prompt. Invent missing epic details. Do not talk to the user.",
            "Video Prompt (Cinematic)": "You are a video prompt machine.\nSTEP 1: Write EXACTLY ONE SENTENCE of brainstorming. Do not overthink. Do not waste tokens.\nSTEP 2: Type the exact word '[REBEL_OUTPUT]' on a new line.\nSTEP 3: Write a detailed video generation prompt focusing on fluid motion and cinematic camera movement. Invent missing epic details. Do not talk to the user.",
            "Editing (Inpainting/I2V)": "You are an image editor machine.\nSTEP 1: Write EXACTLY ONE SENTENCE of brainstorming. Do not overthink. Do not waste tokens.\nSTEP 2: Type the exact word '[REBEL_OUTPUT]' on a new line.\nSTEP 3: Write a precise prompt describing the new transformed scene. Invent missing details. Do not talk to the user."
        }
        
        secret_injected_prompt = f"{system_instructions[purpose]}\n\nRAW IDEA TO EXPAND:\n\" {raw_prompt} \""

        node_dir = os.path.dirname(os.path.abspath(__file__))
        search_terms = {"Efficiency (Q4_K_M)": "q4_k_m", "High Fidelity (F16)": "f16"}
        term = search_terms.get(precision)
        
        try:
            files = [f for f in os.listdir(node_dir) if f.lower().endswith(".gguf") and term in f.lower()]
        except Exception as e:
            print(f"🚀 [Rebel AI] Directory read error: {e}")
            raise

        if not files:
            error_msg = f"Rebels Prompt Enhancer Error: Could not find '{term}' in {node_dir}"
            print(f"🚀 [Rebel AI] {error_msg}")
            raise FileNotFoundError(error_msg)
            
        model_filename = files[0]
        model_path = os.path.join(node_dir, model_filename)
        
        print(f"🚀 [Rebel AI] Loading model: {model_filename} with Seed: {seed}")
        
        from llama_cpp import Llama
        # Wire the ComfyUI seed directly into the LLM so it generates fresh ideas
        llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False, n_ctx=2048, seed=seed)
        
        output = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": secret_injected_prompt}
            ],
            max_tokens=2048,
            temperature=0.6 
        )
        
        raw_output = output['choices'][0]['message']['content']
        
        # Split logic via positional tokens
        if "[REBEL_OUTPUT]" in raw_output:
            parts = raw_output.split("[REBEL_OUTPUT]")
            thoughts = parts[0].strip()
            final_prompt = parts[-1].strip()
        elif "REBEL_OUTPUT" in raw_output:
            parts = raw_output.split("REBEL_OUTPUT")
            thoughts = parts[0].strip()
            final_prompt = parts[-1].strip()
        else:
            paragraphs = [p.strip() for p in raw_output.strip().split('\n') if p.strip()]
            if paragraphs:
                final_prompt = paragraphs[-1]
                thoughts = "\n".join(paragraphs[:-1]) + "\n\n[Warning: Splitting via fallback line rule]"
            else:
                final_prompt = raw_output.strip()
                thoughts = "[Warning: Structural generation failure]"

        # CLEANUP: Delete any hallucinated template trailing tags
        tags_to_destroy = ["[prompt]", "模板提示", "<prompt>", "</prompt>"]
        for tag in tags_to_destroy:
            final_prompt = final_prompt.replace(tag, "").strip()

        # Clean common prefixes
        cleanup_prefixes = [
            "8k image prompt:", "here is the prompt:", "expanded prompt:", 
            "final prompt:", "output:", "prompt:", "enhanced prompt:"
        ]
        for prefix in cleanup_prefixes:
            if final_prompt.lower().startswith(prefix):
                final_prompt = final_prompt[len(prefix):].strip()

        final_prompt = final_prompt.strip('"\' ')
        
        # Flush memory
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print(f"🚀 [Rebel AI] Process complete. Output cleaned and split.")
        return (final_prompt, thoughts)

NODE_CLASS_MAPPINGS = {"RebelsPromptEnhancer": RebelsPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"RebelsPromptEnhancer": "🚀 Rebels Prompt Enhancer"}
