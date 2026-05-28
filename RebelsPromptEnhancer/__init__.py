import torch
import gc
import os
import re
from llama_cpp import Llama

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
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "thought_process")
    FUNCTION = "enhance"
    CATEGORY = "Rebel AI"

    def enhance(self, raw_prompt, purpose, precision):
        # 1. STRICT TEMPLATE INSTRUCTIONS
        # We now force the model into a rigid XML-style structure so it can't get confused.
        template_rule = "\n\nYou MUST structure your ENTIRE response exactly like this:\n<think>\n[Your reasoning here]\n</think>\n<prompt>\n[Your final epic prompt here]\n</prompt>"
        
        system_instructions = {
            "Image Prompt (Photorealistic)": "You are a prompt expansion engine. Aggressively expand the raw idea into a highly detailed, cinematic, photorealistic 8k image prompt with dramatic lighting. INVENT missing details to make it epic. NEVER ask for clarification.",
            "Video Prompt (Cinematic)": "You are a video prompt engine. Aggressively expand the raw idea into a detailed video generation prompt focusing on fluid motion and cinematic camera movement. INVENT missing details to make it epic. NEVER ask for clarification.",
            "Editing (Inpainting/I2V)": "You are an image editor engine. Rewrite the edit instruction into a precise prompt describing the new transformed scene. INVENT missing details if needed. NEVER ask for clarification."
        }
        
        # Secretly combine the instructions, the template rule, and the user's prompt
        secret_injected_prompt = f"{system_instructions[purpose]}{template_rule}\n\nRAW IDEA TO EXPAND:\n\"{raw_prompt}\""

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
        
        print(f"🚀 [Rebel AI] Loading model: {model_filename}")
        
        llm = Llama(model_path=model_path, n_gpu_layers=-1, verbose=False, n_ctx=2048)
        
        output = llm.create_chat_completion(
            messages=[
                {"role": "user", "content": secret_injected_prompt}
            ],
            max_tokens=1024,
            temperature=0.6 
        )
        
        raw_output = output['choices'][0]['message']['content']
        
        # 2. BULLETPROOF EXTRACTION
        # Extract thoughts
        thought_match = re.search(r'<think>(.*?)</think>', raw_output, re.DOTALL | re.IGNORECASE)
        thoughts = thought_match.group(1).strip() if thought_match else "[Warning: Model missed <think> tags.]"
        
        # Extract the exact prompt inside the new <prompt> tags
        prompt_match = re.search(r'<prompt>(.*?)</prompt>', raw_output, re.DOTALL | re.IGNORECASE)
        if prompt_match:
            final_prompt = prompt_match.group(1).strip()
        else:
            # Fallback if it somehow forgets the <prompt> tag, grab everything after </think>
            final_prompt = raw_output[thought_match.end():].strip() if thought_match else raw_output.strip()

        # Final prefix scrubbing just in case it puts "8k image prompt:" INSIDE the <prompt> tags
        cleanup_prefixes = [
            "8k image prompt:", "Here is the", "The expanded prompt", 
            "Final prompt:", "Output:", "Prompt:"
        ]
        for prefix in cleanup_prefixes:
            final_prompt = re.sub(f"^{prefix}", "", final_prompt, flags=re.IGNORECASE).strip()

        final_prompt = final_prompt.strip('"\'')
        
        # Flush VRAM
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        print(f"🚀 [Rebel AI] Enhancement complete. Payload extracted perfectly.")
        return (final_prompt, thoughts)

NODE_CLASS_MAPPINGS = {"RebelsPromptEnhancer": RebelsPromptEnhancer}
NODE_DISPLAY_NAME_MAPPINGS = {"RebelsPromptEnhancer": "🚀 Rebels Prompt Enhancer"}
