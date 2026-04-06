from __future__ import annotations
import json
import argparse
import torch
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lojban_evolution.experiment import generate_dataset, split_dataset
from src.lojban_evolution.m9.engine import M9System1
from transformers import AutoModelForCausalLM, AutoTokenizer

def prepare_m14_5_corpus(size: int, output_dir: str):
    print(f"Generating Unified M14.5 Corpus (Size: {size}, Profile: diverse_v3)...")
    
    # 1. Load M11 Forge to generate target logic traces
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 896
    s1 = M9System1(hidden_size=hidden_size).to(device)
    m11_ckpt = "archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt"
    
    # RESILIENT LOADING: strict=False to accommodate Rg architecture
    state_dict = torch.load(m11_ckpt, map_location=device)
    
    # WEIGHT SURGERY: Deep Manifold Inflation
    if "hyper_mod.net.0.weight" in state_dict:
        print("M14: Inflating HyperModulator bottleneck (224 -> 896).")
        # Inflate bottleneck weights (net.0)
        old_w0 = state_dict["hyper_mod.net.0.weight"]
        old_b0 = state_dict["hyper_mod.net.0.bias"]
        state_dict["hyper_mod.net.0.weight"] = old_w0.repeat(4, 1) # 224 * 4 = 896
        state_dict["hyper_mod.net.0.bias"] = old_b0.repeat(4)
        
        # Inflate output weights (net.2)
        # Old: [896, 224], New: [4480, 896]
        # We handle this by repeating the input dimension (224 -> 896) 
        # and the output dimension (896 -> 4480)
        old_w2 = state_dict["hyper_mod.net.2.weight"]
        old_b2 = state_dict["hyper_mod.net.2.bias"]
        
        # 1. Repeat input dimension
        w2_inflated_in = old_w2.repeat(1, 4) # [896, 896]
        # 2. Repeat output dimension for the 9 roles (Mandated 10-slot topology)
        state_dict["hyper_mod.net.2.weight"] = w2_inflated_in.repeat(9, 1) # [8064, 896]
        state_dict["hyper_mod.net.2.bias"] = old_b2.repeat(9) # [8064]

    s1.load_state_dict(state_dict, strict=False)
    s1.eval()
    
    # 2. Load backbone for hidden state extraction
    # Surgical Load to bypass PEFT auto-resize crash
    base_model = "archive/results/m9/active/RESULTS_M9_SYNCED/synced_model"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    from transformers import AutoConfig
    # Use original Qwen path for config to ensure model_type is found
    base_qwen_path = "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct"
    config = AutoConfig.from_pretrained(base_qwen_path)
    backbone = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # Force alignment with archival weight dimensions
    backbone.resize_token_embeddings(153921)
    
    # Load synced weights manually
    weights_path = Path(base_model) / "adapter_model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        state_dict = load_file(str(weights_path))
        clean_sd = {k.replace("base_model.model.", ""): v for k, v in state_dict.items()}
        backbone.load_state_dict(clean_sd, strict=False)
    
    backbone = backbone.to(device)
    backbone.eval()
    
    # Align dtypes
    s1 = s1.to(device=device, dtype=backbone.dtype)

    # 3. Generate raw puzzles
    ds = generate_dataset(size=size, seed=42, profile="diverse_v3")
    train, val, test = split_dataset(ds)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    def process_split(data, name):
        p = out_path / f"m14_5_unified_{name}.jsonl"
        count = 0
        with p.open("w", encoding="utf-8") as f:
            for item in data:
                # Extract S2 hidden state for the forge
                prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = backbone(**inputs, output_hidden_states=True)
                    h_eng = out.hidden_states[-1][:, -1, :]
                    
                    # Extract prompt embeddings for dynamic PointerBind
                    prompt_embs = backbone.get_input_embeddings()(inputs.input_ids)
                    
                    # Generate logical AST from Forge using the new Rg API
                    op_vec, x_probs, op_idx = s1.build_graph(h_eng, prompt_embs)
                    
                    # Flatten into deterministic 1D string for runway
                    tokens = [f"<loj_{op_idx.item()}>"]
                    for i in range(9):
                        ptr_idx = torch.argmax(x_probs[0, i], dim=-1).item()
                        tokens.append(f"<loj_{2000 + ptr_idx}>")
                    logic_string = " ".join(tokens)

                row = {
                    "prompt": item.prompt,
                    "answer": item.answer,
                    "target_logic": logic_string,
                    "difficulty": item.difficulty
                }
                f.write(json.dumps(row) + "\n")
                count += 1
        print(f"  Wrote {count} samples to {p}")

    process_split(train, "train")
    process_split(val, "val")
    process_split(test, "test")

if __name__ == "__main__":
    prepare_m14_5_corpus(2000, "artifacts/datasets/m14_5_unified")
