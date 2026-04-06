from __future__ import annotations
import torch
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m7.engine import System1Coprocessor, InterleavedRouter
from lojban_evolution.experiment import generate_dataset, split_dataset

def probe_semantics(ckpt_path, base_model_path, adapter_path):
    print(f"\n--- M7 SEMANTIC PROBE: OPERATOR & ROLE AUDIT ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    
    # 2. Load Backbone & Resize
    backbone = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    backbone.resize_token_embeddings(len(tokenizer))
    
    # 3. Load Adapter (151701)
    model = PeftModel.from_pretrained(backbone, adapter_path).to(device)
    
    # 4. Handshake Token Expansion (151702)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<CALL_ADVISOR>"]})
    model.resize_token_embeddings(len(tokenizer))
    
    hidden_size = model.config.hidden_size
    s1 = System1Coprocessor(hidden_size=hidden_size).to(device)
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    s1.load_state_dict(checkpoint["s1_state"])
    s1.eval()
    
    ds = generate_dataset(size=100, seed=42, profile="diverse_v3")
    _, _, test_ds = split_dataset(ds)
    
    semantic_map = {} # Op -> List of (PromptType, Ptr1_Word, Ptr2_Word)
    
    print(f"Probing 20 samples for emergent positional meanings...")
    
    for item in test_ds[:20]:
        with torch.no_grad():
            prompt_ids = tokenizer(item.prompt, return_tensors="pt").input_ids.to(device)
            # Encode to get the latent thought
            out = model(input_ids=prompt_ids, output_hidden_states=True)
            latent = out.hidden_states[-1][:, -1, :]
            
            # System 1 Pass with Dynamic Masking
            prompt_len = prompt_ids.shape[1]
            _, op_idx = s1.codebook.quantize(latent)
            _, ptr_indices, _ = s1(latent, prompt_len=prompt_len)
            
            op = int(op_idx.item())
            p1 = int(ptr_indices[0, 0].item())
            p2 = int(ptr_indices[0, 1].item())
            
            # Map indices back to words for human audit
            tokens = tokenizer.convert_ids_to_tokens(prompt_ids[0])
            word1 = tokens[p1] if p1 < len(tokens) else "OOD"
            word2 = tokens[p2] if p2 < len(tokens) else "OOD"
            
            p_type = "Winograd" if "Winograd" in str(type(item)) or "refused" in item.prompt else "Logic"
            
            if op not in semantic_map: semantic_map[op] = []
            semantic_map[op].append((p_type, word1, word2))

    print(f"\n--- EMERGENT DICTIONARY (M7) ---")
    for op, occurrences in semantic_map.items():
        print(f"Operator G{op}: Used {len(occurrences)} times")
        for p_type, w1, w2 in occurrences[:3]:
            print(f"  [{p_type}] Role 1: {w1} | Role 2: {w2}")

if __name__ == "__main__":
    base = "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct"
    adapter = "runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5"
    ckpt = "archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR/m7_checkpoint.pt"
    probe_semantics(ckpt, base, adapter)
