from __future__ import annotations

import argparse
import torch
from pathlib import Path
import sys
import json

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m9.engine import M9System1

def sync_m11_manifold(base_model_path, adapter_path, forge_ckpt):
    print(f"\n--- M11 MANIFOLD SYNC INITIATED ---")
    device = "cpu"
    
    # 1. Load Tokenizer & Expand to 2256
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # We maintain the 2256-token taxonomy for physical compatibility
    num_new_tokens = 2256
    tokenizer.add_tokens([f"<loj_{i}>" for i in range(num_new_tokens)])
    
    # 2. Load Forge Weights (M11 Architecture)
    print(f"Loading M11 Forge weights from {forge_ckpt}...")
    s1 = M9System1(hidden_size=896)
    s1.load_state_dict(torch.load(forge_ckpt, map_location=device))
    
    # 3. Pre-Blend Provenance Vectors
    # E_final = E_dict(ID) + E_type(Type)
    blended_vectors = torch.zeros(num_new_tokens, 896)
    
    with torch.no_grad():
        # A. Gismu (0-1999)
        gismu_flavor = s1.manifold.type_emb(torch.tensor([0]))
        blended_vectors[:2000] = s1.manifold.gismu_emb.weight + gismu_flavor
        
        # B. Judri (2000-2127) - We map them to the 2000+ range in the taxonomy
        judri_flavor = s1.manifold.type_emb(torch.tensor([2]))
        blended_vectors[2000:2128] = s1.manifold.judri_emb.weight + judri_flavor
        
        print("M11: 2128 provenance-aware vectors pre-blended.")

    # 4. Load Backbone & Adapter
    backbone = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="cpu")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, adapter_path)
    
    # 5. Physical Hard-Copy
    with torch.no_grad():
        # The Lojban partition starts at 151701
        model.get_input_embeddings().weight[-2256:] = blended_vectors
        model.get_output_embeddings().weight[-2256:] = blended_vectors
        print(f"Physical Hard-Copy: SUCCESS (Blended M11 Manifold Synced).")

    # 6. Save Synced Artifact
    output_dir = Path("archive/results/m9/active/RESULTS_M9_SYNCED/synced_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"M11 Symbiote Synced. Saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--forge-ckpt", default="archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt")
    args = parser.parse_args()
    
    sync_m11_manifold(args.base_model, args.adapter, args.forge_ckpt)
