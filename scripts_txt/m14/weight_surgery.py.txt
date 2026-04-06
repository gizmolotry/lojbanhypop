import torch
import torch.nn as nn
from pathlib import Path
import sys

def perform_surgery(input_path: str, output_path: str):
    print(f"\n--- WEIGHT SURGERY: INFALTION PROTOCOL ---")
    print(f"Source: {input_path}")
    
    device = "cpu"
    sd = torch.load(input_path, map_location=device)
    
    # 1. Inflate HyperModulator bottleneck (net.0)
    # Checkpoint: [224, 896] -> Model: [896, 896]
    if "hyper_mod.net.0.weight" in sd:
        print("  Inflating bottleneck net.0 (224 -> 896)...")
        w0 = sd["hyper_mod.net.0.weight"]
        b0 = sd["hyper_mod.net.0.bias"]
        sd["hyper_mod.net.0.weight"] = w0.repeat(4, 1) + torch.randn(896, 896) * 0.01
        sd["hyper_mod.net.0.bias"] = b0.repeat(4) + torch.randn(896) * 0.01

    # 2. Inflate HyperModulator output (net.2)
    # Checkpoint: [896, 224] -> Model: [8064, 896] (9 slots * 896)
    if "hyper_mod.net.2.weight" in sd:
        print("  Inflating Role Basis net.2 (896 -> 8064)...")
        w2 = sd["hyper_mod.net.2.weight"]
        b2 = sd["hyper_mod.net.2.bias"]
        
        # Expand input dim first (224 -> 896)
        w2_expanded_in = w2.repeat(1, 4)
        # Expand output dim for 9 slots
        sd["hyper_mod.net.2.weight"] = w2_expanded_in.repeat(9, 1) + torch.randn(8064, 896) * 0.01
        sd["hyper_mod.net.2.bias"] = b2.repeat(9) + torch.randn(8064) * 0.01

    torch.save(sd, output_path)
    print(f"Surgery Complete. Saved to {output_path}\n")

if __name__ == "__main__":
    src = "archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt"
    dst = "RESULTS_M9_PHASE3/m11_s1_v2_geometry.pt"
    Path("RESULTS_M9_PHASE3").mkdir(parents=True, exist_ok=True)
    perform_surgery(src, dst)
