import torch
import torch.nn as nn
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.lojban_evolution.m9.engine import M9System1

def perform_surgery(ckpt_path: Path, output_path: Path, max_slots: int = 9):
    print(f"Surgery: Loading legacy checkpoint from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu")
    
    # Identify the target layer (final linear layer in HyperModulator)
    # Architecture in engine.py:
    # self.net = nn.Sequential(
    #     nn.Linear(hidden_size, hidden_size), # Index 0
    #     nn.ReLU(),                           # Index 1
    #     nn.Linear(hidden_size, hidden_size * max_slots), # Index 2
    #     nn.Sigmoid()                         # Index 3
    # )
    
    weight_key = "hyper_mod.net.2.weight"
    bias_key = "hyper_mod.net.2.bias"
    
    if weight_key in sd:
        old_weight = sd[weight_key] # [H, H]
        old_bias = sd[bias_key]     # [H]
        
        h = old_weight.shape[1]
        print(f"Surgery: Found legacy layer with output H={h}. Expanding to H*max_slots={h*max_slots}...")
        
        # 1. Expand Weight [9*H, H]
        # Repeat the weights max_slots times
        new_weight = old_weight.repeat(max_slots, 1)
        # Add small Gaussian noise to break symmetry
        new_weight += torch.randn_like(new_weight) * 0.01
        
        # 2. Expand Bias [9*H]
        new_bias = old_bias.repeat(max_slots)
        new_bias += torch.randn_like(new_bias) * 0.01
        
        sd[weight_key] = new_weight
        sd[bias_key] = new_bias
        
        # 3. Handle x_heads vs cmavo_head transition
        # The new architecture removed x_heads (ModuleList) and added cmavo_head.
        # We delete the orphaned x_heads weights to avoid strict=True failures.
        keys_to_delete = [k for k in sd.keys() if "x_heads" in k]
        for k in keys_to_delete:
            del sd[k]
        
        torch.save(sd, output_path)
        print(f"Surgery: SUCCESS. Saved to {output_path}")
    else:
        print(f"Error: {weight_key} not found in checkpoint. Available keys: {list(sd.keys())[:10]}")

if __name__ == "__main__":
    legacy_path = Path("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt")
    target_path = Path("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_v2_geometry.pt")
    
    if legacy_path.exists():
        perform_surgery(legacy_path, target_path)
    else:
        print(f"Legacy checkpoint {legacy_path} not found. Skipping surgery.")
