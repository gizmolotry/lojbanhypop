from __future__ import annotations
import torch
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lojban_evolution.m9.engine import M9System1

def transplant_brain(ckpt_path, output_path):
    print(f"\n--- M9.7 BRAIN TRANSPLANT INITIATED ---")
    print(f"Source: {ckpt_path}")
    
    device = "cpu"
    hidden_size = 896
    
    # 1. Initialize empty M9.7 Symbiote
    s1 = M9System1(hidden_size=hidden_size)
    
    # 2. Load H5.3 Checkpoint
    h5_sd = torch.load(ckpt_path, map_location=device)
    
    # 3. Surgical Mapping
    with torch.no_grad():
        # A. Vocabulary Mapping (0-1999)
        h5_emb = h5_sd['codebook_state']['emb']
        s1.vocabulary.emb[:2000] = h5_emb
        print(f"Vocabulary Transplant: 2000 rows mapped.")
        
        # B. Operator Head Mapping
        h5_rel_w = h5_sd['arity_head_state']['head_rel.weight']
        h5_rel_b = h5_sd['arity_head_state']['head_rel.bias']
        s1.op_head.weight[:2000] = h5_rel_w
        s1.op_head.bias[:2000] = h5_rel_b
        print(f"Operator Head Transplant: SUCCESS.")
        
        # C. Pointer Head Mapping (x1, x2)
        # H5.3 used 32 slots, M9.1 uses 128 slots.
        # PHYSICAL LAW: We must match the statistical distribution of the legacy weights
        # to prevent uniform-probability smearing in the Softmax.
        for i in range(2):
            h5_w = h5_sd['arity_head_state'][f'head_var{i+1}.weight']
            h5_b = h5_sd['arity_head_state'][f'head_var{i+1}.bias']
            
            # Calculate mean and std of the trained weights
            w_mean, w_std = h5_w.mean(), h5_w.std()
            b_mean, b_std = h5_b.mean(), h5_b.std()
            
            # Initialize with matched Gaussian noise
            new_w = torch.randn(128, 896) * w_std + w_mean
            new_b = torch.randn(128) * b_std + b_mean
            
            # Surgically inject legacy weights into the top 32 slots
            new_w[:32, :] = h5_w
            new_b[:32] = h5_b
            
            s1.x_heads[i].weight.data = new_w
            s1.x_heads[i].bias.data = new_b
            
        print(f"Positional Heads (x1, x2) Transplant: SUCCESS (Statistically Padded 32 to 128).")

    # 4. Save Transplanted Checkpoint
    torch.save(s1.state_dict(), output_path)
    print(f"\nTransplant Complete. Saved to {output_path}")

if __name__ == "__main__":
    src = "runs/i_series/20260302_172603/h5_checkpoint.pt"
    dst = "archive/results/m9/active/RESULTS_M9_PHASE3/m9_s1_dark_transplant.pt"
    transplant_brain(src, dst)
