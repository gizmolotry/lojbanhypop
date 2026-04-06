import torch
import torch.nn.functional as F
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lojban_evolution.m9.engine import M9System1

def analyze_manifold(ckpt_path):
    print(f"\n--- M11 EMERGENT DICTIONARY AUTOPSY ---")
    device = "cpu"
    hidden_size = 896
    
    # 1. Load M11 Forge
    s1 = M9System1(hidden_size=hidden_size)
    sd = torch.load(ckpt_path, map_location=device)
    s1.load_state_dict(sd, strict=False)
    
    # 2. Extract Codebooks
    gismu = s1.manifold.gismu_emb.weight # [2000, 896]
    cmavo = s1.manifold.cmavo_emb.weight # [50, 896]
    
    # 3. Analyze Operator Diversity
    # Calculate pairwise cosine similarity within the gismu playground (100-1999)
    playground = gismu[100:1999]
    avg_sim = F.cosine_similarity(playground.unsqueeze(1), playground.unsqueeze(0), dim=-1).mean().item()
    print(f"Playground Self-Similarity: {avg_sim:.4f} (Goal: < 0.1 for high entropy)")
    
    # 4. Analyze "The Smudge" (Anchor Drift)
    day0_path = Path(__file__).parent.parent.parent / "src/lojban_evolution/m9/cmavo_anchors.json"
    if day0_path.exists():
        day0_cmavo = torch.tensor(json.loads(day0_path.read_text()))
        num_anchors = min(len(day0_cmavo), 50)
        drift = F.cosine_similarity(cmavo[:num_anchors], day0_cmavo[:num_anchors], dim=-1).mean().item()
        print(f"Logical Anchor Drift: {drift:.4f} (1.0 = Frozen, < 0.9 = High Smudge)")

    # 5. Analyze Hyper-Modulator Capacity
    # Sample 5 emergent gismu and see their modulation masks
    indices = [100, 500, 1000, 1500, 1900]
    print("\nModulation Variance (The Place Structure Test):")
    for idx in indices:
        vec = s1.manifold.get_vector(torch.tensor([idx]), token_type=0)
        mask = s1.hyper_mod(vec)
        print(f"  Gismu [{idx}] mask variance: {mask.var().item():.6f}")

if __name__ == "__main__":
    analyze_manifold("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt")
