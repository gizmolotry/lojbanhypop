import torch
import torch.nn.functional as F
import json
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

def health_check():
    ckpt_path = Path("archive/results/m9/active/RESULTS_M9_PHASE3/m9_s1_final.pt")
    day0_gismu_path = Path("src/lojban_evolution/m9/gismu_anchors.json")
    day0_cmavo_path = Path("src/lojban_evolution/m9/cmavo_anchors.json")
    day0_role_path = Path("src/lojban_evolution/m9/role_anchors.json")

    print(f"\n--- M9.1 LATENT HEALTH CHECK INITIATED ---")
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint {ckpt_path} not found.")
        return

    # 1. Load Trained Weights
    state_dict = torch.load(ckpt_path, map_location="cpu")
    emb = state_dict["vocabulary.emb"] # [2256, 896]
    
    # --- Probe 1: The Positional Cables (Ptr) ---
    print(f"\nProbe 1: The Positional Cables (Indices 2000-2255)")
    ptr_block = emb[2000:2256] # [256, 896]
    ptr_norm = F.normalize(ptr_block, p=2, dim=1)
    cos_sim_matrix = torch.mm(ptr_norm, ptr_norm.t())
    
    avg_sim = cos_sim_matrix.mean().item()
    diag_mask = torch.eye(256, dtype=torch.bool)
    off_diag_sim = cos_sim_matrix[~diag_mask].mean().item()
    max_off_diag = cos_sim_matrix[~diag_mask].max().item()
    
    print(f"  - Average Pairwise Similarity: {avg_sim:.4f}")
    print(f"  - Off-Diagonal Mean:           {off_diag_sim:.4f} (Target: < 0.3)")
    print(f"  - Off-Diagonal Max:            {max_off_diag:.4f} (Danger: > 0.9)")
    if off_diag_sim < 0.3:
        print("  - Status: HEALTHY (Orthogonal Routing Cables Verified)")
    else:
        print("  - Status: COLLAPSED (Positional Mode Collapse Detected)")

    # --- Probe 2: The gismu Anchor Washout ---
    print(f"\nProbe 2: The gismu Anchor Washout (Indices 0-49)")
    if day0_gismu_path.exists():
        day0_gismu = torch.tensor(json.loads(day0_gismu_path.read_text()))
        final_gismu = emb[:50]
        
        dist = torch.norm(final_gismu - day0_gismu, p=2, dim=1).mean().item()
        print(f"  - Mean L2 Shift from Day-0: {dist:.4f}")
        if dist < 0.1:
            print("  - Status: HEALTHY (AdaHessian Protected Anchors)")
        else:
            print("  - Status: WASHED OUT (Anchor Drift Detected)")
    else:
        print("  - Status: SKIPPED (Day-0 gismu anchors missing)")

    # --- Probe 3: Cluster Analysis (PCA) ---
    print(f"\nProbe 3: The cmavo vs. Emergent Dark Matter")
    
    # Prepare data for PCA
    data = emb.detach().numpy()
    pca = PCA(n_components=2)
    proj = pca.fit_transform(data)
    
    # Clusters
    gismu_p = proj[:50]
    cmavo_p = proj[50:100]
    playground_p = proj[100:2000]
    ptr_p = proj[2000:]
    
    # Calculate Centroids
    g_center = gismu_p.mean(axis=0)
    c_center = cmavo_p.mean(axis=0)
    p_center = playground_p.mean(axis=0)
    
    dist_g_c = np.linalg.norm(g_center - c_center)
    dist_c_p = np.linalg.norm(c_center - p_center)
    
    print(f"  - gismu-to-cmavo Separation:    {dist_g_c:.4f}")
    print(f"  - cmavo-to-Playground Separation: {dist_c_p:.4f}")
    
    # Detect Operator [99] shift
    op99_proj = proj[99]
    dist_99_noise = np.linalg.norm(op99_proj - p_center)
    print(f"  - Operator [99] Escape Velocity: {dist_99_noise:.4f} (Distance from Noise Center)")
    
    if dist_g_c > 0.5 and dist_99_noise > 0.1:
        print("  - Status: HEALTHY (Geometric Partitioning SUCCESS)")
    else:
        print("  - Status: MASHED (Low Topological Diversity Detected)")

if __name__ == "__main__":
    health_check()
