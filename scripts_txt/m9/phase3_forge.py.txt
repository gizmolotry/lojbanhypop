from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import zmq
import json
from pathlib import Path
import sys
import math

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lojban_evolution.m9.engine import M9System1, InfoNCEForge, M9HardNegativeGenerator
from src.lojban_evolution.m9.optimizer import AdaHessian

def main():
    parser = argparse.ArgumentParser(description="M11 Forge: Hyper-Modulation & Differential Smudging.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--lr_emergent", type=float, default=1e-3)
    parser.add_argument("--lr_smudge", type=float, default=1e-5)
    parser.add_argument("--load-ckpt", type=Path, default=Path("archive/results/m9/active/RESULTS_M9_PHASE3/m9_s1_final.pt"))
    args = parser.parse_args()

    device = "cpu" 
    hidden_size = 896
    
    context = zmq.Context()
    pull_socket = context.socket(zmq.PULL); pull_socket.bind(f"tcp://*:{args.port}")
    push_socket = context.socket(zmq.PUSH); push_socket.bind(f"tcp://*:{args.port + 1}")
    
    s1 = M9System1(hidden_size=hidden_size).to(device)
    
    # 1. PARAMETER GROUPS: M11 Differential Smudging
    # Group A: Emergent Playground (Fast Evolution)
    emergent_params = (
        list(s1.manifold.gismu_emb.parameters()) + 
        list(s1.manifold.judri_emb.parameters()) + 
        list(s1.op_head.parameters()) + 
        list(s1.hyper_mod.parameters())
    )
    
    # Group B: Human Seeds (Slow constrained smudging)
    smudge_params = (
        list(s1.manifold.cmavo_emb.parameters()) + 
        list(s1.manifold.type_emb.parameters())
    )
    
    forge = InfoNCEForge(hidden_size=hidden_size).to(device)
    neg_gen = M9HardNegativeGenerator(s1.manifold)
    
    opt = AdaHessian([
        {"params": emergent_params, "lr": args.lr_emergent},
        {"params": smudge_params, "lr": args.lr_smudge},
        {"params": list(forge.parameters()), "lr": args.lr_emergent}
    ], lr=args.lr_emergent)

    if args.load_ckpt.exists():
        print(f"M11: Loading base weights from {args.load_ckpt}...")
        # Note: M11 architecture is different, we primarily load the gismu/heads 
        # and allow the hyper-modulator to initialize from scratch.
        sd = torch.load(args.load_ckpt, map_location=device)
        with torch.no_grad():
            if 'vocabulary.emb' in sd:
                # Map legacy monolithic [2256, 896] to M11 separated codebooks
                legacy_emb = sd['vocabulary.emb']
                s1.manifold.gismu_emb.weight[:2000] = legacy_emb[:2000]
                # Reset cmavo from original anchors to ensure pure Day-0 seeds
                gismu_path = Path(__file__).parent.parent.parent / "src/lojban_evolution/m9/gismu_anchors.json"
                if gismu_path.exists():
                    anchors = torch.tensor(json.loads(gismu_path.read_text()))
                    s1.manifold.gismu_emb.weight[:min(len(anchors), 50)] = anchors[:50]
            
            # Map Heads
            if 'op_head.weight' in sd:
                s1.op_head.weight[:sd['op_head.weight'].shape[0]] = sd['op_head.weight']
                s1.op_head.bias[:sd['op_head.bias'].shape[0]] = sd['op_head.bias']
        print("M11: Manifold migration complete.")

    print(f"\n--- M11 FORGE ACTIVE (HYPER-MODULATION + DIFFERENTIAL) ---")

    step = 0
    try:
        while True:
            payload = pull_socket.recv_json()
            premise = torch.tensor(payload["premise"], device=device).view(1, -1)
            prompt_len = payload["prompt_len"]
            trace = payload.get("trace", None)
            tier = payload.get("tier", "inference")
            
            opt.zero_grad()
            opt.zero_hessian()
            
            op_vector, x_probs, op_idx = s1.build_graph(premise, prompt_len)
            
            tokens = [f"<loj_{op_idx.item()}>"]
            for i in range(s1.num_x_slots):
                ptr_idx = torch.argmax(x_probs[0, i], dim=-1).item()
                # Physically map to the 2000+ pointer range in the synced taxonomy
                tokens.append(f"<loj_{2000 + ptr_idx}>")
            
            push_socket.send_json({"logic_string": " ".join(tokens)})
            
            if trace is not None:
                from scripts.m9.phase2_forge import TRACE_TO_ANCHOR
                base_idx = TRACE_TO_ANCHOR.get(trace[0], 49)
                target_op_idx = (base_idx % 2000)
                
                pos_hyp = s1.manifold.get_vector(torch.tensor([target_op_idx], device=device), token_type=0)
                neg_hyps = neg_gen.generate(premise, op_vector, x_probs, op_idx)
                
                loss = forge(premise, pos_hyp, neg_hyps)
                loss.backward(create_graph=True)
                opt.step()
                
                step += 1
                if step % 10 == 0:
                    print(f"Step {step} - Forge Loss: {loss.item():.4f} | Avg Scale: {s1.hyper_mod(op_vector).mean().item():.4f}")
                if step % 100 == 0:
                    torch.save(s1.state_dict(), Path("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt"))

    except KeyboardInterrupt:
        print("Forge offline.")

if __name__ == "__main__":
    main()
