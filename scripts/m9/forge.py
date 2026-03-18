from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import zmq
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lojban_evolution.m9.engine import M9System1, InfoNCEForge

def main():
    parser = argparse.ArgumentParser(description="M9 Forge: Contrastive NLI Training.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Initialize ZeroMQ PULL socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{args.port}")
    print(f"Forge listening on port {args.port}")

    # 2. Initialize M9 Components
    hidden_size = 896
    s1 = M9System1(hidden_size=hidden_size).to(device)
    forge = InfoNCEForge(hidden_size=hidden_size).to(device)
    
    opt = torch.optim.AdamW(list(s1.parameters()) + list(forge.parameters()), lr=args.lr)

    print(f"\n--- M9 FORGE ACTIVE ---")
    print(f"Algorithm: InfoNCE Contrastive NLI")
    print(f"Mode: Asynchronous PULL")
    
    step = 0
    try:
        while True:
            # PULL premise from Harvester
            payload = socket.recv_json()
            premise_state = torch.tensor(payload["premise"], device=device).view(1, -1) # [1, H]
            prompt_len = payload["prompt_len"]
            
            opt.zero_grad()
            
            # 1. Generate Positive Hypothesis (Grounded Logic)
            # System 1 builds a graph from the premise
            op_state, ptr_logits = s1.build_graph(premise_state, prompt_len)
            pos_hypothesis = op_state # [1, H]
            
            # 2. Generate Negative Hypotheses (Hallucination/Noise)
            # We sample random vectors from the codebook to represent invalid logic
            n_neg = 5
            indices = torch.randint(0, s1.codebook_size, (1, n_neg), device=device)
            neg_hypotheses = s1.emb[indices] # [1, N_neg, H]
            
            # 3. Calculate Contrastive Loss
            # This pulls pos_hypothesis closer to premise_state and repels neg_hypotheses
            loss = forge(premise_state, pos_hypothesis, neg_hypotheses)
            
            loss.backward()
            opt.step()
            
            step += 1
            if step % 10 == 0:
                print(f"Step {step} - InfoNCE Loss: {loss.item():.4f}")
                
            if step % 500 == 0:
                # Periodic Checkpoint
                output_dir = Path("RESULTS_M9_FORGE")
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"s1_state": s1.state_dict(), "forge_state": forge.state_dict()}, output_dir / "m9_forge_checkpoint.pt")

    except KeyboardInterrupt:
        print("\nForge shutting down...")

if __name__ == "__main__":
    main()
