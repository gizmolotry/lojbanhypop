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

from src.lojban_evolution.m9.engine import M9System1

# Mapping from procedural trace tokens to indices
# We use the same TRACE_TO_ANCHOR as Phase 1, but we'll add 'Emergent' logic
TRACE_TO_ANCHOR = {
    "TASK_WINOGRAD": 0, "TASK_SPATIAL": 1, "TASK_TEMPORAL": 2,
    "BIND_E1": 10, "BIND_E2": 11, "BIND_OBJ": 12, "BIND_LOC": 13,
    "LINK_CAUSAL": 20, "LINK_SPATIAL": 21, "LINK_TEMPORAL": 22,
    "PRONOUN_REF": 30, "RESOLVE_PRON_E1": 31, "RESOLVE_PRON_E2": 32,
    "VERIFY_ID": 40, "ANS_E1": 41, "ANS_E2": 42, "ANS_YES": 43, "ANS_NO": 44
}

def main():
    parser = argparse.ArgumentParser(description="M9 Phase 2: Curriculum Forge.")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--load-ckpt", type=Path, default=Path("archive/results/m9/active/RESULTS_M9_PHASE1/m9_s1_phase1.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 896
    
    # 1. Initialize ZeroMQ PULL socket
    context = zmq.Context()
    socket = context.socket(zmq.PULL)
    socket.bind(f"tcp://*:{args.port}")
    
    # 2. Initialize M9 System 1
    s1 = M9System1(hidden_size=hidden_size).to(device)
    if args.load_ckpt.exists():
        s1.load_state_dict(torch.load(args.load_ckpt))
        print(f"Curriculum Forge: Loaded Phase 1 weights.")

    opt = torch.optim.AdamW(s1.parameters(), lr=args.lr)

    print(f"\n--- M9 PHASE 2: CURRICULUM FORGE ACTIVE ---")
    print(f"Goal: Guide emergence into Gaussian playground via tiered complexity.")

    step = 0
    try:
        while True:
            payload = socket.recv_json()
            premise = torch.tensor(payload["premise"], device=device).view(1, -1)
            trace = payload["trace"]
            tier = payload["tier"]
            
            opt.zero_grad()
            
            # Map trace to target index
            base_idx = TRACE_TO_ANCHOR.get(trace[0], 49)
            
            # GUIDED EMERGENCE PHYSICS:
            # If the puzzle is 'hard', we offset the target into the Gaussian Noise range (50-1999)
            # This forces the model to invent a 'Hard Operator' to minimize the loss.
            if tier == "hard":
                # We use a deterministic map for the 'hard' tokens to ensure consistency
                target_op = torch.tensor([50 + (base_idx % 1950)], device=device)
            else:
                target_op = torch.tensor([base_idx], device=device)
            
            logits = s1.op_head(premise)
            loss = F.cross_entropy(logits, target_op)
            
            loss.backward()
            opt.step()
            
            step += 1
            if step % 50 == 0:
                print(f"Step {step} [{tier.upper()}] - Emergence Loss: {loss.item():.4f}")
                
            if step % 500 == 0:
                output_dir = Path("archive/results/m9/active/RESULTS_M9_PHASE2")
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(s1.state_dict(), output_dir / "m9_s1_phase2.pt")

    except KeyboardInterrupt:
        print("Curriculum Forge offline.")

if __name__ == "__main__":
    main()
