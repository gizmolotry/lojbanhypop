from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lojban_evolution.m9.engine import M9System1
from lojban_evolution.experiment import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="M9 Phase 2: Grounded Curriculum Scaling.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-steps", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--load-ckpt", type=Path, default=Path("archive/results/m9/active/RESULTS_M9_PHASE1/m9_s1_phase1.pt"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size = 896
    
    # 1. Load System 2 (The Semantic Source)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    model.eval()

    # 2. Initialize System 1 (The Logical Symbiote)
    s1 = M9System1(hidden_size=hidden_size).to(device)
    if args.load_ckpt.exists():
        s1.load_state_dict(torch.load(args.load_ckpt))
        print(f"Phase 2: Loaded grounded Phase 1 weights.")

    opt = torch.optim.AdamW(s1.parameters(), lr=args.lr)

    print(f"\n--- M9 PHASE 2: GROUNDED CURRICULUM SCALING INITIATED ---")

    # 3. Tiered Grounded Training
    for tier in ["easy", "medium", "hard"]:
        print(f"\nScaling to {tier.upper()} logic puzzles...")
        ds = generate_dataset(size=args.train_steps // 3, seed=7, difficulty_tier=tier, profile="diverse_v3")
        
        for step, item in enumerate(ds):
            # THE SEMANTIC SCAN
            cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
            inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                premise_state = outputs.hidden_states[-1][:, -1, :].detach()
            
            # Grounded Target from trace
            from scripts.m9.phase2_forge import TRACE_TO_ANCHOR
            base_idx = TRACE_TO_ANCHOR.get(item.trace[0], 49)
            # HARD tier pushes into Gaussian Playground (50-1999)
            target_op_idx = 50 + (base_idx % 1950) if tier == "hard" else base_idx
            
            opt.zero_grad()
            
            # Supervision: Force the correct Operator and the 1st Pointer
            op_logits = s1.op_head(premise_state)
            p1_logits = s1.x_heads[0](premise_state)
            
            loss = F.cross_entropy(op_logits, torch.tensor([target_op_idx], device=device))
            loss += F.cross_entropy(p1_logits, torch.tensor([0], device=device)) # Simplified pointer target
            
            loss.backward()
            opt.step()
            
            if (step + 1) % 100 == 0:
                print(f"[{tier}] Step {step+1} - Grounded Emergence Loss: {loss.item():.4f}")

    output_dir = Path("archive/results/m9/active/RESULTS_M9_PHASE2")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(s1.state_dict(), output_dir / "m9_s1_phase2.pt")
    print(f"Phase 2 Complete. Checkpoint saved to {output_dir / 'm9_s1_phase2.pt'}")

if __name__ == "__main__":
    main()
