from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import random

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.lojban_evolution.m9.engine import M9System1
from lojban_evolution.experiment import generate_dataset

def generate_ideal_traces(size: int):
    """
    Generates 10,000 idealized Lojban AST traces for Syntax SFT.
    Format: [Op_Index, Ptr1_Index, Ptr2_Index]
    """
    traces = []
    for _ in range(size):
        # Sample an operator from the Grounded Anchors (0-49)
        op = random.randint(0, 49)
        # Sample two pointers within a typical prompt length (e.g., 0-40)
        p1 = random.randint(0, 40)
        p2 = random.randint(0, 40)
        traces.append((op, p1, p2))
    return traces

def main():
    parser = argparse.ArgumentParser(description="M9 Phase 1: Grounded Syntax SFT.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-4)
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
    opt = torch.optim.AdamW(s1.parameters(), lr=args.lr)

    # 3. Generate Grounded Training Data
    print("Generating grounded SFT samples...")
    ds = generate_dataset(size=args.train_steps, seed=42, profile="diverse_v3")
    
    print(f"\n--- M9 PHASE 1: GROUNDED SYNTAX SFT INITIATED ---")

    for step, item in enumerate(ds):
        # THE SEMANTIC SCAN: Get real English thoughts
        cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            premise_state = outputs.hidden_states[-1][:, -1, :].detach() # [1, 896]
        
        # Target logic triple from procedural trace
        from scripts.m9.phase2_forge import TRACE_TO_ANCHOR
        target_op = TRACE_TO_ANCHOR.get(item.trace[0], 49)
        # We simplify to 1st pointer for SFT phase
        target_p1 = random.randint(0, min(inputs.input_ids.shape[1]-1, 40))
        
        opt.zero_grad()
        
        # Predict 
        op_logits = s1.op_head(premise_state)
        p1_logits = s1.x_heads[0](premise_state)
        
        loss = F.cross_entropy(op_logits, torch.tensor([target_op], device=device))
        loss += F.cross_entropy(p1_logits, torch.tensor([target_p1], device=device))
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{args.train_steps} - Grounded Loss: {loss.item():.4f}")

    output_dir = Path("archive/results/m9/active/RESULTS_M9_PHASE1")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(s1.state_dict(), output_dir / "m9_s1_phase1.pt")
    print(f"Phase 1 Complete. Checkpoint saved to {output_dir / 'm9_s1_phase1.pt'}")

if __name__ == "__main__":
    main()
