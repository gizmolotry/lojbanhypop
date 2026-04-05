from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import json

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m9.engine import MoVGate
from lojban_evolution.experiment import generate_dataset

def main():
    parser = argparse.ArgumentParser(description="M9.5 Training: The MoV Gate.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m9/active/RESULTS_M9_MOV")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Synced Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    base_vocab_size = tokenizer.convert_tokens_to_ids("<loj_0>")
    
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, args.adapter)
    
    hidden_size = model.config.hidden_size
    
    # 2. Initialize MoV Gate
    gate = MoVGate(hidden_size=hidden_size, base_vocab_size=base_vocab_size).to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=args.lr)
    
    # 3. Training Data (Idealized Logic -> Answer sequences)
    ds = generate_dataset(size=args.train_steps, seed=42, profile="diverse_v3")
    
    print(f"\n--- M9.5 MoV GATE TRAINING INITIATED ---")
    print(f"Goal: Learn bimodal routing (Lojban inside tags, English outside).")

    for step, item in enumerate(ds):
        # We simulate the full 10-slot reasoning trace for the gate to learn context
        # <logic> <loj_99> <loj_ptr1> ... <loj_ptr9> </logic> Answer: [ENGLISH]
        ptrs = " ".join([f"<loj_{2000 + (step + i) % 255}>" for i in range(9)])
        logic_str = f"<loj_99> {ptrs}"
        full_text = f"Question: {item.prompt}\n<logic> {logic_str} </logic>\nAnswer: {item.answer}"
        
        inputs = tokenizer(full_text, return_tensors="pt").to(device)
        labels = inputs.input_ids
        
        opt.zero_grad()
        
        # Get final hidden states
        outputs = model(input_ids=inputs.input_ids, output_hidden_states=True)
        h_t = outputs.hidden_states[-1] # [1, L, H]
        
        # Gated mixture forward pass
        # We hijack the lm_head.weight for the dual projection
        lm_weight = model.get_output_embeddings().weight
        log_probs = gate(h_t, lm_weight) # [1, L, V_total]
        
        # NLL Loss on the labels
        # Shift labels for next-token prediction
        shift_log_probs = log_probs[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        
        loss = F.nll_loss(shift_log_probs.view(-1, shift_log_probs.shape[-1]), shift_labels.view(-1))
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            # Measure average gate value to check for bimodal focus
            g_val = torch.sigmoid(gate.router(h_t)).mean().item()
            print(f"Step {step+1}/{args.train_steps} - NLL Loss: {loss.item():.4f} | Avg Gate: {g_val:.4f}")

    # 4. Save Gate Weights
    torch.save(gate.state_dict(), output_dir / "mov_gate.pt")
    print(f"Saved MoV Gate to {output_dir / 'mov_gate.pt'}")

if __name__ == "__main__":
    main()
