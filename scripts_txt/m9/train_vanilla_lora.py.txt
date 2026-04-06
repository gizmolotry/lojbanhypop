from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model
from lojban_evolution.experiment import generate_dataset
from scripts.m9.vanilla_utils import trace_to_english

def main():
    parser = argparse.ArgumentParser(description="M9.4 Control: Vanilla English LoRA Training.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m9/active/RESULTS_M9_CONTROL")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    
    # 2. Apply Standard LoRA (Branch A)
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    
    # 3. Standard AdamW
    opt = AdamW(model.parameters(), lr=args.lr)
    
    # 4. Dataset Generation (Same 7,430 puzzles)
    print("Generating 7,430 puzzles for Vanilla LoRA...")
    ds = generate_dataset(size=7430, seed=42, profile="diverse_v3")
    
    print(f"\n--- BRANCH A: VANILLA ENGLISH LoRA TRAINING INITIATED ---")
    
    for step in range(args.train_steps):
        item = ds[step % len(ds)]
        
        # Format: Question -> Reasoning -> Answer
        reasoning = trace_to_english(item.trace)
        full_text = f"Question: {item.prompt}\nReasoning: {reasoning}\nAnswer: {item.answer}"
        
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        labels = inputs.input_ids.clone()
        
        opt.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 50 == 0:
            print(f"Step {step+1}/{args.train_steps} - CE Loss: {loss.item():.4f}")

    # 5. Save Control Model
    model.save_pretrained(output_dir / "vanilla_lora")
    tokenizer.save_pretrained(output_dir / "vanilla_lora")
    print(f"Saved Vanilla LoRA to {output_dir / 'vanilla_lora'}")

if __name__ == "__main__":
    main()
