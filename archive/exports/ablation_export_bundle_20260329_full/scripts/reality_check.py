from __future__ import annotations

import argparse
import torch
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="Reality Check: True Base Model Performance.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading TRUE BASE MODEL (No Adapters): {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    model.eval()
    
    ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
    _, _, test_ds = split_dataset(ds)
    
    print(f"\n--- REALITY CHECK: TRUE ENGLISH CoT BASELINE ---")
    
    base_correct = 0
    total = 20
    
    for item in test_ds[:total]:
        cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        cot_ids = tokenizer(cot_prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            base_out = model.generate(input_ids=cot_ids, max_new_tokens=64, do_sample=False)
            base_full_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
            
            # Simple exact-word match for accuracy
            base_is_correct = item.answer.lower() in base_full_text.lower()
            if base_is_correct: base_correct += 1
            
        print(f"Target: {item.answer} | Correct: {base_is_correct}")
        print(f"Output: {base_full_text[len(cot_prompt):].strip()}\n")

    print(f"\n--- FINAL REALITY CHECK RESULTS ---")
    print(f"True Base Model Accuracy: {base_correct/total:.4f} ({base_correct}/{total})")

if __name__ == "__main__":
    main()
