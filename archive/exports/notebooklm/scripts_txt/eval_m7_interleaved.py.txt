from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m7.engine import System1Coprocessor, InterleavedRouter, adapter_disabled
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M7 Evaluation: The Interleaved Latent Coprocessor.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-gen-tokens", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=Path("archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Tokenizer from adapter to match Phase 5 vocabulary (151701)
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    
    # Load adapter with 151701 vocab
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map="auto")
    
    # Now expand for <CALL_ADVISOR> (151702)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<CALL_ADVISOR>"]})
    call_advisor_id = tokenizer.convert_tokens_to_ids("<CALL_ADVISOR>")
    model.resize_token_embeddings(len(tokenizer))
    
    hidden_size = model.config.hidden_size
    
    # 2. Initialize Components
    s1_coprocessor = System1Coprocessor(hidden_size=hidden_size).to(device)
    router = InterleavedRouter(s1_coprocessor, hidden_size=hidden_size).to(device)
    
    # Load trained weights
    ckpt_path = args.output_dir / "m7_checkpoint.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        s1_coprocessor.load_state_dict(ckpt["s1_state"])
        router.load_state_dict(ckpt["router_state"])
        print(f"Loaded trained M7 engine from {ckpt_path}")
    else:
        print(f"Warning: No checkpoint found at {ckpt_path}. Evaluating untrained state.")
        
    model.eval()
    s1_coprocessor.eval()
    router.eval()
    
    print("Generating M7 diverse evaluation dataset (Size=5000)...")
    ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
    _, _, test_ds = split_dataset(ds)
    print(f"Evaluation Dataset Size: {len(test_ds)}")
    
    print(f"\n--- M7 INTERLEAVED EVALUATION INITIATED ---")
    
    correct = 0
    total = 20 # Sample size
    results = []
    
    for item in test_ds[:total]:
        try:
            print(f"Evaluating item: {item.prompt[:50]}...")
            
            # Format: Prompt + <CALL_ADVISOR> + Answer
            prompt_text = f"Question: {item.prompt}\n<CALL_ADVISOR>\nAnswer:"
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            
            # Find <CALL_ADVISOR> position
            call_pos_tensor = (input_ids[0] == call_advisor_id).nonzero(as_tuple=True)[0]
            if len(call_pos_tensor) == 0: continue
            call_pos = call_pos_tensor[0].item()
            
            with torch.no_grad():
                # Step 1: System 2 Initial Read
                prompt_embs = model.get_input_embeddings()(input_ids)
                out_s2_initial = model(inputs_embeds=prompt_embs[:, :call_pos+1, :], output_hidden_states=True)
                call_advisor_state = out_s2_initial.hidden_states[-1][:, -1, :]
                
                # Step 2: System 1 Handoff & Discrete Choke
                injected_constraint = router.route_and_inject(prompt_embs, call_advisor_state)
                
                # Step 3: System 2 Autoregressive Resolution
                past_kv = out_s2_initial.past_key_values
                
                # Start generation after "Answer:"
                gen_ids = []
                cur_ids = input_ids[:, call_pos+1:] # Contains "Answer:"
                
                # We need to manually inject the constraint for the first token generation
                for t in range(args.max_gen_tokens):
                    cur_embs = model.get_input_embeddings()(cur_ids)
                    if t == 0:
                        cur_embs[:, 0, :] = cur_embs[:, 0, :] + injected_constraint
                    
                    outputs = model(inputs_embeds=cur_embs, past_key_values=past_kv)
                    next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                    gen_ids.append(next_token_id.item())
                    
                    if next_token_id.item() == tokenizer.eos_token_id:
                        break
                    
                    # Update for next step (purely English now)
                    cur_ids = next_token_id
                    past_kv = outputs.past_key_values
                
                predicted_token = tokenizer.decode([gen_ids[0]], skip_special_tokens=True).strip().lower()
                clean_full_pred = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                clean_target = item.answer.lower().strip()
                
                is_correct = (predicted_token == clean_target)
                if not is_correct and len(clean_target) > 3:
                    is_correct = clean_target in clean_full_pred.lower()
                
                if is_correct: correct += 1
                
                results.append({
                    "prompt": item.prompt,
                    "target_answer": item.answer,
                    "predicted": clean_full_pred,
                    "correct": is_correct
                })
                
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    accuracy = correct / total
    print(f"\nM7 Interleaved Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy": accuracy,
        "correct_count": correct,
        "total_evaluated": total,
        "vector_choke_dim": s1_coprocessor.codebook.choke_dim,
        "details": results
    }
    
    (args.output_dir / "m7_eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {args.output_dir / 'm7_eval_report.json'}")

if __name__ == "__main__":
    main()
