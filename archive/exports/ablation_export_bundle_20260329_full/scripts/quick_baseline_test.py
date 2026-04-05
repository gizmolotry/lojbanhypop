from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m8.engine import CouncilOfOracles, M8InterleavedRouter, adapter_disabled
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M8 vs English CoT Baseline: Head-to-Head.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--num-oracles", type=int, default=4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m8/active/RESULTS_M8_VS_ENGLISH_BASELINE")
    output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # 2. Initialize M8 Components
    council = CouncilOfOracles(hidden_size=hidden_size, num_oracles=args.num_oracles).to(device)
    router = M8InterleavedRouter(council, hidden_size=hidden_size).to(device)
    
    # Load M8 weights
    ckpt_path = Path("archive/results/m8/active/RESULTS_M8_COUNCIL_OF_ORACLES/m8_checkpoint.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        council.load_state_dict(ckpt["council_state"])
        router.load_state_dict(ckpt["router_state"])
        print(f"Loaded M8 engine from {ckpt_path}")
    
    model.eval(); council.eval(); router.eval()
    
    # 3. Load Diverse Dataset
    ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
    _, _, test_ds = split_dataset(ds)
    
    print(f"\n--- M8 VS ENGLISH CoT HEAD-TO-HEAD ---")
    
    m8_correct = 0
    base_correct = 0
    total = 20
    
    for item in test_ds[:total]:
        print(f"Testing: {item.prompt[:50]}...")
        
        # A. ENGLISH CoT BASELINE (Direct Qwen Generation)
        # We use a standard chain-of-thought prompt for the baseline
        cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        cot_ids = tokenizer(cot_prompt, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            # Generate reasoning + answer
            base_out = model.generate(input_ids=cot_ids, max_new_tokens=64, do_sample=False)
            base_full_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
            
            # Simple exact-word match for accuracy
            base_is_correct = item.answer.lower() in base_full_text.lower()
            if base_is_correct: base_correct += 1
            
        # B. M8 INTERLEAVED COUNCIL
        m8_prompt = f"Question: {item.prompt}\n<CALL_ADVISOR>\nAnswer:"
        m8_ids = tokenizer(m8_prompt, return_tensors="pt").input_ids.to(device)
        call_pos = (m8_ids[0] == call_advisor_id).nonzero(as_tuple=True)[0][0].item()
        
        with torch.no_grad():
            prompt_embs = model.get_input_embeddings()(m8_ids)
            out_s2_initial = model(inputs_embeds=prompt_embs[:, :call_pos+1, :], output_hidden_states=True)
            call_advisor_state = out_s2_initial.hidden_states[-1][:, -1, :]
            
            # Council Hypotheses
            hypothesis_matrix_choked = router.route_hypotheses(call_advisor_state, prompt_len=m8_ids.shape[1])
            hypothesis_matrix = router.prepare_for_injection(hypothesis_matrix_choked)
            
            # Supreme Judge Generation
            past_kv = out_s2_initial.past_key_values
            gen_ids = []
            cur_ids = m8_ids[:, call_pos+1:]
            
            for t in range(24):
                cur_embs = model.get_input_embeddings()(cur_ids)
                combined_embs = torch.cat([hypothesis_matrix, cur_embs], dim=1)
                outputs = model(inputs_embeds=combined_embs, past_key_values=past_kv)
                next_token_id = torch.argmax(outputs.logits[:, args.num_oracles + cur_embs.shape[1] - 1, :], dim=-1, keepdim=True)
                gen_ids.append(next_token_id.item())
                if next_token_id.item() == tokenizer.eos_token_id: break
                cur_ids, past_kv = next_token_id, outputs.past_key_values
            
            m8_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            m8_is_correct = item.answer.lower() in m8_text.lower()
            if m8_is_correct: m8_correct += 1

        print(f"  [Base] Correct: {base_is_correct} | [M8] Correct: {m8_is_correct}")

    print(f"\n--- FINAL RESULTS ---")
    print(f"English CoT Baseline Accuracy: {base_correct/total:.4f} ({base_correct}/{total})")
    print(f"M8 Council Accuracy:          {m8_correct/total:.4f} ({m8_correct}/{total})")
    print(f"M8 Accuracy Lift:             {((m8_correct - base_correct)/total):+.4f}")

if __name__ == "__main__":
    main()
