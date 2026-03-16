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
from src.lojban_evolution.m8.engine import CouncilOfOracles, M8InterleavedRouter
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M8 Training: The Council of Oracles.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--num-oracles", type=int, default=4)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("RESULTS_M8_COUNCIL_OF_ORACLES")
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
    
    params = list(router.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    
    ds = generate_dataset(size=100, seed=7)
    train_ds, _, _ = split_dataset(ds)
    
    print(f"\n--- M8 COUNCIL OF ORACLES INITIALIZED ---")
    print(f"Parallel Broadcaster: {args.num_oracles} independent Oracles.")
    print(f"Hypothesis Matrix: [B, {args.num_oracles}, 16]")
    
    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]
        
        prompt_text = f"Question: {item.prompt}\n<CALL_ADVISOR>\nAnswer:"
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        # Find <CALL_ADVISOR> position
        call_pos_tensor = (input_ids[0] == call_advisor_id).nonzero(as_tuple=True)[0]
        if len(call_pos_tensor) == 0: continue
        call_pos = call_pos_tensor[0].item()
        
        opt.zero_grad()
        
        # 1. System 2 Initial Read
        with torch.no_grad():
            prompt_embs = model.get_input_embeddings()(input_ids)
            out_s2_initial = model(inputs_embeds=prompt_embs[:, :call_pos+1, :], output_hidden_states=True)
            call_advisor_state = out_s2_initial.hidden_states[-1][:, -1, :] # [1, H]
        
        # 2. Parallel Latent Broadcast (M8.1) & Hypothesis Matrix (M8.2)
        hypothesis_matrix_choked = router.route_hypotheses(call_advisor_state, prompt_len=input_ids.shape[1]) # [1, N, 16]
        hypothesis_matrix = router.prepare_for_injection(hypothesis_matrix_choked) # [1, N, H]
        
        # 3. Supreme Judge Resolution (System 2)
        # We simulate S2's cross-attention by prepending the N hypotheses to the KV cache
        # System 2 will attend to the hypothesis that aligns with its understanding
        
        # In this PoC, we concatenate the hypothesis matrix to the answer embeddings
        # so System 2 can cross-attend to the N choices.
        answer_embs = model.get_input_embeddings()(target_ids) # [1, Ans_L, H]
        
        # Combine: [Hypothesis 1, Hypothesis 2... Hypothesis N, Answer Token 1, Answer Token 2...]
        # System 2 sees the hypotheses as logical 'prefixes' to the answer.
        combined_embs = torch.cat([hypothesis_matrix, answer_embs], dim=1)
        
        # Provide the past_key_values from the initial read
        past_kv = out_s2_initial.past_key_values
        
        out_s2_final = model(inputs_embeds=combined_embs, past_key_values=past_kv)
        
        # We only care about the logits for the answer portion
        # Answer tokens start at index N in the combined sequence
        logits = out_s2_final.logits[:, args.num_oracles:, :] # [1, Ans_L, Vocab]
        
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}/{args.train_steps} - CE Loss: {loss.item():.4f}")

    print(f"\nM8 Superposition Active. Council size: {args.num_oracles}")
    
    ckpt_path = output_dir / "m8_checkpoint.pt"
    torch.save({
        "council_state": council.state_dict(),
        "router_state": router.state_dict()
    }, ckpt_path)
    print(f"Saved M8 checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
