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
from src.lojban_evolution.m7.engine import System1Coprocessor, InterleavedRouter
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M7 Training: The Interleaved Latent Coprocessor.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Model & Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.adapter), local_files_only=args.local_files_only)
    
    # Add <CALL_ADVISOR> trigger token
    tokenizer.add_special_tokens({"additional_special_tokens": ["<CALL_ADVISOR>"]})
    call_advisor_id = tokenizer.convert_tokens_to_ids("<CALL_ADVISOR>")
    
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, str(args.adapter), local_files_only=args.local_files_only, device_map="auto")
    
    hidden_size = model.config.hidden_size
    
    # 2. Initialize M7 Components
    s1_coprocessor = System1Coprocessor(hidden_size=hidden_size).to(device)
    router = InterleavedRouter(s1_coprocessor, hidden_size=hidden_size).to(device)
    
    params = list(router.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr)
    
    ds = generate_dataset(size=100, seed=7)
    train_ds, _, _ = split_dataset(ds)
    
    print(f"\n--- M7 INTERLEAVED LATENT COPROCESSOR INITIALIZED ---")
    print(f"Topology: Synchronous Loop (System 2 -> System 1 -> System 2)")
    print(f"Vector Choke: Codebook compressed to d={s1_coprocessor.codebook.choke_dim}")
    
    # We will use a PyTorch hook to inject the constraint when <CALL_ADVISOR> is processed
    def injection_hook(module, args_tuple, kwargs_dict, output):
        # output is a tuple for Llama/Qwen layers: (hidden_states, past_key_values, ...)
        hidden_states = output[0] if isinstance(output, tuple) else output
        
        # In a real dynamic forward pass, we'd check if the current token is <CALL_ADVISOR>
        # For teacher forcing, we just inject the constraint at the specified position.
        # This hook is currently a placeholder for the actual dynamic routing logic,
        # which requires modifying the input embeddings or residual stream directly.
        return output

    # Register hook on the middle layer (e.g., layer 12)
    layer_idx = 12
    # hook_handle = model.base_model.model.model.layers[layer_idx].register_forward_hook(injection_hook, with_kwargs=True)

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]
        
        # Format: Prompt + <CALL_ADVISOR> + Answer
        prompt_text = f"Question: {item.prompt}\n<CALL_ADVISOR>\nAnswer:"
        
        input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        target_ids = tokenizer(" " + item.answer, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        # Find <CALL_ADVISOR> position
        call_pos = (input_ids[0] == call_advisor_id).nonzero(as_tuple=True)[0]
        if len(call_pos) == 0:
            continue
        call_pos = call_pos[0].item()
        
        opt.zero_grad()
        
        # 1. System 2 Initial Read (up to <CALL_ADVISOR>)
        with torch.no_grad():
            prompt_embs = model.get_input_embeddings()(input_ids) # [1, L, H]
            # Run through model to get the contextualized hidden state at <CALL_ADVISOR>
            out_s2_initial = model(inputs_embeds=prompt_embs[:, :call_pos+1, :], output_hidden_states=True)
            # Take the hidden state from the last layer at the <CALL_ADVISOR> position
            call_advisor_state = out_s2_initial.hidden_states[-1][:, -1, :] # [1, H]
        
        # 2. The Handoff & Discrete Choke (System 1)
        # S1 generates the discrete matrix and embeds it
        injected_constraint = router.route_and_inject(prompt_embs, call_advisor_state) # [1, H]
        
        # 3. The Injection & Resolution (System 2)
        # We simulate the injection by adding the constraint to the embeddings of the remaining sequence
        # (A true implementation injects into the residual stream, but embedding addition is mathematically similar for this PoC)
        
        # We Teacher-Force the answer generation
        answer_embs = model.get_input_embeddings()(target_ids) # [1, Ans_L, H]
        
        # Inject the constraint into the first token of the answer
        answer_embs[:, 0, :] = answer_embs[:, 0, :] + injected_constraint
        
        # Continue S2 forward pass
        # Provide the past_key_values from the initial read to maintain context
        past_kv = out_s2_initial.past_key_values
        
        out_s2_final = model(inputs_embeds=answer_embs, past_key_values=past_kv)
        logits = out_s2_final.logits # [1, Ans_L, Vocab]
        
        # Calculate Loss
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 5 == 0:
            print(f"Step {step+1}/{args.train_steps} - CE Loss: {loss.item():.4f}")

    print(f"\nM7 Vector Choke Active. Codebook size: {s1_coprocessor.codebook.codebook_size} (dim={s1_coprocessor.codebook.choke_dim})")
    
    ckpt_path = output_dir / "m7_checkpoint.pt"
    torch.save({
        "s1_state": s1_coprocessor.state_dict(),
        "router_state": router.state_dict()
    }, ckpt_path)
    print(f"Saved M7 checkpoint to {ckpt_path}")

if __name__ == "__main__":
    main()
