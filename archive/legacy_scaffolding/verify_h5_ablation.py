from __future__ import annotations
import torch
import torch.nn.functional as F
import json
from pathlib import Path
import sys
import random

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from scripts.train_h5_persistent_vq_advisor import (
    BooleanAnchorTable, CouncilCrossAttentionAdapter, AdvisorArityHead,
    persistent_advisor_hook, build_final_prefix, extract_trace_hidden_states,
    build_logic_prompt
)
from lojban_evolution.experiment import generate_dataset, split_dataset

def run_eval(name, surgery, iron_collar, ckpt_path, base_model_path, adapter_path, rel_bias=0.0):
    print(f"\n--- Evaluating {name} (Surgery={surgery}, IronCollar={iron_collar}, Bias={rel_bias}) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. Load Backbone & Resize
    backbone = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
    backbone.resize_token_embeddings(len(tokenizer))
    
    # 3. Load Adapter
    model = PeftModel.from_pretrained(backbone, adapter_path).to(device)
    
    # 4. Handshake Expansion
    tokenizer.add_special_tokens({"additional_special_tokens": ["[ADVANCE]"]})
    model.resize_token_embeddings(len(tokenizer))
    advance_token_id = tokenizer.convert_tokens_to_ids("[ADVANCE]")
    
    # 5. Load H5 Checkpoint Components
    checkpoint = torch.load(ckpt_path, map_location=device)
    hidden_size = model.config.hidden_size
    
    codebook = BooleanAnchorTable(2000, hidden_size).to(device)
    codebook.load_state_dict(checkpoint["codebook_state"])
    
    adapter_mod = CouncilCrossAttentionAdapter(hidden_size, use_boolean_surgery=surgery).to(device)
    adapter_mod.load_state_dict(checkpoint["advisor_adapter_state"], strict=False)
    
    arity_head = AdvisorArityHead(hidden_size, 2000).to(device)
    # Manual non-strict load for ArityHead
    curr_state = arity_head.state_dict()
    checkpoint_state = checkpoint["arity_head_state"]
    for name_p, param in checkpoint_state.items():
        if name_p in curr_state and curr_state[name_p].shape == param.shape:
            curr_state[name_p].copy_(param)
    arity_head.load_state_dict(curr_state)
    
    model.eval()
    
    ds = generate_dataset(size=100, seed=42)
    _, _, test = split_dataset(ds)
    
    correct = 0
    surgery_hits = 0
    
    for item in test[:10]:
        with torch.no_grad():
            h_t = extract_trace_hidden_states(model, tokenizer, item.prompt, 48)
            z_st, idx, _, _ = codebook.quantize(h_t, relation_bias=rel_bias)
            
            hits = (idx < 5).sum().item() # Track all 5 operators
            surgery_hits += hits
            
            tokens, _, _ = arity_head.decode_with_arity(z_st, use_iron_collar=iron_collar)
            adv_state = torch.cat([arity_head.token_to_embedding(t, codebook) for t in tokens], dim=1)
            adv_ids = torch.stack(tokens, dim=1)
            
            # Use generation to see actual trace words mapped to the codebook
            trace_words = []
            for t_id in tokens:
                tid = t_id.item()
                if tid < 5:
                    word = ["AND", "OR", "NOT", "IMPLIES", "XOR"][tid]
                else:
                    word = f"V{tid}"
                trace_words.append(word)
            
            # English Generation with Gearbox
            prefix = build_final_prefix(item.prompt)
            cur_ids = tokenizer(prefix, return_tensors="pt").input_ids.to(device)
            cur_emb = model.get_input_embeddings()(cur_ids)
            curr_ptr = 0
            
            gen_tokens = []
            for _ in range(24):
                p_ids = torch.full((1, cur_emb.shape[1]), curr_ptr, device=device, dtype=torch.long)
                with persistent_advisor_hook(model, 12, adapter_mod, adv_state, adv_ids, p_ids, 1.0):
                    out = model(inputs_embeds=cur_emb)
                
                next_token_id = torch.argmax(out.logits[:, -1, :], dim=-1)
                gen_tokens.append(next_token_id.item())
                
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                
                # Advance on [ADVANCE]
                if next_token_id.item() == advance_token_id:
                    curr_ptr = min(curr_ptr + 3, (len(tokens)) - 3)
                
                next_emb = model.get_input_embeddings()(next_token_id.unsqueeze(0))
                cur_emb = torch.cat([cur_emb, next_emb], dim=1)
            
            output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            is_correct = item.answer.strip().lower() in output_text.strip().lower()
            if is_correct: correct += 1
            
            print(f"--- Question: {item.prompt[:50]}... ---")
            print(f"Codebook Trace: {' '.join(trace_words[:12])}...")
            print(f"Output: {output_text.strip()}")
            print(f"Correct: {is_correct} | Surgery Hits: {hits}")

    print(f"\nFinal Accuracy: {correct}/10")
    print(f"Total Surgery Hits: {surgery_hits}")

if __name__ == "__main__":
    base = "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct"
    adapter = "runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5"
    
    ckpt_s3 = "runs/i_series/20260302_172603/h5_checkpoint.pt"
    ckpt_s4 = "runs/i_series/20260302_203358/h5_checkpoint.pt"
    ckpt_s5 = "runs/i_series/20260303_013946/h5_checkpoint.pt"
    ckpt_i = "runs/i_series/20260303_031401/h5_checkpoint.pt"
    
    run_eval("H5 Slice 3 (Dark Reasoner)", True, False, ckpt_s3, base, adapter, rel_bias=0.0)
    run_eval("H5 Slice 4 (Iron Collar)", True, True, ckpt_s4, base, adapter, rel_bias=0.0)
    run_eval("H5 Slice 5 (Grounded Fine-Tune)", True, True, ckpt_s5, base, adapter, rel_bias=0.0)
    run_eval("H5 Row I-Matrix (Teacher Distill)", True, True, ckpt_i, base, adapter, rel_bias=0.0)
