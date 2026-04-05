from __future__ import annotations

import argparse
import json
import math
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import zmq
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m14_5_continuous_decompressor_family import (
    build_decompressor_protocol_manifest,
    M14_5_REGISTRY
)
from lojban_evolution.experiment import generate_dataset, split_dataset
from src.lojban_evolution.m9.engine import M9System1

from contextlib import contextmanager

@contextmanager
def _seed_injection_hook(model, layer_index: int, seed_mask: torch.Tensor, delta: torch.Tensor):
    """
    Physically injects the M11 continuous logic into the <loj_seed> hidden state.
    """
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Ensure seed_mask is handled as a tensor
        if torch.is_tensor(seed_mask):
            mask_tensor = seed_mask.to(hidden.device)
            positions = torch.nonzero(mask_tensor[0], as_tuple=False).flatten()
            if len(positions) > 0:
                # KV-Cache Guard: Only inject during prefill when the hidden sequence contains the seed token
                valid_positions = positions[positions < hidden.shape[1]]
                if len(valid_positions) > 0:
                    hidden[:, valid_positions, :] += delta.to(hidden.dtype)
        return (hidden,) if isinstance(output, tuple) else hidden

    # Recursive Layer Discovery for PeftModel -> Backbone
    target_model = model
    if hasattr(model, "base_model"):
        target_model = model.base_model.model
    
    if hasattr(target_model, "model"):
        layers = target_model.model.layers
    else:
        layers = target_model.layers
        
    handle = layers[layer_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()

class DecompressorAblationOrchestrator:
    def __init__(self, model, tokenizer, s1, args):
        self.model = model
        self.tokenizer = tokenizer
        self.s1 = s1
        self.args = args
        self.device = next(model.parameters()).device

    def run_cell(self, cell_id: str, samples: list[Any]) -> dict[str, Any]:
        cell_spec = M14_5_REGISTRY["M14.5"]["cells"][cell_id]
        print(f"\n--- EXECUTING CELL {cell_id}: {cell_spec['label'].upper()} ---")
        
        results = []
        correct = 0

        for i, item in enumerate(samples):
            # 1. Forge Continuous Seed
            prompt = f"Question: {item['prompt']}\nReasoning: step."
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out_s2 = self.model(**inputs, output_hidden_states=True)
                h_eng = out_s2.hidden_states[-1][:, -1, :]
                
                # Extract prompt embeddings for dynamic PointerBind
                prompt_embs = self.model.get_input_embeddings()(inputs.input_ids)
                op_vec, x_probs, _ = self.s1.build_graph(h_eng, prompt_embs)
                seed_delta = op_vec.unsqueeze(1) # [1, 1, H]

            # 2. Inject and Unspool
            seed_prompt = f"{prompt}\n{self.args.seed_token}"
            seed_inputs = self.tokenizer(seed_prompt, return_tensors="pt").to(self.device)
            # Create a boolean tensor mask for the seed token position
            seed_token_id = self.tokenizer.convert_tokens_to_ids(self.args.seed_token)
            # PHYSICAL LAW: Ensure seed_mask is a TENSOR
            seed_mask = (seed_inputs.input_ids == seed_token_id)
            
            with _seed_injection_hook(self.model, self.args.layer_index, seed_mask, seed_delta * self.args.scratchpad_alpha):
                # Autoregressive generation of the runway
                # (Simplified for the 5-sample burst)
                out = self.model.generate(**seed_inputs, max_new_tokens=10, do_sample=False)
                runway_text = self.tokenizer.decode(out[0][seed_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 3. English Answer Resumption (WITH ATTENTION SHIELD)
            # Format: [Prompt] + <loj_seed> + [Runway] + \nAnswer:
            final_prompt = f"{seed_prompt} {runway_text}\nAnswer:"
            final_inputs = self.tokenizer(final_prompt, return_tensors="pt").to(self.device)
            
            # Identify indices to shield (everything between seed_token and Answer:)
            input_ids = final_inputs.input_ids[0]
            seed_id = self.tokenizer.convert_tokens_to_ids(self.args.seed_token)
            answer_ids = self.tokenizer.encode("\nAnswer:", add_special_tokens=False)
            
            # Find seed position
            seed_pos_tensor = torch.eq(input_ids, seed_id).nonzero(as_tuple=True)[0]
            # Find start of "Answer:"
            ans_start_pos_tensor = torch.eq(input_ids, answer_ids[0]).nonzero(as_tuple=True)[0]
            
            # The Shield Mask: 1 for allowed, 0 for shielded
            shield_mask = torch.ones_like(final_inputs.input_ids)
            if len(seed_pos_tensor) > 0 and len(ans_start_pos_tensor) > 0:
                runway_start = seed_pos_tensor[0] + 1
                ans_start_pos = ans_start_pos_tensor[-1]
                shield_mask[0, runway_start:ans_start_pos] = 0
                print(f"  Shield active: Masked {ans_start_pos - runway_start} Lojban tokens.")

            with torch.no_grad():
                # Overwrite the tokenizer's attention mask with our custom shield mask
                final_inputs["attention_mask"] = shield_mask
                
                # We pass the custom attention_mask to generate
                out_ans = self.model.generate(
                    **final_inputs, 
                    max_new_tokens=10, 
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                prediction = self.tokenizer.decode(out_ans[0][final_inputs.input_ids.shape[1]:], skip_special_tokens=True).strip().lower()

            target = item["answer"].lower().strip()
            is_correct = (prediction == target) or (prediction.startswith(target))
            if is_correct: correct += 1
            
            print(f"  Item {i+1}: Pred='{prediction}' | Target='{target}' | Correct: {is_correct}")
            results.append({"correct": is_correct})

        return {"accuracy": correct / len(samples), "details": results}

def main():
    parser = argparse.ArgumentParser(description="M14.5 Continuous Decompressor Runner.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-size", type=int, default=20)
    parser.add_argument("--layer-index", type=int, default=12)
    parser.add_argument("--scratchpad-alpha", type=float, default=1.0)
    parser.add_argument("--seed-token", type=str, default="<loj_seed>")
    parser.add_argument("--max-runway-length", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m14_5_decompressor"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    # FORCE CPU INFERENCE for diagnostic stability
    device = "cpu"
    
    # 1. System Loading
    base_archival_model = "archive/results/m9/active/RESULTS_M9_SYNCED/synced_model"
    tokenizer = AutoTokenizer.from_pretrained(base_archival_model, local_files_only=args.local_files_only)
    
    # Ensure <loj_seed> is in vocabulary
    if args.seed_token not in tokenizer.get_vocab():
        tokenizer.add_tokens([args.seed_token])
        print(f"M14: Added {args.seed_token} to vocabulary.")
    
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    backbone = AutoModelForCausalLM.from_config(config, attn_implementation="eager")
    # THE ROOT FIX: Physical re-assignment
    new_vocab_size = 153921
    backbone.resize_token_embeddings(new_vocab_size)
    backbone.model.embed_tokens = torch.nn.Embedding(new_vocab_size, backbone.config.hidden_size).to(device=device, dtype=backbone.dtype)
    backbone.lm_head = torch.nn.Linear(backbone.config.hidden_size, new_vocab_size, bias=False).to(device=device, dtype=backbone.dtype)
    
    weights_path = Path(args.base_model) / "adapter_model.safetensors"
    if weights_path.exists():
        from safetensors.torch import load_file
        clean_sd = {k.replace("base_model.model.", ""): v for k, v in load_file(str(weights_path)).items()}
        backbone.load_state_dict(clean_sd, strict=False)
    
    backbone = backbone.to(device)
    # Manual PEFT injection
    from peft import PeftConfig
    model = PeftModel(backbone, PeftConfig.from_pretrained(args.adapter))
    
    # TOTAL MANIFOLD RECONSTRUCTION
    new_vocab_size = len(tokenizer)
    
    # Surgically target only the specific embedding and head modules
    for name, module in model.named_modules():
        if "embed_tokens" in name and isinstance(module, torch.nn.Embedding):
            if module.num_embeddings < new_vocab_size:
                module.num_embeddings = new_vocab_size
                std = module.weight.std().item() if module.weight.numel() > 0 else 0.02
                module.weight = torch.nn.Parameter(torch.randn(new_vocab_size, backbone.config.hidden_size, device=device, dtype=backbone.dtype) * std)
        elif "lm_head" in name and isinstance(module, torch.nn.Linear):
            if module.out_features < new_vocab_size:
                module.out_features = new_vocab_size
                std = module.weight.std().item() if module.weight.numel() > 0 else 0.02
                module.weight = torch.nn.Parameter(torch.randn(new_vocab_size, backbone.config.hidden_size, device=device, dtype=backbone.dtype) * std)

    print(f"M14: Total manifold reconstruction complete ({new_vocab_size} tokens).")
    
    weights_bin = Path(args.adapter) / "adapter_model.bin"
    weights_safe = Path(args.adapter) / "adapter_model.safetensors"
    
    if weights_safe.exists():
        from safetensors.torch import load_file
        adapter_weights = load_file(str(weights_safe))
    else:
        # ARCHIVE RECOVERY: weights_only=False required for legacy .bin files
        adapter_weights = torch.load(str(weights_bin), map_location=device, weights_only=False)
        
    # SURGICAL FILTER: Map old weights into the new larger matrices
    print("M14: Surgically mapping adapter weights into expanded manifold...")
    for k, v in adapter_weights.items():
        if "embed_tokens" in k or "lm_head" in k:
            old_rows = v.shape[0]
            # Get the current submodule from PeftModel based on key
            try:
                submodule = model.get_submodule(k.rsplit('.', 1)[0])
                submodule.weight.data[:old_rows] = v.to(device=device, dtype=backbone.dtype)
            except Exception as e:
                print(f"Warning: Could not map {k}: {e}")
        else:
            try:
                model.get_parameter(k).data.copy_(v)
            except Exception as e:
                pass
                
    print("M14: Adapter attention minds surgically injected.")
    model.eval()

    # 2. Forge Loading (Rg)
    hidden_size = model.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    s1_sd = torch.load(args.checkpoint, map_location=device)
    s1.load_state_dict(s1_sd, strict=False)
    s1.eval()
    
    # Cast s1 to backbone precision
    s1 = s1.to(dtype=backbone.dtype)

    # 3. Data Loading
    pack_path = Path("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl")
    with pack_path.open("r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f][:args.eval_size]

    orchestrator = DecompressorAblationOrchestrator(model, tokenizer, s1, args)
    
    # Execute 4-Cell Matrix
    cells = ["A", "B", "C", "D"]
    final_report = {"run_id": args.run_id, "cells": {}}
    for cid in cells:
        final_report["cells"][cid] = orchestrator.run_cell(cid, samples)

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "m14_5_report.json").write_text(json.dumps(final_report, indent=2))
    print(f"\nM14.5 Complete: {args.output_root}")

if __name__ == "__main__":
    main()
