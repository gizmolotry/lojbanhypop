from __future__ import annotations
import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lojban_evolution.m18.salience import M18SalienceSelector
from lojban_evolution.m18.graph_induction import M18RelationalInterpreter, M18BiasCompiler
from lojban_evolution.m18.eval_core import M18TwoPassOrchestrator
from lojban_evolution.m18.registry import M18_REGISTRY

def run_m18_audit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = M18_REGISTRY["M18-v0"]
    defaults = reg["defaults"]
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    # FORCE EAGER ATTENTION to enable intervention hooks
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    backbone.eval()
    
    # Load Test Corpus
    with open(args.data_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f][:args.eval_size]

    results = {"run_id": args.run_id, "cells": {}}

    # Execute Ablation Cells
    for cid, cell in reg["cells"].items():
        print(f"\n--- AUDITING CELL {cid}: {cell['label'].upper()} ---")
        
        # 1. Initialize Components for this cell
        selector = M18SalienceSelector(hidden_size=896, top_k=6).to(device)
        interpreter = M18RelationalInterpreter(hidden_size=896, num_relations=8, ontology=cell["ontology"]).to(device)
        compiler = M18BiasCompiler(num_relations=8, num_heads=backbone.config.num_attention_heads, hidden_size=896).to(device)
        
        # Load joint weights if intervention is active
        if cell["intervention"] != "none":
            path_prefix = "joint_u_v0" if cell["ontology"] == "U" else "joint_l_v0"
            joint_dir = Path("artifacts/models/m18") / path_prefix
            if (joint_dir / "selector_final.pt").exists():
                selector.load_state_dict(torch.load(joint_dir / "selector_final.pt", map_location=device))
                interpreter.load_state_dict(torch.load(joint_dir / "interpreter_final.pt", map_location=device))
                compiler.load_state_dict(torch.load(joint_dir / "compiler_final.pt", map_location=device))

        # Precision Match
        selector = selector.to(dtype=torch.bfloat16); interpreter = interpreter.to(dtype=torch.bfloat16); compiler = compiler.to(dtype=torch.bfloat16)
        
        orchestrator = M18TwoPassOrchestrator(backbone, tokenizer, selector, interpreter, compiler, defaults)
        
        cell_results = []
        correct = 0
        total_tokens = 0
        
        for item in tqdm(samples, desc=cid):
            # Intervention logic implemented via orchestrator
            out = orchestrator.execute(item["prompt"], intervention_active=(cell["intervention"] != "none"))
            
            prediction = out["prediction"].lower().strip()
            # Token count (excluding prompt)
            gen_tokens = len(tokenizer.encode(out["prediction"], add_special_tokens=False))
            total_tokens += gen_tokens

            target = item["answer"].lower().strip()
            # Permissive check: is target word inside the prediction?
            is_correct = target in prediction
            if is_correct: correct += 1
            
            if len(cell_results) < 3:
                print(f"  Prompt: {item['prompt'][:50]}...")
                print(f"  Pred: {prediction!r} | Tokens: {gen_tokens} | Match: {is_correct}")
            
            cell_results.append({
                "correct": is_correct,
                "tokens": gen_tokens,
                "top_k_tokens": out["top_k_tokens"],
                "telemetry": out["telemetry"]
            })
            
        acc = correct / len(samples)
        avg_tokens = total_tokens / len(samples)
        results["cells"][cid] = {"accuracy": acc, "avg_tokens": avg_tokens, "details": cell_results}
        print(f"  Cell {cid} Accuracy: {acc:.4f} | Avg Tokens: {avg_tokens:.2f}")

    # Save Final Report
    output_path = Path("artifacts/runs/m18_v0_audit_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))
    print(f"\nAudit Complete. Report saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--eval-size", type=int, default=10)
    parser.add_argument("--run-id", default="m18_v0_smoke")
    args = parser.parse_args()
    run_m18_audit(args)
