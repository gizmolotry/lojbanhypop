import argparse
import json
import torch
import re
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lojban_evolution.m18.salience import M18SalienceSelector
from lojban_evolution.m18.graph_induction import M18RelationalInterpreter, M18BiasCompiler
from lojban_evolution.m18.eval_core import M18TwoPassOrchestrator
from lojban_evolution.m18.registry import M18_REGISTRY

def scoring_fn(prediction, target):
    """Unified permissive substring match."""
    pred = prediction.lower().strip()
    target = target.lower().strip()
    return target in pred

def run_harmonized_audit(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = M18_REGISTRY["M18-v0"]
    defaults = reg["defaults"]
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    backbone.eval()
    
    # Load Data
    with open("artifacts/datasets/sanity_check_v1.jsonl", "r", encoding="utf-8") as f:
        en_samples = [json.loads(line) for line in f]
    with open("artifacts/datasets/sanity_check_zh_v1.jsonl", "r", encoding="utf-8") as f:
        zh_samples = [json.loads(line) for line in f]

    results = {}

    # Define Test Cells
    # 1. Base Baselines (Pass 1 only)
    baselines = [
        {"id": "EN-CONCISE", "lang": "en", "instruction": "\nAnswer with one word.", "data": en_samples},
        {"id": "EN-COT", "lang": "en", "instruction": "\nThink step-by-step. Final answer one word after 'Answer: '.", "data": en_samples},
        {"id": "ZH-COT", "lang": "zh", "instruction": "\n请逐步思考。最终答案在“答案：”之后的一个词。", "data": zh_samples},
    ]

    for b in baselines:
        print(f"\n--- RUNNING BASELINE: {b['id']} ---")
        correct = 0
        total_tokens = 0
        for item in tqdm(b["data"], desc=b["id"]):
            full_prompt = item["prompt"] + b["instruction"]
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                out = backbone.generate(**inputs, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
            gen_ids = out[0][prompt_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            total_tokens += len(gen_ids)
            if scoring_fn(gen_text, item["answer"]): correct += 1
            
        results[b["id"]] = {"acc": correct / len(b["data"]), "tokens": total_tokens / len(b["data"])}

    # 2. M18 Cells (Two-Pass)
    m18_cells = ["U-TYPED", "L-TYPED", "KILL-LABEL", "KILL-RANDOM"]
    
    for cid in m18_cells:
        cell_cfg = reg["cells"][cid]
        print(f"\n--- RUNNING M18 CELL: {cid} ---")
        
        selector = M18SalienceSelector(hidden_size=896, top_k=6).to(device)
        interpreter = M18RelationalInterpreter(hidden_size=896, num_relations=8, ontology=cell_cfg["ontology"]).to(device)
        compiler = M18BiasCompiler(num_relations=8, num_heads=backbone.config.num_attention_heads, hidden_size=896).to(device)
        
        path_prefix = "joint_u_v0" if cell_cfg["ontology"] == "U" else "joint_l_v0"
        joint_dir = Path("artifacts/models/m18") / path_prefix
        if (joint_dir / "selector_final.pt").exists():
            selector.load_state_dict(torch.load(joint_dir / "selector_final.pt", map_location=device))
            interpreter.load_state_dict(torch.load(joint_dir / "interpreter_final.pt", map_location=device))
            compiler.load_state_dict(torch.load(joint_dir / "compiler_final.pt", map_location=device))

        selector = selector.to(dtype=torch.bfloat16); interpreter = interpreter.to(dtype=torch.bfloat16); compiler = compiler.to(dtype=torch.bfloat16)
        orchestrator = M18TwoPassOrchestrator(backbone, tokenizer, selector, interpreter, compiler, defaults)
        
        correct = 0
        total_tokens = 0
        for item in tqdm(en_samples, desc=cid):
            # Same prompt as EN-CONCISE for fair M18 test
            prompt = item["prompt"] + "\nAnswer with one word."
            out = orchestrator.execute(prompt, intervention_active=True)
            
            gen_tokens = len(tokenizer.encode(out["prediction"], add_special_tokens=False))
            total_tokens += gen_tokens
            if scoring_fn(out["prediction"], item["answer"]): correct += 1
            
        results[cid] = {"acc": correct / len(en_samples), "tokens": total_tokens / len(en_samples)}

    # Final Table
    print("\n" + "="*60)
    print(f"{'Cell':<15} | {'Accuracy':<10} | {'Avg Tokens':<10}")
    print("-" * 60)
    for k, v in results.items():
        print(f"{k:<15} | {v['acc']:<10.4f} | {v['tokens']:<10.2f}")
    print("="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    run_harmonized_audit(args)
