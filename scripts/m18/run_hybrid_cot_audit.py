import argparse
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from collections import Counter

from lojban_evolution.m18.salience import M18SalienceSelector
from lojban_evolution.m18.graph_induction import M18RelationalInterpreter, M18BiasCompiler
from lojban_evolution.m18.eval_core import M18TwoPassOrchestrator
from lojban_evolution.m18.registry import M18_REGISTRY

def scoring_fn(prediction, target):
    pred = prediction.lower().strip()
    target = target.lower().strip()
    return target in pred

def _loop_flag(token_ids: list[int]) -> bool:
    if len(token_ids) < 3:
        return False
    if len(set(token_ids)) <= max(1, len(token_ids) // 2):
        return True
    for n in (1, 2):
        if len(token_ids) < n * 2:
            continue
        seen: set[tuple[int, ...]] = set()
        for i in range(0, len(token_ids) - n + 1):
            gram = tuple(int(t) for t in token_ids[i : i + n])
            if gram in seen:
                return True
            seen.add(gram)
    return False

def run_hybrid_cot_audit(args):
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
    
    with open("artifacts/datasets/sanity_check_v1.jsonl", "r", encoding="utf-8") as f:
        en_samples = [json.loads(line) for line in f]
    with open("artifacts/datasets/sanity_check_zh_v1.jsonl", "r", encoding="utf-8") as f:
        zh_samples = [json.loads(line) for line in f]

    results = {}
    cot_instr = "\nThink step-by-step. Final answer one word after 'Answer: '."
    zh_cot_instr = "\n请逐步思考。最终答案在“答案：”之后的一个词。"

    # 1. Standard Baselines
    baselines = [
        {"id": "EN-CONCISE", "data": en_samples, "instr": "\nAnswer with one word.", "max_tok": 20},
        {"id": "EN-COT", "data": en_samples, "instr": cot_instr, "max_tok": 128},
        {"id": "ZH-COT", "data": zh_samples, "instr": zh_cot_instr, "max_tok": 128},
    ]

    for b in baselines:
        print(f"\n--- BASELINE: {b['id']} ---")
        correct, tokens, loops = 0, 0, 0
        for item in tqdm(b["data"], desc=b["id"]):
            full_prompt = item["prompt"] + b["instr"]
            inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
            prompt_len = inputs.input_ids.shape[1]
            with torch.no_grad():
                out = backbone.generate(**inputs, max_new_tokens=b["max_tok"], do_sample=False, pad_token_id=tokenizer.eos_token_id)
            gen_ids = out[0][prompt_len:].tolist()
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            tokens += len(gen_ids)
            if scoring_fn(gen_text, item["answer"]): correct += 1
            if _loop_flag(gen_ids): loops += 1
        results[b["id"]] = {"acc": correct/10, "tokens": tokens/10, "loops": loops/10}

    # 2. Hybrid M18 + CoT
    m18_cells = ["KILL-RANDOM", "U-TYPED", "L-TYPED"]
    for cid in m18_cells:
        print(f"\n--- HYBRID CELL: EN-COT + {cid} ---")
        cell_cfg = reg["cells"][cid]
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
        
        correct, tokens, loops = 0, 0, 0
        for item in tqdm(en_samples, desc=cid):
            prompt = item["prompt"] + cot_instr
            out = orchestrator.execute(prompt, intervention_active=True, max_new_tokens=128)
            gen_ids = out["token_ids"]
            tokens += len(gen_ids)
            if scoring_fn(out["prediction"], item["answer"]): correct += 1
            if _loop_flag(gen_ids): loops += 1
        results[f"EN-COT+{cid}"] = {"acc": correct/10, "tokens": tokens/10, "loops": loops/10}

    # Output Table
    print("\n" + "="*70)
    print(f"{'Cell':<20} | {'Accuracy':<10} | {'Avg Tokens':<10} | {'Loop Rate':<10}")
    print("-" * 70)
    for k, v in results.items():
        print(f"{k:<20} | {v['acc']:<10.4f} | {v['tokens']:<10.2f} | {v['loops']:<10.2f}")
    print("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    args = parser.parse_args()
    run_hybrid_cot_audit(args)
