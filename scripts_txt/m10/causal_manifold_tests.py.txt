import torch
import torch.nn.functional as F
import json
from pathlib import Path
import sys
import zmq
import random

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.deep_adapter import M10eDeepTranslationAdapter
from lojban_evolution.experiment import generate_dataset, split_dataset

def run_causal_intervention(mode="baseline"):
    print(f"\n--- MANIFOLD INTERVENTION: {mode.upper()} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "archive/results/m9/active/RESULTS_M9_SYNCED/synced_model"
    
    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone = AutoModelForCausalLM.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, "archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m10f_aligned_model")
    model.eval()
    
    hidden_size = backbone.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    s1.load_state_dict(torch.load("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt", map_location=device), strict=False)
    s1.eval()
    
    adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
    adapter.load_state_dict(torch.load("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m10f_joint_adapter.pt", map_location=device))
    adapter.eval()

    context = zmq.Context()
    push = context.socket(zmq.PUSH); push.connect("tcp://127.0.0.1:5555")
    pull = context.socket(zmq.PULL); pull.connect("tcp://127.0.0.1:5556")

    # 2. Data
    all_ds = generate_dataset(size=100, seed=42, profile="diverse_v3")
    _, _, test_ds = split_dataset(all_ds)
    samples = test_ds[:20]

    correct = 0
    all_targets = []
    all_preds = []
    
    for item in samples:
        with torch.no_grad():
            inputs = tokenizer(f"Question: {item.prompt}\nReasoning: step.", return_tensors="pt").to(device)
            h_eng = model(**inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
            
            # Logic Injection
            push.send_json({"premise": h_eng.tolist(), "prompt_len": inputs.input_ids.shape[1]})
            logic_str = pull.recv_json()["logic_string"]
            
            op_vec, x_probs, op_idx = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            
            # --- INTERVENTIONS ---
            if mode == "shuffle_op":
                rand_idx = torch.randint(100, 2000, (1,), device=device)
                op_vec = s1.manifold.get_vector(rand_idx, token_type=0)
            
            elif mode == "scramble_vectors":
                norm = op_vec.norm()
                op_vec = torch.randn_like(op_vec) * (norm / math.sqrt(hidden_size))
                ptr_vecs = torch.randn_like(ptr_vecs) * (ptr_vecs.norm() / math.sqrt(hidden_size))
            
            elif mode == "scramble_pointers":
                ptr_vecs = torch.randn_like(ptr_vecs) * (ptr_vecs.norm() / math.sqrt(hidden_size))
                
            elif mode == "scramble_op":
                norm = op_vec.norm()
                op_vec = torch.randn_like(op_vec) * (norm / math.sqrt(hidden_size))
            
            elif mode == "mask_pointers":
                ptr_vecs = torch.zeros_like(ptr_vecs)

            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)
            
            # Resolution
            res_prompt = f"Question: {item.prompt}\nReasoning: step.\n<logic> {logic_str} </logic>\nAnswer:"
            res_inputs = tokenizer(res_prompt, return_tensors="pt").to(device)
            h_base = model(input_ids=res_inputs.input_ids, output_hidden_states=True).hidden_states[-1][:, -1:, :]
            
            h_final = adapter(h_base, logic_state)
            logits = model.lm_head(h_final)
            pred = tokenizer.decode([torch.argmax(logits, dim=-1).item()]).strip().lower()
            target = item.answer.lower().strip()
            
            is_correct = pred.startswith(target)
            correct += int(is_correct)
            
            if mode == "baseline":
                all_targets.append(target)
                all_preds.append(pred)
                print(f"Target: {target} | Pred: {pred} | Correct: {is_correct}")

    acc = correct / len(samples)
    print(f"Accuracy under {mode}: {acc:.2f}")
    
    if mode == "baseline":
        from collections import Counter
        print("\n--- CONFUSION MATRIX (Baseline) ---")
        pairs = Counter(zip(all_targets, all_preds))
        for (t, p), count in pairs.most_common():
            print(f"Target: {t:15} | Pred: {p:15} | Count: {count}")
        print("-----------------------------------\n")
        
    return acc

import math
if __name__ == "__main__":
    results = {}
    results["baseline"] = run_causal_intervention("baseline")
    results["shuffle_op"] = run_causal_intervention("shuffle_op")
    results["scramble_vectors"] = run_causal_intervention("scramble_vectors")
    results["scramble_pointers"] = run_causal_intervention("scramble_pointers")
    results["scramble_op"] = run_causal_intervention("scramble_op")
    results["mask_pointers"] = run_causal_intervention("mask_pointers")
    
    print("\n--- CAUSAL FINAL VERDICT ---")
    print(json.dumps(results, indent=2))
