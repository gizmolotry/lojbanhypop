from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import zmq
import json
from pathlib import Path
import sys
import math
from datetime import datetime, timezone
from sklearn.metrics import confusion_matrix, f1_score, classification_report

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.deep_adapter import M10eDeepTranslationAdapter
from src.lojban_evolution.m10.english_head import M10cEnglishHead
from lojban_evolution.experiment import generate_dataset, split_dataset

def run_single_seed(seed, args, device, model, tokenizer, s1, adapter, english_head, target_words, id_to_class, num_classes, push_socket, pull_socket):
    print(f"  Audit: Processing Seed {seed}...")
    
    ds = generate_dataset(size=2000, seed=seed, profile="diverse_v3")
    _, _, test_ds = split_dataset(ds)
    test_ds = test_ds[:args.num_samples]

    y_true = []
    y_pred = []
    
    for i, item in enumerate(test_ds):
        try:
            # Semantic Scan
            cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
            inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                out_s2 = model(**inputs, output_hidden_states=True)
                h_eng = out_s2.hidden_states[-1][:, -1, :]
            
            # Logic Request
            push_socket.send_json({"premise": h_eng.tolist(), "prompt_len": inputs.input_ids.shape[1], "target": item.answer})
            logic_payload = pull_socket.recv_json()
            logic_string = logic_payload["logic_string"]
            
            # Resolution
            with torch.no_grad():
                op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
                prompt_embs = model.get_input_embeddings()(inputs.input_ids)
                ptr_vecs = torch.matmul(x_probs, prompt_embs)
                logic_tensor = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

                res_prompt = f"{cot_prompt}\n<logic> {logic_string} </logic>\nAnswer:"
                res_inputs = tokenizer(res_prompt, return_tensors="pt").to(device)
                res_outputs = model(input_ids=res_inputs.input_ids, output_hidden_states=True)
                h_base = res_outputs.hidden_states[-1][:, -1:, :]
                
                h_final = h_base
                if adapter is not None:
                    h_final = adapter(h_base, logic_tensor)
                
                logits = english_head(h_final)
                pred_class = torch.argmax(logits, dim=-1).item()
            
            target_clean = item.answer.lower().strip()
            true_class = id_to_class.get(tokenizer.encode(" " + target_clean, add_special_tokens=False)[0], num_classes - 1)
                
            y_true.append(true_class)
            y_pred.append(pred_class)
            
        except Exception as e:
            continue

    return y_true, y_pred

def main():
    parser = argparse.ArgumentParser(description="M11 Large-Scale Robust Audit.")
    parser.add_argument("--base-model", default="archive/results/m9/active/RESULTS_M9_SYNCED/synced_model")
    parser.add_argument("--forge-ckpt", default="archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt")
    parser.add_argument("--adapter-ckpt", default="archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m11_native_adapter.pt")
    parser.add_argument("--head-ckpt", default="archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m11_native_head.pt")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--disable-adapter", action="store_true")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Systems
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    backbone = AutoModelForCausalLM.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, args.base_model)
    model.eval()
    
    hidden_size = model.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    sd = torch.load(args.forge_ckpt, map_location=device)
    s1.load_state_dict(sd, strict=False)
    s1.eval()

    adapter = None
    if not args.disable_adapter:
        adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
        adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))
        adapter.eval()
        print(f"--- M11 AUDIT: DEEP BRIDGE ACTIVE (N={args.num_samples * 3}) ---")
    else:
        print(f"--- M11 AUDIT: LITE FLOOR (N={args.num_samples * 3}) ---")

    target_words = ["yes", "no", "engineer", "analyst", "box", "shelf", "cabinet", "drawer", "suitcase", "trophy", "riley", "alex", "morgan", "casey"]
    target_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in target_words]
    id_to_class = {tid: i for i, tid in enumerate(target_ids)}
    num_classes = len(target_words) + 1
    
    english_head = M10cEnglishHead(hidden_size=hidden_size, num_classes=num_classes).to(device)
    english_head.load_state_dict(torch.load(args.head_ckpt, map_location=device))
    english_head.eval()

    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH); push_socket.connect(f"tcp://127.0.0.1:{args.port}")
    pull_socket = context.socket(zmq.PULL); pull_socket.connect(f"tcp://127.0.0.1:{args.port + 1}")

    all_y_true = []
    all_y_pred = []
    
    for seed in [42, 7, 123]:
        y_true, y_pred = run_single_seed(seed, args, device, model, tokenizer, s1, adapter, english_head, target_words, id_to_class, num_classes, push_socket, pull_socket)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    # 4. Consolidated Metrics
    label_indices = list(range(num_classes))
    label_names = target_words + ["unknown"]
    
    report_dict = classification_report(all_y_true, all_y_pred, labels=label_indices, target_names=label_names, output_dict=True, zero_division=0)
    conf_mtx = confusion_matrix(all_y_true, all_y_pred, labels=label_indices).tolist()
    
    mean_acc = report_dict["accuracy"]
    macro_f1 = report_dict["macro avg"]["f1-score"]

    print(f"\n--- CONSOLIDATED PUBLICATION METRICS ---")
    print(f"Overall Accuracy: {mean_acc:.4f}")
    print(f"Macro F1 Score:   {macro_f1:.4f}")
    
    output_filename = "final_floor_lock.json" if args.disable_adapter else "final_bridge_audit.json"
    with open(output_dir / output_filename, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_samples": len(all_y_true),
            "accuracy": mean_acc,
            "macro_f1": macro_f1,
            "classification_report": report_dict,
            "confusion_matrix": conf_mtx
        }, f, indent=2)
    print(f"Report written to {output_dir / output_filename}")

if __name__ == "__main__":
    main()
