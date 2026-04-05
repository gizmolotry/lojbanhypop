import json
import sys
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import zmq

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from lojban_evolution.experiment import generate_dataset, split_dataset
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.deep_adapter import M10eDeepTranslationAdapter
from src.lojban_evolution.m10.english_head import M10cEnglishHead

TARGET_WORDS = ["yes", "no", "engineer", "analyst", "box", "shelf", "cabinet", "drawer", "suitcase", "trophy", "riley", "alex", "morgan", "casey"]

def label(i):
    return TARGET_WORDS[i] if i < len(TARGET_WORDS) else "unknown"

def run_tests():
    print("\n--- M11 DEFINITIVE KILL TESTS ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model = "archive/results/m9/active/RESULTS_M9_SYNCED/synced_model"
    
    # 1. Setup
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    backbone = AutoModelForCausalLM.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, base_model)
    model.eval()
    
    hidden_size = backbone.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    s1.load_state_dict(torch.load("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt", map_location=device), strict=False)
    s1.eval()
    
    adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
    adapter.load_state_dict(torch.load("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m10e_deep_adapter.pt", map_location=device))
    adapter.eval()
    
    head = M10cEnglishHead(hidden_size=hidden_size, num_classes=len(TARGET_WORDS) + 1).to(device)
    head.load_state_dict(torch.load("archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m10c_head.pt", map_location=device))
    head.eval()

    context = zmq.Context()
    push = context.socket(zmq.PUSH); push.connect("tcp://127.0.0.1:5555")
    pull = context.socket(zmq.PULL); pull.connect("tcp://127.0.0.1:5556")

    # 2. Data
    all_ds = generate_dataset(size=200, seed=42, profile="diverse_v3")
    _, _, test_ds = split_dataset(all_ds)
    samples = test_ds[:50]

    # TEST 1: Alpha Sweep
    print("\n[TEST 1] Alpha Sweep Diagnostic")
    alpha_settings = [("base", None), ("trained", 0.005), ("mid", 0.05), ("high", 0.2)]
    
    sweep_results = []
    for name, a_val in alpha_settings:
        correct = 0
        ratios = []
        for item in samples:
            with torch.no_grad():
                inputs = tokenizer(f"Question: {item.prompt}\nReasoning: step.", return_tensors="pt").to(device)
                h_eng = model(**inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
                
                # Logic
                push.send_json({"premise": h_eng.tolist(), "prompt_len": inputs.input_ids.shape[1]})
                logic_str = pull.recv_json()["logic_string"]
                op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
                ptr_vecs = torch.matmul(x_probs, model.get_input_embeddings()(inputs.input_ids))
                logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)
                
                # Base State
                res_prompt = f"Question: {item.prompt}\nReasoning: step.\n<logic> {logic_str} </logic>\nAnswer:"
                h_base = model(input_ids=tokenizer(res_prompt, return_tensors="pt").input_ids.to(device), output_hidden_states=True).hidden_states[-1][:, -1:, :]
                
                if a_val is None:
                    h_final = h_base
                    ratio = 0.0
                else:
                    adapter.alpha.data.fill_(a_val)
                    h_final = adapter(h_base, logic_state)
                    ratio = torch.norm(h_final - h_base) / torch.norm(h_base)
                
                probs = torch.softmax(head(h_final), dim=-1)[0]
                pred = label(torch.argmax(probs).item())
                correct += int(pred == item.answer.lower().strip())
                ratios.append(ratio.item() if torch.is_tensor(ratio) else ratio)
        
        acc = correct / len(samples)
        avg_ratio = sum(ratios) / len(samples)
        sweep_results.append({"mode": name, "acc": acc, "ratio": avg_ratio})
        print(f"  Mode {name:8} | Acc: {acc:.2f} | Delta Ratio: {avg_ratio:.2f}")

    # TEST 3: Distribution Check (On trained alpha)
    print("\n[TEST 3] Distribution Sanity Check")
    adapter.alpha.data.fill_(0.005)
    preds = {}
    for item in samples:
        with torch.no_grad():
            inputs = tokenizer(f"Question: {item.prompt}\nReasoning: step.", return_tensors="pt").to(device)
            h_eng = model(**inputs, output_hidden_states=True).hidden_states[-1][:, -1, :]
            push.send_json({"premise": h_eng.tolist(), "prompt_len": inputs.input_ids.shape[1]})
            logic_str = pull.recv_json()["logic_string"]
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            ptr_vecs = torch.matmul(x_probs, model.get_input_embeddings()(inputs.input_ids))
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)
            h_base = model(input_ids=tokenizer(f"Question: {item.prompt}\nReasoning: step.\n<logic> {logic_str} </logic>\nAnswer:", return_tensors="pt").input_ids.to(device), output_hidden_states=True).hidden_states[-1][:, -1:, :]
            h_final = adapter(h_base, logic_state)
            pred = label(torch.argmax(head(h_final)).item())
            preds[pred] = preds.get(pred, 0) + 1
    
    print(f"  Class Distribution: {preds}")

if __name__ == "__main__":
    run_tests()
