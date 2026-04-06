from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import zmq
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.deep_adapter import M10eDeepTranslationAdapter
from lojban_evolution.experiment import generate_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description="M11 Diagnostic: Probability Shift Test.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Systems
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained(
        "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
        local_files_only=args.local_files_only,
        device_map="auto",
    )
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, args.base_model)
    model.eval()

    hidden_size = backbone.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    sd = torch.load(args.forge_ckpt, map_location=device)
    s1.load_state_dict(sd, strict=False)
    s1.eval()

    adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
    adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))
    adapter.eval()

    loj_start = tokenizer.convert_tokens_to_ids("<loj_0>")

    # 2. Data
    ds = generate_dataset(size=10, seed=42, profile="diverse_v3")

    print(f"\n--- M10 DIAGNOSTIC: PROBABILITY SHIFT TEST ---")

    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect(f"tcp://127.0.0.1:{args.port}")
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect(f"tcp://127.0.0.1:{args.port + 1}")

    for i, item in enumerate(ds[:5]):
        prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :]

            # Get Logic
            push_socket.send_json({"premise": h_eng.tolist(), "prompt_len": inputs.input_ids.shape[1], "trace": list(item.trace), "tier": "hard"})
            logic_string = pull_socket.recv_json()["logic_string"]

            # Get Logic Tensors
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

            # Resolution Step (Start of Answer)
            res_prompt = f"{prompt}\n<logic> {logic_string} </logic>\nAnswer:"
            res_inputs = tokenizer(res_prompt, return_tensors="pt").to(device)
            res_outputs = model(input_ids=res_inputs.input_ids, output_hidden_states=True)
            h_base = res_outputs.hidden_states[-1][:, -1:, :]

            # LM Head BEFORE Adapter
            logits_base = model.lm_head(h_base)
            probs_base = torch.softmax(logits_base, dim=-1)

            # LM Head AFTER Adapter
            h_final = adapter(h_base, logic_state)
            logits_final = model.lm_head(h_final)
            probs_final = torch.softmax(logits_final, dim=-1)

            # Analysis
            def get_top_5(probs):
                p, idx = torch.topk(probs[0, 0], 5)
                return [(tokenizer.decode([i]), p[j].item()) for j, i in enumerate(idx)]

            top_base = get_top_5(probs_base)
            top_final = get_top_5(probs_final)

            target_id = tokenizer.encode(" " + item.answer.lower().strip(), add_special_tokens=False)[0]
            prob_target_base = probs_base[0, 0, target_id].item()
            prob_target_final = probs_final[0, 0, target_id].item()

            print(f"\nItem {i + 1}: {item.prompt[:50]}...")
            print(f"Target: '{item.answer}' (ID: {target_id})")
            print(f"  Base Top-5:  {top_base}")
            print(f"  Final Top-5: {top_final}")
            print(f"  Target Prob: {prob_target_base:.6f} -> {prob_target_final:.6f} (Delta: {prob_target_final - prob_target_base:+.6f})")
            print(f"  Alpha:       {adapter.alpha.item():.4f}")


if __name__ == "__main__":
    main()
