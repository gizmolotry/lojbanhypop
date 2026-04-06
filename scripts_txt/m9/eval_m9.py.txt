from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import zmq
import json
from pathlib import Path
import sys
import math
import traceback
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from peft import PeftModel
from src.lojban_evolution.m9.engine import MoVGate, M9System1
from lojban_evolution.experiment import generate_dataset, split_dataset


class LojbanExtortionMask(LogitsProcessor):
    def __init__(self, base_vocab_size: int, total_vocab_size: int):
        self.start_idx = base_vocab_size
        self.end_idx = total_vocab_size

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if scores.shape[-1] > self.start_idx:
            scores[:, self.start_idx:self.end_idx] = -float("inf")
        return scores


def main():
    parser = argparse.ArgumentParser(description="M11 Unified Performance Audit.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--mode", type=str, default="m10e", choices=["natural", "mov", "extort", "m10b", "m10e", "m10c"])
    parser.add_argument("--schedule", type=str, default="full", choices=["full", "token1", "first2", "decay05", "decay07", "cutoff3"])
    parser.add_argument("--num-samples", type=int, default=20)

    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_AUDIT")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "m10_audit_report.json"

    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "mode": args.mode, "status": "failed", "details": []}

    try:
        # 1. ZeroMQ
        context = zmq.Context()
        push_socket = context.socket(zmq.PUSH)
        push_socket.connect(f"tcp://127.0.0.1:{args.port}")
        pull_socket = context.socket(zmq.PULL)
        pull_socket.connect(f"tcp://127.0.0.1:{args.port + 1}")

        # 2. Load Backbone & Adapter
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
        backbone = AutoModelForCausalLM.from_pretrained(
            "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
            local_files_only=args.local_files_only,
            device_map="auto",
        )
        backbone.resize_token_embeddings(len(tokenizer))
        model = PeftModel.from_pretrained(backbone, args.base_model)
        model.eval()

        # 3. Load System 1 (Forge)
        hidden_size = backbone.config.hidden_size
        s1 = M9System1(hidden_size=hidden_size).to(device)
        sd = torch.load(args.forge_ckpt, map_location=device)
        s1.load_state_dict(sd, strict=False)
        s1.eval()

        # 4. Load Mode-Specific Components
        translation_adapter = None
        m10c_head = None
        gate = None
        loj_start = tokenizer.convert_tokens_to_ids("<loj_0>")

        target_words = ["yes", "no", "engineer", "analyst", "box", "shelf", "cabinet", "drawer", "suitcase", "trophy", "riley", "alex", "morgan", "casey"]
        num_classes = len(target_words) + 1

        if args.mode in {"m10b", "m10e", "m10c"}:
            from src.lojban_evolution.m10.deep_adapter import M10eDeepTranslationAdapter

            translation_adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
            adapter_path = Path("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m10e_deep_adapter.pt")
            if not adapter_path.exists():
                raise FileNotFoundError(f"Missing M10e deep adapter checkpoint: {adapter_path}")
            translation_adapter.load_state_dict(torch.load(adapter_path, map_location=device))
            translation_adapter.eval()
            print("Loaded M10e Deep Translation Adapter (SwiGLU).")

            if args.mode == "m10c":
                from src.lojban_evolution.m10.english_head import M10cEnglishHead

                m10c_head = M10cEnglishHead(hidden_size=hidden_size, num_classes=num_classes).to(device)
                m10c_head.load_state_dict(torch.load("archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m10c_head.pt", map_location=device))
                m10c_head.eval()
        elif args.mode == "mov":
            gate = MoVGate(hidden_size=hidden_size, base_vocab_size=loj_start).to(device)
            gate.load_state_dict(torch.load("archive/results/m9/active/RESULTS_M9_MOV/mov_gate.pt", map_location=device))
            gate.eval()
            print("Loaded MoV Routing Gate.")

        ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
        _, _, test_ds = split_dataset(ds)

        print(f"\n--- M10 UNIFIED AUDIT (MODE: {args.mode.upper()}) ---")
        correct = 0

        for i, item in enumerate(test_ds[:args.num_samples]):
            cot_prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
            inputs = tokenizer(cot_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                premise_state = outputs.hidden_states[-1][:, -1, :].detach()

            # Logical Translation
            push_socket.send_json({"premise": premise_state.tolist(), "prompt_len": inputs.input_ids.shape[1], "target": item.answer})
            logic_payload = pull_socket.recv_json()
            logic_string = logic_payload["logic_string"]

            # Extract Logic Tensors for translation
            with torch.no_grad():
                op_vec, x_probs, _ = s1.build_graph(premise_state, inputs.input_ids.shape[1])
                prompt_embs = model.get_input_embeddings()(inputs.input_ids)
                ptr_vecs = torch.matmul(x_probs, prompt_embs)
                logic_tensor = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

            final_prompt = f"{cot_prompt}\n<logic> {logic_string} </logic>\nAnswer:"
            final_inputs = tokenizer(final_prompt, return_tensors="pt").to(device)

            gen_ids = []
            cur_ids = final_inputs.input_ids
            past_kv = None

            for t in range(8):
                with torch.no_grad():
                    out = model(input_ids=cur_ids, past_key_values=past_kv, output_hidden_states=True)
                    h_t = out.hidden_states[-1][:, -1:, :]
                    past_kv = out.past_key_values

                    if args.mode in {"m10b", "m10e", "m10c"}:
                        # TRANSIENT CONDITIONING SCHEDULER
                        gamma = 1.0
                        if args.schedule == "token1" and t > 0: gamma = 0.0
                        elif args.schedule == "first2" and t > 1: gamma = 0.0
                        elif args.schedule == "cutoff3" and t > 2: gamma = 0.0
                        elif args.schedule == "decay05": gamma = 0.5 ** t
                        elif args.schedule == "decay07": gamma = 0.7 ** t
                        
                        if gamma > 0:
                            # We temporarily scale alpha by gamma for this step
                            orig_alpha = translation_adapter.alpha.item()
                            translation_adapter.alpha.data.fill_(orig_alpha * gamma)
                            h_t = translation_adapter(h_t, logic_tensor)
                            translation_adapter.alpha.data.fill_(orig_alpha)

                    if args.mode == "m10c" and m10c_head is not None:
                        # Direct classification lever.
                        logits = m10c_head(h_t)
                        pred_class = torch.argmax(logits, dim=-1).item()
                        if pred_class < len(target_words):
                            prediction_full = target_words[pred_class]
                        else:
                            prediction_full = "unknown"
                        break

                    logits = model.lm_head(h_t)

                    if args.mode == "extort":
                        logits[:, :, loj_start:] = -float("inf")
                    elif args.mode == "mov":
                        lm_weight = model.get_output_embeddings().weight
                        logits = gate(h_t, lm_weight)

                    next_token_id = torch.argmax(logits, dim=-1)

                gen_ids.append(next_token_id.item())
                if next_token_id.item() == tokenizer.eos_token_id:
                    break
                cur_ids = next_token_id

            if args.mode != "m10c":
                prediction_full = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

            prediction = prediction_full.split()[0] if prediction_full else ""
            target = item.answer.lower().strip()
            is_correct = (prediction == target) or prediction.startswith(target)
            if is_correct:
                correct += 1

            print(f"Item {i+1}: Logic={logic_string[:30]}... | Pred={prediction} | Target={target} | Correct={is_correct}")
            report["details"].append({"prompt": item.prompt, "logic": logic_string, "prediction": prediction, "target": target, "correct": is_correct})

        report["accuracy"] = correct / args.num_samples
        report["status"] = "success"
        print(f"\nAudit Complete. Accuracy: {report['accuracy']:.4f}")

    except Exception as e:
        print(f"\nCRASH: {e}")
        traceback.print_exc()
        report["status"] = "crashed"
        report["error"] = str(e)

    finally:
        report_path.write_text(json.dumps(report, indent=2))
        print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()

