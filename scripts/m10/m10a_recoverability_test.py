from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import zmq
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.probe import M10aRecoverabilityProbe
from lojban_evolution.experiment import generate_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description="M10a: Recoverability Probe Training.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_PROBE")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Systems (Frozen)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    model.eval()

    hidden_size = model.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    if Path(args.forge_ckpt).exists():
        sd = torch.load(args.forge_ckpt, map_location=device)
        print("Loading M11-compatible forge checkpoint for M10a probe...")
        s1.load_state_dict(sd, strict=False)
    s1.eval()

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    for p in s1.parameters():
        p.requires_grad = False

    # 2. Identify Target Classes (First tokens of common answers)
    target_words = ["yes", "no", "engineer", "analyst", "box", "shelf", "cabinet", "drawer", "suitcase", "trophy", "riley", "alex", "morgan", "casey"]
    target_ids = []
    for word in target_words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        target_ids.append(ids[0])

    id_to_class = {tid: i for i, tid in enumerate(target_ids)}
    num_classes = len(target_ids) + 1

    # 3. Initialize Probe
    probe = M10aRecoverabilityProbe(hidden_size=hidden_size, num_slots=10, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    # 4. Data
    ds = generate_dataset(size=2000, seed=42, profile="diverse_v3")
    train_ds, _, test_ds = split_dataset(ds)

    print(f"\n--- M10a RECOVERABILITY PROBE INITIATED ---")
    print("Goal: Can we decode answer labels from frozen M9 logic tensors?")

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]

        # A. Semantic Scan (S2)
        prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :]

            # B. Logical Translation (S1)
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)

            # Full 10-slot logic state
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

        # C. Target Mapping
        ans_id = tokenizer.encode(" " + item.answer.lower().strip(), add_special_tokens=False)[0]
        target_class = id_to_class.get(ans_id, num_classes - 1)

        # D. Probe Training
        opt.zero_grad()
        logits = probe(logic_state)
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=device))
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.train_steps} - Probe Loss: {loss.item():.4f}")

    # 5. Validation
    probe.eval()
    correct = 0
    print("\nEvaluating Probe on held-out samples...")
    for item in test_ds[:50]:
        with torch.no_grad():
            inputs = tokenizer(f"Question: {item.prompt}\nReasoning: Let's think step by step.", return_tensors="pt").to(device)
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :]
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

            logits = probe(logic_state)
            pred_class = torch.argmax(logits, dim=-1).item()

            ans_id = tokenizer.encode(" " + item.answer.lower().strip(), add_special_tokens=False)[0]
            target_class = id_to_class.get(ans_id, num_classes - 1)

            if pred_class == target_class:
                correct += 1

    print(f"Probe Accuracy: {correct / 50:.4f}")

    torch.save(probe.state_dict(), output_dir / "m10a_probe.pt")
    print(f"Saved M10a Probe to {output_dir / 'm10a_probe.pt'}")


if __name__ == "__main__":
    main()
