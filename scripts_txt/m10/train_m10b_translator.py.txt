from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import json
from datetime import datetime, timezone

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.adapter import M10bTranslationAdapter
from lojban_evolution.experiment import generate_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description="M10b: Generative Translation Training.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_TRANSLATOR")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Systems (Frozen Backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=args.local_files_only, device_map="auto")
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    hidden_size = backbone.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    # M11-compatible loading for System 1
    sd = torch.load(args.forge_ckpt, map_location=device)
    s1.load_state_dict(sd, strict=False)
    s1.eval()
    for p in s1.parameters():
        p.requires_grad = False

    # 2. Initialize M10b Translator Adapter (Trainable)
    adapter = M10bTranslationAdapter(hidden_size=hidden_size).to(device)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    # 3. Curriculum Data
    ds = generate_dataset(size=2000, seed=42, profile="diverse_v3")
    train_ds, _, _ = split_dataset(ds)

    print(f"\n--- M10b GENERATIVE TRANSLATOR INITIATED ---")
    print("Goal: Learn Logic -> English resolution pass.")

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]

        # A. Semantic Premise Extraction (S2 frozen pass)
        prefix = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prefix, return_tensors="pt").to(device)
        target_text = " Answer: " + item.answer
        targets = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)

        with torch.no_grad():
            out = backbone(**inputs, output_hidden_states=True)
            h_eng_premise = out.hidden_states[-1][:, -1, :]

            # B. Logical Translation (S1 frozen pass)
            op_vec, x_probs, _ = s1.build_graph(h_eng_premise, inputs.input_ids.shape[1])
            prompt_embs = backbone.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

        # C. Generative Cross-Attention pass
        opt.zero_grad()

        # We teacher-force the target answer
        full_ids = torch.cat([inputs.input_ids, targets], dim=1)
        outputs = backbone(input_ids=full_ids, output_hidden_states=True)
        h_answer = outputs.hidden_states[-1][:, inputs.input_ids.shape[1] - 1:-1, :]

        # Apply Translation Adapter
        h_final = adapter(h_answer, logic_state)

        # Calculate Logits via frozen LM head
        logits = backbone.lm_head(h_final)

        # Cross-Entropy Loss on the target tokens
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.train_steps} - Translation Loss: {loss.item():.4f} | Alpha: {adapter.alpha.item():.4f}")

    # 4. Save components
    torch.save(adapter.state_dict(), output_dir / "m10b_adapter.pt")
    print(f"Saved M10b Adapter to {output_dir / 'm10b_adapter.pt'}")


if __name__ == "__main__":
    main()
