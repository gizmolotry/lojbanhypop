from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
import zmq
from pathlib import Path
import sys
import json

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from src.lojban_evolution.m9.engine import M9System1
from src.lojban_evolution.m10.deep_adapter import M10eDeepTranslationAdapter
from src.lojban_evolution.m10.english_head import M10cEnglishHead
from lojban_evolution.experiment import generate_dataset, split_dataset


def main():
    parser = argparse.ArgumentParser(description="M10c: English-only Head Training on M11.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load All Frozen Components
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

    # Freeze all
    for p in model.parameters():
        p.requires_grad = False
    for p in s1.parameters():
        p.requires_grad = False
    for p in adapter.parameters():
        p.requires_grad = False

    # 2. Target Classes
    target_words = ["yes", "no", "engineer", "analyst", "box", "shelf", "cabinet", "drawer", "suitcase", "trophy", "riley", "alex", "morgan", "casey"]
    target_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in target_words]
    id_to_class = {tid: i for i, tid in enumerate(target_ids)}
    num_classes = len(target_ids) + 1

    # 3. Initialize M10c Head
    head = M10cEnglishHead(hidden_size=hidden_size, num_classes=num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    # 4. Data
    ds = generate_dataset(size=2000, seed=42, profile="diverse_v3")
    train_ds, _, _ = split_dataset(ds)

    print(f"\n--- M10c ENGLISH-ONLY HEAD TRAINING INITIATED ---")
    print("Goal: Force logic-to-English mapping by physically excluding Lojban.")

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]

        prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :]

            # Get Logic from S1
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

            # Get conditioned state from the deep adapter at the Answer: position.
            res_prompt = f"{prompt}\n<logic> simulated_logic </logic>\nAnswer:"
            res_inputs = tokenizer(res_prompt, return_tensors="pt").to(device)
            res_outputs = model(input_ids=res_inputs.input_ids, output_hidden_states=True)
            h_base = res_outputs.hidden_states[-1][:, -1:, :]

            h_conditioned = adapter(h_base, logic_state)

        # Train Head
        opt.zero_grad()
        logits = head(h_conditioned)

        ans_id = tokenizer.encode(" " + item.answer.lower().strip(), add_special_tokens=False)[0]
        target_class = id_to_class.get(ans_id, num_classes - 1)

        loss = F.cross_entropy(logits, torch.tensor([target_class], device=device))
        loss.backward()
        opt.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{args.train_steps} - English Head Loss: {loss.item():.4f}")

    # 5. Save Head
    torch.save(head.state_dict(), output_dir / "m10c_head.pt")
    print(f"Saved M10c English Head to {output_dir / 'm10c_head.pt'}")


if __name__ == "__main__":
    main()
