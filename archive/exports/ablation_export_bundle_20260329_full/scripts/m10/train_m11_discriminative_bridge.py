from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F
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
    parser = argparse.ArgumentParser(description="M11 Discriminative Bridge Training (Phase 4.5b).")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Frozen Systems
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, args.base_model)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    hidden_size = model.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    s1.load_state_dict(torch.load(args.forge_ckpt, map_location=device), strict=False)
    s1.eval()
    for p in s1.parameters(): p.requires_grad = False

    # 2. Target Classes
    target_words = ["yes", "no", "engineer", "analyst", "box", "shelf", "cabinet", "drawer", "suitcase", "trophy", "riley", "alex", "morgan", "casey"]
    target_ids = [tokenizer.encode(" " + w, add_special_tokens=False)[0] for w in target_words]
    id_to_class = {tid: i for i, tid in enumerate(target_ids)}
    num_classes = len(target_words) + 1

    # 3. Initialize Bridge & Head (Trainable)
    adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
    with torch.no_grad():
        adapter.alpha.data.fill_(0.001)
        
    head = M10cEnglishHead(hidden_size=hidden_size, num_classes=num_classes).to(device)
    
    # Opt jointly over Bridge + Head
    opt = torch.optim.AdamW(list(adapter.parameters()) + list(head.parameters()), lr=args.lr)

    # 4. Data
    ds = generate_dataset(size=2000, seed=42, profile="diverse_v3")
    train_ds, _, _ = split_dataset(ds)

    print(f"\n--- M11 DISCRIMINATIVE BRIDGE TRAINING INITIATED ---")
    print(f"Goal: Beat the 76.67% M11-lite floor.")

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]
        
        prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :]
            
            op_vec, x_probs, op_idx = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)

            # Reconstruct actual logic string for System 2 context
            tokens = [f"<loj_{op_idx.item()}>"]
            for i in range(s1.num_x_slots):
                ptr_idx = torch.argmax(x_probs[0, i], dim=-1).item()
                tokens.append(f"<loj_{2000 + ptr_idx}>")
            actual_logic_string = " ".join(tokens)

            # Get 'Answer:' token state with ACTUAL logic string
            res_prompt = f"{prompt}\n<logic> {actual_logic_string} </logic>\nAnswer:"
            res_inputs = tokenizer(res_prompt, return_tensors="pt").to(device)
            res_outputs = model(input_ids=res_inputs.input_ids, output_hidden_states=True)
            h_base = res_outputs.hidden_states[-1][:, -1:, :]
        
        # Discriminative Training
        opt.zero_grad()
        h_final = adapter(h_base, logic_state)
        logits = head(h_final)
        
        ans_id = tokenizer.encode(" " + item.answer.lower().strip(), add_special_tokens=False)[0]
        target_class = id_to_class.get(ans_id, num_classes - 1)
        
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=device))
        loss.backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{args.train_steps} - Loss: {loss.item():.4f} | Alpha: {adapter.alpha.item():.4f}")
            
        if (step + 1) % 500 == 0:
            torch.save(adapter.state_dict(), output_dir / f"m11_native_adapter_step_{step+1}.pt")
            print(f"Checkpoint saved at step {step+1}")

    # 5. Save Final
    torch.save(adapter.state_dict(), output_dir / "m11_native_adapter.pt")
    torch.save(head.state_dict(), Path("archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m11_native_head.pt"))
    print(f"Saved M11-native Bridge and Head.")

if __name__ == "__main__":
    main()
