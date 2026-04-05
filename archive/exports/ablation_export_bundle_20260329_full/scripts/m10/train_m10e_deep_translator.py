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
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.ArgumentParser(description="M10e: Deep Translator Training (SwiGLU).")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--load-ckpt", type=str, default=None, help="Resume from previous adapter checkpoint")
    parser.add_argument("--train-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-5) # Lower LR for generative stability
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Systems (Frozen)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, args.base_model)
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    
    hidden_size = backbone.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    sd = torch.load(args.forge_ckpt, map_location=device)
    s1.load_state_dict(sd, strict=False)
    s1.eval()
    for p in s1.parameters(): p.requires_grad = False

    # 2. Initialize M11-Native Deep Adapter (BLANK REBIRTH)
    print("M11: Initializing blank SwiGLU bridge (Guardrails active).")
    adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
    
    if args.load_ckpt and Path(args.load_ckpt).exists():
        print(f"M11: Resuming from checkpoint {args.load_ckpt}...")
        adapter.load_state_dict(torch.load(args.load_ckpt, map_location=device))
    else:
        # Set tiny initial alpha to prevent representational shock
        with torch.no_grad():
            adapter.alpha.data.fill_(0.001)
    
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)

    # 3. Training Loop
    ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
    train_ds, _, _ = split_dataset(ds)

    print(f"\n--- M10e DEEP TRANSLATOR INITIATED (M11 NATIVE) ---")

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]
        
        # A. Semantic Premise Scan
        prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_text = " Answer: " + item.answer
        targets = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :] # [1, H]
            
            # B. Logical Translation (S1)
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1) # [1, 10, H]
        
        # C. Deep Bridge Training
        opt.zero_grad()
        full_ids = torch.cat([inputs.input_ids, targets], dim=1)
        res_outputs = model(input_ids=full_ids, output_hidden_states=True)
        h_answer = res_outputs.hidden_states[-1][:, inputs.input_ids.shape[1]-1:-1, :]
        
        h_final = adapter(h_answer, logic_state)
        
        with torch.no_grad():
            delta = h_final - h_answer
            delta_ratio = torch.norm(delta) / (torch.norm(h_answer) + 1e-12)
        
        logits = model.lm_head(h_final)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        
        if delta_ratio > 5.0:
            loss = loss + (delta_ratio - 5.0) * 0.1
        
        loss.backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{args.train_steps} - Loss: {loss.item():.4f} | Alpha: {adapter.alpha.item():.4f} | Delta Ratio: {delta_ratio.item():.2f}")

        if (step + 1) % 1000 == 0:
            torch.save(adapter.state_dict(), output_dir / f"m10e_deep_adapter_step_{step+1}.pt")
            print(f"Generative Checkpoint saved at step {step+1}")

    torch.save(adapter.state_dict(), output_dir / "m10e_deep_adapter.pt")
    print(f"Saved M11-native Deep Adapter to {output_dir / 'm10e_deep_adapter.pt'}")

if __name__ == "__main__":
    main()
