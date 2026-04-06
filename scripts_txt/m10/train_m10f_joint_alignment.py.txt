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
from lojban_evolution.experiment import generate_dataset, split_dataset

def main():
    parser = argparse.  ArgumentParser(description="M10f: Joint Manifold Alignment.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--forge-ckpt", required=True)
    parser.add_argument("--adapter-ckpt", default="archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m10e_deep_adapter.pt")
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--lr_backbone", type=float, default=1e-5)
    parser.add_argument("--lr_adapter", type=float, default=1e-4)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Systems
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=args.local_files_only)
    backbone = AutoModelForCausalLM.from_pretrained("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", local_files_only=args.local_files_only, device_map="auto")
    backbone.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(backbone, args.base_model)
    
    # 2. SELECTIVE UNFREEZING (M10f Protocol)
    # Freeze everything first
    for p in model.parameters(): p.requires_grad = False
    
    # Unfreeze Top 4 layers (20, 21, 22, 23)
    num_layers = len(model.base_model.model.model.layers)
    for i in range(num_layers - 4, num_layers):
        for p in model.base_model.model.model.layers[i].parameters():
            p.requires_grad = True
            
    # Unfreeze LM Head
    for p in model.base_model.model.lm_head.parameters():
        p.requires_grad = True
        
    print(f"M10f: Unfrozen LM Head and Top 4 layers ({num_layers-4}-{num_layers-1}).")
    
    # Load S1
    hidden_size = model.config.hidden_size
    s1 = M9System1(hidden_size=hidden_size).to(device)
    sd = torch.load(args.forge_ckpt, map_location=device)
    s1.load_state_dict(sd, strict=False)
    s1.eval()
    for p in s1.parameters(): p.requires_grad = False

    # Load Adapter
    adapter = M10eDeepTranslationAdapter(hidden_size=hidden_size).to(device)
    if Path(args.adapter_ckpt).exists():
        adapter.load_state_dict(torch.load(args.adapter_ckpt, map_location=device))
        print(f"M10f: Loaded base adapter from {args.adapter_ckpt}")
    
    # Joint Optimizer
    opt = torch.optim.AdamW([
        {"params": filter(lambda p: p.requires_grad, model.parameters()), "lr": args.lr_backbone},
        {"params": adapter.parameters(), "lr": args.lr_adapter}
    ])

    # 3. Training Loop
    ds = generate_dataset(size=5000, seed=42, profile="diverse_v3")
    train_ds, _, _ = split_dataset(ds)

    print(f"\n--- M10f JOINT ALIGNMENT INITIATED ---")

    for step in range(args.train_steps):
        item = train_ds[step % len(train_ds)]
        
        prompt = f"Question: {item.prompt}\nReasoning: Let's think step by step."
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        target_text = " Answer: " + item.answer
        targets = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            h_eng = outputs.hidden_states[-1][:, -1, :]
            op_vec, x_probs, _ = s1.build_graph(h_eng, inputs.input_ids.shape[1])
            prompt_embs = model.get_input_embeddings()(inputs.input_ids)
            ptr_vecs = torch.matmul(x_probs, prompt_embs)
            logic_state = torch.cat([op_vec.unsqueeze(1), ptr_vecs], dim=1)
        
        opt.zero_grad()
        full_ids = torch.cat([inputs.input_ids, targets], dim=1)
        res_outputs = model(input_ids=full_ids, output_hidden_states=True)
        h_answer = res_outputs.hidden_states[-1][:, inputs.input_ids.shape[1]-1:-1, :]
        
        h_final = adapter(h_answer, logic_state)
        logits = model.lm_head(h_final)
        
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))
        loss.backward()
        opt.step()
        
        if (step + 1) % 100 == 0:
            print(f"Step {step+1}/{args.train_steps} - Joint Loss: {loss.item():.4f} | Alpha: {adapter.alpha.item():.4f}")
            torch.save(adapter.state_dict(), output_dir / f"m10f_joint_adapter_step_{step+1}.pt")
            model.save_pretrained(output_dir / f"m10f_aligned_model_step_{step+1}")

    # 4. Save components
    torch.save(adapter.state_dict(), output_dir / "m10f_joint_adapter.pt")
    model.save_pretrained(output_dir / "m10f_aligned_model")
    print(f"M10f: Joint Alignment Complete. Saved to {output_dir}")

if __name__ == "__main__":
    main()
