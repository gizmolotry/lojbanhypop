from __future__ import annotations
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lojban_evolution.m18.salience import M18SalienceSelector
from lojban_evolution.m18.graph_induction import M18RelationalInterpreter, M18BiasCompiler
from lojban_evolution.m18.attention_patch import apply_m18_bias
from lojban_evolution.m18.registry import M18_REGISTRY

class JointDataset(Dataset):
    def __init__(self, data_path: str):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_fn(batch):
    return batch

def train_phase_c(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reg = M18_REGISTRY["M18-v0"]
    defaults = reg["defaults"]
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    backbone = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager" # Required for hooks
    )
    # Backbone is frozen
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    
    # Initialize Symbiote Stack
    selector = M18SalienceSelector(hidden_size=896, top_k=6).to(device)
    if Path(args.selector_path).exists():
        selector.load_state_dict(torch.load(args.selector_path, map_location=device))
    
    interpreter = M18RelationalInterpreter(hidden_size=896, num_relations=8, ontology=args.ontology).to(device)
    if Path(args.interpreter_path).exists():
        interpreter.load_state_dict(torch.load(args.interpreter_path, map_location=device))
    
    compiler = M18BiasCompiler(num_relations=8, num_heads=backbone.config.num_attention_heads, hidden_size=896).to(device)
    
    # Joint stack cast to BFloat16
    selector = selector.to(dtype=torch.bfloat16)
    interpreter = interpreter.to(dtype=torch.bfloat16)
    compiler = compiler.to(dtype=torch.bfloat16)
    
    params = list(selector.parameters()) + list(interpreter.parameters()) + list(compiler.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    dataset = JointDataset(args.data_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    print(f"--- M18 PHASE C: JOINT CONTROLLER TRAINING (Ontology: {args.ontology}) ---")
    
    intervention_layers = defaults["intervention_layers"]
    intervention_config = {
        layer_idx: list(range(backbone.config.num_attention_heads))
        for layer_idx in intervention_layers
    }

    for epoch in range(args.epochs):
        selector.train(); interpreter.train(); compiler.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            batch_loss = 0
            
            for item in batch:
                prompt = f"Question: {item['prompt']}\nReasoning: step."
                target_answer = item['answer'].lower().strip()
                full_text = f"{prompt}\nAnswer: {target_answer}"
                
                # --- PASS 1: CLEAN ANALYSIS ---
                inputs_p1 = tokenizer(prompt, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    out_p1 = backbone(**inputs_p1, output_hidden_states=True)
                    h_tap = out_p1.hidden_states[defaults["tap_layer"]]
                    
                # --- SYMBIOTE: GRAPH INDUCTION ---
                # We need gradients to flow through here
                scores, top_k_indices, mask = selector(h_tap)
                
                salient_embs = torch.gather(
                    h_tap, 1, 
                    top_k_indices.unsqueeze(-1).expand(-1, -1, h_tap.shape[-1])
                )
                
                adj_matrix = interpreter(salient_embs)
                
                # We compile the bias tensor for the FULL text length to support teacher forcing
                inputs_p2 = tokenizer(full_text, return_tensors="pt").to(device)
                full_seq_l = inputs_p2.input_ids.shape[1]
                
                bias_tensor = compiler.compile(adj_matrix, top_k_indices, full_seq_l)
                
                layer_biases = {layer_idx: bias_tensor for layer_idx in intervention_layers}
                
                # --- PASS 2: BIASED EXECUTION (TEACHER FORCING) ---
                labels = inputs_p2.input_ids.clone()
                # Mask out prompt from loss
                prompt_len = inputs_p1.input_ids.shape[1]
                labels[0, :prompt_len] = -100
                
                with apply_m18_bias(backbone, intervention_layers, intervention_config, layer_biases):
                    # We pass output_attentions=True to trigger the hook on eager attention
                    out_p2 = backbone(
                        **inputs_p2,
                        labels=labels,
                        output_attentions=True
                    )
                    
                loss = out_p2.loss
                loss.backward()
                batch_loss += loss.item()
                
            optimizer.step()
            total_loss += batch_loss / len(batch)
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
            
    # Save the full Joint Controller
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(selector.state_dict(), out_dir / "selector_final.pt")
    torch.save(interpreter.state_dict(), out_dir / "interpreter_final.pt")
    torch.save(compiler.state_dict(), out_dir / "compiler_final.pt")
    print(f"Phase C Complete. Joint stack saved to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--selector-path", required=True)
    parser.add_argument("--interpreter-path", required=True)
    parser.add_argument("--ontology", choices=["U", "L"], default="U")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_phase_c(args)
