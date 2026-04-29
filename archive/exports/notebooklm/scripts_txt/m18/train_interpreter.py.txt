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
from lojban_evolution.m18.graph_induction import M18RelationalInterpreter

class InterpreterDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, selector, backbone, device, max_len: int = 128):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.selector = selector
        self.backbone = backbone
        self.device = device
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = f"Question: {item['prompt']}\nReasoning: step."
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_len, padding="max_length")
        
        with torch.no_grad():
            input_ids = inputs.input_ids.to(self.device)
            out = self.backbone(input_ids, output_hidden_states=True)
            h_tap = out.hidden_states[12] # Layer 12 as per Registry
            _, top_k_indices, _ = self.selector(h_tap.to(torch.float32))
            
            # Extract salient embeddings [1, K, H]
            salient_embs = torch.gather(
                h_tap, 1, 
                top_k_indices.unsqueeze(-1).expand(-1, -1, h_tap.shape[-1])
            )
            
        # Gold Edges: (mockup for v0 warm-start)
        # In reality, we use the logic trace to define directed relations
        # Target shape: [K, K, R]
        gold_adj = torch.zeros(6, 6, 8)
        # Identity baseline: each node relates to itself on type 0
        for i in range(6):
            gold_adj[i, i, 0] = 1.0
        
        return {
            "salient_embs": salient_embs[0].cpu(),
            "gold_adj": gold_adj.cpu()
        }

def train_phase_b(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    backbone.eval()
    
    selector = M18SalienceSelector(hidden_size=896, top_k=6).to(device)
    selector.load_state_dict(torch.load(args.selector_path, map_location=device))
    selector.eval()
    
    interpreter = M18RelationalInterpreter(hidden_size=896, num_relations=8, ontology=args.ontology).to(device)
    # Align dtypes with backbone
    interpreter = interpreter.to(dtype=torch.bfloat16)
    
    optimizer = torch.optim.Adam(interpreter.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    dataset = InterpreterDataset(args.data_path, tokenizer, selector, backbone, device)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    print(f"--- M18 PHASE B: INTERPRETER WARM START (Ontology: {args.ontology}) ---")
    for epoch in range(args.epochs):
        interpreter.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            embs = batch["salient_embs"].to(device=device, dtype=torch.bfloat16)
            gold_adj = batch["gold_adj"].to(device=device, dtype=torch.bfloat16)
            
            pred_adj = interpreter(embs)
            loss = criterion(pred_adj, gold_adj)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(interpreter.state_dict(), args.output_path)
    print(f"Phase B Complete. Interpreter saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--selector-path", required=True)
    parser.add_argument("--ontology", choices=["U", "L"], default="U")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_phase_b(args)
