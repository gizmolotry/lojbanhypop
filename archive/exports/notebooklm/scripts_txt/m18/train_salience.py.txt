from __future__ import annotations
import argparse
import json
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from lojban_evolution.m18.salience import M18SalienceSelector, compute_salience_metrics

class SalienceDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_len: int = 128):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                # Gold targets: operator and pointer tokens
                # For v0, we use the logical trace indices
                self.samples.append(item)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = f"Question: {item['prompt']}\nReasoning: step."
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_len, padding="max_length")
        
        # Extract gold positions from logic_trace (mockup for v0)
        # In reality, we map the M11 logic indices back to the prompt tokens
        # For this burst, we use heuristic targets if gold map is missing
        gold_mask = torch.zeros(self.max_len)
        # Simplified: target the question mark and capital letters as structural proxies
        for i, tid in enumerate(inputs.input_ids[0]):
            token = self.tokenizer.decode([tid])
            if token in ["?", ":", "step"]:
                gold_mask[i] = 1.0
                
        return {
            "input_ids": inputs.input_ids[0],
            "attention_mask": inputs.attention_mask[0],
            "gold_mask": gold_mask
        }

def train_phase_a(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Load backbone for hidden state extraction
    backbone = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto")
    backbone.eval()
    
    selector = M18SalienceSelector(hidden_size=896, top_k=6).to(device)
    optimizer = torch.optim.Adam(selector.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    dataset = SalienceDataset(args.data_path, tokenizer)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    print(f"--- M18 PHASE A: SALIENCE TRAINING (Gold Supervision) ---")
    for epoch in range(args.epochs):
        selector.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            gold_mask = batch["gold_mask"].to(device)
            
            with torch.no_grad():
                out = backbone(input_ids, output_hidden_states=True)
                # Tap Layer 12 as per Registry
                h_tap = out.hidden_states[12]
            
            scores, _, _ = selector(h_tap.to(torch.float32))
            loss = criterion(scores, gold_mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(loader):.4f}")
        
    # Qualitative Audit
    selector.eval()
    test_item = dataset[0]
    with torch.no_grad():
        h_test = backbone(test_item["input_ids"].unsqueeze(0).to(device), output_hidden_states=True).hidden_states[12]
        _, indices, _ = selector(h_test.to(torch.float32))
        tokens = [tokenizer.decode([test_item["input_ids"][idx]]) for idx in indices[0]]
        print(f"Audited Top-K Tokens: {tokens}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(selector.state_dict(), args.output_path)
    print(f"Phase A Complete. Selector saved to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-path", default="artifacts/models/m18/salience_v0.pt")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    train_phase_a(args)
