import torch
import torch.optim as optim
import wandb
import os
import argparse
from tqdm import tqdm

from lojban_evolution.m29.model import M29GFlowNetSymbiote

class MockGauntletDataset(torch.utils.data.Dataset):
    def __init__(self, size=1000, seq_len=16):
        self.size = size
        self.seq_len = seq_len
        self.data = torch.randint(0, 1000, (size, seq_len))
        self.labels = torch.randint(0, 5, (size,)) # Assuming 5 answer labels
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "labels": self.labels[idx]
        }

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--ctb_clip", type=float, default=1.0, help="Gradient clipping for CTB to prevent early thrashing")
    parser.add_argument("--beta", type=float, default=1.0, help="Inverse temperature for Reward Shaping R = exp(-beta * L_CE)")
    parser.add_argument("--init_temp", type=float, default=5.0, help="Initial temperature for Tempered Exploration")
    parser.add_argument("--min_temp", type=float, default=1.0, help="Minimum temperature")
    args = parser.parse_args()

    wandb.init(project="lojban_evolution", config=args, name="M29_GFN_CTB_Proof")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = M29GFlowNetSymbiote(
        vocab_size=1000, 
        hidden_dim=128, 
        num_queries=12,
        target_vocab_size=7
    ).to(device)

    # We use a single optimizer, but you can decouple LR for Answer Head and Q-Former if needed
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Mock dataset for the proof of concept
    dataset = MockGauntletDataset(size=1000, seq_len=16)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    temperature = args.init_temp
    temp_decay = (args.init_temp - args.min_temp) / (args.epochs * 0.8) # Anneal over 80% of training

    for epoch in range(args.epochs):
        model.train()
        total_ctb = 0.0
        total_ce = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            target_answers = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                target_answers=target_answers,
                temperature=temperature,
                beta=args.beta
            )

            ctb_loss = outputs["ctb_loss"]
            answer_head_loss = outputs["answer_head_loss"]
            
            # Joint objective: The Answer Head minimizes CE, Q-Former minimizes CTB
            loss = ctb_loss + answer_head_loss
            loss.backward()

            # Gradient Clipping (specifically critical for CTB)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.ctb_clip)

            optimizer.step()

            total_ctb += ctb_loss.item()
            total_ce += answer_head_loss.item()
            
            pbar.set_postfix({"CTB": ctb_loss.item(), "CE": answer_head_loss.item(), "T": round(temperature, 2)})

        avg_ctb = total_ctb / len(dataloader)
        avg_ce = total_ce / len(dataloader)
        
        wandb.log({
            "train/ctb_loss": avg_ctb,
            "train/ce_loss": avg_ce,
            "train/temperature": temperature
        })
        
        # Anneal temperature
        temperature = max(args.min_temp, temperature - temp_decay)

    print("Training complete.")
    torch.save(model.state_dict(), "m29_gfn_symbiote.pt")
    
if __name__ == "__main__":
    train()
