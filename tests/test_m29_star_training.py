import pytest
import torch
from torch.utils.data import DataLoader

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m25.emergent_bridi import generate_m25_emergent_bridi_examples, M25LooseBridiDataset, m25_collate
from lojban_evolution.m29.model import M29StarQFormerSymbiote
from lojban_evolution.m29.star_training import train_m29_star_symbiote

def test_star_training_loop():
    # 1. Setup minimal dataset
    eval_size = 10
    examples = generate_m25_emergent_bridi_examples(eval_size, seed=42)
    vocab = build_vocab(examples)
    
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=32)
    loader = DataLoader(dataset, batch_size=5, shuffle=False, collate_fn=m25_collate)
    
    # 2. Setup Model
    model = M29StarQFormerSymbiote(
        vocab_size=len(vocab),
        hidden_dim=16,
        num_queries=3,
        target_vocab_size=7
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 3. Run 1 epoch of STaR
    history = train_m29_star_symbiote(
        model=model,
        train_loader=loader,
        optimizer=optimizer,
        epochs=1,
        samples_per_prompt=2, # Keep low for fast test
        device="cpu"
    )
    
    assert len(history) == 1
    assert "loss" in history[0]
