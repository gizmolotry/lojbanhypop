import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tabulate import tabulate

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m25.emergent_bridi import generate_m25_emergent_bridi_examples, M25LooseBridiDataset, m25_collate
from lojban_evolution.m28.baselines import run_m28_baseline_bundle
from lojban_evolution.m29.model import M29StarQFormerSymbiote
from lojban_evolution.m29.rl_training import train_m29_rl_symbiote
from lojban_evolution.m29.runtime import evaluate_m29_star_runtime

def main():
    print("===========================================")
    print("M29 FINAL PROOF: LOJBAN vs ENGLISH vs CHINESE")
    print("===========================================\n")
    
    # 1. Generate Universal Datasets
    print("1. Generating Synthetic Lojbanic Physics World...")
    dataset_size = 10000 # Massive size for final proof
    examples = generate_m25_emergent_bridi_examples(dataset_size, seed=42)
    
    train_size = int(dataset_size * 0.8)
    train_examples = examples[:train_size]
    eval_examples = examples[train_size:]
    
    vocab = build_vocab(examples)
    print(f"Generated {dataset_size} total examples.")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 2. Train the M29 Model using STaR
    print("\n2. Initializing M29 STaR Symbiote...")
    model = M29StarQFormerSymbiote(
        vocab_size=len(vocab),
        hidden_dim=64, # Increased capacity
        num_queries=12,
        target_vocab_size=7
    ).to(device)
    # Give the model a vocab attribute so the baseline script can find it
    model.vocab = vocab
    
    train_dataset = M25LooseBridiDataset(train_examples, vocab, max_symbols=32)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=m25_collate)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("\nStarting RL Grinding Phase (Warm-up + REINFORCE + Dense Syntax Rewards)...")
    history = train_m29_rl_symbiote(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=15,
        samples_per_prompt=20,
        device=device
    )
    
    print("\n3. Testing M29 on the Causal Mediation Gauntlet...")
    model.eval()
    
    # 4. Run the Cross-Lingual Evaluation Suite
    print("\n4. Running Cross-Lingual Evaluation Suite (Chinese & English CoT)...")
    results = run_m28_baseline_bundle(
        learned_model=model,
        train_examples=train_examples,
        eval_examples=eval_examples,
        epochs=10,
        batch_size=128,
        embedding_dim=32,
        hidden_dim=64,
        latent_bottleneck_dim=8,
        seed=42,
        max_symbols=32,
        device=device
    )
    
    print("\n===========================================")
    print("FINAL RESULTS TABLE")
    print("===========================================")
    
    table_data = []
    baseline_results = results["baseline_results"]
    for name, metrics in baseline_results.items():
        table_data.append([
            name, 
            f"{metrics['strict_accuracy']*100:.2f}%", 
            f"{metrics['avg_trace_tokens']:.1f}",
            f"{metrics.get('trace_causality_delta', 0.0)*100:.2f}%"
        ])
    
    table_data.append([
        "M29 Lojban Symbiote",
        f"{results['summary']['m28_learned_logebonic_accuracy']*100:.2f}%",
        f"{results['summary']['m28_learned_trace_token_count']:.1f}",
        f"{results['baseline_results']['learned_logebonic_trace']['trace_causality_delta']*100:.2f}%"
    ])
    
    print(tabulate(table_data, headers=["Architecture / CoT Type", "Accuracy", "Avg Trace Tokens", "Causal Delta"]))
    
    # Dump to JSON
    output_dir = Path("outputs/m29")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "final_proof_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
