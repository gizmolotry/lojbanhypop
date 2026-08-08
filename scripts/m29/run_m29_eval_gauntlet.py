import torch
import json
import os
from pathlib import Path

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m25.emergent_bridi import generate_m25_emergent_bridi_examples
from lojban_evolution.m29.model import M29StarQFormerSymbiote
from lojban_evolution.m29.runtime import evaluate_m29_star_runtime

def main():
    print("Initializing M29 Causal Mediation Gauntlet...")
    
    # 1. Generate Evaluation Dataset
    eval_size = 100
    seed = 42
    print(f"Generating {eval_size} Emergent Bridi examples...")
    eval_examples = generate_m25_emergent_bridi_examples(eval_size, seed=seed)
    vocab = build_vocab(eval_examples)  # Simple vocabulary for English prompt
    
    # 2. Instantiate M29 Model
    model = M29StarQFormerSymbiote(
        vocab_size=len(vocab),
        hidden_dim=32,
        num_queries=5,
        target_vocab_size=7
    )
    
    # 3. Run the Gauntlet (Trace Swapping + Topology Corruption)
    print("Running evaluate_m29_star_runtime probes...")
    results = evaluate_m29_star_runtime(
        model=model,
        examples=eval_examples,
        vocab=vocab,
        batch_size=32,
        seed=seed
    )
    
    metrics = results["metrics"]
    
    print("\n===========================================")
    print("M29 EVALUATIVE GAUNTLET RESULTS")
    print("===========================================")
    print(f"Strict Accuracy:            {metrics['strict_accuracy']:.4f}")
    print(f"Trace Swapped Accuracy:     {metrics['trace_swapped_accuracy']:.4f}")
    print(f"Topology Corrupted Accuracy:{metrics['topology_corrupted_accuracy']:.4f}")
    print(f"Trace Causality Delta:      {metrics['trace_causality_delta']:.4f}")
    
    # 4. Save results
    output_dir = Path("outputs/m29")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "gauntlet_results.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
