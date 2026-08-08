import argparse
import json
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.data.datasets.emergent_bridi import (
    M25LooseBridiDataset,
    generate_m25_emergent_bridi_examples,
    m25_collate,
)
from lojban_evolution.models.baselines.controls import run_m28_baseline_bundle
from lojban_evolution.models.architectures.gflownet_symbiote import M29StarQFormerSymbiote
from lojban_evolution.m29.runtime import evaluate_m29_star_runtime
from lojban_evolution.m29.rl_training import train_m29_rl_symbiote


def accuracy_percent(metrics: dict) -> str:
    return f"{float(metrics.get('strict_accuracy', 0.0) or 0.0) * 100:.2f}%"


def print_accuracy_table(rows: list[tuple[str, str]]) -> None:
    name_width = max(len("Architecture / CoT Type"), *(len(name) for name, _ in rows))
    print(f"{'Architecture / CoT Type'.ljust(name_width)} | Accuracy")
    print(f"{'-' * name_width}-+-{'-' * len('Accuracy')}")
    for name, accuracy in rows:
        print(f"{name.ljust(name_width)} | {accuracy}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Training Shell")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("===========================================")
    print(f"UNIFIED MLOPS RUN: {config['experiment']['name'].upper()}")
    print("===========================================\n")

    DATASET_SIZE = config["data"]["dataset_size"]
    TRAIN_FRACTION = config["data"]["train_fraction"]
    SEED = config["experiment"]["seed"]
    EPOCHS = config["training"]["epochs"]
    BATCH_SIZE = config["data"]["batch_size"]
    MAX_SYMBOLS = config["data"]["max_symbols"]
    SAMPLES_PER_PROMPT = config["training"]["samples_per_prompt"]
    OUTPUT_PATH = Path(config["output"]["path"])

    print(f"1. Generating {DATASET_SIZE:,} synthetic Lojban math/logic problems...")
    examples = generate_m25_emergent_bridi_examples(DATASET_SIZE, seed=SEED, max_symbols=MAX_SYMBOLS)
    train_size = int(DATASET_SIZE * TRAIN_FRACTION)
    train_examples = examples[:train_size]
    eval_examples = examples[train_size:]
    vocab = build_vocab(examples)
    print(f"Generated {len(train_examples):,} train and {len(eval_examples):,} eval examples.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"\\n2. Initializing and training architecture: {config['model']['architecture']}...")
    model = M29StarQFormerSymbiote(
        vocab_size=len(vocab),
        hidden_dim=config["model"]["hidden_dim"],
        num_queries=config["model"]["num_queries"],
        target_vocab_size=config["model"]["target_vocab_size"],
    ).to(device)
    model.vocab = vocab

    train_loader = DataLoader(
        M25LooseBridiDataset(train_examples, vocab, max_symbols=MAX_SYMBOLS),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=m25_collate,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    history = train_m29_rl_symbiote(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        samples_per_prompt=SAMPLES_PER_PROMPT,
        device=device,
    )

    print("\\n3. Running Causal Probes...")
    runtime_results = evaluate_m29_star_runtime(
        model=model,
        examples=eval_examples,
        vocab=vocab,
        batch_size=BATCH_SIZE,
        device=device,
        seed=SEED,
    )

    print("\\n4. Training Chinese CoT and English CoT baselines on the same dataset...")
    baseline_bundle = run_m28_baseline_bundle(
        learned_model=model,
        train_examples=train_examples,
        eval_examples=eval_examples,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        embedding_dim=config["baselines"]["embedding_dim"],
        hidden_dim=config["baselines"]["hidden_dim"],
        latent_bottleneck_dim=config["baselines"]["latent_bottleneck_dim"],
        seed=SEED,
        max_symbols=MAX_SYMBOLS,
        device=device,
    )

    baseline_results = baseline_bundle["baseline_results"]
    rows = [
        ("M29 Lojban STaR Q-Former", accuracy_percent(runtime_results["metrics"])),
        ("Chinese CoT", accuracy_percent(baseline_results["full_chinese_cot"])),
        ("English CoT", accuracy_percent(baseline_results["full_english_cot"])),
    ]

    print("\\n===========================================")
    print("FINAL ACCURACY TABLE")
    print("===========================================")
    print_accuracy_table(rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(
            {
                "config": config,
                "train_size": len(train_examples),
                "eval_size": len(eval_examples),
                "history": history,
                "runtime_results": runtime_results,
                "baseline_bundle": baseline_bundle,
                "final_accuracy_table": [
                    {"architecture": name, "accuracy": accuracy}
                    for name, accuracy in rows
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\\nSaved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
