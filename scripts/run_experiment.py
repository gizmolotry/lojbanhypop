from __future__ import annotations

import argparse
from pathlib import Path

from lojban_evolution.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run identity-tracking language evolution experiments.")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-accept", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_experiment(
        output_root=args.output_dir,
        iterations=args.iterations,
        seed=args.seed,
        dataset_size=args.dataset_size,
        max_accept_per_iteration=args.max_accept,
    )
    print("Experiment complete.")
    print(f"Final vocab size: {payload['final_language']['total_token_count']}")
    print(f"Test accuracy: {payload['test_metrics']['accuracy']:.4f}")
    print(f"Test avg tokens: {payload['test_metrics']['avg_tokens']:.4f}")
    print(f"Test parse success: {payload['test_metrics']['parse_success_rate']:.4f}")


if __name__ == "__main__":
    main()
