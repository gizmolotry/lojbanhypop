from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from lojban_evolution.experiment import run_experiment
from lojban_evolution.repro import safe_git_commit, write_run_manifest
from lojban_evolution.storage import join_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run identity-tracking language evolution experiments.")
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-accept", type=int, default=3)
    parser.add_argument("--output-dir", type=str, default="artifacts/runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    payload = run_experiment(
        output_root=args.output_dir,
        iterations=args.iterations,
        seed=args.seed,
        dataset_size=args.dataset_size,
        max_accept_per_iteration=args.max_accept,
    )
    run_dir = payload["run_dir"]
    write_run_manifest(
        join_path(run_dir, "run_manifest.json"),
        {
            "script": "scripts/run_experiment.py",
            "argv": sys.argv[1:],
            "git_commit": safe_git_commit(repo_root),
            "config": payload["config"],
            "dataset_fingerprint": payload["dataset_fingerprint"],
            "environment": {
                "HF_HOME": os.environ.get("HF_HOME"),
                "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "outputs": {
                "run_dir": str(run_dir),
                "history_json": str(join_path(run_dir, "history.json")),
                "summary_md": str(join_path(run_dir, "summary.md")),
            },
        },
    )
    print("Experiment complete.")
    print(f"Run directory: {run_dir}")
    print(f"Final vocab size: {payload['final_language']['total_token_count']}")
    print(f"Test accuracy: {payload['test_metrics']['accuracy']:.4f}")
    print(f"Test avg tokens: {payload['test_metrics']['avg_tokens']:.4f}")
    print(f"Test parse success: {payload['test_metrics']['parse_success_rate']:.4f}")


if __name__ == "__main__":
    main()
