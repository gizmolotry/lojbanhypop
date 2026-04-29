from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lojban_evolution.j_series_eval import run_j4_operator_curriculum


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-4: operator curriculum dataset build.")
    p.add_argument("--per-operator", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("runs/j_series/j-4.json"))
    p.add_argument("--dataset-output", type=Path, default=Path("runs/j_series/j-4_curriculum.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_j4_operator_curriculum(args.output, args.dataset_output, per_operator=max(1, args.per_operator), seed=args.seed)
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.dataset_output}")
    print(f"sample_count: {payload['metrics']['sample_count']:.0f}")


if __name__ == "__main__":
    main()
