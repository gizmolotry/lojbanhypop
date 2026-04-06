from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lojban_evolution.j_series_eval import run_j5_adversarial_synthesis


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-5: adversarial synthesis + scope scaling + foil validation.")
    p.add_argument("--sample-count", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--novelty-threshold", type=float, default=0.30)
    p.add_argument("--strict-depth-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-attempt-multiplier", type=int, default=100)
    p.add_argument("--output", type=Path, default=Path("runs/j_series/j-5.json"))
    p.add_argument("--dataset-output", type=Path, default=Path("runs/j_series/j-5_adversarial.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_j5_adversarial_synthesis(
        output=args.output,
        dataset_output=args.dataset_output,
        sample_count=max(1, int(args.sample_count)),
        seed=int(args.seed),
        novelty_threshold=float(args.novelty_threshold),
        strict_depth_balance=bool(args.strict_depth_balance),
        max_attempt_multiplier=max(1, int(args.max_attempt_multiplier)),
    )
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.dataset_output}")
    print(f"generator_accept_rate: {payload['metrics']['generator_accept_rate']:.4f}")
    print(f"accepted_foil_pair_accuracy: {payload['metrics']['accepted_foil_pair_accuracy']:.4f}")


if __name__ == "__main__":
    main()
