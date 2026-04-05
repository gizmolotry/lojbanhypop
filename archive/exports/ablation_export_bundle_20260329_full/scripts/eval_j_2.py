from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lojban_evolution.j_series_eval import run_j2_paraphrase_explosion


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-2: paraphrase explosion invariance audit.")
    p.add_argument("--j1-artifact", type=Path, required=True)
    p.add_argument("--variants-per-graph", type=int, default=64)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", type=Path, default=Path("runs/j_series/j-2.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_j2_paraphrase_explosion(args.j1_artifact, args.output, variants_per_graph=max(1, args.variants_per_graph), seed=args.seed)
    print(f"Wrote: {args.output}")
    print(f"invariance_rate: {payload['metrics']['invariance_rate']:.4f}")


if __name__ == "__main__":
    main()
