from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lojban_evolution.j_series_eval import run_j1_graph_target


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-1: graph-target artifact extraction.")
    p.add_argument("--input-artifact", type=Path, default=None, help="Optional artifact with sample prompts (e.g., H5-OOD output).")
    p.add_argument("--output", type=Path, default=Path("runs/j_series/j-1.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_j1_graph_target(args.input_artifact, args.output)
    print(f"Wrote: {args.output}")
    print(f"schema_valid_rate: {payload['metrics']['schema_valid_rate']:.4f}")


if __name__ == "__main__":
    main()
