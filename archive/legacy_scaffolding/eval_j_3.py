from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from lojban_evolution.j_series_eval import run_j3_stopgrad_isolation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="J-3: stop-grad isolation gate.")
    p.add_argument("--source-script", type=Path, default=Path("scripts/train_h5_persistent_vq_advisor.py"))
    p.add_argument("--output", type=Path, default=Path("runs/j_series/j-3.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_j3_stopgrad_isolation(args.source_script, args.output)
    print(f"Wrote: {args.output}")
    print(f"stopgrad_contract_pass: {payload['metrics']['stopgrad_contract_pass']:.0f}")


if __name__ == "__main__":
    main()
