from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M19.32 / M19.3d hyperbolic typed-faithfulness suite.")
    parser.add_argument("--base-model", default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--eval-size", type=int, default=64)
    parser.add_argument("--audit-eval-size", type=int, default=10)
    parser.add_argument("--dictionary-eval-size", type=int, default=64)
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_typed_physics_suite.py"),
        "--track",
        "M19.32",
        "--base-model",
        str(args.base_model),
        "--typed-slot-layout",
        "gismu:2,cmavo:2,judri:4",
        "--arity-router-mode",
        "soft",
        "--geometry-mode",
        "hyperbolic",
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
        "--seed",
        str(int(args.seed)),
        "--epochs",
        str(int(args.epochs)),
        "--eval-size",
        str(int(args.eval_size)),
        "--audit-eval-size",
        str(int(args.audit_eval_size)),
        "--dictionary-eval-size",
        str(int(args.dictionary_eval_size)),
    ]
    if str(args.run_id).strip():
        cmd.extend(["--run-id", str(args.run_id)])
    subprocess.run(cmd, cwd=str(repo_root), check=True)


if __name__ == "__main__":
    main()
