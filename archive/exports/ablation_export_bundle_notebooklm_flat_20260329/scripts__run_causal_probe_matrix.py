from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List


@dataclass
class CellResult:
    regime: str
    k: float
    adapter_dir: str
    gate_json: str
    mean_final_lift: float
    mean_symbolic_lift: float
    gate_pass: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a causal probe matrix over rigidity k and positional regime (RoPE vs NoPE)."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/causal_probe"))
    parser.add_argument("--k-values", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--regimes", nargs="+", choices=["rope", "nope"], default=["rope", "nope"])
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[7, 11])
    parser.add_argument("--sample-size", type=int, default=24)
    parser.add_argument("--dataset-size", type=int, default=1000)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--max-steps", type=int, default=11)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--adapter-init", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--execute", action="store_true", help="Actually run training + eval. Default is dry run.")
    return parser.parse_args()


def iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def k_weights(k: float) -> dict:
    return {
        "trace_loss_multiplier": 1.0 + k,
        "semantic_unambiguity_weight": 0.05 * k,
        "compositional_consistency_weight": 0.10 * k,
        "roundtrip_consistency_weight": 0.05 * k,
        "coverage_regularization_weight": 0.03 * k,
        "compression_regularization_weight": 0.02 * k,
        "trajectory_balance_weight": 0.01 * k,
        "embedding_anchor_weight": 0.02 * k,
    }


def linear_slope(xs: Iterable[float], ys: Iterable[float]) -> float:
    x = list(xs)
    y = list(ys)
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0:
        return 0.0
    return num / den


def run_cmd(cmd: list[str], execute: bool) -> None:
    print("$ " + " ".join(cmd))
    if not execute:
        return
    subprocess.run(cmd, check=True)


def summarize_and_write(
    out_dir: Path,
    matrix_id: str,
    rows: list[CellResult],
) -> None:
    by_regime: dict[str, list[CellResult]] = {}
    for row in rows:
        by_regime.setdefault(row.regime, []).append(row)
    for r in by_regime.values():
        r.sort(key=lambda x: x.k)

    analysis = {}
    for regime, r in by_regime.items():
        ks = [x.k for x in r]
        symbolic = [x.mean_symbolic_lift for x in r]
        final = [x.mean_final_lift for x in r]
        slope_symbolic = linear_slope(ks, symbolic)
        slope_final = linear_slope(ks, final)
        baseline_final = final[0] if final else 0.0
        min_final = min(final) if final else 0.0
        analysis[regime] = {
            "n": len(r),
            "slope_symbolic_lift_vs_k": slope_symbolic,
            "slope_final_lift_vs_k": slope_final,
            "baseline_final_lift_k0": baseline_final,
            "min_final_lift": min_final,
            "max_final_drop_from_k0": baseline_final - min_final,
        }

    payload = {
        "timestamp_utc": iso_utc(),
        "matrix_id": matrix_id,
        "cells": [asdict(x) for x in rows],
        "analysis": analysis,
    }
    json_path = out_dir / "causal_probe_matrix.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Causal Probe Matrix Summary",
        "",
        f"- Matrix ID: `{matrix_id}`",
        f"- Timestamp: `{payload['timestamp_utc']}`",
        "",
        "## Analysis by Regime",
        "",
    ]
    for regime, a in analysis.items():
        lines.append(
            f"- {regime}: slope(symbolic_lift~k)={a['slope_symbolic_lift_vs_k']:.6f}, "
            f"slope(final_lift~k)={a['slope_final_lift_vs_k']:.6f}, "
            f"max_final_drop_from_k0={a['max_final_drop_from_k0']:.6f}"
        )
    lines.extend(["", "## Cell Results", ""])
    for row in rows:
        lines.append(
            f"- {row.regime} k={row.k:.3f}: "
            f"final={row.mean_final_lift:.6f}, symbolic={row.mean_symbolic_lift:.6f}, gate={row.gate_pass}"
        )
    md_path = out_dir / "causal_probe_matrix.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


def main() -> None:
    args = parse_args()
    matrix_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / matrix_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[CellResult] = []
    for regime in args.regimes:
        for k in args.k_values:
            tag = f"{regime}_k{str(k).replace('.', 'p')}"
            adapter_dir = out_dir / f"adapter_{tag}"
            gate_json = out_dir / f"gate_{tag}.json"
            weights = k_weights(k)

            train_cmd = [
                args.python_bin,
                "scripts/train_lora.py",
                "--base-model",
                args.base_model,
                "--dataset",
                str(args.dataset),
                "--output-dir",
                str(adapter_dir),
                "--seed",
                str(args.train_seed),
                "--max-steps",
                str(args.max_steps),
                "--per-device-batch-size",
                str(args.per_device_batch_size),
                "--grad-accum",
                str(args.grad_accum),
                "--max-length",
                str(args.max_length),
                "--phase5-objectives",
                "--no-dynamic-anchor-miner",
                "--trace-loss-multiplier",
                str(weights["trace_loss_multiplier"]),
                "--semantic-unambiguity-weight",
                str(weights["semantic_unambiguity_weight"]),
                "--compositional-consistency-weight",
                str(weights["compositional_consistency_weight"]),
                "--roundtrip-consistency-weight",
                str(weights["roundtrip_consistency_weight"]),
                "--coverage-regularization-weight",
                str(weights["coverage_regularization_weight"]),
                "--compression-regularization-weight",
                str(weights["compression_regularization_weight"]),
                "--trajectory-balance-weight",
                str(weights["trajectory_balance_weight"]),
                "--embedding-anchor-weight",
                str(weights["embedding_anchor_weight"]),
            ]
            if args.adapter_init:
                train_cmd.extend(["--adapter-init", str(args.adapter_init)])
            if args.local_files_only:
                train_cmd.append("--local-files-only")
            if regime == "nope":
                train_cmd.append("--disable-rope")

            eval_cmd = [
                args.python_bin,
                "scripts/eval_hf_dual_mode_gate.py",
                "--base-model",
                args.base_model,
                "--adapter",
                str(adapter_dir),
                "--sample-size",
                str(args.sample_size),
                "--dataset-size",
                str(args.dataset_size),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--seeds",
                *[str(s) for s in args.eval_seeds],
                "--output",
                str(gate_json),
            ]
            if args.local_files_only:
                eval_cmd.append("--local-files-only")

            run_cmd(train_cmd, execute=args.execute)
            run_cmd(eval_cmd, execute=args.execute)

            if args.execute:
                payload = json.loads(gate_json.read_text(encoding="utf-8"))
                rows.append(
                    CellResult(
                        regime=regime,
                        k=float(k),
                        adapter_dir=str(adapter_dir),
                        gate_json=str(gate_json),
                        mean_final_lift=float(payload["mean_lifts"]["final_answer"]),
                        mean_symbolic_lift=float(payload["mean_lifts"]["symbolic"]),
                        gate_pass=bool(payload["gate_pass"]),
                    )
                )

    if args.execute:
        summarize_and_write(out_dir=out_dir, matrix_id=matrix_id, rows=rows)
    else:
        dry_run = {
            "timestamp_utc": iso_utc(),
            "matrix_id": matrix_id,
            "note": "Dry run only. Re-run with --execute to run training and evaluation.",
        }
        dry_path = out_dir / "causal_probe_dry_run.json"
        dry_path.write_text(json.dumps(dry_run, indent=2), encoding="utf-8")
        print(f"Wrote: {dry_path}")


if __name__ == "__main__":
    main()
