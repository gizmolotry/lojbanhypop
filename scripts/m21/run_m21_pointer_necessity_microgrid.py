from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lojban_evolution.m21.family import M21_FAMILY_VERSION, M21_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402
from train_m21_dynamic_bridi import parse_args as parse_train_args  # noqa: E402
from train_m21_dynamic_bridi import run_train  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m21_pointer_microgrid_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m21_pointer_microgrid_{_timestamp()}"


def _parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _metric(row: dict[str, Any], key: str) -> float:
    value = row.get("metrics", {}).get(key, row.get(key, 0.0)) if isinstance(row, dict) else 0.0
    return float(value or 0.0)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    def collect(key: str) -> list[float]:
        return [_metric(row, key) for row in rows]

    strict = collect("strict_accuracy")
    judri_delta = collect("judri_causal_delta")
    pointer_gap = collect("pointer_necessity_gap")
    no_judri = collect("no_judri_accuracy")
    return {
        "mean_strict_accuracy": mean(strict) if strict else 0.0,
        "std_strict_accuracy": pstdev(strict) if len(strict) > 1 else 0.0,
        "mean_judri_causal_delta": mean(judri_delta) if judri_delta else 0.0,
        "mean_no_judri_accuracy": mean(no_judri) if no_judri else 0.0,
        "mean_pointer_necessity_gap": mean(pointer_gap) if pointer_gap else 0.0,
        "mean_loss_pointer_necessity": mean(collect("loss_pointer_necessity")) if rows else 0.0,
        "mean_bridi_trace_exact_accuracy": mean(collect("bridi_trace_exact_accuracy")) if rows else 0.0,
        "mean_brivi_lock_violation_rate": mean(collect("brivi_lock_violation_rate")) if rows else 0.0,
        "mean_active_frames": mean(collect("mean_active_frames")) if rows else 0.0,
        "score": (mean(strict) if strict else 0.0) + 2.0 * (mean(judri_delta) if judri_delta else 0.0) - max(0.0, 0.02 - (mean(pointer_gap) if pointer_gap else 0.0)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M21_REGISTRY["M21"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Run a microgrid over M21 pointer necessity weights.")
    parser.add_argument("--weights", type=str, default="0.0,0.25,0.5,1.0,2.0")
    parser.add_argument("--seed-list", type=str, default="23,29")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--pointer-necessity-margin", type=float, default=0.05)
    parser.add_argument("--frame-necessity-weight", type=float, default=1.0)
    parser.add_argument("--brivi-lock-weight", type=float, default=1.5)
    parser.add_argument("--trace-weight", type=float, default=1.25)
    parser.add_argument("--answer-weight", type=float, default=1.25)
    parser.add_argument("--counterfactual-weight", type=float, default=1.25)
    parser.add_argument("--mdl-weight", type=float, default=0.01)
    parser.add_argument("--geometry-mode", type=str, choices=("euclidean", "poincare"), default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--poincare-max-norm", type=float, default=0.99)
    parser.add_argument("--hyperbolic-topology-weight", type=float, default=0.0)
    parser.add_argument("--judri-bridge-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judri-bridge-gate-temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["pointer_microgrid"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


def run_microgrid(args: argparse.Namespace) -> dict[str, Any]:
    registry = M21_REGISTRY["M21"]
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = _parse_float_list(args.weights)
    seeds = _parse_int_list(args.seed_list)
    cells: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for weight in weights:
        weight_key = f"w_{weight:g}".replace(".", "p")
        rows: list[dict[str, Any]] = []
        for seed in seeds:
            train_args = parse_train_args(
                [
                    "--train-size",
                    str(int(args.train_size)),
                    "--eval-size",
                    str(int(args.eval_size)),
                    "--epochs",
                    str(int(args.epochs)),
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--learning-rate",
                    str(float(args.learning_rate)),
                    "--seed",
                    str(int(seed)),
                    "--embedding-dim",
                    str(int(args.embedding_dim)),
                    "--hidden-dim",
                    str(int(args.hidden_dim)),
                    "--trace-weight",
                    str(float(args.trace_weight)),
                    "--answer-weight",
                    str(float(args.answer_weight)),
                    "--counterfactual-weight",
                    str(float(args.counterfactual_weight)),
                    "--brivi-lock-weight",
                    str(float(args.brivi_lock_weight)),
                    "--frame-necessity-weight",
                    str(float(args.frame_necessity_weight)),
                    "--mdl-weight",
                    str(float(args.mdl_weight)),
                    "--pointer-necessity-weight",
                    str(float(weight)),
                    "--pointer-necessity-margin",
                    str(float(args.pointer_necessity_margin)),
                    "--geometry-mode",
                    str(args.geometry_mode),
                    "--poincare-curvature",
                    str(float(args.poincare_curvature)),
                    "--poincare-max-norm",
                    str(float(args.poincare_max_norm)),
                    "--hyperbolic-topology-weight",
                    str(float(args.hyperbolic_topology_weight)),
                    "--judri-bridge-gate" if bool(args.judri_bridge_gate) else "--no-judri-bridge-gate",
                    "--judri-bridge-gate-temperature",
                    str(float(args.judri_bridge_gate_temperature)),
                    "--device",
                    str(args.device),
                    "--output-root",
                    str(run_dir),
                    "--run-id",
                    f"{weight_key}_seed_{seed}",
                ]
            )
            report = run_train(train_args)
            row = {"seed": int(seed), "pointer_necessity_weight": float(weight), "run_dir": report["run_dir"], "checkpoint_path": report["checkpoint_path"], "metrics": report["metrics"]}
            rows.append(row)
            all_rows.append(row)
        cells[weight_key] = {"pointer_necessity_weight": float(weight), "aggregate_metrics": _summarize(rows), "seed_reports": rows}
    best_key = max(cells, key=lambda key: float(cells[key]["aggregate_metrics"]["score"])) if cells else ""
    payload = {
        "series": series_metadata("M", "M21.pointer_necessity_microgrid", "scripts/m21/run_m21_pointer_necessity_microgrid.py"),
        "track": "M21.pointer_necessity",
        "family_version": M21_FAMILY_VERSION,
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": vars(args),
        "cells": cells,
        "aggregate_metrics": _summarize(all_rows),
        "best_cell": best_key,
        "best_metrics": cells.get(best_key, {}).get("aggregate_metrics", {}),
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path = run_dir / registry["report_names"]["pointer_microgrid"]
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"M21 pointer necessity microgrid report written to {report_path}")
    print(
        "M21 pointer microgrid best: "
        f"{best_key or 'none'} "
        f"strict={payload['best_metrics'].get('mean_strict_accuracy', 0.0):.4f} "
        f"judri_delta={payload['best_metrics'].get('mean_judri_causal_delta', 0.0):.4f} "
        f"gap={payload['best_metrics'].get('mean_pointer_necessity_gap', 0.0):.4f}"
    )
    return payload


if __name__ == "__main__":
    run_microgrid(parse_args())
