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

from lojban_evolution.m20.family import M20_DICTIONARY_FIRST_GRID, M20_FAMILY_VERSION, M20_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402
from train_m20_dictionary import parse_args as parse_train_args  # noqa: E402
from train_m20_dictionary import run_train  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m20_dictionary_suite_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m20_dictionary_suite_{_timestamp()}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M20_REGISTRY["M20"]
    output_root = Path(args.output_root or registry["output_roots"]["suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _variant_args(base: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "factor_weight": float(variant.get("factor_weight", base.factor_weight)),
        "dictionary_commitment_weight": float(variant.get("dictionary_commitment_weight", base.dictionary_commitment_weight)),
        "quotient_invariance_weight": float(variant.get("quotient_invariance_weight", base.quotient_invariance_weight)),
        "brivi_lock_weight": float(variant.get("brivi_lock_weight", base.brivi_lock_weight)),
        "temperature_start": float(variant.get("temperature_start", base.temperature_start)),
        "temperature_end": float(variant.get("temperature_end", base.temperature_end)),
    }


def _summarize_seed_reports(seed_reports: list[dict[str, Any]], stable_threshold: float) -> dict[str, float]:
    metric_rows = [report["metrics"] for report in seed_reports]

    def collect(key: str) -> list[float]:
        return [float(row.get(key, 0.0) or 0.0) for row in metric_rows]

    strict = collect("strict_accuracy")
    locks = collect("lock_pass_rate")
    factor = collect("factorized_exact_accuracy")
    brivi = collect("brivi_gate_accuracy")
    quotient = collect("predicate_identity_stability")
    return {
        "mean_strict_accuracy": mean(strict) if strict else 0.0,
        "std_strict_accuracy": pstdev(strict) if len(strict) > 1 else 0.0,
        "mean_lock_pass_rate": mean(locks) if locks else 0.0,
        "mean_factorized_exact_accuracy": mean(factor) if factor else 0.0,
        "mean_brivi_gate_accuracy": mean(brivi) if brivi else 0.0,
        "mean_predicate_identity_stability": mean(quotient) if quotient else 0.0,
        "stable_seed_rate": sum(1.0 for value in strict if value >= float(stable_threshold)) / max(1, len(strict)),
        "avg_tokens": mean(collect("avg_tokens")) if metric_rows else 0.0,
        "accuracy_per_token": mean(collect("accuracy_per_token")) if metric_rows else 0.0,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M20_REGISTRY["M20"]
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seed_list)
    cells: dict[str, Any] = {}
    all_seed_reports: list[dict[str, Any]] = []
    selected_cells = {item.strip().upper() for item in str(args.cell_list).split(",") if item.strip()}
    grid = [cell for cell in M20_DICTIONARY_FIRST_GRID if not selected_cells or cell["cell_key"] in selected_cells]
    for cell in grid:
        cell_key = str(cell["cell_key"])
        variant = _variant_args(args, dict(cell.get("variant", {})))
        seed_reports: list[dict[str, Any]] = []
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
                    "--codebook-size",
                    str(int(args.codebook_size)),
                    "--embedding-dim",
                    str(int(args.embedding_dim)),
                    "--hidden-dim",
                    str(int(args.hidden_dim)),
                    "--temperature-start",
                    str(variant["temperature_start"]),
                    "--temperature-end",
                    str(variant["temperature_end"]),
                    "--factor-weight",
                    str(variant["factor_weight"]),
                    "--dictionary-commitment-weight",
                    str(variant["dictionary_commitment_weight"]),
                    "--quotient-invariance-weight",
                    str(variant["quotient_invariance_weight"]),
                    "--brivi-lock-weight",
                    str(variant["brivi_lock_weight"]),
                    "--device",
                    str(args.device),
                    "--output-root",
                    str(run_dir),
                    "--run-id",
                    f"{cell_key}_seed_{seed}",
                ]
            )
            seed_reports.append(run_train(train_args))
        all_seed_reports.extend(seed_reports)
        cells[cell_key] = {
            "cell_id": cell["cell_id"],
            "lock": cell["lock"],
            "label": cell["label"],
            "variant_spec": variant,
            "aggregate_metrics": _summarize_seed_reports(seed_reports, float(args.stable_threshold)),
            "seed_reports": [
                {
                    "seed": report["config"]["seed"],
                    "run_dir": report["run_dir"],
                    "checkpoint_path": report["checkpoint_path"],
                    "metrics": report["metrics"],
                    "lock_status": report["lock_status"],
                }
                for report in seed_reports
            ],
        }
    aggregate = _summarize_seed_reports(all_seed_reports, float(args.stable_threshold))
    report_path = run_dir / registry["report_names"]["suite"]
    validate_series_outputs("M", [registry["output_roots"]["suite"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M20.1.dictionary_first_suite", "scripts/m20/run_m20_dictionary_first_suite.py"),
        "track": "M20.1",
        "family_version": M20_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["suite"],
            "dag": registry["dags"]["suite"],
            "output_root": registry["output_roots"]["suite"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": {
            "seed_list": _parse_int_list(args.seed_list),
            "cell_list": [cell["cell_key"] for cell in grid],
            "train_size": int(args.train_size),
            "eval_size": int(args.eval_size),
            "epochs": int(args.epochs),
            "codebook_size": int(args.codebook_size),
            "stable_threshold": float(args.stable_threshold),
        },
        "aggregate_metrics": aggregate,
        "cells": cells,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M20 suite report written to {report_path}")
    print(
        "M20 suite metrics: "
        f"mean_strict={aggregate['mean_strict_accuracy']:.4f} "
        f"std={aggregate['std_strict_accuracy']:.4f} "
        f"mean_locks={aggregate['mean_lock_pass_rate']:.4f} "
        f"stable={aggregate['stable_seed_rate']:.4f}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M20_REGISTRY["M20"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Run the M20 dictionary-first retraining suite.")
    parser.add_argument("--seed-list", type=str, default="23,29")
    parser.add_argument("--cell-list", type=str, default="A,B,C,D,E,F")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--codebook-size", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--temperature-start", type=float, default=1.5)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--dictionary-commitment-weight", type=float, default=0.75)
    parser.add_argument("--quotient-invariance-weight", type=float, default=2.0)
    parser.add_argument("--brivi-lock-weight", type=float, default=1.0)
    parser.add_argument("--stable-threshold", type=float, default=0.70)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["suite"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
