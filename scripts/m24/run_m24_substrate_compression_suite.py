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

from lojban_evolution.m24.compression import (  # noqa: E402
    DEFAULT_M24_MDL_WEIGHT,
    M24_LOCKS,
    metric_lock_status,
    train_m24_substrate_compression,
)
from lojban_evolution.m24.family import M24_FAMILY_VERSION, M24_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m24_substrate_compression_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m24_substrate_compression_{_timestamp()}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M24_REGISTRY["M24"]
    output_root = Path(args.output_root or registry["output_roots"]["suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _collect(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = (
        "strict_accuracy",
        "predicted_trace_accuracy",
        "oracle_trace_accuracy",
        "shuffled_trace_accuracy",
        "random_trace_accuracy",
        "zero_trace_accuracy",
        "prompt_only_accuracy",
        "advisor_vs_prompt_delta",
        "m24_strict_delta_vs_prompt_only",
        "predicted_vs_shuffled_delta",
        "predicted_vs_random_delta",
        "oracle_trace_delta",
        "oracle_trained_oracle_trace_accuracy",
        "oracle_trained_predicted_trace_accuracy",
        "oracle_trained_shuffled_trace_accuracy",
        "oracle_trained_random_trace_accuracy",
        "oracle_trained_zero_trace_accuracy",
        "oracle_trained_trace_delta",
        "predicted_trace_gap_to_oracle_upper_bound",
        "cross_advisor_oracle_gap",
        "trace_advisor_delta",
        "bridi_trace_exact_accuracy",
        "gismu_accuracy",
        "cmavo_accuracy",
        "judri_accuracy",
        "packed_symbol_compression_ratio",
        "packed_symbol_to_prompt_ratio",
        "prompt_to_packed_symbol_ratio",
        "packed_to_prompt_ratio",
        "prompt_to_packed_ratio",
        "token_reduction_ratio",
        "compression_ratio",
        "substrate_token_count",
        "reference_token_count",
        "strict_accuracy_per_substrate_token",
        "accuracy_per_trace_token",
        "accuracy_per_packed_symbol",
        "substrate_claim_score",
        "m24_promotion_gate_pass_rate",
        "m24_promotion_candidate",
        "m24_gate_trace_beats_shuffled",
        "m24_gate_trace_beats_random",
        "m24_gate_trace_beats_zero",
        "m24_gate_trace_matches_oracle_upper_bound",
        "m24_gate_trace_beats_prompt_only",
        "m24_gate_packed_trace_shorter_than_prompt",
        "m24_gate_token_reduction_positive",
        "m24_gate_nonzero_exact_trace_reconstruction",
        "mdl_weight",
        "generator_parameter_max_delta_after_advisor",
        "generator_parameters_unchanged_after_advisor",
    )
    out: dict[str, float] = {}
    for key in keys:
        values = _collect(rows, key)
        out[f"mean_{key}"] = mean(values) if values else 0.0
        if key == "strict_accuracy":
            out["std_strict_accuracy"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M24_REGISTRY["M24"]
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seed_list)
    seed_reports: list[dict[str, Any]] = []
    for seed in seeds:
        result = train_m24_substrate_compression(
            train_size=int(args.train_size),
            eval_size=int(args.eval_size),
            generator_epochs=int(args.generator_epochs),
            advisor_epochs=int(args.advisor_epochs),
            prompt_epochs=int(args.prompt_epochs),
            batch_size=int(args.batch_size),
            generator_learning_rate=float(args.generator_learning_rate),
            advisor_learning_rate=float(args.advisor_learning_rate),
            seed=int(seed),
            embedding_dim=int(args.embedding_dim),
            hidden_dim=int(args.hidden_dim),
            advisor_hidden_dim=int(args.advisor_hidden_dim),
            max_frames=int(args.max_frames),
            max_places=int(args.max_places),
            max_entities=int(args.max_entities),
            trace_weight=float(args.trace_weight),
            answer_weight=float(args.answer_weight),
            mdl_weight=float(args.mdl_weight),
            trace_exact_surrogate_weight=float(args.trace_exact_surrogate_weight),
            clean_train_fraction=float(args.clean_train_fraction),
            clean_eval_fraction=float(args.clean_eval_fraction),
            device=str(args.device),
        )
        metrics = dict(result["metrics"])
        locks = metric_lock_status(metrics)
        seed_reports.append(
            {
                "seed": int(seed),
                "config": result["config"],
                "metrics": metrics,
                "lock_status": locks,
                "stage1_metrics": result["stage1_metrics"],
                "stage1_config": result["stage1_config"],
                "stage1_history": result["stage1_history"],
                "advisor_history": result["advisor_history"],
                "oracle_advisor_history": result["oracle_advisor_history"],
                "prompt_history": result["prompt_history"],
                "sample_eval_rows": [row.to_json() for row in result["eval_examples"][:3]],
            }
        )
    metric_rows = [row["metrics"] for row in seed_reports]
    aggregate = _summarize(metric_rows)
    aggregate["lock_pass_rate"] = mean(
        [sum(1.0 for ok in row["lock_status"].values() if ok) / max(1, len(row["lock_status"])) for row in seed_reports]
    ) if seed_reports else 0.0
    report_path = run_dir / registry["report_names"]["suite"]
    validate_series_outputs("M", [registry["output_roots"]["suite"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M24.substrate_first_compression", "scripts/m24/run_m24_substrate_compression_suite.py"),
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile=registry["dataset_defaults"]["profile"],
            difficulty_tier="m23_decoy_balanced_symbolic_trace_compression",
        ),
        "track": "M24",
        "family_version": M24_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["suite"],
            "output_root": registry["output_roots"]["suite"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": {
            "seed_list": seeds,
            "train_size": int(args.train_size),
            "eval_size": int(args.eval_size),
            "generator_epochs": int(args.generator_epochs),
            "advisor_epochs": int(args.advisor_epochs),
            "prompt_epochs": int(args.prompt_epochs),
            "trace_weight": float(args.trace_weight),
            "answer_weight": float(args.answer_weight),
            "mdl_weight": float(args.mdl_weight),
            "trace_exact_surrogate_weight": float(args.trace_exact_surrogate_weight),
        },
        "architecture_locks": M24_LOCKS,
        "aggregate_metrics": aggregate,
        "seed_reports": seed_reports,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["prompt_only_accuracy"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M24 substrate compression report written to {report_path}")
    print(
        "M24 metrics: "
        f"strict={aggregate.get('mean_strict_accuracy', 0.0):.4f} "
        f"trace_exact={aggregate.get('mean_bridi_trace_exact_accuracy', 0.0):.4f} "
        f"substrate_score={aggregate.get('mean_substrate_claim_score', 0.0):.4f}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M24_REGISTRY["M24"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Run the M24 substrate-first packed bridi trace compression suite.")
    parser.add_argument("--seed-list", type=str, default="24")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--generator-epochs", type=int, default=8)
    parser.add_argument("--advisor-epochs", type=int, default=8)
    parser.add_argument("--prompt-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--generator-learning-rate", type=float, default=2e-3)
    parser.add_argument("--advisor-learning-rate", type=float, default=2e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--advisor-hidden-dim", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-places", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--trace-weight", type=float, default=2.5)
    parser.add_argument("--answer-weight", type=float, default=0.2)
    parser.add_argument("--mdl-weight", type=float, default=DEFAULT_M24_MDL_WEIGHT)
    parser.add_argument("--trace-exact-surrogate-weight", type=float, default=0.5)
    parser.add_argument("--clean-train-fraction", type=float, default=0.35)
    parser.add_argument("--clean-eval-fraction", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["suite"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
