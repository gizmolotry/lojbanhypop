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

from lojban_evolution.m25.emergent_bridi import DEFAULT_M25_MDL_WEIGHT, train_m25_emergent_bridi  # noqa: E402
from lojban_evolution.m25.family import M25_FAMILY_VERSION, M25_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m25_emergent_bridi_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m25_emergent_bridi_{_timestamp()}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M25_REGISTRY["M25"]
    output_root = Path(args.output_root or registry["output_roots"]["suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _collect(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in rows]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "strict_accuracy",
        "predicted_stream_accuracy",
        "oracle_stream_accuracy",
        "shuffled_stream_accuracy",
        "random_stream_accuracy",
        "zero_stream_accuracy",
        "prompt_only_accuracy",
        "m25_strict_delta_vs_prompt_only",
        "predicted_vs_shuffled_delta",
        "predicted_vs_random_delta",
        "oracle_stream_delta",
        "oracle_trained_predicted_stream_accuracy",
        "oracle_trained_oracle_stream_accuracy",
        "oracle_trained_random_stream_accuracy",
        "oracle_trained_stream_delta",
        "stream_advisor_delta",
        "loose_stream_exact_accuracy",
        "stream_type_accuracy",
        "stream_value_accuracy",
        "stream_aux_accuracy",
        "mean_predicted_emitted_symbols_after_bottleneck",
        "mean_oracle_emitted_symbols_after_bottleneck",
        "mean_prompt_tokens",
        "loose_symbol_to_prompt_ratio",
        "prompt_to_loose_symbol_ratio",
        "token_reduction_ratio",
        "accuracy_per_loose_symbol",
        "accuracy_per_prompt_token",
        "loose_symbol_budget",
        "hard_symbol_budget_active",
        "advisor_primary_trace_is_symbolic",
        "continuous_trace_smuggling_detected",
        "generator_parameter_max_delta_after_advisor",
        "generator_parameters_unchanged_after_advisor",
        "m25_promotion_gate_pass_rate",
        "m25_promotion_candidate",
        "m25_gate_strict_accuracy_retained",
        "m25_gate_stream_beats_shuffled",
        "m25_gate_stream_beats_random",
        "m25_gate_token_reduction_positive",
        "m25_gate_nonzero_stream_reconstruction",
        "m25_gate_symbolic_trace_only",
    )
    out: dict[str, float] = {}
    for key in keys:
        values = _collect(rows, key)
        out[f"mean_{key}"] = mean(values) if values else 0.0
        if key == "strict_accuracy":
            out["std_strict_accuracy"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M25_REGISTRY["M25"]
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seed_list)
    seed_reports: list[dict[str, Any]] = []
    for seed in seeds:
        result = train_m25_emergent_bridi(
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
            max_symbols=int(args.max_symbols),
            symbol_budget=int(args.symbol_budget) if int(args.symbol_budget) > 0 else None,
            trace_weight=float(args.trace_weight),
            answer_weight=float(args.answer_weight),
            mdl_weight=float(args.mdl_weight),
            clean_train_fraction=float(args.clean_train_fraction),
            clean_eval_fraction=float(args.clean_eval_fraction),
            device=str(args.device),
        )
        seed_reports.append(
            {
                "seed": int(seed),
                "config": result["config"],
                "metrics": result["metrics"],
                "surface_metrics": result["surface_metrics"],
                "generator_history": result["generator_history"],
                "advisor_history": result["advisor_history"],
                "oracle_advisor_history": result["oracle_advisor_history"],
                "prompt_history": result["prompt_history"],
                "sample_eval_rows": [row.to_json() for row in result["eval_examples"][:3]],
            }
        )
    aggregate = _summarize([row["metrics"] for row in seed_reports])
    report_path = run_dir / registry["report_names"]["suite"]
    validate_series_outputs("M", [registry["output_roots"]["suite"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M25.emergent_bridi_grammar", "scripts/m25/run_m25_emergent_bridi_suite.py"),
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile=registry["dataset_defaults"]["profile"],
            difficulty_tier="m23_decoy_balanced_loose_bridi_stream",
        ),
        "track": "M25",
        "family_version": M25_FAMILY_VERSION,
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
            "max_symbols": int(args.max_symbols),
            "symbol_budget": int(args.symbol_budget),
            "trace_weight": float(args.trace_weight),
            "answer_weight": float(args.answer_weight),
            "mdl_weight": float(args.mdl_weight),
            "device": str(args.device),
        },
        "architecture_locks": [
            "loose_integer_bridi_stream_only",
            "frozen_generator_before_advisor_training",
            "predicted_oracle_shuffled_random_zero_controls",
        ],
        "seed_reports": seed_reports,
        "aggregate_metrics": aggregate,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M25 emergent bridi report written to {report_path}")
    print(json.dumps(aggregate, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M25 emergent bridi grammar stream suite.")
    parser.add_argument("--seed-list", default="23,29")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--eval-size", type=int, default=1500)
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
    parser.add_argument("--max-symbols", type=int, default=32)
    parser.add_argument("--symbol-budget", type=int, default=0)
    parser.add_argument("--trace-weight", type=float, default=2.0)
    parser.add_argument("--answer-weight", type=float, default=0.25)
    parser.add_argument("--mdl-weight", type=float, default=DEFAULT_M25_MDL_WEIGHT)
    parser.add_argument("--clean-train-fraction", type=float, default=0.35)
    parser.add_argument("--clean-eval-fraction", type=float, default=0.35)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--run-id", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run_suite(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
