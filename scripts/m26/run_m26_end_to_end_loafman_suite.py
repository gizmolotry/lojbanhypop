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

from lojban_evolution.m25.emergent_bridi import DEFAULT_M25_MDL_WEIGHT  # noqa: E402
from lojban_evolution.m26.end_to_end import (  # noqa: E402
    DEFAULT_M26_ANSWER_WEIGHT,
    DEFAULT_M26_TRACE_WEIGHT,
    train_m26_end_to_end_loafman,
)
from lojban_evolution.m26.family import M26_FAMILY_VERSION, M26_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m26_end_to_end_loafman_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m26_end_to_end_loafman_{_timestamp()}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M26_REGISTRY["M26"]
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
        "phrase_accuracy",
        "end_to_end_answer_accuracy",
        "shuffled_trace_accuracy",
        "random_trace_accuracy",
        "zero_trace_accuracy",
        "prompt_only_accuracy",
        "matched_prompt_accuracy",
        "m26_strict_delta_vs_prompt_only",
        "m26_strict_delta_vs_matched_prompt",
        "predicted_vs_shuffled_delta",
        "predicted_vs_random_delta",
        "predicted_vs_zero_delta",
        "loose_stream_exact_accuracy",
        "stream_type_accuracy",
        "stream_value_accuracy",
        "stream_aux_accuracy",
        "mean_predicted_emitted_symbols_after_bottleneck",
        "mean_prompt_tokens",
        "mean_matched_prompt_tokens",
        "loose_symbol_to_prompt_ratio",
        "loose_symbol_to_matched_prompt_ratio",
        "matched_prompt_token_reduction_ratio",
        "accuracy_per_loose_symbol",
        "accuracy_per_prompt_token",
        "matched_prompt_accuracy_per_token",
        "m26_accuracy_per_symbol_delta_vs_matched_prompt",
        "matched_prompt_token_budget",
        "m26_gate_beats_matched_prompt",
        "single_optimizer_end_to_end_training",
        "hard_argmax_training_cut_detected",
        "torch_no_grad_training_cut_detected",
        "advisor_primary_trace_is_differentiable",
        "answer_loss_generator_grad_norm",
        "answer_loss_symbol_head_grad_norm",
        "answer_loss_advisor_grad_norm",
        "answer_loss_reaches_generator",
        "answer_loss_reaches_symbol_heads",
        "trainable_parameter_count",
        "generator_trainable_parameter_count",
        "advisor_trainable_parameter_count",
        "m26_gate_answer_loss_reaches_generator",
        "m26_gate_answer_loss_reaches_symbol_heads",
        "m26_gate_single_optimizer",
        "m26_gate_no_hard_training_cut",
        "m26_gate_stream_beats_zero",
        "m26_gate_beats_matched_prompt",
        "m26_spinal_cord_gate_pass_rate",
        "m26_spinal_cord_candidate",
        "m26_prompt_comparable_candidate",
        "m26_promotion_candidate",
    )
    out: dict[str, float] = {}
    for key in keys:
        values = _collect(rows, key)
        out[f"mean_{key}"] = mean(values) if values else 0.0
        if key == "strict_accuracy":
            out["std_strict_accuracy"] = pstdev(values) if len(values) > 1 else 0.0
    return out


def _summarize_surfaces(seed_reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    buckets: dict[str, list[float]] = {}
    counts: dict[str, float] = {}
    for report in seed_reports:
        surfaces = report.get("surface_metrics", {})
        if not isinstance(surfaces, dict):
            continue
        for surface, metrics in surfaces.items():
            if not isinstance(metrics, dict):
                continue
            if "strict_accuracy" not in metrics:
                continue
            buckets.setdefault(str(surface), []).append(float(metrics["strict_accuracy"]))
            counts[str(surface)] = counts.get(str(surface), 0.0) + float(metrics.get("count", 0.0) or 0.0)
    return {
        surface: {
            "mean_strict_accuracy": mean(values),
            "min_strict_accuracy": min(values),
            "max_strict_accuracy": max(values),
            "total_count": counts.get(surface, 0.0),
        }
        for surface, values in sorted(buckets.items())
        if values
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M26_REGISTRY["M26"]
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seed_list)
    seed_reports: list[dict[str, Any]] = []
    for seed in seeds:
        result = train_m26_end_to_end_loafman(
            train_size=int(args.train_size),
            eval_size=int(args.eval_size),
            epochs=int(args.epochs),
            prompt_epochs=int(args.prompt_epochs),
            batch_size=int(args.batch_size),
            learning_rate=float(args.learning_rate),
            seed=int(seed),
            embedding_dim=int(args.embedding_dim),
            hidden_dim=int(args.hidden_dim),
            advisor_hidden_dim=int(args.advisor_hidden_dim),
            max_frames=int(args.max_frames),
            max_symbols=int(args.max_symbols),
            symbol_budget=int(args.symbol_budget) if int(args.symbol_budget) > 0 else None,
            matched_prompt_budget=int(args.matched_prompt_budget) if int(args.matched_prompt_budget) > 0 else None,
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
                "history": result["history"],
                "prompt_history": result["prompt_history"],
                "matched_prompt_history": result["matched_prompt_history"],
                "sample_eval_rows": [row.to_json() for row in result["eval_examples"][:3]],
            }
        )
    aggregate = _summarize([row["metrics"] for row in seed_reports])
    aggregate_surface_metrics = _summarize_surfaces(seed_reports)
    report_path = run_dir / registry["report_names"]["suite"]
    validate_series_outputs("M", [registry["output_roots"]["suite"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M26.end_to_end_lojban_symbiote", "scripts/m26/run_m26_end_to_end_loafman_suite.py"),
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile=registry["dataset_defaults"]["profile"],
            difficulty_tier="m25_loose_bridi_end_to_end_spinal_cord",
        ),
        "track": "M26",
        "family_version": M26_FAMILY_VERSION,
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
            "epochs": int(args.epochs),
            "prompt_epochs": int(args.prompt_epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "embedding_dim": int(args.embedding_dim),
            "hidden_dim": int(args.hidden_dim),
            "advisor_hidden_dim": int(args.advisor_hidden_dim),
            "max_frames": int(args.max_frames),
            "max_symbols": int(args.max_symbols),
            "symbol_budget": int(args.symbol_budget),
            "matched_prompt_budget": int(args.matched_prompt_budget),
            "trace_weight": float(args.trace_weight),
            "answer_weight": float(args.answer_weight),
            "mdl_weight": float(args.mdl_weight),
            "device": str(args.device),
        },
        "architecture_locks": [
            "single_optimizer_generator_and_advisor",
            "differentiable_soft_bridi_trace_handoff",
            "answer_loss_gradient_probe_into_generator",
            "no_training_path_argmax_or_no_grad_cut",
        ],
        "seed_reports": seed_reports,
        "aggregate_metrics": aggregate,
        "aggregate_surface_metrics": aggregate_surface_metrics,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M26 end-to-end Lojban symbiote report written to {report_path}")
    print(json.dumps(aggregate, indent=2))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run M26 end-to-end Lojban symbiote spinal-cord suite.")
    parser.add_argument("--seed-list", default="23,29")
    parser.add_argument("--train-size", type=int, default=6000)
    parser.add_argument("--eval-size", type=int, default=1500)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--prompt-epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--advisor-hidden-dim", type=int, default=64)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-symbols", type=int, default=32)
    parser.add_argument("--symbol-budget", type=int, default=0)
    parser.add_argument("--matched-prompt-budget", type=int, default=0)
    parser.add_argument("--trace-weight", type=float, default=DEFAULT_M26_TRACE_WEIGHT)
    parser.add_argument("--answer-weight", type=float, default=DEFAULT_M26_ANSWER_WEIGHT)
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
