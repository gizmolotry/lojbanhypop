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

from lojban_evolution.m23.family import M23_FAMILY_VERSION, M23_REGISTRY, M23_RELEVANCE_GRID  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402
from train_m23_relevance_router import parse_args as parse_train_args  # noqa: E402
from train_m23_relevance_router import run_train  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m23_relevance_suite_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m23_relevance_suite_{_timestamp()}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M23_REGISTRY["M23"]
    output_root = Path(args.output_root or registry["output_roots"]["suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _variant_args(base: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "use_relevance_router": bool(variant.get("use_relevance_router", base.use_relevance_router)),
        "relevance_rank_weight": float(variant.get("relevance_rank_weight", base.relevance_rank_weight)),
        "relevance_margin": float(variant.get("relevance_margin", base.relevance_margin)),
        "trace_weight": float(variant.get("trace_weight", base.trace_weight)),
        "trace_exact_surrogate_weight": float(variant.get("trace_exact_surrogate_weight", base.trace_exact_surrogate_weight)),
        "clean_train_fraction": float(variant.get("clean_train_fraction", base.clean_train_fraction)),
        "judri_bridge_gate": bool(variant.get("judri_bridge_gate", base.judri_bridge_gate)),
    }


def _collect(metric_rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row.get(key, 0.0) or 0.0) for row in metric_rows]


def _summarize(seed_reports: list[dict[str, Any]], stable_threshold: float) -> dict[str, float]:
    rows = [report["metrics"] for report in seed_reports]
    strict = _collect(rows, "strict_accuracy")
    decoy = _collect(rows, "decoy_relation_ood_accuracy")
    worst = _collect(rows, "worst_surface_accuracy")
    return {
        "mean_strict_accuracy": mean(strict) if strict else 0.0,
        "std_strict_accuracy": pstdev(strict) if len(strict) > 1 else 0.0,
        "mean_decoy_relation_ood_accuracy": mean(decoy) if decoy else 0.0,
        "mean_worst_surface_accuracy": mean(worst) if worst else 0.0,
        "mean_bridi_trace_exact_accuracy": mean(_collect(rows, "bridi_trace_exact_accuracy")) if rows else 0.0,
        "mean_relevance_top1_accuracy": mean(_collect(rows, "relevance_top1_accuracy")) if rows else 0.0,
        "mean_relevance_margin": mean(_collect(rows, "relevance_margin")) if rows else 0.0,
        "mean_loss_trace_exact_surrogate": mean(_collect(rows, "loss_trace_exact_surrogate")) if rows else 0.0,
        "mean_trace_exact_surrogate_weight": mean(_collect(rows, "trace_exact_surrogate_weight")) if rows else 0.0,
        "mean_oracle_relevance_accuracy": mean(_collect(rows, "oracle_relevance_accuracy")) if rows else 0.0,
        "mean_random_relevance_accuracy": mean(_collect(rows, "random_relevance_accuracy")) if rows else 0.0,
        "mean_no_relevance_accuracy": mean(_collect(rows, "no_relevance_accuracy")) if rows else 0.0,
        "mean_decoy_only_accuracy": mean(_collect(rows, "decoy_only_accuracy")) if rows else 0.0,
        "mean_oracle_relevance_delta": mean(_collect(rows, "oracle_relevance_delta")) if rows else 0.0,
        "mean_random_relevance_delta": mean(_collect(rows, "random_relevance_delta")) if rows else 0.0,
        "mean_decoy_only_delta": mean(_collect(rows, "decoy_only_delta")) if rows else 0.0,
        "avg_tokens": mean(_collect(rows, "avg_tokens")) if rows else 0.0,
        "accuracy_per_token": mean(_collect(rows, "accuracy_per_token")) if rows else 0.0,
        "trace_tokens": mean(_collect(rows, "trace_tokens")) if rows else 0.0,
        "accuracy_per_trace_token": mean(_collect(rows, "accuracy_per_trace_token")) if rows else 0.0,
        "stable_seed_rate": sum(1.0 for value in strict if value >= float(stable_threshold)) / max(1, len(strict)),
    }


def _interpret(cells: dict[str, Any]) -> dict[str, Any]:
    a = cells.get("A", {}).get("aggregate_metrics", {})
    b = cells.get("B", {}).get("aggregate_metrics", {})
    c = cells.get("C", {}).get("aggregate_metrics", {})
    if c and not (a or b):
        return {
            "conclusion": "trace_punishment_diagnostic_only",
            "m23_router_decoy_lift_vs_scale": 0.0,
            "m23_router_worst_surface_lift_vs_scale": 0.0,
            "m23_oracle_relevance_lift": 0.0,
            "m23_trace_punish_trace_exact_lift_vs_scale": 0.0,
            "m23_trace_punish_decoy_delta_vs_scale": 0.0,
            "m23_trace_punish_strict_delta_vs_scale": 0.0,
        }
    scale_decoy = float(a.get("mean_decoy_relation_ood_accuracy", 0.0) or 0.0)
    router_decoy = float(b.get("mean_decoy_relation_ood_accuracy", 0.0) or 0.0)
    scale_worst = float(a.get("mean_worst_surface_accuracy", 0.0) or 0.0)
    router_worst = float(b.get("mean_worst_surface_accuracy", 0.0) or 0.0)
    scale_trace = float(a.get("mean_bridi_trace_exact_accuracy", 0.0) or 0.0)
    scale_strict = float(a.get("mean_strict_accuracy", 0.0) or 0.0)
    oracle_delta = float(b.get("mean_oracle_relevance_delta", 0.0) or a.get("mean_oracle_relevance_delta", 0.0) or 0.0)
    learned_lift = router_decoy - scale_decoy
    if scale_decoy >= 0.70 and router_decoy <= scale_decoy + 0.01:
        conclusion = "more_data_training_was_enough"
    elif learned_lift >= 0.02 and router_worst >= scale_worst - 0.02:
        conclusion = "explicit_relevance_selection_is_causally_useful"
    elif oracle_delta >= 0.02:
        conclusion = "architecture_can_use_relevance_but_router_objective_is_insufficient"
    else:
        conclusion = "failure_is_downstream_or_trace_decoding_not_frame_selection"
    return {
        "conclusion": conclusion,
        "m23_router_decoy_lift_vs_scale": learned_lift,
        "m23_router_worst_surface_lift_vs_scale": router_worst - scale_worst,
        "m23_oracle_relevance_lift": oracle_delta,
        "m23_trace_punish_trace_exact_lift_vs_scale": float(c.get("mean_bridi_trace_exact_accuracy", 0.0) or 0.0) - scale_trace if c else 0.0,
        "m23_trace_punish_decoy_delta_vs_scale": float(c.get("mean_decoy_relation_ood_accuracy", 0.0) or 0.0) - scale_decoy if c else 0.0,
        "m23_trace_punish_strict_delta_vs_scale": float(c.get("mean_strict_accuracy", 0.0) or 0.0) - scale_strict if c else 0.0,
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M23_REGISTRY["M23"]
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seed_list)
    selected_cells = {item.strip().upper() for item in str(args.cell_list).split(",") if item.strip()}
    grid = [cell for cell in M23_RELEVANCE_GRID if not selected_cells or cell["cell_key"] in selected_cells]
    cells: dict[str, Any] = {}
    all_reports: list[dict[str, Any]] = []
    for cell in grid:
        cell_key = str(cell["cell_key"])
        variant = _variant_args(args, dict(cell.get("variant", {})))
        seed_reports: list[dict[str, Any]] = []
        for seed in seeds:
            train_args = parse_train_args(
                [
                    "--train-size", str(int(args.train_size)),
                    "--eval-size", str(int(args.eval_size)),
                    "--epochs", str(int(args.epochs)),
                    "--batch-size", str(int(args.batch_size)),
                    "--learning-rate", str(float(args.learning_rate)),
                    "--seed", str(int(seed)),
                    "--embedding-dim", str(int(args.embedding_dim)),
                    "--hidden-dim", str(int(args.hidden_dim)),
                    "--max-frames", str(int(args.max_frames)),
                    "--max-places", str(int(args.max_places)),
                    "--max-entities", str(int(args.max_entities)),
                    "--trace-weight", str(variant["trace_weight"]),
                    "--answer-weight", str(float(args.answer_weight)),
                    "--counterfactual-weight", str(float(args.counterfactual_weight)),
                    "--brivi-lock-weight", str(float(args.brivi_lock_weight)),
                    "--frame-necessity-weight", str(float(args.frame_necessity_weight)),
                    "--mdl-weight", str(float(args.mdl_weight)),
                    "--necessity-margin", str(float(args.necessity_margin)),
                    "--pointer-necessity-weight", str(float(args.pointer_necessity_weight)),
                    "--pointer-necessity-margin", str(float(args.pointer_necessity_margin)),
                    "--relevance-rank-weight", str(variant["relevance_rank_weight"]),
                    "--relevance-margin", str(variant["relevance_margin"]),
                    "--trace-exact-surrogate-weight", str(variant["trace_exact_surrogate_weight"]),
                    "--use-relevance-router" if variant["use_relevance_router"] else "--no-use-relevance-router",
                    "--clean-train-fraction", str(variant["clean_train_fraction"]),
                    "--clean-eval-fraction", str(float(args.clean_eval_fraction)),
                    "--geometry-mode", str(args.geometry_mode),
                    "--poincare-curvature", str(float(args.poincare_curvature)),
                    "--poincare-max-norm", str(float(args.poincare_max_norm)),
                    "--riemannian-gradient-scale" if bool(args.riemannian_gradient_scale) else "--no-riemannian-gradient-scale",
                    "--judri-bridge-gate" if variant["judri_bridge_gate"] else "--no-judri-bridge-gate",
                    "--judri-bridge-gate-temperature", str(float(args.judri_bridge_gate_temperature)),
                    "--device", str(args.device),
                    "--output-root", str(run_dir),
                    "--run-id", f"{cell_key}_seed_{seed}",
                ]
            )
            seed_reports.append(run_train(train_args))
        all_reports.extend(seed_reports)
        cells[cell_key] = {
            "cell_id": cell["cell_id"],
            "lock": cell["lock"],
            "label": cell["label"],
            "variant_spec": variant,
            "aggregate_metrics": _summarize(seed_reports, float(args.stable_threshold)),
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
    aggregate = _summarize(all_reports, float(args.stable_threshold))
    interpretation = _interpret(cells)
    report_path = run_dir / registry["report_names"]["suite"]
    validate_series_outputs("M", [registry["output_roots"]["suite"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M23.causal_relevance_router_suite", "scripts/m23/run_m23_relevance_suite.py"),
        "track": "M23",
        "family_version": M23_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["suite"],
            "dag": registry["dags"].get("suite"),
            "output_root": registry["output_roots"]["suite"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "config": {
            "seed_list": seeds,
            "cell_list": [cell["cell_key"] for cell in grid],
            "train_size": int(args.train_size),
            "eval_size": int(args.eval_size),
            "epochs": int(args.epochs),
            "stable_threshold": float(args.stable_threshold),
        },
        "aggregate_metrics": aggregate | interpretation,
        "cells": cells,
        "hypothesis_interpretation": interpretation,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M23 suite report written to {report_path}")
    print(
        "M23 suite metrics: "
        f"strict={aggregate['mean_strict_accuracy']:.4f} "
        f"decoy={aggregate['mean_decoy_relation_ood_accuracy']:.4f} "
        f"worst={aggregate['mean_worst_surface_accuracy']:.4f} "
        f"conclusion={interpretation['conclusion']}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M23_REGISTRY["M23"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Run the M23 causal relevance router suite.")
    parser.add_argument("--seed-list", type=str, default="23,29,31,37,41,43")
    parser.add_argument("--cell-list", type=str, default="A,B,C")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-places", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--trace-weight", type=float, default=1.25)
    parser.add_argument("--answer-weight", type=float, default=1.25)
    parser.add_argument("--counterfactual-weight", type=float, default=1.25)
    parser.add_argument("--brivi-lock-weight", type=float, default=1.5)
    parser.add_argument("--frame-necessity-weight", type=float, default=1.0)
    parser.add_argument("--mdl-weight", type=float, default=0.01)
    parser.add_argument("--necessity-margin", type=float, default=0.04)
    parser.add_argument("--pointer-necessity-weight", type=float, default=0.0)
    parser.add_argument("--pointer-necessity-margin", type=float, default=0.05)
    parser.add_argument("--relevance-rank-weight", type=float, default=0.0)
    parser.add_argument("--relevance-margin", type=float, default=0.15)
    parser.add_argument("--trace-exact-surrogate-weight", type=float, default=0.0)
    parser.add_argument("--use-relevance-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clean-train-fraction", type=float, default=0.35)
    parser.add_argument("--clean-eval-fraction", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, choices=("euclidean", "poincare"), default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--poincare-max-norm", type=float, default=0.99)
    parser.add_argument("--riemannian-gradient-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judri-bridge-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judri-bridge-gate-temperature", type=float, default=1.0)
    parser.add_argument("--stable-threshold", type=float, default=0.70)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["suite"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
