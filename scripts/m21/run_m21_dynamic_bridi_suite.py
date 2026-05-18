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

from lojban_evolution.m21.family import M21_DYNAMIC_BRIDI_GRID, M21_FAMILY_VERSION, M21_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402
from train_m21_dynamic_bridi import parse_args as parse_train_args  # noqa: E402
from train_m21_dynamic_bridi import run_train  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m21_dynamic_bridi_suite_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m21_dynamic_bridi_suite_{_timestamp()}"


def _parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M21_REGISTRY["M21"]
    output_root = Path(args.output_root or registry["output_roots"]["suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _variant_args(base: argparse.Namespace, variant: dict[str, Any]) -> dict[str, Any]:
    return {
        "trace_weight": float(variant.get("trace_weight", base.trace_weight)),
        "answer_weight": float(variant.get("answer_weight", base.answer_weight)),
        "counterfactual_weight": float(variant.get("counterfactual_weight", base.counterfactual_weight)),
        "brivi_lock_weight": float(variant.get("brivi_lock_weight", base.brivi_lock_weight)),
        "frame_necessity_weight": float(variant.get("frame_necessity_weight", base.frame_necessity_weight)),
        "mdl_weight": float(variant.get("mdl_weight", base.mdl_weight)),
        "necessity_margin": float(variant.get("necessity_margin", base.necessity_margin)),
        "pointer_necessity_weight": float(variant.get("pointer_necessity_weight", base.pointer_necessity_weight)),
        "pointer_necessity_margin": float(variant.get("pointer_necessity_margin", base.pointer_necessity_margin)),
        "hyperbolic_topology_weight": float(variant.get("hyperbolic_topology_weight", base.hyperbolic_topology_weight)),
        "judri_bridge_gate": bool(variant.get("judri_bridge_gate", base.judri_bridge_gate)),
        "judri_bridge_gate_temperature": float(variant.get("judri_bridge_gate_temperature", base.judri_bridge_gate_temperature)),
        "adversarial_train_fraction": float(variant.get("adversarial_train_fraction", base.adversarial_train_fraction)),
        "adversarial_train_surfaces": str(variant.get("adversarial_train_surfaces", base.adversarial_train_surfaces)),
    }


def _summarize_seed_reports(seed_reports: list[dict[str, Any]], stable_threshold: float) -> dict[str, float]:
    metric_rows = [report["metrics"] for report in seed_reports]

    def collect(key: str) -> list[float]:
        return [float(row.get(key, 0.0) or 0.0) for row in metric_rows]

    strict = collect("strict_accuracy")
    return {
        "mean_strict_accuracy": mean(strict) if strict else 0.0,
        "std_strict_accuracy": pstdev(strict) if len(strict) > 1 else 0.0,
        "mean_bridi_trace_exact_accuracy": mean(collect("bridi_trace_exact_accuracy")) if metric_rows else 0.0,
        "mean_gismu_accuracy": mean(collect("gismu_accuracy")) if metric_rows else 0.0,
        "mean_cmavo_accuracy": mean(collect("cmavo_accuracy")) if metric_rows else 0.0,
        "mean_judri_binding_accuracy": mean(collect("judri_binding_accuracy")) if metric_rows else 0.0,
        "mean_frame_count_mae": mean(collect("frame_count_mae")) if metric_rows else 0.0,
        "mean_brivi_lock_violation_rate": mean(collect("brivi_lock_violation_rate")) if metric_rows else 0.0,
        "mean_brivi_gate_accuracy": mean(collect("brivi_gate_accuracy")) if metric_rows else 0.0,
        "mean_lock_pass_rate": mean(collect("lock_pass_rate")) if metric_rows else 0.0,
        "mean_full_accuracy": mean(collect("full_accuracy")) if metric_rows else 0.0,
        "mean_no_cmavo_accuracy": mean(collect("no_cmavo_accuracy")) if metric_rows else 0.0,
        "mean_no_judri_accuracy": mean(collect("no_judri_accuracy")) if metric_rows else 0.0,
        "mean_gismu_only_accuracy": mean(collect("gismu_only_accuracy")) if metric_rows else 0.0,
        "mean_frame_drop_delta": mean(collect("frame_drop_delta")) if metric_rows else 0.0,
        "mean_cmavo_causal_delta": mean(collect("cmavo_causal_delta")) if metric_rows else 0.0,
        "mean_judri_causal_delta": mean(collect("judri_causal_delta")) if metric_rows else 0.0,
        "mean_loss_pointer_necessity": mean(collect("loss_pointer_necessity")) if metric_rows else 0.0,
        "mean_pointer_necessity_gap": mean(collect("pointer_necessity_gap")) if metric_rows else 0.0,
        "mean_active_frames": mean(collect("mean_active_frames")) if metric_rows else 0.0,
        "mean_hyperbolic_projection_clip_rate": mean(collect("hyperbolic_projection_clip_rate")) if metric_rows else 0.0,
        "mean_hyperbolic_max_norm": mean(collect("hyperbolic_max_norm")) if metric_rows else 0.0,
        "mean_hyperbolic_distance_mean": mean(collect("hyperbolic_distance_mean")) if metric_rows else 0.0,
        "mean_hyperbolic_tangent_handoff_norm_mean": mean(collect("hyperbolic_tangent_handoff_norm_mean")) if metric_rows else 0.0,
        "mean_hyperbolic_tangent_handoff_finite_rate": mean(collect("hyperbolic_tangent_handoff_finite_rate")) if metric_rows else 0.0,
        "mean_judri_bridge_gate_mean": mean(collect("judri_bridge_gate_mean")) if metric_rows else 0.0,
        "mean_judri_bridge_gate_active_mean": mean(collect("judri_bridge_gate_active_mean")) if metric_rows else 0.0,
        "mean_judri_bridge_gate_silenced_predicate_energy_mean": mean(collect("judri_bridge_gate_silenced_predicate_energy_mean")) if metric_rows else 0.0,
        "mean_judri_bridge_gate_enabled": mean(collect("judri_bridge_gate_enabled")) if metric_rows else 0.0,
        "mean_adversarial_train_fraction": mean(collect("adversarial_train_fraction")) if metric_rows else 0.0,
        "adversarial_training_exposure_rate": sum(1.0 for value in collect("adversarial_train_fraction") if value > 0.0) / max(1, len(metric_rows)),
        "mean_active_code_fraction_reachable": mean(collect("active_code_fraction_reachable")) if metric_rows else 0.0,
        "avg_tokens": mean(collect("avg_tokens")) if metric_rows else 0.0,
        "accuracy_per_token": mean(collect("accuracy_per_token")) if metric_rows else 0.0,
        "trace_tokens": mean(collect("trace_tokens")) if metric_rows else 0.0,
        "accuracy_per_trace_token": mean(collect("accuracy_per_trace_token")) if metric_rows else 0.0,
        "stable_seed_rate": sum(1.0 for value in strict if value >= float(stable_threshold)) / max(1, len(strict)),
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M21_REGISTRY["M21"]
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    seeds = _parse_int_list(args.seed_list)
    selected_cells = {item.strip().upper() for item in str(args.cell_list).split(",") if item.strip()}
    grid = [cell for cell in M21_DYNAMIC_BRIDI_GRID if not selected_cells or cell["cell_key"] in selected_cells]
    cells: dict[str, Any] = {}
    all_seed_reports: list[dict[str, Any]] = []
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
                    "--embedding-dim",
                    str(int(args.embedding_dim)),
                    "--hidden-dim",
                    str(int(args.hidden_dim)),
                    "--max-frames",
                    str(int(args.max_frames)),
                    "--max-cmavo-per-frame",
                    str(int(args.max_cmavo_per_frame)),
                    "--max-places",
                    str(int(args.max_places)),
                    "--max-entities",
                    str(int(args.max_entities)),
                    "--trace-weight",
                    str(variant["trace_weight"]),
                    "--answer-weight",
                    str(variant["answer_weight"]),
                    "--counterfactual-weight",
                    str(variant["counterfactual_weight"]),
                    "--brivi-lock-weight",
                    str(variant["brivi_lock_weight"]),
                    "--frame-necessity-weight",
                    str(variant["frame_necessity_weight"]),
                    "--mdl-weight",
                    str(variant["mdl_weight"]),
                    "--necessity-margin",
                    str(variant["necessity_margin"]),
                    "--pointer-necessity-weight",
                    str(variant["pointer_necessity_weight"]),
                    "--pointer-necessity-margin",
                    str(variant["pointer_necessity_margin"]),
                    "--hyperbolic-topology-weight",
                    str(variant["hyperbolic_topology_weight"]),
                    "--judri-bridge-gate" if bool(variant["judri_bridge_gate"]) else "--no-judri-bridge-gate",
                    "--judri-bridge-gate-temperature",
                    str(variant["judri_bridge_gate_temperature"]),
                    "--adversarial-train-fraction",
                    str(variant["adversarial_train_fraction"]),
                    "--adversarial-train-surfaces",
                    str(variant["adversarial_train_surfaces"]),
                    "--geometry-mode",
                    str(args.geometry_mode),
                    "--poincare-curvature",
                    str(float(args.poincare_curvature)),
                    "--poincare-max-norm",
                    str(float(args.poincare_max_norm)),
                    "--riemannian-gradient-scale" if bool(args.riemannian_gradient_scale) else "--no-riemannian-gradient-scale",
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
        "series": series_metadata("M", "M21.1.dynamic_bridi_suite", "scripts/m21/run_m21_dynamic_bridi_suite.py"),
        "track": "M21.1",
        "family_version": M21_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["suite"],
            "dag": registry["dags"]["suite"],
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
            "geometry_mode": str(args.geometry_mode),
            "poincare_curvature": float(args.poincare_curvature),
            "poincare_max_norm": float(args.poincare_max_norm),
            "riemannian_gradient_scale": bool(args.riemannian_gradient_scale),
            "judri_bridge_gate": bool(args.judri_bridge_gate),
            "judri_bridge_gate_temperature": float(args.judri_bridge_gate_temperature),
            "adversarial_train_fraction": float(args.adversarial_train_fraction),
            "adversarial_train_surfaces": str(args.adversarial_train_surfaces),
        },
        "aggregate_metrics": aggregate,
        "cells": cells,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M21 suite report written to {report_path}")
    print(
        "M21 suite metrics: "
        f"mean_strict={aggregate['mean_strict_accuracy']:.4f} "
        f"trace_exact={aggregate['mean_bridi_trace_exact_accuracy']:.4f} "
        f"locks={aggregate['mean_lock_pass_rate']:.4f} "
        f"stable={aggregate['stable_seed_rate']:.4f}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M21_REGISTRY["M21"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Run the M21 dynamic bridi Q-former suite.")
    parser.add_argument("--seed-list", type=str, default="23,29")
    parser.add_argument("--cell-list", type=str, default="A,B,C,D,E,F,G,H,I,J,K,L")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-cmavo-per-frame", type=int, default=3)
    parser.add_argument("--max-places", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--trace-weight", type=float, default=1.0)
    parser.add_argument("--answer-weight", type=float, default=1.0)
    parser.add_argument("--counterfactual-weight", type=float, default=1.0)
    parser.add_argument("--brivi-lock-weight", type=float, default=1.0)
    parser.add_argument("--frame-necessity-weight", type=float, default=0.5)
    parser.add_argument("--mdl-weight", type=float, default=0.01)
    parser.add_argument("--necessity-margin", type=float, default=0.04)
    parser.add_argument("--pointer-necessity-weight", type=float, default=0.0)
    parser.add_argument("--pointer-necessity-margin", type=float, default=0.05)
    parser.add_argument("--hyperbolic-topology-weight", type=float, default=0.0)
    parser.add_argument("--geometry-mode", type=str, choices=("euclidean", "poincare"), default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--poincare-max-norm", type=float, default=0.99)
    parser.add_argument("--riemannian-gradient-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judri-bridge-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judri-bridge-gate-temperature", type=float, default=1.0)
    parser.add_argument("--adversarial-train-fraction", type=float, default=0.0)
    parser.add_argument("--adversarial-train-surfaces", type=str, default="heldout_paraphrase,clausal_permutation")
    parser.add_argument("--stable-threshold", type=float, default=0.70)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["suite"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
