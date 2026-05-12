from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.artifact_contract import run_if_needed
from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


def _track_key(track: str) -> str:
    candidate = str(track or "").strip()
    return candidate if candidate in M19_REGISTRY else "M19"


def _resolve_output_root(track: str, output_root: Path) -> Path:
    track_key = _track_key(track)
    registry = M19_REGISTRY[track_key]
    default_root = Path(M19_REGISTRY["M19"]["output_roots"]["stability_microgrid"])
    if track_key != "M19" and Path(output_root) == default_root:
        return Path(registry["output_roots"]["stability_microgrid"])
    return Path(output_root)


def _apply_track_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = M19_REGISTRY.get(_track_key(args.track), {}).get("defaults", {})
    if defaults:
        if not str(args.typed_physics_config).strip() and defaults.get("typed_physics_config"):
            args.typed_physics_config = str(defaults["typed_physics_config"])
        if not str(args.typed_slot_layout).strip() and defaults.get("typed_slot_layout"):
            args.typed_slot_layout = str(defaults["typed_slot_layout"])
        if str(args.arity_router_mode).strip() == "soft" and defaults.get("arity_router_mode"):
            args.arity_router_mode = str(defaults["arity_router_mode"])
        if str(args.geometry_mode).strip() == "euclidean" and defaults.get("geometry_mode"):
            args.geometry_mode = str(defaults["geometry_mode"])
        if not args.gumbel_hard and str(defaults.get("arity_router_mode", "")).strip() == "gumbel_hard":
            args.gumbel_hard = True
    return args


def _typed_replication_cli_args(args: argparse.Namespace) -> list[str]:
    cli_args = [
        "--typed-physics-config",
        str(args.typed_physics_config),
        "--typed-slot-layout",
        str(args.typed_slot_layout),
        "--arity-router-mode",
        str(args.arity_router_mode),
        "--gumbel-temp-end",
        str(float(args.gumbel_temp_end)),
        "--geometry-mode",
        str(args.geometry_mode),
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
    ]
    if args.gumbel_hard:
        cli_args.append("--gumbel-hard")
    return cli_args


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(
        description="Run a ledger-native stabilization micro-grid over weak M19 seeds."
    )
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", type=Path, default=Path(registry["dataset_defaults"]["train"]))
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--audit-data-path", type=Path, default=Path(registry["dataset_defaults"]["audit"]))
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--audit-eval-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate-list", type=str, default="5e-5,1e-4")
    parser.add_argument("--augmentation-prob-list", type=str, default="0.0,0.5")
    parser.add_argument("--format-augmentation-prob-list", type=str, default="0.0")
    parser.add_argument("--seed-list", type=str, default="23,29")
    parser.add_argument("--surface-consistency-weight", type=float, default=0.0)
    parser.add_argument("--surface-consistency-weight-list", type=str, default="")
    parser.add_argument("--surface-consistency-entity-rename", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--surface-consistency-format-flatten", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--surface-consistency-combined", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--surface-consistency-max-variants", type=int, default=2)
    parser.add_argument("--surface-consistency-task-weight", type=float, default=0.15)
    parser.add_argument("--pointer-necessity-weight", type=float, default=0.0)
    parser.add_argument("--pointer-necessity-weight-list", type=str, default="")
    parser.add_argument("--pointer-necessity-margin", type=float, default=0.05)
    parser.add_argument("--pointer-necessity-ablation-mode", type=str, default="no_judri")
    parser.add_argument("--typed-family-weight", type=float, default=0.05)
    parser.add_argument("--typed-arity-weight", type=float, default=0.05)
    parser.add_argument("--family-separation-weight", type=float, default=0.02)
    parser.add_argument("--slot-usage-balance-weight", type=float, default=0.01)
    parser.add_argument("--operator-balance-weight", type=float, default=0.0)
    parser.add_argument("--operator-top1-cap", type=float, default=0.30)
    parser.add_argument("--typed-physics-config", type=str, default="")
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--checkpoint-selection-policy", type=str, default="final_only")
    parser.add_argument("--selection-purged-eval-size", type=int, default=None)
    parser.add_argument("--selection-surface-size", type=int, default=200)
    parser.add_argument("--selection-regimes", type=str, default="")
    parser.add_argument("--promotion-baseline-accuracy", type=float, default=0.60)
    parser.add_argument("--weak-seed-baseline-accuracy", type=float, default=0.27)
    parser.add_argument("--promotion-accuracy-tolerance", type=float, default=0.05)
    parser.add_argument("--track", type=str, default="M19")
    parser.add_argument("--cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["stability_microgrid"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return _apply_track_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = _resolve_output_root(str(args.track), Path(args.output_root))
    assert_output_path_allowed("M", output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = _parse_float_list(args.learning_rate_list)
    augmentation_probs = _parse_float_list(args.augmentation_prob_list)
    format_augmentation_probs = _parse_float_list(args.format_augmentation_prob_list)
    surface_consistency_weights = (
        _parse_float_list(args.surface_consistency_weight_list)
        if str(args.surface_consistency_weight_list).strip()
        else [float(args.surface_consistency_weight)]
    )
    pointer_necessity_weights = (
        _parse_float_list(args.pointer_necessity_weight_list)
        if str(args.pointer_necessity_weight_list).strip()
        else [float(args.pointer_necessity_weight)]
    )
    seeds = _parse_int_list(args.seed_list)

    combo_rows: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        for augmentation_prob in augmentation_probs:
            for format_augmentation_prob in format_augmentation_probs:
                for surface_consistency_weight in surface_consistency_weights:
                    for pointer_necessity_weight in pointer_necessity_weights:
                        combo_slug = _combo_slug(
                            learning_rate,
                            augmentation_prob,
                            format_augmentation_prob,
                            surface_consistency_weight,
                            pointer_necessity_weight,
                        )
                        combo_dir = run_dir / combo_slug
                        combo_dir.mkdir(parents=True, exist_ok=True)
                        replication_report_path = combo_dir / M19_REGISTRY["M19"]["report_names"]["replication"]
                        _run_replication_if_needed(
                            repo_root=repo_root,
                            args=args,
                            run_id=f"{run_id}_{combo_slug}",
                            output_root=combo_dir,
                            report_path=replication_report_path,
                            learning_rate=learning_rate,
                            augmentation_prob=augmentation_prob,
                            format_augmentation_prob=format_augmentation_prob,
                            surface_consistency_weight=surface_consistency_weight,
                            pointer_necessity_weight=pointer_necessity_weight,
                            seeds=seeds,
                        )
                        replication_payload = _read_json(replication_report_path)
                        combo_rows.append(
                            _summarize_combo(
                                combo_slug=combo_slug,
                                learning_rate=learning_rate,
                                augmentation_prob=augmentation_prob,
                                format_augmentation_prob=format_augmentation_prob,
                                surface_consistency_weight=surface_consistency_weight,
                                pointer_necessity_weight=pointer_necessity_weight,
                                seeds=seeds,
                                report_path=replication_report_path,
                                payload=replication_payload,
                                args=args,
                            )
                        )

    best_by_mean = _best_row(combo_rows, "mean_accuracy")
    best_by_stability = _best_row(combo_rows, "stable_seed_rate", "mean_accuracy", "mean_audit_qformer_accuracy")
    best_balanced = _best_balanced_row(combo_rows)

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.stability_microgrid", "scripts/m19/run_m19_stability_microgrid.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "data_path": str(args.data_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "audit_data_path": str(args.audit_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "audit_eval_size": int(args.audit_eval_size),
            "epochs": int(args.epochs),
            "learning_rate_list": learning_rates,
            "augmentation_prob_list": augmentation_probs,
            "format_augmentation_prob_list": format_augmentation_probs,
            "surface_consistency_weight_list": surface_consistency_weights,
            "pointer_necessity_weight_list": pointer_necessity_weights,
            "pointer_necessity_margin": float(args.pointer_necessity_margin),
            "pointer_necessity_ablation_mode": str(args.pointer_necessity_ablation_mode),
            "seed_list": seeds,
            "surface_consistency_entity_rename": bool(args.surface_consistency_entity_rename),
            "surface_consistency_format_flatten": bool(args.surface_consistency_format_flatten),
            "surface_consistency_combined": bool(args.surface_consistency_combined),
            "surface_consistency_max_variants": int(args.surface_consistency_max_variants),
            "surface_consistency_task_weight": float(args.surface_consistency_task_weight),
            "typed_family_weight": float(args.typed_family_weight),
            "typed_arity_weight": float(args.typed_arity_weight),
            "family_separation_weight": float(args.family_separation_weight),
            "slot_usage_balance_weight": float(args.slot_usage_balance_weight),
            "operator_balance_weight": float(args.operator_balance_weight),
            "operator_top1_cap": float(args.operator_top1_cap),
            "cell_id": str(args.cell_id),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "max_latent_steps": int(args.max_latent_steps),
            "random_scale": float(args.random_scale),
            "checkpoint_selection_policy": str(args.checkpoint_selection_policy),
            "selection_purged_eval_size": _selection_surface_size(args),
            "selection_surface_size": _selection_surface_size(args),
            "selection_regimes": _selection_regimes(args),
            "promotion_baseline_accuracy": float(args.promotion_baseline_accuracy),
            "weak_seed_baseline_accuracy": float(args.weak_seed_baseline_accuracy),
            "promotion_accuracy_tolerance": float(args.promotion_accuracy_tolerance),
        },
        "grid_rows": combo_rows,
        "headline": {
            "configs_tested": len(combo_rows),
            "best_mean_accuracy": best_by_mean.get("mean_accuracy") if best_by_mean else None,
            "best_mean_accuracy_config": best_by_mean.get("combo_slug") if best_by_mean else None,
            "best_stable_seed_rate": best_by_stability.get("stable_seed_rate") if best_by_stability else None,
            "best_stability_config": best_by_stability.get("combo_slug") if best_by_stability else None,
            "best_balanced_config": best_balanced.get("combo_slug") if best_balanced else None,
            "best_balanced_mean_accuracy": best_balanced.get("mean_accuracy") if best_balanced else None,
            "best_balanced_stable_seed_rate": best_balanced.get("stable_seed_rate") if best_balanced else None,
            "recovered_seed_count": max((int(row.get("recovered_seed_count", 0)) for row in combo_rows), default=0),
            "promotion_gate_pass_count": sum(1 for row in combo_rows if bool(row.get("promotion_gate_pass"))),
        },
        "best_configs": {
            "best_by_mean_accuracy": best_by_mean,
            "best_by_stability": best_by_stability,
            "best_balanced": best_balanced,
        },
        "notes": [
            "This sweep targets weak-seed stabilization rather than architecture changes.",
            "Each cell is a full replication sub-run with its own JSON artifact so selection stays ledger-compatible.",
            "Recovered seeds count how many of the weak seeds cross the current stability threshold inside each combo.",
        ],
    }

    report_path = (
        Path(args.output_path)
        if args.output_path
        else (run_dir / M19_REGISTRY["M19"]["report_names"]["stability_microgrid"])
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote stability micro-grid report: {report_path}")


def _run_replication_if_needed(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    run_id: str,
    output_root: Path,
    report_path: Path,
    learning_rate: float,
    augmentation_prob: float,
    format_augmentation_prob: float,
    surface_consistency_weight: float,
    pointer_necessity_weight: float,
    seeds: list[int],
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_replication_suite.py"),
        "--base-model",
        str(args.base_model),
        "--data-path",
        str(args.data_path),
        "--eval-data-path",
        str(args.eval_data_path),
        "--audit-data-path",
        str(args.audit_data_path),
        "--eval-size",
        str(int(args.eval_size)),
        "--audit-eval-size",
        str(int(args.audit_eval_size)),
        "--epochs",
        str(int(args.epochs)),
        "--learning-rate",
        str(float(learning_rate)),
        "--seed-list",
        ",".join(str(seed) for seed in seeds),
        "--surface-consistency-weight",
        str(float(surface_consistency_weight)),
        "--surface-consistency-max-variants",
        str(int(args.surface_consistency_max_variants)),
        "--surface-consistency-task-weight",
        str(float(args.surface_consistency_task_weight)),
        "--pointer-necessity-weight",
        str(float(pointer_necessity_weight)),
        "--pointer-necessity-margin",
        str(float(args.pointer_necessity_margin)),
        "--pointer-necessity-ablation-mode",
        str(args.pointer_necessity_ablation_mode),
        "--typed-family-weight",
        str(float(args.typed_family_weight)),
        "--typed-arity-weight",
        str(float(args.typed_arity_weight)),
        "--family-separation-weight",
        str(float(args.family_separation_weight)),
        "--slot-usage-balance-weight",
        str(float(args.slot_usage_balance_weight)),
        "--operator-balance-weight",
        str(float(args.operator_balance_weight)),
        "--operator-top1-cap",
        str(float(args.operator_top1_cap)),
        "--track",
        str(args.track),
        "--cell-id",
        str(args.cell_id),
        "--num-queries",
        str(int(args.num_queries)),
        "--bottleneck-dim",
        str(int(args.bottleneck_dim)),
        "--scratchpad-length",
        str(int(args.scratchpad_length)),
        "--max-latent-steps",
        str(int(args.max_latent_steps)),
        "--random-scale",
        str(float(args.random_scale)),
        "--entity-rename-augmentation-prob",
        str(float(augmentation_prob)),
        "--format-flatten-augmentation-prob",
        str(float(format_augmentation_prob)),
        "--checkpoint-selection-policy",
        str(args.checkpoint_selection_policy),
        "--selection-purged-eval-size",
        str(int(_selection_surface_size(args))),
        "--selection-surface-size",
        str(int(_selection_surface_size(args))),
        "--selection-regimes",
        _selection_regimes(args),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--output-path",
        str(report_path),
        *_typed_replication_cli_args(args),
    ] + (
        ["--surface-consistency-entity-rename"] if bool(args.surface_consistency_entity_rename) else ["--no-surface-consistency-entity-rename"]
    ) + (
        ["--surface-consistency-format-flatten"] if bool(args.surface_consistency_format_flatten) else ["--no-surface-consistency-format-flatten"]
    ) + (
        ["--surface-consistency-combined"] if bool(args.surface_consistency_combined) else ["--no-surface-consistency-combined"]
    )
    run_if_needed(Path(report_path), cmd, Path(repo_root))


def _summarize_combo(
    *,
    combo_slug: str,
    learning_rate: float,
    augmentation_prob: float,
    format_augmentation_prob: float,
    surface_consistency_weight: float,
    pointer_necessity_weight: float,
    seeds: list[int],
    report_path: Path,
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    seed_runs = payload.get("seed_runs", []) if isinstance(payload.get("seed_runs"), list) else []
    recovered_seed_count = sum(1 for row in seed_runs if _is_recovered_seed(row))
    best_seed = _best_seed_row(seed_runs)
    seed_29_accuracy = _seed_accuracy(seed_runs, 29)
    mean_accuracy = metrics.get("mean_accuracy")
    promotion_gate_pass = _promotion_gate_pass(
        mean_accuracy=mean_accuracy,
        weak_seed_accuracy=seed_29_accuracy,
        baseline_accuracy=float(args.promotion_baseline_accuracy),
        weak_seed_baseline=float(args.weak_seed_baseline_accuracy),
        tolerance=float(args.promotion_accuracy_tolerance),
    )
    return {
        "combo_slug": combo_slug,
        "learning_rate": float(learning_rate),
        "entity_rename_augmentation_prob": float(augmentation_prob),
        "format_flatten_augmentation_prob": float(format_augmentation_prob),
        "surface_consistency_weight": float(surface_consistency_weight),
        "surface_consistency_task_weight": float(args.surface_consistency_task_weight),
        "pointer_necessity_weight": float(pointer_necessity_weight),
        "pointer_necessity_margin": float(args.pointer_necessity_margin),
        "pointer_necessity_ablation_mode": str(args.pointer_necessity_ablation_mode),
        "seed_list": seeds,
        "report_path": str(report_path).replace("\\", "/"),
        "mean_accuracy": mean_accuracy,
        "std_accuracy": metrics.get("std_accuracy"),
        "mean_avg_tokens": metrics.get("mean_avg_tokens"),
        "mean_avg_runway_tokens": metrics.get("mean_avg_runway_tokens"),
        "mean_audit_qformer_accuracy": metrics.get("mean_audit_qformer_accuracy"),
        "mean_lift_vs_random": metrics.get("mean_lift_vs_random"),
        "mean_lift_vs_scratchpad_only": metrics.get("mean_lift_vs_scratchpad_only"),
        "stable_seed_rate": metrics.get("stable_seed_rate"),
        "recovered_seed_count": recovered_seed_count,
        "seed_29_accuracy": seed_29_accuracy,
        "promotion_gate_pass": promotion_gate_pass,
        "promotion_gate": {
            "mean_accuracy_floor": float(args.promotion_baseline_accuracy) - float(args.promotion_accuracy_tolerance),
            "weak_seed_floor": float(args.weak_seed_baseline_accuracy),
            "strict_accuracy_canonical": True,
        },
        "best_seed": best_seed,
    }


def _best_seed_row(seed_runs: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_key: tuple[float, float, float] | None = None
    for row in seed_runs:
        key = (
            _float_or_neg_inf(row.get("overall_accuracy")),
            _float_or_neg_inf(row.get("audit_qformer_accuracy")),
            _float_or_neg_inf(row.get("lift_vs_random")),
        )
        if best_key is None or key > best_key:
            best = {
                "seed": row.get("seed"),
                "overall_accuracy": row.get("overall_accuracy"),
                "audit_qformer_accuracy": row.get("audit_qformer_accuracy"),
                "lift_vs_random": row.get("lift_vs_random"),
                "avg_tokens": row.get("avg_tokens"),
                "avg_runway_tokens": row.get("avg_runway_tokens"),
                "top_predictions": row.get("top_predictions", [])[:3]
                if isinstance(row.get("top_predictions"), list)
                else [],
            }
            best_key = key
    return best


def _best_row(rows: list[dict[str, Any]], *keys: str) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for row in rows:
        current = tuple(_float_or_neg_inf(row.get(key)) for key in keys)
        if best_key is None or current > best_key:
            best = row
            best_key = current
    return best


def _best_balanced_row(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    best: dict[str, Any] | None = None
    best_score: tuple[float, float, float, float] | None = None
    for row in rows:
        current = (
            _float_or_neg_inf(row.get("recovered_seed_count")),
            _float_or_neg_inf(row.get("stable_seed_rate")),
            _float_or_neg_inf(row.get("mean_accuracy")),
            _float_or_neg_inf(row.get("mean_audit_qformer_accuracy")),
        )
        if best_score is None or current > best_score:
            best = row
            best_score = current
    return best


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if not values:
        raise ValueError("expected at least one float value")
    return values


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("expected at least one integer value")
    return values


def _combo_slug(
    learning_rate: float,
    augmentation_prob: float,
    format_augmentation_prob: float,
    surface_consistency_weight: float,
    pointer_necessity_weight: float,
) -> str:
    lr = f"{learning_rate:.0e}".replace("-", "m").replace("+", "")
    aug = str(augmentation_prob).replace(".", "p")
    fmt = str(format_augmentation_prob).replace(".", "p")
    surf = str(surface_consistency_weight).replace(".", "p")
    ptr = str(pointer_necessity_weight).replace(".", "p")
    return f"lr_{lr}_aug_{aug}_fmt_{fmt}_surf_{surf}_ptr_{ptr}"


def _is_recovered_seed(row: dict[str, Any]) -> bool:
    try:
        return (
            float(row.get("overall_accuracy")) >= 0.20
            and float(row.get("lift_vs_random")) >= 0.10
            and float(row.get("audit_qformer_accuracy")) >= 0.90
        )
    except (TypeError, ValueError):
        return False


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _selection_surface_size(args: argparse.Namespace) -> int:
    explicit_surface = getattr(args, "selection_surface_size", None)
    legacy_purged = getattr(args, "selection_purged_eval_size", None)
    if explicit_surface is not None:
        return max(1, int(explicit_surface))
    if legacy_purged is not None:
        return max(1, int(legacy_purged))
    return 200


def _selection_regimes(args: argparse.Namespace) -> str:
    explicit = str(getattr(args, "selection_regimes", "") or "").strip()
    return explicit or f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}"


def _seed_accuracy(seed_runs: list[dict[str, Any]], seed: int) -> float | None:
    for row in seed_runs:
        try:
            if int(row.get("seed")) == int(seed):
                return float(row.get("overall_accuracy"))
        except (TypeError, ValueError):
            continue
    return None


def _promotion_gate_pass(
    *,
    mean_accuracy: object,
    weak_seed_accuracy: float | None,
    baseline_accuracy: float,
    weak_seed_baseline: float,
    tolerance: float,
) -> bool:
    try:
        mean_value = float(mean_accuracy)
    except (TypeError, ValueError):
        return False
    if weak_seed_accuracy is None:
        return False
    return mean_value >= float(baseline_accuracy) - float(tolerance) and float(weak_seed_accuracy) > float(weak_seed_baseline)


def _float_or_neg_inf(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


if __name__ == "__main__":
    main()
