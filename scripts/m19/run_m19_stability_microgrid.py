from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


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
    parser.add_argument("--checkpoint-selection-policy", type=str, default="final_only")
    parser.add_argument("--selection-purged-eval-size", type=int, default=100)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    learning_rates = _parse_float_list(args.learning_rate_list)
    augmentation_probs = _parse_float_list(args.augmentation_prob_list)
    format_augmentation_probs = _parse_float_list(args.format_augmentation_prob_list)
    seeds = _parse_int_list(args.seed_list)

    combo_rows: list[dict[str, Any]] = []
    for learning_rate in learning_rates:
        for augmentation_prob in augmentation_probs:
            for format_augmentation_prob in format_augmentation_probs:
                combo_slug = _combo_slug(learning_rate, augmentation_prob, format_augmentation_prob)
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
                    seeds=seeds,
                )
                replication_payload = _read_json(replication_report_path)
                combo_rows.append(
                    _summarize_combo(
                        combo_slug=combo_slug,
                        learning_rate=learning_rate,
                        augmentation_prob=augmentation_prob,
                        format_augmentation_prob=format_augmentation_prob,
                        seeds=seeds,
                        report_path=replication_report_path,
                        payload=replication_payload,
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
            "seed_list": seeds,
            "cell_id": str(args.cell_id),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "max_latent_steps": int(args.max_latent_steps),
            "random_scale": float(args.random_scale),
            "checkpoint_selection_policy": str(args.checkpoint_selection_policy),
            "selection_purged_eval_size": int(args.selection_purged_eval_size),
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
    seeds: list[int],
) -> None:
    if Path(report_path).exists():
        return
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
        str(int(args.selection_purged_eval_size)),
        "--output-root",
        str(output_root),
        "--run-id",
        run_id,
        "--output-path",
        str(report_path),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _summarize_combo(
    *,
    combo_slug: str,
    learning_rate: float,
    augmentation_prob: float,
    format_augmentation_prob: float,
    seeds: list[int],
    report_path: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    seed_runs = payload.get("seed_runs", []) if isinstance(payload.get("seed_runs"), list) else []
    recovered_seed_count = sum(1 for row in seed_runs if _is_recovered_seed(row))
    best_seed = _best_seed_row(seed_runs)
    return {
        "combo_slug": combo_slug,
        "learning_rate": float(learning_rate),
        "entity_rename_augmentation_prob": float(augmentation_prob),
        "format_flatten_augmentation_prob": float(format_augmentation_prob),
        "seed_list": seeds,
        "report_path": str(report_path).replace("\\", "/"),
        "mean_accuracy": metrics.get("mean_accuracy"),
        "std_accuracy": metrics.get("std_accuracy"),
        "mean_avg_tokens": metrics.get("mean_avg_tokens"),
        "mean_audit_qformer_accuracy": metrics.get("mean_audit_qformer_accuracy"),
        "mean_lift_vs_random": metrics.get("mean_lift_vs_random"),
        "mean_lift_vs_scratchpad_only": metrics.get("mean_lift_vs_scratchpad_only"),
        "stable_seed_rate": metrics.get("stable_seed_rate"),
        "recovered_seed_count": recovered_seed_count,
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


def _combo_slug(learning_rate: float, augmentation_prob: float, format_augmentation_prob: float) -> str:
    lr = f"{learning_rate:.0e}".replace("-", "m").replace("+", "")
    aug = str(augmentation_prob).replace(".", "p")
    fmt = str(format_augmentation_prob).replace(".", "p")
    return f"lr_{lr}_aug_{aug}_fmt_{fmt}"


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


def _float_or_neg_inf(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


if __name__ == "__main__":
    main()
