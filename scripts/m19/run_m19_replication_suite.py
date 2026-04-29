from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.m19.integrity import write_jsonl
from lojban_evolution.m19.kill_tests import build_format_flattened_rows, build_purged_eval_rows
from lojban_evolution.m19.training import checkpoint_selection_score, select_best_checkpoint
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Run seed replications for the active M19 mainline cell.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", type=Path, default=Path(registry["dataset_defaults"]["train"]))
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--audit-data-path", type=Path, default=Path(registry["dataset_defaults"]["audit"]))
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--audit-eval-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed-list", type=str, default="19,23,29")
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--entity-rename-augmentation-prob", type=float, default=0.0)
    parser.add_argument("--format-flatten-augmentation-prob", type=float, default=0.0)
    parser.add_argument("--query-repulsion-weight", type=float, default=0.0)
    parser.add_argument("--query-repulsion-margin", type=float, default=0.15)
    parser.add_argument("--checkpoint-selection-policy", type=str, default="final_only")
    parser.add_argument("--selection-purged-eval-size", type=int, default=100)
    parser.add_argument("--track", type=str, default="M19")
    parser.add_argument("--cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["replication"]))
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
    if str(args.checkpoint_selection_policy) not in {"final_only", "audit_purged", "audit_purged_format"}:
        raise ValueError("checkpoint_selection_policy must be one of: final_only, audit_purged, audit_purged_format")

    seeds = [int(part.strip()) for part in str(args.seed_list).split(",") if part.strip()]
    selection_paths = _prepare_selection_slices(
        run_dir=run_dir,
        train_path=Path(args.data_path),
        eval_path=Path(args.eval_data_path),
        selection_purged_eval_size=int(args.selection_purged_eval_size),
    )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = seed_dir / f"{args.cell_id}.pt"
        train_report_path = seed_dir / "train_report.json"
        benchmark_report_path = seed_dir / "benchmark_report.json"
        audit_report_path = seed_dir / "audit_report.json"

        _run_if_missing(
            train_report_path,
            [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "train_m19_mainline.py"),
                "--base-model",
                str(args.base_model),
                "--data-path",
                str(args.data_path),
                "--num-queries",
                str(int(args.num_queries)),
                "--bottleneck-dim",
                str(int(args.bottleneck_dim)),
                "--scratchpad-length",
                str(int(args.scratchpad_length)),
                "--epochs",
                str(int(args.epochs)),
                "--learning-rate",
                str(float(args.learning_rate)),
                "--seed",
                str(int(seed)),
                "--track",
                str(args.track),
                "--cell-id",
                str(args.cell_id),
                "--entity-rename-augmentation-prob",
                str(float(args.entity_rename_augmentation_prob)),
                "--format-flatten-augmentation-prob",
                str(float(args.format_flatten_augmentation_prob)),
                "--query-repulsion-weight",
                str(float(args.query_repulsion_weight)),
                "--query-repulsion-margin",
                str(float(args.query_repulsion_margin)),
                "--save-epoch-checkpoints" if str(args.checkpoint_selection_policy) != "final_only" else "--no-save-epoch-checkpoints",
                "--checkpoint-output-path",
                str(checkpoint_path),
                "--report-output-path",
                str(train_report_path),
            ],
            repo_root,
        )
        train_report = _read_json(train_report_path)
        selection_payload = _select_checkpoint_if_needed(
            repo_root=repo_root,
            args=args,
            seed=seed,
            seed_dir=seed_dir,
            train_report=train_report,
            final_checkpoint_path=checkpoint_path,
            selection_paths=selection_paths,
        )
        selected_checkpoint_path = Path(str(selection_payload["selected_checkpoint_path"]))
        _run_if_missing(
            benchmark_report_path,
            [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
                "--base-model",
                str(args.base_model),
                "--bridge-path",
                str(selected_checkpoint_path),
                "--eval-data-path",
                str(args.eval_data_path),
                "--eval-size",
                str(int(args.eval_size)),
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
                "--seed",
                str(int(seed)),
                "--track",
                str(args.track),
                "--cell-id",
                str(args.cell_id),
                "--regimes",
                f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}",
                "--output-path",
                str(benchmark_report_path),
            ],
            repo_root,
        )
        _run_if_missing(
            audit_report_path,
            [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "run_m19_audit.py"),
                "--base-model",
                str(args.base_model),
                "--bridge-path",
                str(selected_checkpoint_path),
                "--dataset-path",
                str(args.audit_data_path),
                "--eval-size",
                str(int(args.audit_eval_size)),
                "--scratchpad-length",
                str(int(args.scratchpad_length)),
                "--num-queries",
                str(int(args.num_queries)),
                "--bottleneck-dim",
                str(int(args.bottleneck_dim)),
                "--max-latent-steps",
                str(int(args.max_latent_steps)),
                "--random-scale",
                str(float(args.random_scale)),
                "--seed",
                str(int(seed)),
                "--track",
                str(args.track),
                "--cell-id",
                str(args.cell_id),
                "--output-path",
                str(audit_report_path),
            ],
            repo_root,
        )

        benchmark_report = _read_json(benchmark_report_path)
        audit_report = _read_json(audit_report_path)
        benchmark_metrics = benchmark_report.get("metrics", {}) if isinstance(benchmark_report.get("metrics"), dict) else {}
        benchmark_results = benchmark_report.get("results", {}) if isinstance(benchmark_report.get("results"), dict) else {}
        prediction_summaries = (
            benchmark_report.get("prediction_summaries", {})
            if isinstance(benchmark_report.get("prediction_summaries"), dict)
            else {}
        )
        mainline_summary = (
            prediction_summaries.get(str(args.cell_id), {})
            if isinstance(prediction_summaries.get(str(args.cell_id), {}), dict)
            else {}
        )
        audit_headline = audit_report.get("headline", {}) if isinstance(audit_report.get("headline"), dict) else {}
        random_accuracy = _result_accuracy(benchmark_results, "RANDOM-SHAPE")
        scratchpad_accuracy = _result_accuracy(benchmark_results, "SCRATCHPAD-ONLY")
        mainline_accuracy = benchmark_metrics.get("overall_accuracy", benchmark_metrics.get("strict_accuracy"))
        rows.append(
            {
                "seed": int(seed),
                "checkpoint_path": str(selected_checkpoint_path).replace("\\", "/"),
                "final_checkpoint_path": str(checkpoint_path).replace("\\", "/"),
                "train_report": str(train_report_path).replace("\\", "/"),
                "benchmark_report": str(benchmark_report_path).replace("\\", "/"),
                "audit_report": str(audit_report_path).replace("\\", "/"),
                "final_mean_loss": train_report.get("final_mean_loss"),
                "checkpoint_selection_policy": str(args.checkpoint_selection_policy),
                "selection": selection_payload,
                "overall_accuracy": mainline_accuracy,
                "avg_tokens": benchmark_metrics.get("avg_tokens"),
                "accuracy_per_token": benchmark_metrics.get("accuracy_per_token"),
                "lift_vs_en_cot": benchmark_metrics.get("lift_vs_en_cot"),
                "lift_vs_random": benchmark_metrics.get("lift_vs_random"),
                "random_accuracy": random_accuracy,
                "scratchpad_only_accuracy": scratchpad_accuracy,
                "lift_vs_scratchpad_only": _safe_delta(mainline_accuracy, scratchpad_accuracy),
                "unique_prediction_count": mainline_summary.get("unique_prediction_count"),
                "empty_prediction_rate": mainline_summary.get("empty_prediction_rate"),
                "top_predictions": mainline_summary.get("top_predictions", [])[:5]
                if isinstance(mainline_summary.get("top_predictions"), list)
                else [],
                "audit_qformer_accuracy": audit_headline.get("qformer_accuracy"),
                "audit_lift_vs_random": audit_headline.get("lift_vs_random"),
            }
        )

    accuracy_values = [float(row["overall_accuracy"]) for row in rows if row.get("overall_accuracy") is not None]
    token_values = [float(row["avg_tokens"]) for row in rows if row.get("avg_tokens") is not None]
    audit_values = [float(row["audit_qformer_accuracy"]) for row in rows if row.get("audit_qformer_accuracy") is not None]
    random_lift_values = [float(row["lift_vs_random"]) for row in rows if row.get("lift_vs_random") is not None]
    scratchpad_lift_values = [
        float(row["lift_vs_scratchpad_only"]) for row in rows if row.get("lift_vs_scratchpad_only") is not None
    ]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.replication", "scripts/m19/run_m19_replication_suite.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "data_path": str(args.data_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "audit_data_path": str(args.audit_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "audit_eval_size": int(args.audit_eval_size),
            "epochs": int(args.epochs),
            "learning_rate": float(args.learning_rate),
            "seed_list": seeds,
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "max_latent_steps": int(args.max_latent_steps),
            "random_scale": float(args.random_scale),
            "entity_rename_augmentation_prob": float(args.entity_rename_augmentation_prob),
            "format_flatten_augmentation_prob": float(args.format_flatten_augmentation_prob),
            "query_repulsion_weight": float(args.query_repulsion_weight),
            "query_repulsion_margin": float(args.query_repulsion_margin),
            "checkpoint_selection_policy": str(args.checkpoint_selection_policy),
            "selection_purged_eval_size": int(args.selection_purged_eval_size),
            "cell_id": str(args.cell_id),
        },
        "seed_runs": rows,
        "metrics": {
            "replication_count": len(rows),
            "mean_accuracy": _stat_mean(accuracy_values),
            "std_accuracy": _stat_std(accuracy_values),
            "min_accuracy": min(accuracy_values) if accuracy_values else None,
            "max_accuracy": max(accuracy_values) if accuracy_values else None,
            "mean_avg_tokens": _stat_mean(token_values),
            "std_avg_tokens": _stat_std(token_values),
            "mean_audit_qformer_accuracy": _stat_mean(audit_values),
            "min_audit_qformer_accuracy": min(audit_values) if audit_values else None,
            "max_audit_qformer_accuracy": max(audit_values) if audit_values else None,
            "mean_lift_vs_random": _stat_mean(random_lift_values),
            "mean_lift_vs_scratchpad_only": _stat_mean(scratchpad_lift_values),
            "stable_seed_rate": _stable_seed_rate(rows),
        },
        "headline": {
            "mean_accuracy": _stat_mean(accuracy_values),
            "std_accuracy": _stat_std(accuracy_values),
            "mean_avg_tokens": _stat_mean(token_values),
            "mean_audit_qformer_accuracy": _stat_mean(audit_values),
            "stable_seed_rate": _stable_seed_rate(rows),
        },
        "notes": [
            "Replication suite retrains the active M19 mainline cell across multiple seeds under the current training contract.",
            "Each seed emits train, benchmark, and audit artifacts so stability can be compared without console-only claims.",
            "Checkpoint selection can optionally prioritize purged robustness plus audit behavior rather than final train loss alone.",
        ],
    }

    report_path = Path(args.output_path) if args.output_path else (run_dir / M19_REGISTRY["M19"]["report_names"]["replication"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote replication report: {report_path}")


def _run_if_missing(report_path: Path, cmd: list[str], repo_root: Path) -> None:
    if Path(report_path).exists():
        return
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _prepare_selection_slices(
    *,
    run_dir: Path,
    train_path: Path,
    eval_path: Path,
    selection_purged_eval_size: int,
) -> dict[str, object]:
    _, purged_rows = build_purged_eval_rows(train_path, eval_path, selection_purged_eval_size)
    format_rows = build_format_flattened_rows(purged_rows)
    selection_dir = run_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    purged_path = write_jsonl(selection_dir / "selection_purged_eval.jsonl", purged_rows)
    format_path = write_jsonl(selection_dir / "selection_format_eval.jsonl", format_rows)
    return {
        "purged_eval_path": purged_path,
        "format_eval_path": format_path,
        "purged_count": len(purged_rows),
        "format_count": len(format_rows),
    }


def _select_checkpoint_if_needed(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    seed: int,
    seed_dir: Path,
    train_report: dict[str, object],
    final_checkpoint_path: Path,
    selection_paths: dict[str, object],
) -> dict[str, Any]:
    policy = str(args.checkpoint_selection_policy)
    if policy == "final_only":
        return {
            "selected_checkpoint_path": str(final_checkpoint_path).replace("\\", "/"),
            "selected_epoch": None,
            "selection_score": checkpoint_selection_score(
                purged_accuracy=None,
                audit_qformer_accuracy=None,
                format_accuracy=None,
                final_mean_loss=_safe_float(train_report.get("final_mean_loss")),
                policy=policy,
            ),
            "candidates": [],
        }

    epoch_paths = train_report.get("epoch_checkpoint_paths")
    epoch_metrics = train_report.get("epoch_metrics")
    if not isinstance(epoch_paths, list) or not epoch_paths:
        return {
            "selected_checkpoint_path": str(final_checkpoint_path).replace("\\", "/"),
            "selected_epoch": None,
            "selection_score": None,
            "candidates": [],
            "fallback_reason": "missing_epoch_checkpoints",
        }

    epoch_metric_map: dict[int, dict[str, object]] = {}
    if isinstance(epoch_metrics, list):
        for row in epoch_metrics:
            if isinstance(row, dict) and row.get("epoch") is not None:
                try:
                    epoch_metric_map[int(float(row["epoch"]))] = row
                except (TypeError, ValueError):
                    continue

    selection_dir = seed_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selection_purged_eval_path = Path(str(selection_paths["purged_eval_path"]))
    selection_format_eval_path = Path(str(selection_paths["format_eval_path"]))
    selection_purged_count = int(selection_paths["purged_count"])
    selection_format_count = int(selection_paths["format_count"])
    candidates: list[dict[str, Any]] = []
    for idx, raw_path in enumerate(epoch_paths, start=1):
        epoch_checkpoint_path = Path(str(raw_path))
        epoch_purged_report_path = selection_dir / f"epoch_{idx}_purged.json"
        epoch_format_report_path = selection_dir / f"epoch_{idx}_format.json"
        epoch_audit_report_path = selection_dir / f"epoch_{idx}_audit.json"
        _run_if_missing(
            epoch_purged_report_path,
            [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
                "--base-model",
                str(args.base_model),
                "--bridge-path",
                str(epoch_checkpoint_path),
                "--eval-data-path",
                str(selection_purged_eval_path),
                "--eval-size",
                str(int(selection_purged_count)),
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
                "--seed",
                str(int(seed)),
                "--track",
                str(args.track),
                "--cell-id",
                str(args.cell_id),
                "--regimes",
                f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}",
                "--output-path",
                str(epoch_purged_report_path),
            ],
            repo_root,
        )
        if policy == "audit_purged_format":
            _run_if_missing(
                epoch_format_report_path,
                [
                    sys.executable,
                    str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
                    "--base-model",
                    str(args.base_model),
                    "--bridge-path",
                    str(epoch_checkpoint_path),
                    "--eval-data-path",
                    str(selection_format_eval_path),
                    "--eval-size",
                    str(int(selection_format_count)),
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
                    "--seed",
                    str(int(seed)),
                    "--track",
                    str(args.track),
                    "--cell-id",
                    str(args.cell_id),
                    "--regimes",
                    f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}",
                    "--output-path",
                    str(epoch_format_report_path),
                ],
                repo_root,
            )
        _run_if_missing(
            epoch_audit_report_path,
            [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "run_m19_audit.py"),
                "--base-model",
                str(args.base_model),
                "--bridge-path",
                str(epoch_checkpoint_path),
                "--dataset-path",
                str(args.audit_data_path),
                "--eval-size",
                str(int(args.audit_eval_size)),
                "--scratchpad-length",
                str(int(args.scratchpad_length)),
                "--num-queries",
                str(int(args.num_queries)),
                "--bottleneck-dim",
                str(int(args.bottleneck_dim)),
                "--max-latent-steps",
                str(int(args.max_latent_steps)),
                "--random-scale",
                str(float(args.random_scale)),
                "--seed",
                str(int(seed)),
                "--track",
                str(args.track),
                "--cell-id",
                str(args.cell_id),
                "--output-path",
                str(epoch_audit_report_path),
            ],
            repo_root,
        )
        purged_report = _read_json(epoch_purged_report_path)
        format_report = _read_json(epoch_format_report_path) if epoch_format_report_path.exists() else {}
        audit_report = _read_json(epoch_audit_report_path)
        purged_metrics = purged_report.get("metrics", {}) if isinstance(purged_report.get("metrics"), dict) else {}
        format_metrics = format_report.get("metrics", {}) if isinstance(format_report.get("metrics"), dict) else {}
        audit_headline = audit_report.get("headline", {}) if isinstance(audit_report.get("headline"), dict) else {}
        epoch_metric = epoch_metric_map.get(idx, {})
        candidate = {
            "epoch": idx,
            "checkpoint_path": str(epoch_checkpoint_path).replace("\\", "/"),
            "purged_accuracy": purged_metrics.get("overall_accuracy", purged_metrics.get("strict_accuracy")),
            "format_accuracy": format_metrics.get("overall_accuracy", format_metrics.get("strict_accuracy")),
            "audit_qformer_accuracy": audit_headline.get("qformer_accuracy"),
            "final_mean_loss": epoch_metric.get("mean_loss"),
            "purged_report": str(epoch_purged_report_path).replace("\\", "/"),
            "format_report": str(epoch_format_report_path).replace("\\", "/") if epoch_format_report_path.exists() else None,
            "audit_report": str(epoch_audit_report_path).replace("\\", "/"),
        }
        candidate["selection_score"] = checkpoint_selection_score(
            purged_accuracy=_safe_float(candidate.get("purged_accuracy")),
            audit_qformer_accuracy=_safe_float(candidate.get("audit_qformer_accuracy")),
            format_accuracy=_safe_float(candidate.get("format_accuracy")),
            final_mean_loss=_safe_float(candidate.get("final_mean_loss")),
            policy=policy,
        )
        candidates.append(candidate)

    best = select_best_checkpoint(candidates, policy)
    if best is None:
        return {
            "selected_checkpoint_path": str(final_checkpoint_path).replace("\\", "/"),
            "selected_epoch": None,
            "selection_score": None,
            "candidates": candidates,
            "fallback_reason": "no_selection_candidates",
        }
    return {
        "selected_checkpoint_path": str(best["checkpoint_path"]),
        "selected_epoch": best.get("epoch"),
        "selection_score": best.get("selection_score"),
        "candidates": candidates,
    }


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _result_accuracy(results: object, key: str) -> float | None:
    if not isinstance(results, dict):
        return None
    row = results.get(key)
    if not isinstance(row, dict):
        return None
    value = row.get("accuracy")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_delta(left: object, right: object) -> float | None:
    try:
        if left is None or right is None:
            return None
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_seed_rate(rows: list[dict[str, object]]) -> float | None:
    if not rows:
        return None
    stable = 0
    for row in rows:
        accuracy = row.get("overall_accuracy")
        random_lift = row.get("lift_vs_random")
        audit_accuracy = row.get("audit_qformer_accuracy")
        try:
            if float(accuracy) >= 0.30 and float(random_lift) >= 0.20 and float(audit_accuracy) >= 0.90:
                stable += 1
        except (TypeError, ValueError):
            continue
    return stable / max(1, len(rows))


def _stat_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _stat_std(values: list[float]) -> float | None:
    if len(values) <= 1:
        return 0.0 if values else None
    return pstdev(values)


if __name__ == "__main__":
    main()
