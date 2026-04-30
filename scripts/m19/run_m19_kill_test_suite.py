from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.m19.kill_tests import (
    build_entity_anonymized_rows,
    build_entity_renamed_rows,
    build_format_flattened_rows,
    build_numeric_normalized_rows,
    build_purged_eval_rows,
    classify_kill_test_status,
    compute_kill_test_metrics,
)
from lojban_evolution.m19.integrity import write_jsonl
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


def _track_key(track: str) -> str:
    candidate = str(track or "").strip()
    return candidate if candidate in M19_REGISTRY else "M19"


def _resolve_output_root(track: str, output_root: Path) -> Path:
    track_key = _track_key(track)
    registry = M19_REGISTRY[track_key]
    default_root = Path(M19_REGISTRY["M19"]["output_roots"]["kill_tests"])
    if track_key != "M19" and Path(output_root) == default_root:
        return Path(registry["output_roots"]["kill_tests"])
    return Path(output_root)


def _apply_track_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = M19_REGISTRY.get(_track_key(args.track), {}).get("defaults", {})
    if defaults:
        if not str(args.typed_slot_layout).strip() and defaults.get("typed_slot_layout"):
            args.typed_slot_layout = str(defaults["typed_slot_layout"])
        if str(args.arity_router_mode).strip() == "soft" and defaults.get("arity_router_mode"):
            args.arity_router_mode = str(defaults["arity_router_mode"])
        if str(args.geometry_mode).strip() == "euclidean" and defaults.get("geometry_mode"):
            args.geometry_mode = str(defaults["geometry_mode"])
        if not args.gumbel_hard and str(defaults.get("arity_router_mode", "")).strip() == "gumbel_hard":
            args.gumbel_hard = True
    return args


def _typed_bridge_cli_args(args: argparse.Namespace) -> list[str]:
    cli_args = [
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
    parser = argparse.ArgumentParser(description="Run the broader M19 kill-test suite on the purged benchmark slice.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", required=True, type=Path)
    parser.add_argument("--train-data-path", type=Path, default=Path(registry["dataset_defaults"]["train"]))
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--track", type=str, default="M19")
    parser.add_argument("--cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--purged-report-path", type=Path, default=None)
    parser.add_argument("--masked-report-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["kill_tests"]))
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

    slices_dir = run_dir / "slices"
    _, purged_rows = build_purged_eval_rows(Path(args.train_data_path), Path(args.eval_data_path), int(args.eval_size))
    entity_rows = build_entity_anonymized_rows(purged_rows)
    entity_renamed_rows = build_entity_renamed_rows(purged_rows)
    format_rows = build_format_flattened_rows(purged_rows)
    numeric_rows = build_numeric_normalized_rows(purged_rows)

    purged_eval_path = write_jsonl(slices_dir / "purged_eval.jsonl", purged_rows)
    entity_eval_path = write_jsonl(slices_dir / "entity_eval.jsonl", entity_rows)
    entity_renamed_eval_path = write_jsonl(slices_dir / "entity_renamed_eval.jsonl", entity_renamed_rows)
    format_eval_path = write_jsonl(slices_dir / "format_eval.jsonl", format_rows)
    numeric_eval_path = write_jsonl(slices_dir / "numeric_eval.jsonl", numeric_rows)

    purged_report_path = args.purged_report_path or (run_dir / "benchmark_purged.json")
    masked_report_path = args.masked_report_path or (run_dir / "benchmark_masked.json")
    entity_report_path = run_dir / "benchmark_entity.json"
    entity_renamed_report_path = run_dir / "benchmark_entity_renamed.json"
    format_report_path = run_dir / "benchmark_format.json"
    numeric_report_path = run_dir / "benchmark_numeric.json"

    _run_benchmark_if_needed(
        output_path=purged_report_path,
        repo_root=repo_root,
        args=args,
        eval_data_path=purged_eval_path,
    )
    _run_benchmark_if_needed(
        output_path=entity_report_path,
        repo_root=repo_root,
        args=args,
        eval_data_path=entity_eval_path,
    )
    _run_benchmark_if_needed(
        output_path=entity_renamed_report_path,
        repo_root=repo_root,
        args=args,
        eval_data_path=entity_renamed_eval_path,
    )
    _run_benchmark_if_needed(
        output_path=format_report_path,
        repo_root=repo_root,
        args=args,
        eval_data_path=format_eval_path,
    )
    _run_benchmark_if_needed(
        output_path=numeric_report_path,
        repo_root=repo_root,
        args=args,
        eval_data_path=numeric_eval_path,
    )

    purged_report = _read_json(purged_report_path)
    entity_report = _read_json(entity_report_path)
    entity_renamed_report = _read_json(entity_renamed_report_path)
    format_report = _read_json(format_report_path)
    numeric_report = _read_json(numeric_report_path)
    masked_report = _read_json(masked_report_path) if Path(masked_report_path).exists() else None

    metrics = compute_kill_test_metrics(
        purged_report=purged_report,
        masked_report=masked_report,
        entity_report=entity_report,
        entity_renamed_report=entity_renamed_report,
        format_report=format_report,
        numeric_report=numeric_report,
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.kill_tests", "scripts/m19/run_m19_kill_test_suite.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/"),
            "train_data_path": str(args.train_data_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "max_latent_steps": int(args.max_latent_steps),
            "random_scale": float(args.random_scale),
            "seed": int(args.seed),
            "cell_id": str(args.cell_id),
        },
        "dataset_slices": {
            "purged_count": len(purged_rows),
            "purged_eval_path": str(purged_eval_path).replace("\\", "/"),
            "entity_eval_path": str(entity_eval_path).replace("\\", "/"),
            "entity_renamed_eval_path": str(entity_renamed_eval_path).replace("\\", "/"),
            "format_eval_path": str(format_eval_path).replace("\\", "/"),
            "numeric_eval_path": str(numeric_eval_path).replace("\\", "/"),
        },
        "reports": {
            "purged": str(purged_report_path).replace("\\", "/"),
            "masked": str(masked_report_path).replace("\\", "/") if masked_report_path else None,
            "entity": str(entity_report_path).replace("\\", "/"),
            "entity_renamed": str(entity_renamed_report_path).replace("\\", "/"),
            "format": str(format_report_path).replace("\\", "/"),
            "numeric": str(numeric_report_path).replace("\\", "/"),
        },
        "metrics": metrics,
        "headline": {
            "purged_accuracy": metrics.get("purged_accuracy"),
            "entity_accuracy": metrics.get("entity_accuracy"),
            "entity_renamed_accuracy": metrics.get("entity_renamed_accuracy"),
            "format_accuracy": metrics.get("format_accuracy"),
            "numeric_accuracy": metrics.get("numeric_accuracy"),
            "masked_accuracy": metrics.get("masked_accuracy"),
            "kill_test_status": classify_kill_test_status(metrics),
        },
        "notes": [
            "Kill tests are evaluated on the purged slice only so train-eval overlap does not inflate robustness claims.",
            "Entity anonymization, format flattening, and numeric normalization preserve task semantics while removing easy lexical anchors.",
            "Masked accuracy is imported from the integrity suite when available so the broader kill surface stays ledger-compatible.",
        ],
    }

    report_path = Path(args.output_path) if args.output_path else (run_dir / M19_REGISTRY["M19"]["report_names"]["kill_tests"])
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote kill-test report: {report_path}")


def _run_benchmark_if_needed(
    *,
    output_path: Path,
    repo_root: Path,
    args: argparse.Namespace,
    eval_data_path: Path,
) -> None:
    if Path(output_path).exists():
        return
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
        "--base-model",
        str(args.base_model),
        "--bridge-path",
        str(args.bridge_path),
        "--eval-data-path",
        str(eval_data_path),
        "--eval-size",
        str(_count_rows(eval_data_path)),
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
        str(int(args.seed)),
        "--track",
        str(args.track),
        "--cell-id",
        str(args.cell_id),
        "--regimes",
        f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}",
        "--output-path",
        str(output_path),
        *_typed_bridge_cli_args(args),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _count_rows(path: Path) -> int:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
