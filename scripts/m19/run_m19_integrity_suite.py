from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.m19.integrity import (
    build_masked_eval_rows,
    build_train_pair_index,
    classify_integrity_status,
    compute_integrity_metrics,
    load_jsonl_rows,
    split_eval_rows_by_overlap,
    write_jsonl,
)
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19"]
    parser = argparse.ArgumentParser(description="Run the M19 integrity suite with full, purged, overlap, masked, and audit controls.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", required=True, type=Path)
    parser.add_argument("--train-data-path", type=Path, default=Path(registry["dataset_defaults"]["train"]))
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--audit-data-path", type=Path, default=Path(registry["dataset_defaults"]["audit"]))
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--audit-eval-size", type=int, default=10)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--track", type=str, default="M19")
    parser.add_argument("--cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument("--full-report-path", type=Path, default=None)
    parser.add_argument("--purged-report-path", type=Path, default=None)
    parser.add_argument("--overlap-report-path", type=Path, default=None)
    parser.add_argument("--masked-report-path", type=Path, default=None)
    parser.add_argument("--audit-report-path", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m19_integrity_suite"))
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

    slices_dir = run_dir / "slices"
    full_rows = load_jsonl_rows(Path(args.eval_data_path))[: int(args.eval_size)]
    train_rows = load_jsonl_rows(Path(args.train_data_path))
    train_pairs = build_train_pair_index(train_rows)
    overlap_rows, purged_rows = split_eval_rows_by_overlap(full_rows, train_pairs)
    masked_purged_rows = build_masked_eval_rows(purged_rows)

    full_eval_path = write_jsonl(slices_dir / "full_eval.jsonl", full_rows)
    overlap_eval_path = write_jsonl(slices_dir / "overlap_eval.jsonl", overlap_rows)
    purged_eval_path = write_jsonl(slices_dir / "purged_eval.jsonl", purged_rows)
    masked_eval_path = write_jsonl(slices_dir / "masked_purged_eval.jsonl", masked_purged_rows)

    benchmark_full_path = args.full_report_path or (run_dir / "benchmark_full.json")
    benchmark_purged_path = args.purged_report_path or (run_dir / "benchmark_purged.json")
    benchmark_overlap_path = args.overlap_report_path or (run_dir / "benchmark_overlap.json")
    benchmark_masked_path = args.masked_report_path or (run_dir / "benchmark_masked_purged.json")
    audit_report_path = args.audit_report_path or (run_dir / "audit_report.json")

    _run_benchmark_if_needed(
        output_path=benchmark_full_path,
        repo_root=repo_root,
        base_model=args.base_model,
        bridge_path=args.bridge_path,
        eval_data_path=full_eval_path,
        eval_size=len(full_rows),
        num_queries=args.num_queries,
        bottleneck_dim=args.bottleneck_dim,
        scratchpad_length=args.scratchpad_length,
        max_latent_steps=args.max_latent_steps,
        random_scale=args.random_scale,
        seed=args.seed,
        track=args.track,
        cell_id=args.cell_id,
    )
    _run_benchmark_if_needed(
        output_path=benchmark_purged_path,
        repo_root=repo_root,
        base_model=args.base_model,
        bridge_path=args.bridge_path,
        eval_data_path=purged_eval_path,
        eval_size=len(purged_rows),
        num_queries=args.num_queries,
        bottleneck_dim=args.bottleneck_dim,
        scratchpad_length=args.scratchpad_length,
        max_latent_steps=args.max_latent_steps,
        random_scale=args.random_scale,
        seed=args.seed,
        track=args.track,
        cell_id=args.cell_id,
    )
    _run_benchmark_if_needed(
        output_path=benchmark_overlap_path,
        repo_root=repo_root,
        base_model=args.base_model,
        bridge_path=args.bridge_path,
        eval_data_path=overlap_eval_path,
        eval_size=len(overlap_rows),
        num_queries=args.num_queries,
        bottleneck_dim=args.bottleneck_dim,
        scratchpad_length=args.scratchpad_length,
        max_latent_steps=args.max_latent_steps,
        random_scale=args.random_scale,
        seed=args.seed,
        track=args.track,
        cell_id=args.cell_id,
    )
    _run_benchmark_if_needed(
        output_path=benchmark_masked_path,
        repo_root=repo_root,
        base_model=args.base_model,
        bridge_path=args.bridge_path,
        eval_data_path=masked_eval_path,
        eval_size=len(masked_purged_rows),
        num_queries=args.num_queries,
        bottleneck_dim=args.bottleneck_dim,
        scratchpad_length=args.scratchpad_length,
        max_latent_steps=args.max_latent_steps,
        random_scale=args.random_scale,
        seed=args.seed,
        track=args.track,
        cell_id=args.cell_id,
        regimes=f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}",
    )
    _run_audit_if_needed(
        output_path=audit_report_path,
        repo_root=repo_root,
        base_model=args.base_model,
        bridge_path=args.bridge_path,
        audit_data_path=args.audit_data_path,
        audit_eval_size=args.audit_eval_size,
        scratchpad_length=args.scratchpad_length,
        num_queries=args.num_queries,
        bottleneck_dim=args.bottleneck_dim,
        max_latent_steps=args.max_latent_steps,
        random_scale=args.random_scale,
        seed=args.seed,
        track=args.track,
        cell_id=args.cell_id,
    )

    full_report = _read_json(benchmark_full_path)
    purged_report = _read_json(benchmark_purged_path)
    overlap_report = _read_json(benchmark_overlap_path)
    masked_report = _read_json(benchmark_masked_path)
    audit_report = _read_json(audit_report_path)
    metrics = compute_integrity_metrics(
        full_report=full_report,
        purged_report=purged_report,
        overlap_report=overlap_report,
        masked_report=masked_report,
        audit_report=audit_report,
        overlap_size=len(overlap_rows),
        purged_size=len(purged_rows),
        eval_size=len(full_rows),
    )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.integrity", "scripts/m19/run_m19_integrity_suite.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/"),
            "train_data_path": str(args.train_data_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "audit_data_path": str(args.audit_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "audit_eval_size": int(args.audit_eval_size),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "max_latent_steps": int(args.max_latent_steps),
            "random_scale": float(args.random_scale),
            "seed": int(args.seed),
            "cell_id": str(args.cell_id),
        },
        "dataset_slices": {
            "full_count": len(full_rows),
            "overlap_count": len(overlap_rows),
            "purged_count": len(purged_rows),
            "masked_purged_count": len(masked_purged_rows),
            "full_eval_path": str(full_eval_path).replace("\\", "/"),
            "overlap_eval_path": str(overlap_eval_path).replace("\\", "/"),
            "purged_eval_path": str(purged_eval_path).replace("\\", "/"),
            "masked_eval_path": str(masked_eval_path).replace("\\", "/"),
        },
        "reports": {
            "benchmark_full": str(benchmark_full_path).replace("\\", "/"),
            "benchmark_purged": str(benchmark_purged_path).replace("\\", "/"),
            "benchmark_overlap": str(benchmark_overlap_path).replace("\\", "/"),
            "benchmark_masked_purged": str(benchmark_masked_path).replace("\\", "/"),
            "audit": str(audit_report_path).replace("\\", "/"),
        },
        "metrics": metrics,
        "headline": {
            "purged_accuracy": metrics.get("purged_accuracy"),
            "overlap_gap": metrics.get("overlap_gap"),
            "masked_accuracy": metrics.get("masked_accuracy"),
            "audit_qformer_accuracy": metrics.get("audit_qformer_accuracy"),
            "integrity_status": classify_integrity_status(metrics),
        },
        "notes": [
            "Purged and overlap slices are built from exact prompt-answer overlap against the current M19 training curriculum.",
            "Masked control is applied only to the purged slice so lexical blindfolding is not confounded by train-eval overlap.",
            "The suite reuses the standard M19 benchmark and audit runners so all regime metrics stay contract-comparable.",
        ],
    }

    report_path = Path(args.output_path) if args.output_path else (run_dir / "m19_integrity_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote integrity suite report: {report_path}")


def _run_benchmark_if_needed(
    *,
    output_path: Path,
    repo_root: Path,
    base_model: str,
    bridge_path: Path,
    eval_data_path: Path,
    eval_size: int,
    num_queries: int,
    bottleneck_dim: int,
    scratchpad_length: int,
    max_latent_steps: int,
    random_scale: float,
    seed: int,
    track: str,
    cell_id: str,
    regimes: str = "",
) -> None:
    if Path(output_path).exists():
        return
    _run_benchmark(
        output_path=output_path,
        repo_root=repo_root,
        base_model=base_model,
        bridge_path=bridge_path,
        eval_data_path=eval_data_path,
        eval_size=eval_size,
        num_queries=num_queries,
        bottleneck_dim=bottleneck_dim,
        scratchpad_length=scratchpad_length,
        max_latent_steps=max_latent_steps,
        random_scale=random_scale,
        seed=seed,
        track=track,
        cell_id=cell_id,
        regimes=regimes,
    )


def _run_benchmark(
    *,
    output_path: Path,
    repo_root: Path,
    base_model: str,
    bridge_path: Path,
    eval_data_path: Path,
    eval_size: int,
    num_queries: int,
    bottleneck_dim: int,
    scratchpad_length: int,
    max_latent_steps: int,
    random_scale: float,
    seed: int,
    track: str,
    cell_id: str,
    regimes: str = "",
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
        "--base-model",
        str(base_model),
        "--bridge-path",
        str(bridge_path),
        "--eval-data-path",
        str(eval_data_path),
        "--eval-size",
        str(int(eval_size)),
        "--num-queries",
        str(int(num_queries)),
        "--bottleneck-dim",
        str(int(bottleneck_dim)),
        "--scratchpad-length",
        str(int(scratchpad_length)),
        "--max-latent-steps",
        str(int(max_latent_steps)),
        "--random-scale",
        str(float(random_scale)),
        "--seed",
        str(int(seed)),
        "--track",
        str(track),
        "--cell-id",
        str(cell_id),
        "--output-path",
        str(output_path),
    ]
    if str(regimes).strip():
        cmd.extend(["--regimes", str(regimes).strip()])
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _run_audit_if_needed(
    *,
    output_path: Path,
    repo_root: Path,
    base_model: str,
    bridge_path: Path,
    audit_data_path: Path,
    audit_eval_size: int,
    scratchpad_length: int,
    num_queries: int,
    bottleneck_dim: int,
    max_latent_steps: int,
    random_scale: float,
    seed: int,
    track: str,
    cell_id: str,
) -> None:
    if Path(output_path).exists():
        return
    _run_audit(
        output_path=output_path,
        repo_root=repo_root,
        base_model=base_model,
        bridge_path=bridge_path,
        audit_data_path=audit_data_path,
        audit_eval_size=audit_eval_size,
        scratchpad_length=scratchpad_length,
        num_queries=num_queries,
        bottleneck_dim=bottleneck_dim,
        max_latent_steps=max_latent_steps,
        random_scale=random_scale,
        seed=seed,
        track=track,
        cell_id=cell_id,
    )


def _run_audit(
    *,
    output_path: Path,
    repo_root: Path,
    base_model: str,
    bridge_path: Path,
    audit_data_path: Path,
    audit_eval_size: int,
    scratchpad_length: int,
    num_queries: int,
    bottleneck_dim: int,
    max_latent_steps: int,
    random_scale: float,
    seed: int,
    track: str,
    cell_id: str,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_audit.py"),
        "--base-model",
        str(base_model),
        "--bridge-path",
        str(bridge_path),
        "--dataset-path",
        str(audit_data_path),
        "--eval-size",
        str(int(audit_eval_size)),
        "--scratchpad-length",
        str(int(scratchpad_length)),
        "--num-queries",
        str(int(num_queries)),
        "--bottleneck-dim",
        str(int(bottleneck_dim)),
        "--max-latent-steps",
        str(int(max_latent_steps)),
        "--random-scale",
        str(float(random_scale)),
        "--seed",
        str(int(seed)),
        "--track",
        str(track),
        "--cell-id",
        str(cell_id),
        "--output-path",
        str(output_path),
    ]
    subprocess.run(cmd, cwd=str(repo_root), check=True)


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
