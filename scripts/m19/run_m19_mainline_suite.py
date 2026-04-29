from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys as _sys

_sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
_sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lojban_evolution.m19.family import M19_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, validate_series_outputs


def _run(cmd: list[str], cwd: Path) -> None:
    print(f"Executing: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train, audit, and benchmark the M19 mainline suite with stable artifact outputs.")
    parser.add_argument("--base-model", default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-path", default="artifacts/datasets/m19_mixed_curriculum_v1.jsonl")
    parser.add_argument("--eval-data-path", default="artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl")
    parser.add_argument("--audit-data-path", default="artifacts/datasets/sanity_check_v1.jsonl")
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--min-latent-steps", type=int, default=4)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--include-random-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynamic-pacing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--track", type=str, default="M19")
    parser.add_argument("--static-bridge-path", type=str, default="artifacts/models/m19/grid/m19_isolation_grid_20260409_v3/M19.3_8Q_128D_8S.pt")
    parser.add_argument("--static-cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument("--static-num-queries", type=int, default=8)
    parser.add_argument("--static-bottleneck-dim", type=int, default=128)
    parser.add_argument("--static-scratchpad-length", type=int, default=8)
    parser.add_argument("--output-root", type=Path, default=Path(M19_REGISTRY["M19"]["output_roots"]["mainline"]))
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/models/m19/mainline_suite"))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    track_key = str(args.track if str(args.track) in M19_REGISTRY else "M19")
    registry = M19_REGISTRY[track_key]
    output_root = Path(args.output_root)
    if track_key != "M19" and output_root == Path(M19_REGISTRY["M19"]["output_roots"]["mainline"]):
        output_root = Path(registry["output_roots"]["mainline"])
    assert_output_path_allowed("M", output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = args.model_output_root / run_id / "m19_mainline_bridge.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    train_report_path = run_dir / "train_report.json"
    audit_report_path = run_dir / "audit_report.json"
    benchmark_report_path = run_dir / "benchmark_report.json"

    _run(
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
            "--min-latent-steps",
            str(int(args.min_latent_steps)),
            "--max-latent-steps",
            str(int(args.max_latent_steps)),
            "--epochs",
            str(int(args.epochs)),
            "--seed",
            str(int(args.seed)),
            "--track",
            track_key,
            "--cell-id",
            "M19.4" if bool(args.dynamic_pacing or track_key == "M19.4") else "MAINLINE",
            "--checkpoint-output-path",
            str(checkpoint_path),
            "--report-output-path",
            str(train_report_path),
        ],
        repo_root,
    )

    if not bool(args.dynamic_pacing or track_key == "M19.4"):
        _run(
            [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "run_m19_audit.py"),
                "--base-model",
                str(args.base_model),
                "--bridge-path",
                str(checkpoint_path),
                "--data-path",
                str(args.audit_data_path),
                "--scratchpad-length",
                str(int(args.scratchpad_length)),
                "--num-queries",
                str(int(args.num_queries)),
                "--bottleneck-dim",
                str(int(args.bottleneck_dim)),
                "--random-scale",
                str(float(args.random_scale)),
                "--track",
                track_key,
                "--cell-id",
                "MAINLINE",
                "--output-path",
                str(audit_report_path),
            ],
            repo_root,
        )

    benchmark_cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
        "--base-model",
        str(args.base_model),
        "--bridge-path",
        str(checkpoint_path),
        "--eval-data-path",
        str(args.eval_data_path),
        "--eval-size",
        str(int(args.eval_size)),
        "--scratchpad-length",
        str(int(args.scratchpad_length)),
        "--min-latent-steps",
        str(int(args.min_latent_steps)),
        "--max-latent-steps",
        str(int(args.max_latent_steps)),
        "--num-queries",
        str(int(args.num_queries)),
        "--bottleneck-dim",
        str(int(args.bottleneck_dim)),
        "--cell-id",
        "M19.4" if bool(args.dynamic_pacing or track_key == "M19.4") else "MAINLINE",
        "--static-cell-id",
        str(args.static_cell_id),
        "--static-num-queries",
        str(int(args.static_num_queries)),
        "--static-bottleneck-dim",
        str(int(args.static_bottleneck_dim)),
        "--static-scratchpad-length",
        str(int(args.static_scratchpad_length)),
        "--track",
        track_key,
        "--random-scale",
        str(float(args.random_scale)),
        "--output-path",
        str(benchmark_report_path),
    ]
    if bool(args.dynamic_pacing or track_key == "M19.4"):
        benchmark_cmd.append("--dynamic-pacing")
    if str(args.static_bridge_path).strip():
        benchmark_cmd.extend(["--static-bridge-path", str(args.static_bridge_path)])
    benchmark_cmd.append("--include-random-control" if bool(args.include_random_control) else "--no-include-random-control")
    _run(benchmark_cmd, repo_root)

    train_report = json.loads(train_report_path.read_text(encoding="utf-8"))
    audit_report = json.loads(audit_report_path.read_text(encoding="utf-8")) if audit_report_path.exists() else {}
    benchmark_report = json.loads(benchmark_report_path.read_text(encoding="utf-8"))

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": track_key,
        "run_id": run_id,
        "series": {
            "series_id": "M",
            "track": track_key,
            "script": "scripts/m19/run_m19_mainline_suite.py",
        },
        "family_contract": {
            "family_name": registry["family"],
            "implementation_label": registry["implementation_label"],
            "runner_script": "scripts/m19/run_m19_mainline_suite.py",
            "dag": registry["dags"]["mainline"],
        },
        "config": {
            "base_model": str(args.base_model),
            "data_path": str(args.data_path),
            "eval_data_path": str(args.eval_data_path),
            "audit_data_path": str(args.audit_data_path),
            "eval_size": int(args.eval_size),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "track": track_key,
            "dynamic_pacing": bool(args.dynamic_pacing or track_key == "M19.4"),
            "num_queries": int(args.num_queries),
            "bottleneck_dim": int(args.bottleneck_dim),
            "scratchpad_length": int(args.scratchpad_length),
            "min_latent_steps": int(args.min_latent_steps),
            "max_latent_steps": int(args.max_latent_steps),
            "random_scale": float(args.random_scale),
            "include_random_control": bool(args.include_random_control),
            "static_bridge_path": str(args.static_bridge_path) or None,
            "static_cell_id": str(args.static_cell_id),
        },
        "checkpoint_output_path": train_report.get("checkpoint_output_path"),
        "train": {
            "final_mean_loss": train_report.get("final_mean_loss"),
            "epoch_mean_losses": train_report.get("epoch_mean_losses", []),
            "dataset_size": train_report.get("dataset_size"),
        },
        "metrics": {
            **benchmark_report.get("metrics", {}),
            "audit_qformer_accuracy": audit_report.get("headline", {}).get("qformer_accuracy"),
            "audit_random_accuracy": audit_report.get("headline", {}).get("random_accuracy"),
            "audit_lift_vs_base": audit_report.get("headline", {}).get("lift_vs_base"),
            "audit_lift_vs_random": audit_report.get("headline", {}).get("lift_vs_random"),
        },
        "report_paths": {
            "train_report": str(train_report_path).replace("\\", "/"),
            "audit_report": str(audit_report_path).replace("\\", "/") if audit_report_path.exists() else None,
            "benchmark_report": str(benchmark_report_path).replace("\\", "/"),
        },
    }
    manifest_path = run_dir / registry["report_names"]["mainline"]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"M19 mainline suite complete: {run_dir}")


if __name__ == "__main__":
    main()
