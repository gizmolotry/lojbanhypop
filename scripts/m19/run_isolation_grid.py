from __future__ import annotations

import argparse
import hashlib
import json
import shutil
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _desired_train_contract(args: argparse.Namespace, cell_cfg: dict[str, Any], seed: int) -> dict[str, Any]:
    return {
        "cell_id": str(cell_cfg["cell_id"]),
        "num_queries": int(cell_cfg["num_queries"]),
        "bottleneck_dim": int(cell_cfg["bottleneck_dim"]),
        "scratchpad_length": int(cell_cfg["scratchpad_length"]),
        "epochs": int(args.epochs),
        "data_path": str(args.data_path).replace("\\", "/"),
        "base_model": str(args.base_model),
        "seed": int(seed),
    }


def _contract_hash(contract: dict[str, Any]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _find_archival_checkpoint(repo_root: Path, cell_label: str) -> Path | None:
    archival_root = repo_root / "artifacts" / "models" / "m19" / "grid"
    direct = archival_root / f"{cell_label}.pt"
    if direct.exists():
        return direct
    matches = sorted(archival_root.rglob(f"{cell_label}.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _find_matching_train_report(repo_root: Path, checkpoint_path: Path, desired_contract: dict[str, Any]) -> Path | None:
    telemetry_root = repo_root / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_isolation_grid"
    if not telemetry_root.exists():
        return None
    desired_hash = _contract_hash(desired_contract)
    candidates = sorted(telemetry_root.rglob("train_report.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    checkpoint_norm = str(checkpoint_path.resolve()).replace("\\", "/").lower()
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
        contract = {
            "cell_id": str(config.get("cell_id") or payload.get("cell_id") or ""),
            "num_queries": int(config.get("num_queries") or -1),
            "bottleneck_dim": int(config.get("bottleneck_dim") or -1),
            "scratchpad_length": int(config.get("scratchpad_length") or -1),
            "epochs": int(config.get("epochs") or -1),
            "data_path": str(config.get("data_path") or "").replace("\\", "/"),
            "base_model": str(config.get("base_model") or ""),
            "seed": int(config.get("seed") or -1),
        }
        checkpoint_out = str(payload.get("checkpoint_output_path") or payload.get("checkpoint_path") or "").replace("\\", "/")
        source_ckpt = str(payload.get("source_checkpoint") or "").replace("\\", "/")
        if checkpoint_out:
            try:
                checkpoint_out = str(Path(checkpoint_out).resolve()).replace("\\", "/").lower()
            except Exception:
                checkpoint_out = checkpoint_out.lower()
        if source_ckpt:
            try:
                source_ckpt = str(Path(source_ckpt).resolve()).replace("\\", "/").lower()
            except Exception:
                source_ckpt = source_ckpt.lower()
        if checkpoint_norm not in {checkpoint_out, source_ckpt}:
            continue
        if _contract_hash(contract) == desired_hash:
            return candidate
    return None


def _cell_specs(include_replications: bool, replication_seeds: list[int]) -> list[tuple[str, dict[str, Any]]]:
    spec = M19_REGISTRY["M19"]
    rows: list[tuple[str, dict[str, Any]]] = [(str(row["cell_key"]), dict(row)) for row in spec["default_grid"]]
    if not include_replications:
        return rows

    default_cells = {str(row["cell_key"]): dict(row) for row in spec["default_grid"]}
    for rep_key, rep_spec in spec["replication_cells"].items():
        base = default_cells[str(rep_spec["base_cell"])]
        for index, seed in enumerate(replication_seeds, start=1):
            cell_key = f"{rep_key}{index}"
            rows.append(
                (
                    cell_key,
                    {
                        **base,
                        "label": f"{rep_spec['label']}_seed{seed}",
                        "role": str(rep_spec["role"]),
                        "base_cell": str(rep_spec["base_cell"]),
                        "seed": int(seed),
                    },
                )
            )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the M19.3 isolation grid with explicit per-cell artifacts and structured reports.")
    parser.add_argument("--base-model", default="C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--data-path", default="artifacts/datasets/m19_mixed_curriculum_v1.jsonl")
    parser.add_argument("--eval-data-path", default="artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl")
    parser.add_argument("--audit-data-path", default="artifacts/datasets/sanity_check_v1.jsonl")
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--include-random-control", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-replications", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replication-seeds", type=str, default="43,44")
    parser.add_argument("--replication-runs", type=int, default=0)
    parser.add_argument("--use-existing-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--force-retrain", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume-existing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--only-cells", type=str, default="")
    parser.add_argument("--output-root", type=Path, default=Path(M19_REGISTRY["M19"]["output_roots"]["grid"]))
    parser.add_argument("--train-output-root", type=Path, default=None)
    parser.add_argument("--benchmark-output-root", type=Path, default=None)
    parser.add_argument("--model-output-root", type=Path, default=Path("artifacts/models/m19/grid"))
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    assert_output_path_allowed("M", args.output_root)
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    replication_seeds = [int(part.strip()) for part in str(args.replication_seeds).split(",") if part.strip()]
    if int(args.replication_runs) > 0:
        replication_seeds = [int(args.seed) + 1 + i for i in range(int(args.replication_runs))]
    model_output_root = Path(args.model_output_root or args.train_output_root or "artifacts/models/m19/grid")
    cells = _cell_specs(bool(args.include_replications), replication_seeds)
    if str(args.only_cells).strip():
        wanted = {part.strip() for part in str(args.only_cells).split(",") if part.strip()}
        cells = [(cell_key, cell_cfg) for cell_key, cell_cfg in cells if cell_key in wanted]

    summary: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "track": "M19.3",
        "run_id": run_id,
        "series": {
            "series_id": "M",
            "track": "M19.3",
            "script": "scripts/m19/run_isolation_grid.py",
        },
        "family_contract": {
            "family_name": M19_REGISTRY["M19"]["family"],
            "implementation_label": "runway_capacity_isolation_grid",
            "runner_script": "scripts/m19/run_isolation_grid.py",
            "dag": M19_REGISTRY["M19"]["dags"]["grid"],
        },
        "config": {
            "base_model": str(args.base_model),
            "data_path": str(args.data_path),
            "eval_data_path": str(args.eval_data_path),
            "audit_data_path": str(args.audit_data_path),
            "eval_size": int(args.eval_size),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "random_scale": float(args.random_scale),
            "include_random_control": bool(args.include_random_control),
            "include_replications": bool(args.include_replications),
            "replication_seeds": replication_seeds,
        },
        "cells": {},
    }

    for offset, (cell_key, cell_cfg) in enumerate(cells):
        cell_seed = int(cell_cfg.get("seed", int(args.seed) + offset))
        cell_label = str(cell_cfg["cell_id"])
        cell_dir = run_dir / cell_key
        cell_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = model_output_root / run_id / f"{cell_label}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        train_report_path = cell_dir / "train_report.json"
        benchmark_report_path = cell_dir / "benchmark_report.json"
        audit_report_path = cell_dir / "audit_report.json"
        existing_cell_summary_path = cell_dir / "cell_summary.json"

        if bool(args.resume_existing) and existing_cell_summary_path.exists():
            existing_cell_summary = _read_json(existing_cell_summary_path)
            if (
                train_report_path.exists()
                and benchmark_report_path.exists()
                and audit_report_path.exists()
                and existing_cell_summary.get("variant_spec", {}).get("cell_id") == cell_label
            ):
                summary["cells"][cell_key] = existing_cell_summary
                _write_json(run_dir / "m19_isolation_grid_report.partial.json", summary)
                continue

        reused = False
        archival_ckpt = _find_archival_checkpoint(repo_root, cell_label)
        desired_contract = _desired_train_contract(args, cell_cfg, cell_seed)
        source_train_report = (
            _find_matching_train_report(repo_root, archival_ckpt, desired_contract)
            if archival_ckpt is not None and archival_ckpt.exists()
            else None
        )
        if (
            bool(args.use_existing_checkpoints)
            and not bool(args.force_retrain)
            and not str(cell_cfg.get("base_cell", "")).strip()
            and archival_ckpt is not None
            and archival_ckpt.exists()
            and source_train_report is not None
        ):
            shutil.copy2(archival_ckpt, checkpoint_path)
            train_report_path.write_text(
                json.dumps(
                    {
                        "track": "M19.3",
                        "cell_id": cell_label,
                        "checkpoint_output_path": str(checkpoint_path).replace("\\", "/"),
                        "source_checkpoint": str(archival_ckpt).replace("\\", "/"),
                        "source_train_report": str(source_train_report).replace("\\", "/"),
                        "reuse_contract_hash": _contract_hash(desired_contract),
                        "reuse_validated": True,
                        "epoch_mean_losses": [],
                        "final_mean_loss": None,
                        "note": "reused pre-existing checkpoint with matching train manifest contract",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            reused = True

        if not reused:
            if not (bool(args.resume_existing) and train_report_path.exists() and checkpoint_path.exists()):
                train_cmd = [
                    sys.executable,
                    str(repo_root / "scripts" / "m19" / "train_m19_mainline.py"),
                    "--base-model",
                    str(args.base_model),
                    "--data-path",
                    str(args.data_path),
                    "--num-queries",
                    str(int(cell_cfg["num_queries"])),
                    "--bottleneck-dim",
                    str(int(cell_cfg["bottleneck_dim"])),
                    "--scratchpad-length",
                    str(int(cell_cfg["scratchpad_length"])),
                    "--epochs",
                    str(int(args.epochs)),
                    "--learning-rate",
                    str(float(args.learning_rate)),
                    "--seed",
                    str(cell_seed),
                    "--track",
                    "M19.3",
                    "--cell-id",
                    cell_label,
                    "--checkpoint-output-path",
                    str(checkpoint_path),
                    "--report-output-path",
                    str(train_report_path),
                ]
                if bool(args.local_files_only):
                    train_cmd.append("--local-files-only")
                _run(train_cmd, repo_root)

        if not (bool(args.resume_existing) and benchmark_report_path.exists()):
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
                str(int(cell_cfg["scratchpad_length"])),
                "--num-queries",
                str(int(cell_cfg["num_queries"])),
                "--bottleneck-dim",
                str(int(cell_cfg["bottleneck_dim"])),
                "--cell-id",
                cell_label,
                "--track",
                "M19.3",
                "--random-scale",
                str(float(args.random_scale)),
                "--output-path",
                str(benchmark_report_path),
            ]
            benchmark_cmd.append("--include-random-control" if bool(args.include_random_control) else "--no-include-random-control")
            if bool(args.local_files_only):
                benchmark_cmd.append("--local-files-only")
            _run(benchmark_cmd, repo_root)

        if not (bool(args.resume_existing) and audit_report_path.exists()):
            audit_cmd = [
                sys.executable,
                str(repo_root / "scripts" / "m19" / "run_m19_audit.py"),
                "--base-model",
                str(args.base_model),
                "--bridge-path",
                str(checkpoint_path),
                "--data-path",
                str(args.audit_data_path),
                "--scratchpad-length",
                str(int(cell_cfg["scratchpad_length"])),
                "--num-queries",
                str(int(cell_cfg["num_queries"])),
                "--bottleneck-dim",
                str(int(cell_cfg["bottleneck_dim"])),
                "--random-scale",
                str(float(args.random_scale)),
                "--track",
                "M19.3",
                "--cell-id",
                cell_label,
                "--output-path",
                str(audit_report_path),
            ]
            if bool(args.local_files_only):
                audit_cmd.append("--local-files-only")
            _run(audit_cmd, repo_root)

        train_report = _read_json(train_report_path)
        benchmark_report = _read_json(benchmark_report_path)
        audit_report = _read_json(audit_report_path)

        metrics = dict(benchmark_report.get("metrics", {}))
        metrics["audit_qformer_accuracy"] = audit_report.get("headline", {}).get("qformer_accuracy")
        metrics["audit_random_accuracy"] = audit_report.get("headline", {}).get("random_accuracy")
        metrics["audit_lift_vs_random"] = audit_report.get("headline", {}).get("lift_vs_random")

        cell_summary = {
            "label": str(cell_cfg["label"]),
            "variant_spec": {
                "cell_id": cell_label,
                "num_queries": int(cell_cfg["num_queries"]),
                "bottleneck_dim": int(cell_cfg["bottleneck_dim"]),
                "scratchpad_length": int(cell_cfg["scratchpad_length"]),
                "role": str(cell_cfg.get("role", "")),
                "base_cell": str(cell_cfg.get("base_cell", "")) or None,
                "seed": int(cell_seed),
            },
            "reused_checkpoint": bool(reused),
            "reuse_validated": bool(reused),
            "train": {
                "checkpoint_output_path": train_report.get("checkpoint_output_path"),
                "final_mean_loss": train_report.get("final_mean_loss"),
                "epoch_mean_losses": train_report.get("epoch_mean_losses", []),
                "source_train_report": train_report.get("source_train_report"),
            },
            "metrics": metrics,
            "audit_metrics": audit_report.get("headline", {}),
            "report_paths": {
                "train_report": str(train_report_path).replace("\\", "/"),
                "benchmark_report": str(benchmark_report_path).replace("\\", "/"),
                "audit_report": str(audit_report_path).replace("\\", "/"),
            },
        }
        summary["cells"][cell_key] = cell_summary
        _write_json(cell_dir / "cell_summary.json", cell_summary)
        _write_json(run_dir / "m19_isolation_grid_report.partial.json", summary)

    leaderboard = sorted(
        (
            {
                "cell": cell_key,
                "label": payload["label"],
                "overall_accuracy": float(payload["metrics"].get("overall_accuracy", 0.0)),
                "lift_vs_en_cot": payload["metrics"].get("lift_vs_en_cot"),
                "lift_vs_zh_cot": payload["metrics"].get("lift_vs_zh_cot"),
                "lift_vs_random": payload["metrics"].get("lift_vs_random"),
                "avg_tokens": payload["metrics"].get("avg_tokens"),
            }
            for cell_key, payload in summary["cells"].items()
        ),
        key=lambda row: (float(row["overall_accuracy"]), float(row.get("lift_vs_en_cot") or -999.0)),
        reverse=True,
    )
    summary["leaderboard"] = leaderboard
    summary["best_cell"] = leaderboard[0]["cell"] if leaderboard else None
    summary["best_label"] = leaderboard[0]["label"] if leaderboard else None

    summary_path = run_dir / "m19_isolation_grid_report.json"
    _write_json(summary_path, summary)

    md_lines = [
        "# M19.3 Isolation Grid",
        "",
        f"- run_id: `{run_id}`",
        f"- summary: `{str(summary_path).replace('\\', '/')}`",
        "",
        "| Cell | Label | Acc | vs EN-CoT | vs ZH-CoT | vs Random | Audit QF | Loss | Avg Tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in leaderboard:
        payload = summary["cells"][row["cell"]]
        md_lines.append(
            f"| {row['cell']} | {row['label']} | {float(row['overall_accuracy']):.3f} | "
            f"{float(row.get('lift_vs_en_cot') or 0.0):.3f} | {float(row.get('lift_vs_zh_cot') or 0.0):.3f} | "
            f"{float(row.get('lift_vs_random') or 0.0):.3f} | "
            f"{float(payload['audit_metrics'].get('qformer_accuracy') or 0.0):.3f} | "
            f"{float(payload['train'].get('final_mean_loss') or 0.0):.4f} | {float(row.get('avg_tokens') or 0.0):.2f} |"
        )
    (run_dir / "m19_isolation_grid_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"M19 isolation grid complete: {run_dir}")


if __name__ == "__main__":
    main()
