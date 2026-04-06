from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.m_bridge_ablation_family import BRIDGE_ABLATION_REGISTRY, BRIDGE_ABLATION_FAMILY_VERSION, BRIDGE_NORMALIZED_METRICS
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)


def _wait_for_port(host: str, port: int, timeout_s: float) -> None:
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _run_logged(cmd: list[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8") as so, stderr_path.open("w", encoding="utf-8") as se:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=so, stderr=se, text=True)
    return int(proc.returncode)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the M11 discriminative bridge suite with unified-ledger output and Airflow-friendly forge orchestration.")
    p.add_argument("--base-model", default="archive/results/m9/active/RESULTS_M9_SYNCED/synced_model")
    p.add_argument("--forge-ckpt", type=Path, default=Path("archive/results/m9/active/RESULTS_M9_PHASE3/m11_s1_final.pt"))
    p.add_argument("--adapter-ckpt", type=Path, default=Path("archive/results/m10/active/RESULTS_M10_DEEP_TRANSLATOR/m11_native_adapter.pt"))
    p.add_argument("--head-ckpt", type=Path, default=Path("archive/results/m10/active/RESULTS_M10_ENGLISH_HEAD/m11_native_head.pt"))
    p.add_argument("--train-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--disable-adapter", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--forge-startup-timeout", type=float, default=45.0)
    p.add_argument("--baseline-manifest", type=Path, default=Path("docs/baselines/m_series_bridge_baseline_manifest.json"))
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m11_discriminative_suite"))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    assert_output_path_allowed("M", args.output_root)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    train_mode = (not bool(args.skip_train)) and int(args.train_steps) > 0
    python_bin = sys.executable

    manifest: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M11.discriminative_suite", "scripts/run_m11_discriminative_suite.py"),
        "track": "M11.discriminative_suite",
        "lineage": lineage_metadata(
            "train" if train_mode else "eval_only",
            checkpoint_in=str(args.forge_ckpt).replace("\\", "/"),
            checkpoint_out=str(args.adapter_ckpt).replace("\\", "/"),
            dataset_profile="diverse_v3",
            difficulty_tier="all",
        ),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "config": {
            "base_model": str(args.base_model),
            "forge_ckpt": str(args.forge_ckpt).replace("\\", "/"),
            "adapter_ckpt": str(args.adapter_ckpt).replace("\\", "/"),
            "head_ckpt": str(args.head_ckpt).replace("\\", "/"),
            "train_steps": int(args.train_steps),
            "lr": float(args.lr),
            "num_samples": int(args.num_samples),
            "disable_adapter": bool(args.disable_adapter),
            "skip_train": bool(args.skip_train),
            "port": int(args.port),
            "local_files_only": bool(args.local_files_only),
        },
        "ablation_contract": {
            "family_version": BRIDGE_ABLATION_FAMILY_VERSION,
            "family_name": "m_bridge_ablations",
            "runner_script": "scripts/run_m11_discriminative_suite.py",
            "dag": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["dag"],
            "implementation_label": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["implementation_label"],
            "tensor_flow": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["tensor_flow"],
            "lora_positioning": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["lora_positioning"],
            "parameter_axes": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["parameter_axes"],
            "loss_profile": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["loss_profile"],
            "variant_cells": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["cells"],
            "normalized_metric_aliases": BRIDGE_NORMALIZED_METRICS,
            "checkpoint_in": str(args.forge_ckpt).replace("\\", "/"),
        },
        "steps": {},
    }

    if train_mode:
        train_cmd = [
            python_bin,
            "scripts/m10/train_m11_discriminative_bridge.py",
            "--base-model",
            str(args.base_model),
            "--forge-ckpt",
            str(args.forge_ckpt),
            "--train-steps",
            str(int(args.train_steps)),
            "--lr",
            str(float(args.lr)),
        ]
        if bool(args.local_files_only):
            train_cmd.append("--local-files-only")
        train_stdout = run_dir / "train_m11_discriminative_bridge.stdout.log"
        train_stderr = run_dir / "train_m11_discriminative_bridge.stderr.log"
        rc = _run_logged(train_cmd, repo_root, train_stdout, train_stderr)
        manifest["steps"]["train_m11_discriminative_bridge"] = {
            "command": train_cmd,
            "return_code": int(rc),
            "stdout": str(train_stdout).replace("\\", "/"),
            "stderr": str(train_stderr).replace("\\", "/"),
        }
        if rc != 0:
            raise RuntimeError(f"M11 discriminative bridge training failed with return code {rc}")

    if not args.adapter_ckpt.exists():
        raise FileNotFoundError(f"adapter checkpoint not found: {args.adapter_ckpt}")
    if not args.head_ckpt.exists():
        raise FileNotFoundError(f"head checkpoint not found: {args.head_ckpt}")

    forge_stdout = run_dir / "phase3_forge.stdout.log"
    forge_stderr = run_dir / "phase3_forge.stderr.log"
    audit_stdout = run_dir / "final_audit.stdout.log"
    audit_stderr = run_dir / "final_audit.stderr.log"

    forge_cmd = [
        python_bin,
        "-u",
        "scripts/m9/phase3_forge.py",
        "--port",
        str(int(args.port)),
        "--load-ckpt",
        str(args.forge_ckpt),
    ]

    with forge_stdout.open("w", encoding="utf-8") as fo, forge_stderr.open("w", encoding="utf-8") as fe:
        forge_proc = subprocess.Popen(forge_cmd, cwd=str(repo_root), stdout=fo, stderr=fe, text=True)
        try:
            _wait_for_port("127.0.0.1", int(args.port), float(args.forge_startup_timeout))

            audit_cmd = [
                python_bin,
                "scripts/m10/final_audit.py",
                "--base-model",
                str(args.base_model),
                "--forge-ckpt",
                str(args.forge_ckpt),
                "--adapter-ckpt",
                str(args.adapter_ckpt),
                "--head-ckpt",
                str(args.head_ckpt),
                "--num-samples",
                str(int(args.num_samples)),
                "--port",
                str(int(args.port)),
            ]
            if bool(args.disable_adapter):
                audit_cmd.append("--disable-adapter")

            rc = _run_logged(audit_cmd, repo_root, audit_stdout, audit_stderr)
            manifest["steps"]["final_audit"] = {
                "command": audit_cmd,
                "return_code": int(rc),
                "stdout": str(audit_stdout).replace("\\", "/"),
                "stderr": str(audit_stderr).replace("\\", "/"),
            }
            if rc != 0:
                raise RuntimeError(f"M11 final audit failed with return code {rc}")
        finally:
            forge_proc.terminate()
            try:
                forge_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                forge_proc.kill()
                forge_proc.wait(timeout=10)

    source_report = repo_root / "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT" / ("final_floor_lock.json" if bool(args.disable_adapter) else "final_bridge_audit.json")
    if not source_report.exists():
        raise FileNotFoundError(f"Expected audit report not found: {source_report}")

    copied_report = run_dir / source_report.name
    shutil.copy2(source_report, copied_report)
    report_payload = json.loads(source_report.read_text(encoding="utf-8"))

    publication_metrics = repo_root / "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT" / "final_publication_metrics.json"
    if publication_metrics.exists():
        shutil.copy2(publication_metrics, run_dir / publication_metrics.name)

    manifest["artifacts"] = {
        "audit_report": str(copied_report).replace("\\", "/"),
        "source_report": str(source_report).replace("\\", "/"),
        "adapter_ckpt": str(args.adapter_ckpt).replace("\\", "/"),
        "head_ckpt": str(args.head_ckpt).replace("\\", "/"),
    }
    manifest["metrics"] = {
        "accuracy": report_payload.get("accuracy"),
        "macro_f1": report_payload.get("macro_f1"),
        "num_samples": report_payload.get("num_samples"),
    }

    manifest_path = run_dir / "m11_discriminative_suite_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    md_lines = [
        "# M11 Discriminative Suite",
        "",
        f"- accuracy: `{report_payload.get('accuracy')}`",
        f"- macro_f1: `{report_payload.get('macro_f1')}`",
        f"- num_samples: `{report_payload.get('num_samples')}`",
        f"- disable_adapter: `{bool(args.disable_adapter)}`",
        f"- adapter_ckpt: `{str(args.adapter_ckpt).replace('\\', '/')}`",
        f"- head_ckpt: `{str(args.head_ckpt).replace('\\', '/')}`",
        f"- train_executed: `{train_mode}`",
        "",
        f"- manifest: `{str(manifest_path).replace('\\', '/')}`",
        f"- report: `{str(copied_report).replace('\\', '/')}`",
    ]
    (run_dir / "m11_discriminative_suite_summary.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"M11 discriminative suite complete: {run_dir}")


if __name__ == "__main__":
    main()


