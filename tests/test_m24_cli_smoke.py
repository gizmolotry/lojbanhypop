from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_m24_suite_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/m24/run_m24_substrate_compression_suite.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--generator-epochs" in result.stdout
    assert "--advisor-epochs" in result.stdout
    assert "--trace-weight" in result.stdout


def test_m24_suite_cli_tiny_smoke_writes_report() -> None:
    run_id = "pytest_m24_cli_smoke"
    report = REPO_ROOT / "artifacts/runs/telemetry/raw/ablation/hypercube/m24_substrate_compression" / run_id / "m24_substrate_compression_report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/m24/run_m24_substrate_compression_suite.py"),
            "--seed-list",
            "24",
            "--train-size",
            "18",
            "--eval-size",
            "18",
            "--generator-epochs",
            "1",
            "--advisor-epochs",
            "1",
            "--prompt-epochs",
            "1",
            "--batch-size",
            "18",
            "--embedding-dim",
            "8",
            "--hidden-dim",
            "16",
            "--advisor-hidden-dim",
            "16",
            "--run-id",
            run_id,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "M24 substrate compression report written" in result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["aggregate_metrics"]["mean_strict_accuracy"] >= 0.0
    assert "mean_substrate_claim_score" in payload["aggregate_metrics"]
    assert payload["seed_reports"][0]["lock_status"]["symbolic_trace_only"] is True
