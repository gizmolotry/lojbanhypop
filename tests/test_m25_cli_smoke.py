from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import load_script_module


REPO_ROOT = Path(__file__).resolve().parents[1]
runner = load_script_module("run_m25_emergent_bridi_suite", "scripts/m25/run_m25_emergent_bridi_suite.py")


def test_m25_suite_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/m25/run_m25_emergent_bridi_suite.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--max-symbols" in result.stdout
    assert "--symbol-budget" in result.stdout
    assert "--generator-epochs" in result.stdout
    assert "--advisor-epochs" in result.stdout


def test_m25_suite_cli_symbol_budget_default_is_disabled() -> None:
    args = runner.parse_args([])
    assert args.symbol_budget == 0


def test_m25_suite_cli_tiny_smoke_writes_report() -> None:
    run_id = "pytest_m25_cli_smoke"
    report = (
        REPO_ROOT
        / "artifacts/runs/telemetry/raw/ablation/hypercube/m25_emergent_bridi"
        / run_id
        / "m25_emergent_bridi_report.json"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/m25/run_m25_emergent_bridi_suite.py"),
            "--seed-list",
            "25",
            "--train-size",
            "18",
            "--eval-size",
            "12",
            "--generator-epochs",
            "1",
            "--advisor-epochs",
            "1",
            "--prompt-epochs",
            "1",
            "--batch-size",
            "6",
            "--embedding-dim",
            "8",
            "--hidden-dim",
            "16",
            "--advisor-hidden-dim",
            "16",
            "--max-symbols",
            "16",
            "--symbol-budget",
            "8",
            "--mdl-weight",
            "0.1",
            "--run-id",
            run_id,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "M25 emergent bridi report written" in result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["track"] == "M25"
    assert payload["config"]["max_symbols"] == 16
    assert payload["config"]["symbol_budget"] == 8
    assert payload["seed_reports"][0]["metrics"]["advisor_primary_trace_is_symbolic"] == 1.0
    assert "mean_strict_accuracy" in payload["aggregate_metrics"]
    assert "mean_loose_stream_exact_accuracy" in payload["aggregate_metrics"]
    assert "mean_token_reduction_ratio" in payload["aggregate_metrics"]
