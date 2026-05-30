from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import load_script_module


REPO_ROOT = Path(__file__).resolve().parents[1]
runner = load_script_module("run_m26_end_to_end_loafman_suite", "scripts/m26/run_m26_end_to_end_loafman_suite.py")


def test_m26_suite_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/m26/run_m26_end_to_end_loafman_suite.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--epochs" in result.stdout
    assert "--symbol-budget" in result.stdout
    assert "--answer-weight" in result.stdout


def test_m26_suite_cli_defaults_are_spinal_cord_defaults() -> None:
    args = runner.parse_args([])
    assert args.symbol_budget == 0
    assert args.answer_weight == 1.0


def test_m26_suite_cli_tiny_smoke_writes_report() -> None:
    run_id = "pytest_m26_cli_smoke"
    report = (
        REPO_ROOT
        / "artifacts/runs/telemetry/raw/ablation/hypercube/m26_end_to_end_loafman"
        / run_id
        / "m26_end_to_end_loafman_report.json"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/m26/run_m26_end_to_end_loafman_suite.py"),
            "--seed-list",
            "26",
            "--train-size",
            "18",
            "--eval-size",
            "12",
            "--epochs",
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
    assert "M26 end-to-end Loafman report written" in result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["track"] == "M26"
    assert payload["config"]["max_symbols"] == 16
    assert payload["config"]["symbol_budget"] == 8
    assert "single_optimizer_generator_and_advisor" in payload["architecture_locks"]
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_generator"] == 1.0
    assert "mean_answer_loss_reaches_generator" in payload["aggregate_metrics"]
    assert "mean_m26_spinal_cord_gate_pass_rate" in payload["aggregate_metrics"]
