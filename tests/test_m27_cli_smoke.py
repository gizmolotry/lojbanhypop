from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from conftest import load_script_module


REPO_ROOT = Path(__file__).resolve().parents[1]
runner = load_script_module("run_m27_coconut_bridi_runtime_suite", "scripts/m27/run_m27_coconut_bridi_runtime_suite.py")


def test_m27_suite_cli_help() -> None:
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts/m27/run_m27_coconut_bridi_runtime_suite.py"), "--help"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--max-steps" in result.stdout
    assert "--matched-prompt-budget" in result.stdout
    assert "--answer-weight" in result.stdout
    assert "--enable-relevance-runtime" in result.stdout


def test_m27_suite_cli_defaults_are_runtime_defaults() -> None:
    args = runner.parse_args([])
    assert args.max_prompt_length == 128
    assert args.language_layers == 1
    assert args.language_heads == 2
    assert args.symbol_budget == 0
    assert args.matched_prompt_budget == 0
    assert args.answer_weight == 1.0
    assert args.enable_relevance_runtime is False


def test_m27_suite_cli_tiny_smoke_writes_report() -> None:
    run_id = "pytest_m27_cli_smoke"
    report = (
        REPO_ROOT
        / "artifacts/runs/telemetry/raw/ablation/hypercube/m27_coconut_bridi_runtime"
        / run_id
        / "m27_coconut_bridi_runtime_report.json"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/m27/run_m27_coconut_bridi_runtime_suite.py"),
            "--seed-list",
            "27",
            "--train-size",
            "18",
            "--eval-size",
            "8",
            "--epochs",
            "1",
            "--prompt-epochs",
            "1",
            "--batch-size",
            "4",
            "--embedding-dim",
            "8",
            "--hidden-dim",
            "16",
            "--advisor-hidden-dim",
            "16",
            "--max-symbols",
            "8",
            "--max-steps",
            "8",
            "--symbol-budget",
            "8",
            "--matched-prompt-budget",
            "8",
            "--mdl-weight",
            "0.1",
            "--enable-relevance-runtime",
            "--relevance-rank-weight",
            "0.25",
            "--use-relevance-answer",
            "--run-id",
            run_id,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "M27 Coconut-Bridi runtime report written" in result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["track"] == "M27"
    assert payload["config"]["max_symbols"] == 8
    assert payload["config"]["max_steps"] == 8
    assert payload["seed_reports"][0]["config"]["organism_mode"] == "coconut_autoregressive_lm_hidden_bridi_bridge"
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_coconut_cell"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_recurrent_bridi_feedback"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["m27_relevance_runtime_enabled"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["m27_inherited_contract_bundle_present"] == 1.0
    assert "mean_answer_loss_reaches_coconut_cell" in payload["aggregate_metrics"]
    assert "mean_m27_full_organism_gate_pass_rate" in payload["aggregate_metrics"]
    assert "mean_m27_relevance_runtime_enabled" in payload["aggregate_metrics"]
    assert "mean_m27_inherited_contract_bundle_present" in payload["aggregate_metrics"]
