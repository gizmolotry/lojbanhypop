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
    assert "--prompt-epochs" in result.stdout
    assert "--max-prompt-length" in result.stdout
    assert "--language-layers" in result.stdout
    assert "--language-heads" in result.stdout
    assert "--symbol-budget" in result.stdout
    assert "--matched-prompt-budget" in result.stdout
    assert "--answer-weight" in result.stdout


def test_m26_suite_cli_defaults_are_full_organism_defaults() -> None:
    args = runner.parse_args([])
    assert args.max_prompt_length == 128
    assert args.language_layers == 1
    assert args.language_heads == 2
    assert args.symbol_budget == 0
    assert args.matched_prompt_budget == 0
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
            "--max-prompt-length",
            "128",
            "--language-layers",
            "1",
            "--language-heads",
            "2",
            "--symbol-budget",
            "8",
            "--matched-prompt-budget",
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
    assert "M26 end-to-end Lojban symbiote report written" in result.stdout
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["track"] == "M26"
    assert payload["config"]["max_symbols"] == 16
    assert payload["config"]["symbol_budget"] == 8
    assert payload["config"]["matched_prompt_budget"] == 8
    assert payload["config"]["batch_size"] == 6
    assert payload["config"]["embedding_dim"] == 8
    assert payload["config"]["max_prompt_length"] == 128
    assert payload["config"]["language_layers"] == 1
    assert payload["config"]["language_heads"] == 2
    assert "single_optimizer_generator_and_advisor" in payload["architecture_locks"]
    assert "language_hidden_state_stream_before_bridi_generation" in payload["architecture_locks"]
    assert "bridi_generator_reads_language_hidden_states" in payload["architecture_locks"]
    assert "trace_language_cross_attention_bridge" in payload["architecture_locks"]
    assert "answer_head_reads_fused_language_trace_state" in payload["architecture_locks"]
    assert "raw_prompt_bypass_blocked" in payload["architecture_locks"]
    assert payload["seed_reports"][0]["config"]["organism_mode"] == "lm_hidden_bridi_bridge"
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_generator"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_trace_slot_advisor"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_advisor_classifier"] == 0.0
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_language_backbone"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["answer_loss_reaches_bridge"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["lm_hidden_state_stream_active"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["bridi_generator_reads_lm_hidden_states"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["trace_bridge_reads_prompt_hidden_states"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["answer_head_reads_fused_lm_trace_state"] == 1.0
    assert payload["seed_reports"][0]["metrics"]["raw_prompt_bypass_blocked"] == 1.0
    assert "prompt_history" in payload["seed_reports"][0]
    assert "matched_prompt_history" in payload["seed_reports"][0]
    assert "mean_answer_loss_reaches_generator" in payload["aggregate_metrics"]
    assert "mean_answer_loss_reaches_language_backbone" in payload["aggregate_metrics"]
    assert "mean_answer_loss_reaches_bridge" in payload["aggregate_metrics"]
    assert "mean_answer_loss_reaches_trace_slot_advisor" in payload["aggregate_metrics"]
    assert "mean_answer_loss_reaches_advisor_classifier" in payload["aggregate_metrics"]
    assert "mean_answer_loss_language_backbone_grad_norm" in payload["aggregate_metrics"]
    assert "mean_answer_loss_bridge_grad_norm" in payload["aggregate_metrics"]
    assert "mean_answer_loss_trace_slot_advisor_grad_norm" in payload["aggregate_metrics"]
    assert "mean_answer_loss_advisor_classifier_grad_norm" in payload["aggregate_metrics"]
    assert "mean_lm_hidden_state_stream_active" in payload["aggregate_metrics"]
    assert "mean_bridi_generator_reads_lm_hidden_states" in payload["aggregate_metrics"]
    assert "mean_trace_bridge_reads_prompt_hidden_states" in payload["aggregate_metrics"]
    assert "mean_answer_head_reads_fused_lm_trace_state" in payload["aggregate_metrics"]
    assert "mean_raw_prompt_bypass_blocked" in payload["aggregate_metrics"]
    assert "mean_phrase_accuracy" in payload["aggregate_metrics"]
    assert "mean_matched_prompt_accuracy" in payload["aggregate_metrics"]
    assert "mean_m26_spinal_cord_gate_pass_rate" in payload["aggregate_metrics"]
    assert "mean_m26_full_organism_gate_pass_rate" in payload["aggregate_metrics"]
    assert "mean_m26_full_organism_candidate" in payload["aggregate_metrics"]
    assert payload["aggregate_surface_metrics"]
