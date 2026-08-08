from __future__ import annotations

import os
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_rel_path: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / script_rel_path), "--help"],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_run_experiment_help() -> None:
    out = _run_help("scripts/legacy/run_experiment.py")
    assert "usage:" in out.lower()
    assert "--iterations" in out


def test_run_phase_ablation_help() -> None:
    out = _run_help("scripts/control_plane/pipeline_eval_manifold.py")
    assert "usage:" in out.lower()
    assert "--input-artifact" in out
    assert "--output" in out


def test_build_mixed_dataset_help() -> None:
    out = _run_help("scripts/data/build_mixed_curriculum_dataset.py")
    assert "usage:" in out.lower()
    assert "--output" in out


def test_run_direct_unified_eval_help() -> None:
    out = _run_help("scripts/control_plane/run_direct_unified_eval.py")
    assert "usage:" in out.lower()
    assert "--family" in out
    assert "--execute-m19-direct" in out
    assert "--m24-compression-report" in out
    assert "--m26-end-to-end-report" in out
    assert "--m27-coconut-runtime-report" in out
    assert "--m28-model-report" in out
    assert "--m28-suite-report" in out


def test_run_m28_logebonic_model_suite_help() -> None:
    out = _run_help("scripts/m28/run_m28_logebonic_model_suite.py")
    assert "usage:" in out.lower()
    assert "--seed-list" in out
    assert "--stable-accuracy-threshold" in out
    assert "--no-baselines" in out


def test_run_ablation_test_matrix_help() -> None:
    out = _run_help("scripts/control_plane/run_ablation_test_matrix.py")
    assert "usage:" in out.lower()
    assert "--lane" in out
    assert "--family" in out
    assert "--execute" in out


def test_run_direct_unified_eval_m24_defaults_track_to_family() -> None:
    report = (
        REPO_ROOT
        / "artifacts/runs/telemetry/raw/ablation/hypercube/m24_substrate_compression/pytest_m24_direct_track_fake/m24_substrate_compression_report.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "track": "M24",
                "aggregate_metrics": {
                    "mean_strict_accuracy": 0.25,
                    "mean_predicted_trace_accuracy": 0.25,
                    "mean_random_trace_accuracy": 0.05,
                    "mean_zero_trace_accuracy": 0.05,
                    "mean_prompt_only_accuracy": 0.30,
                    "mean_predicted_vs_random_delta": 0.20,
                    "mean_substrate_claim_score": 0.10,
                    "mean_compression_ratio": 0.50,
                    "mean_substrate_token_count": 12.0,
                    "mean_reference_token_count": 6.0,
                    "mean_m24_promotion_candidate": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    run_id = "pytest_m24_direct_track_default"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/control_plane/run_direct_unified_eval.py"),
            "--family",
            "M24",
            "--m24-compression-report",
            str(report),
            "--run-id",
            run_id,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = (
        REPO_ROOT
        / "artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval"
        / run_id
        / "direct_unified_eval_manifest.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["family_key"] == "M24"
    assert payload["track"] == "M24"
    assert payload["config"]["track"] == "M24"


def test_run_direct_unified_eval_m28_defaults_track_to_family() -> None:
    report = (
        REPO_ROOT
        / "artifacts/tmp/pytest_m28_direct_track_fake/m28_logebonic_model_report.json"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(
            {
                "track": "M28",
                "metrics": {
                    "strict_accuracy": 0.25,
                    "m28_actual_model_artifact": 1.0,
                    "checkpoint_roundtrip_pass": 1.0,
                    "model_inference_api_pass": 1.0,
                    "trace_schema_saved": 1.0,
                    "m28_baseline_comparison_bundle_present": 1.0,
                    "m28_baseline_count": 7.0,
                    "m28_learned_logebonic_accuracy": 0.25,
                    "m28_best_non_logebonic_baseline_accuracy": 0.20,
                    "m28_learned_vs_best_baseline_delta": 0.05,
                    "m28_trace_causality_delta": 0.05,
                },
            }
        ),
        encoding="utf-8",
    )
    run_id = "pytest_m28_direct_track_default"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/control_plane/run_direct_unified_eval.py"),
            "--family",
            "M28",
            "--m28-model-report",
            str(report),
            "--output-root",
            "artifacts/runs/telemetry/raw/ablation/hypercube/test_direct_unified_eval",
            "--run-id",
            run_id,
        ],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = (
        REPO_ROOT
        / "artifacts/runs/telemetry/raw/ablation/hypercube/test_direct_unified_eval"
        / run_id
        / "direct_unified_eval_manifest.json"
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["family_key"] == "M28"
    assert payload["track"] == "M28"
    assert payload["config"]["track"] == "M28"
    assert payload["config"]["m28_model_report"].endswith("m28_logebonic_model_report.json")


def test_run_m19_integrity_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_integrity_suite.py")
    assert "usage:" in out.lower()
    assert "--train-data-path" in out
    assert "--bridge-path" in out


def test_run_m19_replication_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_replication_suite.py")
    assert "usage:" in out.lower()
    assert "--seed-list" in out
    assert "--eval-data-path" in out
    assert "--checkpoint-selection-policy" in out
    assert "--query-repulsion-weight" in out
    assert "--pointer-necessity-weight" in out


def test_run_m19_stability_microgrid_help() -> None:
    out = _run_help("scripts/m19/run_m19_stability_microgrid.py")
    assert "usage:" in out.lower()
    assert "--learning-rate-list" in out
    assert "--augmentation-prob-list" in out
    assert "--format-augmentation-prob-list" in out
    assert "--pointer-necessity-weight-list" in out


def test_run_m19_kill_test_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_kill_test_suite.py")
    assert "usage:" in out.lower()
    assert "--train-data-path" in out
    assert "--bridge-path" in out


def test_run_m19_dictionary_audit_help() -> None:
    out = _run_help("scripts/m19/run_m19_dictionary_audit.py")
    assert "usage:" in out.lower()
    assert "--bridge-spec" in out
    assert "--dataset-path" in out
    assert "--typed-slot-layout" in out


def test_run_m19_typed_physics_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_typed_physics_suite.py")
    assert "usage:" in out.lower()
    assert "--track" in out
    assert "--typed-slot-layout" in out


def test_run_m19_bridge_channel_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_bridge_channel_suite.py")
    assert "usage:" in out.lower()
    assert "--mode-list" in out
    assert "--channel-causality-threshold" in out


def test_run_m19_pointer_counterfactual_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_pointer_counterfactual_suite.py")
    assert "usage:" in out.lower()
    assert "--mode-list" in out
    assert "--pointer-causality-threshold" in out


def test_run_m19_order_sensitivity_suite_help() -> None:
    out = _run_help("scripts/m19/run_m19_order_sensitivity_suite.py")
    assert "usage:" in out.lower()
    assert "--slice-list" in out
    assert "--order-sensitivity-threshold" in out


def test_run_m19_gumbel_and_hyperbolic_suite_help() -> None:
    gumbel = _run_help("scripts/m19/run_m19_gumbel_faithfulness_suite.py")
    hyper = _run_help("scripts/m19/run_m19_hyperbolic_faithfulness_suite.py")
    assert "--epochs" in gumbel
    assert "--poincare-curvature" in hyper
