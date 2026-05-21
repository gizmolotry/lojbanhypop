from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from lojban_evolution.m22.generalization import build_m22_semantic_generalization_payload


def _load_m22_runner():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "m22" / "run_m22_semantic_generalization.py"
    spec = importlib.util.spec_from_file_location("run_m22_semantic_generalization", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m22_generalization_gate_requires_semantic_lift_without_judri_regression() -> None:
    suite_payload = {
        "aggregate_metrics": {
            "mean_strict_accuracy": 0.85,
            "mean_bridi_trace_exact_accuracy": 0.999,
            "mean_gismu_accuracy": 1.0,
            "mean_cmavo_accuracy": 0.999,
            "mean_judri_binding_accuracy": 0.999,
            "mean_cmavo_causal_delta": 0.47,
            "mean_judri_causal_delta": 0.79,
            "stable_seed_rate": 1.0,
        }
    }
    adversarial_payload = {
        "aggregate_metrics": {
            "semantic_coverage_strict_accuracy": 0.43,
            "semantic_coverage_worst_surface_accuracy": 0.35,
            "semantic_coverage_judri_causal_delta": 0.31,
            "semantic_coverage_oov_token_rate": 0.12,
            "semantic_coverage_training_exposure_rate": 1.0,
            "semantic_isolation_cell_count": 8.0,
        }
    }
    control_manifest = {
        "headline_metrics": {
            "strict_accuracy": 0.85,
            "semantic_coverage_strict_accuracy": 0.39,
            "semantic_coverage_worst_surface_accuracy": 0.30,
            "judri_causal_delta": 0.80,
        }
    }

    payload = build_m22_semantic_generalization_payload(
        suite_payload=suite_payload,
        adversarial_payload=adversarial_payload,
        control_manifest_payload=control_manifest,
    )
    metrics = payload["metrics"]

    assert metrics["m22_semantic_strict_delta_vs_m21_control"] == 0.03999999999999998
    assert metrics["m22_semantic_worst_delta_vs_m21_control"] == 0.04999999999999999
    assert metrics["m22_clean_accuracy_drop_vs_m21_control"] == 0.0
    assert metrics["m22_promotion_candidate"] == 1.0
    assert payload["comparison_policy"]["delta_baseline"] == "explicit_m21_control_direct_manifest"


def test_m22_generalization_gate_blocks_clean_or_semantic_regression() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.79,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.69,
            }
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.38,
                "semantic_coverage_worst_surface_accuracy": 0.28,
                "semantic_coverage_judri_causal_delta": 0.20,
            }
        },
        control_manifest_payload={
            "headline_metrics": {
                "strict_accuracy": 0.85,
                "semantic_coverage_strict_accuracy": 0.39,
                "semantic_coverage_worst_surface_accuracy": 0.30,
                "judri_causal_delta": 0.80,
            }
        },
    )

    assert payload["metrics"]["m22_promotion_candidate"] == 0.0
    assert payload["promotion_gates"]["clean_accuracy_not_collapsed"] is False
    assert payload["promotion_gates"]["semantic_strict_improves_control"] is False


def test_m22_generalization_gate_requires_exposure_isolation_and_control() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            }
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.95,
                "semantic_coverage_worst_surface_accuracy": 0.95,
                "semantic_coverage_judri_causal_delta": 0.95,
                "semantic_coverage_training_exposure_rate": 0.0,
                "semantic_isolation_cell_count": 0.0,
            }
        },
        control_manifest_payload={},
    )

    assert payload["metrics"]["m22_promotion_candidate"] == 0.0
    assert payload["promotion_gates"]["semantic_training_exposed"] is False
    assert payload["promotion_gates"]["semantic_isolation_evidence_present"] is False
    assert payload["promotion_gates"]["explicit_m21_control_present"] is False


def test_m22_generalization_preserves_explicit_zero_semantic_metrics() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            }
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.0,
                "semantic_coverage_worst_surface_accuracy": 0.0,
                "semantic_coverage_judri_causal_delta": 0.0,
                "mean_adversarial_strict_accuracy": 0.9,
                "mean_adversarial_worst_surface_accuracy": 0.9,
                "mean_adversarial_judri_causal_delta": 0.9,
                "semantic_coverage_training_exposure_rate": 1.0,
                "semantic_isolation_cell_count": 8.0,
            }
        },
        control_manifest_payload={
            "headline_metrics": {
                "strict_accuracy": 0.85,
                "semantic_coverage_strict_accuracy": 0.39,
                "semantic_coverage_worst_surface_accuracy": 0.30,
                "judri_causal_delta": 0.80,
            }
        },
    )

    assert payload["metrics"]["semantic_coverage_strict_accuracy"] == 0.0
    assert payload["metrics"]["semantic_coverage_worst_surface_accuracy"] == 0.0
    assert payload["metrics"]["semantic_coverage_judri_causal_delta"] == 0.0
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_runner_writes_report_from_fixture_paths(tmp_path: Path) -> None:
    runner = _load_m22_runner()
    suite_path = tmp_path / "suite.json"
    adversarial_path = tmp_path / "adversarial.json"
    control_path = tmp_path / "control.json"
    suite_path.write_text(
        json.dumps(
            {
                "aggregate_metrics": {
                    "mean_strict_accuracy": 0.85,
                    "mean_bridi_trace_exact_accuracy": 0.999,
                    "mean_judri_causal_delta": 0.79,
                }
            }
        ),
        encoding="utf-8",
    )
    adversarial_path.write_text(
        json.dumps(
            {
                "aggregate_metrics": {
                    "semantic_coverage_strict_accuracy": 0.43,
                    "semantic_coverage_worst_surface_accuracy": 0.35,
                    "semantic_coverage_judri_causal_delta": 0.31,
                    "semantic_coverage_training_exposure_rate": 1.0,
                    "semantic_isolation_cell_count": 8.0,
                }
            }
        ),
        encoding="utf-8",
    )
    control_path.write_text(
        json.dumps(
            {
                "headline_metrics": {
                    "strict_accuracy": 0.85,
                    "semantic_coverage_strict_accuracy": 0.39,
                    "semantic_coverage_worst_surface_accuracy": 0.30,
                    "judri_causal_delta": 0.80,
                }
            }
        ),
        encoding="utf-8",
    )

    output_root = Path("artifacts/runs/telemetry/raw/ablation/hypercube/m22_semantic_generalization_test")
    args = runner.parse_args(
        [
            "--suite-report",
            str(suite_path),
            "--adversarial-audit-report",
            str(adversarial_path),
            "--m21-control-direct-manifest",
            str(control_path),
            "--output-root",
            str(output_root),
            "--run-id",
            "fixture",
        ]
    )
    payload = runner.run_generalization(args)
    report_path = output_root / "fixture" / "m22_semantic_generalization_report.json"

    assert report_path.exists()
    assert payload["track"] == "M22"
    assert payload["source_reports"]["m21_suite_report"] == str(suite_path)
    assert "metrics" in payload
    assert "promotion_gates" in payload
