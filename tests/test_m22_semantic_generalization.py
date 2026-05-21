from __future__ import annotations

from lojban_evolution.m22.generalization import build_m22_semantic_generalization_payload


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
