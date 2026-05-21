from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_whole_grid_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "control_plane" / "run_whole_ablation_grid.py"
    spec = importlib.util.spec_from_file_location("run_whole_ablation_grid", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m19_direct_unified_eval_headline_metrics_are_not_clobbered() -> None:
    whole_grid = _load_whole_grid_module()
    whole_grid.DEFAULT_M19_REPLICATION_ROOT = Path("runs") / "missing_replication_for_test"
    whole_grid.DEFAULT_M19_KILL_ROOT = Path("runs") / "missing_kill_for_test"
    payload = {
        "headline_metrics": {
            "overall_accuracy": 0.3775,
            "avg_tokens": 32.0,
            "lift_vs_en_cot": 0.375,
            "lift_vs_random": 0.3575,
            "audit_qformer_accuracy": 1.0,
            "purged_accuracy": 0.373,
            "masked_accuracy": 0.0,
            "replication_mean_accuracy": 0.2,
            "replication_std_accuracy": 0.2,
            "entity_accuracy": 0.0,
            "format_accuracy": 0.2625,
            "numeric_accuracy": 0.3625,
            "typed_family_accuracy": 0.88,
            "masked_pointer_zero_rate": 1.0,
            "family_slot_entropy": 0.22,
        },
        "metrics": {},
    }

    metrics = whole_grid._special_stage_metrics("M19", payload)

    assert metrics["mainline_overall_accuracy"] == 0.3775
    assert metrics["mainline_avg_tokens"] == 32.0
    assert metrics["mainline_lift_vs_en_cot"] == 0.375
    assert metrics["mainline_lift_vs_random"] == 0.3575
    assert metrics["mainline_audit_qformer_accuracy"] == 1.0
    assert metrics["purged_accuracy"] == 0.373
    assert metrics["replication_mean_accuracy"] == 0.2
    assert metrics["replication_std_accuracy"] == 0.2
    assert metrics["kill_entity_accuracy"] == 0.0
    assert metrics["kill_format_accuracy"] == 0.2625
    assert metrics["kill_numeric_accuracy"] == 0.3625
    assert metrics["typed_family_accuracy"] == 0.88
    assert metrics["masked_pointer_zero_rate"] == 1.0
    assert metrics["family_slot_entropy"] == 0.22


def test_m19_legacy_metrics_can_supplement_headline_metrics() -> None:
    whole_grid = _load_whole_grid_module()
    whole_grid.DEFAULT_M19_REPLICATION_ROOT = Path("runs") / "missing_replication_for_test"
    whole_grid.DEFAULT_M19_KILL_ROOT = Path("runs") / "missing_kill_for_test"
    payload = {
        "headline_metrics": {
            "overall_accuracy": 0.31,
            "avg_tokens": 17.0,
        },
        "metrics": {
            "premature_stop_rate": 0.02,
            "max_cap_hit_rate": 0.01,
            "caa_manifold_entanglement_score": 0.13,
        },
    }

    metrics = whole_grid._special_stage_metrics("M19", payload)

    assert metrics["mainline_overall_accuracy"] == 0.31
    assert metrics["mainline_avg_tokens"] == 17.0
    assert metrics["mainline_premature_stop_rate"] == 0.02
    assert metrics["mainline_max_cap_hit_rate"] == 0.01
    assert metrics["mainline_caa_entanglement"] == 0.13


def test_m19_direct_contract_updates_stale_spine_policy() -> None:
    whole_grid = _load_whole_grid_module()
    stage = {
        "stage_key": "M19",
        "required_test_contracts": ["m19.runway_efficiency"],
        "comparison_targets": [{"target": "M18"}],
        "historical_comparison_families": ["J"],
    }
    payload = {
        "comparison_contract": {
            "required_test_contract_ids": [
                "m19.runway_efficiency",
                "m19.replication_stability",
                "m19.kill_test_suite",
            ],
            "comparison_targets": [{"target": "M19"}, {"target": "M18"}],
            "historical_comparison_families": ["J", "L"],
        }
    }

    updated = whole_grid._stage_with_direct_contract(stage, payload)

    assert updated["required_test_contracts"] == [
        "m19.runway_efficiency",
        "m19.replication_stability",
        "m19.kill_test_suite",
    ]
    assert [target["target"] for target in updated["comparison_targets"]] == ["M19", "M18"]
    assert updated["historical_comparison_families"] == ["J", "L"]


def test_m21_special_stage_metrics_include_dynamic_bridi_and_causal_deltas() -> None:
    whole_grid = _load_whole_grid_module()
    whole_grid.DEFAULT_M21_ACTUAL_ROOT = Path("runs") / "missing_m21_actual_for_test"
    whole_grid.DEFAULT_M21_LOCK_ROOT = Path("runs") / "missing_m21_lock_for_test"
    payload = {
        "aggregate_metrics": {
            "mean_strict_accuracy": 0.72,
            "mean_bridi_trace_exact_accuracy": 0.66,
            "mean_gismu_accuracy": 0.89,
            "mean_cmavo_accuracy": 0.81,
            "mean_judri_binding_accuracy": 0.77,
            "mean_frame_drop_delta": 0.11,
            "mean_cmavo_causal_delta": 0.21,
            "mean_judri_causal_delta": 0.09,
            "accuracy_per_trace_token": 0.08,
            "semantic_coverage_strict_accuracy": 0.37,
            "semantic_coverage_worst_surface_accuracy": 0.29,
            "semantic_coverage_judri_causal_delta": 0.31,
            "semantic_coverage_training_exposure_rate": 1.0,
            "semantic_coverage_surface_count": 2.0,
            "semantic_isolation_cell_count": 8.0,
            "semantic_coverage_lexical_shift_effect_strict_accuracy_delta": 0.14,
            "semantic_coverage_role_binding_effect_strict_accuracy_delta": 0.11,
            "semantic_coverage_combined_effect_strict_accuracy_delta": 0.20,
            "semantic_coverage_fraction_effect_strict_accuracy_delta": 0.02,
            "semantic_coverage_role_curriculum_effect_strict_accuracy_delta": 0.09,
        }
    }

    metrics = whole_grid._special_stage_metrics("M21", payload)

    assert metrics["strict_accuracy"] == 0.72
    assert metrics["bridi_trace_exact_accuracy"] == 0.66
    assert metrics["gismu_accuracy"] == 0.89
    assert metrics["cmavo_accuracy"] == 0.81
    assert metrics["judri_binding_accuracy"] == 0.77
    assert metrics["frame_drop_delta"] == 0.11
    assert metrics["cmavo_causal_delta"] == 0.21
    assert metrics["judri_causal_delta"] == 0.09
    assert metrics["accuracy_per_trace_token"] == 0.08
    assert metrics["semantic_coverage_strict_accuracy"] == 0.37
    assert metrics["semantic_coverage_worst_surface_accuracy"] == 0.29
    assert metrics["semantic_coverage_judri_causal_delta"] == 0.31
    assert metrics["semantic_coverage_training_exposure_rate"] == 1.0
    assert metrics["semantic_coverage_surface_count"] == 2.0
    assert metrics["semantic_isolation_cell_count"] == 8.0
    assert metrics["semantic_coverage_lexical_shift_effect_strict_accuracy_delta"] == 0.14
    assert metrics["semantic_coverage_role_binding_effect_strict_accuracy_delta"] == 0.11
    assert metrics["semantic_coverage_combined_effect_strict_accuracy_delta"] == 0.20
    assert metrics["semantic_coverage_fraction_effect_strict_accuracy_delta"] == 0.02
    assert metrics["semantic_coverage_role_curriculum_effect_strict_accuracy_delta"] == 0.09


def test_m22_is_visible_in_whole_grid_stage_order_and_metrics() -> None:
    whole_grid = _load_whole_grid_module()
    payload = {
        "metrics": {
            "strict_accuracy": 0.849,
            "semantic_coverage_strict_accuracy": 0.653,
            "semantic_coverage_worst_surface_accuracy": 0.181,
            "semantic_coverage_judri_causal_delta": 0.600,
            "m22_semantic_generalization_score": 0.181,
            "m22_semantic_strict_delta_vs_m21_control": 0.260,
            "m22_semantic_worst_delta_vs_m21_control": -0.116,
            "m22_clean_accuracy_drop_vs_m21_control": 0.001,
            "m22_judri_delta_drop_vs_m21_control": 0.001,
            "m22_promotion_gate_pass_rate": 0.833,
            "m22_promotion_candidate": 0.0,
        }
    }

    metrics = whole_grid._special_stage_metrics("M22", payload)

    assert "M22" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M21") < whole_grid.STAGE_ORDER.index("M22")
    assert metrics["strict_accuracy"] == 0.849
    assert metrics["semantic_coverage_strict_accuracy"] == 0.653
    assert metrics["m22_semantic_generalization_score"] == 0.181
    assert metrics["m22_promotion_candidate"] == 0.0
