from __future__ import annotations

import importlib.util
import json
import os
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


def test_latest_direct_eval_anchor_prefers_stable_artifacts_over_pytest_smoke(tmp_path: Path) -> None:
    whole_grid = _load_whole_grid_module()
    root = tmp_path / "direct_unified_eval"
    smoke = root / "pytest_m24_direct_track_default" / "direct_unified_eval_manifest.json"
    stable = root / "m24_2_full_direct_20260528" / "direct_unified_eval_manifest.json"
    smoke.parent.mkdir(parents=True)
    stable.parent.mkdir(parents=True)
    smoke.write_text(json.dumps({"family_key": "M24"}), encoding="utf-8")
    stable.write_text(json.dumps({"family_key": "M24"}), encoding="utf-8")
    os.utime(stable, (1000, 1000))
    os.utime(smoke, (2000, 2000))
    whole_grid.DEFAULT_DIRECT_UNIFIED_EVAL_ROOT = root

    assert whole_grid._latest_direct_unified_eval_anchor("M24") == stable


def test_latest_ablation_test_matrix_anchor_prefers_latest_executed_over_dry_runs(tmp_path: Path) -> None:
    whole_grid = _load_whole_grid_module()
    root = tmp_path / "ablation_test_matrix"
    passed = root / "passed" / "ablation_test_matrix_manifest.json"
    failed = root / "failed" / "ablation_test_matrix_manifest.json"
    dry = root / "dry" / "ablation_test_matrix_manifest.json"
    passed.parent.mkdir(parents=True)
    failed.parent.mkdir(parents=True)
    dry.parent.mkdir(parents=True)
    passed.write_text(
        json.dumps({"status": "passed", "execute": True, "metrics": {"pytest_executed": 1.0, "pytest_passed": 1.0}}),
        encoding="utf-8",
    )
    failed.write_text(
        json.dumps({"status": "failed", "execute": True, "metrics": {"pytest_executed": 1.0, "pytest_passed": 0.0}}),
        encoding="utf-8",
    )
    dry.write_text(json.dumps({"status": "dry_run", "execute": False, "metrics": {"pytest_executed": 0.0}}), encoding="utf-8")
    os.utime(passed, (1000, 1000))
    os.utime(failed, (2000, 2000))
    os.utime(dry, (3000, 3000))
    whole_grid.DEFAULT_ABLATION_TEST_MATRIX_ROOT = root

    assert whole_grid._latest_ablation_test_matrix_anchor() == failed


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
            "m22_candidate_cell_count": 4.0,
            "m22_candidate_cells_present": 1.0,
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
    assert list(metrics)[:2] == ["m22_promotion_candidate", "m22_promotion_gate_pass_rate"]
    assert metrics["strict_accuracy"] == 0.849
    assert metrics["m22_candidate_cell_count"] == 4.0
    assert metrics["semantic_coverage_strict_accuracy"] == 0.653
    assert metrics["m22_semantic_generalization_score"] == 0.181
    assert metrics["m22_promotion_candidate"] == 0.0


def test_m23_special_stage_metrics_and_order() -> None:
    whole_grid = _load_whole_grid_module()
    payload = {
        "aggregate_metrics": {
            "mean_strict_accuracy": 0.72,
            "mean_decoy_relation_ood_accuracy": 0.64,
            "mean_worst_surface_accuracy": 0.61,
            "mean_relevance_top1_accuracy": 0.83,
            "m23_router_decoy_lift_vs_scale": 0.12,
            "m23_oracle_relevance_lift": 0.03,
        }
    }

    metrics = whole_grid._special_stage_metrics("M23", payload)

    assert "M23" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M22") < whole_grid.STAGE_ORDER.index("M23")
    assert metrics["strict_accuracy"] == 0.72
    assert metrics["decoy_relation_ood_accuracy"] == 0.64
    assert metrics["relevance_top1_accuracy"] == 0.83
    assert metrics["m23_router_decoy_lift_vs_scale"] == 0.12


def test_m24_special_stage_metrics_and_order() -> None:
    whole_grid = _load_whole_grid_module()
    payload = {
        "headline_metrics": {
            "strict_accuracy": 0.69,
            "overall_phrase_accuracy": 0.74,
            "phrase_accuracy": 0.75,
            "substrate_token_count": 6.5,
            "reference_token_count": 18.0,
            "compression_ratio": 0.3611,
            "token_reduction_ratio": 0.6389,
            "token_ratio_vs_m23": 0.42,
            "compression_lift_vs_m23": 0.11,
            "avg_tokens": 9.0,
            "trace_tokens": 6.5,
            "accuracy_per_token": 0.0767,
            "accuracy_per_trace_token": 0.1062,
            "compression_adjusted_strict_accuracy": 1.91,
            "strict_accuracy_per_substrate_token": 0.1062,
            "shuffled_trace_accuracy": 0.11,
            "predicted_vs_shuffled_delta": 0.58,
            "mdl_weight": 0.025,
            "m24_gate_packed_trace_shorter_than_prompt": 1.0,
            "m24_gate_trace_beats_random": 1.0,
            "m24_gate_trace_beats_zero": 1.0,
            "m24_gate_trace_beats_shuffled": 1.0,
            "m24_gate_trace_matches_oracle_upper_bound": 0.0,
            "m24_gate_trace_beats_prompt_only": 0.0,
            "m24_gate_nonzero_exact_trace_reconstruction": 1.0,
            "m24_gate_token_reduction_positive": 1.0,
            "m24_2_hard_bottleneck_strict_accuracy": 0.67,
            "m24_2_hard_bottleneck_token_count": 4.0,
            "m24_2_hard_bottleneck_compression_ratio": 0.2222,
            "m24_2_hard_bottleneck_accuracy_per_token": 0.1675,
            "m24_2_hard_bottleneck_delta_vs_m24_1": -0.02,
            "m24_2_hard_bottleneck_delta_vs_prompt_only": -0.07,
            "m24_2_hard_bottleneck_symbol_error_rate": 0.03,
            "m24_2_hard_bottleneck_score": 0.41,
            "m24_2_promotion_gate_pass_rate": 0.6,
            "m24_2_promotion_candidate": 0.0,
            "m24_2_gate_hard_bottleneck_configured": 1.0,
            "m24_2_gate_strict_accuracy_retained": 1.0,
            "m24_2_gate_trace_beats_shuffled_strong": 0.0,
            "m24_2_gate_trace_beats_random_strong": 1.0,
            "m24_2_gate_trace_exact_floor": 1.0,
            "m24_2_gate_symbol_budget_respected": 1.0,
            "m24_2_gate_hard_trace_beats_random": 1.0,
            "m24_2_gate_hard_trace_beats_prompt_only": 0.0,
            "m24_2_gate_token_reduction_positive": 1.0,
        },
        "aggregate_metrics": {
            "mean_m24_2_hard_bottleneck_trace_exact_accuracy": 0.62,
            "mean_judri_binding_accuracy": 0.66,
        },
        "cells": {
            "phrase_only": {"metrics": {"strict_accuracy": 0.10, "overall_accuracy": 0.99}},
            "strict_best": {"metrics": {"strict_accuracy": 0.80, "overall_accuracy": 0.70}},
        },
    }

    metrics = whole_grid._special_stage_metrics("M24", payload)

    assert "M24" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M23") < whole_grid.STAGE_ORDER.index("M24")
    assert list(metrics)[:4] == [
        "strict_accuracy",
        "m24_2_promotion_candidate",
        "m24_2_promotion_gate_pass_rate",
        "m24_2_hard_bottleneck_compression_ratio",
    ]
    assert metrics["strict_accuracy"] == 0.69
    assert metrics["overall_phrase_accuracy"] == 0.74
    assert metrics["compression_ratio"] == 0.3611
    assert metrics["token_reduction_ratio"] == 0.6389
    assert metrics["shuffled_trace_accuracy"] == 0.11
    assert metrics["predicted_vs_shuffled_delta"] == 0.58
    assert metrics["mdl_weight"] == 0.025
    assert metrics["m24_gate_packed_trace_shorter_than_prompt"] == 1.0
    assert metrics["m24_gate_trace_beats_random"] == 1.0
    assert metrics["m24_gate_trace_beats_zero"] == 1.0
    assert metrics["m24_gate_trace_beats_shuffled"] == 1.0
    assert metrics["m24_gate_trace_matches_oracle_upper_bound"] == 0.0
    assert metrics["m24_gate_trace_beats_prompt_only"] == 0.0
    assert metrics["m24_gate_nonzero_exact_trace_reconstruction"] == 1.0
    assert metrics["m24_2_hard_bottleneck_strict_accuracy"] == 0.67
    assert metrics["m24_2_hard_bottleneck_trace_exact_accuracy"] == 0.62
    assert metrics["m24_2_hard_bottleneck_token_count"] == 4.0
    assert metrics["m24_2_hard_bottleneck_score"] == 0.41
    assert metrics["m24_2_promotion_candidate"] == 0.0
    assert metrics["m24_2_gate_strict_accuracy_retained"] == 1.0
    assert metrics["m24_2_gate_trace_beats_shuffled_strong"] == 0.0
    assert metrics["m24_2_gate_trace_exact_floor"] == 1.0
    assert metrics["strict_accuracy_per_substrate_token"] == 0.1062
    assert metrics["judri_binding_accuracy"] == 0.66
    assert metrics["best_cell_accuracy"] == 0.80


def test_m25_special_stage_metrics_and_order() -> None:
    whole_grid = _load_whole_grid_module()
    payload = {
        "aggregate_metrics": {
            "mean_strict_accuracy": 0.71,
            "mean_predicted_stream_accuracy": 0.71,
            "mean_oracle_stream_accuracy": 0.78,
            "mean_shuffled_stream_accuracy": 0.42,
            "mean_random_stream_accuracy": 0.39,
            "mean_prompt_only_accuracy": 0.69,
            "mean_m25_strict_delta_vs_prompt_only": 0.02,
            "mean_matched_prompt_accuracy": 0.52,
            "mean_m25_strict_delta_vs_matched_prompt": 0.19,
            "mean_predicted_vs_shuffled_delta": 0.29,
            "mean_predicted_vs_random_delta": 0.32,
            "mean_loose_stream_exact_accuracy": 0.33,
            "mean_stream_type_accuracy": 0.81,
            "mean_token_reduction_ratio": 0.44,
            "mean_accuracy_per_loose_symbol": 0.12,
            "mean_matched_prompt_accuracy_per_token": 0.065,
            "mean_m25_accuracy_per_symbol_delta_vs_matched_prompt": 0.055,
            "mean_m25_gate_beats_matched_prompt": 1.0,
            "mean_m25_promotion_gate_pass_rate": 1.0,
            "mean_m25_promotion_candidate": 1.0,
            "mean_m25_gate_stream_beats_shuffled": 1.0,
        }
    }

    metrics = whole_grid._special_stage_metrics("M25", payload)

    assert "M25" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M24") < whole_grid.STAGE_ORDER.index("M25")
    assert whole_grid.STAGE_ORDER.index("M25") < whole_grid.STAGE_ORDER.index("Control Plane")
    assert list(metrics)[:4] == [
        "strict_accuracy",
        "m25_promotion_candidate",
        "m25_promotion_gate_pass_rate",
        "loose_stream_exact_accuracy",
    ]
    assert metrics["strict_accuracy"] == 0.71
    assert metrics["predicted_stream_accuracy"] == 0.71
    assert metrics["shuffled_stream_accuracy"] == 0.42
    assert metrics["predicted_vs_shuffled_delta"] == 0.29
    assert metrics["matched_prompt_accuracy"] == 0.52
    assert metrics["m25_strict_delta_vs_matched_prompt"] == 0.19
    assert metrics["loose_stream_exact_accuracy"] == 0.33
    assert metrics["stream_type_accuracy"] == 0.81
    assert metrics["token_reduction_ratio"] == 0.44
    assert metrics["m25_promotion_candidate"] == 1.0


def test_m26_special_stage_metrics_and_order() -> None:
    whole_grid = _load_whole_grid_module()
    payload = {
        "aggregate_metrics": {
            "mean_strict_accuracy": 0.44,
            "mean_phrase_accuracy": 0.99,
            "mean_end_to_end_answer_accuracy": 0.44,
            "mean_zero_trace_accuracy": 0.20,
            "mean_matched_prompt_accuracy": 0.40,
            "mean_m26_strict_delta_vs_matched_prompt": 0.04,
            "mean_predicted_vs_zero_delta": 0.24,
            "mean_answer_loss_generator_grad_norm": 1.5,
            "mean_answer_loss_symbol_head_grad_norm": 0.8,
            "mean_answer_loss_advisor_grad_norm": 0.9,
            "mean_answer_loss_trace_slot_advisor_grad_norm": 0.9,
            "mean_answer_loss_advisor_classifier_grad_norm": 0.0,
            "mean_answer_loss_language_backbone_grad_norm": 2.5,
            "mean_answer_loss_bridge_grad_norm": 0.7,
            "mean_answer_loss_reaches_generator": 1.0,
            "mean_answer_loss_reaches_symbol_heads": 1.0,
            "mean_answer_loss_reaches_trace_slot_advisor": 1.0,
            "mean_answer_loss_reaches_advisor_classifier": 0.0,
            "mean_answer_loss_reaches_language_backbone": 1.0,
            "mean_answer_loss_reaches_bridge": 1.0,
            "mean_single_optimizer_end_to_end_training": 1.0,
            "mean_hard_argmax_training_cut_detected": 0.0,
            "mean_lm_hidden_state_stream_active": 1.0,
            "mean_bridi_generator_reads_lm_hidden_states": 1.0,
            "mean_trace_bridge_reads_prompt_hidden_states": 1.0,
            "mean_answer_head_reads_fused_lm_trace_state": 1.0,
            "mean_raw_prompt_bypass_blocked": 1.0,
            "mean_bridge_gate_value": 0.42,
            "mean_bridge_delta_norm": 0.13,
            "mean_trace_attention_entropy": 0.2,
            "mean_trace_active_mass": 8.0,
            "mean_trainable_parameter_count": 123.0,
            "mean_language_backbone_trainable_parameter_count": 45.0,
            "mean_m26_gate_beats_matched_prompt": 1.0,
            "mean_m26_gate_answer_loss_reaches_language_backbone": 1.0,
            "mean_m26_gate_answer_loss_reaches_bridge": 1.0,
            "mean_m26_gate_bridi_generator_reads_lm_hidden_states": 1.0,
            "mean_m26_gate_trace_bridge_reads_prompt_hidden_states": 1.0,
            "mean_m26_gate_answer_head_reads_fused_lm_trace_state": 1.0,
            "mean_m26_gate_raw_prompt_bypass_blocked": 1.0,
            "mean_m26_spinal_cord_gate_pass_rate": 1.0,
            "mean_m26_spinal_cord_candidate": 1.0,
            "mean_m26_full_organism_gate_pass_rate": 1.0,
            "mean_m26_full_organism_candidate": 1.0,
            "mean_m26_prompt_comparable_candidate": 1.0,
            "mean_m26_promotion_candidate": 1.0,
        }
    }

    metrics = whole_grid._special_stage_metrics("M26", payload)

    assert "M26" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M25") < whole_grid.STAGE_ORDER.index("M26")
    assert whole_grid.STAGE_ORDER.index("M26") < whole_grid.STAGE_ORDER.index("Control Plane")
    if "m26_full_organism_gate_pass_rate" in metrics:
        assert list(metrics)[:7] == [
            "strict_accuracy",
            "m26_promotion_candidate",
            "m26_full_organism_gate_pass_rate",
            "m26_full_organism_candidate",
            "m26_spinal_cord_gate_pass_rate",
            "m26_spinal_cord_candidate",
            "m26_prompt_comparable_candidate",
        ]
    else:
        assert list(metrics)[:5] == [
            "strict_accuracy",
            "m26_promotion_candidate",
            "m26_spinal_cord_gate_pass_rate",
            "m26_spinal_cord_candidate",
            "m26_prompt_comparable_candidate",
        ]
    assert metrics["answer_loss_reaches_generator"] == 1.0
    assert metrics["strict_accuracy"] == 0.44
    assert metrics["phrase_accuracy"] == 0.99
    assert metrics["matched_prompt_accuracy"] == 0.40
    assert metrics["m26_strict_delta_vs_matched_prompt"] == 0.04
    assert metrics["answer_loss_generator_grad_norm"] == 1.5
    assert metrics["answer_loss_advisor_grad_norm"] == 0.9
    assert metrics["answer_loss_trace_slot_advisor_grad_norm"] == 0.9
    assert metrics["answer_loss_advisor_classifier_grad_norm"] == 0.0
    assert metrics["answer_loss_reaches_trace_slot_advisor"] == 1.0
    assert metrics["answer_loss_reaches_advisor_classifier"] == 0.0
    assert metrics["predicted_vs_zero_delta"] == 0.24
    assert metrics["trainable_parameter_count"] == 123.0
    assert metrics["m26_gate_beats_matched_prompt"] == 1.0
    if "m26_full_organism_gate_pass_rate" in metrics:
        assert metrics["m26_full_organism_gate_pass_rate"] == 1.0
        assert metrics["m26_full_organism_candidate"] == 1.0
        assert metrics["answer_loss_reaches_language_backbone"] == 1.0
        assert metrics["answer_loss_reaches_bridge"] == 1.0
        assert metrics["answer_loss_language_backbone_grad_norm"] == 2.5
        assert metrics["answer_loss_bridge_grad_norm"] == 0.7
        assert metrics["lm_hidden_state_stream_active"] == 1.0
        assert metrics["bridi_generator_reads_lm_hidden_states"] == 1.0
        assert metrics["trace_bridge_reads_prompt_hidden_states"] == 1.0
        assert metrics["answer_head_reads_fused_lm_trace_state"] == 1.0
        assert metrics["raw_prompt_bypass_blocked"] == 1.0

    assert (
        whole_grid._direct_promotion_status(
            {"contract_results": [{"test_id": "m26.end_to_end_spinal_cord", "promotion_status": "m26_spinal_promoted_prompt_gap"}]}
        )
        == "m26_spinal_promoted_prompt_gap"
    )


def test_m27_special_stage_row_is_full_row_with_runtime_metrics(tmp_path: Path) -> None:
    whole_grid = _load_whole_grid_module()
    report = tmp_path / "m27_coconut_bridi_runtime_report.json"
    report.write_text(
        json.dumps(
            {
                "aggregate_metrics": {
                    "mean_strict_accuracy": 0.52,
                    "mean_soft_free_run_strict_accuracy": 0.52,
                    "mean_hard_free_run_strict_accuracy": 0.50,
                    "mean_m27_promotion_candidate": 0.0,
                    "mean_m27_full_organism_gate_pass_rate": 1.0,
                    "mean_m27_wiring_candidate": 1.0,
                    "mean_m27_full_organism_candidate": 1.0,
                    "mean_m27_prompt_comparable_candidate": 1.0,
                    "mean_m27_step_dependency_delta": 0.03,
                    "mean_m27_relevance_runtime_enabled": 1.0,
                    "mean_m27_relevance_runtime_active": 1.0,
                    "mean_m27_relevance_top1_accuracy": 0.75,
                    "mean_m27_relevance_margin": 0.20,
                    "mean_m27_relevance_full_vs_random_delta": 0.12,
                    "mean_m27_inherited_contract_bundle_present": 1.0,
                    "mean_answer_loss_reaches_generator": 1.0,
                    "mean_answer_loss_reaches_coconut_cell": 1.0,
                    "mean_answer_loss_reaches_recurrent_bridi_feedback": 1.0,
                    "mean_answer_loss_reaches_symbol_heads": 1.0,
                    "mean_answer_loss_reaches_language_backbone": 1.0,
                    "mean_answer_loss_reaches_bridge": 1.0,
                    "mean_m27_gate_answer_loss_trains_soft_free_run": 1.0,
                    "mean_predicted_vs_zero_delta": 0.0,
                }
            }
        ),
        encoding="utf-8",
    )
    stage = {
        "stage_key": "M27",
        "title": "M27",
        "executed_count": 0,
        "report_count": 1,
        "archival_count": 0,
        "deferred_count": 0,
        "artifact_roots": [str(report)],
    }

    whole_grid.DEFAULT_DIRECT_UNIFIED_EVAL_ROOT = tmp_path / "missing_direct"
    whole_grid.DEFAULT_M27_COCONUT_ROOT = tmp_path / "missing_m27"
    metrics = whole_grid._special_stage_metrics("M27", json.loads(report.read_text(encoding="utf-8")))
    row = whole_grid._special_stage_row(stage)
    markdown = whole_grid._render_markdown({"stage_rows": [row], "coverage_summary": {}, "source_manifests": {}})

    assert "M27" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M26") < whole_grid.STAGE_ORDER.index("M27")
    assert row["stage_key"] == "M27"
    assert row["surface_kind"] == "artifact_anchor"
    assert "headline_metrics" in row
    assert row["headline_metrics"]["m27_promotion_candidate"] == 0.0
    assert row["headline_metrics"]["m27_full_organism_gate_pass_rate"] == 1.0
    assert row["headline_metrics"]["m27_wiring_candidate"] == 1.0
    assert row["headline_metrics"]["m27_relevance_runtime_enabled"] == 1.0
    assert row["headline_metrics"]["m27_relevance_full_vs_random_delta"] == 0.12
    assert row["headline_metrics"]["m27_inherited_contract_bundle_present"] == 1.0
    assert row["headline_metrics"]["answer_loss_reaches_generator"] == 1.0
    assert row["headline_metrics"]["answer_loss_reaches_language_backbone"] == 1.0
    assert metrics["m27_gate_answer_loss_trains_soft_free_run"] == 1.0
    assert "| `M27` | `artifact_anchor` |" in markdown
    assert "m27_promotion_candidate=0" in markdown
    assert "m27_inherited_contract_bundle_present=1" in markdown


def test_m28_special_stage_row_surfaces_actual_model_metrics(tmp_path: Path) -> None:
    whole_grid = _load_whole_grid_module()
    report = tmp_path / "m28_logebonic_model_report.json"
    report.write_text(
        json.dumps(
            {
                "metrics": {
                    "strict_accuracy": 0.42,
                    "m28_actual_model_artifact": 1.0,
                    "checkpoint_roundtrip_pass": 1.0,
                    "model_inference_api_pass": 1.0,
                    "trace_schema_saved": 1.0,
                    "m28_baseline_comparison_bundle_present": 1.0,
                    "m28_baseline_count": 7.0,
                    "m28_learned_logebonic_accuracy": 0.42,
                    "m28_best_non_logebonic_baseline_accuracy": 0.38,
                    "m28_learned_vs_best_baseline_delta": 0.04,
                    "m28_trace_causality_delta": 0.11,
                    "m27_full_organism_gate_pass_rate": 1.0,
                    "answer_loss_reaches_generator": 1.0,
                    "answer_loss_reaches_coconut_cell": 1.0,
                    "answer_loss_reaches_recurrent_bridi_feedback": 1.0,
                    "answer_loss_reaches_language_backbone": 1.0,
                    "answer_loss_reaches_bridge": 1.0,
                },
                "baseline_comparison": {
                    "summary": {
                        "m28_baseline_comparison_bundle_present": 1.0,
                        "m28_baseline_count": 7.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stage = {
        "stage_key": "M28",
        "title": "M28",
        "executed_count": 0,
        "report_count": 1,
        "archival_count": 0,
        "deferred_count": 0,
        "artifact_roots": [str(report)],
    }

    whole_grid.DEFAULT_DIRECT_UNIFIED_EVAL_ROOT = tmp_path / "missing_direct"
    whole_grid.DEFAULT_M28_LOGEBONIC_SUITE_ROOT = tmp_path / "missing_m28_suite"
    whole_grid.DEFAULT_M28_LOGEBONIC_ROOT = tmp_path / "missing_m28"
    metrics = whole_grid._special_stage_metrics("M28", json.loads(report.read_text(encoding="utf-8")))
    row = whole_grid._special_stage_row(stage)
    markdown = whole_grid._render_markdown({"stage_rows": [row], "coverage_summary": {}, "source_manifests": {}})

    assert "M28" in whole_grid.STAGE_ORDER
    assert whole_grid.STAGE_ORDER.index("M27") < whole_grid.STAGE_ORDER.index("M28")
    assert row["stage_key"] == "M28"
    assert row["surface_kind"] == "artifact_anchor"
    assert row["headline_metrics"]["m28_actual_model_artifact"] == 1.0
    assert row["headline_metrics"]["checkpoint_roundtrip_pass"] == 1.0
    assert row["headline_metrics"]["model_inference_api_pass"] == 1.0
    assert row["headline_metrics"]["trace_schema_saved"] == 1.0
    assert row["headline_metrics"]["m28_baseline_comparison_bundle_present"] == 1.0
    assert row["headline_metrics"]["m28_learned_vs_best_baseline_delta"] == 0.04
    assert metrics["m28_trace_causality_delta"] == 0.11
    assert "| `M28` | `artifact_anchor` |" in markdown
    assert "m28_actual_model_artifact=1" in markdown
    assert "m28_learned_vs_best_baseline_delta=0.04" in markdown


def test_control_plane_row_surfaces_ablation_test_matrix_metrics() -> None:
    whole_grid = _load_whole_grid_module()
    stage = {
        "stage_key": "Control Plane",
        "title": "Control Plane",
        "executed_count": 0,
        "report_count": 1,
        "archival_count": 0,
        "deferred_count": 0,
    }
    matrix_manifest = {
        "status": "passed",
        "selected_group_count": 7,
        "selected_test_count": 23,
        "duration_seconds": 12.5,
        "metrics": {
            "selected_test_count": 23.0,
            "matrix_unique_test_count": 50.0,
            "matrix_discovered_test_count": 50.0,
            "matrix_unlisted_test_count": 0.0,
            "matrix_extra_listed_test_count": 0.0,
            "pytest_returncode": 0.0,
            "pytest_passed": 1.0,
            "pytest_executed": 1.0,
        },
    }

    row = whole_grid._control_plane_row(
        stage,
        Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/run/ablation_history_manifest.json"),
        Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_program_spine/run/ablation_program_spine_manifest.json"),
        Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_test_matrix/run/ablation_test_matrix_manifest.json"),
        None,
        None,
        matrix_manifest,
    )

    assert row["surface_kind"] == "control_plane_manifest"
    assert "ablation_test_matrix" in " ".join(row["supplemental_paths"])
    assert row["headline_metrics"]["pytest_passed"] == 1.0
    assert row["headline_metrics"]["pytest_executed"] == 1.0
    assert row["headline_metrics"]["selected_test_count"] == 23.0
    assert row["headline_metrics"]["matrix_unique_test_count"] == 50.0
    assert row["headline_metrics"]["matrix_discovered_test_count"] == 50.0
    assert row["headline_metrics"]["matrix_unlisted_test_count"] == 0.0
    assert row["headline_metrics"]["matrix_extra_listed_test_count"] == 0.0
    assert row["headline_metrics"]["pytest_returncode"] == 0.0
    assert row["headline_metrics"]["test_matrix_status_passed"] == 1.0
