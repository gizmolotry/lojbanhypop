from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .control_plane.artifact_io import (
    latest_json as _artifact_latest_json,
    latest_named_manifest as _artifact_latest_named_manifest,
    path_allowed_for_discovery as _artifact_path_allowed_for_discovery,
    read_json_optional as _artifact_read_json_optional,
    repo_string as _artifact_repo_string,
)
from .experiment_taxonomy import build_comparison_index, load_taxonomy_config
from .m19.family import M19_REGISTRY
from .m20.family import M20_REGISTRY
from .m21.family import M21_REGISTRY
from .m22.family import M22_REGISTRY
from .m23.family import M23_REGISTRY
from .repo_paths import REPO_ROOT, repo_relative
from .series_contract import series_metadata

try:
    from .m24.family import M24_REGISTRY
except ModuleNotFoundError:  # pragma: no cover - M24 core registry may land in a parallel branch.
    M24_REGISTRY: dict[str, dict[str, Any]] = {}


DIRECT_UNIFIED_EVAL_VERSION = "1.0"
DIRECT_UNIFIED_EVAL_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "direct_unified_eval"
)

M22_REQUIRED_PROMOTION_GATES = {
    "clean_accuracy_not_collapsed",
    "trace_reconstruction_preserved",
    "judri_causality_preserved",
    "semantic_judri_causality_preserved",
    "semantic_strict_improves_control",
    "semantic_worst_improves_control",
    "clean_drop_within_tolerance",
    "semantic_training_exposed",
    "semantic_coverage_metrics_available",
    "relation_ood_metrics_available",
    "relation_ood_surfaces_complete",
    "relation_ood_surfaces_unseen_in_training",
    "relation_ood_judri_causality_preserved",
    "relation_ood_score_positive",
    "m22_candidate_cell_evidence_present",
    "m22_audit_candidate_cell_evidence_present",
    "m22_blended_candidate_audit_evidence_present",
    "explicit_m21_control_present",
}

KEY_METRICS = (
    "strict_accuracy",
    "overall_accuracy",
    "overall_phrase_accuracy",
    "avg_tokens",
    "accuracy_per_token",
    "lift_vs_base",
    "lift_vs_en_cot",
    "lift_vs_zh_cot",
    "lift_vs_random",
    "retention_vs_en_cot",
    "token_ratio_vs_en_cot",
    "compression_adjusted_retention",
    "premature_stop_rate",
    "max_cap_hit_rate",
    "scratchpad_bleed_rate",
    "caa_manifold_entanglement_score",
    "base_accuracy",
    "en_cot_accuracy",
    "zh_cot_accuracy",
    "static_m19_3_accuracy",
    "audit_qformer_accuracy",
    "audit_random_accuracy",
    "audit_lift_vs_base",
    "audit_lift_vs_random",
    "full_accuracy",
    "purged_accuracy",
    "overlap_accuracy",
    "masked_accuracy",
    "overlap_gap",
    "masked_collapse_gap",
    "purged_lift_vs_scratchpad_only",
    "mean_accuracy",
    "std_accuracy",
    "mean_avg_tokens",
    "mean_audit_qformer_accuracy",
    "best_mean_accuracy",
    "best_stable_seed_rate",
    "recovered_seed_count",
    "entity_accuracy",
    "entity_renamed_accuracy",
    "format_accuracy",
    "numeric_accuracy",
    "typed_family_accuracy",
    "arity_violation_rate",
    "masked_pointer_zero_rate",
    "family_slot_entropy",
    "symbolic_trace_alignment",
    "predicate_pointer_radial_gap",
    "family_radius_violation_rate",
    "hyperbolic_geodesic_margin",
    "hyperbolic_projection_clip_rate",
    "hyperbolic_tangent_handoff_norm_mean",
    "hyperbolic_tangent_handoff_finite_rate",
    "judri_bridge_gate_enabled",
    "judri_bridge_gate_mean",
    "judri_bridge_gate_active_mean",
    "judri_bridge_gate_silenced_predicate_energy_mean",
    "integrity_overlap_flag",
    "integrity_mask_flag",
    "integrity_audit_flag",
    "mean_intervention_delta_gold",
    "resume_first_token_accuracy",
    "english_fluency_score",
    "loop_rate",
    "contamination_rate",
    "headline_accuracy",
    "headline_macro_f1",
    "logical_accuracy",
    "held_out_accuracy",
    "constraint_scope",
    "constraint_identity",
    "dictionary_coverage",
    "dictionary_precedence_violation_rate",
    "oov_predicate_rate",
    "factorized_exact_accuracy",
    "domain_accuracy",
    "polarity_accuracy",
    "relation_type_accuracy",
    "arity_accuracy",
    "role_schema_accuracy",
    "predicate_identity_stability",
    "counterfactual_quotient_consistency",
    "quotient_collision_rate",
    "brivi_gate_accuracy",
    "brivi_formation_valid_rate",
    "brivi_lock_violation_rate",
    "ungrounded_predicate_energy_mean",
    "synthetic_world_accuracy",
    "synthetic_world_generalization_accuracy",
    "soft_dictionary_entropy",
    "soft_hard_dictionary_agreement",
    "hard_dictionary_activation_rate",
    "hard_code_utilization_count",
    "active_code_fraction",
    "lock_pass_rate",
    "entity_leakage_proxy",
    "bridi_trace_exact_accuracy",
    "gismu_accuracy",
    "cmavo_accuracy",
    "judri_binding_accuracy",
    "frame_count_mae",
    "stop_accuracy",
    "mean_active_frames",
    "frame_count_entropy",
    "active_gismu_count",
    "active_cmavo_count",
    "active_code_fraction_reachable",
    "active_code_fraction_total",
    "no_cmavo_accuracy",
    "no_judri_accuracy",
    "gismu_only_accuracy",
    "random_trace_accuracy",
    "scratchpad_only_accuracy",
    "frame_drop_delta",
    "cmavo_causal_delta",
    "judri_causal_delta",
    "trace_tokens",
    "accuracy_per_trace_token",
    "actual_bridge_transfer_score",
    "loss_pointer_necessity",
    "pointer_necessity_gap",
    "m19_gauntlet_worst_surface_accuracy",
    "m19_gauntlet_order_sensitivity_spread",
    "adversarial_strict_accuracy",
    "adversarial_bridi_trace_exact_accuracy",
    "adversarial_gismu_accuracy",
    "adversarial_cmavo_accuracy",
    "adversarial_judri_binding_accuracy",
    "adversarial_no_judri_accuracy",
    "adversarial_judri_causal_delta",
    "adversarial_worst_surface_accuracy",
    "adversarial_oov_token_rate",
    "adversarial_oov_synonym_accuracy",
    "adversarial_oov_synonym_trace_exact_accuracy",
    "adversarial_train_fraction",
    "mean_adversarial_train_fraction",
    "adversarial_training_exposure_rate",
    "semantic_coverage_strict_accuracy",
    "semantic_coverage_worst_surface_accuracy",
    "semantic_coverage_judri_causal_delta",
    "semantic_coverage_oov_token_rate",
    "semantic_coverage_oov_synonym_accuracy",
    "semantic_coverage_oov_synonym_trace_exact_accuracy",
    "semantic_coverage_surface_seed_std_max",
    "semantic_coverage_surface_seed_min_accuracy",
    "semantic_coverage_metrics_present",
    "semantic_coverage_training_exposure_rate",
    "semantic_coverage_train_fraction",
    "semantic_coverage_surface_count",
    "semantic_isolation_cell_count",
    "m22_relation_ood_metrics_present",
    "m22_relation_ood_strict_accuracy",
    "m22_relation_ood_worst_surface_accuracy",
    "m22_relation_ood_bridi_trace_exact_accuracy",
    "m22_relation_ood_judri_causal_delta",
    "m22_relation_ood_oov_token_rate",
    "m22_relation_ood_surface_count",
    "m22_relation_ood_surface_seed_std_max",
    "m22_relation_ood_surface_seed_min_accuracy",
    "m22_relation_ood_surface_training_overlap_rate",
    "m22_hard_relation_ood_score",
    "semantic_coverage_lexical_shift_effect_strict_accuracy_delta",
    "semantic_coverage_lexical_shift_effect_worst_surface_accuracy_delta",
    "semantic_coverage_lexical_shift_effect_judri_causal_delta_delta",
    "semantic_coverage_role_binding_effect_strict_accuracy_delta",
    "semantic_coverage_role_binding_effect_worst_surface_accuracy_delta",
    "semantic_coverage_role_binding_effect_judri_causal_delta_delta",
    "semantic_coverage_combined_effect_strict_accuracy_delta",
    "semantic_coverage_combined_effect_worst_surface_accuracy_delta",
    "semantic_coverage_combined_effect_judri_causal_delta_delta",
    "semantic_coverage_fraction_effect_strict_accuracy_delta",
    "semantic_coverage_fraction_effect_worst_surface_accuracy_delta",
    "semantic_coverage_fraction_effect_judri_causal_delta_delta",
    "semantic_coverage_role_curriculum_effect_strict_accuracy_delta",
    "semantic_coverage_role_curriculum_effect_worst_surface_accuracy_delta",
    "semantic_coverage_role_curriculum_effect_judri_causal_delta_delta",
    "semantic_coverage_role_swap_effect_strict_accuracy_delta",
    "semantic_coverage_role_swap_effect_worst_surface_accuracy_delta",
    "semantic_coverage_role_swap_effect_judri_causal_delta_delta",
    "semantic_coverage_role_curriculum_fraction_effect_strict_accuracy_delta",
    "semantic_coverage_role_curriculum_fraction_effect_worst_surface_accuracy_delta",
    "semantic_coverage_role_curriculum_fraction_effect_judri_causal_delta_delta",
    "gauntlet_integrity_full_accuracy",
    "gauntlet_integrity_purged_accuracy",
    "gauntlet_integrity_masked_accuracy",
    "gauntlet_kill_worst_surface_accuracy",
    "gauntlet_order_accuracy_spread",
    "m22_candidate_cell_count",
    "m22_candidate_cells_present",
    "m22_suite_candidate_cell_count",
    "m22_audit_candidate_cell_count",
    "m22_audit_candidate_cells_present",
    "m22_blended_candidate_cell_count",
    "m22_audit_blended_candidate_cell_count",
    "m22_audit_blended_candidate_present",
    "m22_semantic_generalization_score",
    "m22_semantic_strict_delta_vs_m21_control",
    "m22_semantic_worst_delta_vs_m21_control",
    "m22_clean_accuracy_drop_vs_m21_control",
    "m22_judri_delta_drop_vs_m21_control",
    "m22_promotion_gate_pass_rate",
    "m22_promotion_candidate",
    "clean_accuracy",
    "worst_surface_accuracy",
    "decoy_relation_ood_accuracy",
    "relevance_top1_accuracy",
    "relevance_margin",
    "loss_trace_exact_surrogate",
    "trace_exact_surrogate_weight",
    "oracle_relevance_accuracy",
    "random_relevance_accuracy",
    "no_relevance_accuracy",
    "decoy_only_accuracy",
    "oracle_relevance_delta",
    "random_relevance_delta",
    "decoy_only_delta",
    "m23_router_decoy_lift_vs_scale",
    "m23_router_worst_surface_lift_vs_scale",
    "m23_oracle_relevance_lift",
    "m23_trace_punish_trace_exact_lift_vs_scale",
    "m23_trace_punish_decoy_delta_vs_scale",
    "m23_trace_punish_strict_delta_vs_scale",
    "predicted_trace_accuracy",
    "oracle_trace_accuracy",
    "random_trace_accuracy",
    "shuffled_trace_accuracy",
    "zero_trace_accuracy",
    "prompt_only_accuracy",
    "advisor_vs_prompt_delta",
    "m24_strict_delta_vs_prompt_only",
    "predicted_vs_random_delta",
    "predicted_vs_shuffled_delta",
    "oracle_trained_oracle_trace_accuracy",
    "oracle_trained_predicted_trace_accuracy",
    "oracle_trained_random_trace_accuracy",
    "oracle_trained_trace_delta",
    "predicted_trace_gap_to_oracle_upper_bound",
    "cross_advisor_oracle_gap",
    "substrate_claim_score",
    "m24_promotion_gate_pass_rate",
    "m24_promotion_candidate",
    "generator_parameter_max_delta_after_advisor",
    "generator_parameters_unchanged_after_advisor",
    "substrate_token_count",
    "mean_substrate_token_count",
    "avg_substrate_token_count",
    "substrate_tokens",
    "mean_substrate_tokens",
    "avg_substrate_tokens",
    "reference_token_count",
    "mean_reference_token_count",
    "reference_tokens",
    "mean_reference_tokens",
    "baseline_token_count",
    "mean_baseline_token_count",
    "baseline_tokens",
    "mean_baseline_tokens",
    "compression_ratio",
    "mean_compression_ratio",
    "packed_symbol_to_prompt_ratio",
    "prompt_to_packed_symbol_ratio",
    "packed_to_prompt_ratio",
    "prompt_to_packed_ratio",
    "packed_symbol_compression_ratio",
    "mdl_weight",
    "token_reduction_ratio",
    "mean_token_reduction_ratio",
    "token_ratio_vs_m23",
    "compression_lift_vs_m23",
    "compression_adjusted_strict_accuracy",
    "strict_accuracy_per_substrate_token",
    "m24_gate_packed_trace_shorter_than_prompt",
    "phrase_accuracy",
    "phrase_exact_accuracy",
)

_REFERENCE_ROOTS: dict[str, list[Path]] = {
    "J": [
        REPO_ROOT / "runs" / "j_series",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw",
    ],
    "L": [
        REPO_ROOT / "runs" / "l_series",
        REPO_ROOT / "artifacts" / "runs" / "models" / "frozen_manifolds",
    ],
    "M11": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m_bridge_ablation_test_suite",
    ],
    "M14": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m14_symbiote_scratchpad",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m14_5_decompressor",
    ],
    "M18": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m18_controller_family",
    ],
    "M19": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_mainline_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_dynamic_pacing_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_isolation_grid",
    ],
    "M20": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m20_dictionary_first_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m20_lock_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m20_predicate_induction",
    ],
    "M21": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_dynamic_bridi_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_synthetic_assay_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_actual_bridge_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_lock_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_pointer_necessity_microgrid",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_gauntlet_suite",
    ],
    "M22": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m22_semantic_generalization",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "direct_unified_eval",
    ],
    "M23": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m23_relevance_suite",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "direct_unified_eval",
    ],
    "M24": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m24_substrate_compression",
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "direct_unified_eval",
    ],
    "M10": [
        REPO_ROOT / "archive" / "results" / "m10" / "active",
    ],
    "M9": [
        REPO_ROOT / "archive" / "results" / "m9" / "active",
    ],
    "M8": [
        REPO_ROOT / "archive" / "results" / "m8" / "active",
    ],
    "M7": [
        REPO_ROOT / "archive" / "results" / "m7" / "active",
    ],
    "M6": [
        REPO_ROOT / "archive" / "results" / "m6" / "20260314",
        REPO_ROOT / "archive" / "results" / "m6_1" / "active",
        REPO_ROOT / "archive" / "results" / "m6_2" / "active",
        REPO_ROOT / "archive" / "results" / "m6_3" / "active",
        REPO_ROOT / "archive" / "results" / "m6_6" / "active",
    ],
    "M5": [
        REPO_ROOT / "archive" / "results" / "m5" / "20260313",
    ],
    "M4": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube",
    ],
    "M3": [
        REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube",
    ],
}


def discover_m19_surfaces(
    *,
    track: str = "M19",
    benchmark_report_path: Path | None = None,
    audit_report_path: Path | None = None,
    integrity_report_path: Path | None = None,
    replication_report_path: Path | None = None,
    stability_report_path: Path | None = None,
    kill_test_report_path: Path | None = None,
    dictionary_audit_report_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    track_key = _track_key(track)
    benchmark_path = benchmark_report_path or _latest_named_manifest(
        REPO_ROOT / M19_REGISTRY[track_key]["output_roots"]["benchmark"],
        M19_REGISTRY[track_key]["report_names"]["benchmark"],
    )
    audit_path = audit_report_path
    if audit_path is None and track_key == "M19":
        audit_path = _latest_named_manifest(
            REPO_ROOT / M19_REGISTRY["M19"]["output_roots"]["audit"],
            M19_REGISTRY["M19"]["report_names"]["audit"],
        )
    integrity_path = integrity_report_path
    if integrity_path is None and track_key == "M19":
        integrity_path = _latest_named_manifest(
            REPO_ROOT / M19_REGISTRY["M19"]["output_roots"]["integrity"],
            M19_REGISTRY["M19"]["report_names"]["integrity"],
        )
    replication_path = replication_report_path
    if replication_path is None and track_key == "M19":
        replication_path = _latest_named_manifest(
            REPO_ROOT / M19_REGISTRY["M19"]["output_roots"]["replication"],
            M19_REGISTRY["M19"]["report_names"]["replication"],
        )
    stability_path = stability_report_path
    if stability_path is None and track_key == "M19":
        stability_path = _latest_named_manifest(
            REPO_ROOT / M19_REGISTRY["M19"]["output_roots"]["stability_microgrid"],
            M19_REGISTRY["M19"]["report_names"]["stability_microgrid"],
        )
    kill_test_path = kill_test_report_path
    if kill_test_path is None and track_key == "M19":
        kill_test_path = _latest_named_manifest(
            REPO_ROOT / M19_REGISTRY["M19"]["output_roots"]["kill_tests"],
            M19_REGISTRY["M19"]["report_names"]["kill_tests"],
        )
    dictionary_audit_path = dictionary_audit_report_path
    if (
        dictionary_audit_path is None
        and track_key in M19_REGISTRY
        and "dictionary_audit" in M19_REGISTRY[track_key].get("output_roots", {})
        and "dictionary_audit" in M19_REGISTRY[track_key].get("report_names", {})
    ):
        dictionary_audit_path = _latest_named_manifest(
            REPO_ROOT / M19_REGISTRY[track_key]["output_roots"]["dictionary_audit"],
            M19_REGISTRY[track_key]["report_names"]["dictionary_audit"],
        )

    surfaces: dict[str, dict[str, Any]] = {
        "benchmark": _surface_record("benchmark", benchmark_path),
        "audit": _surface_record("audit", audit_path),
        "integrity": _surface_record("integrity", integrity_path),
        "replication": _surface_record("replication", replication_path),
        "stability_microgrid": _surface_record("stability_microgrid", stability_path),
        "kill_tests": _surface_record("kill_tests", kill_test_path),
        "dictionary_audit": _surface_record("dictionary_audit", dictionary_audit_path),
    }
    return surfaces


def discover_m20_surfaces(
    *,
    suite_report_path: Path | None = None,
    lock_report_path: Path | None = None,
    induction_report_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    registry = M20_REGISTRY["M20"]
    suite_path = suite_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["suite"],
        registry["report_names"]["suite"],
    )
    lock_path = lock_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["lock_suite"],
        registry["report_names"]["lock_suite"],
    )
    induction_path = induction_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["predicate_induction"],
        registry["report_names"]["predicate_induction"],
    )
    return {
        "suite": _surface_record("suite", suite_path),
        "lock_suite": _surface_record("lock_suite", lock_path),
        "predicate_induction": _surface_record("predicate_induction", induction_path),
    }


def discover_m21_surfaces(
    *,
    suite_report_path: Path | None = None,
    synthetic_assay_report_path: Path | None = None,
    actual_bridge_report_path: Path | None = None,
    lock_report_path: Path | None = None,
    pointer_microgrid_report_path: Path | None = None,
    gauntlet_report_path: Path | None = None,
    adversarial_audit_report_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    registry = M21_REGISTRY["M21"]
    suite_path = suite_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["suite"],
        registry["report_names"]["suite"],
    )
    synthetic_path = synthetic_assay_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["synthetic_assay"],
        registry["report_names"]["synthetic_assay"],
    )
    actual_path = actual_bridge_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["actual_bridge"],
        registry["report_names"]["actual_bridge"],
    )
    lock_path = lock_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["lock_suite"],
        registry["report_names"]["lock_suite"],
    )
    pointer_path = pointer_microgrid_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["pointer_microgrid"],
        registry["report_names"]["pointer_microgrid"],
    )
    gauntlet_path = gauntlet_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["gauntlet"],
        registry["report_names"]["gauntlet"],
    )
    adversarial_path = adversarial_audit_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["adversarial_audit"],
        registry["report_names"]["adversarial_audit"],
    )
    return {
        "suite": _surface_record("suite", suite_path),
        "synthetic_assay": _surface_record("synthetic_assay", synthetic_path),
        "actual_bridge": _surface_record("actual_bridge", actual_path),
        "lock_suite": _surface_record("lock_suite", lock_path),
        "pointer_microgrid": _surface_record("pointer_microgrid", pointer_path),
        "gauntlet": _surface_record("gauntlet", gauntlet_path),
        "adversarial_audit": _surface_record("adversarial_audit", adversarial_path),
    }


def discover_m22_surfaces(
    *,
    generalization_report_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    registry = M22_REGISTRY["M22"]
    generalization_path = generalization_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["generalization"],
        registry["report_names"]["generalization"],
    )
    return {
        "generalization": _surface_record("generalization", generalization_path),
    }


def discover_m23_surfaces(
    *,
    relevance_report_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    registry = M23_REGISTRY["M23"]
    relevance_path = relevance_report_path or _latest_named_manifest(
        REPO_ROOT / registry["output_roots"]["suite"],
        registry["report_names"]["suite"],
    )
    return {
        "relevance_suite": _surface_record("relevance_suite", relevance_path),
    }


def discover_m24_surfaces(
    *,
    compression_report_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    registry = M24_REGISTRY.get("M24", {})
    output_roots = registry.get("output_roots", {}) if isinstance(registry, dict) else {}
    report_names = registry.get("report_names", {}) if isinstance(registry, dict) else {}
    root = output_roots.get("substrate_compression") or output_roots.get("suite")
    report_name = (
        report_names.get("substrate_compression")
        or report_names.get("suite")
        or "m24_substrate_compression_report.json"
    )
    compression_path = compression_report_path
    if compression_path is None:
        compression_path = _latest_named_manifest(
            REPO_ROOT / str(root or "artifacts/runs/telemetry/raw/ablation/hypercube/m24_substrate_compression"),
            str(report_name),
        )
    return {
        "substrate_compression": _surface_record("substrate_compression", compression_path),
    }


def build_direct_unified_eval_manifest(
    *,
    family_key: str,
    track: str = "",
    benchmark_report_path: Path | None = None,
    audit_report_path: Path | None = None,
    integrity_report_path: Path | None = None,
    replication_report_path: Path | None = None,
    stability_report_path: Path | None = None,
    kill_test_report_path: Path | None = None,
    dictionary_audit_report_path: Path | None = None,
    m20_suite_report_path: Path | None = None,
    m20_lock_report_path: Path | None = None,
    m20_induction_report_path: Path | None = None,
    m21_suite_report_path: Path | None = None,
    m21_synthetic_assay_report_path: Path | None = None,
    m21_actual_bridge_report_path: Path | None = None,
    m21_lock_report_path: Path | None = None,
    m21_pointer_microgrid_report_path: Path | None = None,
    m21_gauntlet_report_path: Path | None = None,
    m21_adversarial_audit_report_path: Path | None = None,
    m22_generalization_report_path: Path | None = None,
    m23_relevance_report_path: Path | None = None,
    m24_compression_report_path: Path | None = None,
    history_manifest_path: Path | None = None,
    taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or load_taxonomy_config()
    comparison_index = build_comparison_index(taxonomy)
    if family_key not in comparison_index:
        raise ValueError(f"Unknown family_key '{family_key}'.")
    contract = comparison_index[family_key]
    if family_key == "M19":
        resolved_track = _track_key(track or family_key)
        direct_surfaces = discover_m19_surfaces(
            track=resolved_track,
            benchmark_report_path=benchmark_report_path,
            audit_report_path=audit_report_path,
            integrity_report_path=integrity_report_path,
            replication_report_path=replication_report_path,
            stability_report_path=stability_report_path,
            kill_test_report_path=kill_test_report_path,
            dictionary_audit_report_path=dictionary_audit_report_path,
        )
    elif family_key == "M20":
        resolved_track = str(track or "M20.1")
        direct_surfaces = discover_m20_surfaces(
            suite_report_path=m20_suite_report_path,
            lock_report_path=m20_lock_report_path,
            induction_report_path=m20_induction_report_path,
        )
    elif family_key == "M21":
        resolved_track = str(track or "M21.1")
        direct_surfaces = discover_m21_surfaces(
            suite_report_path=m21_suite_report_path,
            synthetic_assay_report_path=m21_synthetic_assay_report_path,
            actual_bridge_report_path=m21_actual_bridge_report_path,
            lock_report_path=m21_lock_report_path,
            pointer_microgrid_report_path=m21_pointer_microgrid_report_path,
            gauntlet_report_path=m21_gauntlet_report_path,
            adversarial_audit_report_path=m21_adversarial_audit_report_path,
        )
    elif family_key == "M22":
        resolved_track = str(track or "M22")
        direct_surfaces = discover_m22_surfaces(
            generalization_report_path=m22_generalization_report_path,
        )
    elif family_key == "M23":
        resolved_track = str(track or "M23")
        direct_surfaces = discover_m23_surfaces(
            relevance_report_path=m23_relevance_report_path,
        )
    elif family_key == "M24":
        resolved_track = str(track or "M24")
        direct_surfaces = discover_m24_surfaces(
            compression_report_path=m24_compression_report_path,
        )
    else:
        raise NotImplementedError(f"Direct unified eval is currently implemented for family '{family_key}' only.")

    benchmark_payload = direct_surfaces.get("benchmark", {}).get("payload")
    audit_payload = direct_surfaces.get("audit", {}).get("payload")
    integrity_payload = direct_surfaces.get("integrity", {}).get("payload")
    replication_payload = direct_surfaces.get("replication", {}).get("payload")
    stability_payload = direct_surfaces.get("stability_microgrid", {}).get("payload")
    kill_test_payload = direct_surfaces.get("kill_tests", {}).get("payload")
    dictionary_audit_payload = direct_surfaces.get("dictionary_audit", {}).get("payload")
    m20_suite_payload = direct_surfaces.get("suite", {}).get("payload") if family_key == "M20" else None
    m20_lock_payload = direct_surfaces.get("lock_suite", {}).get("payload") if family_key == "M20" else None
    m20_induction_payload = direct_surfaces.get("predicate_induction", {}).get("payload") if family_key == "M20" else None
    m21_suite_payload = direct_surfaces.get("suite", {}).get("payload") if family_key == "M21" else None
    m21_synthetic_payload = direct_surfaces.get("synthetic_assay", {}).get("payload")
    m21_actual_payload = direct_surfaces.get("actual_bridge", {}).get("payload")
    m21_lock_payload = direct_surfaces.get("lock_suite", {}).get("payload") if family_key == "M21" else None
    m21_pointer_payload = direct_surfaces.get("pointer_microgrid", {}).get("payload") if family_key == "M21" else None
    m21_gauntlet_payload = direct_surfaces.get("gauntlet", {}).get("payload") if family_key == "M21" else None
    m21_adversarial_payload = direct_surfaces.get("adversarial_audit", {}).get("payload") if family_key == "M21" else None
    m22_generalization_payload = direct_surfaces.get("generalization", {}).get("payload") if family_key == "M22" else None
    m23_relevance_payload = direct_surfaces.get("relevance_suite", {}).get("payload") if family_key == "M23" else None
    m24_compression_payload = direct_surfaces.get("substrate_compression", {}).get("payload") if family_key == "M24" else None
    historical_references = _resolve_historical_family_references(contract, history_manifest_path)
    comparison_targets = _resolve_comparison_targets(contract, history_manifest_path)
    reference_surface_index = _build_reference_surface_index(historical_references, comparison_targets)
    contract_results = _evaluate_contracts(
        family_key=family_key,
        required_test_contracts=contract.get("required_test_contracts", []),
        benchmark_payload=benchmark_payload,
        audit_payload=audit_payload,
        integrity_payload=integrity_payload,
        replication_payload=replication_payload,
        stability_payload=stability_payload,
        kill_test_payload=kill_test_payload,
        dictionary_audit_payload=dictionary_audit_payload,
        m20_suite_payload=m20_suite_payload,
        m20_lock_payload=m20_lock_payload,
        m20_induction_payload=m20_induction_payload,
        m21_suite_payload=m21_suite_payload,
        m21_synthetic_payload=m21_synthetic_payload,
        m21_actual_payload=m21_actual_payload,
        m21_lock_payload=m21_lock_payload,
        m21_pointer_payload=m21_pointer_payload,
        m21_gauntlet_payload=m21_gauntlet_payload,
        m21_adversarial_payload=m21_adversarial_payload,
        m22_generalization_payload=m22_generalization_payload,
        m23_relevance_payload=m23_relevance_payload,
        m24_compression_payload=m24_compression_payload,
        reference_surface_index=reference_surface_index,
    )

    headline_metrics = _build_headline_metrics(
        benchmark_payload,
        audit_payload,
        integrity_payload,
        replication_payload,
        stability_payload,
        kill_test_payload,
        dictionary_audit_payload,
        m20_suite_payload=m20_suite_payload,
        m20_lock_payload=m20_lock_payload,
        m20_induction_payload=m20_induction_payload,
        m21_suite_payload=m21_suite_payload,
        m21_synthetic_payload=m21_synthetic_payload,
        m21_actual_payload=m21_actual_payload,
        m21_lock_payload=m21_lock_payload,
        m21_pointer_payload=m21_pointer_payload,
        m21_gauntlet_payload=m21_gauntlet_payload,
        m21_adversarial_payload=m21_adversarial_payload,
        m22_generalization_payload=m22_generalization_payload,
        m23_relevance_payload=m23_relevance_payload,
        m24_compression_payload=m24_compression_payload,
    )
    direct_report_paths = {
        name: surface["path"]
        for name, surface in direct_surfaces.items()
        if surface.get("path")
    }
    all_paths = [path for path in direct_report_paths.values()]
    all_paths.extend(
        row["anchor_path"]
        for row in historical_references
        if row.get("anchor_path")
    )
    all_paths.extend(
        row["anchor_path"]
        for row in comparison_targets
        if row.get("anchor_path")
    )

    notes = [
        "Direct unified eval separates direct checkpoint-backed surfaces from inherited historical comparison obligations.",
        "Legacy J and L comparators are carried forward as reference anchors unless a family-specific direct adapter exists.",
    ]
    if family_key == "M19" and resolved_track != "M19":
        notes.append(f"Track {resolved_track} is evaluated under the M19 family contract.")
    if family_key == "M20":
        notes.append("M20 direct surfaces are dictionary-first substrate reports, not downstream English bridge evaluations.")
    if family_key == "M21":
        notes.append("M21 direct surfaces combine dynamic bridi synthetic assay, lock-suite, minimal actual bridge, and M19-style gauntlet adapter reports.")
    if family_key == "M22":
        notes.append("M22 direct surfaces evaluate semantic coverage generalization over fixed M21 dynamic bridi controls.")
    if family_key == "M23":
        notes.append("M23 direct surfaces test whether an explicit frame relevance selector beats M22-style scale on decoy relation OOD.")
    if family_key == "M24":
        notes.append(
            "M24 direct surfaces test M24.1 matched trace corruption and compression pressure; "
            "strict accuracy remains canonical and phrase accuracy is diagnostic only."
        )

    manifest = {
        "schema_version": DIRECT_UNIFIED_EVAL_VERSION,
        "generated_utc": _now_utc_iso(),
        "series": series_metadata("M", f"{family_key}.direct_unified_eval", "scripts/control_plane/run_direct_unified_eval.py"),
        "family_key": family_key,
        "track": resolved_track,
        "comparison_contract": {
            "family_key": contract.get("family_key"),
            "compare_within_family": bool(contract.get("compare_within_family", False)),
            "compare_against_ancestors": bool(contract.get("compare_against_ancestors", False)),
            "inherit_required_test_contracts": bool(contract.get("inherit_required_test_contracts", False)),
            "historical_comparison_families": list(contract.get("historical_comparison_families", [])),
            "required_test_contract_ids": list(contract.get("required_test_contract_ids", [])),
            "comparison_targets": list(contract.get("comparison_targets", [])),
        },
        "direct_surfaces": {
            name: {
                "status": surface["status"],
                "path": surface["path"],
                "metrics_digest": _extract_metric_digest(surface["payload"]),
            }
            for name, surface in direct_surfaces.items()
        },
        "contract_results": contract_results,
        "historical_family_references": historical_references,
        "comparison_targets_resolved": comparison_targets,
        "headline_metrics": headline_metrics,
        "source_paths": _unique_strings(all_paths),
        "notes": notes,
    }
    return manifest


def render_direct_unified_eval_markdown(manifest: dict[str, Any]) -> str:
    family_key = str(manifest.get("family_key", ""))
    track = str(manifest.get("track", ""))
    lines = [
        f"# Direct Unified Eval: {family_key} ({track})",
        "",
        "## Headline",
        "",
    ]
    headline = manifest.get("headline_metrics", {})
    if isinstance(headline, dict) and headline:
        for key, value in headline.items():
            lines.append(f"- `{key}`: {_format_scalar(value)}")
    else:
        lines.append("- no direct headline metrics resolved")

    lines.extend(["", "## Contract Results", ""])
    for row in manifest.get("contract_results", []):
        lines.append(
            f"- `{row.get('test_id')}`: status={row.get('status')}, provenance={row.get('provenance')}, "
            f"surface={row.get('surface')}"
        )
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict) and metrics:
            preview = ", ".join(f"{k}={_format_scalar(v)}" for k, v in list(metrics.items())[:5])
            lines.append(f"  metrics: {preview}")
        notes = row.get("notes", [])
        if isinstance(notes, list):
            for note in notes[:2]:
                lines.append(f"  note: {note}")

    lines.extend(["", "## Historical References", ""])
    refs = manifest.get("historical_family_references", [])
    if not refs:
        lines.append("- none")
    else:
        for row in refs:
            lines.append(
                f"- `{row.get('family')}`: status={row.get('status')}, provenance={row.get('provenance')}, "
                f"path={row.get('anchor_path') or 'n/a'}"
            )

    lines.extend(["", "## Comparison Targets", ""])
    targets = manifest.get("comparison_targets_resolved", [])
    if not targets:
        lines.append("- none")
    else:
        for row in targets:
            lines.append(
                f"- `{row.get('target')}` ({row.get('kind')}): status={row.get('status')}, "
                f"path={row.get('anchor_path') or 'n/a'}"
            )
    lines.append("")
    return "\n".join(lines)


def _evaluate_contracts(
    *,
    family_key: str,
    required_test_contracts: list[dict[str, Any]],
    benchmark_payload: dict[str, Any] | None,
    audit_payload: dict[str, Any] | None,
    integrity_payload: dict[str, Any] | None,
    replication_payload: dict[str, Any] | None,
    stability_payload: dict[str, Any] | None,
    kill_test_payload: dict[str, Any] | None,
    dictionary_audit_payload: dict[str, Any] | None,
    m20_suite_payload: dict[str, Any] | None = None,
    m20_lock_payload: dict[str, Any] | None = None,
    m20_induction_payload: dict[str, Any] | None = None,
    m21_suite_payload: dict[str, Any] | None = None,
    m21_synthetic_payload: dict[str, Any] | None = None,
    m21_actual_payload: dict[str, Any] | None = None,
    m21_lock_payload: dict[str, Any] | None = None,
    m21_pointer_payload: dict[str, Any] | None = None,
    m21_gauntlet_payload: dict[str, Any] | None = None,
    m21_adversarial_payload: dict[str, Any] | None = None,
    m22_generalization_payload: dict[str, Any] | None = None,
    m23_relevance_payload: dict[str, Any] | None = None,
    m24_compression_payload: dict[str, Any] | None = None,
    reference_surface_index: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    reference_surface_index = reference_surface_index or {}
    rows: list[dict[str, Any]] = []
    for contract in required_test_contracts:
        test_id = str(contract.get("test_id", "")).strip()
        if family_key == "M19":
            row = _evaluate_m19_contract(
                test_id,
                contract,
                benchmark_payload,
                audit_payload,
                integrity_payload,
                replication_payload,
                stability_payload,
                kill_test_payload,
                dictionary_audit_payload,
            )
            rows.append(_attach_reference_surface(row, contract, reference_surface_index))
            continue
        if family_key == "M20":
            row = _evaluate_m20_contract(test_id, contract, m20_suite_payload, m20_lock_payload, m20_induction_payload)
            rows.append(_attach_reference_surface(row, contract, reference_surface_index))
            continue
        if family_key == "M21":
            row = _evaluate_m21_contract(
                test_id,
                contract,
                m21_suite_payload,
                m21_synthetic_payload,
                m21_actual_payload,
                m21_lock_payload,
                m21_pointer_payload,
                m21_gauntlet_payload,
                m21_adversarial_payload,
            )
            rows.append(_attach_reference_surface(row, contract, reference_surface_index))
            continue
        if family_key == "M22":
            row = _evaluate_m22_contract(test_id, contract, m22_generalization_payload)
            rows.append(_attach_reference_surface(row, contract, reference_surface_index))
            continue
        if family_key == "M23":
            row = _evaluate_m23_contract(test_id, contract, m23_relevance_payload)
            rows.append(_attach_reference_surface(row, contract, reference_surface_index))
            continue
        if family_key == "M24":
            row = _evaluate_m24_contract(test_id, contract, m24_compression_payload)
            rows.append(_attach_reference_surface(row, contract, reference_surface_index))
            continue
        rows.append(
            {
                "test_id": test_id,
                "surface": contract.get("surface"),
                "status": "unimplemented",
                "provenance": "none",
                "metrics": {},
                "notes": [f"No direct evaluator exists yet for family {family_key} contract {test_id}."],
            }
        )
    return rows


def _evaluate_m24_contract(
    test_id: str,
    contract: dict[str, Any],
    compression_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = _filtered_metrics(_m24_compression_metrics(compression_payload), tuple(contract.get("metrics", [])))
    if not metrics:
        return _missing_contract_row(test_id, contract, f"missing M24 direct surface for {test_id}")
    notes = [
        "M24 contract is evaluated from the M24.1 matched trace corruption and compression-pressure report.",
        "strict_accuracy is canonical; phrase accuracy metrics are diagnostic only.",
    ]
    if float(metrics.get("m24_promotion_candidate", 0.0) or 0.0) < 1.0:
        notes.append("M24 remains explicitly non-promoted unless m24_promotion_candidate=1.0.")
    return {
        "test_id": test_id,
        "surface": contract.get("surface"),
        "status": "available",
        "provenance": "artifact",
        "metrics": metrics,
        "notes": notes,
    }


def _evaluate_m23_contract(
    test_id: str,
    contract: dict[str, Any],
    relevance_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = _filtered_metrics(_m23_suite_metrics(relevance_payload), tuple(contract.get("metrics", [])))
    if not metrics:
        return _missing_contract_row(test_id, contract, f"missing M23 direct surface for {test_id}")
    notes = ["M23 contract is evaluated from the causal relevance router suite over M21/M22 dynamic bridi traces."]
    if test_id == "m23.causal_relevance_router":
        aggregate = relevance_payload.get("aggregate_metrics", {}) if isinstance(relevance_payload, dict) else {}
        conclusion = str(aggregate.get("conclusion", ""))
        notes.append(f"hypothesis interpretation: {conclusion or 'not reported'}")
    return {
        "test_id": test_id,
        "surface": contract.get("surface"),
        "status": "available",
        "provenance": "artifact",
        "metrics": metrics,
        "notes": notes,
    }


def _evaluate_m22_contract(
    test_id: str,
    contract: dict[str, Any],
    generalization_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = {}
    if isinstance(generalization_payload, dict) and isinstance(generalization_payload.get("metrics"), dict):
        metrics = _filtered_metrics(generalization_payload["metrics"], tuple(contract.get("metrics", [])))
    if not metrics:
        return _missing_contract_row(test_id, contract, f"missing M22 direct surface for {test_id}")
    if test_id == "m22.semantic_coverage_generalization":
        report_metrics = generalization_payload.get("metrics", {})
        if float(report_metrics.get("m22_semantic_generalization_score", 0.0) or 0.0) <= 0.0:
            row = _missing_contract_row(test_id, contract, "missing positive M22 semantic generalization score")
            row["metrics"] = metrics
            return row
        if float(report_metrics.get("m22_promotion_candidate", 0.0) or 0.0) < 1.0:
            row = _missing_contract_row(test_id, contract, "M22 semantic generalization gate failed promotion")
            row["metrics"] = metrics
            return row
        promotion_gates = generalization_payload.get("promotion_gates", {})
        if not isinstance(promotion_gates, dict) or not promotion_gates:
            row = _missing_contract_row(test_id, contract, "missing M22 promotion gate evidence")
            row["metrics"] = metrics
            return row
        failed_gates = [str(key) for key, value in promotion_gates.items() if not bool(value)]
        if failed_gates:
            row = _missing_contract_row(test_id, contract, f"M22 promotion report has failed gates: {', '.join(failed_gates)}")
            row["metrics"] = metrics
            return row
        missing_gates = sorted(M22_REQUIRED_PROMOTION_GATES.difference(str(key) for key in promotion_gates))
        if missing_gates:
            row = _missing_contract_row(test_id, contract, f"M22 promotion report is missing gates: {', '.join(missing_gates)}")
            row["metrics"] = metrics
            return row
    return {
        "test_id": test_id,
        "surface": contract.get("surface"),
        "status": "available",
        "provenance": "artifact",
        "metrics": metrics,
        "notes": ["M22 contract is evaluated from the semantic generalization report over fixed M21 controls."],
    }


def _evaluate_m21_contract(
    test_id: str,
    contract: dict[str, Any],
    suite_payload: dict[str, Any] | None,
    synthetic_payload: dict[str, Any] | None,
    actual_payload: dict[str, Any] | None,
    lock_payload: dict[str, Any] | None,
    pointer_payload: dict[str, Any] | None,
    gauntlet_payload: dict[str, Any] | None,
    adversarial_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    suite_metrics = _m21_suite_metrics(suite_payload)
    synthetic_metrics = _m21_suite_metrics(synthetic_payload)
    pointer_metrics = _m21_suite_metrics(pointer_payload)
    actual_metrics = actual_payload.get("metrics", {}) if isinstance(actual_payload, dict) else {}
    lock_metrics = lock_payload.get("metrics", {}) if isinstance(lock_payload, dict) else {}
    gauntlet_metrics = gauntlet_payload.get("metrics", {}) if isinstance(gauntlet_payload, dict) else {}
    adversarial_metrics = _m21_suite_metrics(adversarial_payload)
    merged = {}
    surface = str(contract.get("surface", "")).strip()
    if test_id == "m21.pointer_necessity" or surface == "m21_pointer_necessity_microgrid":
        sources = (pointer_metrics, actual_metrics)
    elif test_id == "m21.m19_gauntlet_port" or surface == "m21_gauntlet_suite":
        sources = (gauntlet_metrics,)
    elif test_id == "m21.adversarial_heldout" or (
        surface == "m21_adversarial_audit" and test_id not in {"m21.adversarial_augmentation", "m21.semantic_coverage"}
    ):
        sources = (adversarial_metrics,)
    elif test_id == "m21.adversarial_augmentation":
        sources = (suite_metrics, actual_metrics, adversarial_metrics)
    elif test_id == "m21.semantic_coverage":
        sources = (suite_metrics, adversarial_metrics)
    elif surface == "m21_dynamic_bridi_suite":
        sources = (suite_metrics,)
    elif surface == "m21_actual_bridge_suite":
        sources = (suite_metrics, actual_metrics)
    elif surface == "m21_synthetic_assay_suite":
        sources = (suite_metrics, synthetic_metrics)
    elif surface == "m21_lock_suite":
        sources = (suite_metrics, lock_metrics)
    else:
        sources = (suite_metrics, synthetic_metrics, actual_metrics, lock_metrics, pointer_metrics, gauntlet_metrics, adversarial_metrics)
    for source in sources:
        if isinstance(source, dict):
            for key, value in source.items():
                if value is not None:
                    merged[key] = value
    metric_keys = tuple(contract.get("metrics", []))
    metrics = _filtered_metrics(merged, metric_keys)
    if test_id == "m21.adversarial_augmentation" and float(adversarial_metrics.get("adversarial_training_exposure_rate", 0.0) or 0.0) <= 0.0:
        row = _missing_contract_row(test_id, contract, "missing M21 adversarial training exposure for augmentation contract")
        row["metrics"] = _filtered_metrics(adversarial_metrics, metric_keys)
        return row
    if test_id == "m21.semantic_coverage" and not _has_semantic_isolation_evidence(adversarial_metrics):
        row = _missing_contract_row(test_id, contract, "missing M21 semantic-coverage isolation evidence for semantic coverage contract")
        row["metrics"] = _filtered_metrics(adversarial_metrics, metric_keys)
        return row
    if test_id == "m21.dynamic_frame_count" and isinstance(suite_payload, dict):
        metrics["cell_count"] = len(suite_payload.get("cells", {})) if isinstance(suite_payload.get("cells"), dict) else 0
    if not metrics:
        return _missing_contract_row(test_id, contract, f"missing M21 direct surface for {test_id}")
    return {
        "test_id": test_id,
        "surface": contract.get("surface"),
        "status": "available",
        "provenance": "artifact",
        "metrics": metrics,
        "notes": ["M21 contract is evaluated from dynamic bridi suite, synthetic assay, lock-suite, actual bridge, adversarial audit, and gauntlet-adapter reports."],
    }


def _has_semantic_isolation_evidence(metrics: dict[str, Any]) -> bool:
    if float(metrics.get("semantic_coverage_training_exposure_rate", 0.0) or 0.0) <= 0.0:
        return False
    if float(metrics.get("semantic_isolation_cell_count", 0.0) or 0.0) < 5.0:
        return False
    required = (
        "semantic_coverage_lexical_shift_effect_strict_accuracy_delta",
        "semantic_coverage_role_binding_effect_strict_accuracy_delta",
        "semantic_coverage_combined_effect_strict_accuracy_delta",
        "semantic_coverage_fraction_effect_strict_accuracy_delta",
    )
    return all(metrics.get(key) is not None for key in required)


def _evaluate_m20_contract(
    test_id: str,
    contract: dict[str, Any],
    suite_payload: dict[str, Any] | None,
    lock_payload: dict[str, Any] | None,
    induction_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    suite_metrics = _m20_suite_metrics(suite_payload)
    lock_metrics = lock_payload.get("metrics", {}) if isinstance(lock_payload, dict) else {}
    induction_metrics = induction_payload.get("metrics", {}) if isinstance(induction_payload, dict) else {}
    merged = {}
    if isinstance(induction_metrics, dict):
        merged.update(induction_metrics)
    if isinstance(suite_metrics, dict):
        merged.update(suite_metrics)
    if isinstance(lock_metrics, dict):
        merged.update(lock_metrics)

    metric_keys = tuple(contract.get("metrics", []))
    metrics = _filtered_metrics(merged, metric_keys)
    if test_id == "m20.synthetic_world_pretraining" and isinstance(suite_payload, dict):
        metrics["cell_count"] = len(suite_payload.get("cells", {})) if isinstance(suite_payload.get("cells"), dict) else 0
    if not metrics:
        return _missing_contract_row(test_id, contract, f"missing M20 direct surface for {test_id}")
    return {
        "test_id": test_id,
        "surface": contract.get("surface"),
        "status": "available",
        "provenance": "artifact",
        "metrics": metrics,
        "notes": ["M20 contract is evaluated from dictionary-first suite, lock-suite, and predicate-induction reports."],
    }


def _evaluate_m19_contract(
    test_id: str,
    contract: dict[str, Any],
    benchmark_payload: dict[str, Any] | None,
    audit_payload: dict[str, Any] | None,
    integrity_payload: dict[str, Any] | None,
    replication_payload: dict[str, Any] | None,
    stability_payload: dict[str, Any] | None,
    kill_test_payload: dict[str, Any] | None,
    dictionary_audit_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    notes: list[str] = []
    if test_id == "m19.runway_efficiency":
        metrics = {}
        if benchmark_payload:
            source = benchmark_payload.get("metrics", {})
            metrics = _filtered_metrics(
                source,
                (
                    "strict_accuracy",
                    "overall_accuracy",
                    "avg_tokens",
                    "accuracy_per_token",
                    "retention_vs_en_cot",
                    "token_ratio_vs_en_cot",
                    "compression_adjusted_retention",
                    "lift_vs_base",
                    "lift_vs_en_cot",
                    "lift_vs_zh_cot",
                ),
            )
        if not metrics:
            return _missing_contract_row(test_id, contract, "missing benchmark report for runway efficiency surface")
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": notes,
        }

    if test_id == "m19.zh_branch_control":
        metrics = {}
        if benchmark_payload:
            metrics = _filtered_metrics(
                benchmark_payload.get("metrics", {}),
                (
                    "zh_cot_accuracy",
                    "zh_cot_avg_tokens",
                    "lift_vs_zh_cot",
                    "token_ratio_vs_en_cot",
                    "compression_adjusted_retention",
                ),
            )
            results = benchmark_payload.get("results", {})
            if isinstance(results, dict):
                metrics.update(
                    {
                        "m19_accuracy": _nested_metric(results, benchmark_payload.get("config", {}).get("cell_id"), "accuracy"),
                        "en_cot_accuracy": _nested_metric(results, "EN-COT", "accuracy"),
                        "zh_cot_accuracy_table": _nested_metric(results, "ZH-COT", "accuracy"),
                    }
                )
                metrics = {k: v for k, v in metrics.items() if v is not None}
        if not metrics:
            return _missing_contract_row(test_id, contract, "missing benchmark report for Chinese branch control surface")
        notes.append("Chinese branch control is evaluated as a peer comparator, not as a replacement reasoning substrate.")
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": notes,
        }

    if test_id == "m19.dynamic_pacing_guardrails":
        if not benchmark_payload:
            return _missing_contract_row(test_id, contract, "missing benchmark report for dynamic pacing guardrails")
        is_dynamic = bool(benchmark_payload.get("config", {}).get("dynamic_pacing", False))
        if not is_dynamic:
            return {
                "test_id": test_id,
                "surface": contract.get("surface"),
                "status": "not_applicable",
                "provenance": "artifact",
                "metrics": {},
                "notes": ["Static M19 tracks do not emit dynamic pacing guardrails; this obligation is carried by M19.4."],
            }
        metrics = _filtered_metrics(
            benchmark_payload.get("metrics", {}),
            (
                "premature_stop_rate",
                "max_cap_hit_rate",
                "scratchpad_bleed_rate",
                "caa_manifold_entanglement_score",
                "avg_tokens",
                "overall_accuracy",
            ),
        )
        dynamic_rollup = benchmark_payload.get("dynamic_rollup", {})
        if isinstance(dynamic_rollup, dict):
            metrics.update({k: v for k, v in dynamic_rollup.items() if isinstance(v, (int, float, bool))})
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": notes,
        }

    if test_id == "m19.integrity_controls":
        if not integrity_payload:
            return _missing_contract_row(test_id, contract, "missing integrity suite report for leakage and masked-control surface")
        metrics = _filtered_metrics(
            integrity_payload.get("metrics", {}),
            (
                "full_accuracy",
                "purged_accuracy",
                "overlap_accuracy",
                "masked_accuracy",
                "overlap_gap",
                "masked_collapse_gap",
                "purged_lift_vs_random",
                "purged_lift_vs_scratchpad_only",
                "audit_qformer_accuracy",
                "audit_lift_vs_random",
                "integrity_overlap_flag",
                "integrity_mask_flag",
                "integrity_audit_flag",
            ),
        )
        headline = integrity_payload.get("headline", {})
        if isinstance(headline, dict):
            metrics.update({f"integrity_{k}": v for k, v in headline.items() if isinstance(v, (int, float, bool, str))})
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": [
                "Integrity suite tracks exact train-eval overlap, purged performance, lexical blindfold masking, and audit lift on the active checkpoint."
            ],
        }

    if test_id == "m19.replication_stability":
        if not replication_payload:
            return _missing_contract_row(test_id, contract, "missing replication suite report for multi-seed stability")
        metrics = _filtered_metrics(
            replication_payload.get("metrics", {}),
            (
                "replication_count",
                "mean_accuracy",
                "std_accuracy",
                "min_accuracy",
                "max_accuracy",
                "mean_avg_tokens",
                "std_avg_tokens",
                "mean_audit_qformer_accuracy",
            ),
        )
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": ["Replication suite measures seed stability under the current M19 training contract."],
        }

    if test_id == "m19.kill_test_suite":
        if not kill_test_payload:
            return _missing_contract_row(test_id, contract, "missing broader kill-test suite report")
        metrics = _filtered_metrics(
            kill_test_payload.get("metrics", {}),
            (
                "purged_accuracy",
                "entity_accuracy",
                "entity_drop_vs_purged",
                "entity_renamed_accuracy",
                "entity_renamed_drop_vs_purged",
                "format_accuracy",
                "format_drop_vs_purged",
                "numeric_accuracy",
                "numeric_drop_vs_purged",
                "masked_accuracy",
                "masked_drop_vs_purged",
            ),
        )
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": ["Broader kill tests extend the integrity bundle with entity, format, and numeric perturbation checks on the purged slice."],
        }

    if test_id == "m19.typed_faithfulness":
        if not dictionary_audit_payload:
            return _missing_contract_row(test_id, contract, "missing dictionary audit report for typed-faithfulness surface")
        checkpoints = dictionary_audit_payload.get("checkpoints", [])
        if not isinstance(checkpoints, list) or not checkpoints:
            return _missing_contract_row(test_id, contract, "dictionary audit report has no checkpoint rows")
        first = checkpoints[0]
        faithfulness = first.get("typed_faithfulness", {}) if isinstance(first, dict) else {}
        metrics = _filtered_metrics(
            faithfulness,
            (
                "typed_family_accuracy",
                "arity_violation_rate",
                "masked_pointer_zero_rate",
                "family_slot_entropy",
                "symbolic_trace_alignment",
            ),
        )
        if not metrics:
            return _missing_contract_row(test_id, contract, "dictionary audit did not emit typed-faithfulness metrics")
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": ["Typed faithfulness measures Lojban-inspired family prediction, arity discipline, pointer masking, and trace alignment on the active typed bridge."],
        }

    if test_id == "m19.hyperbolic_geometry":
        if not dictionary_audit_payload:
            return _missing_contract_row(test_id, contract, "missing dictionary audit report for hyperbolic geometry surface")
        checkpoints = dictionary_audit_payload.get("checkpoints", [])
        if not isinstance(checkpoints, list) or not checkpoints:
            return _missing_contract_row(test_id, contract, "dictionary audit report has no checkpoint rows")
        first = checkpoints[0]
        faithfulness = first.get("typed_faithfulness", {}) if isinstance(first, dict) else {}
        metrics = _filtered_metrics(
            faithfulness,
            (
                "predicate_pointer_radial_gap",
                "family_radius_violation_rate",
                "hyperbolic_geodesic_margin",
                "hyperbolic_projection_clip_rate",
            ),
        )
        if not metrics:
            return _missing_contract_row(test_id, contract, "dictionary audit did not emit hyperbolic geometry metrics")
        return {
            "test_id": test_id,
            "surface": contract.get("surface"),
            "status": "available",
            "provenance": "artifact",
            "metrics": metrics,
            "notes": ["Hyperbolic geometry tracks typed family separation in the Poincare-ball branch and is non-applicable for plain Euclidean baselines without emitted metrics."],
        }

    return _missing_contract_row(test_id, contract, f"no evaluator is registered for {test_id}")


def _missing_contract_row(test_id: str, contract: dict[str, Any], note: str) -> dict[str, Any]:
    return {
        "test_id": test_id,
        "surface": contract.get("surface"),
        "status": "missing",
        "provenance": "none",
        "metrics": {},
        "notes": [note],
    }


def _resolve_historical_family_references(
    contract: dict[str, Any],
    history_manifest_path: Path | None,
) -> list[dict[str, Any]]:
    history_entries = _load_history_entries(history_manifest_path)
    rows: list[dict[str, Any]] = []
    for family in contract.get("historical_comparison_families", []):
        anchor = _resolve_reference_anchor(str(family), history_entries)
        rows.append(
            {
                "family": str(family),
                "status": "resolved" if anchor else "missing",
                "provenance": "artifact" if anchor else "none",
                "anchor_path": _repo_string(anchor) if anchor else None,
                "metrics_digest": _extract_metric_digest(_read_json_optional(anchor) if anchor else None),
            }
        )
    return rows


def _resolve_comparison_targets(
    contract: dict[str, Any],
    history_manifest_path: Path | None,
) -> list[dict[str, Any]]:
    history_entries = _load_history_entries(history_manifest_path)
    rows: list[dict[str, Any]] = []
    for target in contract.get("comparison_targets", []):
        if not isinstance(target, dict):
            continue
        target_name = str(target.get("target", "")).strip()
        anchor = _resolve_reference_anchor(target_name, history_entries)
        rows.append(
            {
                "kind": target.get("kind"),
                "target": target_name,
                "reason": target.get("reason"),
                "status": "resolved" if anchor else "unresolved",
                "anchor_path": _repo_string(anchor) if anchor else None,
                "metrics_digest": _extract_metric_digest(_read_json_optional(anchor) if anchor else None),
            }
        )
    return rows


def _build_reference_surface_index(
    historical_references: list[dict[str, Any]],
    comparison_targets: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    historical_map = {str(row.get("family", "")).strip().upper(): row for row in historical_references}
    comparison_map = {str(row.get("target", "")).strip().upper(): row for row in comparison_targets}

    if "J" in historical_map:
        index["legacy_j_series"] = historical_map["J"]
    if "L" in historical_map:
        index["legacy_l_series"] = historical_map["L"]
    if "M11" in comparison_map:
        index["m11_discriminative_suite"] = comparison_map["M11"]
    if "M14" in comparison_map:
        index["m14_symbiote_scratchpad"] = comparison_map["M14"]
    if "M18" in comparison_map:
        index["m18_controller_family"] = comparison_map["M18"]
    if "M3" in comparison_map:
        index["m_bridge_ablation_suite"] = comparison_map["M3"]
    if "M5" in comparison_map:
        index["m5_series"] = comparison_map["M5"]
    if "M4" in comparison_map:
        index["m4_series"] = comparison_map["M4"]
    return index


def _attach_reference_surface(
    row: dict[str, Any],
    contract: dict[str, Any],
    reference_surface_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if row.get("status") != "missing":
        return row
    surface = str(contract.get("surface", "")).strip()
    reference = reference_surface_index.get(surface)
    if not reference or not reference.get("anchor_path"):
        return row
    notes = list(row.get("notes", []))
    notes.append(f"using reference-only anchor from {reference.get('anchor_path')}")
    return {
        **row,
        "status": "reference_only",
        "provenance": "reference",
        "metrics": dict(reference.get("metrics_digest", {})),
        "reference_anchor_path": reference.get("anchor_path"),
        "notes": notes,
    }


def _resolve_reference_anchor(target: str, history_entries: list[dict[str, Any]]) -> Path | None:
    target_key = str(target).strip()
    if not target_key:
        return None
    history_match = _resolve_from_history(target_key, history_entries)
    if history_match is not None:
        return history_match

    family_hint = target_key.split(".")[0]
    roots = _REFERENCE_ROOTS.get(family_hint, [])
    preferred_names = _preferred_names_for_target(target_key)
    for root in roots:
        hit = _latest_json(root, preferred_names=preferred_names)
        if hit is not None:
            return hit
    return None


def _resolve_from_history(target: str, history_entries: list[dict[str, Any]]) -> Path | None:
    wanted = str(target).strip().upper()
    for entry in history_entries:
        aliases = [str(entry.get("normalized_canonical_id", ""))]
        aliases.extend(str(alias) for alias in entry.get("aliases", []))
        aliases.extend(str(alias) for alias in entry.get("lookup_aliases", []))
        aliases.extend([str(entry.get("canonical_id", ""))])
        if wanted not in {alias.strip().upper() for alias in aliases if alias.strip()}:
            continue
        for root in entry.get("artifact_roots", []):
            path = _repo_path(root)
            if path.is_file():
                return path
            if path.is_dir():
                hit = _latest_json(path)
                if hit is not None:
                    return hit
        archive_path = entry.get("archive_path")
        if archive_path:
            path = _repo_path(archive_path)
            if path.exists():
                if path.is_file():
                    return path
                hit = _latest_json(path)
                if hit is not None:
                    return hit
    return None


def _load_history_entries(history_manifest_path: Path | None) -> list[dict[str, Any]]:
    if history_manifest_path is None:
        return []
    payload = _read_json_optional(history_manifest_path)
    if not isinstance(payload, dict):
        return []
    entries = payload.get("entries", [])
    if isinstance(entries, list):
        return [row for row in entries if isinstance(row, dict)]
    return []


def _build_headline_metrics(
    benchmark_payload: dict[str, Any] | None,
    audit_payload: dict[str, Any] | None,
    integrity_payload: dict[str, Any] | None,
    replication_payload: dict[str, Any] | None,
    stability_payload: dict[str, Any] | None,
    kill_test_payload: dict[str, Any] | None,
    dictionary_audit_payload: dict[str, Any] | None,
    *,
    m20_suite_payload: dict[str, Any] | None = None,
    m20_lock_payload: dict[str, Any] | None = None,
    m20_induction_payload: dict[str, Any] | None = None,
    m21_suite_payload: dict[str, Any] | None = None,
    m21_synthetic_payload: dict[str, Any] | None = None,
    m21_actual_payload: dict[str, Any] | None = None,
    m21_lock_payload: dict[str, Any] | None = None,
    m21_pointer_payload: dict[str, Any] | None = None,
    m21_gauntlet_payload: dict[str, Any] | None = None,
    m21_adversarial_payload: dict[str, Any] | None = None,
    m22_generalization_payload: dict[str, Any] | None = None,
    m23_relevance_payload: dict[str, Any] | None = None,
    m24_compression_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    headline: dict[str, Any] = {}
    if isinstance(benchmark_payload, dict):
        headline.update(_filtered_metrics(benchmark_payload.get("metrics", {}), KEY_METRICS))
        if isinstance(benchmark_payload.get("headline"), dict):
            headline.update({f"benchmark_{k}": v for k, v in benchmark_payload["headline"].items()})
    if isinstance(audit_payload, dict):
        headline.update(
            {
                "audit_qformer_accuracy": audit_payload.get("headline", {}).get("qformer_accuracy"),
                "audit_random_accuracy": audit_payload.get("headline", {}).get("random_accuracy"),
                "audit_lift_vs_base": audit_payload.get("headline", {}).get("lift_vs_base"),
                "audit_lift_vs_random": audit_payload.get("headline", {}).get("lift_vs_random"),
            }
        )
    if isinstance(integrity_payload, dict):
        for key, value in _filtered_metrics(integrity_payload.get("metrics", {}), KEY_METRICS).items():
            headline.setdefault(key, value)
        if isinstance(integrity_payload.get("headline"), dict):
            for key, value in integrity_payload["headline"].items():
                headline[f"integrity_{key}"] = value
    if isinstance(replication_payload, dict):
        for key, value in _filtered_metrics(replication_payload.get("metrics", {}), KEY_METRICS).items():
            headline.setdefault(key, value)
        if isinstance(replication_payload.get("headline"), dict):
            for key, value in replication_payload["headline"].items():
                headline[f"replication_{key}"] = value
    if isinstance(stability_payload, dict):
        for key, value in _filtered_metrics(stability_payload.get("headline", {}), KEY_METRICS).items():
            headline.setdefault(key, value)
        best_balanced = stability_payload.get("best_configs", {}).get("best_balanced")
        if isinstance(best_balanced, dict):
            for key in ("combo_slug", "mean_accuracy", "stable_seed_rate", "mean_audit_qformer_accuracy"):
                if key in best_balanced and best_balanced.get(key) is not None:
                    headline[f"stability_{key}"] = best_balanced.get(key)
    if isinstance(kill_test_payload, dict):
        for key, value in _filtered_metrics(kill_test_payload.get("metrics", {}), KEY_METRICS).items():
            headline.setdefault(key, value)
        if isinstance(kill_test_payload.get("headline"), dict):
            for key, value in kill_test_payload["headline"].items():
                headline[f"kill_{key}"] = value
    if isinstance(dictionary_audit_payload, dict):
        checkpoints = dictionary_audit_payload.get("checkpoints", [])
        if isinstance(checkpoints, list) and checkpoints:
            first = checkpoints[0]
            if isinstance(first, dict):
                for key, value in _filtered_metrics(first.get("typed_faithfulness", {}), KEY_METRICS).items():
                    headline.setdefault(key, value)
    if isinstance(m20_suite_payload, dict):
        for key, value in _filtered_metrics(_m20_suite_metrics(m20_suite_payload), KEY_METRICS).items():
            headline.setdefault(key, value)
    if isinstance(m20_lock_payload, dict):
        for key, value in _filtered_metrics(m20_lock_payload.get("metrics", {}), KEY_METRICS).items():
            headline.setdefault(key, value)
    if isinstance(m20_induction_payload, dict):
        for key, value in _filtered_metrics(m20_induction_payload.get("metrics", {}), KEY_METRICS).items():
            headline.setdefault(key, value)
    if isinstance(m21_suite_payload, dict):
        for key, value in _filtered_metrics(_m21_suite_metrics(m21_suite_payload), KEY_METRICS).items():
            headline.setdefault(key, value)
    if isinstance(m21_synthetic_payload, dict):
        for key, value in _filtered_metrics(_m21_suite_metrics(m21_synthetic_payload), KEY_METRICS).items():
            headline.setdefault(key, value)
    if isinstance(m21_actual_payload, dict):
        for key, value in _filtered_metrics(m21_actual_payload.get("metrics", {}), KEY_METRICS).items():
            if key in {
                "strict_accuracy",
                "full_accuracy",
                "actual_bridge_transfer_score",
                "no_cmavo_accuracy",
                "no_judri_accuracy",
                "gismu_only_accuracy",
                "random_trace_accuracy",
                "scratchpad_only_accuracy",
                "frame_drop_delta",
                "cmavo_causal_delta",
                "judri_causal_delta",
            }:
                headline[key] = value
            else:
                headline.setdefault(key, value)
    if isinstance(m21_lock_payload, dict):
        for key, value in _filtered_metrics(m21_lock_payload.get("metrics", {}), KEY_METRICS).items():
            if key in {"lock_pass_rate", "brivi_lock_violation_rate", "brivi_gate_accuracy"}:
                headline[key] = value
            else:
                headline.setdefault(key, value)
    if isinstance(m21_pointer_payload, dict):
        for key, value in _filtered_metrics(_m21_suite_metrics(m21_pointer_payload), KEY_METRICS).items():
            if key in {"loss_pointer_necessity", "pointer_necessity_gap"}:
                headline[key] = value
            else:
                headline.setdefault(key, value)
    if isinstance(m21_gauntlet_payload, dict):
        for key, value in _filtered_metrics(m21_gauntlet_payload.get("metrics", {}), KEY_METRICS).items():
            if str(key).startswith("gauntlet_") or str(key).startswith("m19_gauntlet_"):
                headline[key] = value
            else:
                headline.setdefault(key, value)
    if isinstance(m21_adversarial_payload, dict):
        for key, value in _filtered_metrics(_m21_suite_metrics(m21_adversarial_payload), KEY_METRICS).items():
            if (
                str(key).startswith("adversarial_")
                or str(key).startswith("semantic_coverage_")
                or str(key).startswith("semantic_isolation_")
            ):
                headline[key] = value
            else:
                headline.setdefault(key, value)
    if isinstance(m22_generalization_payload, dict) and isinstance(m22_generalization_payload.get("metrics"), dict):
        for key, value in _filtered_metrics(m22_generalization_payload["metrics"], KEY_METRICS).items():
            headline[key] = value
    if isinstance(m23_relevance_payload, dict):
        for key, value in _filtered_metrics(_m23_suite_metrics(m23_relevance_payload), KEY_METRICS).items():
            headline[key] = value
    if isinstance(m24_compression_payload, dict):
        for key, value in _filtered_metrics(_m24_compression_metrics(m24_compression_payload), KEY_METRICS).items():
            headline[key] = value
    return {k: v for k, v in headline.items() if v is not None}


def _m24_compression_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, Any] = {}
    for source_name in ("headline_metrics", "metrics", "aggregate_metrics"):
        source = payload.get(source_name, {})
        if isinstance(source, dict):
            metrics.update(source)
    aggregate = payload.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
        metrics.setdefault("strict_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("overall_phrase_accuracy", aggregate.get("mean_overall_phrase_accuracy"))
        metrics.setdefault("phrase_accuracy", aggregate.get("mean_phrase_accuracy"))
        metrics.setdefault("phrase_exact_accuracy", aggregate.get("mean_phrase_exact_accuracy"))
        metrics.setdefault("avg_tokens", aggregate.get("mean_avg_tokens"))
        metrics.setdefault("trace_tokens", aggregate.get("mean_trace_tokens"))
        metrics.setdefault("accuracy_per_token", aggregate.get("mean_accuracy_per_token"))
        metrics.setdefault("accuracy_per_trace_token", aggregate.get("mean_accuracy_per_trace_token"))
        metrics.setdefault("substrate_token_count", aggregate.get("mean_substrate_token_count"))
        metrics.setdefault("substrate_tokens", aggregate.get("mean_substrate_tokens"))
        metrics.setdefault("reference_token_count", aggregate.get("mean_reference_token_count"))
        metrics.setdefault("reference_tokens", aggregate.get("mean_reference_tokens"))
        metrics.setdefault("baseline_token_count", aggregate.get("mean_baseline_token_count"))
        metrics.setdefault("baseline_tokens", aggregate.get("mean_baseline_tokens"))
        metrics.setdefault("compression_ratio", aggregate.get("mean_compression_ratio"))
        metrics.setdefault("predicted_trace_accuracy", aggregate.get("mean_predicted_trace_accuracy"))
        metrics.setdefault("oracle_trace_accuracy", aggregate.get("mean_oracle_trace_accuracy"))
        metrics.setdefault("random_trace_accuracy", aggregate.get("mean_random_trace_accuracy"))
        metrics.setdefault("shuffled_trace_accuracy", aggregate.get("mean_shuffled_trace_accuracy"))
        metrics.setdefault("zero_trace_accuracy", aggregate.get("mean_zero_trace_accuracy"))
        metrics.setdefault("prompt_only_accuracy", aggregate.get("mean_prompt_only_accuracy"))
        metrics.setdefault("advisor_vs_prompt_delta", aggregate.get("mean_advisor_vs_prompt_delta"))
        metrics.setdefault("m24_strict_delta_vs_prompt_only", aggregate.get("mean_m24_strict_delta_vs_prompt_only"))
        metrics.setdefault("predicted_vs_random_delta", aggregate.get("mean_predicted_vs_random_delta"))
        metrics.setdefault("predicted_vs_shuffled_delta", aggregate.get("mean_predicted_vs_shuffled_delta"))
        metrics.setdefault("oracle_trained_oracle_trace_accuracy", aggregate.get("mean_oracle_trained_oracle_trace_accuracy"))
        metrics.setdefault("oracle_trained_predicted_trace_accuracy", aggregate.get("mean_oracle_trained_predicted_trace_accuracy"))
        metrics.setdefault("oracle_trained_random_trace_accuracy", aggregate.get("mean_oracle_trained_random_trace_accuracy"))
        metrics.setdefault("oracle_trained_trace_delta", aggregate.get("mean_oracle_trained_trace_delta"))
        metrics.setdefault("predicted_trace_gap_to_oracle_upper_bound", aggregate.get("mean_predicted_trace_gap_to_oracle_upper_bound"))
        metrics.setdefault("cross_advisor_oracle_gap", aggregate.get("mean_cross_advisor_oracle_gap"))
        metrics.setdefault("substrate_claim_score", aggregate.get("mean_substrate_claim_score"))
        metrics.setdefault("m24_promotion_gate_pass_rate", aggregate.get("mean_m24_promotion_gate_pass_rate"))
        metrics.setdefault("m24_promotion_candidate", aggregate.get("mean_m24_promotion_candidate"))
        metrics.setdefault("generator_parameter_max_delta_after_advisor", aggregate.get("mean_generator_parameter_max_delta_after_advisor"))
        metrics.setdefault("generator_parameters_unchanged_after_advisor", aggregate.get("mean_generator_parameters_unchanged_after_advisor"))
        metrics.setdefault("bridi_trace_exact_accuracy", aggregate.get("mean_bridi_trace_exact_accuracy"))
        metrics.setdefault("gismu_accuracy", aggregate.get("mean_gismu_accuracy"))
        metrics.setdefault("cmavo_accuracy", aggregate.get("mean_cmavo_accuracy"))
        metrics.setdefault("judri_binding_accuracy", aggregate.get("mean_judri_accuracy"))
        metrics.setdefault("packed_symbol_to_prompt_ratio", aggregate.get("mean_packed_symbol_to_prompt_ratio"))
        metrics.setdefault("prompt_to_packed_symbol_ratio", aggregate.get("mean_prompt_to_packed_symbol_ratio"))
        metrics.setdefault("packed_to_prompt_ratio", aggregate.get("mean_packed_to_prompt_ratio"))
        metrics.setdefault("prompt_to_packed_ratio", aggregate.get("mean_prompt_to_packed_ratio"))
        metrics.setdefault("packed_symbol_compression_ratio", aggregate.get("mean_packed_symbol_compression_ratio"))
        metrics.setdefault("mdl_weight", aggregate.get("mean_mdl_weight"))
        metrics.setdefault("compression_ratio", aggregate.get("mean_compression_ratio"))
        if metrics.get("compression_ratio") is None:
            metrics["compression_ratio"] = aggregate.get("mean_prompt_to_packed_symbol_ratio")
        metrics.setdefault("token_reduction_ratio", aggregate.get("mean_token_reduction_ratio"))
        metrics.setdefault(
            "m24_gate_packed_trace_shorter_than_prompt",
            aggregate.get("mean_m24_gate_packed_trace_shorter_than_prompt"),
        )
    for source_name in ("headline_metrics", "metrics"):
        source = payload.get(source_name, {})
        if isinstance(source, dict):
            metrics.setdefault("strict_accuracy", source.get("mean_strict_accuracy"))
            metrics.setdefault("overall_phrase_accuracy", source.get("phrase_accuracy"))
            metrics.setdefault("phrase_accuracy", source.get("overall_phrase_accuracy"))
            metrics.setdefault("substrate_token_count", source.get("mean_substrate_token_count"))
            metrics.setdefault("compression_ratio", source.get("mean_compression_ratio"))
            metrics.setdefault("mdl_weight", source.get("mean_mdl_weight"))
            metrics.setdefault("token_reduction_ratio", source.get("mean_token_reduction_ratio"))
            metrics.setdefault(
                "m24_gate_packed_trace_shorter_than_prompt",
                source.get("mean_m24_gate_packed_trace_shorter_than_prompt"),
            )
    seed_rows: list[dict[str, Any]] = []
    cells = payload.get("cells", {})
    if isinstance(cells, dict):
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            for row in cell.get("seed_reports", []):
                if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                    seed_rows.append(row["metrics"])
    top_level_seed_rows = payload.get("seed_reports", [])
    if isinstance(top_level_seed_rows, list):
        for row in top_level_seed_rows:
            if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                seed_rows.append(row["metrics"])
    for key in KEY_METRICS:
        values = [float(row[key]) for row in seed_rows if isinstance(row.get(key), (int, float))]
        if values and key not in metrics:
            metrics[key] = sum(values) / len(values)
    return {key: value for key, value in metrics.items() if value is not None}


def _m23_suite_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, Any] = {}
    aggregate = payload.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
        metrics.update(aggregate)
        metrics.setdefault("strict_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("synthetic_world_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("bridi_trace_exact_accuracy", aggregate.get("mean_bridi_trace_exact_accuracy"))
        metrics.setdefault("decoy_relation_ood_accuracy", aggregate.get("mean_decoy_relation_ood_accuracy"))
        metrics.setdefault("worst_surface_accuracy", aggregate.get("mean_worst_surface_accuracy"))
        metrics.setdefault("relevance_top1_accuracy", aggregate.get("mean_relevance_top1_accuracy"))
        metrics.setdefault("relevance_margin", aggregate.get("mean_relevance_margin"))
        metrics.setdefault("loss_trace_exact_surrogate", aggregate.get("mean_loss_trace_exact_surrogate"))
        metrics.setdefault("trace_exact_surrogate_weight", aggregate.get("mean_trace_exact_surrogate_weight"))
        metrics.setdefault("oracle_relevance_accuracy", aggregate.get("mean_oracle_relevance_accuracy"))
        metrics.setdefault("random_relevance_accuracy", aggregate.get("mean_random_relevance_accuracy"))
        metrics.setdefault("no_relevance_accuracy", aggregate.get("mean_no_relevance_accuracy"))
        metrics.setdefault("decoy_only_accuracy", aggregate.get("mean_decoy_only_accuracy"))
        metrics.setdefault("oracle_relevance_delta", aggregate.get("mean_oracle_relevance_delta"))
        metrics.setdefault("random_relevance_delta", aggregate.get("mean_random_relevance_delta"))
        metrics.setdefault("decoy_only_delta", aggregate.get("mean_decoy_only_delta"))
        metrics.setdefault("m23_router_decoy_lift_vs_scale", aggregate.get("m23_router_decoy_lift_vs_scale"))
        metrics.setdefault("m23_router_worst_surface_lift_vs_scale", aggregate.get("m23_router_worst_surface_lift_vs_scale"))
        metrics.setdefault("m23_oracle_relevance_lift", aggregate.get("m23_oracle_relevance_lift"))
        metrics.setdefault("m23_trace_punish_trace_exact_lift_vs_scale", aggregate.get("m23_trace_punish_trace_exact_lift_vs_scale"))
        metrics.setdefault("m23_trace_punish_decoy_delta_vs_scale", aggregate.get("m23_trace_punish_decoy_delta_vs_scale"))
        metrics.setdefault("m23_trace_punish_strict_delta_vs_scale", aggregate.get("m23_trace_punish_strict_delta_vs_scale"))
        metrics.setdefault("avg_tokens", aggregate.get("avg_tokens"))
        metrics.setdefault("accuracy_per_token", aggregate.get("accuracy_per_token"))
        metrics.setdefault("trace_tokens", aggregate.get("trace_tokens"))
        metrics.setdefault("accuracy_per_trace_token", aggregate.get("accuracy_per_trace_token"))
    seed_rows: list[dict[str, Any]] = []
    cells = payload.get("cells", {})
    if isinstance(cells, dict):
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            for row in cell.get("seed_reports", []):
                if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                    seed_rows.append(row["metrics"])
    for key in KEY_METRICS:
        values = [float(row[key]) for row in seed_rows if isinstance(row.get(key), (int, float))]
        if values and key not in metrics:
            metrics[key] = sum(values) / len(values)
    return {key: value for key, value in metrics.items() if value is not None}


def _m21_suite_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, Any] = {}
    aggregate = payload.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
        metrics.update(aggregate)
        metrics.setdefault("strict_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("synthetic_world_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("bridi_trace_exact_accuracy", aggregate.get("mean_bridi_trace_exact_accuracy"))
        metrics.setdefault("gismu_accuracy", aggregate.get("mean_gismu_accuracy"))
        metrics.setdefault("cmavo_accuracy", aggregate.get("mean_cmavo_accuracy"))
        metrics.setdefault("judri_binding_accuracy", aggregate.get("mean_judri_binding_accuracy"))
        metrics.setdefault("frame_count_mae", aggregate.get("mean_frame_count_mae"))
        metrics.setdefault("lock_pass_rate", aggregate.get("mean_lock_pass_rate"))
        metrics.setdefault("full_accuracy", aggregate.get("mean_full_accuracy"))
        metrics.setdefault("no_cmavo_accuracy", aggregate.get("mean_no_cmavo_accuracy"))
        metrics.setdefault("no_judri_accuracy", aggregate.get("mean_no_judri_accuracy"))
        metrics.setdefault("gismu_only_accuracy", aggregate.get("mean_gismu_only_accuracy"))
        metrics.setdefault("frame_drop_delta", aggregate.get("mean_frame_drop_delta"))
        metrics.setdefault("cmavo_causal_delta", aggregate.get("mean_cmavo_causal_delta"))
        metrics.setdefault("judri_causal_delta", aggregate.get("mean_judri_causal_delta"))
        metrics.setdefault("loss_pointer_necessity", aggregate.get("mean_loss_pointer_necessity"))
        metrics.setdefault("pointer_necessity_gap", aggregate.get("mean_pointer_necessity_gap"))
        metrics.setdefault("hyperbolic_tangent_handoff_norm_mean", aggregate.get("mean_hyperbolic_tangent_handoff_norm_mean"))
        metrics.setdefault("hyperbolic_tangent_handoff_finite_rate", aggregate.get("mean_hyperbolic_tangent_handoff_finite_rate"))
        metrics.setdefault("judri_bridge_gate_enabled", aggregate.get("mean_judri_bridge_gate_enabled"))
        metrics.setdefault("judri_bridge_gate_mean", aggregate.get("mean_judri_bridge_gate_mean"))
        metrics.setdefault("judri_bridge_gate_active_mean", aggregate.get("mean_judri_bridge_gate_active_mean"))
        metrics.setdefault("judri_bridge_gate_silenced_predicate_energy_mean", aggregate.get("mean_judri_bridge_gate_silenced_predicate_energy_mean"))
        metrics.setdefault("adversarial_strict_accuracy", aggregate.get("mean_adversarial_strict_accuracy"))
        metrics.setdefault("adversarial_bridi_trace_exact_accuracy", aggregate.get("mean_adversarial_bridi_trace_exact_accuracy"))
        metrics.setdefault("adversarial_gismu_accuracy", aggregate.get("mean_adversarial_gismu_accuracy"))
        metrics.setdefault("adversarial_cmavo_accuracy", aggregate.get("mean_adversarial_cmavo_accuracy"))
        metrics.setdefault("adversarial_judri_binding_accuracy", aggregate.get("mean_adversarial_judri_binding_accuracy"))
        metrics.setdefault("adversarial_no_judri_accuracy", aggregate.get("mean_adversarial_no_judri_accuracy"))
        metrics.setdefault("adversarial_judri_causal_delta", aggregate.get("mean_adversarial_judri_causal_delta"))
        metrics.setdefault("adversarial_worst_surface_accuracy", aggregate.get("mean_adversarial_worst_surface_accuracy"))
        metrics.setdefault("adversarial_oov_token_rate", aggregate.get("mean_adversarial_oov_token_rate"))
        metrics.setdefault("adversarial_oov_synonym_accuracy", aggregate.get("mean_adversarial_oov_synonym_accuracy"))
        metrics.setdefault(
            "adversarial_oov_synonym_trace_exact_accuracy",
            aggregate.get("mean_adversarial_oov_synonym_trace_exact_accuracy"),
        )
        metrics.setdefault("mean_adversarial_train_fraction", aggregate.get("mean_adversarial_train_fraction"))
        metrics.setdefault("adversarial_training_exposure_rate", aggregate.get("adversarial_training_exposure_rate"))
        metrics.setdefault("semantic_coverage_strict_accuracy", aggregate.get("semantic_coverage_strict_accuracy"))
        metrics.setdefault("semantic_coverage_worst_surface_accuracy", aggregate.get("semantic_coverage_worst_surface_accuracy"))
        metrics.setdefault("semantic_coverage_judri_causal_delta", aggregate.get("semantic_coverage_judri_causal_delta"))
        metrics.setdefault("semantic_coverage_oov_token_rate", aggregate.get("semantic_coverage_oov_token_rate"))
        metrics.setdefault("semantic_coverage_oov_synonym_accuracy", aggregate.get("semantic_coverage_oov_synonym_accuracy"))
        metrics.setdefault(
            "semantic_coverage_oov_synonym_trace_exact_accuracy",
            aggregate.get("semantic_coverage_oov_synonym_trace_exact_accuracy"),
        )
        metrics.setdefault("semantic_coverage_surface_seed_std_max", aggregate.get("semantic_coverage_surface_seed_std_max"))
        metrics.setdefault(
            "semantic_coverage_surface_seed_min_accuracy",
            aggregate.get("semantic_coverage_surface_seed_min_accuracy"),
        )
        metrics.setdefault("semantic_coverage_training_exposure_rate", aggregate.get("semantic_coverage_training_exposure_rate"))
        metrics.setdefault("semantic_coverage_train_fraction", aggregate.get("semantic_coverage_train_fraction"))
        metrics.setdefault("semantic_coverage_surface_count", aggregate.get("semantic_coverage_surface_count"))
        metrics.setdefault("semantic_isolation_cell_count", aggregate.get("semantic_isolation_cell_count"))
        for key in (
            "m22_relation_ood_strict_accuracy",
            "m22_relation_ood_worst_surface_accuracy",
            "m22_relation_ood_bridi_trace_exact_accuracy",
            "m22_relation_ood_judri_causal_delta",
            "m22_relation_ood_oov_token_rate",
            "m22_relation_ood_surface_count",
            "m22_relation_ood_surface_seed_std_max",
            "m22_relation_ood_surface_seed_min_accuracy",
            "m22_relation_ood_surface_training_overlap_rate",
        ):
            metrics.setdefault(key, aggregate.get(key))
        for key, value in aggregate.items():
            if str(key).startswith("semantic_coverage_") and str(key).endswith("_delta"):
                metrics.setdefault(str(key), value)
            if str(key).startswith("semantic_isolation_"):
                metrics.setdefault(str(key), value)
        metrics.setdefault("mean_active_frames", aggregate.get("mean_active_frames"))
        metrics.setdefault("active_code_fraction_reachable", aggregate.get("mean_active_code_fraction_reachable"))
    seed_rows: list[dict[str, Any]] = []
    cells = payload.get("cells", {})
    if isinstance(cells, dict):
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            for row in cell.get("seed_reports", []):
                if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                    seed_rows.append(row["metrics"])
    top_level_seed_rows = payload.get("seed_reports", [])
    if isinstance(top_level_seed_rows, list):
        for row in top_level_seed_rows:
            if isinstance(row, dict) and isinstance(row.get("metrics"), dict):
                seed_rows.append(row["metrics"])
    for key in KEY_METRICS:
        values = [float(row[key]) for row in seed_rows if isinstance(row.get(key), (int, float))]
        if values and key not in metrics:
            metrics[key] = sum(values) / len(values)
    return metrics


def _m20_suite_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, Any] = {}
    aggregate = payload.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
        metrics.update(aggregate)
        metrics.setdefault("strict_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("synthetic_world_accuracy", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("dictionary_coverage", aggregate.get("mean_strict_accuracy"))
        metrics.setdefault("factorized_exact_accuracy", aggregate.get("mean_factorized_exact_accuracy"))
        metrics.setdefault("brivi_gate_accuracy", aggregate.get("mean_brivi_gate_accuracy"))
        metrics.setdefault("predicate_identity_stability", aggregate.get("mean_predicate_identity_stability"))
        metrics.setdefault("lock_pass_rate", aggregate.get("mean_lock_pass_rate"))
    cells = payload.get("cells", {})
    if isinstance(cells, dict):
        seed_metric_values: dict[str, list[float]] = {}
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            for seed_report in cell.get("seed_reports", []):
                if not isinstance(seed_report, dict):
                    continue
                for key, value in seed_report.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        seed_metric_values.setdefault(str(key), []).append(float(value))
        for key, values in seed_metric_values.items():
            metrics.setdefault(key, sum(values) / max(1, len(values)))
    return {key: value for key, value in metrics.items() if value is not None}


def _surface_record(name: str, path: Path | None) -> dict[str, Any]:
    payload = _read_json_optional(path)
    return {
        "name": name,
        "status": "resolved" if payload is not None else "missing",
        "path": _repo_string(path) if path else None,
        "payload": payload,
    }


def _extract_metric_digest(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return _filtered_metrics(metrics, KEY_METRICS)
    aggregate = payload.get("aggregate_metrics")
    if isinstance(aggregate, dict):
        digest = _filtered_metrics(aggregate, KEY_METRICS)
        if "mean_strict_accuracy" in aggregate:
            digest.setdefault("strict_accuracy", aggregate.get("mean_strict_accuracy"))
        if "mean_lock_pass_rate" in aggregate:
            digest.setdefault("lock_pass_rate", aggregate.get("mean_lock_pass_rate"))
        return digest
    headline = payload.get("headline")
    if isinstance(headline, dict):
        return {k: v for k, v in headline.items() if isinstance(v, (int, float, bool)) or v is None}
    return {}


def _filtered_metrics(source: Any, keys: tuple[str, ...] | list[str]) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {}
    output: dict[str, Any] = {}
    for key in keys:
        if key in source:
            output[str(key)] = source[key]
    return output


def _nested_metric(source: Any, row_key: Any, metric_key: str) -> Any:
    if not isinstance(source, dict):
        return None
    row = source.get(row_key)
    if not isinstance(row, dict):
        return None
    return row.get(metric_key)


def _latest_named_manifest(root: Path, file_name: str) -> Path | None:
    return _artifact_latest_named_manifest(root, file_name, recursive=True, path_filter=_path_allowed_for_discovery)


def _latest_json(root: Path, preferred_names: list[str] | None = None) -> Path | None:
    return _artifact_latest_json(root, preferred_names or [], path_filter=_path_allowed_for_discovery)


def _preferred_names_for_target(target: str) -> list[str]:
    upper = str(target).strip().upper()
    if upper == "J":
        return ["j-5.json", "j5.json", "j_series_summary.json"]
    if upper == "L":
        return ["l6_ablation_manifest.json", "l_series_summary.json"]
    if upper == "M10":
        return ["final_publication_metrics.json", "m10_audit_report.json", "final_bridge_audit.json"]
    if upper == "M9":
        return ["m9_audit_report.json", "duel_report.json"]
    if upper == "M8":
        return ["m8_eval_report.json"]
    if upper == "M7":
        return ["m7_eval_report.json"]
    if upper == "M6":
        return ["m6_eval_report.json", "m6_directed_eval_report.json", "m6_expansive_report.json"]
    if upper == "M5":
        return ["m5_eval_report.json", "m5_family_report.json"]
    if upper == "M4":
        return ["ablation_hypercube_report.json"]
    if upper == "M3":
        return ["m_bridge_ablation_suite_manifest.json", "ablation_hypercube_report.json"]
    if upper == "M11":
        return ["m_bridge_ablation_suite_manifest.json", "m11_publication_summary.json"]
    if upper == "M14":
        return ["m14_5_report.json", "m14_family_report.json"]
    if upper == "M18":
        return ["m18_family_report.json", "harmonized_audit_report.json"]
    if upper.startswith("M19"):
        return ["m19_mainline_report.json", "m19_4_mainline_report.json", "m19_isolation_grid_report.json"]
    if upper.startswith("M20"):
        return ["m20_dictionary_first_suite_report.json", "m20_lock_suite_report.json", "m20_predicate_induction_report.json"]
    if upper.startswith("M21"):
        return [
            "m21_dynamic_bridi_suite_report.json",
            "m21_synthetic_assay_report.json",
            "m21_actual_bridge_report.json",
            "m21_lock_suite_report.json",
            "m21_pointer_necessity_microgrid_report.json",
            "m21_gauntlet_report.json",
            "m21_adversarial_audit_report.json",
        ]
    if upper.startswith("M22"):
        return ["m22_semantic_generalization_report.json", "direct_unified_eval_manifest.json"]
    if upper.startswith("M23"):
        return ["m23_relevance_suite_report.json", "m23_relevance_train_report.json", "direct_unified_eval_manifest.json"]
    if upper.startswith("M24"):
        return ["m24_substrate_compression_report.json", "direct_unified_eval_manifest.json"]
    return []


def _path_allowed_for_discovery(path: Path) -> bool:
    return _artifact_path_allowed_for_discovery(path)


def _read_json_optional(path: Path | None) -> dict[str, Any] | None:
    return _artifact_read_json_optional(path)


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _repo_string(path: Path | None) -> str | None:
    try:
        return repo_relative(path) if path is not None else None
    except ValueError:
        return _artifact_repo_string(path, REPO_ROOT)


def _track_key(track: str) -> str:
    candidate = str(track).strip()
    return candidate if candidate in M19_REGISTRY else "M19"


def _unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def _format_scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
