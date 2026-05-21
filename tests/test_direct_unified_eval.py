from __future__ import annotations

import json
from pathlib import Path
import uuid

from lojban_evolution.direct_unified_eval import (
    build_direct_unified_eval_manifest,
    render_direct_unified_eval_markdown,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _scratch_dir() -> Path:
    root = Path("runs") / "test_direct_unified_eval" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_build_direct_unified_eval_manifest_static_m19() -> None:
    tmp_path = _scratch_dir()
    benchmark_path = _write_json(
        tmp_path / "benchmark.json",
        {
            "config": {
                "dynamic_pacing": False,
                "cell_id": "M19.3_8Q_128D_8S",
            },
            "metrics": {
                "strict_accuracy": 0.19,
                "overall_accuracy": 0.19,
                "avg_tokens": 30.32,
                "accuracy_per_token": 0.00626,
                "token_ratio_vs_en_cot": 0.24,
                "lift_vs_zh_cot": 0.19,
                "zh_cot_accuracy": 0.0,
                "zh_cot_avg_tokens": 117.9,
            },
            "results": {
                "M19.3_8Q_128D_8S": {"accuracy": 0.19},
                "EN-COT": {"accuracy": 0.0},
                "ZH-COT": {"accuracy": 0.0},
            },
            "headline": {"overall_accuracy": 0.19},
        },
    )
    audit_path = _write_json(
        tmp_path / "audit.json",
        {
            "headline": {
                "qformer_accuracy": 0.1,
                "random_accuracy": 0.0,
                "lift_vs_base": 0.1,
                "lift_vs_random": 0.1,
            }
        },
    )
    integrity_path = _write_json(
        tmp_path / "integrity.json",
        {
            "metrics": {
                "purged_accuracy": 0.18,
                "overlap_gap": 0.02,
                "masked_accuracy": 0.03,
                "audit_qformer_accuracy": 1.0,
                "audit_lift_vs_random": 1.0,
            },
            "headline": {
                "integrity_status": "pass",
            },
        },
    )
    replication_path = _write_json(
        tmp_path / "replication.json",
        {
            "metrics": {
                "replication_count": 3,
                "mean_accuracy": 0.36,
                "std_accuracy": 0.01,
                "mean_avg_tokens": 31.9,
                "mean_audit_qformer_accuracy": 0.95,
            }
        },
    )
    stability_path = _write_json(
        tmp_path / "stability.json",
        {
            "headline": {
                "configs_tested": 4,
                "best_mean_accuracy": 0.28,
                "best_stable_seed_rate": 0.5,
                "recovered_seed_count": 1,
            },
            "best_configs": {
                "best_balanced": {
                    "combo_slug": "lr_5em05_aug_0p0",
                    "mean_accuracy": 0.28,
                    "stable_seed_rate": 0.5,
                    "mean_audit_qformer_accuracy": 0.95,
                }
            },
        },
    )
    kill_path = _write_json(
        tmp_path / "kill.json",
        {
            "metrics": {
                "purged_accuracy": 0.18,
                "entity_accuracy": 0.17,
                "format_accuracy": 0.18,
                "numeric_accuracy": 0.16,
                "masked_accuracy": 0.03,
            }
        },
    )
    dictionary_path = _write_json(
        tmp_path / "dictionary.json",
        {
            "checkpoints": [
                {
                    "label": "seed23",
                    "typed_faithfulness": {
                        "typed_family_accuracy": 0.88,
                        "arity_violation_rate": 0.05,
                        "masked_pointer_zero_rate": 1.0,
                        "family_slot_entropy": 0.22,
                        "symbolic_trace_alignment": 0.74,
                        "predicate_pointer_radial_gap": 0.31,
                        "family_radius_violation_rate": 0.02,
                        "hyperbolic_geodesic_margin": 0.56,
                        "hyperbolic_projection_clip_rate": 0.04,
                    },
                }
            ]
        },
    )
    j_anchor = _write_json(tmp_path / "j-5.json", {"metrics": {"accepted_foil_pair_accuracy": 0.77}})
    l_anchor = _write_json(tmp_path / "l_series_summary.json", {"metrics": {"constraint_scope": 0.92}})
    history_path = _write_json(
        tmp_path / "history.json",
        {
            "entries": [
                {
                    "canonical_id": "legacy.j",
                    "normalized_canonical_id": "J",
                    "aliases": ["J"],
                    "lookup_aliases": ["J"],
                    "artifact_roots": [str(j_anchor)],
                },
                {
                    "canonical_id": "legacy.l",
                    "normalized_canonical_id": "L",
                    "aliases": ["L"],
                    "lookup_aliases": ["L"],
                    "artifact_roots": [str(l_anchor)],
                },
            ]
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M19",
        track="M19",
        benchmark_report_path=benchmark_path,
        audit_report_path=audit_path,
        integrity_report_path=integrity_path,
        replication_report_path=replication_path,
        stability_report_path=stability_path,
        kill_test_report_path=kill_path,
        dictionary_audit_report_path=dictionary_path,
        history_manifest_path=history_path,
    )

    assert manifest["family_key"] == "M19"
    assert manifest["track"] == "M19"
    assert len(manifest["contract_results"]) >= 3

    runway = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.runway_efficiency")
    assert runway["status"] == "available"
    assert runway["metrics"]["overall_accuracy"] == 0.19

    guardrails = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.dynamic_pacing_guardrails")
    assert guardrails["status"] == "not_applicable"
    integrity = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.integrity_controls")
    assert integrity["status"] == "available"
    assert integrity["metrics"]["purged_accuracy"] == 0.18
    replication = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.replication_stability")
    assert replication["status"] == "available"
    assert replication["metrics"]["mean_accuracy"] == 0.36
    kill = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.kill_test_suite")
    assert kill["status"] == "available"
    assert kill["metrics"]["entity_accuracy"] == 0.17
    typed = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.typed_faithfulness")
    assert typed["status"] == "available"
    assert typed["metrics"]["typed_family_accuracy"] == 0.88
    hyper = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.hyperbolic_geometry")
    assert hyper["status"] == "available"
    assert hyper["metrics"]["predicate_pointer_radial_gap"] == 0.31

    inherited = {row["test_id"] for row in manifest["contract_results"]}
    assert "m14.scratchpad_bleed" in inherited
    assert "m11.native_discriminative_oracle" in inherited

    refs = {row["family"]: row for row in manifest["historical_family_references"]}
    assert refs["J"]["status"] == "resolved"
    assert refs["L"]["status"] == "resolved"

    assert manifest["headline_metrics"]["overall_accuracy"] == 0.19
    assert manifest["headline_metrics"]["audit_qformer_accuracy"] == 0.1
    assert manifest["headline_metrics"]["purged_accuracy"] == 0.18
    assert manifest["headline_metrics"]["mean_accuracy"] == 0.36
    assert manifest["headline_metrics"]["best_mean_accuracy"] == 0.28
    assert manifest["headline_metrics"]["stability_combo_slug"] == "lr_5em05_aug_0p0"
    assert manifest["headline_metrics"]["entity_accuracy"] == 0.17
    assert manifest["headline_metrics"]["typed_family_accuracy"] == 0.88

    rendered = render_direct_unified_eval_markdown(manifest)
    assert "Direct Unified Eval: M19 (M19)" in rendered
    assert "m19.runway_efficiency" in rendered


def test_build_direct_unified_eval_manifest_dynamic_m19() -> None:
    tmp_path = _scratch_dir()
    benchmark_path = _write_json(
        tmp_path / "m19_4_benchmark_report.json",
        {
            "config": {
                "dynamic_pacing": True,
                "cell_id": "M19.4",
            },
            "metrics": {
                "overall_accuracy": 0.31,
                "avg_tokens": 17.0,
                "premature_stop_rate": 0.02,
                "max_cap_hit_rate": 0.0,
                "scratchpad_bleed_rate": 0.01,
                "caa_manifold_entanglement_score": 0.13,
            },
            "dynamic_rollup": {
                "mean_latent_steps": 8.4,
            },
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M19",
        track="M19.4",
        benchmark_report_path=benchmark_path,
        audit_report_path=None,
        integrity_report_path=None,
        history_manifest_path=None,
    )

    guardrails = next(row for row in manifest["contract_results"] if row["test_id"] == "m19.dynamic_pacing_guardrails")
    assert guardrails["status"] == "available"
    assert guardrails["metrics"]["premature_stop_rate"] == 0.02
    assert guardrails["metrics"]["mean_latent_steps"] == 8.4


def test_m21_direct_eval_keeps_pointer_gauntlet_surfaces_separate() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m21_suite.json",
        {
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.8,
                "mean_frame_count_mae": 0.1,
                "mean_active_frames": 1.5,
                "mean_bridi_trace_exact_accuracy": 0.9,
                "mean_gismu_accuracy": 0.91,
                "mean_cmavo_accuracy": 0.92,
                "mean_judri_binding_accuracy": 0.93,
            },
            "cells": {"F": {"seed_reports": [{"metrics": {"strict_accuracy": 0.8, "mean_active_frames": 1.5}}]}},
        },
    )
    actual_path = _write_json(
        tmp_path / "m21_actual.json",
        {
            "metrics": {
                "full_accuracy": 0.81,
                "no_cmavo_accuracy": 0.4,
                "cmavo_causal_delta": 0.41,
                "no_judri_accuracy": 0.5,
                "judri_causal_delta": 0.31,
                "gismu_only_accuracy": 0.45,
                "actual_bridge_transfer_score": 0.31,
                "scratchpad_only_accuracy": 0.0,
                "accuracy_per_trace_token": 0.08,
            }
        },
    )
    pointer_path = _write_json(
        tmp_path / "m21_pointer.json",
        {
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.2,
                "mean_active_frames": 0.0,
                "mean_loss_pointer_necessity": 0.07,
                "mean_pointer_necessity_gap": 0.03,
            }
        },
    )
    gauntlet_path = _write_json(
        tmp_path / "m21_gauntlet.json",
        {
            "metrics": {
                "purged_accuracy": 0.78,
                "format_accuracy": 0.77,
                "entity_accuracy": 0.76,
                "entity_renamed_accuracy": 0.75,
                "numeric_accuracy": 0.74,
                "m19_gauntlet_worst_surface_accuracy": 0.74,
                "m19_gauntlet_order_sensitivity_spread": 0.02,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M21",
        track="M21.1",
        m21_suite_report_path=suite_path,
        m21_actual_bridge_report_path=actual_path,
        m21_pointer_microgrid_report_path=pointer_path,
        m21_gauntlet_report_path=gauntlet_path,
        history_manifest_path=None,
    )

    dynamic = next(row for row in manifest["contract_results"] if row["test_id"] == "m21.dynamic_frame_count")
    assert dynamic["metrics"]["mean_active_frames"] == 1.5
    pointer = next(row for row in manifest["contract_results"] if row["test_id"] == "m21.pointer_necessity")
    assert pointer["metrics"]["loss_pointer_necessity"] == 0.07
    assert pointer["metrics"]["full_accuracy"] == 0.81
    gauntlet = next(row for row in manifest["contract_results"] if row["test_id"] == "m21.m19_gauntlet_port")
    assert gauntlet["metrics"]["m19_gauntlet_worst_surface_accuracy"] == 0.74


def test_build_direct_unified_eval_manifest_m20_dictionary_first() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m20_dictionary_first_suite_report.json",
        {
            "track": "M20.1",
            "aggregate_metrics": {
                "mean_strict_accuracy": 1.0,
                "mean_lock_pass_rate": 1.0,
                "mean_factorized_exact_accuracy": 1.0,
                "mean_brivi_gate_accuracy": 1.0,
                "mean_predicate_identity_stability": 0.999,
                "avg_tokens": 8.6,
                "accuracy_per_token": 0.11,
            },
            "cells": {
                "A": {
                    "seed_reports": [
                        {
                            "metrics": {
                                "dictionary_coverage": 1.0,
                                "factorized_exact_accuracy": 1.0,
                                "predicate_identity_stability": 0.999,
                                "synthetic_world_accuracy": 1.0,
                                "soft_dictionary_entropy": 0.7,
                                "soft_hard_dictionary_agreement": 1.0,
                                "hard_dictionary_activation_rate": 0.007,
                                "active_code_fraction": 0.007,
                            }
                        }
                    ]
                }
            },
        },
    )
    lock_path = _write_json(
        tmp_path / "m20_lock_suite_report.json",
        {
            "metrics": {
                "brivi_gate_accuracy": 1.0,
                "brivi_formation_valid_rate": 1.0,
                "brivi_lock_violation_rate": 0.0,
                "ungrounded_predicate_energy_mean": 0.0001,
                "lock_pass_rate": 1.0,
            }
        },
    )
    induction_path = _write_json(
        tmp_path / "m20_predicate_induction_report.json",
        {
            "metrics": {
                "dictionary_coverage": 1.0,
                "oov_predicate_rate": 0.0,
                "dictionary_precedence_violation_rate": 0.0,
                "counterfactual_quotient_consistency": 1.0,
                "quotient_collision_rate": 0.0,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M20",
        track="M20.1",
        m20_suite_report_path=suite_path,
        m20_lock_report_path=lock_path,
        m20_induction_report_path=induction_path,
        history_manifest_path=None,
    )

    assert manifest["family_key"] == "M20"
    assert manifest["track"] == "M20.1"
    assert manifest["headline_metrics"]["strict_accuracy"] == 1.0
    assert manifest["headline_metrics"]["lock_pass_rate"] == 1.0

    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m20.dictionary_precedence"] == "available"
    assert statuses["m20.factorized_predicate_dictionary"] == "available"
    assert statuses["m20.counterfactual_quotient"] == "available"
    assert statuses["m20.brivi_lock"] == "available"
    assert statuses["m20.synthetic_world_pretraining"] == "available"
    assert statuses["m20.soft_dictionary_annealing"] == "available"

    rendered = render_direct_unified_eval_markdown(manifest)
    assert "Direct Unified Eval: M20 (M20.1)" in rendered
    assert "m20.dictionary_precedence" in rendered


def test_build_direct_unified_eval_manifest_m21_dynamic_bridi() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m21_dynamic_bridi_suite_report.json",
        {
            "track": "M21.1",
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.82,
                "mean_bridi_trace_exact_accuracy": 0.76,
                "mean_gismu_accuracy": 0.91,
                "mean_cmavo_accuracy": 0.84,
                "mean_judri_binding_accuracy": 0.79,
                "mean_frame_count_mae": 0.18,
                "mean_lock_pass_rate": 0.83,
                "mean_frame_drop_delta": 0.12,
                "accuracy_per_trace_token": 0.09,
            },
            "cells": {
                "F": {
                    "seed_reports": [
                        {
                            "metrics": {
                                "strict_accuracy": 0.82,
                                "bridi_trace_exact_accuracy": 0.76,
                                "gismu_accuracy": 0.91,
                                "cmavo_accuracy": 0.84,
                                "judri_binding_accuracy": 0.79,
                                "stop_accuracy": 0.95,
                                "mean_active_frames": 2.1,
                                "active_code_fraction_reachable": 0.64,
                            }
                        }
                    ]
                }
            },
        },
    )
    actual_path = _write_json(
        tmp_path / "m21_actual_bridge_report.json",
        {
            "metrics": {
                "strict_accuracy": 0.68,
                "full_accuracy": 0.68,
                "no_cmavo_accuracy": 0.43,
                "no_judri_accuracy": 0.52,
                "gismu_only_accuracy": 0.39,
                "random_trace_accuracy": 0.06,
                "scratchpad_only_accuracy": 0.12,
                "frame_drop_delta": 0.14,
                "cmavo_causal_delta": 0.25,
                "judri_causal_delta": 0.16,
                "actual_bridge_transfer_score": 0.16,
                "accuracy_per_trace_token": 0.08,
            }
        },
    )
    lock_path = _write_json(
        tmp_path / "m21_lock_suite_report.json",
        {
            "metrics": {
                "lock_pass_rate": 0.83,
                "brivi_lock_violation_rate": 0.04,
            }
        },
    )
    adversarial_path = _write_json(
        tmp_path / "m21_adversarial_audit_report.json",
        {
            "aggregate_metrics": {
                "mean_adversarial_strict_accuracy": 0.61,
                "mean_adversarial_bridi_trace_exact_accuracy": 0.58,
                "mean_adversarial_no_judri_accuracy": 0.06,
                "mean_adversarial_judri_causal_delta": 0.55,
                "mean_adversarial_worst_surface_accuracy": 0.49,
                "mean_adversarial_oov_token_rate": 0.22,
                "mean_adversarial_train_fraction": 0.25,
                "adversarial_training_exposure_rate": 1.0,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M21",
        track="M21.1",
        m21_suite_report_path=suite_path,
        m21_synthetic_assay_report_path=suite_path,
        m21_actual_bridge_report_path=actual_path,
        m21_lock_report_path=lock_path,
        m21_adversarial_audit_report_path=adversarial_path,
        history_manifest_path=None,
    )

    assert manifest["family_key"] == "M21"
    assert manifest["track"] == "M21.1"
    assert manifest["headline_metrics"]["strict_accuracy"] == 0.68
    assert manifest["headline_metrics"]["bridi_trace_exact_accuracy"] == 0.76

    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m21.dynamic_frame_count"] == "available"
    assert statuses["m21.bridi_reconstruction"] == "available"
    assert statuses["m21.cmavo_causality"] == "available"
    assert statuses["m21.judri_binding"] == "available"
    assert statuses["m21.frame_necessity"] == "available"
    assert statuses["m21.actual_bridge_transfer"] == "available"
    assert statuses["m21.adversarial_heldout"] == "available"
    assert statuses["m21.adversarial_augmentation"] == "available"
    assert statuses["m21.semantic_coverage"] == "missing"

    rendered = render_direct_unified_eval_markdown(manifest)
    assert "Direct Unified Eval: M21 (M21.1)" in rendered
    assert "m21.actual_bridge_transfer" in rendered
    assert "m21.adversarial_heldout" in rendered
    assert "m21.adversarial_augmentation" in rendered


def test_m21_adversarial_augmentation_requires_training_exposure() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m21_dynamic_bridi_suite_report.json",
        {"aggregate_metrics": {"mean_strict_accuracy": 0.82, "mean_judri_causal_delta": 0.6}},
    )
    actual_path = _write_json(
        tmp_path / "m21_actual_bridge_report.json",
        {"metrics": {"strict_accuracy": 0.82, "judri_causal_delta": 0.6}},
    )
    adversarial_path = _write_json(
        tmp_path / "m21_adversarial_audit_report.json",
        {
            "aggregate_metrics": {
                "mean_adversarial_strict_accuracy": 0.48,
                "mean_adversarial_judri_causal_delta": 0.42,
                "mean_adversarial_worst_surface_accuracy": 0.33,
                "adversarial_training_exposure_rate": 0.0,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M21",
        track="M21.1",
        m21_suite_report_path=suite_path,
        m21_actual_bridge_report_path=actual_path,
        m21_adversarial_audit_report_path=adversarial_path,
        history_manifest_path=None,
    )

    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m21.adversarial_heldout"] == "available"
    assert statuses["m21.adversarial_augmentation"] == "missing"


def test_m21_semantic_coverage_requires_semantic_training_exposure() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m21_dynamic_bridi_suite_report.json",
        {"aggregate_metrics": {"mean_strict_accuracy": 0.82, "mean_judri_causal_delta": 0.6}},
    )
    adversarial_path = _write_json(
        tmp_path / "m21_adversarial_audit_report.json",
        {
            "aggregate_metrics": {
                "mean_adversarial_strict_accuracy": 0.48,
                "mean_adversarial_judri_causal_delta": 0.42,
                "mean_adversarial_worst_surface_accuracy": 0.33,
                "adversarial_training_exposure_rate": 1.0,
                "semantic_coverage_training_exposure_rate": 0.0,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M21",
        track="M21.1",
        m21_suite_report_path=suite_path,
        m21_adversarial_audit_report_path=adversarial_path,
        history_manifest_path=None,
    )

    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m21.adversarial_augmentation"] == "available"
    assert statuses["m21.semantic_coverage"] == "missing"


def test_m21_semantic_coverage_requires_isolation_deltas_not_prefixed_metrics_only() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m21_dynamic_bridi_suite_report.json",
        {"aggregate_metrics": {"mean_strict_accuracy": 0.82, "mean_judri_causal_delta": 0.6}},
    )
    actual_path = _write_json(
        tmp_path / "m21_actual_bridge_report.json",
        {"metrics": {"strict_accuracy": 0.82, "actual_bridge_transfer_score": 0.4, "judri_causal_delta": 0.6}},
    )
    adversarial_path = _write_json(
        tmp_path / "m21_adversarial_audit_report.json",
        {
            "aggregate_metrics": {
                "mean_adversarial_strict_accuracy": 0.48,
                "mean_adversarial_judri_causal_delta": 0.42,
                "mean_adversarial_worst_surface_accuracy": 0.33,
                "adversarial_training_exposure_rate": 1.0,
                "semantic_coverage_strict_accuracy": 0.37,
                "semantic_coverage_worst_surface_accuracy": 0.29,
                "semantic_coverage_judri_causal_delta": 0.31,
                "semantic_coverage_training_exposure_rate": 1.0,
                "semantic_coverage_train_fraction": 0.25,
                "semantic_coverage_surface_count": 2.0,
                "semantic_coverage_oov_token_rate": 0.18,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M21",
        track="M21.1",
        m21_suite_report_path=suite_path,
        m21_actual_bridge_report_path=actual_path,
        m21_adversarial_audit_report_path=adversarial_path,
        history_manifest_path=None,
    )

    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m21.semantic_coverage"] == "missing"
    assert manifest["headline_metrics"]["strict_accuracy"] == 0.82
    assert manifest["headline_metrics"]["actual_bridge_transfer_score"] == 0.4
    assert manifest["headline_metrics"]["semantic_coverage_strict_accuracy"] == 0.37
    assert "semantic_coverage_lexical_shift_effect_strict_accuracy_delta" not in manifest["headline_metrics"]


def test_m21_semantic_coverage_available_with_isolation_deltas() -> None:
    tmp_path = _scratch_dir()
    suite_path = _write_json(
        tmp_path / "m21_dynamic_bridi_suite_report.json",
        {"aggregate_metrics": {"mean_strict_accuracy": 0.82, "mean_judri_causal_delta": 0.6}},
    )
    actual_path = _write_json(
        tmp_path / "m21_actual_bridge_report.json",
        {"metrics": {"strict_accuracy": 0.82, "actual_bridge_transfer_score": 0.4, "judri_causal_delta": 0.6}},
    )
    adversarial_path = _write_json(
        tmp_path / "m21_adversarial_audit_report.json",
        {
            "aggregate_metrics": {
                "mean_adversarial_strict_accuracy": 0.48,
                "mean_adversarial_judri_causal_delta": 0.42,
                "mean_adversarial_worst_surface_accuracy": 0.33,
                "adversarial_training_exposure_rate": 1.0,
                "semantic_coverage_strict_accuracy": 0.37,
                "semantic_coverage_worst_surface_accuracy": 0.29,
                "semantic_coverage_judri_causal_delta": 0.31,
                "semantic_coverage_training_exposure_rate": 1.0,
                "semantic_coverage_train_fraction": 0.25,
                "semantic_coverage_surface_count": 2.0,
                "semantic_coverage_oov_token_rate": 0.18,
                "semantic_isolation_cell_count": 8.0,
                "semantic_coverage_lexical_shift_effect_strict_accuracy_delta": 0.14,
                "semantic_coverage_role_binding_effect_strict_accuracy_delta": 0.11,
                "semantic_coverage_combined_effect_strict_accuracy_delta": 0.20,
                "semantic_coverage_fraction_effect_strict_accuracy_delta": 0.02,
                "semantic_coverage_role_curriculum_effect_strict_accuracy_delta": 0.09,
            }
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M21",
        track="M21.1",
        m21_suite_report_path=suite_path,
        m21_actual_bridge_report_path=actual_path,
        m21_adversarial_audit_report_path=adversarial_path,
        history_manifest_path=None,
    )

    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m21.semantic_coverage"] == "available"
    assert manifest["headline_metrics"]["strict_accuracy"] == 0.82
    assert manifest["headline_metrics"]["actual_bridge_transfer_score"] == 0.4
    assert manifest["headline_metrics"]["semantic_coverage_strict_accuracy"] == 0.37
    assert manifest["headline_metrics"]["semantic_coverage_lexical_shift_effect_strict_accuracy_delta"] == 0.14
    assert manifest["headline_metrics"]["semantic_coverage_role_curriculum_effect_strict_accuracy_delta"] == 0.09
    semantic_row = next(row for row in manifest["contract_results"] if row["test_id"] == "m21.semantic_coverage")
    assert semantic_row["metrics"]["semantic_coverage_surface_count"] == 2.0
    assert semantic_row["metrics"]["semantic_isolation_cell_count"] == 8.0


def test_build_direct_unified_eval_manifest_m22_semantic_generalization() -> None:
    tmp_path = _scratch_dir()
    generalization_path = _write_json(
        tmp_path / "m22_semantic_generalization_report.json",
        {
            "track": "M22",
            "metrics": {
                "strict_accuracy": 0.85,
                "bridi_trace_exact_accuracy": 0.999,
                "judri_causal_delta": 0.79,
                "semantic_coverage_strict_accuracy": 0.43,
                "semantic_coverage_worst_surface_accuracy": 0.35,
                "semantic_coverage_judri_causal_delta": 0.31,
                "m22_semantic_generalization_score": 0.31,
                "m22_semantic_strict_delta_vs_m21_control": 0.04,
                "m22_semantic_worst_delta_vs_m21_control": 0.03,
                "m22_clean_accuracy_drop_vs_m21_control": 0.0,
                "m22_judri_delta_drop_vs_m21_control": 0.01,
                "m22_promotion_gate_pass_rate": 1.0,
                "m22_promotion_candidate": 1.0,
            },
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M22",
        track="M22",
        m22_generalization_report_path=generalization_path,
        history_manifest_path=None,
    )

    assert manifest["family_key"] == "M22"
    assert manifest["track"] == "M22"
    assert manifest["headline_metrics"]["m22_semantic_generalization_score"] == 0.31
    statuses = {row["test_id"]: row["status"] for row in manifest["contract_results"]}
    assert statuses["m22.semantic_coverage_generalization"] == "available"
    assert any(row["target"] == "M21.1.O" for row in manifest["comparison_targets_resolved"])

    rendered = render_direct_unified_eval_markdown(manifest)
    assert "Direct Unified Eval: M22 (M22)" in rendered
    assert "m22.semantic_coverage_generalization" in rendered


def test_build_direct_unified_eval_manifest_m22_failed_promotion_is_not_available() -> None:
    tmp_path = _scratch_dir()
    generalization_path = _write_json(
        tmp_path / "m22_semantic_generalization_report.json",
        {
            "track": "M22",
            "metrics": {
                "strict_accuracy": 0.85,
                "bridi_trace_exact_accuracy": 0.999,
                "judri_causal_delta": 0.79,
                "semantic_coverage_strict_accuracy": 0.43,
                "semantic_coverage_worst_surface_accuracy": 0.12,
                "semantic_coverage_judri_causal_delta": 0.31,
                "m22_semantic_generalization_score": 0.12,
                "m22_semantic_strict_delta_vs_m21_control": 0.04,
                "m22_semantic_worst_delta_vs_m21_control": -0.18,
                "m22_clean_accuracy_drop_vs_m21_control": 0.0,
                "m22_judri_delta_drop_vs_m21_control": 0.01,
                "m22_promotion_gate_pass_rate": 0.8,
                "m22_promotion_candidate": 0.0,
            },
        },
    )

    manifest = build_direct_unified_eval_manifest(
        family_key="M22",
        track="M22",
        m22_generalization_report_path=generalization_path,
        history_manifest_path=None,
    )

    rows = {row["test_id"]: row for row in manifest["contract_results"]}
    semantic_row = rows["m22.semantic_coverage_generalization"]
    assert semantic_row["status"] == "missing"
    assert "failed promotion" in " ".join(semantic_row["notes"])
    assert semantic_row["metrics"]["m22_promotion_candidate"] == 0.0
