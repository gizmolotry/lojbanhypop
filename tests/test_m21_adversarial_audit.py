from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_audit_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "m21" / "run_m21_adversarial_audit.py"
    spec = importlib.util.spec_from_file_location("run_m21_adversarial_audit", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m21_adversarial_audit_summarizes_semantic_isolation_deltas() -> None:
    audit = _load_audit_module()
    rows = [
        _row("H", "heldout_paraphrase,clausal_permutation", 0.30, 0.25, 0.20),
        _row("I", "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train", 0.50, 0.40, 0.45),
        _row("J", "heldout_paraphrase,clausal_permutation,lexical_shift_train", 0.44, 0.35, 0.38),
        _row("K", "heldout_paraphrase,clausal_permutation,role_binding_train", 0.41, 0.34, 0.36),
        _row("L", "heldout_paraphrase,clausal_permutation,lexical_shift_train,role_binding_train", 0.52, 0.42, 0.46, fraction=0.50),
        _row("M", "heldout_paraphrase,clausal_permutation,role_binding_train,role_binding_pair_train,role_binding_swap_train,role_binding_chain_train", 0.55, 0.45, 0.50),
        _row("N", "heldout_paraphrase,clausal_permutation,role_binding_swap_train", 0.47, 0.39, 0.41),
        _row("O", "heldout_paraphrase,clausal_permutation,role_binding_train,role_binding_pair_train,role_binding_swap_train,role_binding_chain_train", 0.53, 0.43, 0.48, fraction=0.35),
    ]

    summary = audit._summarize(rows)

    assert summary["semantic_isolation_cell_count"] == 8.0
    assert summary["mean_adversarial_oov_synonym_accuracy"] == pytest.approx(0.30)
    assert summary["semantic_coverage_oov_synonym_accuracy"] == pytest.approx(0.30)
    assert summary["semantic_coverage_surface_seed_std_max"] > 0.0
    assert summary["semantic_coverage_surface_accuracy"]["oov_synonym"]["seed_count"] == 7.0
    assert summary["semantic_coverage_lexical_shift_effect_strict_accuracy_delta"] == pytest.approx(0.14)
    assert summary["semantic_coverage_role_binding_effect_strict_accuracy_delta"] == pytest.approx(0.11)
    assert summary["semantic_coverage_combined_effect_strict_accuracy_delta"] == pytest.approx(0.20)
    assert summary["semantic_coverage_fraction_effect_strict_accuracy_delta"] == pytest.approx(0.02)
    assert summary["semantic_coverage_lexical_shift_effect_worst_surface_accuracy_delta"] == pytest.approx(0.10)
    assert summary["semantic_coverage_role_binding_effect_judri_causal_delta_delta"] == pytest.approx(0.16)
    assert summary["semantic_coverage_role_curriculum_effect_strict_accuracy_delta"] == pytest.approx(0.14)
    assert summary["semantic_coverage_role_swap_effect_worst_surface_accuracy_delta"] == pytest.approx(0.14)
    assert summary["semantic_coverage_role_curriculum_fraction_effect_judri_causal_delta_delta"] == pytest.approx(-0.02)


def test_semantic_coverage_surface_count_includes_m22_generalization_surfaces() -> None:
    audit = _load_audit_module()

    count = audit._semantic_coverage_surface_count(
        {
            "adversarial_train_surfaces": (
                "heldout_paraphrase,"
                "role_binding_train,"
                "role_binding_swap_train,"
                "relational_synonym_train,"
                "role_chain_generalization_train,"
                "polarity_reframe_train"
            )
        }
    )

    assert count == 5


def test_m22_semantic_cells_count_as_isolation_evidence_without_breaking_h_o_effects() -> None:
    audit = _load_audit_module()
    rows = [
        _row("P", "heldout_paraphrase,clausal_permutation,relational_synonym_train", 0.60, 0.30, 0.50),
        _row("Q", "heldout_paraphrase,clausal_permutation,role_chain_generalization_train", 0.62, 0.28, 0.52),
        _row("R", "heldout_paraphrase,clausal_permutation,polarity_reframe_train", 0.58, 0.27, 0.49),
        _row(
            "S",
            "heldout_paraphrase,clausal_permutation,role_binding_train,role_binding_pair_train,role_binding_swap_train,role_binding_chain_train,relational_synonym_train,role_chain_generalization_train,polarity_reframe_train",
            0.70,
            0.33,
            0.60,
        ),
    ]

    summary = audit._summarize(rows)

    assert summary["semantic_isolation_cell_count"] == 4.0
    assert summary["semantic_isolation_p_strict_accuracy"] == 0.60
    assert summary["semantic_isolation_q_worst_surface_accuracy"] == 0.28
    assert summary["semantic_isolation_s_judri_causal_delta"] == 0.60
    assert summary["semantic_coverage_oov_synonym_accuracy"] == pytest.approx(0.30)
    assert "semantic_coverage_lexical_shift_effect_strict_accuracy_delta" not in summary


def test_m22_relation_ood_metrics_report_seed_surface_and_overlap_diagnostics() -> None:
    audit = _load_audit_module()
    rows = [
        _relation_ood_row(
            "S",
            "relation_composition_ood,role_binding_train",
            {
                "relation_composition_ood": (0.80, 0.90, 4.0),
                "role_inversion_ood": (0.60, 0.80, 4.0),
                "polarity_scope_ood": (0.70, 0.85, 4.0),
                "decoy_relation_ood": (0.50, 0.75, 4.0),
            },
        ),
        _relation_ood_row(
            "S",
            "role_binding_train",
            {
                "relation_composition_ood": (0.70, 0.88, 4.0),
                "role_inversion_ood": (0.55, 0.78, 4.0),
                "polarity_scope_ood": (0.65, 0.83, 4.0),
                "decoy_relation_ood": (0.45, 0.73, 4.0),
            },
        ),
    ]

    summary = audit._summarize(rows)

    assert summary["m22_relation_ood_seed_count"] == 2.0
    assert summary["m22_relation_ood_surface_count"] == 4.0
    assert summary["m22_relation_ood_surface_count_mean"] == 4.0
    assert summary["m22_relation_ood_strict_accuracy_mean"] == pytest.approx(0.61875)
    assert summary["m22_relation_ood_strict_accuracy_min"] == pytest.approx(0.5875)
    assert summary["m22_relation_ood_worst_surface_accuracy_mean"] == pytest.approx(0.475)
    assert summary["m22_relation_ood_bridi_trace_exact_accuracy_mean"] == pytest.approx(0.815)
    assert summary["m22_relation_ood_training_overlap_rate"] == pytest.approx(0.125)
    assert summary["m22_relation_ood_training_overlap_surface_count_mean"] == pytest.approx(0.5)
    assert summary["m22_relation_ood_surface_accuracy"]["decoy_relation_ood"]["min"] == pytest.approx(0.45)
    assert summary["m22_relation_ood_surface_trace_exact_accuracy"]["role_inversion_ood"]["mean"] == pytest.approx(0.79)
    assert summary["m22_relation_ood_surface_seed_std_max"] > 0.0
    assert summary["m22_relation_ood_surface_seed_min_accuracy"] == pytest.approx(0.45)


def test_m22_relation_ood_audit_profile_resolves_surfaces_without_breaking_overrides() -> None:
    audit = _load_audit_module()

    profiled_args = audit.parse_args(["--suite-report", "suite.json", "--audit-profile", "m22-relation-ood"])
    override_args = audit.parse_args(
        [
            "--suite-report",
            "suite.json",
            "--audit-profile",
            "m22-relation-ood",
            "--surfaces",
            "oov_synonym",
        ]
    )

    assert audit._resolve_audit_surfaces(profiled_args) == audit.M22_RELATION_OOD_AUDIT_SURFACES
    assert audit._resolve_audit_surfaces(override_args) == ("oov_synonym",)


def _row(
    cell: str,
    surfaces: str,
    strict: float,
    worst: float,
    judri_delta: float,
    *,
    fraction: float = 0.25,
) -> dict[str, object]:
    return {
        "cell_key": cell,
        "config": {
            "adversarial_train_fraction": fraction,
            "adversarial_train_surfaces": surfaces,
        },
        "metrics": {
            "adversarial_strict_accuracy": strict,
            "adversarial_worst_surface_accuracy": worst,
            "adversarial_judri_causal_delta": judri_delta,
            "adversarial_oov_synonym_accuracy": 0.30,
            "adversarial_oov_token_rate": 0.1,
            "surface_metrics": {
                "heldout_paraphrase": {"strict_accuracy": strict},
                "oov_synonym": {"strict_accuracy": 0.30},
                "role_distractor": {"strict_accuracy": worst},
            },
        },
    }


def _relation_ood_row(
    cell: str,
    train_surfaces: str,
    relation_surface_metrics: dict[str, tuple[float, float, float]],
) -> dict[str, object]:
    strict = sum(item[0] for item in relation_surface_metrics.values()) / len(relation_surface_metrics)
    trace = sum(item[1] for item in relation_surface_metrics.values()) / len(relation_surface_metrics)
    return {
        "cell_key": cell,
        "config": {
            "adversarial_train_fraction": 0.25,
            "adversarial_train_surfaces": train_surfaces,
        },
        "metrics": {
            "adversarial_strict_accuracy": strict,
            "adversarial_bridi_trace_exact_accuracy": trace,
            "adversarial_worst_surface_accuracy": min(item[0] for item in relation_surface_metrics.values()),
            "adversarial_judri_causal_delta": 0.50,
            "adversarial_oov_synonym_accuracy": 0.0,
            "adversarial_oov_token_rate": 0.0,
            "surface_metrics": {
                surface: {
                    "strict_accuracy": strict_accuracy,
                    "bridi_trace_exact_accuracy": trace_accuracy,
                    "count": count,
                }
                for surface, (strict_accuracy, trace_accuracy, count) in relation_surface_metrics.items()
            },
        },
    }
