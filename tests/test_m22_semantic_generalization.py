from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lojban_evolution.m22.generalization import build_m22_semantic_generalization_payload
from lojban_evolution.m22.family import m22_track_spec


def _load_m22_runner():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "m22" / "run_m22_semantic_generalization.py"
    spec = importlib.util.spec_from_file_location("run_m22_semantic_generalization", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_m22_seed_stability_runner():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "m22" / "run_m22_seed_stability_aggregate.py"
    spec = importlib.util.spec_from_file_location("run_m22_seed_stability_aggregate", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_m22_registry_tracks_explicit_pqr_candidate_cells() -> None:
    targets = set(m22_track_spec()["comparison_targets"])

    assert {"M21.1.P", "M21.1.Q", "M21.1.R", "M21.1.S", "M21.1.T"}.issubset(targets)


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
        },
        "cells": {"P": {}, "Q": {}, "R": {}, "S": {}},
    }
    adversarial_payload = {
        "aggregate_metrics": {
            "semantic_coverage_strict_accuracy": 0.43,
            "semantic_coverage_worst_surface_accuracy": 0.35,
            "semantic_coverage_judri_causal_delta": 0.79,
            "semantic_coverage_oov_token_rate": 0.12,
            "semantic_coverage_oov_synonym_accuracy": 0.34,
            "semantic_coverage_surface_seed_std_max": 0.04,
            "semantic_coverage_surface_seed_min_accuracy": 0.29,
            "semantic_coverage_training_exposure_rate": 1.0,
            "semantic_isolation_cell_count": 8.0,
            **_hard_relation_ood_metrics(),
        },
        "seed_reports": [{"cell_key": "P"}, {"cell_key": "Q"}, {"cell_key": "R"}, {"cell_key": "S"}],
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
    assert metrics["m22_candidate_cell_count"] == 4.0
    assert metrics["m22_audit_candidate_cell_count"] == 4.0
    assert metrics["m22_audit_blended_candidate_present"] == 1.0
    assert metrics["semantic_coverage_oov_synonym_accuracy"] == 0.34
    assert metrics["semantic_coverage_surface_seed_std_max"] == 0.04
    assert metrics["m22_relation_ood_strict_accuracy"] == 0.42
    assert metrics["m22_hard_relation_ood_score"] == 0.36
    assert payload["promotion_gates"]["relation_ood_metrics_available"] is True
    assert metrics["m22_promotion_candidate"] == 1.0
    assert payload["candidate_cells"] == ["P", "Q", "R", "S"]
    assert payload["comparison_policy"]["delta_baseline"] == "explicit_m21_control_direct_manifest"


def test_m22_generalization_gate_blocks_clean_or_semantic_regression() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.79,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.69,
            },
            "cells": {"P": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.38,
                "semantic_coverage_worst_surface_accuracy": 0.28,
                "semantic_coverage_judri_causal_delta": 0.20,
            },
            "seed_reports": [{"cell_key": "P"}],
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
            },
            "cells": {"P": {}},
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
    assert payload["promotion_gates"]["m22_candidate_cell_evidence_present"] is True
    assert payload["promotion_gates"]["m22_audit_candidate_cell_evidence_present"] is False
    assert payload["promotion_gates"]["m22_blended_candidate_audit_evidence_present"] is False
    assert payload["promotion_gates"]["explicit_m21_control_present"] is False


def test_m22_generalization_gate_requires_explicit_pqr_candidate_cells() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"H": {}, "I": {}, "O": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.95,
                "semantic_coverage_worst_surface_accuracy": 0.95,
                "semantic_coverage_judri_causal_delta": 0.95,
                "semantic_coverage_training_exposure_rate": 1.0,
                "semantic_isolation_cell_count": 8.0,
            },
            "seed_reports": [{"cell_key": "H"}, {"cell_key": "I"}, {"cell_key": "O"}],
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

    assert payload["candidate_cells"] == []
    assert payload["metrics"]["m22_candidate_cell_count"] == 0.0
    assert payload["promotion_gates"]["m22_candidate_cell_evidence_present"] is False
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_generalization_gate_rejects_h_o_audit_for_m22_candidate_suite() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"S": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.95,
                "semantic_coverage_worst_surface_accuracy": 0.95,
                "semantic_coverage_judri_causal_delta": 0.95,
                "semantic_coverage_training_exposure_rate": 1.0,
            },
            "seed_reports": [{"cell_key": "H"}, {"cell_key": "I"}, {"cell_key": "O"}],
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

    assert payload["suite_candidate_cells"] == ["S"]
    assert payload["audit_candidate_cells"] == []
    assert payload["promotion_gates"]["m22_audit_candidate_cell_evidence_present"] is False
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_generalization_gate_requires_blended_candidate_not_p_only() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"P": {}, "Q": {}, "R": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.95,
                "semantic_coverage_worst_surface_accuracy": 0.95,
                "semantic_coverage_judri_causal_delta": 0.95,
                "semantic_coverage_training_exposure_rate": 1.0,
            },
            "seed_reports": [{"cell_key": "P"}, {"cell_key": "Q"}, {"cell_key": "R"}],
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

    assert payload["audit_candidate_cells"] == ["P", "Q", "R"]
    assert payload["audit_blended_candidate_cells"] == []
    assert payload["promotion_gates"]["m22_blended_candidate_audit_evidence_present"] is False
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_generalization_gate_blocks_low_semantic_judri_delta() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"S": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.95,
                "semantic_coverage_worst_surface_accuracy": 0.95,
                "semantic_coverage_judri_causal_delta": 0.05,
                "semantic_coverage_training_exposure_rate": 1.0,
            },
            "seed_reports": [{"cell_key": "S"}],
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

    assert payload["promotion_gates"]["semantic_judri_causality_preserved"] is False
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_generalization_gate_does_not_promote_generic_adversarial_fallback() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"S": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "mean_adversarial_strict_accuracy": 0.95,
                "mean_adversarial_worst_surface_accuracy": 0.95,
                "mean_adversarial_judri_causal_delta": 0.95,
                "semantic_coverage_training_exposure_rate": 1.0,
            },
            "seed_reports": [{"cell_key": "S"}],
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

    assert payload["metrics"]["semantic_coverage_strict_accuracy"] == 0.95
    assert payload["promotion_gates"]["semantic_coverage_metrics_available"] is False
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_generalization_gate_requires_hard_relation_ood_evidence() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"S": {}},
        },
        adversarial_payload={
            "aggregate_metrics": {
                "semantic_coverage_strict_accuracy": 0.95,
                "semantic_coverage_worst_surface_accuracy": 0.95,
                "semantic_coverage_judri_causal_delta": 0.95,
                "semantic_coverage_training_exposure_rate": 1.0,
            },
            "seed_reports": [{"cell_key": "S"}],
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

    assert payload["promotion_gates"]["relation_ood_metrics_available"] is False
    assert payload["promotion_gates"]["relation_ood_score_positive"] is False
    assert payload["metrics"]["m22_promotion_candidate"] == 0.0


def test_m22_generalization_preserves_explicit_zero_semantic_metrics() -> None:
    payload = build_m22_semantic_generalization_payload(
        suite_payload={
            "aggregate_metrics": {
                "mean_strict_accuracy": 0.85,
                "mean_bridi_trace_exact_accuracy": 0.999,
                "mean_judri_causal_delta": 0.79,
            },
            "cells": {"P": {}},
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
            },
            "seed_reports": [{"cell_key": "P"}],
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
                },
                "cells": {"S": {}},
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
                    "semantic_coverage_judri_causal_delta": 0.79,
                    "semantic_coverage_training_exposure_rate": 1.0,
                    "semantic_isolation_cell_count": 8.0,
                    **_hard_relation_ood_metrics(),
                },
                "seed_reports": [{"cell_key": "S"}],
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
    assert payload["candidate_cells"] == ["S"]
    assert payload["source_reports"]["m21_suite_report"] == str(suite_path)
    assert "metrics" in payload
    assert "promotion_gates" in payload


def test_m22_seed_stability_aggregate_reports_ood_accuracy_and_surface_variance(tmp_path: Path) -> None:
    runner = _load_m22_seed_stability_runner()
    suite_path = tmp_path / "suite.json"
    audit_path = tmp_path / "audit.json"
    gate_path = tmp_path / "gate.json"
    suite_path.write_text(
        json.dumps(
            {
                "run_id": "suite_a",
                "cells": {
                    "S": {
                        "seed_reports": [
                            {"seed": 23, "metrics": {"strict_accuracy": 0.85, "bridi_trace_exact_accuracy": 0.999, "judri_causal_delta": 0.79}},
                            {"seed": 29, "metrics": {"strict_accuracy": 0.83, "bridi_trace_exact_accuracy": 0.998, "judri_causal_delta": 0.78}},
                        ]
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps(
            {
                "run_id": "audit_a",
                "seed_reports": [
                    {
                        "cell_key": "S",
                        "seed": 23,
                        "metrics": {
                            "adversarial_strict_accuracy": 0.84,
                            "adversarial_worst_surface_accuracy": 0.31,
                            "adversarial_judri_causal_delta": 0.80,
                            "adversarial_oov_synonym_accuracy": 0.31,
                            "adversarial_oov_token_rate": 0.04,
                            "surface_metrics": {
                                "oov_synonym": {"strict_accuracy": 0.31},
                                "role_distractor": {"strict_accuracy": 0.70},
                            },
                        },
                    },
                    {
                        "cell_key": "S",
                        "seed": 29,
                        "metrics": {
                            "adversarial_strict_accuracy": 0.82,
                            "adversarial_worst_surface_accuracy": 0.27,
                            "adversarial_judri_causal_delta": 0.77,
                            "adversarial_oov_synonym_accuracy": 0.27,
                            "adversarial_oov_token_rate": 0.05,
                            "surface_metrics": {
                                "oov_synonym": {"strict_accuracy": 0.27},
                                "role_distractor": {"strict_accuracy": 0.72},
                            },
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    gate_path.write_text(
        json.dumps(
            {
                "run_id": "gate_a",
                "metrics": {
                    "strict_accuracy": 0.84,
                    "semantic_coverage_strict_accuracy": 0.83,
                    "semantic_coverage_worst_surface_accuracy": 0.29,
                    "semantic_coverage_oov_synonym_accuracy": 0.29,
                    **_hard_relation_ood_metrics(),
                    "m22_promotion_candidate": 1.0,
                },
                "promotion_gates": {
                    "clean_accuracy_not_collapsed": True,
                    "trace_reconstruction_preserved": True,
                    "judri_causality_preserved": True,
                    "semantic_judri_causality_preserved": True,
                    "semantic_strict_improves_control": True,
                    "semantic_worst_improves_control": True,
                    "clean_drop_within_tolerance": True,
                    "semantic_training_exposed": True,
                    "semantic_coverage_metrics_available": True,
                    "relation_ood_metrics_available": True,
                    "relation_ood_surfaces_complete": True,
                    "relation_ood_surfaces_unseen_in_training": True,
                    "relation_ood_judri_causality_preserved": True,
                    "relation_ood_score_positive": True,
                    "m22_candidate_cell_evidence_present": True,
                    "m22_audit_candidate_cell_evidence_present": True,
                    "m22_blended_candidate_audit_evidence_present": True,
                    "explicit_m21_control_present": True,
                },
            }
        ),
        encoding="utf-8",
    )

    output_root = Path("artifacts/runs/telemetry/raw/ablation/hypercube/m22_seed_stability_test")
    args = runner.parse_args(
        [
            "--suite-reports",
            str(suite_path),
            "--adversarial-audit-reports",
            str(audit_path),
            "--gate-reports",
            str(gate_path),
            "--output-root",
            str(output_root),
            "--run-id",
            "fixture",
        ]
    )
    payload = runner.run_seed_stability(args)
    metrics = payload["metrics"]

    assert metrics["m22_seed_stability_suite_seed_count"] == 2.0
    assert metrics["suite_seed_strict_accuracy_mean"] == 0.84
    assert metrics["audit_seed_adversarial_oov_synonym_accuracy_mean"] == 0.29000000000000004
    assert metrics["m22_seed_stability_surface_accuracy"]["oov_synonym"]["std"] == pytest.approx(0.02)
    assert metrics["m22_seed_stability_promotion_rate"] == 1.0
    assert metrics["m22_seed_stability_gate_evidence_rate"] == 1.0
    assert (output_root / "fixture" / "m22_seed_stability_report.json").exists()


def _hard_relation_ood_metrics() -> dict[str, float]:
    return {
        "m22_relation_ood_strict_accuracy": 0.42,
        "m22_relation_ood_worst_surface_accuracy": 0.36,
        "m22_relation_ood_bridi_trace_exact_accuracy": 0.98,
        "m22_relation_ood_judri_causal_delta": 0.79,
        "m22_relation_ood_oov_token_rate": 0.04,
        "m22_relation_ood_surface_count": 4.0,
        "m22_relation_ood_surface_seed_std_max": 0.03,
        "m22_relation_ood_surface_seed_min_accuracy": 0.34,
        "m22_relation_ood_surface_training_overlap_rate": 0.0,
    }
