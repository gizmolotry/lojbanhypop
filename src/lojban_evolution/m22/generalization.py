from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lojban_evolution.series_contract import series_metadata


def build_m22_semantic_generalization_payload(
    *,
    suite_payload: dict[str, Any],
    adversarial_payload: dict[str, Any] | None = None,
    control_manifest_payload: dict[str, Any] | None = None,
    run_id: str = "",
    suite_report_path: str | Path | None = None,
    adversarial_audit_report_path: str | Path | None = None,
    control_manifest_path: str | Path | None = None,
    min_semantic_delta: float = 0.02,
    max_clean_drop: float = 0.02,
    min_judri_delta: float = 0.70,
) -> dict[str, Any]:
    suite_metrics = _suite_metrics(suite_payload)
    adversarial_metrics = _suite_metrics(adversarial_payload)
    control_metrics = _control_metrics(control_manifest_payload)
    suite_candidate_cells = _m22_candidate_cells(suite_payload)
    audit_candidate_cells = _m22_candidate_cells(adversarial_payload)
    candidate_cells = sorted(set(suite_candidate_cells).union(audit_candidate_cells))
    blended_candidate_cells = sorted(set(candidate_cells).intersection(_m22_blended_candidate_cells()))
    audit_blended_candidate_cells = sorted(set(audit_candidate_cells).intersection(_m22_blended_candidate_cells()))
    semantic_metric_keys = (
        "semantic_coverage_strict_accuracy",
        "semantic_coverage_worst_surface_accuracy",
        "semantic_coverage_judri_causal_delta",
    )
    relation_ood_metric_keys = (
        "m22_relation_ood_strict_accuracy",
        "m22_relation_ood_worst_surface_accuracy",
        "m22_relation_ood_judri_causal_delta",
        "m22_relation_ood_surface_count",
    )
    semantic_coverage_metrics_present = all(key in adversarial_metrics for key in semantic_metric_keys)
    relation_ood_metrics_present = all(key in adversarial_metrics for key in relation_ood_metric_keys)
    metrics: dict[str, Any] = {
        "m22_candidate_cell_count": float(len(candidate_cells)),
        "m22_candidate_cells_present": float(len(candidate_cells) > 0),
        "m22_suite_candidate_cell_count": float(len(suite_candidate_cells)),
        "m22_audit_candidate_cell_count": float(len(audit_candidate_cells)),
        "m22_audit_candidate_cells_present": float(len(audit_candidate_cells) > 0),
        "m22_blended_candidate_cell_count": float(len(blended_candidate_cells)),
        "m22_audit_blended_candidate_cell_count": float(len(audit_blended_candidate_cells)),
        "m22_audit_blended_candidate_present": float(len(audit_blended_candidate_cells) > 0),
        "semantic_coverage_metrics_present": float(semantic_coverage_metrics_present),
        "m22_relation_ood_metrics_present": float(relation_ood_metrics_present),
        "strict_accuracy": suite_metrics.get("strict_accuracy", 0.0),
        "bridi_trace_exact_accuracy": suite_metrics.get("bridi_trace_exact_accuracy", 0.0),
        "gismu_accuracy": suite_metrics.get("gismu_accuracy", 0.0),
        "cmavo_accuracy": suite_metrics.get("cmavo_accuracy", 0.0),
        "judri_binding_accuracy": suite_metrics.get("judri_binding_accuracy", 0.0),
        "cmavo_causal_delta": suite_metrics.get("cmavo_causal_delta", 0.0),
        "judri_causal_delta": suite_metrics.get("judri_causal_delta", 0.0),
        "stable_seed_rate": suite_metrics.get("stable_seed_rate", 0.0),
        "semantic_coverage_strict_accuracy": _first_present(
            adversarial_metrics,
            "semantic_coverage_strict_accuracy",
            "adversarial_strict_accuracy",
            default=0.0,
        ),
        "semantic_coverage_worst_surface_accuracy": _first_present(
            adversarial_metrics,
            "semantic_coverage_worst_surface_accuracy",
            "adversarial_worst_surface_accuracy",
            default=0.0,
        ),
        "semantic_coverage_judri_causal_delta": _first_present(
            adversarial_metrics,
            "semantic_coverage_judri_causal_delta",
            "adversarial_judri_causal_delta",
            default=0.0,
        ),
        "semantic_coverage_oov_token_rate": _first_present(
            adversarial_metrics,
            "semantic_coverage_oov_token_rate",
            "adversarial_oov_token_rate",
            default=0.0,
        ),
        "semantic_coverage_oov_synonym_accuracy": _first_present(
            adversarial_metrics,
            "semantic_coverage_oov_synonym_accuracy",
            "adversarial_oov_synonym_accuracy",
            default=0.0,
        ),
        "semantic_coverage_oov_synonym_trace_exact_accuracy": _first_present(
            adversarial_metrics,
            "semantic_coverage_oov_synonym_trace_exact_accuracy",
            "adversarial_oov_synonym_trace_exact_accuracy",
            default=0.0,
        ),
        "semantic_coverage_surface_seed_std_max": adversarial_metrics.get(
            "semantic_coverage_surface_seed_std_max", 0.0
        ),
        "semantic_coverage_surface_seed_min_accuracy": adversarial_metrics.get(
            "semantic_coverage_surface_seed_min_accuracy", 0.0
        ),
        "semantic_coverage_training_exposure_rate": adversarial_metrics.get("semantic_coverage_training_exposure_rate", 0.0),
        "semantic_isolation_cell_count": adversarial_metrics.get("semantic_isolation_cell_count", 0.0),
        "m22_relation_ood_strict_accuracy": _first_present(
            adversarial_metrics,
            "m22_relation_ood_strict_accuracy",
            "m22_relation_ood_strict_accuracy_mean",
            default=0.0,
        ),
        "m22_relation_ood_worst_surface_accuracy": _first_present(
            adversarial_metrics,
            "m22_relation_ood_worst_surface_accuracy",
            "m22_relation_ood_worst_surface_accuracy_mean",
            default=0.0,
        ),
        "m22_relation_ood_bridi_trace_exact_accuracy": _first_present(
            adversarial_metrics,
            "m22_relation_ood_bridi_trace_exact_accuracy",
            "m22_relation_ood_bridi_trace_exact_accuracy_mean",
            default=0.0,
        ),
        "m22_relation_ood_judri_causal_delta": _first_present(
            adversarial_metrics,
            "m22_relation_ood_judri_causal_delta",
            "m22_relation_ood_judri_causal_delta_mean",
            default=0.0,
        ),
        "m22_relation_ood_oov_token_rate": _first_present(
            adversarial_metrics,
            "m22_relation_ood_oov_token_rate",
            "m22_relation_ood_oov_token_rate_mean",
            default=0.0,
        ),
        "m22_relation_ood_surface_count": adversarial_metrics.get("m22_relation_ood_surface_count", 0.0),
        "m22_relation_ood_surface_seed_std_max": adversarial_metrics.get(
            "m22_relation_ood_surface_seed_std_max", 0.0
        ),
        "m22_relation_ood_surface_seed_min_accuracy": adversarial_metrics.get(
            "m22_relation_ood_surface_seed_min_accuracy", 0.0
        ),
        "m22_relation_ood_surface_training_overlap_rate": adversarial_metrics.get(
            "m22_relation_ood_surface_training_overlap_rate", 1.0
        ),
        "m21_control_strict_accuracy": control_metrics.get("strict_accuracy"),
        "m21_control_semantic_coverage_strict_accuracy": control_metrics.get("semantic_coverage_strict_accuracy"),
        "m21_control_semantic_coverage_worst_surface_accuracy": control_metrics.get(
            "semantic_coverage_worst_surface_accuracy"
        ),
        "m21_control_judri_causal_delta": control_metrics.get("judri_causal_delta"),
    }
    metrics["m22_semantic_generalization_score"] = min(
        _num(metrics.get("semantic_coverage_strict_accuracy")),
        _num(metrics.get("semantic_coverage_worst_surface_accuracy")),
        _num(metrics.get("semantic_coverage_judri_causal_delta")),
    )
    metrics["m22_hard_relation_ood_score"] = min(
        _num(metrics.get("m22_relation_ood_strict_accuracy")),
        _num(metrics.get("m22_relation_ood_worst_surface_accuracy")),
        _num(metrics.get("m22_relation_ood_judri_causal_delta")),
    )
    metrics["m22_semantic_strict_delta_vs_m21_control"] = _delta(
        metrics.get("semantic_coverage_strict_accuracy"),
        metrics.get("m21_control_semantic_coverage_strict_accuracy"),
    )
    metrics["m22_semantic_worst_delta_vs_m21_control"] = _delta(
        metrics.get("semantic_coverage_worst_surface_accuracy"),
        metrics.get("m21_control_semantic_coverage_worst_surface_accuracy"),
    )
    metrics["m22_clean_accuracy_drop_vs_m21_control"] = _delta(
        metrics.get("m21_control_strict_accuracy"),
        metrics.get("strict_accuracy"),
    )
    metrics["m22_judri_delta_drop_vs_m21_control"] = _delta(
        metrics.get("m21_control_judri_causal_delta"),
        metrics.get("judri_causal_delta"),
    )
    gates = {
        "clean_accuracy_not_collapsed": _num(metrics.get("strict_accuracy")) >= 0.80,
        "trace_reconstruction_preserved": _num(metrics.get("bridi_trace_exact_accuracy")) >= 0.99,
        "judri_causality_preserved": _num(metrics.get("judri_causal_delta")) >= float(min_judri_delta),
        "semantic_judri_causality_preserved": _num(metrics.get("semantic_coverage_judri_causal_delta"))
        >= float(min_judri_delta),
        "semantic_strict_improves_control": _num(metrics.get("m22_semantic_strict_delta_vs_m21_control"))
        >= float(min_semantic_delta),
        "semantic_worst_improves_control": _num(metrics.get("m22_semantic_worst_delta_vs_m21_control"))
        >= float(min_semantic_delta),
        "clean_drop_within_tolerance": _num(metrics.get("m22_clean_accuracy_drop_vs_m21_control")) <= float(max_clean_drop),
        "semantic_training_exposed": _num(metrics.get("semantic_coverage_training_exposure_rate")) > 0.0,
        "semantic_coverage_metrics_available": semantic_coverage_metrics_present,
        "relation_ood_metrics_available": relation_ood_metrics_present,
        "relation_ood_surfaces_complete": _num(metrics.get("m22_relation_ood_surface_count")) >= 4.0,
        "relation_ood_surfaces_unseen_in_training": _num(
            metrics.get("m22_relation_ood_surface_training_overlap_rate"), default=1.0
        )
        == 0.0,
        "relation_ood_judri_causality_preserved": _num(metrics.get("m22_relation_ood_judri_causal_delta"))
        >= float(min_judri_delta),
        "relation_ood_score_positive": _num(metrics.get("m22_hard_relation_ood_score")) > 0.0,
        "m22_candidate_cell_evidence_present": _num(metrics.get("m22_candidate_cells_present")) > 0.0,
        "m22_audit_candidate_cell_evidence_present": _num(metrics.get("m22_audit_candidate_cells_present")) > 0.0,
        "m22_blended_candidate_audit_evidence_present": _num(metrics.get("m22_audit_blended_candidate_present")) > 0.0,
        "explicit_m21_control_present": all(
            metrics.get(key) is not None
            for key in (
                "m21_control_strict_accuracy",
                "m21_control_semantic_coverage_strict_accuracy",
                "m21_control_semantic_coverage_worst_surface_accuracy",
                "m21_control_judri_causal_delta",
            )
        ),
    }
    metrics["m22_promotion_gate_pass_rate"] = sum(1.0 for value in gates.values() if value) / max(1, len(gates))
    metrics["m22_promotion_candidate"] = float(all(gates.values()))
    return {
        "series": series_metadata("M", "M22.semantic_coverage_generalization", "scripts/m22/run_m22_semantic_generalization.py"),
        "track": "M22",
        "family_version": "0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id or f"m22_semantic_generalization_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "source_reports": {
            "m21_suite_report": str(suite_report_path or ""),
            "m21_adversarial_audit_report": str(adversarial_audit_report_path or ""),
            "m21_control_direct_manifest": str(control_manifest_path or ""),
        },
        "candidate_cells": candidate_cells,
        "suite_candidate_cells": suite_candidate_cells,
        "audit_candidate_cells": audit_candidate_cells,
        "blended_candidate_cells": blended_candidate_cells,
        "audit_blended_candidate_cells": audit_blended_candidate_cells,
        "comparison_policy": {
            "delta_baseline": "explicit_m21_control_direct_manifest",
            "candidate_suite": "m21_suite_report",
            "candidate_reserved_semantic_audit": "m21_adversarial_audit_report",
            "candidate_hard_relation_ood_audit": "m21_adversarial_audit_report",
            "promotion_requires_audited_blended_candidate": sorted(_m22_blended_candidate_cells()),
            "promotion_requires_hard_relation_ood_surfaces": [
                "relation_composition_ood",
                "role_inversion_ood",
                "polarity_scope_ood",
                "decoy_relation_ood",
            ],
        },
        "metrics": metrics,
        "promotion_gates": gates,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": [
            "phrase_accuracy",
            "m22_relation_ood_oov_token_rate",
            "m22_relation_ood_surface_seed_std_max",
            "m22_relation_ood_surface_seed_min_accuracy",
        ],
        "notes": [
            "M22 is a semantic-generalization layer over M21; it does not introduce a new model architecture.",
            "Promotion requires semantic coverage lift over the explicit M21 control manifest without clean-accuracy or judri-causality regression.",
            "M22 deltas are not implicitly computed against the candidate M21 suite unless that direct manifest is passed as the control.",
        ],
    }


def _suite_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    metrics: dict[str, Any] = {}
    aggregate = payload.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
        metrics.update(aggregate)
        aliases = {
            "strict_accuracy": "mean_strict_accuracy",
            "bridi_trace_exact_accuracy": "mean_bridi_trace_exact_accuracy",
            "gismu_accuracy": "mean_gismu_accuracy",
            "cmavo_accuracy": "mean_cmavo_accuracy",
            "judri_binding_accuracy": "mean_judri_binding_accuracy",
            "cmavo_causal_delta": "mean_cmavo_causal_delta",
            "judri_causal_delta": "mean_judri_causal_delta",
            "stable_seed_rate": "stable_seed_rate",
            "adversarial_strict_accuracy": "mean_adversarial_strict_accuracy",
            "adversarial_worst_surface_accuracy": "mean_adversarial_worst_surface_accuracy",
            "adversarial_judri_causal_delta": "mean_adversarial_judri_causal_delta",
            "adversarial_oov_token_rate": "mean_adversarial_oov_token_rate",
            "adversarial_oov_synonym_accuracy": "mean_adversarial_oov_synonym_accuracy",
            "adversarial_oov_synonym_trace_exact_accuracy": "mean_adversarial_oov_synonym_trace_exact_accuracy",
        }
        for canonical, source in aliases.items():
            if source in aggregate:
                metrics.setdefault(canonical, aggregate.get(source))
    top_metrics = payload.get("metrics", {})
    if isinstance(top_metrics, dict):
        metrics.update(top_metrics)
    headline = payload.get("headline_metrics", {})
    if isinstance(headline, dict):
        metrics.update(headline)
    return {key: value for key, value in metrics.items() if value is not None}


def _control_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    headline = payload.get("headline_metrics", {})
    return dict(headline) if isinstance(headline, dict) else {}


def _m22_candidate_cells(*payloads: dict[str, Any] | None) -> list[str]:
    allowed = {"P", "Q", "R", "S", "T"}
    cells: set[str] = set()
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        raw_cells = payload.get("cells")
        if isinstance(raw_cells, dict):
            cells.update(str(key).strip().upper() for key in raw_cells if str(key).strip().upper() in allowed)
        for row in payload.get("seed_reports", []) if isinstance(payload.get("seed_reports"), list) else []:
            if isinstance(row, dict):
                cell = str(row.get("cell_key", "")).strip().upper()
                if cell in allowed:
                    cells.add(cell)
    return sorted(cells)


def _m22_blended_candidate_cells() -> set[str]:
    return {"S", "T"}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _first_present(metrics: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return default


def _delta(value: Any, baseline: Any) -> float:
    if baseline is None or value is None:
        return 0.0
    return _num(value) - _num(baseline)
