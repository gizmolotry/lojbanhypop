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
    metrics: dict[str, Any] = {
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
        "semantic_coverage_training_exposure_rate": adversarial_metrics.get("semantic_coverage_training_exposure_rate", 0.0),
        "semantic_isolation_cell_count": adversarial_metrics.get("semantic_isolation_cell_count", 0.0),
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
        "semantic_strict_improves_control": _num(metrics.get("m22_semantic_strict_delta_vs_m21_control"))
        >= float(min_semantic_delta),
        "semantic_worst_improves_control": _num(metrics.get("m22_semantic_worst_delta_vs_m21_control"))
        >= float(min_semantic_delta),
        "clean_drop_within_tolerance": _num(metrics.get("m22_clean_accuracy_drop_vs_m21_control")) <= float(max_clean_drop),
        "semantic_training_exposed": _num(metrics.get("semantic_coverage_training_exposure_rate")) > 0.0,
        "semantic_isolation_evidence_present": _num(metrics.get("semantic_isolation_cell_count")) > 0.0,
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
        "comparison_policy": {
            "delta_baseline": "explicit_m21_control_direct_manifest",
            "candidate_suite": "m21_suite_report",
            "candidate_reserved_semantic_audit": "m21_adversarial_audit_report",
        },
        "metrics": metrics,
        "promotion_gates": gates,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
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
