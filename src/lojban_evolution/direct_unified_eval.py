from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_taxonomy import build_comparison_index, load_taxonomy_config
from .m19.family import M19_REGISTRY
from .repo_paths import REPO_ROOT, repo_relative
from .series_contract import series_metadata


DIRECT_UNIFIED_EVAL_VERSION = "1.0"
DIRECT_UNIFIED_EVAL_OUTPUT_ROOT = (
    REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "direct_unified_eval"
)

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

    surfaces: dict[str, dict[str, Any]] = {
        "benchmark": _surface_record("benchmark", benchmark_path),
        "audit": _surface_record("audit", audit_path),
        "integrity": _surface_record("integrity", integrity_path),
        "replication": _surface_record("replication", replication_path),
        "stability_microgrid": _surface_record("stability_microgrid", stability_path),
        "kill_tests": _surface_record("kill_tests", kill_test_path),
    }
    return surfaces


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
    history_manifest_path: Path | None = None,
    taxonomy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    taxonomy = taxonomy or load_taxonomy_config()
    comparison_index = build_comparison_index(taxonomy)
    if family_key not in comparison_index:
        raise ValueError(f"Unknown family_key '{family_key}'.")
    contract = comparison_index[family_key]
    if family_key != "M19":
        raise NotImplementedError(f"Direct unified eval is currently implemented for family '{family_key}' only.")

    resolved_track = _track_key(track or family_key)
    direct_surfaces = discover_m19_surfaces(
        track=resolved_track,
        benchmark_report_path=benchmark_report_path,
        audit_report_path=audit_report_path,
        integrity_report_path=integrity_report_path,
        replication_report_path=replication_report_path,
        stability_report_path=stability_report_path,
        kill_test_report_path=kill_test_report_path,
    )
    benchmark_payload = direct_surfaces["benchmark"]["payload"]
    audit_payload = direct_surfaces["audit"]["payload"]
    integrity_payload = direct_surfaces["integrity"]["payload"]
    replication_payload = direct_surfaces["replication"]["payload"]
    stability_payload = direct_surfaces["stability_microgrid"]["payload"]
    kill_test_payload = direct_surfaces["kill_tests"]["payload"]
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
        reference_surface_index=reference_surface_index,
    )

    headline_metrics = _build_headline_metrics(
        benchmark_payload,
        audit_payload,
        integrity_payload,
        replication_payload,
        stability_payload,
        kill_test_payload,
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
    if resolved_track != "M19":
        notes.append(f"Track {resolved_track} is evaluated under the M19 family contract.")

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
    reference_surface_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
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
            )
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


def _evaluate_m19_contract(
    test_id: str,
    contract: dict[str, Any],
    benchmark_payload: dict[str, Any] | None,
    audit_payload: dict[str, Any] | None,
    integrity_payload: dict[str, Any] | None,
    replication_payload: dict[str, Any] | None,
    stability_payload: dict[str, Any] | None,
    kill_test_payload: dict[str, Any] | None,
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
    return {k: v for k, v in headline.items() if v is not None}


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
    if not root.exists():
        return None
    matches = [path for path in root.rglob(file_name) if _path_allowed_for_discovery(path)]
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return matches[0]


def _latest_json(root: Path, preferred_names: list[str] | None = None) -> Path | None:
    if not root.exists():
        return None
    preferred_names = preferred_names or []
    candidates = [path for path in root.rglob("*.json") if _path_allowed_for_discovery(path)]
    if not candidates:
        return None
    ranked: list[tuple[int, float, Path]] = []
    for path in candidates:
        score = 0
        if path.name in preferred_names:
            score += 100
        if "manifest" in path.name:
            score += 20
        if "report" in path.name:
            score += 20
        if "summary" in path.name:
            score += 10
        ranked.append((score, path.stat().st_mtime, path))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


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
    return []


def _path_allowed_for_discovery(path: Path) -> bool:
    bad_parts = {"__pycache__"}
    for part in path.parts:
        lowered = part.lower()
        if lowered in bad_parts:
            return False
        if lowered.startswith("test_"):
            return False
    return True


def _read_json_optional(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _repo_string(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return repo_relative(path)
    except ValueError:
        return str(path).replace("\\", "/")


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
