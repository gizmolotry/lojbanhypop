from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import re
from typing import Any


ABLATION_HISTORY_SCHEMA_VERSION = "1.0"
EVIDENCE_CLASS_ORDER = {"git_reported": 0, "doc_reported": 1, "artifact": 2, "mixed": 3}
CONFIDENCE_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2}
REPRODUCIBILITY_ORDER = {
    "git_only": 0,
    "doc_only": 1,
    "orphaned": 2,
    "artifact_only": 3,
    "partial": 4,
    "runnable": 5,
}


def slugify(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def unique_list(values: list[Any]) -> list[Any]:
    seen: set[str] = set()
    output: list[Any] = []
    for value in values:
        if value is None:
            continue
        key = str(value).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(value)
    return output


def ensure_entry(
    registry: dict[str, dict[str, Any]],
    *,
    canonical_id: str,
    family: str,
    title: str,
    aliases: list[str] | None = None,
    lookup_aliases: list[str] | None = None,
    objective: str | None = None,
    baseline_relation: str | None = None,
    family_group: str | None = None,
    script_paths: list[str] | None = None,
    dag_paths: list[str] | None = None,
    artifact_roots: list[str] | None = None,
    derived_from: list[str] | None = None,
    supersedes: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    if canonical_id not in registry:
        registry[canonical_id] = {
            "canonical_id": canonical_id,
            "family": family,
            "family_group": family_group or family,
            "title": title,
            "aliases": unique_list(list(aliases or [])),
            "lookup_aliases": unique_list(list(lookup_aliases or [])),
            "objective": objective,
            "baseline_relation": baseline_relation,
            "script_paths": unique_list(list(script_paths or [])),
            "dag_paths": unique_list(list(dag_paths or [])),
            "artifact_roots": unique_list(list(artifact_roots or [])),
            "derived_from": unique_list(list(derived_from or [])),
            "supersedes": unique_list(list(supersedes or [])),
            "notes": unique_list(list(notes or [])),
            "evidence_records": [],
        }
        return registry[canonical_id]

    entry = registry[canonical_id]
    entry["aliases"] = unique_list([*entry.get("aliases", []), *(aliases or [])])
    entry["lookup_aliases"] = unique_list([*entry.get("lookup_aliases", []), *(lookup_aliases or [])])
    entry["script_paths"] = unique_list([*entry.get("script_paths", []), *(script_paths or [])])
    entry["dag_paths"] = unique_list([*entry.get("dag_paths", []), *(dag_paths or [])])
    entry["artifact_roots"] = unique_list([*entry.get("artifact_roots", []), *(artifact_roots or [])])
    entry["derived_from"] = unique_list([*entry.get("derived_from", []), *(derived_from or [])])
    entry["supersedes"] = unique_list([*entry.get("supersedes", []), *(supersedes or [])])
    entry["notes"] = unique_list([*entry.get("notes", []), *(notes or [])])
    if objective and not entry.get("objective"):
        entry["objective"] = objective
    if baseline_relation and not entry.get("baseline_relation"):
        entry["baseline_relation"] = baseline_relation
    return entry


def build_evidence_record(
    *,
    source_label: str,
    source_paths: list[str],
    evidence_class: str,
    confidence_level: str,
    reproducibility_status: str,
    metrics: dict[str, Any] | None = None,
    normalized_metrics: dict[str, float | None] | None = None,
    reported_at: str | None = None,
    notes: str | None = None,
    lineage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "source_label": source_label,
        "source_paths": unique_list(list(source_paths)),
        "evidence_class": evidence_class,
        "confidence_level": confidence_level,
        "reproducibility_status": reproducibility_status,
        "reported_at": reported_at,
        "metrics": deepcopy(metrics or {}),
        "normalized_metrics": deepcopy(normalized_metrics or {}),
        "lineage": deepcopy(lineage or {}),
        "notes": notes,
    }


def add_evidence(entry: dict[str, Any], record: dict[str, Any]) -> None:
    entry.setdefault("evidence_records", []).append(deepcopy(record))
    entry["artifact_roots"] = unique_list(
        [*entry.get("artifact_roots", []), *record.get("source_paths", [])]
    )


def normalize_metric_surface(metrics: dict[str, Any]) -> dict[str, float | None]:
    flat: dict[str, float | None] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                flat[f"{key}.{child_key}"] = safe_float(child_value)
        else:
            flat[key] = safe_float(value)

    held_out_accuracy = _first_metric(
        flat,
        (
            "overall_accuracy",
            "held_out_accuracy",
            "logical_accuracy",
            "accuracy",
            "mean_accuracy",
            "ood_accuracy",
            "final_accuracy",
            "adapter_acc",
            "adapter_accuracy",
            "control_base_final_acc",
            "enhanced_t2t_final_acc",
            "coconut_handoff_final_acc",
            "handoff_acc",
            "dptr_dynamic_accuracy",
        ),
    )
    logical_accuracy = _first_metric(
        flat,
        (
            "logical_accuracy",
            "logical_acc",
            "value",
            "adapter_symbolic_acc",
            "enhanced_t2t_symbolic_acc",
            "coconut_handoff_symbolic_acc",
            "control_base_symbolic_acc",
        ),
    )
    final_ce_loss = _first_metric(
        flat,
        (
            "final_ce_loss",
            "ce_loss_final",
            "ce_loss",
            "loss",
        ),
    )
    geometry_retention = _first_metric(
        flat,
        (
            "mean_step_cosine",
            "geometry_retention",
            "provenance_exact_match_ratio_eps",
        ),
    )
    surgery_trigger_rate = _first_metric(
        flat,
        (
            "surgery_trigger_rate",
            "stopgrad_contract_pass",
            "schema_valid_rate",
        ),
    )
    return {
        "held_out_accuracy": held_out_accuracy,
        "logical_accuracy": logical_accuracy,
        "macro_f1": _first_metric(flat, ("macro_f1", "headline_macro_f1")),
        "final_ce_loss": final_ce_loss,
        "intervention_effect_on_gold": _first_metric(flat, ("mean_intervention_delta_gold",)),
        "resume_first_token_accuracy": _first_metric(flat, ("resume_first_token_accuracy",)),
        "english_fluency_score": _first_metric(flat, ("english_fluency_score",)),
        "contamination_rate": _first_metric(flat, ("contamination_rate",)),
        "loop_rate": _first_metric(flat, ("loop_rate",)),
        "scratchpad_bleed_rate": _first_metric(flat, ("scratchpad_bleed_rate",)),
        "surgery_trigger_rate": surgery_trigger_rate,
        "geometry_retention": geometry_retention,
        "final_answer_lift": _first_metric(flat, ("mean_lifts.final_answer", "mean_lifts.adapter_final_answer", "handoff_lift")),
        "symbolic_lift": _first_metric(flat, ("mean_lifts.symbolic", "mean_lifts.adapter_symbolic")),
    }


def finalize_registry(registry: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    entries = [deepcopy(entry) for entry in registry.values()]
    lookup_seen: dict[str, str] = {}
    for entry in entries:
        entry["aliases"] = unique_list(entry.get("aliases", []))
        entry["lookup_aliases"] = unique_list(entry.get("lookup_aliases", []))
        for alias in entry.get("lookup_aliases", []):
            if alias in lookup_seen and lookup_seen[alias] != entry["canonical_id"]:
                raise ValueError(
                    f"lookup alias '{alias}' is duplicated by {lookup_seen[alias]} and {entry['canonical_id']}"
                )
            lookup_seen[alias] = entry["canonical_id"]

        records = entry.get("evidence_records", [])
        classes = {str(r.get("evidence_class")) for r in records if r.get("evidence_class")}
        confidences = {str(r.get("confidence_level")) for r in records if r.get("confidence_level")}
        repros = {str(r.get("reproducibility_status")) for r in records if r.get("reproducibility_status")}
        dates = [str(r.get("reported_at")) for r in records if r.get("reported_at")]

        entry["evidence_counts"] = {
            "total": len(records),
            "artifact": sum(1 for r in records if r.get("evidence_class") == "artifact"),
            "doc_reported": sum(1 for r in records if r.get("evidence_class") == "doc_reported"),
            "git_reported": sum(1 for r in records if r.get("evidence_class") == "git_reported"),
        }
        entry["evidence_class"] = _dominant_evidence_class(classes)
        entry["confidence_level"] = _dominant_confidence(confidences)
        entry["reproducibility_status"] = _dominant_repro(repros)
        entry["date_window"] = {
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None,
        }
        entry["metrics_summary"] = _metrics_summary(records)
        entry["artifact_roots"] = unique_list(entry.get("artifact_roots", []))
        entry["script_paths"] = unique_list(entry.get("script_paths", []))
        entry["dag_paths"] = unique_list(entry.get("dag_paths", []))
        entry["derived_from"] = unique_list(entry.get("derived_from", []))
        entry["supersedes"] = unique_list(entry.get("supersedes", []))
        entry["notes"] = unique_list(entry.get("notes", []))
        entry["evidence_records"] = sorted(
            records,
            key=lambda record: (
                str(record.get("reported_at") or ""),
                str(record.get("source_label") or ""),
            ),
        )

    entries.sort(key=lambda entry: (_date_sort_key(entry.get("date_window", {}).get("earliest")), entry["canonical_id"]))
    return entries


def build_lookup_index(entries: list[dict[str, Any]]) -> dict[str, str]:
    index: dict[str, str] = {}
    for entry in entries:
        for alias in entry.get("lookup_aliases", []):
            index[alias] = entry["canonical_id"]
    return index


def flatten_history_rows(entries: list[dict[str, Any]], mode: str = "all") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if not _entry_in_mode(entry, mode):
            continue
        metrics_summary = entry.get("metrics_summary", {})
        rows.append(
            {
                "canonical_id": entry.get("canonical_id"),
                "normalized_canonical_id": entry.get("normalized_canonical_id"),
                "series_major": entry.get("series_major"),
                "series_minor": entry.get("series_minor"),
                "series_cell": entry.get("series_cell"),
                "family": entry.get("family"),
                "family_group": entry.get("family_group"),
                "title": entry.get("title"),
                "aliases": "|".join(entry.get("aliases", [])),
                "lookup_aliases": "|".join(entry.get("lookup_aliases", [])),
                "evidence_class": entry.get("evidence_class"),
                "confidence_level": entry.get("confidence_level"),
                "reproducibility_status": entry.get("reproducibility_status"),
                "earliest_reported_at": entry.get("date_window", {}).get("earliest"),
                "latest_reported_at": entry.get("date_window", {}).get("latest"),
                "held_out_accuracy": metrics_summary.get("best_held_out_accuracy"),
                "logical_accuracy": metrics_summary.get("best_logical_accuracy"),
                "macro_f1": metrics_summary.get("best_macro_f1"),
                "final_ce_loss": metrics_summary.get("lowest_final_ce_loss"),
                "intervention_effect_on_gold": metrics_summary.get("best_intervention_effect_on_gold"),
                "resume_first_token_accuracy": metrics_summary.get("best_resume_first_token_accuracy"),
                "english_fluency_score": metrics_summary.get("best_english_fluency_score"),
                "contamination_rate": metrics_summary.get("lowest_contamination_rate"),
                "loop_rate": metrics_summary.get("lowest_loop_rate"),
                "scratchpad_bleed_rate": metrics_summary.get("lowest_scratchpad_bleed_rate"),
                "surgery_trigger_rate": metrics_summary.get("best_surgery_trigger_rate"),
                "geometry_retention": metrics_summary.get("best_geometry_retention"),
                "final_answer_lift": metrics_summary.get("best_final_answer_lift"),
                "symbolic_lift": metrics_summary.get("best_symbolic_lift"),
                "artifact_count": entry.get("evidence_counts", {}).get("artifact"),
                "doc_count": entry.get("evidence_counts", {}).get("doc_reported"),
                "git_count": entry.get("evidence_counts", {}).get("git_reported"),
                "script_paths": "|".join(entry.get("script_paths", [])),
                "dag_paths": "|".join(entry.get("dag_paths", [])),
                "artifact_roots": "|".join(entry.get("artifact_roots", [])),
                "baseline_relation": entry.get("baseline_relation"),
                "question_boundary": entry.get("question_boundary"),
                "architectural_thesis": entry.get("architectural_thesis"),
                "inherits_from": "|".join(entry.get("inherits_from", [])),
                "inherits_components": "|".join(entry.get("inherits_components", [])),
                "changed_components": "|".join(entry.get("changed_components", [])),
                "dropped_components": "|".join(entry.get("dropped_components", [])),
                "baseline_manifest": entry.get("baseline_manifest"),
                "archive_path": entry.get("archive_path"),
                "active_doc_path": entry.get("active_doc_path"),
                "derived_from": "|".join(entry.get("derived_from", [])),
                "supersedes": "|".join(entry.get("supersedes", [])),
                "notes": " | ".join(entry.get("notes", [])),
            }
        )
    return rows


def build_history_slice(entries: list[dict[str, Any]], mode: str = "all") -> dict[str, Any]:
    rows = flatten_history_rows(entries, mode)
    families: dict[str, int] = {}
    family_groups: dict[str, int] = {}
    for row in rows:
        family = str(row.get("family") or "")
        family_group = str(row.get("family_group") or "")
        families[family] = families.get(family, 0) + 1
        family_groups[family_group] = family_groups.get(family_group, 0) + 1
    return {
        "mode": mode,
        "entry_count": len(rows),
        "family_counts": dict(sorted(families.items(), key=lambda item: item[0])),
        "family_group_counts": dict(sorted(family_groups.items(), key=lambda item: item[0])),
    }


def _entry_in_mode(entry: dict[str, Any], mode: str) -> bool:
    normalized = str(mode).strip().lower()
    status = str(entry.get("reproducibility_status") or "")
    evidence_class = str(entry.get("evidence_class") or "")
    if normalized == "all":
        return True
    if normalized == "artifact_only":
        return evidence_class in {"artifact", "mixed"} and status in {"artifact_only", "partial", "runnable"}
    if normalized == "runnable_only":
        return status == "runnable"
    raise ValueError(f"Unsupported history slice mode '{mode}'.")


def _dominant_evidence_class(classes: set[str]) -> str:
    if not classes:
        return "doc_reported"
    if len(classes) == 1:
        return next(iter(classes))
    return "mixed"


def _dominant_confidence(confidences: set[str]) -> str:
    if not confidences:
        return "low"
    return max(confidences, key=lambda value: CONFIDENCE_LEVEL_ORDER.get(value, -1))


def _dominant_repro(repros: set[str]) -> str:
    if not repros:
        return "doc_only"
    return max(repros, key=lambda value: REPRODUCIBILITY_ORDER.get(value, -1))


def _metrics_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {
        "best_held_out_accuracy": None,
        "best_logical_accuracy": None,
        "best_macro_f1": None,
        "lowest_final_ce_loss": None,
        "best_intervention_effect_on_gold": None,
        "best_resume_first_token_accuracy": None,
        "best_english_fluency_score": None,
        "lowest_contamination_rate": None,
        "lowest_loop_rate": None,
        "lowest_scratchpad_bleed_rate": None,
        "best_surgery_trigger_rate": None,
        "best_geometry_retention": None,
        "best_final_answer_lift": None,
        "best_symbolic_lift": None,
    }
    for record in records:
        metrics = record.get("normalized_metrics", {})
        summary["best_held_out_accuracy"] = _max_metric(summary["best_held_out_accuracy"], metrics.get("held_out_accuracy"))
        summary["best_logical_accuracy"] = _max_metric(summary["best_logical_accuracy"], metrics.get("logical_accuracy"))
        summary["best_macro_f1"] = _max_metric(summary["best_macro_f1"], metrics.get("macro_f1"))
        summary["lowest_final_ce_loss"] = _min_metric(summary["lowest_final_ce_loss"], metrics.get("final_ce_loss"))
        summary["best_intervention_effect_on_gold"] = _max_metric(summary["best_intervention_effect_on_gold"], metrics.get("intervention_effect_on_gold"))
        summary["best_resume_first_token_accuracy"] = _max_metric(summary["best_resume_first_token_accuracy"], metrics.get("resume_first_token_accuracy"))
        summary["best_english_fluency_score"] = _max_metric(summary["best_english_fluency_score"], metrics.get("english_fluency_score"))
        summary["lowest_contamination_rate"] = _min_metric(summary["lowest_contamination_rate"], metrics.get("contamination_rate"))
        summary["lowest_loop_rate"] = _min_metric(summary["lowest_loop_rate"], metrics.get("loop_rate"))
        summary["lowest_scratchpad_bleed_rate"] = _min_metric(summary["lowest_scratchpad_bleed_rate"], metrics.get("scratchpad_bleed_rate"))
        summary["best_surgery_trigger_rate"] = _max_metric(summary["best_surgery_trigger_rate"], metrics.get("surgery_trigger_rate"))
        summary["best_geometry_retention"] = _max_metric(summary["best_geometry_retention"], metrics.get("geometry_retention"))
        summary["best_final_answer_lift"] = _max_metric(summary["best_final_answer_lift"], metrics.get("final_answer_lift"))
        summary["best_symbolic_lift"] = _max_metric(summary["best_symbolic_lift"], metrics.get("symbolic_lift"))
    return summary


def _first_metric(flat: dict[str, float | None], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = flat.get(key)
        if value is not None:
            return value
    return None


def _max_metric(current: float | None, candidate: Any) -> float | None:
    value = safe_float(candidate)
    if value is None:
        return current
    if current is None:
        return value
    return max(current, value)


def _min_metric(current: float | None, candidate: Any) -> float | None:
    value = safe_float(candidate)
    if value is None:
        return current
    if current is None:
        return value
    return min(current, value)


def _date_sort_key(value: str | None) -> tuple[int, str]:
    if not value:
        return (1, "")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return (0, parsed.isoformat())
    except ValueError:
        return (0, value)
