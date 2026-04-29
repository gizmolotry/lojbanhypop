from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


HISTORY_ROOT = Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill")


GROUP_ORDER = [
    "core_matrix",
    "control_duel",
    "h_series",
    "h5_bridge",
    "j_series",
    "l_series",
    "a_to_g_matrix",
    "phase5_train_ablation",
    "phase5_objective_ablation",
    "m_track",
    "historical_gap",
]


GROUP_DESCRIPTIONS = {
    "core_matrix": "Earliest documented core controls and challengers before later reruns and telemetry consolidation.",
    "control_duel": "Direct English-vs-Lojban control comparisons used to test whether medium alone explained performance.",
    "h_series": "Mid-layer bridge and related H-family latent handoff experiments.",
    "h5_bridge": "Late H5 gearbox, boolean surgery, and forced-manifold bridge rows.",
    "j_series": "Data invariance, adversarial synthesis, and acceptance-diagnostic series.",
    "l_series": "Constraint-optimized Lagrangian training lineage, including the L-series charter, L6 branch, and L-rooted M3+/M4/M5 branch work.",
    "a_to_g_matrix": "Artifact-backed reruns of the legacy A-G matrix under the modern telemetry stack.",
    "phase5_train_ablation": "Training-stack ablations along the phase-5 path.",
    "phase5_objective_ablation": "Objective and loss-surface ablations along the phase-5 path.",
    "m_track": "Modern telemetry-rooted M-series work that is not historically rooted in L-series.",
    "historical_gap": "Known missing artifacts, orphaned checkpoints, and preserved gaps.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a human-readable Markdown catalog of every ablation point from the canonical history manifest.")
    parser.add_argument(
        "--history-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/ABLATION_HISTORY_FULL.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.history_manifest = _resolve_manifest(args.history_manifest)
    payload = json.loads(args.history_manifest.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        group = str(entry.get("family_group") or "misc")
        grouped.setdefault(group, []).append(entry)

    lines: list[str] = []
    lines.append("# Full Ablation History")
    lines.append("")
    lines.append(f"- Source manifest: `{str(args.history_manifest).replace(chr(92), '/')}`")
    lines.append(f"- Generated from run: `{payload.get('run_id')}`")
    lines.append(f"- Total canonical entries: `{len(entries)}`")
    lines.append(f"- Artifact-backed entries: `{payload.get('history_slices', {}).get('artifact_only', {}).get('entry_count')}`")
    lines.append(f"- Runnable entries: `{payload.get('history_slices', {}).get('runnable_only', {}).get('entry_count')}`")
    lines.append("")
    lines.append("This document lists every ablation point currently tracked in the unified ledger. Each item includes a brief description, provenance status, and the best-known metric surface when one exists.")
    lines.append("")
    lines.append("## Group Index")
    lines.append("")
    for group, count in _ordered_counts(payload.get("family_group_counts", _count_group_entries(entries))):
        lines.append(f"- `{group}`: `{count}`")
    lines.append("")

    ordered_groups = sorted(grouped, key=lambda group: (GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER), group))
    for group in ordered_groups:
        lines.append(f"## {group}")
        lines.append("")
        description = GROUP_DESCRIPTIONS.get(group)
        if description:
            lines.append(description)
            lines.append("")
        family_map: dict[str, list[dict[str, Any]]] = {}
        for entry in grouped[group]:
            family_map.setdefault(str(entry.get("family") or "misc"), []).append(entry)
        for family in sorted(family_map):
            family_entries = sorted(family_map[family], key=_entry_sort_key)
            lines.append(f"### {family} ({len(family_entries)})")
            lines.append("")
            for entry in family_entries:
                lines.extend(_render_entry(entry))
        lines.append("")

    gaps = payload.get("historical_gaps", [])
    if gaps:
        lines.append("## historical_gaps")
        lines.append("")
        for gap in gaps:
            lines.append(f"### {gap.get('gap_id')}")
            lines.append("")
            lines.append(f"- Kind: `{gap.get('kind')}`")
            lines.append(f"- Path: `{gap.get('path')}`")
            lines.append(f"- Source: `{gap.get('reported_in')}`")
            if gap.get("notes"):
                lines.append(f"- Description: {gap.get('notes')}")
            lines.append("")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {args.output}")


def _render_entry(entry: dict[str, Any]) -> list[str]:
    metrics = entry.get("metrics_summary", {})
    lines = [f"#### {entry.get('title')}"]
    lines.append("")
    lines.append(f"- Canonical ID: `{entry.get('canonical_id')}`")
    if entry.get("normalized_canonical_id") and entry.get("normalized_canonical_id") != entry.get("canonical_id"):
        lines.append(f"- Normalized ID: `{entry.get('normalized_canonical_id')}`")
    if entry.get("series_major") is not None:
        lines.append(
            f"- Taxonomy: major=`{entry.get('series_major')}` minor=`{entry.get('series_minor')}` cell=`{entry.get('series_cell')}`"
        )
    if entry.get("aliases"):
        lines.append(f"- Aliases: `{', '.join(entry.get('aliases', []))}`")
    if entry.get("lookup_aliases"):
        lines.append(f"- Lookup aliases: `{', '.join(entry.get('lookup_aliases', []))}`")
    if entry.get("objective"):
        lines.append(f"- Brief: {entry.get('objective')}")
    if entry.get("question_boundary"):
        lines.append(f"- Architectural question: `{entry.get('question_boundary')}`")
    lines.append(
        f"- Provenance: evidence=`{entry.get('evidence_class')}` confidence=`{entry.get('confidence_level')}` reproducibility=`{entry.get('reproducibility_status')}`"
    )
    if entry.get("baseline_relation"):
        lines.append(f"- Baseline relation: {entry.get('baseline_relation')}")
    if entry.get("inherits_from"):
        lines.append(f"- Inherits from: `{', '.join(entry.get('inherits_from', []))}`")
    metric_line = _metric_line(metrics)
    if metric_line:
        lines.append(f"- Best-known metrics: {metric_line}")
    if entry.get("script_paths"):
        lines.append(f"- Scripts: `{', '.join(entry.get('script_paths', []))}`")
    if entry.get("dag_paths"):
        lines.append(f"- DAGs: `{', '.join(entry.get('dag_paths', []))}`")
    if entry.get("notes"):
        lines.append(f"- Notes: {' | '.join(entry.get('notes', []))}")
    lines.append("")
    return lines


def _resolve_manifest(manifest: Path | None) -> Path:
    if manifest is not None:
        return manifest
    candidates = sorted(HISTORY_ROOT.rglob("ablation_history_manifest.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No ablation history manifest found under {HISTORY_ROOT}")
    return candidates[-1]


def _entry_sort_key(entry: dict[str, Any]) -> tuple[str, str, str]:
    date_window = entry.get("date_window", {})
    earliest = str(date_window.get("earliest") or "")
    title = str(entry.get("title") or "")
    canonical_id = str(entry.get("canonical_id") or "")
    return (earliest, title, canonical_id)


def _count_group_entries(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        group = str(entry.get("family_group") or "misc")
        counts[group] = counts.get(group, 0) + 1
    return counts


def _ordered_counts(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(
        counts.items(),
        key=lambda item: (GROUP_ORDER.index(item[0]) if item[0] in GROUP_ORDER else len(GROUP_ORDER), item[0]),
    )


def _metric_line(metrics: dict[str, Any]) -> str:
    parts: list[str] = []
    if _num(metrics.get("best_held_out_accuracy")) is not None:
        parts.append(f"held_out_accuracy=`{_fmt(metrics.get('best_held_out_accuracy'))}`")
    if _num(metrics.get("best_logical_accuracy")) is not None:
        parts.append(f"logical_accuracy=`{_fmt(metrics.get('best_logical_accuracy'))}`")
    if _num(metrics.get("best_macro_f1")) is not None:
        parts.append(f"macro_f1=`{_fmt(metrics.get('best_macro_f1'))}`")
    if _num(metrics.get("lowest_final_ce_loss")) is not None:
        parts.append(f"final_ce_loss=`{_fmt(metrics.get('lowest_final_ce_loss'))}`")
    if _num(metrics.get("best_intervention_effect_on_gold")) is not None:
        parts.append(f"intervention_effect_on_gold=`{_fmt(metrics.get('best_intervention_effect_on_gold'))}`")
    if _num(metrics.get("best_resume_first_token_accuracy")) is not None:
        parts.append(f"resume_first_token_accuracy=`{_fmt(metrics.get('best_resume_first_token_accuracy'))}`")
    if _num(metrics.get("best_english_fluency_score")) is not None:
        parts.append(f"english_fluency_score=`{_fmt(metrics.get('best_english_fluency_score'))}`")
    if _num(metrics.get("best_final_answer_lift")) is not None:
        parts.append(f"final_answer_lift=`{_fmt(metrics.get('best_final_answer_lift'))}`")
    if _num(metrics.get("best_symbolic_lift")) is not None:
        parts.append(f"symbolic_lift=`{_fmt(metrics.get('best_symbolic_lift'))}`")
    if _num(metrics.get("best_geometry_retention")) is not None:
        parts.append(f"geometry_retention=`{_fmt(metrics.get('best_geometry_retention'))}`")
    if _num(metrics.get("best_surgery_trigger_rate")) is not None:
        parts.append(f"surgery_trigger_rate=`{_fmt(metrics.get('best_surgery_trigger_rate'))}`")
    return ", ".join(parts)


def _num(value: Any) -> float | None:
    try:
        if value is None or isinstance(value, bool):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any) -> str:
    number = _num(value)
    if number is None:
        return "n/a"
    return f"{number:.6f}"


if __name__ == "__main__":
    main()
