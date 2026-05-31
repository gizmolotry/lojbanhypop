from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "src"))

from lojban_evolution.control_plane.path_registry import (  # noqa: E402
    canonical_repo_path,
    latest_history_manifest,
    repo_relative,
)

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_program_spine"
DEFAULT_DOC_OUTPUT = REPO_ROOT / "docs" / "ABLATION_PROGRAM_SPINE.md"

LETTER_STAGE_ORDER = ["A-G", "H", "H5", "J", "L", "J/L Hypercube", "Phase Eval"]
M_STAGE_ORDER = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11", "M14", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25", "M26"]
EXTRA_STAGE_DAGS = {
    "M19": ["airflow/dags/m19/lojban_m19_family_dag.py"],
    "M20": ["airflow/dags/m20/lojban_m20_dictionary_first_dag.py"],
    "M21": ["airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py"],
    "M22": ["airflow/dags/m22/lojban_m22_semantic_generalization_dag.py"],
    "M23": ["airflow/dags/m23/lojban_m23_relevance_router_dag.py"],
    "M24": ["airflow/dags/m24/lojban_m24_substrate_compression_dag.py"],
    "M25": ["airflow/dags/m25/lojban_m25_emergent_bridi_dag.py"],
    "M26": ["airflow/dags/m26/lojban_m26_end_to_end_loafman_dag.py"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one ordered ablation program spine from the canonical history manifest.")
    parser.add_argument("--history-manifest", type=Path, default=None)
    parser.add_argument("--taxonomy-config", type=Path, default=REPO_ROOT / "configs" / "experiment_taxonomy.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_OUTPUT)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_manifest = args.history_manifest or _latest_history_manifest()
    taxonomy = json.loads(args.taxonomy_config.read_text(encoding="utf-8"))
    history = json.loads(history_manifest.read_text(encoding="utf-8"))

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("ablation_program_spine_%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = history.get("entries", [])
    series_families = {row["series_key"]: row for row in history.get("series_family_manifests", [])}
    transitions = _merge_transition_manifests(history.get("transition_manifests", []), taxonomy.get("transition_manifests", []))
    comparison_contracts = {
        **taxonomy.get("family_comparison_contracts", {}),
        **history.get("comparison_contracts", {}),
    }
    major_families = taxonomy.get("major_families", {})

    stages: list[dict[str, Any]] = []
    for key in LETTER_STAGE_ORDER:
        stages.append(_build_letter_stage(key, series_families.get(key)))
    for key in M_STAGE_ORDER:
        stages.append(_build_major_stage(key, major_families.get(key, {}), entries, transitions, comparison_contracts.get(key, {})))
    stages.append(_build_control_plane_stage(history_manifest))

    manifest = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_history_manifest": _repo_relative(history_manifest),
        "source_taxonomy_config": _repo_relative(args.taxonomy_config),
        "stage_count": len(stages),
        "stages": stages,
    }
    _canonicalize_manifest_paths(manifest)

    manifest_path = output_dir / "ablation_program_spine_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.doc_output.write_text(_render_markdown(manifest), encoding="utf-8")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {args.doc_output}")


def _latest_history_manifest() -> Path:
    return latest_history_manifest()


def _repo_relative(path: Path) -> str:
    return repo_relative(path, REPO_ROOT) or ""


def _build_letter_stage(stage_key: str, row: dict[str, Any] | None) -> dict[str, Any]:
    row = row or {}
    return {
        "stage_key": stage_key,
        "stage_kind": "legacy_series",
        "program_layer": "legacy_orchestration",
        "title": row.get("title") or stage_key,
        "objective": row.get("objective") or "",
        "entry_count": int(row.get("entry_count") or 0),
        "runnable_count": int(row.get("runnable_count") or 0),
        "artifact_only_count": int(row.get("artifact_only_count") or 0),
        "doc_only_count": int(row.get("doc_only_count") or 0),
        "normalized_ids": list(row.get("normalized_ids", [])),
        "legacy_aliases": list(row.get("legacy_aliases", [])),
        "docs": list(row.get("doc_paths", [])),
        "scripts": list(row.get("script_paths", [])),
        "dags": list(row.get("dag_paths", [])),
        "artifact_roots": list(row.get("artifact_roots", [])),
        "question_boundary": None,
        "architectural_thesis": None,
        "allowed_ablation_axes": [],
        "forbidden_drift_axes": [],
        "promotion_basis": [],
        "metrics_primary": [],
        "metrics_guardrail": [],
        "required_test_contracts": [],
        "historical_comparison_families": [],
        "comparison_targets": [],
        "baseline_manifest": None,
        "selected_upstream": None,
        "inherits_components": [],
        "reopened_components": [],
        "rejected_components": [],
        "legacy_origin": None,
    }


def _build_major_stage(
    stage_key: str,
    taxonomy_row: dict[str, Any],
    entries: list[dict[str, Any]],
    transitions: list[dict[str, Any]],
    comparison_contract: dict[str, Any],
) -> dict[str, Any]:
    major_num = int(stage_key[1:])
    stage_entries = [entry for entry in entries if int(entry.get("series_major") or -1) == major_num]
    normalized_ids = sorted({str(entry.get("normalized_canonical_id") or "") for entry in stage_entries if str(entry.get("normalized_canonical_id") or "").strip()})
    legacy_aliases = sorted({alias for entry in stage_entries for alias in entry.get("aliases", []) if str(alias).strip()})
    scripts = sorted({path for entry in stage_entries for path in entry.get("script_paths", []) if str(path).strip()})
    dags = sorted(
        {
            *[path for entry in stage_entries for path in entry.get("dag_paths", []) if str(path).strip()],
            *EXTRA_STAGE_DAGS.get(stage_key, []),
        }
    )
    artifact_roots = sorted({path for entry in stage_entries for path in entry.get("artifact_roots", []) if str(path).strip()})
    transition = next((row for row in transitions if str(row.get("to_major")) == stage_key), None)

    legacy_origin = None
    if stage_key == "M1":
        legacy_origin = "J"
    elif stage_key == "M2":
        legacy_origin = "L"

    required_contracts = comparison_contract.get("required_test_contract_ids") or comparison_contract.get("required_test_contracts") or []
    comparison_targets = comparison_contract.get("comparison_targets")
    if not isinstance(comparison_targets, list):
        comparison_targets = []
    explicit_entries = comparison_contract.get("explicit_compare_entries")
    if not comparison_targets and isinstance(explicit_entries, list):
        comparison_targets = [
            {"kind": "explicit_entry", "target": str(target), "reason": "family-specific explicit comparator"}
            for target in explicit_entries
            if str(target).strip()
        ]

    return {
        "stage_key": stage_key,
        "stage_kind": "major_series",
        "program_layer": _major_program_layer(major_num),
        "title": stage_key,
        "objective": taxonomy_row.get("architectural_thesis", ""),
        "entry_count": len(stage_entries),
        "runnable_count": sum(1 for entry in stage_entries if str(entry.get("reproducibility_status") or "") == "runnable"),
        "artifact_only_count": sum(1 for entry in stage_entries if str(entry.get("reproducibility_status") or "") == "artifact_only"),
        "doc_only_count": sum(1 for entry in stage_entries if str(entry.get("reproducibility_status") or "") == "doc_only"),
        "normalized_ids": normalized_ids,
        "legacy_aliases": legacy_aliases,
        "docs": [],
        "scripts": scripts,
        "dags": dags,
        "artifact_roots": artifact_roots,
        "question_boundary": taxonomy_row.get("question_boundary"),
        "architectural_thesis": taxonomy_row.get("architectural_thesis"),
        "allowed_ablation_axes": list(taxonomy_row.get("allowed_ablation_axes", [])),
        "forbidden_drift_axes": list(taxonomy_row.get("forbidden_drift_axes", [])),
        "promotion_basis": list(taxonomy_row.get("promotion_basis", [])),
        "metrics_primary": list(taxonomy_row.get("metrics_primary", [])),
        "metrics_guardrail": list(taxonomy_row.get("metrics_guardrail", [])),
        "required_test_contracts": [str(item) for item in required_contracts],
        "historical_comparison_families": list(comparison_contract.get("historical_comparison_families", [])),
        "comparison_targets": comparison_targets,
        "baseline_manifest": taxonomy_row.get("baseline_manifest"),
        "selected_upstream": transition.get("selected_upstream") if transition else None,
        "inherits_components": list(transition.get("inherits_components", [])) if transition else [],
        "reopened_components": list(transition.get("reopened_components", [])) if transition else [],
        "rejected_components": list(transition.get("rejected_components", [])) if transition else [],
        "legacy_origin": legacy_origin,
    }


def _build_control_plane_stage(history_manifest: Path) -> dict[str, Any]:
    return {
        "stage_key": "Control Plane",
        "stage_kind": "control_plane",
        "program_layer": "control_plane",
        "title": "Control Plane",
        "objective": "backfill, normalize, aggregate, and render the full ablation program so historical and modern runs live in one auditable surface.",
        "entry_count": 0,
        "runnable_count": 1,
        "artifact_only_count": 0,
        "doc_only_count": 0,
        "normalized_ids": [],
        "legacy_aliases": [],
        "docs": [
            _repo_relative(REPO_ROOT / "docs" / "ABLATION_HISTORY_FULL.md"),
            _repo_relative(REPO_ROOT / "docs" / "ABLATION_PROGRAM_MAP.md"),
        ],
        "scripts": [
            "scripts/control_plane/run_ablation_history_backfill.py",
            "scripts/control_plane/run_ablation_test_matrix.py",
            "scripts/control_plane/run_whole_ablation_grid.py",
            "scripts/control_plane/build_ablation_program_map.py",
            "scripts/control_plane/build_ablation_program_spine.py",
            "scripts/m_bridge/run_m_bridge_ablation_test_suite.py",
        ],
        "dags": [
            "airflow/dags/control_plane/lojban_ablation_history_backfill_dag.py",
            "airflow/dags/control_plane/lojban_ablation_test_matrix_dag.py",
            "airflow/dags/control_plane/lojban_ablation_program_spine_dag.py",
            "airflow/dags/control_plane/lojban_whole_ablation_grid_dag.py",
            "airflow/dags/m_bridge/lojban_m_bridge_ablation_test_suite_dag.py",
        ],
        "artifact_roots": [_repo_relative(history_manifest.parent)],
        "question_boundary": "program governance and reproducibility",
        "architectural_thesis": "a coherent research program needs one canonical control plane for lineage, aggregation, and orchestration.",
        "allowed_ablation_axes": ["history backfill", "program map rendering", "suite aggregation"],
        "forbidden_drift_axes": ["silent family creation", "implicit lineage jumps"],
        "promotion_basis": ["manifest completeness", "path coherence", "family coverage"],
        "metrics_primary": ["entry_count", "family_count"],
        "metrics_guardrail": ["historical_gap_count"],
        "required_test_contracts": [],
        "historical_comparison_families": [],
        "comparison_targets": [],
        "baseline_manifest": None,
        "selected_upstream": None,
        "inherits_components": [],
        "reopened_components": [],
        "rejected_components": [],
        "legacy_origin": None,
    }


def _merge_transition_manifests(history_transitions: list[dict[str, Any]], taxonomy_transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in [*history_transitions, *taxonomy_transitions]:
        if not isinstance(row, dict):
            continue
        transition_id = str(row.get("transition_id") or "").strip()
        if not transition_id:
            continue
        merged[transition_id] = row
    return list(merged.values())


def _major_program_layer(major_num: int) -> str:
    if major_num <= 2:
        return "legacy_orchestration"
    if major_num <= 5:
        return "bridge_and_serialization"
    if major_num == 20:
        return "dictionary_first_substrate"
    if major_num == 21:
        return "dynamic_bridi_substrate"
    if major_num == 22:
        return "semantic_generalization_substrate"
    if major_num == 23:
        return "causal_relevance_substrate"
    if major_num == 24:
        return "substrate_compression"
    if major_num == 25:
        return "emergent_bridi_grammar"
    if major_num == 26:
        return "end_to_end_lojban_symbiote"
    return "manifold_and_return_path"


def _canonical_path(value: str) -> str:
    return canonical_repo_path(value, REPO_ROOT)


def _canonicalize_manifest_paths(manifest: dict[str, Any]) -> None:
    for stage in manifest.get("stages", []):
        stage["scripts"] = sorted({_canonical_path(path) for path in stage.get("scripts", []) if str(path).strip()})
        stage["dags"] = sorted({_canonical_path(path) for path in stage.get("dags", []) if str(path).strip()})


def _render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Ablation Program Spine",
        "",
        f"- Generated UTC: `{manifest['generated_utc']}`",
        f"- Source history manifest: `{manifest['source_history_manifest']}`",
        f"- Source taxonomy config: `{manifest['source_taxonomy_config']}`",
        f"- Stage count: `{manifest['stage_count']}`",
        "",
        "This is the ordered research spine of the project: legacy letter-series families, normalized M-major families, and the control plane that keeps the program auditable.",
        "",
    ]
    for index, stage in enumerate(manifest["stages"], start=1):
        lines.append(f"## {index}. {stage['stage_key']}")
        lines.append("")
        lines.append(f"- Kind: `{stage['stage_kind']}`")
        lines.append(f"- Layer: `{stage['program_layer']}`")
        lines.append(f"- Objective: {stage.get('objective') or 'n/a'}")
        if stage.get("legacy_origin"):
            lines.append(f"- Legacy origin: `{stage['legacy_origin']}`")
        lines.append(f"- Entry count: `{stage['entry_count']}`")
        lines.append(f"- Runnable rows: `{stage['runnable_count']}`")
        if stage.get("artifact_only_count"):
            lines.append(f"- Artifact-only rows: `{stage['artifact_only_count']}`")
        if stage.get("doc_only_count"):
            lines.append(f"- Doc-only rows: `{stage['doc_only_count']}`")
        if stage.get("question_boundary"):
            lines.append(f"- Question boundary: {stage['question_boundary']}")
        if stage.get("architectural_thesis"):
            lines.append(f"- Thesis: {stage['architectural_thesis']}")
        if stage.get("selected_upstream"):
            lines.append(f"- Selected upstream: `{stage['selected_upstream']}`")
        if stage.get("inherits_components"):
            lines.append(f"- Inherits: `{', '.join(stage['inherits_components'])}`")
        if stage.get("reopened_components"):
            lines.append(f"- Reopens: `{', '.join(stage['reopened_components'])}`")
        if stage.get("rejected_components"):
            lines.append(f"- Rejects: `{', '.join(stage['rejected_components'])}`")
        if stage.get("normalized_ids"):
            lines.append(f"- IDs: `{', '.join(stage['normalized_ids'][:40])}`")
        if stage.get("legacy_aliases"):
            lines.append(f"- Aliases: `{', '.join(stage['legacy_aliases'][:40])}`")
        if stage.get("allowed_ablation_axes"):
            lines.append(f"- Allowed axes: `{', '.join(stage['allowed_ablation_axes'])}`")
        if stage.get("forbidden_drift_axes"):
            lines.append(f"- Frozen/forbidden drift: `{', '.join(stage['forbidden_drift_axes'])}`")
        if stage.get("promotion_basis"):
            lines.append(f"- Promotion basis: `{', '.join(stage['promotion_basis'])}`")
        if stage.get("metrics_primary"):
            lines.append(f"- Primary metrics: `{', '.join(stage['metrics_primary'])}`")
        if stage.get("metrics_guardrail"):
            lines.append(f"- Guardrail metrics: `{', '.join(stage['metrics_guardrail'])}`")
        if stage.get("historical_comparison_families"):
            lines.append(f"- Historical comparison families: `{', '.join(stage['historical_comparison_families'])}`")
        if stage.get("required_test_contracts"):
            lines.append(f"- Required test contracts: `{', '.join(stage['required_test_contracts'])}`")
        if stage.get("comparison_targets"):
            compare_targets = [str(row.get("target", "")) for row in stage["comparison_targets"] if isinstance(row, dict) and str(row.get("target", "")).strip()]
            if compare_targets:
                lines.append(f"- Automatic comparison targets: `{', '.join(compare_targets)}`")
        if stage.get("baseline_manifest"):
            lines.append(f"- Baseline manifest: `{stage['baseline_manifest']}`")
        if stage.get("docs"):
            lines.append(f"- Docs: `{', '.join(stage['docs'][:12])}`")
        if stage.get("scripts"):
            lines.append(f"- Scripts: `{', '.join(stage['scripts'][:12])}`")
        if stage.get("dags"):
            lines.append(f"- DAGs: `{', '.join(stage['dags'][:12])}`")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
