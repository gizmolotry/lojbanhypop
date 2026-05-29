from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_program_map"
DEFAULT_DOC_OUTPUT = REPO_ROOT / "docs" / "ABLATION_PROGRAM_MAP.md"

FAMILY_DAG_MAP: dict[str, list[str]] = {
    "A-G": ["airflow/dags/control_plane/lojban_ablation_matrix_dag.py"],
    "J": ["airflow/dags/legacy/lojban_j_series_dag.py"],
    "L": ["airflow/dags/legacy/lojban_l_series_dag.py"],
    "J/L Hypercube": ["airflow/dags/control_plane/lojban_ablation_hypercube_report_dag.py"],
    "Phase Eval": ["airflow/dags/control_plane/lojban_phase_ablation_dag.py"],
    "M3": ["airflow/dags/lojban_m3_plus_dag.py"],
    "M3.5": ["airflow/dags/lojban_m3_5_symmetry_dag.py"],
    "M3.6": ["airflow/dags/lojban_m3_6_symmetry_oracle_dag.py"],
    "M3.7": ["airflow/dags/lojban_m3_7_shadow_alignment_dag.py"],
    "M3.8": ["airflow/dags/lojban_m3_8_operator_diversification_dag.py"],
    "M3.9": ["airflow/dags/lojban_m3_9_primitive_probe_dag.py"],
    "M3.10": ["airflow/dags/lojban_m3_10_ood_accuracy_probe_dag.py"],
    "M3.11": ["airflow/dags/lojban_m3_11_winograd_failure_anatomy_dag.py"],
    "M3.12": ["airflow/dags/lojban_m3_12_geometric_return_stream_dag.py"],
    "M3.14": ["airflow/dags/lojban_m3_14_structural_alignment_bridge_dag.py"],
    "M3.15": [
        "airflow/dags/lojban_m3_15_rotary_coconut_dag.py",
        "airflow/dags/lojban_m3_15_rotary_coconut_seven_dag.py",
    ],
    "M3.15b": ["airflow/dags/lojban_m3_15b_relation_local_rotary_dag.py"],
    "M3.15c": ["airflow/dags/lojban_m3_15c_family_conditioned_bridge_dag.py"],
    "M3.15d": ["airflow/dags/lojban_m3_15d_answer_path_forcing_dag.py"],
    "M3.16": ["airflow/dags/lojban_m3_16_continuous_graph_bias_dag.py"],
    "M3.17": ["airflow/dags/lojban_m3_17_advisor_reentry_bridge_dag.py"],
    "M3.18": ["airflow/dags/m3/lojban_m3_18_decoder_reentry_resume_dag.py"],
    "M3.19": ["airflow/dags/lojban_m3_19_d_mainline_grid_dag.py"],
    "M4": ["airflow/dags/lojban_m4_series_dag.py"],
    "M4.0": ["airflow/dags/lojban_m4_0_semantic_probe_dag.py"],
    "M4.2": ["airflow/dags/lojban_m4_2_predicate_grounding_dag.py"],
    "M5": ["airflow/dags/lojban_m5_autoformalization_dag.py"],
    "M5.1": ["airflow/dags/lojban_m5_padded_nary_dag.py"],
    "M5.2": ["airflow/dags/lojban_m5_2_autoregressive_chain_dag.py"],
    "M5.3": ["airflow/dags/lojban_m5_3_masked_pair_chain_dag.py"],
    "M11": ["airflow/dags/lojban_m11_discriminative_suite_dag.py"],
    "M14": ["airflow/dags/lojban_m14_symbiote_scratchpad_dag.py"],
    "M18": ["airflow/dags/lojban_m18_controller_family_dag.py"],
    "M19": [
        "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
        "airflow/dags/m19/lojban_m19_isolation_grid_dag.py",
        "airflow/dags/m19/lojban_m19_replication_suite_dag.py",
        "airflow/dags/m19/lojban_m19_kill_test_suite_dag.py",
        "airflow/dags/m19/lojban_m19_paper_package_dag.py",
        "airflow/dags/m19/lojban_m19_family_dag.py",
    ],
    "M20": ["airflow/dags/m20/lojban_m20_dictionary_first_dag.py"],
    "M21": ["airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py"],
    "M22": ["airflow/dags/m22/lojban_m22_semantic_generalization_dag.py"],
    "M23": ["airflow/dags/m23/lojban_m23_relevance_router_dag.py"],
    "M24": ["airflow/dags/m24/lojban_m24_substrate_compression_dag.py"],
    "M25": ["airflow/dags/m25/lojban_m25_emergent_bridi_dag.py"],
    "History": [
        "airflow/dags/control_plane/lojban_ablation_history_backfill_dag.py",
        "airflow/dags/m_bridge/lojban_m_bridge_ablation_test_suite_dag.py",
    ],
}


LETTER_FAMILY_ORDER = ["A-G", "H", "H5", "J", "L", "J/L Hypercube", "Phase Eval"]
M_FAMILY_ORDER = [
    "M1",
    "M2",
    "M3",
    "M3.5",
    "M3.6",
    "M3.7",
    "M3.8",
    "M3.9",
    "M3.10",
    "M3.11",
    "M3.12",
    "M3.13",
    "M3.14",
    "M3.15",
    "M3.15b",
    "M3.15c",
    "M3.15d",
    "M3.16",
    "M3.17",
    "M3.18",
    "M3.19",
    "M4",
    "M4.0",
    "M4.2",
    "M5",
    "M5.1",
    "M5.2",
    "M5.3",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11",
    "M14",
    "M18",
    "M19",
    "M20",
    "M21",
    "M22",
    "M23",
    "M24",
    "M25",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one concentrated ablation program map from the canonical history manifest.")
    parser.add_argument("--history-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_OUTPUT)
    parser.add_argument("--run-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    history_manifest = args.history_manifest or _latest_history_manifest()
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("ablation_program_map_%Y%m%d_%H%M%S")
    output_dir = args.output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    history = json.loads(history_manifest.read_text(encoding="utf-8"))
    entries = history.get("entries", [])
    series_family_manifests = history.get("series_family_manifests", [])
    transitions = history.get("transition_manifests", [])

    families = _build_family_map(entries, series_family_manifests)
    ordered_families = _ordered_family_rows(families)
    manifest = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_history_manifest": _repo_relative(history_manifest),
        "family_count": len(ordered_families),
        "families": ordered_families,
        "transitions": transitions,
    }
    _canonicalize_manifest_paths(manifest)

    manifest_path = output_dir / "ablation_program_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.doc_output.write_text(_render_markdown(manifest), encoding="utf-8")
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {args.doc_output}")


def _latest_history_manifest() -> Path:
    return latest_history_manifest()


def _repo_relative(path: Path) -> str:
    return repo_relative(path, REPO_ROOT) or ""


def _family_key(entry: dict[str, Any]) -> str | None:
    canonical_id = str(entry.get("canonical_id") or "")
    family_group = str(entry.get("family_group") or "")
    normalized = str(entry.get("normalized_canonical_id") or "")

    if family_group == "a_to_g_matrix":
        return "A-G"
    if family_group == "h_series":
        return "H5" if "h5" in canonical_id.lower() else "H"
    if family_group == "j_series":
        return "J"
    if family_group == "l_series":
        return "L"
    if canonical_id.startswith("phase5."):
        return "Phase Eval"

    if normalized.startswith("M1."):
        return "M1"
    if normalized.startswith("M2."):
        return "M2"
    if normalized.startswith("M3.5"):
        return "M3.5"
    if normalized.startswith("M3.6"):
        return "M3.6"
    if normalized.startswith("M3.7"):
        return "M3.7"
    if normalized.startswith("M3.8"):
        return "M3.8"
    if normalized.startswith("M3.9"):
        return "M3.9"
    if normalized.startswith("M3.10"):
        return "M3.10"
    if normalized.startswith("M3.11"):
        return "M3.11"
    if normalized.startswith("M3.12"):
        return "M3.12"
    if normalized.startswith("M3.13"):
        return "M3.13"
    if normalized.startswith("M3.14"):
        return "M3.14"
    if normalized.startswith("M3.15b"):
        return "M3.15b"
    if normalized.startswith("M3.15c"):
        return "M3.15c"
    if normalized.startswith("M3.15d"):
        return "M3.15d"
    if normalized.startswith("M3.15"):
        return "M3.15"
    if normalized.startswith("M3.16"):
        return "M3.16"
    if normalized.startswith("M3.17"):
        return "M3.17"
    if normalized.startswith("M3.18"):
        return "M3.18"
    if normalized.startswith("M3.19"):
        return "M3.19"
    if normalized.startswith("M3."):
        return "M3"
    if normalized == "M4":
        return "M4"
    if normalized.startswith("M4.0"):
        return "M4.0"
    if normalized.startswith("M4.2"):
        return "M4.2"
    if normalized.startswith("M5.0"):
        return "M5"
    if normalized.startswith("M5.1"):
        return "M5.1"
    if normalized.startswith("M5.2"):
        return "M5.2"
    if normalized.startswith("M5.3"):
        return "M5.3"
    if normalized.startswith("M6."):
        return "M6"
    if normalized.startswith("M7."):
        return "M7"
    if normalized.startswith("M8."):
        return "M8"
    if normalized.startswith("M9."):
        return "M9"
    if normalized.startswith("M10."):
        return "M10"
    if normalized.startswith("M11"):
        return "M11"
    if normalized.startswith("M14"):
        return "M14"
    if normalized.startswith("M18"):
        return "M18"
    if normalized.startswith("M19.3"):
        return "M19"
    if normalized.startswith("M19"):
        return "M19"
    if normalized.startswith("M20"):
        return "M20"
    if normalized.startswith("M21"):
        return "M21"
    if normalized.startswith("M22"):
        return "M22"
    if normalized.startswith("M23"):
        return "M23"
    if normalized.startswith("M24"):
        return "M24"
    if normalized.startswith("M25"):
        return "M25"
    return None


def _program_layer(family_key: str) -> str:
    if family_key in {"A-G", "H", "H5", "J", "L", "J/L Hypercube", "Phase Eval"}:
        return "legacy_orchestration"
    if family_key in {"M1", "M2", "M3", "M3.5", "M3.6", "M3.7", "M3.8", "M3.9", "M3.10", "M3.11", "M3.12", "M3.13", "M3.14", "M3.15", "M3.15b", "M3.15c", "M3.15d", "M3.16", "M3.17", "M3.18", "M3.19", "M4", "M4.0", "M4.2", "M5", "M5.1", "M5.2", "M5.3"}:
        return "bridge_and_serialization"
    if family_key == "M20":
        return "dictionary_first_substrate"
    if family_key == "M21":
        return "dynamic_bridi_substrate"
    if family_key == "M22":
        return "semantic_generalization_substrate"
    if family_key == "M23":
        return "causal_relevance_substrate"
    if family_key == "M24":
        return "substrate_compression"
    if family_key == "M25":
        return "emergent_bridi_grammar"
    return "manifold_and_return_path"


def _legacy_family_key(entry: dict[str, Any]) -> str | None:
    family_group = str(entry.get("family_group") or "")
    canonical_id = str(entry.get("canonical_id") or "")
    if family_group in {"a_to_g_matrix", "core_matrix", "control_duel"}:
        return "A-G"
    if family_group == "h_series":
        return "H5" if "h5" in canonical_id.lower() else "H"
    if family_group == "h5_bridge":
        return "H5"
    if family_group == "j_series":
        return "J"
    if family_group == "l_series":
        return "L"
    if canonical_id.startswith("phase5."):
        return "Phase Eval"
    return None


def _seed_legacy_families(series_family_manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = {}
    for manifest in series_family_manifests:
        family_key = str(manifest.get("series_key") or "").strip()
        if not family_key:
            continue
        families[family_key] = {
            "family_key": family_key,
            "program_layer": _program_layer(family_key),
            "normalized_ids": sorted(manifest.get("normalized_ids", []), key=_normalized_sort_key),
            "legacy_aliases": sorted(manifest.get("legacy_aliases", [])),
            "canonical_ids": sorted(manifest.get("canonical_ids", [])),
            "title_summary": str(manifest.get("objective") or manifest.get("title") or ""),
            "script_paths": sorted(set(manifest.get("script_paths", []))),
            "dag_paths": sorted(set([*manifest.get("dag_paths", []), *FAMILY_DAG_MAP.get(family_key, [])])),
            "doc_paths": sorted(set(manifest.get("doc_paths", []))),
            "artifact_roots": sorted(set(manifest.get("artifact_roots", []))),
            "family_groups": sorted(set(manifest.get("family_groups", []))),
            "reproducibility": [],
            "evidence_classes": [],
            "entry_count": int(manifest.get("entry_count") or 0),
            "runnable_count": int(manifest.get("runnable_count") or 0),
            "artifact_only_count": int(manifest.get("artifact_only_count") or 0),
            "doc_only_count": int(manifest.get("doc_only_count") or 0),
            "status_summary": "mixed_historical",
        }
        families[family_key]["status_summary"] = _status_summary(families[family_key])
    return families


def _build_family_map(entries: list[dict[str, Any]], series_family_manifests: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    families: dict[str, dict[str, Any]] = _seed_legacy_families(series_family_manifests)
    for entry in entries:
        if _legacy_family_key(entry) is not None:
            continue
        family_key = _family_key(entry)
        if family_key is None:
            continue
        if family_key not in families:
            families[family_key] = {
                "family_key": family_key,
                "program_layer": _program_layer(family_key),
                "normalized_ids": [],
                "legacy_aliases": set(),
                "canonical_ids": [],
                "titles": [],
                "script_paths": set(),
                "dag_paths": set(),
                "doc_paths": set(),
                "artifact_roots": set(),
                "family_groups": set(),
                "reproducibility": set(),
                "evidence_classes": set(),
                "entry_count": 0,
                "runnable_count": 0,
                "artifact_only_count": 0,
                "doc_only_count": 0,
            }
        row = families[family_key]
        row["entry_count"] += 1
        row["canonical_ids"].append(str(entry.get("canonical_id") or ""))
        row["titles"].append(str(entry.get("title") or ""))
        normalized = str(entry.get("normalized_canonical_id") or "")
        if normalized and normalized not in row["normalized_ids"]:
            row["normalized_ids"].append(normalized)
        for alias in entry.get("aliases", []):
            alias_str = str(alias).strip()
            if alias_str and alias_str != normalized:
                row["legacy_aliases"].add(alias_str)
        for path in entry.get("script_paths", []):
            if str(path).strip():
                row["script_paths"].add(str(path))
        for path in entry.get("dag_paths", []):
            if str(path).strip():
                row["dag_paths"].add(str(path))
        for path in entry.get("doc_paths", []):
            if str(path).strip():
                row["doc_paths"].add(str(path))
        for path in entry.get("artifact_roots", []):
            if str(path).strip():
                row["artifact_roots"].add(str(path))
        family_group = str(entry.get("family_group") or "").strip()
        if family_group:
            row["family_groups"].add(family_group)
        repro = str(entry.get("reproducibility_status") or "")
        if repro:
            row["reproducibility"].add(repro)
        ev = str(entry.get("evidence_class") or "")
        if ev:
            row["evidence_classes"].add(ev)
        if repro == "runnable":
            row["runnable_count"] += 1
        elif repro == "artifact_only":
            row["artifact_only_count"] += 1
        elif repro == "doc_only":
            row["doc_only_count"] += 1

    for family_key, row in families.items():
        dag_paths = set(row.get("dag_paths", []))
        dag_paths.update(FAMILY_DAG_MAP.get(family_key, []))
        row["dag_paths"] = sorted(dag_paths)
        row["normalized_ids"] = sorted(row["normalized_ids"], key=_normalized_sort_key)
        row["canonical_ids"] = sorted(set(row["canonical_ids"]))
        row["legacy_aliases"] = sorted(set(row["legacy_aliases"]))
        row["script_paths"] = sorted(set(row["script_paths"]))
        row["doc_paths"] = sorted(set(row.get("doc_paths", [])))
        row["artifact_roots"] = sorted(set(row.get("artifact_roots", [])))
        row["family_groups"] = sorted(set(row.get("family_groups", [])))
        row["reproducibility"] = sorted(set(row["reproducibility"]))
        row["evidence_classes"] = sorted(set(row["evidence_classes"]))
        if "titles" in row:
            row["title_summary"] = _title_summary(row["titles"])
        row["status_summary"] = _status_summary(row)
    return families


def _status_summary(row: dict[str, Any]) -> str:
    if row["runnable_count"] > 0:
        return "partially_runnable" if row["artifact_only_count"] > 0 or row["doc_only_count"] > 0 else "runnable"
    if row["artifact_only_count"] > 0:
        return "artifact_only"
    if row["doc_only_count"] > 0:
        return "historical_only"
    return "mixed_historical"


def _title_summary(titles: list[str]) -> str:
    unique_titles = [title for title in sorted(set(titles)) if title]
    if not unique_titles:
        return ""
    if len(unique_titles) == 1:
        return unique_titles[0]
    return f"{unique_titles[0]} + {len(unique_titles) - 1} more"


def _normalized_sort_key(value: str) -> tuple[Any, ...]:
    parts = str(value).replace("M", "").split(".")
    out: list[tuple[int, Any]] = []
    for part in parts:
        if part.isdigit():
            out.append((0, int(part)))
        else:
            out.append((1, part))
    return tuple(out)


def _ordered_family_rows(families: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_keys = [*LETTER_FAMILY_ORDER, *M_FAMILY_ORDER, "History"]
    rows: list[dict[str, Any]] = []
    for key in ordered_keys:
        if key in families:
            rows.append(families[key])
    for key in sorted(k for k in families if k not in ordered_keys):
        rows.append(families[key])
    rows.append(
        {
            "family_key": "History",
            "program_layer": "control_plane",
            "normalized_ids": [],
            "legacy_aliases": [],
            "canonical_ids": [],
            "title_summary": "Backfill and aggregate suite control plane",
            "script_paths": [
                "scripts/control_plane/run_ablation_history_backfill.py",
                "scripts/control_plane/render_ablation_history_catalog.py",
                "scripts/m_bridge/run_m_bridge_ablation_test_suite.py",
                "scripts/control_plane/build_ablation_program_map.py",
                "scripts/control_plane/build_ablation_program_spine.py",
            ],
            "dag_paths": sorted([*FAMILY_DAG_MAP["History"], "airflow/dags/control_plane/lojban_ablation_program_spine_dag.py", "airflow/dags/control_plane/lojban_ablation_master_spine_dag.py"]),
            "doc_paths": [
                "docs/ABLATION_HISTORY_FULL.md",
                "docs/ABLATION_PROGRAM_MAP.md",
                "docs/ABLATION_PROGRAM_SPINE.md",
            ],
            "artifact_roots": [],
            "family_groups": ["control_plane"],
            "reproducibility": ["runnable"],
            "evidence_classes": ["artifact"],
            "entry_count": 0,
            "runnable_count": 1,
            "artifact_only_count": 0,
            "doc_only_count": 0,
            "status_summary": "runnable",
        }
    )
    return rows


def _canonical_path(value: str) -> str:
    return canonical_repo_path(value, REPO_ROOT)


def _canonicalize_manifest_paths(manifest: dict[str, Any]) -> None:
    for row in manifest.get("families", []):
        row["script_paths"] = sorted({_canonical_path(path) for path in row.get("script_paths", []) if str(path).strip()})
        row["dag_paths"] = sorted({_canonical_path(path) for path in row.get("dag_paths", []) if str(path).strip()})


def _render_markdown(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Ablation Program Map")
    lines.append("")
    lines.append(f"- Generated UTC: `{manifest['generated_utc']}`")
    lines.append(f"- Source history manifest: `{manifest['source_history_manifest']}`")
    lines.append(f"- Concentrated family count: `{manifest['family_count']}`")
    lines.append("")
    lines.append("## Program Layers")
    lines.append("")
    lines.append("- `legacy_orchestration`: letter-era experiments and their early DAG architecture")
    lines.append("- `bridge_and_serialization`: early-to-mid M-series bridge, grounding, and serialization families")
    lines.append("- `manifold_and_return_path`: later manifold/native/discriminative/re-entry families")
    lines.append("- `dictionary_first_substrate`: M20 dictionary-first substrate branch")
    lines.append("- `dynamic_bridi_substrate`: M21 dynamic bridi substrate branch")
    lines.append("- `semantic_generalization_substrate`: M22 semantic coverage generalization gate")
    lines.append("- `causal_relevance_substrate`: M23 causal relevance-router fork")
    lines.append("- `substrate_compression`: M24 substrate compression branch")
    lines.append("- `emergent_bridi_grammar`: M25 loose bridi grammar-action stream branch")
    lines.append("- `control_plane`: the backfill, catalog, and aggregate-suite layer")
    lines.append("")
    lines.append("## Concentrated Families")
    lines.append("")
    for row in manifest["families"]:
        lines.append(f"### {row['family_key']}")
        lines.append("")
        lines.append(f"- Layer: `{row['program_layer']}`")
        lines.append(f"- Status: `{row['status_summary']}`")
        if row.get("normalized_ids"):
            lines.append(f"- Normalized IDs: `{', '.join(row['normalized_ids'])}`")
        if row.get("legacy_aliases"):
            lines.append(f"- Legacy aliases: `{', '.join(row['legacy_aliases'][:24])}`")
        lines.append(f"- Entry count: `{row['entry_count']}`")
        lines.append(f"- Runnable rows: `{row['runnable_count']}`")
        if row.get("artifact_only_count"):
            lines.append(f"- Artifact-only rows: `{row['artifact_only_count']}`")
        if row.get("doc_only_count"):
            lines.append(f"- Doc-only rows: `{row['doc_only_count']}`")
        if row.get("title_summary"):
            lines.append(f"- Brief: {row['title_summary']}")
        if row.get("family_groups"):
            lines.append(f"- Family groups: `{', '.join(row['family_groups'])}`")
        if row.get("doc_paths"):
            lines.append(f"- Docs: `{', '.join(row['doc_paths'])}`")
        if row.get("script_paths"):
            lines.append(f"- Scripts: `{', '.join(row['script_paths'])}`")
        if row.get("dag_paths"):
            lines.append(f"- DAGs: `{', '.join(row['dag_paths'])}`")
        if row.get("artifact_roots"):
            lines.append(f"- Artifact roots: `{', '.join(row['artifact_roots'][:8])}`")
        lines.append("")
    lines.append("## Transition Spine")
    lines.append("")
    for transition in manifest.get("transitions", []):
        lines.append(f"- `{transition['transition_id']}`: `{transition['from_major']} -> {transition['to_major']}` via `{transition['selected_upstream']}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
