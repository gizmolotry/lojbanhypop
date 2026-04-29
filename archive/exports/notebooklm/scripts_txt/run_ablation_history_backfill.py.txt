from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.ablation_history_registry import (
    ABLATION_HISTORY_SCHEMA_VERSION,
    add_evidence,
    build_evidence_record,
    build_history_slice,
    build_lookup_index,
    ensure_entry,
    finalize_registry,
    flatten_history_rows,
    normalize_metric_surface,
    safe_float,
    slugify,
)
from lojban_evolution.experiment_taxonomy import (
    build_transition_index,
    enrich_history_entries,
    load_taxonomy_config,
)
from lojban_evolution.repo_paths import legacy_doc_path, repo_relative
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


DOC_SOURCES = {
    "canonical_ledger": repo_relative(legacy_doc_path("CANONICAL_LEDGER.md")),
    "audit_report": repo_relative(legacy_doc_path("AUDIT_REPORT.md")),
    "h5_ablation_report": repo_relative(legacy_doc_path("H5_ABLATION_REPORT.md")),
    "l_series_doc": "docs/L_SERIES.md",
    "series_charter": "docs/SERIES_CHARTER.md",
}


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


LETTER_SERIES_SPECS: list[dict[str, Any]] = [
    {
        "series_key": "A-G",
        "title": "A-G matrix",
        "family_groups": ["a_to_g_matrix", "core_matrix", "control_duel"],
        "objective": "early benchmark matrix covering base control, projected handoff, coconut variants, and English-vs-Lojban control comparisons.",
        "doc_paths": [
            "docs/ledger/CANONICAL_LEDGER.md",
            "docs/history/reports/AUDIT_REPORT.md",
        ],
        "script_paths": ["scripts/run_coconut_ablation_matrix.py"],
        "dag_paths": ["airflow/dags/lojban_ablation_matrix_dag.py"],
    },
    {
        "series_key": "H",
        "title": "H bridge series",
        "family_groups": ["h_series"],
        "family_prefixes": ["legacy.h."],
        "exclude_prefixes": ["legacy.h5.", "h.series.h5"],
        "objective": "mid-layer bridge experiments testing linear and SwiGLU geometric handoff into the host decoder.",
        "doc_paths": [
            "docs/ledger/CANONICAL_LEDGER.md",
            "docs/history/reports/NUMERICAL_AUDIT.md",
        ],
        "script_paths": ["scripts/true_coconut.py"],
        "dag_paths": [],
    },
    {
        "series_key": "H5",
        "title": "H5 bridge extension",
        "family_groups": ["h5_bridge"],
        "family_prefixes": ["legacy.h5.", "h.series.h5"],
        "objective": "boolean-surgery, persistent-advisor, and bridge-extension experiments that fed into the later J and L stacks.",
        "doc_paths": [
            "docs/history/reports/H5_ABLATION_REPORT.md",
            "docs/history/reports/H5_SUMMARY.md",
            "docs/H5_ABLATION_EXTENSION.md",
        ],
        "script_paths": [
            "scripts/train_h5_persistent_vq_advisor.py",
            "scripts/eval_h5_dynamic_pointer_refactor.py",
            "scripts/eval_h5_ood_stress.py",
            "scripts/trace_h5_provenance.py",
        ],
        "dag_paths": [],
    },
    {
        "series_key": "J",
        "title": "J series",
        "family_groups": ["j_series"],
        "objective": "advisor-side data generation and adversarial synthesis family that seeded the later numeric M-line.",
        "doc_paths": [
            "docs/SERIES_CHARTER.md",
            "docs/ledger/CANONICAL_LEDGER.md",
        ],
        "script_paths": [
            "scripts/eval_j_1.py",
            "scripts/eval_j_2.py",
            "scripts/eval_j_3.py",
            "scripts/eval_j_4.py",
            "scripts/eval_j_5.py",
            "scripts/train_h5_persistent_vq_advisor.py",
        ],
        "dag_paths": ["airflow/dags/lojban_j_series_dag.py"],
    },
    {
        "series_key": "L",
        "title": "L series",
        "family_groups": ["l_series"],
        "objective": "lagrangian constrained-manifold family and its branch lineages before the later M unification.",
        "doc_paths": [
            "docs/L_SERIES.md",
            "docs/SERIES_CHARTER.md",
            "archive/reports/relevant/REPORTS_RELEVANT/l6_ablation_manifest.md",
        ],
        "script_paths": [
            "scripts/train_l_series_mvs.py",
            "scripts/run_l6_ablation_branch.py",
        ],
        "dag_paths": ["airflow/dags/lojban_l_series_dag.py"],
    },
    {
        "series_key": "J/L Hypercube",
        "title": "J/L hypercube aggregation",
        "family_groups": [],
        "canonical_ids": ["legacy.hypercube.j_l"],
        "objective": "cross-family orchestration and hypercube reporting layer that consolidated J/L-era runs before the modern M suite.",
        "doc_paths": [
            "archive/reports/relevant/REPORTS_RELEVANT/ablation_hypercube_report.md",
            "archive/results/legacy_misc/20260305/RESULTS_FULL_GRID_20260305/ablation_hypercube_report.md",
        ],
        "script_paths": [],
        "dag_paths": ["airflow/dags/lojban_ablation_hypercube_report_dag.py"],
    },
    {
        "series_key": "Phase Eval",
        "title": "Phase-5 eval families",
        "family_groups": ["phase5_train_ablation", "phase5_objective_ablation"],
        "objective": "phase-5 train/objective ablations used to stress semantic and compression loss surfaces before later M-series serialization work.",
        "doc_paths": [
            "docs/SERIES_CHARTER.md",
            "docs/CAUSAL_PROBE_PROTOCOL.md",
        ],
        "script_paths": [
            "scripts/run_phase5_train_ablation.py",
            "scripts/run_phase5_objective_ablation.py",
        ],
        "dag_paths": ["airflow/dags/lojban_phase_ablation_dag.py"],
    },
]


L_ORIGIN_TRACKS = {
    "M3+",
    "M3.5",
    "M3.6",
    "M3.7",
    "M3.8",
    "M4",
}


ARCHIVE_M_REPORTS: list[dict[str, Any]] = [
    {
        "path": "archive/results/m6/20260314/RESULTS_M6_SEVERED_BRIDGE_20260314/m6_eval_report.json",
        "canonical_id": "m.track.m6_0",
        "title": "M6 severed bridge",
        "aliases": ["M6.0", "RESULTS_M6_SEVERED_BRIDGE_20260314"],
        "lookup_aliases": ["M6.0", "m6:severed_bridge"],
        "objective": "direct logic-engine bridge baseline before later alignment and scratchpad variants",
        "baseline_relation": "m6 baseline",
        "script_paths": ["scripts/train_m6_logic_engine.py", "scripts/eval_m6_logic_engine.py"],
    },
    {
        "path": "archive/results/m6_1/active/RESULTS_M6_1_ALIGNMENT_70ACC/m6_eval_report.json",
        "canonical_id": "m.track.m6_1",
        "title": "M6.1 alignment 70acc",
        "aliases": ["M6.1", "RESULTS_M6_1_ALIGNMENT_70ACC"],
        "lookup_aliases": ["M6.1", "m6:1"],
        "objective": "alignment-focused M6 variant with improved held-out accuracy",
        "baseline_relation": "m6 aligned variant",
        "script_paths": ["scripts/train_m6_logic_engine.py", "scripts/eval_m6_logic_engine.py"],
    },
    {
        "path": "archive/results/m6_2/active/RESULTS_M6_2_ALIGNED_30ACC/m6_eval_report.json",
        "canonical_id": "m.track.m6_2",
        "title": "M6.2 aligned 30acc",
        "aliases": ["M6.2", "RESULTS_M6_2_ALIGNED_30ACC"],
        "lookup_aliases": ["M6.2", "m6:2"],
        "objective": "recalibrated M6 alignment follow-up with lower headline accuracy",
        "baseline_relation": "m6 aligned variant",
        "script_paths": ["scripts/train_m6_logic_engine.py", "scripts/eval_m6_logic_engine.py"],
    },
    {
        "path": "archive/results/m6_3/active/RESULTS_M6_3_SCRATCHPAD_35ACC/m6_directed_eval_report.json",
        "canonical_id": "m.track.m6_3",
        "title": "M6.3 scratchpad 35acc",
        "aliases": ["M6.3", "RESULTS_M6_3_SCRATCHPAD_35ACC"],
        "lookup_aliases": ["M6.3", "m6:3"],
        "objective": "scratchpad-directed M6 branch before later dedicated scratchpad re-entry families",
        "baseline_relation": "m6 scratchpad variant",
        "script_paths": ["scripts/train_m6_logic_engine.py", "scripts/eval_m6_logic_engine.py"],
    },
    {
        "path": "archive/results/m6_6/active/RESULTS_M6_6_DIRECTED_AST_FINAL/m6_expansive_report.json",
        "canonical_id": "m.track.m6_6",
        "title": "M6.6 directed AST final",
        "aliases": ["M6.6", "RESULTS_M6_6_DIRECTED_AST_FINAL"],
        "lookup_aliases": ["M6.6", "m6:6"],
        "objective": "late M6 directed-AST expansion branch",
        "baseline_relation": "m6 expansive variant",
        "script_paths": ["scripts/train_m6_logic_engine.py", "scripts/eval_m6_logic_engine.py"],
    },
    {
        "path": "archive/results/m7/active/RESULTS_M7_INTERLEAVED_COPROCESSOR/m7_eval_report.json",
        "canonical_id": "m.track.m7_0",
        "title": "M7 interleaved coprocessor",
        "aliases": ["M7.0", "M7", "RESULTS_M7_INTERLEAVED_COPROCESSOR"],
        "lookup_aliases": ["M7.0", "M7", "m7:interleaved"],
        "objective": "interleaved coprocessor rollout after the M6 bridge family",
        "baseline_relation": "m7 family row",
        "script_paths": ["scripts/train_m7_interleaved.py", "scripts/eval_m7_interleaved.py"],
    },
    {
        "path": "archive/results/m8/active/RESULTS_M8_COUNCIL_OF_ORACLES/m8_eval_report.json",
        "canonical_id": "m.track.m8_0",
        "title": "M8 council of oracles",
        "aliases": ["M8.0", "M8", "RESULTS_M8_COUNCIL_OF_ORACLES"],
        "lookup_aliases": ["M8.0", "M8", "m8:council"],
        "objective": "council-of-oracles composition family",
        "baseline_relation": "m8 family row",
        "script_paths": ["scripts/train_m8_council.py", "scripts/eval_m8_council.py"],
    },
    {
        "path": "archive/results/m9/active/RESULTS_M9_AUDIT/m9_audit_report.json",
        "canonical_id": "m.track.m9_0",
        "title": "M9 provenance manifold audit",
        "aliases": ["M9.0", "M9.audit", "RESULTS_M9_AUDIT"],
        "lookup_aliases": ["M9.0", "M9.audit", "m9:audit"],
        "objective": "audit the provenance-manifold system after phase synchronization",
        "baseline_relation": "m9 audit row",
        "script_paths": ["scripts/m9/eval_m9.py"],
    },
    {
        "path": "archive/results/m9/active/RESULTS_M9_HYPERCUBE/duel_report.json",
        "canonical_id": "m.track.m9_1",
        "title": "M9 duel hypercube",
        "aliases": ["M9.1", "M9.hypercube", "RESULTS_M9_HYPERCUBE"],
        "lookup_aliases": ["M9.1", "M9.hypercube", "m9:hypercube"],
        "objective": "hypercube duel comparison inside the M9 manifold regime",
        "baseline_relation": "m9 hypercube row",
        "script_paths": ["scripts/m9/eval_m9.py"],
    },
    {
        "path": "archive/results/m10/active/RESULTS_M10_AUDIT/m10_audit_report.json",
        "canonical_id": "m.track.m10_0",
        "title": "M10 audit",
        "aliases": ["M10.0", "M10.audit", "RESULTS_M10_AUDIT"],
        "lookup_aliases": ["M10.0", "M10.audit", "m10:audit"],
        "objective": "translator and English-head audit before final M11-native discriminative redirection",
        "baseline_relation": "m10 audit row",
        "script_paths": ["scripts/m10/final_audit.py"],
    },
    {
        "path": "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_bridge_audit.json",
        "canonical_id": "m.track.m10_1",
        "title": "M10 final bridge audit",
        "aliases": ["M10.1", "M10.final_bridge", "final_bridge_audit"],
        "lookup_aliases": ["M10.1", "M10.final_bridge", "m10:final_bridge"],
        "objective": "bridge-side final audit on the M10 translator/head stack",
        "baseline_relation": "m10 final bridge row",
        "script_paths": ["scripts/m10/final_audit.py"],
    },
    {
        "path": "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_floor_lock.json",
        "canonical_id": "m.track.m10_2",
        "title": "M10 floor lock audit",
        "aliases": ["M10.2", "M10.floor_lock", "final_floor_lock"],
        "lookup_aliases": ["M10.2", "M10.floor_lock", "m10:floor_lock"],
        "objective": "larger-sample M10 floor-lock evaluation",
        "baseline_relation": "m10 floor-lock row",
        "script_paths": ["scripts/m10/final_audit.py"],
    },
    {
        "path": "archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_publication_metrics.json",
        "canonical_id": "m.track.m10_3",
        "title": "M10 publication metrics",
        "aliases": ["M10.3", "M10.publication", "final_publication_metrics"],
        "lookup_aliases": ["M10.3", "M10.publication", "m10:publication"],
        "objective": "publication-facing summary metrics for the late M10 line",
        "baseline_relation": "m10 publication row",
        "script_paths": ["scripts/m10/final_audit.py"],
    },
]


DOC_CLAIMS: list[dict[str, Any]] = [
    {
        "canonical_id": "legacy.core.a",
        "family": "legacy_core_matrix",
        "family_group": "core_matrix",
        "title": "Base Model (No Adapter)",
        "aliases": ["A", "Core/A"],
        "lookup_aliases": ["legacy:A", "core:A"],
        "objective": "core baseline control without a symbolic adapter",
        "baseline_relation": "base control",
        "metrics": {"held_out_accuracy": 0.167},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
        "notes": "Documented as the foundational control group in the canonical ledger.",
    },
    {
        "canonical_id": "legacy.core.b",
        "family": "legacy_core_matrix",
        "family_group": "core_matrix",
        "title": "Rigid Symbolic (Phase 5)",
        "aliases": ["B", "Core/B", "Run B"],
        "lookup_aliases": ["legacy:B", "core:B", "run:B"],
        "objective": "control baseline for two-stage symbolic recovery",
        "baseline_relation": "control baseline",
        "metrics": {"held_out_accuracy": 0.396},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
        "notes": "Canonical control baseline before later bridge and reboot series.",
    },
    {
        "canonical_id": "legacy.core.c",
        "family": "legacy_core_matrix",
        "family_group": "core_matrix",
        "title": "KV Handoff",
        "aliases": ["C", "Core/C"],
        "lookup_aliases": ["legacy:C", "core:C"],
        "objective": "test direct KV handoff without explicit topology",
        "baseline_relation": "negative bridge ablation",
        "metrics": {"held_out_accuracy": 0.104},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.core.e",
        "family": "legacy_core_matrix",
        "family_group": "core_matrix",
        "title": "Babel Expansion",
        "aliases": ["E", "Core/E"],
        "lookup_aliases": ["legacy:E", "core:E"],
        "objective": "test unconstrained vocabulary expansion",
        "baseline_relation": "negative control",
        "metrics": {"held_out_accuracy": 0.167},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.core.f",
        "family": "legacy_core_matrix",
        "family_group": "core_matrix",
        "title": "Self-Correction",
        "aliases": ["F", "Core/F"],
        "lookup_aliases": ["legacy:F", "core:F"],
        "objective": "measure rollback-heavy self-correction as an alternative to rigid topology",
        "baseline_relation": "high-compute ablation",
        "metrics": {"held_out_accuracy": 0.312},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.core.g",
        "family": "legacy_core_matrix",
        "family_group": "core_matrix",
        "title": "True Coconut",
        "aliases": ["G", "Core/G"],
        "lookup_aliases": ["legacy:G", "core:G"],
        "objective": "test continuous latent prefix without English output until the end",
        "baseline_relation": "negative control",
        "metrics": {"held_out_accuracy": 0.104},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.duel.english_cot",
        "family": "english_vs_lojban_duel",
        "family_group": "control_duel",
        "title": "Monolithic English CoT",
        "aliases": ["English CoT", "Control Duel/English"],
        "lookup_aliases": ["duel:english", "control_duel:english"],
        "objective": "directly compare pure English chain-of-thought against rigid topology",
        "baseline_relation": "language-medium control",
        "metrics": {"held_out_accuracy": 0.0, "final_answer_lift": -0.25},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.duel.lojban_topology",
        "family": "english_vs_lojban_duel",
        "family_group": "control_duel",
        "title": "Rigid Lojban Topology",
        "aliases": ["Lojban Dual", "Control Duel/Lojban"],
        "lookup_aliases": ["duel:lojban", "control_duel:lojban"],
        "objective": "directly compare rigid topology against English chain-of-thought",
        "baseline_relation": "language-medium challenger",
        "metrics": {"held_out_accuracy": 0.417, "final_answer_lift": 0.167},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h.h1",
        "family": "legacy_h_series",
        "family_group": "h_series",
        "title": "Linear Bridge",
        "aliases": ["H1"],
        "lookup_aliases": ["legacy:H1"],
        "objective": "inject learned topological latent space into middle English layers via linear bridge",
        "baseline_relation": "mid-layer bridge ablation",
        "metrics": {"held_out_accuracy": 0.0, "final_answer_lift": -0.167, "geometry_retention": 0.456},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h.h2",
        "family": "legacy_h_series",
        "family_group": "h_series",
        "title": "Deep Linear Bridge",
        "aliases": ["H2"],
        "lookup_aliases": ["legacy:H2"],
        "objective": "test deeper linear mid-layer bridge for geometry retention",
        "baseline_relation": "mid-layer bridge ablation",
        "metrics": {"held_out_accuracy": 0.042, "final_answer_lift": -0.125, "geometry_retention": 0.923},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h.h3",
        "family": "legacy_h_series",
        "family_group": "h_series",
        "title": "SwiGLU Non-Linear Bridge",
        "aliases": ["H3"],
        "lookup_aliases": ["legacy:H3"],
        "objective": "test non-linear mid-layer bridge with SwiGLU expansion",
        "baseline_relation": "mid-layer bridge ablation",
        "metrics": {"held_out_accuracy": 0.0, "final_answer_lift": -0.167, "geometry_retention": 0.934},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h.h4",
        "family": "legacy_h_series",
        "family_group": "h_series",
        "title": "Deep SwiGLU Bridge",
        "aliases": ["H4"],
        "lookup_aliases": ["legacy:H4"],
        "objective": "test deeper non-linear mid-layer bridge with SwiGLU",
        "baseline_relation": "mid-layer bridge ablation",
        "metrics": {"held_out_accuracy": 0.083, "final_answer_lift": -0.083, "geometry_retention": 0.925},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h5.h5_2a",
        "family": "legacy_h5_bridge",
        "family_group": "h5_bridge",
        "title": "H5.2a Gearbox Control",
        "aliases": ["H5.2a", "Gearbox Control"],
        "lookup_aliases": ["legacy:H5.2a", "h5:2a"],
        "objective": "monotonic pointer gearbox bridge without boolean surgery",
        "baseline_relation": "slice-2 control",
        "metrics": {"logical_accuracy": 0.375, "final_ce_loss": 13.24},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h5.h5_2b",
        "family": "legacy_h5_bridge",
        "family_group": "h5_bridge",
        "title": "H5.2b True Neuro-Symbolic Bridge",
        "aliases": ["H5.2b", "True Neuro-Symbolic"],
        "lookup_aliases": ["legacy:H5.2b", "h5:2b"],
        "objective": "slice-2 bridge with log-space boolean surgery enabled",
        "baseline_relation": "slice-2 bridge intervention",
        "metrics": {"logical_accuracy": 0.375, "final_ce_loss": 13.24, "surgery_trigger_rate": 0.0},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h5.h5_4",
        "family": "legacy_h5_bridge",
        "family_group": "h5_bridge",
        "title": "H5.4 Forced Boolean Manifold",
        "aliases": ["H5.4", "Iron Collar"],
        "lookup_aliases": ["legacy:H5.4", "h5:4"],
        "objective": "force strict arity mask and boolean slot typing",
        "baseline_relation": "forced manifold intervention",
        "metrics": {"logical_accuracy": 0.0, "surgery_trigger_rate": 1.0},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
    {
        "canonical_id": "legacy.h5.h5_5",
        "family": "legacy_h5_bridge",
        "family_group": "h5_bridge",
        "title": "H5.5 Grounded Fine-Tune",
        "aliases": ["H5.5"],
        "lookup_aliases": ["legacy:H5.5", "h5:5"],
        "objective": "recover semantic decoding while preserving the iron collar manifold",
        "baseline_relation": "semantic recovery",
        "metrics": {"logical_accuracy": 1.0},
        "source_key": "canonical_ledger",
        "reported_at": "2026-03-02T00:00:00+00:00",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill the full ablation history into one canonical registry and ledger.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill"),
    )
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--include-git", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("M", args.output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("history_backfill_%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    registry: dict[str, dict[str, Any]] = {}
    source_paths: set[str] = set()
    historical_gaps: list[dict[str, Any]] = []

    _collect_doc_claims(registry, source_paths, historical_gaps)
    _collect_l_series_docs(registry, source_paths)
    _collect_l_series_artifacts(registry, source_paths)
    _collect_phase5_train_manifests(registry, source_paths)
    _collect_phase5_datapack(registry, source_paths)
    _collect_phase5_objective_ablation(registry, source_paths)
    _collect_a_to_g_artifacts(registry, source_paths)
    _collect_h_series_artifacts(registry, source_paths)
    _collect_j_series_artifacts(registry, source_paths)
    _collect_telemetry_reports(registry, source_paths)
    _collect_archive_m_reports(registry, source_paths)
    if args.include_git:
        _collect_git_evidence(registry, source_paths)

    entries = finalize_registry(registry)
    taxonomy = load_taxonomy_config()
    entries = enrich_history_entries(entries, taxonomy)
    series_family_manifests = _build_letter_series_manifests(entries)
    lookup_index = build_lookup_index(entries)
    all_rows = flatten_history_rows(entries, mode="all")
    artifact_rows = flatten_history_rows(entries, mode="artifact_only")
    runnable_rows = flatten_history_rows(entries, mode="runnable_only")
    family_counts = _count_entries(entries, "family")
    family_group_counts = _count_entries(entries, "family_group")

    manifest = {
        "schema_version": ABLATION_HISTORY_SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M.ablation_history_backfill", "scripts/run_ablation_history_backfill.py"),
        "lineage": lineage_metadata(
            "eval_only",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile="historical_ablation_registry",
            difficulty_tier="mixed",
        ),
        "run_id": run_id,
        "output_root": str(args.output_root).replace("\\", "/"),
        "source_paths": sorted(source_paths),
        "family_counts": family_counts,
        "family_group_counts": family_group_counts,
        "series_family_counts": {row["series_key"]: int(row["entry_count"]) for row in series_family_manifests},
        "series_family_manifests": series_family_manifests,
        "transition_manifests": build_transition_index(taxonomy),
        "history_slices": {
            "all": build_history_slice(entries, mode="all"),
            "artifact_only": build_history_slice(entries, mode="artifact_only"),
            "runnable_only": build_history_slice(entries, mode="runnable_only"),
        },
        "lookup_index": lookup_index,
        "historical_gaps": historical_gaps,
        "entries": entries,
    }

    manifest_path = run_dir / "ablation_history_manifest.json"
    ledger_path = run_dir / "ablation_history_ledger.md"
    letter_series_path = run_dir / "letter_series_families.md"
    rows_all_path = run_dir / "ablation_history_rows_all.csv"
    rows_artifact_path = run_dir / "ablation_history_rows_artifact_only.csv"
    rows_runnable_path = run_dir / "ablation_history_rows_runnable_only.csv"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_csv(rows_all_path, all_rows)
    _write_csv(rows_artifact_path, artifact_rows)
    _write_csv(rows_runnable_path, runnable_rows)
    _write_ledger_md(ledger_path, manifest)
    _write_letter_series_md(letter_series_path, series_family_manifests)
    print(f"Ablation history backfill complete: {run_dir}")


def _collect_doc_claims(
    registry: dict[str, dict[str, Any]],
    source_paths: set[str],
    historical_gaps: list[dict[str, Any]],
) -> None:
    for claim in DOC_CLAIMS:
        path = DOC_SOURCES[claim["source_key"]]
        source_paths.add(path)
        entry = ensure_entry(
            registry,
            canonical_id=claim["canonical_id"],
            family=claim["family"],
            family_group=claim["family_group"],
            title=claim["title"],
            aliases=claim.get("aliases"),
            lookup_aliases=claim.get("lookup_aliases"),
            objective=claim.get("objective"),
            baseline_relation=claim.get("baseline_relation"),
            notes=[claim.get("notes")] if claim.get("notes") else None,
        )
        metrics = claim.get("metrics", {})
        add_evidence(
            entry,
            build_evidence_record(
                source_label=f"doc:{claim['source_key']}",
                source_paths=[path],
                evidence_class="doc_reported",
                confidence_level="medium",
                reproducibility_status="doc_only",
                metrics=metrics,
                normalized_metrics=normalize_metric_surface(metrics),
                reported_at=claim.get("reported_at"),
                notes=claim.get("notes"),
            ),
        )

    source_paths.add(DOC_SOURCES["h5_ablation_report"])
    source_paths.add(DOC_SOURCES["audit_report"])
    baseline_entry = ensure_entry(
        registry,
        canonical_id="restoration.phase5.two_stage_baseline",
        family="historical_gap",
        family_group="historical_gap",
        title="Two-stage phase-5 baseline restoration path",
        aliases=["Restoration Path", "Lost Asset"],
        lookup_aliases=["restoration:phase5_baseline"],
        objective="document the missing production baseline asset and the restoration command path",
        baseline_relation="restoration target",
        script_paths=["scripts/run_phase5_two_stage_recovery.py", "scripts/mine_compositional_anchors.py"],
        notes=[
            "Doc-reported missing asset for the original two-stage control baseline.",
            "Serves as gap handling rather than a confirmed artifact-backed ablation run.",
        ],
    )
    restoration_metrics = {"held_out_accuracy": 0.396, "logical_accuracy": 0.417}
    add_evidence(
        baseline_entry,
        build_evidence_record(
            source_label="doc:audit_report",
            source_paths=[DOC_SOURCES["audit_report"]],
            evidence_class="doc_reported",
            confidence_level="medium",
            reproducibility_status="orphaned",
            metrics=restoration_metrics,
            normalized_metrics=normalize_metric_surface(restoration_metrics),
            reported_at="2026-03-01T00:00:00+00:00",
            notes="Audit report marks the original adapter as lost and documents the restoration steps.",
        ),
    )
    historical_gaps.append(
        {
            "gap_id": "missing.phase5_two_stage_stage2",
            "kind": "missing_artifact",
            "reported_in": DOC_SOURCES["audit_report"],
            "path": "runs/phase5_two_stage_recovery_anchors/20260224_225142/stage2_phase5",
            "notes": "Original Run B control baseline asset reported missing in audit report.",
        }
    )
    historical_gaps.append(
        {
            "gap_id": "orphaned.h3_projection_weights",
            "kind": "orphaned_checkpoint",
            "reported_in": DOC_SOURCES["audit_report"],
            "path": "runs/projections/swiglu_midlayer_bridge_h3_exp4.pt",
            "notes": "Audit report describes H3/H4 projection weights as mathematically orphaned relative to the deleted adapter manifold.",
        }
    )


def _collect_l_series_docs(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    path = DOC_SOURCES["l_series_doc"]
    source_paths.add(path)
    ensure_entry(
        registry,
        canonical_id="l.series.charter",
        family="l_series_charter",
        family_group="l_series",
        title="L-Series Lexicographic Augmented Lagrangian Controller",
        aliases=["L-Series", "Lagrangian Series"],
        lookup_aliases=["l_series", "L-series"],
        objective="replace static weighted-loss blending with lexicographic augmented Lagrangian control over Tier A/B/C constraints",
        baseline_relation="series charter",
        script_paths=["scripts/train_l_series_mvs.py"],
        dag_paths=["airflow/dags/lojban_l_series_dag.py"],
        notes=[
            "Documented in docs/L_SERIES.md as the main lexicographic constraint-control phase.",
            "Serves as the umbrella lineage for L6 and the early M3+/M4/M5 branch experiments rooted in runs/l_series.",
        ],
    )


def _collect_l_series_artifacts(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    root = Path("runs/l_series")
    if not root.exists():
        return

    for path in _iter_json_paths(root, "l6_ablation_manifest.json"):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        source_paths.add(_path_str(path))
        timestamp = str(payload.get("timestamp") or "")
        for row in payload.get("rows", []):
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            metrics = {
                "constraint_arity_strict": safe_float(row.get("final_constraint_arity_strict")),
                "constraint_scope": safe_float(row.get("final_constraint_scope")),
                "constraint_identity": safe_float(row.get("final_constraint_identity")),
                "tier_b_enabled": 1.0 if bool(row.get("tier_b_enabled")) else 0.0,
                "tier_c_enabled": 1.0 if bool(row.get("tier_c_enabled")) else 0.0,
            }
            aliases = [run_id]
            if row.get("m_series_alias"):
                aliases.append(str(row.get("m_series_alias")))
            entry = ensure_entry(
                registry,
                canonical_id=f"l.series.l6.{slugify(run_id)}",
                family="l_series_l6",
                family_group="l_series",
                title=str(row.get("name") or run_id),
                aliases=aliases,
                lookup_aliases=[f"l6:{run_id}", *([str(row.get("m_series_alias"))] if row.get("m_series_alias") else [])],
                objective="L6 branch ablation over scope drills and Tier B/C forcing",
                baseline_relation="l6 branch cell",
                script_paths=["scripts/run_l6_ablation_branch.py", "scripts/train_l_series_mvs.py"],
                artifact_roots=[_path_str(path.parent)],
            )
            add_evidence(
                entry,
                build_evidence_record(
                    source_label=f"artifact:l6:{path.parent.name}",
                    source_paths=[_path_str(path), _path_str(Path(str(row.get("run_dir") or ""))) if row.get("run_dir") else _path_str(path.parent)],
                    evidence_class="artifact",
                    confidence_level="high",
                    reproducibility_status="runnable",
                    metrics=metrics,
                    normalized_metrics=normalize_metric_surface(metrics),
                    reported_at=timestamp or None,
                    notes="L6 multi-front L-series ablation branch.",
                ),
            )

    summary_patterns = {
        "l_series_summary.json",
        "m5_padded_nary_summary.json",
        "m5_2_autoregressive_chain_summary.json",
        "m5_3_masked_pair_chain_summary.json",
    }
    for path in root.rglob("*.json"):
        if path.name not in summary_patterns:
            continue
        if "l6_ablation" in {part.lower() for part in path.parts}:
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        canonical = _infer_l_series_entry(path, payload)
        if canonical is None:
            continue
        source_paths.add(_path_str(path))
        entry = ensure_entry(
            registry,
            canonical_id=canonical["canonical_id"],
            family=canonical["family"],
            family_group="l_series",
            title=canonical["title"],
            aliases=canonical["aliases"],
            lookup_aliases=canonical["lookup_aliases"],
            objective=canonical["objective"],
            baseline_relation=canonical["baseline_relation"],
            script_paths=canonical["script_paths"],
            artifact_roots=[_path_str(path.parent)],
        )
        add_evidence(
            entry,
            build_evidence_record(
                source_label=f"artifact:l_series:{canonical['family']}",
                source_paths=[_path_str(path)],
                evidence_class="artifact",
                confidence_level="high",
                reproducibility_status="artifact_only" if canonical["family"] != "l_series_branch_modern" else "runnable",
                metrics=_extract_l_series_metrics(payload),
                normalized_metrics=normalize_metric_surface(_extract_l_series_metrics(payload)),
                reported_at=str(payload.get("timestamp") or "") or None,
                notes=canonical["notes"],
            ),
        )


def _collect_phase5_train_manifests(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    for path in _iter_json_paths(Path("src/runs/phase5_train_ablation"), "ablation_manifest.json"):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        source_paths.add(_path_str(path))
        ts = str(payload.get("timestamp") or "")
        for row in payload.get("variants", []):
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            canonical_id = f"phase5.train.{slugify(name)}"
            entry = ensure_entry(
                registry,
                canonical_id=canonical_id,
                family="phase5_train_ablation",
                family_group="phase5_train_ablation",
                title=name,
                aliases=[name],
                lookup_aliases=[f"phase5.train:{name}"],
                objective="phase-5 training objective weight ablation",
                baseline_relation="phase5 train ablation variant",
                script_paths=["scripts/run_phase5_train_ablation.py"],
                artifact_roots=[_path_str(path.parent)],
            )
            metrics = {
                "phase5_enabled": 1.0 if bool(row.get("phase5")) else 0.0,
                "return_code": safe_float(row.get("return_code")),
            }
            add_evidence(
                entry,
                build_evidence_record(
                    source_label=f"artifact:phase5_train_manifest:{path.parent.name}",
                    source_paths=[_path_str(path)],
                    evidence_class="artifact",
                    confidence_level="high",
                    reproducibility_status="runnable",
                    metrics=metrics,
                    normalized_metrics=normalize_metric_surface(metrics),
                    reported_at=ts or None,
                    notes="Phase-5 training ablation manifest entry.",
                ),
            )


def _collect_phase5_datapack(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    path = Path("GRANULAR_DATAPACK.json")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return
    source_paths.add(_path_str(path))
    results = payload.get("phase5_ablation_results", {})
    if not isinstance(results, dict):
        return
    for name, row in results.items():
        if not isinstance(row, dict):
            continue
        canonical_id = f"phase5.train.{slugify(name)}"
        metrics = {
            "mean_lifts.final_answer": safe_float(row.get("mean_lifts", {}).get("final_answer")),
            "mean_lifts.symbolic": safe_float(row.get("mean_lifts", {}).get("symbolic")),
            "gate_pass": 1.0 if bool(row.get("gate_pass")) else 0.0,
        }
        entry = ensure_entry(
            registry,
            canonical_id=canonical_id,
            family="phase5_train_ablation",
            family_group="phase5_train_ablation",
            title=name,
            aliases=[name],
            lookup_aliases=[f"phase5.train:{name}"],
            objective="phase-5 train ablation measured from granular datapack",
            baseline_relation="phase5 train ablation variant",
            script_paths=["scripts/run_phase5_train_ablation.py"],
            artifact_roots=[str(row.get("adapter", "")).replace("\\", "/")],
        )
        add_evidence(
            entry,
            build_evidence_record(
                source_label="artifact:granular_datapack",
                source_paths=[_path_str(path)],
                evidence_class="artifact",
                confidence_level="high",
                reproducibility_status="artifact_only",
                metrics=metrics,
                normalized_metrics=normalize_metric_surface(metrics),
                reported_at=str(row.get("timestamp") or "") or None,
                notes="Backfilled from phase5_ablation_results in GRANULAR_DATAPACK.json.",
            ),
        )


def _collect_phase5_objective_ablation(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    path = Path("src/runs/phase5_objective_ablation.json")
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return
    source_paths.add(_path_str(path))
    terms = payload.get("terms", {})
    base_ts = str(payload.get("timestamp") or "")
    for row in payload.get("variants", []):
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        canonical_id = f"phase5.objective.{slugify(name)}"
        metrics = {
            "total_regularizer": safe_float(row.get("total_regularizer")),
            "delta_vs_full": safe_float(row.get("delta_vs_full")),
            **{f"term.{k}": safe_float(v) for k, v in terms.items()},
        }
        entry = ensure_entry(
            registry,
            canonical_id=canonical_id,
            family="phase5_objective_ablation",
            family_group="phase5_objective_ablation",
            title=name,
            aliases=[name],
            lookup_aliases=[f"phase5.objective:{name}"],
            objective="differentiate the contribution of individual phase-5 regularizer terms",
            baseline_relation="objective-term ablation",
            script_paths=["scripts/run_phase5_objective_ablation.py"],
            artifact_roots=[_path_str(path.parent)],
        )
        add_evidence(
            entry,
            build_evidence_record(
                source_label="artifact:phase5_objective_ablation",
                source_paths=[_path_str(path)],
                evidence_class="artifact",
                confidence_level="high",
                reproducibility_status="runnable",
                metrics=metrics,
                normalized_metrics=normalize_metric_surface(metrics),
                reported_at=base_ts or None,
                notes="Synthetic objective-term ablation surface.",
            ),
        )


def _collect_a_to_g_artifacts(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    for path in _iter_json_paths(Path("runs/ablation/a_to_g"), "ablation_matrix.json"):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        source_paths.add(_path_str(path))
        timestamp = str(payload.get("timestamp") or "")
        for row in payload.get("runs", []):
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "").strip()
            if not run_id:
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            normalized = _normalize_a_to_g_row(run_id, metrics)
            canonical_id = f"a_to_g.{slugify(run_id)}"
            entry = ensure_entry(
                registry,
                canonical_id=canonical_id,
                family="a_to_g_matrix",
                family_group="a_to_g_matrix",
                title=str(row.get("name") or run_id),
                aliases=[run_id, f"A-G/{run_id}"],
                lookup_aliases=[f"a_to_g:{run_id}"],
                objective="artifact-backed rerun of the A-G coconut ablation matrix",
                baseline_relation="a-to-g matrix cell",
                script_paths=["scripts/run_coconut_ablation_matrix.py"],
                dag_paths=["airflow/dags/lojban_ablation_matrix_dag.py"],
                artifact_roots=[_path_str(path.parent)],
            )
            add_evidence(
                entry,
                build_evidence_record(
                    source_label=f"artifact:a_to_g:{path.parent.name}",
                    source_paths=[_path_str(path), _path_str(path.parent)],
                    evidence_class="artifact",
                    confidence_level="high",
                    reproducibility_status="artifact_only" if run_id in {"A", "B.1", "B.2", "C", "D"} else "doc_only",
                    metrics=metrics,
                    normalized_metrics=normalized,
                    reported_at=timestamp or None,
                    notes=str(row.get("notes") or ""),
                ),
            )


def _collect_h_series_artifacts(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    search_roots = [Path("runs/h_series"), Path("runs/true_coconut_h_series")]
    for root in search_roots:
        for path in _iter_json_paths(root, "run_h_series.json"):
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            source_paths.add(_path_str(path))
            timestamp = str(payload.get("timestamp") or "")
            for row in payload.get("runs", []):
                if not isinstance(row, dict):
                    continue
                run_id = str(row.get("run_id") or "").strip()
                if not run_id:
                    continue
                metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
                canonical_id = f"h.series.{slugify(run_id)}"
                entry = ensure_entry(
                    registry,
                    canonical_id=canonical_id,
                    family="h_series_artifact",
                    family_group="h_series",
                    title=str(row.get("name") or run_id),
                    aliases=[run_id],
                    lookup_aliases=[f"h:{run_id}"],
                    objective="artifact-backed H/H5 extension ablation or evaluation",
                    baseline_relation="h-series family row",
                    script_paths=_command_scripts(row.get("command")),
                    artifact_roots=[_path_str(path.parent)],
                )
                add_evidence(
                    entry,
                    build_evidence_record(
                        source_label=f"artifact:h_series:{path.parent.name}",
                        source_paths=[_path_str(path), *[_path_str(Path(out)) for out in row.get("output_files", []) if isinstance(out, str)]],
                        evidence_class="artifact",
                        confidence_level="high",
                        reproducibility_status="artifact_only",
                        metrics=metrics,
                        normalized_metrics=normalize_metric_surface(metrics),
                        reported_at=timestamp or None,
                        notes=str(row.get("notes") or ""),
                    ),
                )


def _collect_j_series_artifacts(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    for path in _iter_json_paths(Path("runs/j_series"), "run_h_series.json"):
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        source_paths.add(_path_str(path))
        timestamp = str(payload.get("timestamp") or "")
        for row in payload.get("runs", []):
            if not isinstance(row, dict):
                continue
            run_id = str(row.get("run_id") or "").strip()
            if not run_id or not run_id.upper().startswith("J-"):
                continue
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            aliases = [run_id]
            if row.get("m_series_alias"):
                aliases.append(str(row.get("m_series_alias")))
            canonical_id = f"j.series.{slugify(run_id)}"
            entry = ensure_entry(
                registry,
                canonical_id=canonical_id,
                family="j_series",
                family_group="j_series",
                title=str(row.get("name") or run_id),
                aliases=aliases,
                lookup_aliases=[f"j:{run_id}"],
                objective="J-series invariance, curriculum, or adversarial synthesis ablation/eval",
                baseline_relation="j-series row",
                script_paths=_command_scripts(row.get("command")),
                artifact_roots=[_path_str(path.parent)],
            )
            add_evidence(
                entry,
                build_evidence_record(
                    source_label=f"artifact:j_series:{path.parent.name}",
                    source_paths=[_path_str(path), *[_path_str(Path(out)) for out in row.get("output_files", []) if isinstance(out, str)]],
                    evidence_class="artifact",
                    confidence_level="high",
                    reproducibility_status="runnable",
                    metrics=metrics,
                    normalized_metrics=normalize_metric_surface(metrics),
                    reported_at=timestamp or None,
                    notes=str(row.get("notes") or ""),
                ),
            )


def _extract_report_track(payload: dict[str, Any]) -> str:
    track = str(payload.get("track") or "").strip()
    if track:
        return track
    series = payload.get("series")
    if isinstance(series, dict):
        return str(series.get("track") or "").strip()
    return ""


def _report_timestamp(payload: dict[str, Any]) -> str:
    for key in ("timestamp", "generated_utc"):
        value = str(payload.get(key) or "").strip()
        if value:
            return value
    summary = payload.get("summary")
    if isinstance(summary, dict):
        return str(summary.get("timestamp") or "").strip()
    return ""


def _flatten_numeric_metrics(value: Any, prefix: str = "", depth: int = 0, max_depth: int = 4) -> dict[str, Any]:
    if depth > max_depth or not isinstance(value, dict):
        return {}
    metrics: dict[str, Any] = {}
    for key, child in value.items():
        key_str = str(key).strip()
        if not key_str:
            continue
        full_key = f"{prefix}.{key_str}" if prefix else key_str
        if isinstance(child, dict):
            metrics.update(_flatten_numeric_metrics(child, full_key, depth + 1, max_depth=max_depth))
            continue
        numeric = safe_float(child)
        if numeric is not None:
            metrics[full_key] = numeric
    return metrics


def _track_identity(track: str) -> tuple[str, str, str]:
    track = str(track or "").strip()
    mapping = {
        "M3+": ("l.branch.m3", "l_series_branch_telemetry", "l_series"),
        "M3.5": ("l.branch.m3_5", "l_series_branch_telemetry", "l_series"),
        "M3.6": ("l.branch.m3_6", "l_series_branch_telemetry", "l_series"),
        "M3.7": ("l.branch.m3_7", "l_series_branch_telemetry", "l_series"),
        "M3.8": ("l.branch.m3_8", "l_series_branch_telemetry", "l_series"),
        "M3.9": ("m.track.m3_9", "m_telemetry", "m_track"),
        "M3.10": ("m.track.m3_10", "m_telemetry", "m_track"),
        "M3.11": ("m.track.m3_11", "m_telemetry", "m_track"),
        "M3.12": ("m.track.m3_12", "m_telemetry", "m_track"),
        "M3.13": ("m.track.m3_13", "m_telemetry", "m_track"),
        "M3.14": ("m.track.m3_14", "m_telemetry", "m_track"),
        "M3.15": ("m.track.m3_15", "m_telemetry", "m_track"),
        "M3.15b": ("m.track.m3_15b", "m_telemetry", "m_track"),
        "M3.15c": ("m.track.m3_15c", "m_telemetry", "m_track"),
        "M3.15d": ("m.track.m3_15d", "m_telemetry", "m_track"),
        "M3.16": ("m.track.m3_16", "m_telemetry", "m_track"),
        "M3.17": ("m.track.m3_17", "m_telemetry", "m_track"),
        "M3.18": ("m.track.m3_18", "m_telemetry", "m_track"),
        "M3.19": ("m.track.m3_19", "m_telemetry", "m_track"),
        "M4": ("m.track.m4", "m_telemetry", "m_track"),
        "M4.0": ("m.track.m4_0", "m_telemetry", "m_track"),
        "M4.2": ("m.track.m4_2", "m_telemetry", "m_track"),
        "M5": ("m.track.m5_0", "m_telemetry", "m_track"),
        "M5.padded_nary_family": ("m.track.m5_1", "m_telemetry", "m_track"),
        "M5.2.autoregressive_chain": ("m.track.m5_2", "m_telemetry", "m_track"),
        "M5.2.autoregressive_chain.run": ("m.track.m5_2", "m_telemetry", "m_track"),
        "M5.3.masked_pair_chain": ("m.track.m5_3", "m_telemetry", "m_track"),
        "M5.3.masked_pair_chain.run": ("m.track.m5_3", "m_telemetry", "m_track"),
        "M11.discriminative": ("m.track.m11_discriminative", "m_telemetry", "m_track"),
        "M14": ("m.track.m14", "m_telemetry", "m_track"),
    }
    if track in mapping:
        return mapping[track]

    match = re.match(r"^M(\d+)\.([0-9]+[a-z]?)$", track)
    if match:
        return f"m.track.m{match.group(1)}_{match.group(2)}", "m_telemetry", "m_track"

    match = re.match(r"^M(\d+)$", track)
    if match:
        return f"m.track.m{match.group(1)}", "m_telemetry", "m_track"

    return f"m.track.{slugify(track)}", "m_telemetry", "m_track"


def _canonical_track_row_id(track: str, run_id: str) -> str:
    family_id, _, _ = _track_identity(track)
    row_slug = slugify(run_id)

    if family_id.startswith("l.branch.m3"):
        return f"{family_id}.{row_slug}"

    if family_id == "m.track.m5_0" and row_slug.startswith("m5_"):
        row_slug = row_slug[3:]
    elif family_id == "m.track.m5_1" and row_slug.startswith("m5_"):
        row_slug = row_slug[3:]

    return f"{family_id}.{row_slug}"


def _generic_report_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = _flatten_numeric_metrics(payload)
    if not metrics:
        return {}
    filtered: dict[str, Any] = {}
    for key, value in metrics.items():
        if any(blocked in key for blocked in ("details", "examples", "representative_examples", "inputs", "config", "args")):
            continue
        filtered[key] = value
    return filtered


def _collect_telemetry_reports(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    root = Path("artifacts/runs/telemetry/raw/ablation/hypercube")
    for path in root.rglob("*.json"):
        lowered = path.name.lower()
        if lowered not in {
            "m3_18_report.json",
            "m3_19_grid_report.json",
            "m14_report.json",
            "m11_discriminative_suite_manifest.json",
        } and not lowered.endswith("_report.json"):
            continue
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        track = _extract_report_track(payload)
        if not track:
            continue
        source_paths.add(_path_str(path))
        if track == "M11.discriminative":
            _register_family_level_manifest(
                registry,
                canonical_id="m.track.m11_discriminative",
                family="m_telemetry",
                family_group="m_track",
                title="M11 discriminative suite",
                aliases=["M11.discriminative"],
                lookup_aliases=["M11.discriminative"],
                objective="native M11 discriminative bridge suite",
                baseline_relation="m11 suite",
                metrics=payload.get("headline", {}),
                path=path,
                timestamp=str(payload.get("timestamp") or ""),
                script_paths=["scripts/run_m11_discriminative_suite.py"],
                dag_paths=["airflow/dags/lojban_m11_discriminative_suite_dag.py"],
            )
            continue

        family_id, family, family_group = _track_identity(track)

        if isinstance(payload.get("cells"), dict):
            for cell_name, cell_payload in payload.get("cells", {}).items():
                if not isinstance(cell_payload, dict):
                    continue
                if "_" in str(cell_name):
                    continue
                metrics = cell_payload.get("metrics") if isinstance(cell_payload.get("metrics"), dict) else {}
                if not metrics and isinstance(cell_payload.get("validation_metrics"), dict):
                    metrics = cell_payload.get("validation_metrics")
                canonical_id = f"{family_id}.{slugify(str(cell_name))}"
                title = str(cell_payload.get("label") or cell_payload.get("variant_spec", {}).get("label") or f"{track} {cell_name}")
                entry = ensure_entry(
                    registry,
                    canonical_id=canonical_id,
                    family=family,
                    family_group=family_group,
                    title=title,
                    aliases=[f"{track}.{cell_name}", str(cell_name)],
                    lookup_aliases=[f"{track}.{cell_name}"],
                    objective=f"telemetry-rooted {track} ablation cell",
                    baseline_relation=f"{track} cell",
                    artifact_roots=[_path_str(path.parent)],
                )
                add_evidence(
                    entry,
                    build_evidence_record(
                        source_label=f"artifact:telemetry:{track}",
                        source_paths=[_path_str(path)],
                        evidence_class="artifact",
                        confidence_level="high",
                        reproducibility_status="runnable" if track in {"M3.18", "M3.19", "M14"} else "artifact_only",
                        metrics=metrics,
                        normalized_metrics=normalize_metric_surface(metrics),
                        reported_at=str(payload.get("timestamp") or payload.get("generated_utc") or "") or None,
                        notes=str(cell_payload.get("variant_spec", {}).get("label") or ""),
                    ),
                )
            continue

        if isinstance(payload.get("cells"), list):
            for row in payload.get("cells", []):
                if not isinstance(row, dict):
                    continue
                run_id = str(row.get("run_id") or "").strip()
                if not run_id:
                    continue
                canonical_id = _canonical_track_row_id(track, run_id)
                entry = ensure_entry(
                    registry,
                    canonical_id=canonical_id,
                    family=family,
                    family_group=family_group,
                    title=str(row.get("name") or run_id),
                    aliases=[run_id],
                    lookup_aliases=[f"{track}:{run_id}"],
                    objective=f"telemetry-rooted {track} family row",
                    baseline_relation=f"{track} row",
                    artifact_roots=[_path_str(path.parent)],
                )
                add_evidence(
                    entry,
                    build_evidence_record(
                        source_label=f"artifact:telemetry:{track}",
                        source_paths=[_path_str(path)],
                        evidence_class="artifact",
                        confidence_level="high",
                        reproducibility_status="artifact_only",
                        metrics=row,
                        normalized_metrics=normalize_metric_surface(row),
                        reported_at=_report_timestamp(payload) or None,
                        notes=str(row.get("status") or ""),
                    ),
                )
            continue

        if isinstance(payload.get("rows"), list):
            for row in payload.get("rows", []):
                if not isinstance(row, dict):
                    continue
                run_id = str(row.get("run_id") or "").strip()
                if not run_id:
                    continue
                canonical_id = _canonical_track_row_id(track, run_id)
                entry = ensure_entry(
                    registry,
                    canonical_id=canonical_id,
                    family=family,
                    family_group=family_group,
                    title=str(row.get("name") or run_id),
                    aliases=[run_id],
                    lookup_aliases=[f"{track}:{run_id}"],
                    objective=f"telemetry-rooted {track} family row",
                    baseline_relation=f"{track} row",
                    artifact_roots=[_path_str(path.parent)],
                )
                source_list = [_path_str(path)]
                if row.get("summary_path"):
                    source_list.append(_path_str(Path(str(row.get("summary_path")))))
                add_evidence(
                    entry,
                    build_evidence_record(
                        source_label=f"artifact:telemetry:{track}",
                        source_paths=source_list,
                        evidence_class="artifact",
                        confidence_level="high",
                        reproducibility_status="artifact_only",
                        metrics=row,
                        normalized_metrics=normalize_metric_surface(row),
                        reported_at=str(payload.get("generated_utc") or "") or None,
                        notes=str(row.get("status") or ""),
                    ),
                )
            continue

        metrics = _generic_report_metrics(payload)
        if metrics:
            _register_family_level_manifest(
                registry,
                canonical_id=family_id,
                family=family,
                family_group=family_group,
                title=f"{track} telemetry report",
                aliases=[track],
                lookup_aliases=[track],
                objective=f"telemetry-rooted {track} family report",
                baseline_relation=f"{track} family report",
                metrics=metrics,
                path=path,
                timestamp=_report_timestamp(payload),
                script_paths=[str(payload.get("series", {}).get("script") or "")] if isinstance(payload.get("series"), dict) and payload.get("series", {}).get("script") else [],
                dag_paths=[],
            )


def _collect_archive_m_reports(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    for spec in ARCHIVE_M_REPORTS:
        path = Path(str(spec["path"]))
        payload = _load_json(path)
        if not isinstance(payload, dict):
            continue
        source_paths.add(_path_str(path))
        metrics = _generic_report_metrics(payload)
        entry = ensure_entry(
            registry,
            canonical_id=str(spec["canonical_id"]),
            family="m_archive_results",
            family_group="m_track",
            title=str(spec["title"]),
            aliases=list(spec.get("aliases", [])),
            lookup_aliases=list(spec.get("lookup_aliases", [])),
            objective=str(spec["objective"]),
            baseline_relation=str(spec["baseline_relation"]),
            script_paths=list(spec.get("script_paths", [])),
            artifact_roots=[_path_str(path.parent)],
        )
        add_evidence(
            entry,
            build_evidence_record(
                source_label=f"artifact:archive:{path.parent.name}",
                source_paths=[_path_str(path)],
                evidence_class="artifact",
                confidence_level="high",
                reproducibility_status="artifact_only",
                metrics=metrics,
                normalized_metrics=normalize_metric_surface(metrics),
                reported_at=_report_timestamp(payload) or None,
                notes=str(spec["title"]),
            ),
        )



def _collect_git_evidence(registry: dict[str, dict[str, Any]], source_paths: set[str]) -> None:
    cmd = [
        "git",
        "log",
        "--date=iso-strict",
        "--pretty=format:%H\t%ad\t%s",
        "--all",
        "--grep=ablation",
        "--grep=coconut",
        "--grep=H5",
        "--grep=M3.",
        "--grep=M14",
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except Exception:
        return
    if result.returncode != 0:
        return
    source_paths.add("git_log:ablation_backfill")
    commit_lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not commit_lines:
        return
    alias_map: dict[str, str] = {}
    for entry in registry.values():
        for alias in entry.get("aliases", []):
            alias_map[alias.lower()] = entry["canonical_id"]
        for alias in entry.get("lookup_aliases", []):
            alias_map[alias.lower()] = entry["canonical_id"]

    for line in commit_lines:
        parts = line.split("\t", 2)
        if len(parts) != 3:
            continue
        commit_hash, commit_date, subject = parts
        lowered = subject.lower()
        matched_id = None
        for alias, canonical_id in alias_map.items():
            if alias and alias.lower() in lowered:
                matched_id = canonical_id
                break
        if matched_id is None:
            track_match = re.search(r"(m\d+(?:\.\d+)?(?:[a-z])?)", lowered)
            if track_match:
                raw = track_match.group(1).upper()
                matched_id = f"git.only.{slugify(raw)}"
                ensure_entry(
                    registry,
                    canonical_id=matched_id,
                    family="git_reported_only",
                    family_group="git_only",
                    title=raw,
                    aliases=[raw],
                    lookup_aliases=[f"git:{raw}"],
                    objective="git-era ablation reference without confirmed local artifact",
                    baseline_relation="git-only reference",
                )
        if matched_id is None:
            continue
        entry = registry[matched_id]
        add_evidence(
            entry,
            build_evidence_record(
                source_label=f"git:{commit_hash[:8]}",
                source_paths=[f"git:{commit_hash}"],
                evidence_class="git_reported",
                confidence_level="low",
                reproducibility_status="git_only",
                metrics={},
                normalized_metrics={},
                reported_at=commit_date,
                notes=subject,
            ),
        )


def _infer_l_series_entry(path: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    normalized_parts = [part.lower() for part in path.parts]
    text_path = _path_str(path).lower()
    track = str(payload.get("series", {}).get("track") or payload.get("track") or "").strip()
    parent_name = path.parent.name

    if "/runs/l_series/" not in text_path.replace("\\", "/"):
        return None

    if re.fullmatch(r"\d{8}_\d{6}", path.parent.name) and path.name == "l_series_summary.json":
        return {
            "canonical_id": f"l.series.core.{slugify(path.parent.name)}",
            "family": "l_series_core_run",
            "title": f"L-series core run {path.parent.name}",
            "aliases": [path.parent.name],
            "lookup_aliases": [f"l.core:{path.parent.name}"],
            "objective": "standalone L-series augmented-Lagrangian training run",
            "baseline_relation": "l-series core run",
            "script_paths": ["scripts/train_l_series_mvs.py"],
            "notes": "Core L-series run saved directly under runs/l_series/<timestamp>/.",
        }

    if "m5_autoformalization" in normalized_parts:
        cell = _nearest_prefixed_parent(path, "m5_")
        return {
            "canonical_id": f"l.branch.m5_autoformalization.{slugify(cell or parent_name)}",
            "family": "l_series_branch_modern",
            "title": f"M5 autoformalization {cell or parent_name}",
            "aliases": [cell or parent_name, track or "M5.autoformalization"],
            "lookup_aliases": [f"l:m5_autoformalization:{cell or parent_name}"],
            "objective": "autoformalization branch rooted in the L-series controller stack",
            "baseline_relation": "l-series branch cell",
            "script_paths": ["scripts/train_l_series_mvs.py"],
            "notes": "Recovered from runs/l_series/m5_autoformalization summaries.",
        }

    if "m5_padded_nary" in normalized_parts:
        cell = _nearest_prefixed_parent(path, "m5_n")
        return {
            "canonical_id": f"l.branch.m5_padded_nary.{slugify(cell or parent_name)}",
            "family": "l_series_branch_modern",
            "title": f"M5 padded n-ary {cell or parent_name}",
            "aliases": [cell or parent_name, track or "M5.padded_nary"],
            "lookup_aliases": [f"l:m5_padded_nary:{cell or parent_name}"],
            "objective": "padded n-ary operator branch rooted in L-series constraints",
            "baseline_relation": "l-series branch cell",
            "script_paths": ["scripts/train_m5_padded_nary.py"],
            "notes": "Recovered from runs/l_series/m5_padded_nary summaries.",
        }

    if "m5_2_autoregressive_chain" in normalized_parts:
        cell = _nearest_prefixed_parent(path, "m5_2")
        return {
            "canonical_id": f"l.branch.m5_2_autoregressive_chain.{slugify(cell or parent_name)}",
            "family": "l_series_branch_modern",
            "title": f"M5.2 autoregressive chain {cell or parent_name}",
            "aliases": [cell or parent_name, track or "M5.2"],
            "lookup_aliases": [f"l:m5_2:{cell or parent_name}"],
            "objective": "autoregressive chain branch built on top of the L-series training stack",
            "baseline_relation": "l-series branch cell",
            "script_paths": ["scripts/train_m5_2_autoregressive_chain.py"],
            "notes": "Recovered from runs/l_series/m5_2_autoregressive_chain summaries.",
        }

    if "m5_3_masked_pair_chain" in normalized_parts:
        cell = _nearest_prefixed_parent(path, "m5_3")
        return {
            "canonical_id": f"l.branch.m5_3_masked_pair_chain.{slugify(cell or parent_name)}",
            "family": "l_series_branch_modern",
            "title": f"M5.3 masked pair chain {cell or parent_name}",
            "aliases": [cell or parent_name, track or "M5.3"],
            "lookup_aliases": [f"l:m5_3:{cell or parent_name}"],
            "objective": "masked pair chain branch built on top of the L-series training stack",
            "baseline_relation": "l-series branch cell",
            "script_paths": ["scripts/train_m5_3_masked_pair_chain.py"],
            "notes": "Recovered from runs/l_series/m5_3_masked_pair_chain summaries.",
        }

    if any(name in normalized_parts for name in {"m3_5_symmetry", "m3_7_shadow", "m3_8_diversification", "m3_plus", "m3_plus_nary", "m4"}):
        branch_root = next(
            (part for part in normalized_parts if part in {"m3_5_symmetry", "m3_7_shadow", "m3_8_diversification", "m3_plus", "m3_plus_nary", "m4"}),
            "l_branch",
        )
        cell = _nearest_branch_cell(path)
        label = track or branch_root.upper()
        return {
            "canonical_id": f"l.branch.{slugify(branch_root)}.{slugify(cell or parent_name)}",
            "family": "l_series_branch_modern",
            "title": f"{label} {cell or parent_name}",
            "aliases": [label, cell or parent_name],
            "lookup_aliases": [f"l:{branch_root}:{cell or parent_name}"],
            "objective": "early branch experiment rooted in runs/l_series before the later bridge/re-entry families diverged",
            "baseline_relation": "l-series branch cell",
            "script_paths": ["scripts/train_l_series_mvs.py"],
            "notes": "Recovered from runs/l_series branch summaries.",
        }
    return None


def _extract_l_series_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    final_step = payload.get("final_step", {}) if isinstance(payload.get("final_step"), dict) else {}
    eval_block = payload.get("eval", {}) if isinstance(payload.get("eval"), dict) else {}
    metrics = {
        "constraint_arity_strict": safe_float(final_step.get("constraint_arity_strict")),
        "constraint_scope": safe_float(final_step.get("constraint_scope")),
        "constraint_identity": safe_float(final_step.get("constraint_identity")),
        "constraint_scope_unbound": safe_float(final_step.get("constraint_scope_unbound")),
        "task_loss": safe_float(final_step.get("task_loss")),
        "total_loss": safe_float(final_step.get("total_loss")),
        "operator_entropy": safe_float(final_step.get("operator_entropy")) or safe_float(final_step.get("operator_entropy_batch")),
        "top1_op_share": safe_float(final_step.get("top1_op_share")) or safe_float(final_step.get("top1_op_share_batch")),
        "arity_crystallization_rate": safe_float(final_step.get("arity_crystallization_rate")),
        "winograd_val_accuracy": safe_float(eval_block.get("winograd_val", {}).get("accuracy")),
        "winograd_eval_accuracy": safe_float(eval_block.get("winograd_eval", {}).get("accuracy")),
        "semantic_val_accuracy": safe_float(eval_block.get("semantic_val", {}).get("accuracy")),
        "semantic_val_ce": safe_float(eval_block.get("semantic_val", {}).get("mean_ce")),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _nearest_branch_cell(path: Path) -> str | None:
    for part in reversed(path.parts):
        lowered = part.lower()
        if re.fullmatch(r"m\d+(_\d+)?_[a-z0-9]+", lowered) or re.fullmatch(r"m\d+_\d+", lowered):
            return part
    return None


def _nearest_prefixed_parent(path: Path, prefix: str) -> str | None:
    prefix = prefix.lower()
    for part in reversed(path.parts):
        if part.lower().startswith(prefix):
            return part
    return None


def _register_family_level_manifest(
    registry: dict[str, dict[str, Any]],
    *,
    canonical_id: str,
    family: str,
    family_group: str,
    title: str,
    aliases: list[str],
    lookup_aliases: list[str],
    objective: str,
    baseline_relation: str,
    metrics: dict[str, Any],
    path: Path,
    timestamp: str,
    script_paths: list[str],
    dag_paths: list[str],
) -> None:
    entry = ensure_entry(
        registry,
        canonical_id=canonical_id,
        family=family,
        family_group=family_group,
        title=title,
        aliases=aliases,
        lookup_aliases=lookup_aliases,
        objective=objective,
        baseline_relation=baseline_relation,
        script_paths=script_paths,
        dag_paths=dag_paths,
        artifact_roots=[_path_str(path.parent)],
    )
    add_evidence(
        entry,
        build_evidence_record(
            source_label=f"artifact:{canonical_id}",
            source_paths=[_path_str(path)],
            evidence_class="artifact",
            confidence_level="high",
            reproducibility_status="runnable",
            metrics=metrics,
            normalized_metrics=normalize_metric_surface(metrics),
            reported_at=timestamp or None,
            notes=title,
        ),
    )


def _normalize_a_to_g_row(run_id: str, metrics: dict[str, Any]) -> dict[str, float | None]:
    run_id = run_id.strip().upper()
    if run_id == "A":
        return {
            "held_out_accuracy": safe_float(metrics.get("control_base_final_acc")),
            "logical_accuracy": safe_float(metrics.get("control_base_symbolic_acc")),
        }
    if run_id in {"B.1", "B.2"}:
        return {
            "held_out_accuracy": safe_float(metrics.get("enhanced_t2t_final_acc")),
            "logical_accuracy": safe_float(metrics.get("enhanced_t2t_symbolic_acc")),
            "final_answer_lift": safe_float(metrics.get("mean_lifts", {}).get("adapter_final_answer")),
            "symbolic_lift": safe_float(metrics.get("mean_lifts", {}).get("adapter_symbolic")),
        }
    if run_id == "C":
        return {
            "held_out_accuracy": safe_float(metrics.get("coconut_handoff_final_acc")),
            "logical_accuracy": safe_float(metrics.get("coconut_handoff_symbolic_acc")),
            "final_answer_lift": safe_float(metrics.get("mean_lifts", {}).get("handoff_final_answer")),
            "symbolic_lift": safe_float(metrics.get("mean_lifts", {}).get("handoff_symbolic")),
        }
    if run_id == "D":
        return {
            "held_out_accuracy": safe_float(metrics.get("handoff_acc")),
            "final_answer_lift": safe_float(metrics.get("handoff_lift")),
        }
    return normalize_metric_surface(metrics)


def _command_scripts(command: Any) -> list[str]:
    if not isinstance(command, list):
        return []
    output: list[str] = []
    for part in command:
        if not isinstance(part, str):
            continue
        text = part.replace("\\", "/")
        if text.endswith(".py"):
            output.append(text)
    return sorted(set(output))


def _iter_json_paths(root: Path, file_name: str):
    if not root.exists():
        return
    try:
        matches = sorted(root.rglob(file_name))
    except OSError:
        return
    for path in matches:
        yield path


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _path_str(path: Path | str) -> str:
    return str(path).replace("\\", "/")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "canonical_id",
        "normalized_canonical_id",
        "series_major",
        "series_minor",
        "series_cell",
        "family",
        "family_group",
        "title",
        "aliases",
        "lookup_aliases",
        "evidence_class",
        "confidence_level",
        "reproducibility_status",
        "earliest_reported_at",
        "latest_reported_at",
        "held_out_accuracy",
        "logical_accuracy",
        "macro_f1",
        "final_ce_loss",
        "intervention_effect_on_gold",
        "resume_first_token_accuracy",
        "english_fluency_score",
        "contamination_rate",
        "loop_rate",
        "scratchpad_bleed_rate",
        "surgery_trigger_rate",
        "geometry_retention",
        "final_answer_lift",
        "symbolic_lift",
        "artifact_count",
        "doc_count",
        "git_count",
        "script_paths",
        "dag_paths",
        "artifact_roots",
        "baseline_relation",
        "question_boundary",
        "architectural_thesis",
        "inherits_from",
        "inherits_components",
        "changed_components",
        "dropped_components",
        "baseline_manifest",
        "archive_path",
        "active_doc_path",
        "derived_from",
        "supersedes",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _write_ledger_md(path: Path, manifest: dict[str, Any]) -> None:
    entries = manifest.get("entries", [])
    by_group: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        group = str(entry.get("family_group") or "misc")
        by_group.setdefault(group, []).append(entry)

    lines = [
        "# Ablation History Ledger",
        "",
        f"- schema_version: `{manifest.get('schema_version')}`",
        f"- generated_utc: `{manifest.get('generated_utc')}`",
        f"- run_id: `{manifest.get('run_id')}`",
        f"- total_entries: `{len(entries)}`",
        f"- artifact_entries: `{manifest.get('history_slices', {}).get('artifact_only', {}).get('entry_count')}`",
        f"- runnable_entries: `{manifest.get('history_slices', {}).get('runnable_only', {}).get('entry_count')}`",
        "",
        "## Group Counts",
        "",
    ]
    for group, count in _ordered_counts(manifest.get("family_group_counts", {})).items():
        lines.append(f"- `{group}`: `{count}`")
    lines.extend([
        "",
        "## Groups",
        "",
    ])
    for group in _ordered_groups(by_group):
        lines.append(f"### {group}")
        for entry in by_group[group]:
            metrics = entry.get("metrics_summary", {})
            lines.append(
                f"- `{entry.get('canonical_id')}` aliases=`{' | '.join(entry.get('aliases', []))}` "
                f"evidence=`{entry.get('evidence_class')}` confidence=`{entry.get('confidence_level')}` "
                f"repro=`{entry.get('reproducibility_status')}` "
                f"acc=`{_fmt(metrics.get('best_held_out_accuracy'))}` "
                f"logical=`{_fmt(metrics.get('best_logical_accuracy'))}` "
                f"ce=`{_fmt(metrics.get('lowest_final_ce_loss'))}`"
            )
        lines.append("")
    if manifest.get("historical_gaps"):
        lines.extend(["## Historical Gaps", ""])
        for gap in manifest["historical_gaps"]:
            lines.append(
                f"- `{gap.get('gap_id')}` kind=`{gap.get('kind')}` path=`{gap.get('path')}` source=`{gap.get('reported_in')}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_letter_series_manifests(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for spec in LETTER_SERIES_SPECS:
        matched = [entry for entry in entries if _entry_matches_letter_series(entry, spec)]
        normalized_ids = sorted({str(entry.get("normalized_canonical_id") or "") for entry in matched if str(entry.get("normalized_canonical_id") or "").strip()})
        canonical_ids = sorted({str(entry.get("canonical_id") or "") for entry in matched if str(entry.get("canonical_id") or "").strip()})
        aliases = sorted({alias for entry in matched for alias in entry.get("aliases", []) if str(alias).strip()})
        doc_paths = [path for path in spec.get("doc_paths", []) if Path(path).exists()]
        manifests.append(
            {
                "series_key": spec["series_key"],
                "title": spec["title"],
                "objective": spec["objective"],
                "entry_count": len(matched),
                "runnable_count": sum(1 for entry in matched if str(entry.get("reproducibility_status") or "") == "runnable"),
                "artifact_only_count": sum(1 for entry in matched if str(entry.get("reproducibility_status") or "") == "artifact_only"),
                "doc_only_count": sum(1 for entry in matched if str(entry.get("reproducibility_status") or "") == "doc_only"),
                "family_groups": list(spec.get("family_groups", [])),
                "normalized_ids": normalized_ids,
                "canonical_ids": canonical_ids,
                "legacy_aliases": aliases,
                "doc_paths": doc_paths,
                "script_paths": list(spec.get("script_paths", [])),
                "dag_paths": list(spec.get("dag_paths", [])),
                "artifact_roots": sorted({root for entry in matched for root in entry.get("artifact_roots", []) if str(root).strip()}),
            }
        )
    return manifests


def _entry_matches_letter_series(entry: dict[str, Any], spec: dict[str, Any]) -> bool:
    canonical_id = str(entry.get("canonical_id") or "")
    family_group = str(entry.get("family_group") or "")

    if family_group in set(spec.get("family_groups", [])):
        if any(canonical_id.startswith(prefix) for prefix in spec.get("exclude_prefixes", [])):
            return False
        return True

    if canonical_id in set(spec.get("canonical_ids", [])):
        return True

    for prefix in spec.get("family_prefixes", []):
        if canonical_id.startswith(prefix):
            return True
    return False


def _write_letter_series_md(path: Path, families: list[dict[str, Any]]) -> None:
    lines = [
        "# Letter Series Families",
        "",
        "Formal family-level registry for the pre-M letter-era experiment program and its orchestration surfaces.",
        "",
    ]
    for row in families:
        lines.append(f"## {row['series_key']}")
        lines.append("")
        lines.append(f"- Title: `{row['title']}`")
        lines.append(f"- Objective: {row['objective']}")
        lines.append(f"- Entry count: `{row['entry_count']}`")
        lines.append(f"- Runnable rows: `{row['runnable_count']}`")
        lines.append(f"- Artifact-only rows: `{row['artifact_only_count']}`")
        lines.append(f"- Doc-only rows: `{row['doc_only_count']}`")
        if row.get("family_groups"):
            lines.append(f"- Family groups: `{', '.join(row['family_groups'])}`")
        if row.get("normalized_ids"):
            lines.append(f"- Normalized IDs: `{', '.join(row['normalized_ids'])}`")
        if row.get("legacy_aliases"):
            lines.append(f"- Legacy aliases: `{', '.join(row['legacy_aliases'][:40])}`")
        if row.get("doc_paths"):
            lines.append(f"- Docs: `{', '.join(row['doc_paths'])}`")
        if row.get("script_paths"):
            lines.append(f"- Scripts: `{', '.join(row['script_paths'])}`")
        if row.get("dag_paths"):
            lines.append(f"- DAGs: `{', '.join(row['dag_paths'])}`")
        if row.get("artifact_roots"):
            lines.append(f"- Artifact roots: `{', '.join(row['artifact_roots'][:12])}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: Any) -> str:
    number = safe_float(value)
    if number is None:
        return "n/a"
    return f"{number:.6f}"


def _count_entries(entries: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(entry.get(key) or "")
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0]))


def _ordered_groups(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(
        grouped,
        key=lambda group: (GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER), group),
    )


def _ordered_counts(counts: dict[str, int]) -> dict[str, int]:
    return {
        key: counts[key]
        for key in _ordered_groups({group: [] for group in counts})
    }


if __name__ == "__main__":
    main()
