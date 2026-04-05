from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.utils.task_group import TaskGroup


REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_PATH = REPO_ROOT / "configs" / "experiment_taxonomy.json"
HISTORY_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_history_backfill"


LEGACY_STAGES: list[dict[str, Any]] = [
    {
        "task_id": "a_to_g_matrix",
        "stage_key": "A-G",
        "title": "A-G Matrix",
        "summary": "Early core matrix covering control, projected handoff, coconut variants, and English-vs-Lojban duel baselines.",
        "status_note": "historical family with partial artifact recovery",
        "child_dags": ["lojban_ablation_a_to_g"],
    },
    {
        "task_id": "h_bridge_series",
        "stage_key": "H",
        "title": "H Bridge Series",
        "summary": "Linear and SwiGLU bridge experiments that tested whether geometric handoff could perturb the host decoder usefully.",
        "status_note": "archival checkpoint; not currently a modern runnable family",
        "child_dags": [],
    },
    {
        "task_id": "h5_bridge_extension",
        "stage_key": "H5",
        "title": "H5 Bridge Extension",
        "summary": "Boolean-surgery, persistent-advisor, and extension work that seeded later J and L families.",
        "status_note": "archival checkpoint with artifact-backed reports",
        "child_dags": [],
    },
    {
        "task_id": "j_series_m1_origin",
        "stage_key": "J",
        "title": "J Series / M1 Origin",
        "summary": "Advisor-side invariance and adversarial synthesis program; normalized forward as the M1 family.",
        "status_note": "runnable legacy family",
        "child_dags": ["lojban_j_series_invariance"],
    },
    {
        "task_id": "l_series_m2_origin",
        "stage_key": "L",
        "title": "L Series / M2 Origin",
        "summary": "Lagrangian constrained-manifold program; normalized forward as the M2 family and the early M3 branch line.",
        "status_note": "partially runnable legacy family",
        "child_dags": ["lojban_l_series_train"],
    },
    {
        "task_id": "jl_hypercube_aggregation",
        "stage_key": "J/L Hypercube",
        "title": "J/L Hypercube",
        "summary": "Cross-family aggregation layer that consolidated J/L-era reports before the modern M suite.",
        "status_note": "orchestration surface only",
        "child_dags": ["lojban_ablation_hypercube_report"],
    },
    {
        "task_id": "phase5_eval_surface",
        "stage_key": "Phase Eval",
        "title": "Phase-5 Eval Families",
        "summary": "Train/objective ablation surface for semantic and compression losses before later M-series serialization work.",
        "status_note": "history-backed phase family; script surfaces exist even where family DAG coverage is incomplete",
        "child_dags": [],
    },
]


M_STAGE_SPECS: list[dict[str, Any]] = [
    {
        "task_id": "m1_invariance",
        "stage_key": "M1",
        "title": "M1",
        "legacy_origin": "J",
        "child_dags": ["lojban_j_series_invariance"],
    },
    {
        "task_id": "m2_lagrangian",
        "stage_key": "M2",
        "title": "M2",
        "legacy_origin": "L",
        "child_dags": ["lojban_l_series_train"],
    },
    {
        "task_id": "m3_bridge_core",
        "stage_key": "M3",
        "title": "M3",
        "child_dags": ["lojban_m3_plus_family"],
    },
    {
        "task_id": "m3_5_symmetry",
        "stage_key": "M3.5",
        "title": "M3.5",
        "child_dags": ["lojban_m3_5_symmetry_family"],
    },
    {
        "task_id": "m3_6_symmetry_oracle",
        "stage_key": "M3.6",
        "title": "M3.6",
        "child_dags": ["lojban_m3_6_symmetry_oracle_suite"],
    },
    {
        "task_id": "m3_7_shadow_alignment",
        "stage_key": "M3.7",
        "title": "M3.7",
        "child_dags": ["lojban_m3_7_shadow_alignment"],
    },
    {
        "task_id": "m3_8_operator_diversification",
        "stage_key": "M3.8",
        "title": "M3.8",
        "child_dags": ["lojban_m3_8_operator_diversification"],
    },
    {
        "task_id": "m3_9_primitive_probe",
        "stage_key": "M3.9",
        "title": "M3.9",
        "child_dags": ["lojban_m3_9_primitive_probe"],
    },
    {
        "task_id": "m3_10_ood_accuracy_probe",
        "stage_key": "M3.10",
        "title": "M3.10",
        "child_dags": ["lojban_m3_10_ood_accuracy_probe"],
    },
    {
        "task_id": "m3_11_failure_anatomy",
        "stage_key": "M3.11",
        "title": "M3.11",
        "child_dags": ["lojban_m3_11_winograd_failure_anatomy"],
    },
    {
        "task_id": "m3_12_return_stream",
        "stage_key": "M3.12",
        "title": "M3.12",
        "child_dags": ["lojban_m3_12_geometric_return_stream"],
    },
    {
        "task_id": "m3_13_relational_grid",
        "stage_key": "M3.13",
        "title": "M3.13",
        "child_dags": [],
    },
    {
        "task_id": "m3_14_structural_alignment",
        "stage_key": "M3.14",
        "title": "M3.14",
        "child_dags": ["lojban_m3_14_structural_alignment_bridge"],
    },
    {
        "task_id": "m3_15_rotary_coconut",
        "stage_key": "M3.15",
        "title": "M3.15",
        "child_dags": ["lojban_m3_15_rotary_coconut", "lojban_m3_15_rotary_coconut_seven"],
    },
    {
        "task_id": "m3_15b_local_rotary",
        "stage_key": "M3.15b",
        "title": "M3.15b",
        "child_dags": ["lojban_m3_15b_relation_local_rotary"],
    },
    {
        "task_id": "m3_15c_family_conditioned",
        "stage_key": "M3.15c",
        "title": "M3.15c",
        "child_dags": ["lojban_m3_15c_family_conditioned_bridge"],
    },
    {
        "task_id": "m3_15d_answer_path_forcing",
        "stage_key": "M3.15d",
        "title": "M3.15d",
        "child_dags": ["lojban_m3_15d_answer_path_forcing"],
    },
    {
        "task_id": "m3_16_continuous_graph_bias",
        "stage_key": "M3.16",
        "title": "M3.16",
        "child_dags": ["lojban_m3_16_continuous_graph_bias"],
    },
    {
        "task_id": "m3_17_advisor_reentry",
        "stage_key": "M3.17",
        "title": "M3.17",
        "child_dags": ["lojban_m3_17_advisor_reentry_bridge"],
    },
    {
        "task_id": "m3_18_decoder_reentry",
        "stage_key": "M3.18",
        "title": "M3.18",
        "child_dags": ["lojban_m3_18_decoder_reentry_resume"],
    },
    {
        "task_id": "m3_19_d_mainline_grid",
        "stage_key": "M3.19",
        "title": "M3.19",
        "child_dags": ["lojban_m3_19_d_mainline_grid"],
    },
    {
        "task_id": "m4_semantic_grounding",
        "stage_key": "M4",
        "title": "M4",
        "child_dags": ["lojban_m4_series"],
    },
    {
        "task_id": "m4_0_semantic_probe",
        "stage_key": "M4.0",
        "title": "M4.0",
        "child_dags": ["lojban_m4_0_semantic_probe"],
    },
    {
        "task_id": "m4_2_predicate_grounding",
        "stage_key": "M4.2",
        "title": "M4.2",
        "child_dags": ["lojban_m4_2_predicate_grounding"],
    },
    {
        "task_id": "m5_text_autoformalization",
        "stage_key": "M5",
        "title": "M5",
        "child_dags": ["lojban_m5_autoformalization"],
    },
    {
        "task_id": "m5_1_padded_nary",
        "stage_key": "M5.1",
        "title": "M5.1",
        "child_dags": ["lojban_m5_padded_nary"],
    },
    {
        "task_id": "m5_2_autoregressive_chain",
        "stage_key": "M5.2",
        "title": "M5.2",
        "child_dags": ["lojban_m5_2_autoregressive_chain"],
    },
    {
        "task_id": "m5_3_masked_pair_chain",
        "stage_key": "M5.3",
        "title": "M5.3",
        "child_dags": ["lojban_m5_3_masked_pair_chain"],
    },
    {
        "task_id": "m6_logic_engine_bridge",
        "stage_key": "M6",
        "title": "M6",
        "child_dags": [],
    },
    {
        "task_id": "m7_interleaved_coprocessor",
        "stage_key": "M7",
        "title": "M7",
        "child_dags": [],
    },
    {
        "task_id": "m8_council_of_oracles",
        "stage_key": "M8",
        "title": "M8",
        "child_dags": [],
    },
    {
        "task_id": "m9_provenance_manifold",
        "stage_key": "M9",
        "title": "M9",
        "child_dags": [],
    },
    {
        "task_id": "m10_return_path_adaptation",
        "stage_key": "M10",
        "title": "M10",
        "child_dags": [],
    },
    {
        "task_id": "m11_native_discriminative",
        "stage_key": "M11",
        "title": "M11",
        "child_dags": ["lojban_m11_discriminative_suite"],
    },
    {
        "task_id": "m14_symbiote_scratchpad",
        "stage_key": "M14",
        "title": "M14",
        "child_dags": ["lojban_m14_symbiote_scratchpad"],
    },
]


def _load_taxonomy() -> dict[str, Any]:
    if not TAXONOMY_PATH.exists():
        return {}
    return json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))


def _latest_history_manifest() -> Path | None:
    if not HISTORY_ROOT.exists():
        return None
    candidates = sorted(HISTORY_ROOT.glob("*/ablation_history_manifest.json"), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def _load_history_index() -> dict[str, Any]:
    history_manifest = _latest_history_manifest()
    if history_manifest is None:
        return {"legacy": {}, "major": {}, "path": None}
    payload = json.loads(history_manifest.read_text(encoding="utf-8"))
    legacy = {row.get("series_key"): row for row in payload.get("series_family_manifests", [])}
    major: dict[str, dict[str, int]] = {}
    for entry in payload.get("entries", []):
        major_num = entry.get("series_major")
        if major_num is None:
            continue
        stage_key = f"M{int(major_num)}"
        bucket = major.setdefault(stage_key, {"entry_count": 0, "runnable_count": 0, "artifact_only_count": 0, "doc_only_count": 0})
        bucket["entry_count"] += 1
        status = str(entry.get("reproducibility_status") or "").strip()
        if status == "runnable":
            bucket["runnable_count"] += 1
        elif status == "artifact_only":
            bucket["artifact_only_count"] += 1
        elif status == "doc_only":
            bucket["doc_only_count"] += 1
    return {"legacy": legacy, "major": major, "path": history_manifest}


TAXONOMY = _load_taxonomy()
HISTORY_INDEX = _load_history_index()
MAJOR_FAMILIES = TAXONOMY.get("major_families", {})


def _legacy_doc(stage: dict[str, Any]) -> str:
    family = HISTORY_INDEX["legacy"].get(stage["stage_key"], {})
    return "\n".join(
        [
            f"### {stage['title']}",
            "",
            stage["summary"],
            "",
            f"- Stage key: `{stage['stage_key']}`",
            f"- Status: {stage['status_note']}",
            f"- Entry count: `{int(family.get('entry_count') or 0)}`",
            f"- Runnable rows: `{int(family.get('runnable_count') or 0)}`",
            f"- Artifact-only rows: `{int(family.get('artifact_only_count') or 0)}`",
            f"- Doc-only rows: `{int(family.get('doc_only_count') or 0)}`",
            f"- Child DAGs: `{', '.join(stage['child_dags']) if stage['child_dags'] else 'none'}`",
            f"- Normalized IDs: `{', '.join(family.get('normalized_ids', [])[:24]) or 'n/a'}`",
            f"- Legacy aliases: `{', '.join(family.get('legacy_aliases', [])[:24]) or 'n/a'}`",
        ]
    )


def _major_doc(stage: dict[str, Any]) -> str:
    taxonomy_row = MAJOR_FAMILIES.get(stage["stage_key"], {})
    counts = HISTORY_INDEX["major"].get(stage["stage_key"], {})
    lines = [
        f"### {stage['title']}",
        "",
        taxonomy_row.get("architectural_thesis", "Normalized M-series stage."),
        "",
        f"- Stage key: `{stage['stage_key']}`",
    ]
    if stage.get("legacy_origin"):
        lines.append(f"- Legacy origin: `{stage['legacy_origin']}`")
    lines.extend(
        [
            f"- Entry count: `{int(counts.get('entry_count') or 0)}`",
            f"- Runnable rows: `{int(counts.get('runnable_count') or 0)}`",
            f"- Artifact-only rows: `{int(counts.get('artifact_only_count') or 0)}`",
            f"- Doc-only rows: `{int(counts.get('doc_only_count') or 0)}`",
            f"- Question boundary: {taxonomy_row.get('question_boundary', 'n/a')}",
            f"- Allowed axes: `{', '.join(taxonomy_row.get('allowed_ablation_axes', [])) or 'n/a'}`",
            f"- Forbidden drift: `{', '.join(taxonomy_row.get('forbidden_drift_axes', [])) or 'n/a'}`",
            f"- Promotion basis: `{', '.join(taxonomy_row.get('promotion_basis', [])) or 'n/a'}`",
            f"- Child DAGs: `{', '.join(stage['child_dags']) if stage['child_dags'] else 'archival-only or family-specific scripts only'}`",
        ]
    )
    return "\n".join(lines)


MASTER_DOC = "\n".join(
    [
        "# Ablation Master Spine",
        "",
        "This is the canonical Airflow graph for the entire research program.",
        "",
        "It is intentionally split into three layers:",
        "",
        "1. legacy letter-series foundation",
        "2. normalized M-series progression",
        "3. control-plane refresh",
        "",
        "Historical-only families are kept as explicit archival checkpoints so the graph preserves the intellectual progression even when a family is not yet fully rerunnable.",
        "",
        f"- Latest history manifest: `{HISTORY_INDEX['path']}`" if HISTORY_INDEX["path"] else "- Latest history manifest: `unavailable`",
    ]
)


with DAG(
    dag_id="lojban_ablation_master_spine",
    description="Canonical master DAG showing the full ablation program from letter-era foundations through the normalized M-series and control-plane refresh.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "master", "spine", "history", "m-series"],
    params={
        "run_id": Param("", type="string"),
    },
) as dag:
    dag.doc_md = MASTER_DOC

    start = EmptyOperator(task_id="start")

    with TaskGroup(group_id="legacy_letter_series", tooltip="Letter-era foundations and orchestration") as legacy_letter_series:
        previous = None
        for stage in LEGACY_STAGES:
            task = EmptyOperator(
                task_id=stage["task_id"],
                doc_md=_legacy_doc(stage),
            )
            if previous is not None:
                previous >> task
            previous = task

    with TaskGroup(group_id="m_series_progression", tooltip="Normalized M-series progression") as m_series_progression:
        previous = None
        for stage in M_STAGE_SPECS:
            task = EmptyOperator(
                task_id=stage["task_id"],
                doc_md=_major_doc(stage),
            )
            if previous is not None:
                previous >> task
            previous = task

    with TaskGroup(group_id="control_plane_refresh", tooltip="Canonical ledger and suite refresh") as control_plane_refresh:
        history_registry = EmptyOperator(
            task_id="history_registry",
            doc_md="\n".join(
                [
                    "### History Registry",
                    "",
                    "Canonical history backfill and family normalization surface.",
                    "",
                    "- Child DAG: `lojban_ablation_history_backfill`",
                    "- Output: refreshed ablation history manifest and family ledgers",
                ]
            ),
        )

        family_map_and_spine = EmptyOperator(
            task_id="family_map_and_spine",
            doc_md="\n".join(
                [
                    "### Family Map And Spine",
                    "",
                    "Concentrated family map plus ordered program spine. This is the readable control-plane layer a senior engineer should open first.",
                    "",
                    "- Child DAG: `lojban_ablation_program_spine`",
                    "- Outputs: `docs/ABLATION_PROGRAM_MAP.md`, `docs/ABLATION_PROGRAM_SPINE.md`",
                ]
            ),
        )

        refresh_program_control_plane = TriggerDagRunOperator(
            task_id="refresh_program_control_plane",
            trigger_dag_id="lojban_ablation_program_spine",
            wait_for_completion=False,
            conf={"run_id": "{{ dag_run.run_id }}__program"},
        )

        history_registry >> family_map_and_spine >> refresh_program_control_plane

    end = EmptyOperator(task_id="end")

    start >> legacy_letter_series >> m_series_progression >> control_plane_refresh >> end
