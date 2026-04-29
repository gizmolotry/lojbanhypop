from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M19_FAMILY_VERSION = "1.0"
M19_HIDDEN_SIZE = 896
M19_SCRATCHPAD_TOKEN = "<symbiote>"
M19_SYMBIOTE_END_TOKEN = "<symbiote_end>"


def m19_cell_id(num_queries: int, bottleneck_dim: int, scratchpad_length: int) -> str:
    return f"M19.3_{int(num_queries)}Q_{int(bottleneck_dim)}D_{int(scratchpad_length)}S"


def m19_cell_label(num_queries: int, bottleneck_dim: int, scratchpad_length: int) -> str:
    return f"{int(num_queries)}Q / {int(bottleneck_dim)}D / {int(scratchpad_length)}S"


M19_ISOLATION_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m19_cell_id(8, 64, 8),
        "label": m19_cell_label(8, 64, 8),
        "num_queries": 8,
        "bottleneck_dim": 64,
        "scratchpad_length": 8,
        "role": "baseline",
        "thesis": "rerun the concise baseline in the exact same harness before reading any width or query claims.",
    },
    {
        "cell_key": "B",
        "cell_id": m19_cell_id(8, 128, 8),
        "label": m19_cell_label(8, 128, 8),
        "num_queries": 8,
        "bottleneck_dim": 128,
        "scratchpad_length": 8,
        "role": "width_probe",
        "thesis": "isolate bottleneck width lift while keeping concise runway.",
    },
    {
        "cell_key": "C",
        "cell_id": m19_cell_id(16, 64, 8),
        "label": m19_cell_label(16, 64, 8),
        "num_queries": 16,
        "bottleneck_dim": 64,
        "scratchpad_length": 8,
        "role": "query_probe",
        "thesis": "isolate query-count expansion under the concise runway baseline.",
    },
    {
        "cell_key": "D",
        "cell_id": m19_cell_id(8, 64, 12),
        "label": m19_cell_label(8, 64, 12),
        "num_queries": 8,
        "bottleneck_dim": 64,
        "scratchpad_length": 12,
        "role": "runway_probe",
        "thesis": "isolate longer runway under the narrow baseline bottleneck.",
    },
    {
        "cell_key": "E",
        "cell_id": m19_cell_id(8, 128, 12),
        "label": m19_cell_label(8, 128, 12),
        "num_queries": 8,
        "bottleneck_dim": 128,
        "scratchpad_length": 12,
        "role": "width_runway_interaction",
        "thesis": "interaction test for wider bottleneck plus longer runway.",
    },
    {
        "cell_key": "F",
        "cell_id": m19_cell_id(16, 64, 12),
        "label": m19_cell_label(16, 64, 12),
        "num_queries": 16,
        "bottleneck_dim": 64,
        "scratchpad_length": 12,
        "role": "query_runway_interaction",
        "thesis": "interaction test for more queries plus longer runway.",
    },
]


def _clone_track(base_key: str, **updates: Any) -> dict[str, Any]:
    track = deepcopy(M19_REGISTRY[base_key]) if "M19_REGISTRY" in globals() else {}
    track.update(updates)
    return track


M19_REGISTRY: dict[str, dict[str, Any]] = {
    "M19": {
        "family": "symbiote_runway_mainline",
        "implementation_label": "qformer_runway_bridge",
        "runner_scripts": {
            "train": "scripts/m19/train_m19_mainline.py",
            "benchmark": "scripts/m19/run_m19_godtier_benchmark.py",
            "audit": "scripts/m19/run_m19_audit.py",
            "dictionary_audit": "scripts/m19/run_m19_dictionary_audit.py",
            "integrity": "scripts/m19/run_m19_integrity_suite.py",
            "replication": "scripts/m19/run_m19_replication_suite.py",
            "stability_microgrid": "scripts/m19/run_m19_stability_microgrid.py",
            "kill_tests": "scripts/m19/run_m19_kill_test_suite.py",
            "paper_package": "scripts/control_plane/render_m19_paper_package.py",
            "grid": "scripts/m19/run_m19_isolation_grid.py",
            "suite": "scripts/m19/run_m19_mainline_suite.py",
            "typed_suite": "scripts/m19/run_m19_typed_physics_suite.py",
            "gumbel_faithfulness": "scripts/m19/run_m19_gumbel_faithfulness_suite.py",
            "hyperbolic_faithfulness": "scripts/m19/run_m19_hyperbolic_faithfulness_suite.py",
        },
        "dags": {
            "train": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
            "benchmark": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
            "audit": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
            "dictionary_audit": "airflow/dags/m19/lojban_m19_dictionary_audit_dag.py",
            "integrity": "airflow/dags/m19/lojban_m19_integrity_suite_dag.py",
            "replication": "airflow/dags/m19/lojban_m19_replication_suite_dag.py",
            "stability_microgrid": "airflow/dags/m19/lojban_m19_stability_microgrid_dag.py",
            "kill_tests": "airflow/dags/m19/lojban_m19_kill_test_suite_dag.py",
            "paper_package": "airflow/dags/m19/lojban_m19_paper_package_dag.py",
            "mainline": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
            "grid": "airflow/dags/m19/lojban_m19_isolation_grid_dag.py",
            "typed_suite": "airflow/dags/m19/lojban_m19_typed_physics_suite_dag.py",
            "gumbel_faithfulness": "airflow/dags/m19/lojban_m19_gumbel_faithfulness_suite_dag.py",
            "hyperbolic_faithfulness": "airflow/dags/m19/lojban_m19_hyperbolic_faithfulness_suite_dag.py",
        },
        "output_roots": {
            "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_mainline_train",
            "benchmark": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_godtier_benchmark",
            "audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_audit",
            "dictionary_audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_dictionary_audit",
            "integrity": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_integrity_suite",
            "replication": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_replication_suite",
            "stability_microgrid": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_stability_microgrid",
            "kill_tests": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_kill_test_suite",
            "paper_package": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_paper_package",
            "mainline": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_mainline_suite",
            "grid": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_isolation_grid",
        },
        "report_names": {
            "train": "m19_train_manifest.json",
            "benchmark": "m19_benchmark_report.json",
            "audit": "m19_audit_report.json",
            "dictionary_audit": "m19_dictionary_audit_report.json",
            "integrity": "m19_integrity_report.json",
            "replication": "m19_replication_report.json",
            "stability_microgrid": "m19_stability_microgrid_report.json",
            "kill_tests": "m19_kill_test_report.json",
            "paper_package": "m19_paper_package_manifest.json",
            "mainline": "m19_mainline_report.json",
            "grid": "m19_isolation_grid_report.json",
        },
        "dataset_defaults": {
            "train": "artifacts/datasets/m19_mixed_curriculum_v1.jsonl",
            "benchmark": "artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl",
            "audit": "artifacts/datasets/sanity_check_v1.jsonl",
        },
        "thesis": "give the decoder a bounded continuous scratchpad runway by injecting learned residuals only into repeated <symbiote> token positions before final English continuation.",
        "architecture": {
            "stage_1": "english_prompt_encoding",
            "stage_2": "query_former_symbiote_compression",
            "stage_3": "scratchpad_only_layer_injection",
            "stage_4": "english_resumption_after_runway",
        },
        "parameter_axes": [
            "num_queries",
            "bottleneck_dim",
            "scratchpad_length",
            "tap_layer",
            "epochs",
            "learning_rate",
            "seed",
        ],
        "comparison_targets": [
            "BASE",
            "EN-COT",
            "RANDOM-SHAPE",
            "M11.discriminative",
        ],
        "default_grid": deepcopy(M19_ISOLATION_GRID),
        "replication_cells": {
            "R1": {"label": "replicate_width_probe", "base_cell": "B", "role": "winner_replication"},
            "R2": {"label": "replicate_baseline", "base_cell": "A", "role": "baseline_replication"},
        },
    },
    "M19.4": {
        "family": "symbiote_runway_dynamic_pacing",
        "implementation_label": "qformer_dynamic_pacing_bridge",
        "runner_scripts": {
            "train": "scripts/m19/train_m19_mainline.py",
            "benchmark": "scripts/m19/run_m19_godtier_benchmark.py",
            "suite": "scripts/m19/run_m19_mainline_suite.py",
        },
        "dags": {
            "train": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
            "benchmark": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
            "mainline": "airflow/dags/m19/lojban_m19_mainline_suite_dag.py",
        },
        "output_roots": {
            "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_dynamic_pacing_train",
            "benchmark": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_dynamic_pacing_benchmark",
            "mainline": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_dynamic_pacing_suite",
        },
        "report_names": {
            "train": "m19_4_train_manifest.json",
            "benchmark": "m19_4_benchmark_report.json",
            "mainline": "m19_4_mainline_report.json",
        },
        "dataset_defaults": {
            "train": "artifacts/datasets/m19_mixed_curriculum_v1.jsonl",
            "benchmark": "artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl",
        },
        "thesis": "replace fixed-length runway injection with autoregressive symbiote pacing that halts via a native <symbiote_end> token and exposes passive halt-similarity telemetry without using CAA as a control channel.",
        "architecture": {
            "stage_1": "english_prompt_encoding",
            "stage_2": "query_former_dynamic_reservoir_projection",
            "stage_3": "autoregressive_symbiote_pacing_loop",
            "stage_4": "symbiote_end_halt_and_english_resumption",
        },
        "parameter_axes": [
            "num_queries",
            "bottleneck_dim",
            "max_latent_steps",
            "min_latent_steps",
            "tap_layer",
            "epochs",
            "learning_rate",
            "seed",
        ],
        "comparison_targets": [
            "BASE",
            "EN-COT",
            "ZH-COT",
            "RANDOM-DYNAMIC",
            "M19.3.STATIC",
        ],
        "defaults": {
            "min_latent_steps": 4,
            "max_latent_steps": 64,
            "tap_layer": 12,
        },
    },
}

M19_REGISTRY["M19.31"] = _clone_track(
    "M19",
    family="symbiote_runway_typed_gumbel",
    implementation_label="typed_euclidean_gumbel_arity_bridge",
    thesis="retrofit the bounded runway with typed Lojban-inspired predicate, operator, and pointer slot families plus hard differentiable arity routing over judri slots.",
    output_roots={
        **M19_REGISTRY["M19"]["output_roots"],
        "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_train",
        "benchmark": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_benchmark",
        "audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_audit",
        "dictionary_audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_dictionary_audit",
        "integrity": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_integrity",
        "replication": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_replication",
        "kill_tests": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_kill_tests",
        "mainline": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_31_typed_gumbel_suite",
    },
    report_names={
        **M19_REGISTRY["M19"]["report_names"],
        "train": "m19_31_train_manifest.json",
        "benchmark": "m19_31_benchmark_report.json",
        "audit": "m19_31_audit_report.json",
        "dictionary_audit": "m19_31_dictionary_audit_report.json",
        "integrity": "m19_31_integrity_report.json",
        "replication": "m19_31_replication_report.json",
        "kill_tests": "m19_31_kill_test_report.json",
        "mainline": "m19_31_mainline_report.json",
    },
    defaults={
        "typed_slot_layout": "gismu:2,cmavo:2,judri:4",
        "arity_router_mode": "gumbel_hard",
        "geometry_mode": "euclidean",
        "typed_physics_config": "configs/m19_typed_physics_ontology.json",
    },
    aliases=["M19.3c"],
)

M19_REGISTRY["M19.32"] = _clone_track(
    "M19",
    family="symbiote_runway_typed_hyperbolic",
    implementation_label="typed_hyperbolic_codebook_bridge",
    thesis="retrofit the bounded runway with typed Lojban-inspired slot families and forward-pass hyperbolic radius-band separation for predicate, operator, and pointer roles.",
    output_roots={
        **M19_REGISTRY["M19"]["output_roots"],
        "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_train",
        "benchmark": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_benchmark",
        "audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_audit",
        "dictionary_audit": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_dictionary_audit",
        "integrity": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_integrity",
        "replication": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_replication",
        "kill_tests": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_kill_tests",
        "mainline": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_32_typed_hyperbolic_suite",
    },
    report_names={
        **M19_REGISTRY["M19"]["report_names"],
        "train": "m19_32_train_manifest.json",
        "benchmark": "m19_32_benchmark_report.json",
        "audit": "m19_32_audit_report.json",
        "dictionary_audit": "m19_32_dictionary_audit_report.json",
        "integrity": "m19_32_integrity_report.json",
        "replication": "m19_32_replication_report.json",
        "kill_tests": "m19_32_kill_test_report.json",
        "mainline": "m19_32_mainline_report.json",
    },
    defaults={
        "typed_slot_layout": "gismu:2,cmavo:2,judri:4",
        "arity_router_mode": "soft",
        "geometry_mode": "hyperbolic",
        "typed_physics_config": "configs/m19_typed_physics_ontology.json",
    },
    aliases=["M19.3d"],
)

M19_REGISTRY["M19.3c"] = deepcopy(M19_REGISTRY["M19.31"])
M19_REGISTRY["M19.3d"] = deepcopy(M19_REGISTRY["M19.32"])


def m19_track_spec(track: str = "M19") -> dict[str, Any]:
    return deepcopy(M19_REGISTRY[track])


def m19_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M19_ISOLATION_GRID)


def m19_default_output_root(kind: str) -> Path:
    return Path(M19_REGISTRY["M19"]["output_roots"][kind])
