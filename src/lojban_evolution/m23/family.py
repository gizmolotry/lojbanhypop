from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M23_FAMILY_VERSION = "0.1"


def m23_cell_id(cell_key: str) -> str:
    return f"M23.{str(cell_key).strip().upper()}"


M23_RELEVANCE_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m23_cell_id("A"),
        "lock": "scale_control",
        "label": "Scale control with decoy-balanced data and no relevance router",
        "variant": {
            "use_relevance_router": False,
            "relevance_rank_weight": 0.0,
            "clean_train_fraction": 0.35,
            "judri_bridge_gate": True,
        },
    },
    {
        "cell_key": "B",
        "cell_id": m23_cell_id("B"),
        "lock": "relevance_router",
        "label": "Causal relevance router over active bridi frames",
        "variant": {
            "use_relevance_router": True,
            "relevance_rank_weight": 1.0,
            "relevance_margin": 0.15,
            "clean_train_fraction": 0.35,
            "judri_bridge_gate": True,
        },
    },
    {
        "cell_key": "C",
        "cell_id": m23_cell_id("C"),
        "lock": "trace_punishment",
        "label": "Scale control plus exact bridi-trace punishment",
        "variant": {
            "use_relevance_router": False,
            "relevance_rank_weight": 0.0,
            "trace_weight": 2.5,
            "trace_exact_surrogate_weight": 0.5,
            "clean_train_fraction": 0.35,
            "judri_bridge_gate": True,
        },
    },
]


M23_REGISTRY: dict[str, dict[str, Any]] = {
    "M23": {
        "family": "causal_relevance_router",
        "implementation_label": "m21_m22_dynamic_bridi_frame_relevance_selector",
        "runner_scripts": {
            "train": "scripts/m23/train_m23_relevance_router.py",
            "suite": "scripts/m23/run_m23_relevance_suite.py",
        },
        "dags": {
            "suite": "airflow/dags/m23/lojban_m23_relevance_router_dag.py",
        },
        "output_roots": {
            "train": "artifacts/runs/telemetry/raw/ablation/hypercube/m23_relevance_train",
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m23_relevance_suite",
        },
        "report_names": {
            "train": "m23_relevance_train_report.json",
            "suite": "m23_relevance_suite_report.json",
        },
        "dataset_defaults": {
            "profile": "m23_decoy_balanced_dynamic_bridi_relevance_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "test whether the M21/M22 dynamic bridi scratchpad needs an explicit selector over "
            "answer-causal frames, or whether QKV plus decoy-balanced scale learns relevance unaided."
        ),
        "architecture": {
            "stage_1": "reuse_m21_m22_dynamic_bridi_trace_substrate",
            "stage_2": "attach_relevance_and_decoy_frame_metadata",
            "stage_3": "train_scale_control_without_router",
            "stage_4": "train_lightweight_relevance_scoring_head_with_single_rank_loss",
            "stage_5": "evaluate_oracle_random_uniform_and_decoy_only_read_paths",
        },
        "parameter_axes": [
            "use_relevance_router",
            "relevance_rank_weight",
            "relevance_margin",
            "clean_train_fraction",
            "seed",
        ],
        "comparison_targets": [
            "M22",
            "M21.1.S",
            "M21.1.T",
            "M23.A",
            "M23.B",
            "M23.C",
        ],
        "default_grid": deepcopy(M23_RELEVANCE_GRID),
    }
}


def m23_track_spec(track: str = "M23") -> dict[str, Any]:
    return deepcopy(M23_REGISTRY[track])


def m23_default_output_root(kind: str) -> Path:
    return Path(M23_REGISTRY["M23"]["output_roots"][kind])


def m23_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M23_RELEVANCE_GRID)
