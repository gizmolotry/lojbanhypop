from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M26_FAMILY_VERSION = "0.1"


def m26_cell_id(cell_key: str) -> str:
    return f"M26.{str(cell_key).strip().upper()}"


M26_END_TO_END_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m26_cell_id("A"),
        "lock": "differentiable_loose_bridi_spinal_cord",
        "label": "M25 loose stream emitter with differentiable trace-only advisor under one optimizer",
        "variant": {
            "trace_weight": 2.0,
            "answer_weight": 1.0,
            "mdl_weight": 0.01,
            "max_symbols": 32,
            "symbol_budget": 0,
        },
    }
]


M26_REGISTRY: dict[str, dict[str, Any]] = {
    "M26": {
        "family": "end_to_end_lojban_symbiote_spinal_cord",
        "implementation_label": "single_optimizer_prompt_to_bridi_to_advisor_model",
        "runner_scripts": {
            "suite": "scripts/m26/run_m26_end_to_end_loafman_suite.py",
        },
        "dags": {
            "suite": "airflow/dags/m26/lojban_m26_end_to_end_loafman_dag.py",
        },
        "output_roots": {
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m26_end_to_end_loafman",
        },
        "report_names": {
            "suite": "m26_end_to_end_loafman_report.json",
        },
        "dataset_defaults": {
            "profile": "m25_loose_bridi_stream_with_differentiable_advisor_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "test whether the Lojban symbiote substrate exists as one trainable organism by proving final "
            "answer loss reaches the bridi stream generator through a differentiable advisor path."
        ),
        "architecture": {
            "stage_1": "reuse M25 loose bridi stream supervision and M23/M25 semantic data source",
            "stage_2": "emit soft type/value/aux symbol distributions instead of hard packed integer symbols",
            "stage_3": "read the soft bridi trace with a trace-only advisor under the same optimizer",
            "stage_4": "probe answer-loss gradients into the generator and symbol heads",
        },
        "parameter_axes": [
            "trace_weight",
            "answer_weight",
            "mdl_weight",
            "max_symbols",
            "symbol_budget",
            "advisor_hidden_dim",
            "seed",
        ],
        "comparison_targets": ["M25", "M24.2", "M23"],
        "default_grid": deepcopy(M26_END_TO_END_GRID),
    }
}


def m26_track_spec(track: str = "M26") -> dict[str, Any]:
    return deepcopy(M26_REGISTRY[track])


def m26_default_output_root(kind: str) -> Path:
    return Path(M26_REGISTRY["M26"]["output_roots"][kind])


def m26_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M26_END_TO_END_GRID)
