from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M24_FAMILY_VERSION = "0.1"


def m24_cell_id(cell_key: str) -> str:
    return f"M24.{str(cell_key).strip().upper()}"


M24_SUBSTRATE_COMPRESSION_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m24_cell_id("A"),
        "lock": "substrate_first_compression",
        "label": "M23 substrate generator plus frozen packed-trace advisor",
        "variant": {
            "trace_weight": 2.5,
            "answer_weight": 0.2,
            "mdl_weight": 0.01,
            "trace_exact_surrogate_weight": 0.5,
            "active_frame_budget": 0,
            "trace_symbol_budget": 0,
        },
    }
]


M24_REGISTRY: dict[str, dict[str, Any]] = {
    "M24": {
        "family": "substrate_first_compression",
        "implementation_label": "m23_dynamic_bridi_generator_frozen_packed_symbolic_trace_advisor",
        "runner_scripts": {
            "suite": "scripts/m24/run_m24_substrate_compression_suite.py",
        },
        "dags": {},
        "output_roots": {
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m24_substrate_compression",
        },
        "report_names": {
            "suite": "m24_substrate_compression_report.json",
        },
        "dataset_defaults": {
            "profile": "m23_decoy_balanced_dynamic_bridi_relevance_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "test whether a compressed symbolic bridi substrate produced by the M23 dynamic generator "
            "is sufficient for downstream answers without reading continuous frame or prompt states."
        ),
        "architecture": {
            "stage_1": "reuse train_m23_relevance_router with trace-heavy answer-light weights",
            "stage_2": "freeze generator and pack predicted bridi logits into integer symbolic traces",
            "stage_3": "train a separate trace-only advisor on packed predicted symbolic traces",
            "stage_4": "evaluate predicted/oracle/random/zero trace and prompt-only controls",
        },
        "parameter_axes": [
            "trace_weight",
            "answer_weight",
            "mdl_weight",
            "trace_exact_surrogate_weight",
            "active_frame_budget",
            "trace_symbol_budget",
            "advisor_hidden_dim",
            "seed",
        ],
        "comparison_targets": ["M21", "M23", "M24.A"],
        "default_grid": deepcopy(M24_SUBSTRATE_COMPRESSION_GRID),
    }
}


def m24_track_spec(track: str = "M24") -> dict[str, Any]:
    return deepcopy(M24_REGISTRY[track])


def m24_default_output_root(kind: str) -> Path:
    return Path(M24_REGISTRY["M24"]["output_roots"][kind])


def m24_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M24_SUBSTRATE_COMPRESSION_GRID)
