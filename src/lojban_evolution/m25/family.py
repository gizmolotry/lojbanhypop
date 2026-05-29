from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M25_FAMILY_VERSION = "0.1"


def m25_cell_id(cell_key: str) -> str:
    return f"M25.{str(cell_key).strip().upper()}"


M25_EMERGENT_BRIDI_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m25_cell_id("A"),
        "lock": "emergent_loose_bridi_stream",
        "label": "M23 relevance data with variable typed bridi grammar-action stream",
        "variant": {
            "trace_weight": 2.0,
            "answer_weight": 0.25,
            "mdl_weight": 0.01,
            "max_symbols": 32,
            "symbol_budget": 0,
        },
    }
]


M25_REGISTRY: dict[str, dict[str, Any]] = {
    "M25": {
        "family": "emergent_bridi_grammar",
        "implementation_label": "loose_typed_bridi_stream_generator_and_trace_only_advisor",
        "runner_scripts": {
            "suite": "scripts/m25/run_m25_emergent_bridi_suite.py",
        },
        "dags": {},
        "output_roots": {
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m25_emergent_bridi",
        },
        "report_names": {
            "suite": "m25_emergent_bridi_report.json",
        },
        "dataset_defaults": {
            "profile": "m23_decoy_balanced_to_loose_bridi_stream_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "test whether a looser learned bridi grammar-action stream can preserve causal symbolic "
            "compression without locking the substrate into fixed predicate rows."
        ),
        "architecture": {
            "stage_1": "reuse M23 decoy-balanced dynamic bridi examples as semantic source",
            "stage_2": "linearize frames into variable typed actions OPEN/PRED/MOD/ARG/LINK/CLOSE/STOP",
            "stage_3": "train a Q-former-like emitter over stream positions, not fixed frame slots",
            "stage_4": "freeze emitter and train a trace-only integer-stream advisor with shuffled/random controls",
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
        "comparison_targets": ["M23", "M24", "M24.2", "M25.A"],
        "default_grid": deepcopy(M25_EMERGENT_BRIDI_GRID),
    }
}


def m25_track_spec(track: str = "M25") -> dict[str, Any]:
    return deepcopy(M25_REGISTRY[track])


def m25_default_output_root(kind: str) -> Path:
    return Path(M25_REGISTRY["M25"]["output_roots"][kind])


def m25_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M25_EMERGENT_BRIDI_GRID)
