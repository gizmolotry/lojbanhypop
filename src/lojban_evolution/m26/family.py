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
        "lock": "differentiable_lm_hidden_bridi_bridge_organism",
        "label": "tiny LM hidden stream plus M25 loose bridi generator plus differentiable trace-language bridge",
        "variant": {
            "trace_weight": 2.0,
            "answer_weight": 1.0,
            "mdl_weight": 0.01,
            "max_symbols": 32,
            "symbol_budget": 0,
            "max_prompt_length": 128,
            "language_layers": 1,
            "language_heads": 2,
        },
    }
]


M26_REGISTRY: dict[str, dict[str, Any]] = {
    "M26": {
        "family": "end_to_end_lojban_symbiote_full_organism",
        "implementation_label": "single_optimizer_lm_hidden_to_bridi_to_trace_language_bridge_model",
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
            "answer loss reaches an English hidden-state stream, the bridi stream generator, and a "
            "trace-language bridge without a hard symbolic cut."
        ),
        "architecture": {
            "stage_1": "encode the English prompt with a trainable LM-shaped hidden-state stream",
            "stage_2": "feed those hidden states into the M25 loose bridi stream generator",
            "stage_3": "emit soft type/value/aux symbol distributions instead of hard packed integer symbols",
            "stage_4": "cross-attend from the prompt stream into the soft bridi trace through a differentiable bridge",
            "stage_5": "choke the final answer head so it reads the trace-conditioned bridge residual, not raw prompt state",
            "stage_6": "probe answer-loss gradients into the language backbone, generator, symbol heads, advisor, and bridge",
        },
        "parameter_axes": [
            "trace_weight",
            "answer_weight",
            "mdl_weight",
            "max_symbols",
            "symbol_budget",
            "advisor_hidden_dim",
            "max_prompt_length",
            "language_layers",
            "language_heads",
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
