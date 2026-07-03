from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M27_FAMILY_VERSION = "0.1"


def m27_cell_id(cell_key: str) -> str:
    return f"M27.{str(cell_key).strip().upper()}"


M27_COCONUT_BRIDI_GRID: list[dict[str, Any]] = [
    {
        "cell_key": "A",
        "cell_id": m27_cell_id("A"),
        "lock": "coconut_autoregressive_bridi_runtime",
        "label": "M26 full organism with the M25/M26 parallel emitter replaced by a recurrent Coconut loose-bridi runtime",
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
    },
    {
        "cell_key": "B",
        "cell_id": m27_cell_id("B"),
        "lock": "coconut_autoregressive_bridi_runtime_with_relevance_router",
        "label": "M27.A plus an optional M23-style relevance runtime organ over emitted loose-bridi slots",
        "variant": {
            "trace_weight": 2.0,
            "answer_weight": 1.0,
            "mdl_weight": 0.01,
            "max_symbols": 32,
            "symbol_budget": 0,
            "max_prompt_length": 128,
            "language_layers": 1,
            "language_heads": 2,
            "enable_relevance_runtime": True,
            "relevance_rank_weight": 0.25,
            "relevance_margin": 0.15,
            "use_relevance_answer": True,
            "relevance_temperature": 1.0,
        },
    }
]


M27_REGISTRY: dict[str, dict[str, Any]] = {
    "M27": {
        "family": "coconut_bridi_runtime_full_organism",
        "implementation_label": "single_optimizer_lm_hidden_to_autoregressive_coconut_bridi_to_trace_language_bridge_model",
        "runner_scripts": {
            "suite": "scripts/m27/run_m27_coconut_bridi_runtime_suite.py",
        },
        "dags": {
            "suite": "airflow/dags/m27/lojban_m27_coconut_bridi_runtime_dag.py",
        },
        "output_roots": {
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m27_coconut_bridi_runtime",
        },
        "report_names": {
            "suite": "m27_coconut_bridi_runtime_report.json",
        },
        "dataset_defaults": {
            "profile": "m25_loose_bridi_stream_with_m26_bridge_and_coconut_autoregression_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "test whether the Lojban symbiote exists as a real autoregressive hidden-state organism: "
            "an English hidden stream seeds a Coconut recurrent scratchpad, the scratchpad emits one loose-bridi "
            "symbol at a time, and the final answer loss reaches the recurrent feedback loop through a differentiable bridge."
        ),
        "architecture": {
            "stage_1": "encode the English prompt with the trainable M26 tiny language backbone",
            "stage_2": "initialize a recurrent Coconut latent state from the prompt stream",
            "stage_3": "emit loose-bridi type/value/aux symbols one step at a time with differentiable feedback",
            "stage_4": "support hard free-run inference separately from soft teacher-forced training",
            "stage_5": "reuse the M26 trace-language cross-attention bridge and choked answer head",
            "stage_6": "compare the full runtime against zero/shuffle/random/no-recurrence and prompt-only controls",
            "stage_7": "optionally score emitted trace slots with an M23-style relevance organ before the bridge reads them",
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
            "enable_relevance_runtime",
            "relevance_rank_weight",
            "use_relevance_answer",
            "relevance_temperature",
            "seed",
        ],
        "comparison_targets": ["M26", "M25", "M24.2", "M23"],
        "default_grid": deepcopy(M27_COCONUT_BRIDI_GRID),
    }
}


def m27_track_spec(track: str = "M27") -> dict[str, Any]:
    return deepcopy(M27_REGISTRY[track])


def m27_default_output_root(kind: str) -> Path:
    return Path(M27_REGISTRY["M27"]["output_roots"][kind])


def m27_default_grid() -> list[dict[str, Any]]:
    return deepcopy(M27_COCONUT_BRIDI_GRID)
