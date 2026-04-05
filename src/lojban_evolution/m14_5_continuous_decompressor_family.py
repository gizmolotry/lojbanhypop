from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


DECOMPRESSOR_FAMILY_VERSION = "1.0"


M14_5_REGISTRY: dict[str, dict[str, Any]] = {
    "M14.5": {
        "family": "continuous_decompressor",
        "implementation_label": "hybrid_decompression_runway",
        "runner_script": "scripts/run_m14_5_decompressor.py",
        "dag": "airflow/dags/lojban_m14_5_decompressor_dag.py",
        "output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m14_5_decompressor",
        "report_name": "m14_5_report.json",
        "baseline_manifest": "RESULTS_M10_FINAL_AUDIT/m11_discriminative_manifest.json",
        "thesis": "a continuous semantic seed injected at Layer 12 can causally constrain the autoregressive generation of a discrete Lojban runway, and this generated trace will govern the final English answer better than sequence-native or continuous-only methods",
        "architecture": {
            "stage_1": "continuous_seed_injection",
            "stage_2": "discrete_runway_unspooling",
            "stage_3": "english_answer_resolution",
        },
        "tensor_flow": {
            "seed": "M11 Forge continuous AST tensor injected into the residual stream of a single <loj_seed> token",
            "unspooling": "Model autoregressively emits <loj_i> tokens (or is teacher-forced)",
            "reentry": "Model hits </loj_think> and emits English answer",
        },
        "parameter_axes": {
            "shared": [
                "base_model",
                "adapter",
                "checkpoint",
                "train_steps",
                "eval_size",
                "lr",
                "layer_index",
                "scratchpad_alpha",
                "seed",
            ],
            "track_specific": [
                "runway_token",
                "seed_token",
                "max_runway_length",
            ],
        },
        "decompression_metrics": [
            "operator_match_accuracy",
            "pointer_match_rate",
            "trace_edit_distance",
            "seed_faithfulness_delta",
        ],
        "answer_metrics": [
            "held_out_accuracy",
            "english_fluency_score",
            "contamination_rate",
            "causal_drop_shuffled_trace",
            "causal_drop_no_seed",
        ],
        "cells": {
            "A": {
                "label": "free unspooling",
                "injection_policy": "active_seed",
                "runway_policy": "autoregressive_generation",
            },
            "B": {
                "label": "teacher-forced gold",
                "injection_policy": "active_seed",
                "runway_policy": "teacher_forced_gold_trace",
            },
            "C": {
                "label": "no-seed gold",
                "injection_policy": "zero_seed",
                "runway_policy": "teacher_forced_gold_trace",
            },
            "D": {
                "label": "seed + foil trace",
                "injection_policy": "active_seed",
                "runway_policy": "shuffled_gold_trace",
            },
        },
        "promotion_requirements": [
            "decompression_fidelity_beats_random",
            "answer_depends_on_trace_content",
            "fluency_preserved",
            "seed_faithfulness_delta_positive",
        ],
        "comparison_targets": [
            "M14.C",
            "M15.baseline",
            "M11.discriminative",
        ],
    }
}


def decompressor_track_spec(track: str = "M14.5") -> dict[str, Any]:
    return deepcopy(M14_5_REGISTRY[track])


def build_decompressor_protocol_manifest(
    *,
    track: str,
    run_id: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = decompressor_track_spec(track)
    return {
        "track": track,
        "run_id": run_id,
        "family_contract": {
            "family_version": DECOMPRESSOR_FAMILY_VERSION,
            "family_name": spec["family"],
            "runner_script": spec["runner_script"],
            "dag": spec["dag"],
            "architecture": spec["architecture"],
            "tensor_flow": spec["tensor_flow"],
            "decompression_metrics": spec["decompression_metrics"],
            "answer_metrics": spec["answer_metrics"],
            "promotion_requirements": spec["promotion_requirements"],
            "cells": spec["cells"],
        },
        "config": deepcopy(config),
    }
