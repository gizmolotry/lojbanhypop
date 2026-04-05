from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


SYMBIOTE_SCRATCHPAD_FAMILY_VERSION = "1.0"


SYMBIOTE_SCRATCHPAD_REGISTRY: dict[str, dict[str, Any]] = {
    "M14": {
        "family": "symbiote_scratchpad",
        "implementation_label": "hybrid_symbiote_scratchpad",
        "runner_script": "scripts/run_m14_symbiote_scratchpad.py",
        "dag": "airflow/dags/lojban_m14_symbiote_scratchpad_dag.py",
        "output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m14_symbiote_scratchpad",
        "report_name": "m14_report.json",
        "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
        "thesis": "continuous advisor state should be injected only into bounded scratchpad token positions so the decoder can absorb logic over several normal attention hops before English resumption",
        "architecture": {
            "stage_1": "english_prompt_encoding",
            "stage_2": "advisor_reasoning_space",
            "stage_3": "symbiote_scratchpad_injection",
            "stage_4": "english_continuation",
        },
        "tensor_flow": {
            "carrier": "advisor summary projected into scratchpad token hidden states",
            "exposure": "layer-selective residual injection over bounded scratchpad positions only",
            "reentry": "four-token compute horizon before final answer continuation",
            "scoring_surface": "language_model_answer_logits_and_short_continuation_rollout",
        },
        "parameter_axes": {
            "shared": [
                "base_model",
                "adapter",
                "checkpoint",
                "train_steps",
                "eval_size",
                "lr",
                "bottleneck_dim",
                "max_logic_new_tokens",
                "layer_index",
                "relation_vocab",
                "var_min_id",
                "answer_weight",
                "return_norm_weight",
                "continuation_target_max_tokens",
                "continuation_eval_tokens",
                "scratchpad_length",
                "scratchpad_alpha",
                "seed",
                "strict_balance",
            ],
            "track_specific": [
                "scratchpad_token",
                "b_guard_threshold",
                "c_guard_threshold",
                "d_guard_threshold",
                "residual_guard_weight",
            ],
        },
        "loss_profile": {
            "primary": "continuation_cross_entropy",
            "regularizers": [
                "scratchpad_residual_norm_penalty",
                "scratchpad_guard_overflow_penalty",
            ],
        },
        "continuation_metrics": [
            "held_out_accuracy",
            "english_fluency_score",
            "contamination_rate",
            "loop_rate",
            "continuation_overlap_f1",
            "gold_mention_rate",
            "exact_match_rate",
            "english_stays_english_rate",
            "resume_first_token_accuracy",
            "mean_intervention_delta_gold",
            "mean_answer_delta",
            "mean_scope",
            "scratchpad_attention_mass",
            "scratchpad_residual_norm",
            "scratchpad_gate_mean",
            "scratchpad_gate_overflow_rate",
            "scratchpad_bleed_rate",
            "seed_stability",
        ],
        "cells": {
            "A": {
                "label": "scratchpad-only control",
                "scratchpad_policy": "four_real_tokens_no_injection",
                "advisor_policy": "none",
                "reentry_policy": "baseline_decoder_with_blank_scratchpad",
                "guard_threshold": None,
            },
            "B": {
                "label": "strict residual scratchpad",
                "scratchpad_policy": "advisor_residual_injection",
                "advisor_policy": "continuous_summary",
                "reentry_policy": "layer_injection_into_symbiote_positions",
                "guard_threshold": 0.01,
            },
            "C": {
                "label": "relaxed residual scratchpad",
                "scratchpad_policy": "advisor_residual_injection",
                "advisor_policy": "continuous_summary",
                "reentry_policy": "layer_injection_into_symbiote_positions",
                "guard_threshold": 0.05,
            },
            "D": {
                "label": "severance-threshold residual scratchpad",
                "scratchpad_policy": "advisor_residual_injection",
                "advisor_policy": "continuous_summary",
                "reentry_policy": "layer_injection_into_symbiote_positions",
                "guard_threshold": 0.10,
            },
            "E": {
                "label": "token-only scratchpad baseline",
                "scratchpad_policy": "trainable_token_offsets_only",
                "advisor_policy": "none",
                "reentry_policy": "scratchpad_token_baseline_without_continuous_injection",
                "guard_threshold": None,
            },
        },
        "promotion_requirements": [
            "mean_intervention_delta_gold_positive",
            "resume_first_token_accuracy_gain",
            "fluency_preserved",
            "contamination_below_threshold",
            "loop_rate_below_threshold",
            "scratchpad_bleed_below_threshold",
        ],
        "comparison_targets": [
            "M3.18.D",
            "M3.19.D0",
            "M3.19.D1",
            "M3.19.D2",
            "M3.19.D3",
            "M11.discriminative",
        ],
    }
}


def scratchpad_track_spec(track: str = "M14") -> dict[str, Any]:
    return deepcopy(SYMBIOTE_SCRATCHPAD_REGISTRY[track])


def scratchpad_cell_labels(track: str = "M14") -> dict[str, str]:
    spec = SYMBIOTE_SCRATCHPAD_REGISTRY[track]
    return {cell: str(payload["label"]) for cell, payload in spec["cells"].items()}


def build_scratchpad_protocol_manifest(
    *,
    track: str,
    run_id: str,
    baseline_manifest_path: Path,
    baseline_id: str,
    upstream_m318_report: str | None,
    upstream_m319_report: str | None,
    upstream_m11_manifest: str | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = scratchpad_track_spec(track)
    return {
        "track": track,
        "run_id": run_id,
        "baseline_manifest": str(baseline_manifest_path).replace("\\", "/"),
        "baseline_id": str(baseline_id),
        "family_contract": {
            "family_version": SYMBIOTE_SCRATCHPAD_FAMILY_VERSION,
            "family_name": spec["family"],
            "implementation_label": spec["implementation_label"],
            "runner_script": spec["runner_script"],
            "dag": spec["dag"],
            "architecture": spec["architecture"],
            "tensor_flow": spec["tensor_flow"],
            "parameter_axes": spec["parameter_axes"],
            "loss_profile": spec["loss_profile"],
            "continuation_metrics": spec["continuation_metrics"],
            "promotion_requirements": spec["promotion_requirements"],
            "comparison_targets": spec["comparison_targets"],
            "cells": spec["cells"],
        },
        "upstream_evidence": {
            "m3_18_report": upstream_m318_report,
            "m3_19_report": upstream_m319_report,
            "m11_manifest": upstream_m11_manifest,
        },
        "config": deepcopy(config),
    }
