from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


REENTRY_REBOOT_FAMILY_VERSION = "1.0"


REENTRY_REBOOT_REGISTRY: dict[str, dict[str, Any]] = {
    "M3.18": {
        "family": "reentry_architecture",
        "implementation_label": "decoder_reentry_resume",
        "runner_script": "scripts/run_m3_18_decoder_reentry_resume.py",
        "dag": "airflow/dags/lojban_m3_18_decoder_reentry_resume_dag.py",
        "output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume",
        "baseline_manifest": "docs/baselines/m_series_bridge_baseline_manifest.json",
        "thesis": "advisor reasoning should hand back a compact decoder-native summary rather than remain directly exposed during generation",
        "architecture": {
            "stage_1": "english_context_encoder",
            "stage_2": "advisor_reasoning_space",
            "stage_3": "reentry_encoder",
            "stage_4": "english_resumption",
        },
        "advisor_axes": [
            "state_type",
            "binding_mechanism",
            "concept_neighborhood_support",
            "pointer_awareness",
        ],
        "return_channel_axes": [
            "single_return_token",
            "multi_return_tokens",
            "residual_summary_vector",
            "hybrid_token_plus_residual",
        ],
        "continuation_metrics": [
            "held_out_accuracy",
            "role_resolution_accuracy",
            "ambiguity_handling_accuracy",
            "multi_hop_consistency",
            "english_fluency_score",
            "contamination_rate",
            "loop_rate",
            "continuation_overlap_f1",
            "gold_mention_rate",
            "exact_match_rate",
            "english_stays_english_rate",
            "resume_first_token_accuracy",
            "kill_test_accuracy",
            "first_token_accuracy",
            "gold_vs_foil_margin",
            "intervention_effect_on_gold",
            "state_norm",
            "residual_guard_overflow_rate",
            "state_gate",
            "state_attention_entropy",
            "seed_stability",
        ],
        "cells": {
            "A": {
                "label": "control no advisor",
                "advisor_state": "none",
                "return_channel": "none",
                "rollout_exposure": "none",
                "resume_policy": "baseline_decoder_only",
            },
            "B": {
                "label": "frozen single return token",
                "advisor_state": "frozen_hybrid",
                "return_channel": "single_return_token",
                "rollout_exposure": "one_shot_pre_answer_only",
                "resume_policy": "decoder_resumes_after_single_token",
            },
            "C": {
                "label": "frozen multi-return token bundle",
                "advisor_state": "frozen_hybrid",
                "return_channel": "multi_return_tokens",
                "rollout_exposure": "one_shot_pre_answer_only",
                "resume_policy": "decoder_resumes_after_token_bundle",
            },
            "D": {
                "label": "learned residual continuation vector",
                "advisor_state": "continuous_summary",
                "return_channel": "residual_summary_vector",
                "rollout_exposure": "one_shot_hidden_delta_only",
                "resume_policy": "decoder_resumes_from_hidden_delta",
            },
            "E": {
                "label": "hybrid token plus residual translator",
                "advisor_state": "hybrid",
                "return_channel": "hybrid_token_plus_residual",
                "rollout_exposure": "one_shot_compact_resume_state",
                "resume_policy": "decoder_resumes_from_compound_reentry_state",
            },
        },
        "promotion_requirements": [
            "held_out_accuracy_gain",
            "fluency_preserved",
            "contamination_below_threshold",
            "loop_rate_below_threshold",
            "first_token_preserved",
            "kill_test_gain",
        ],
        "upstream_dependencies": [
            "M3.15d",
            "M3.16",
            "M3.17",
            "M11.discriminative",
        ],
    }
}


def reentry_track_spec(track: str = "M3.18") -> dict[str, Any]:
    return deepcopy(REENTRY_REBOOT_REGISTRY[track])


def reentry_cell_labels(track: str = "M3.18") -> dict[str, str]:
    spec = REENTRY_REBOOT_REGISTRY[track]
    return {cell: str(payload["label"]) for cell, payload in spec["cells"].items()}


def build_reentry_protocol_manifest(
    *,
    track: str,
    run_id: str,
    baseline_manifest_path: Path,
    baseline_id: str,
    upstream_bridge_suite: str | None,
    upstream_m11_manifest: str | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    spec = reentry_track_spec(track)
    return {
        "track": track,
        "run_id": run_id,
        "baseline_manifest": str(baseline_manifest_path).replace("\\", "/"),
        "baseline_id": str(baseline_id),
        "family_contract": {
            "family_version": REENTRY_REBOOT_FAMILY_VERSION,
            "family_name": "reentry_architecture",
            "implementation_label": spec["implementation_label"],
            "runner_script": spec["runner_script"],
            "dag": spec["dag"],
            "architecture": spec["architecture"],
            "advisor_axes": spec["advisor_axes"],
            "return_channel_axes": spec["return_channel_axes"],
            "continuation_metrics": spec["continuation_metrics"],
            "promotion_requirements": spec["promotion_requirements"],
            "cells": spec["cells"],
            "upstream_dependencies": spec["upstream_dependencies"],
        },
        "upstream_evidence": {
            "bridge_suite_manifest": upstream_bridge_suite,
            "m11_manifest": upstream_m11_manifest,
        },
        "config": deepcopy(config),
    }
