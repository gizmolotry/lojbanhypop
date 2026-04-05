from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


BRIDGE_ABLATION_FAMILY_VERSION = "1.0"


BRIDGE_ABLATION_REGISTRY: dict[str, dict[str, Any]] = {
    "M3.15d": {
        "family": "bridge_ablation",
        "implementation_label": "answer_path_forcing",
        "runner_script": "scripts/run_m3_15d_answer_path_forcing.py",
        "dag": "airflow/dags/lojban_m3_15d_answer_path_forcing_dag.py",
        "search_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_15d_answer_path_forcing",
        "report_name": "m3_15d_report.json",
        "tensor_flow": {
            "carrier": "persistent advisor trace",
            "exposure": "answer-path cross-attention during continuation",
            "reentry": "none_direct_bridge",
            "scoring_surface": "language_model_answer_logits_or_pointer_head",
        },
        "lora_positioning": {
            "base_adapter": "phase5_two_stage_recovery_anchor_adapter",
            "bridge_coupling_locus": "answer_path_resolution",
            "bridge_application": "during_continuation_and_final_resolution",
        },
        "parameter_axes": {
            "shared": [
                "base_model",
                "adapter",
                "checkpoint",
                "train_steps",
                "eval_size",
                "lr",
                "max_logic_new_tokens",
                "layer_index",
                "relation_vocab",
                "var_min_id",
                "answer_weight",
                "margin",
                "seed",
                "strict_balance",
            ],
            "track_specific": [
                "max_nodes",
                "anti_collapse_weight",
                "collapse_entropy_floor_ratio",
                "collapse_top1_margin",
                "collapse_top1_weight",
                "collapse_kl_weight",
                "bridge_train_gate_cap",
                "runtime_gate_cap",
                "runtime_cue_norm_cap",
                "runtime_enable_min_acc_gain",
            ],
        },
        "loss_profile": {
            "primary": "answer_path_margin",
            "regularizers": [
                "anti_collapse_entropy_floor",
                "anti_collapse_top1_margin",
                "anti_collapse_kl",
            ],
        },
        "cells": {
            "A": {"label": "control from de-collapsed bridge base", "mode": "control", "exposure_policy": "none", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "B": {"label": "direct-drive gradient scalpel", "mode": "direct_drive", "exposure_policy": "full_answer_path", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "C": {"label": "late-stage causal blindfold", "mode": "blindfold", "exposure_policy": "masked_question_context", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "D": {"label": "topological candidate pointer head", "mode": "pointer", "exposure_policy": "relation_local_pointer", "reentry_policy": "none", "scorer": "candidate_pointer_head"},
        },
    },
    "M3.16": {
        "family": "bridge_ablation",
        "implementation_label": "continuous_graph_bias",
        "runner_script": "scripts/run_m3_16_continuous_graph_bias.py",
        "dag": "airflow/dags/lojban_m3_16_continuous_graph_bias_dag.py",
        "search_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_16_continuous_graph_bias",
        "report_name": "m3_16_report.json",
        "tensor_flow": {
            "carrier": "continuous graph bias",
            "exposure": "single-step additive bias at answer selection",
            "reentry": "none_direct_bias",
            "scoring_surface": "language_model_answer_logits",
        },
        "lora_positioning": {
            "base_adapter": "phase5_two_stage_recovery_anchor_adapter",
            "bridge_coupling_locus": "answer_token_scoring",
            "bridge_application": "soft_prompt_field_bias",
        },
        "parameter_axes": {
            "shared": [
                "base_model",
                "adapter",
                "checkpoint",
                "train_steps",
                "eval_size",
                "lr",
                "max_logic_new_tokens",
                "layer_index",
                "relation_vocab",
                "var_min_id",
                "answer_weight",
                "margin",
                "seed",
                "strict_balance",
            ],
            "track_specific": ["bias_norm_weight"],
        },
        "loss_profile": {
            "primary": "answer_path_margin",
            "regularizers": ["bias_norm_penalty"],
        },
        "cells": {
            "A": {"label": "control frozen bridge base", "mode": "control", "exposure_policy": "none", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "B": {"label": "candidate-only soft graph bias", "mode": "candidate_only", "exposure_policy": "candidate_spans", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "C": {"label": "candidate+cue soft graph bias", "mode": "candidate_plus_cue", "exposure_policy": "candidate_spans_plus_cue", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "D": {"label": "global prompt soft graph bias", "mode": "global_prompt_bias", "exposure_policy": "global_prompt_field", "reentry_policy": "none", "scorer": "lm_answer_logits"},
        },
    },
    "M3.17": {
        "family": "bridge_ablation",
        "implementation_label": "advisor_reentry_bridge",
        "runner_script": "scripts/run_m3_17_advisor_reentry_bridge.py",
        "dag": "airflow/dags/lojban_m3_17_advisor_reentry_bridge_dag.py",
        "search_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_17_advisor_reentry_bridge",
        "report_name": "m3_17_report.json",
        "tensor_flow": {
            "carrier": "advisor state summary",
            "exposure": "compressed return-state handoff before continuation",
            "reentry": "single_or_multi_return_tokens_or_residual_summary",
            "scoring_surface": "language_model_answer_logits_from_reentry_state",
        },
        "lora_positioning": {
            "base_adapter": "phase5_two_stage_recovery_anchor_adapter",
            "bridge_coupling_locus": "pre_answer_reentry_state",
            "bridge_application": "single_or_multi_state_decoder_resumption",
        },
        "parameter_axes": {
            "shared": [
                "base_model",
                "adapter",
                "checkpoint",
                "train_steps",
                "eval_size",
                "lr",
                "max_logic_new_tokens",
                "layer_index",
                "relation_vocab",
                "var_min_id",
                "answer_weight",
                "margin",
                "seed",
                "strict_balance",
            ],
            "track_specific": ["bottleneck_dim", "num_return_tokens", "return_norm_weight"],
        },
        "loss_profile": {
            "primary": "answer_path_margin",
            "regularizers": ["return_norm_penalty"],
        },
        "cells": {
            "A": {"label": "control no re-entry", "mode": "control_no_reentry", "exposure_policy": "none", "reentry_policy": "none", "scorer": "lm_answer_logits"},
            "B": {"label": "single return-state bottleneck", "mode": "single_return_state", "exposure_policy": "single_summary_state", "reentry_policy": "single_return_token", "scorer": "lm_answer_logits"},
            "C": {"label": "three return-state bottleneck", "mode": "three_return_states", "exposure_policy": "short_summary_bundle", "reentry_policy": "multi_return_tokens", "scorer": "lm_answer_logits"},
            "D": {"label": "direct residual re-encoder", "mode": "direct_residual_reencoder", "exposure_policy": "single_residual_summary", "reentry_policy": "residual_hidden_delta", "scorer": "lm_head_from_residual_hidden"},
        },
    },
    "M11.discriminative": {
        "family": "m11_native_eval",
        "implementation_label": "native_discriminative_bridge",
        "runner_script": "scripts/run_m11_discriminative_suite.py",
        "dag": "airflow/dags/lojban_m11_discriminative_suite_dag.py",
        "search_root": "artifacts/runs/telemetry/raw/ablation/hypercube/m11_discriminative_suite",
        "manifest_name": "m11_discriminative_suite_manifest.json",
        "tensor_flow": {
            "carrier": "provenance manifold plus native translator/head",
            "exposure": "adapter then discriminative head",
            "reentry": "native checkpoint pair rather than rollout-time bridge",
            "scoring_surface": "discriminative_head",
        },
        "lora_positioning": {
            "base_adapter": "m11_native_adapter",
            "bridge_coupling_locus": "translator_plus_english_head",
            "bridge_application": "post_reasoning_discriminative_resolution",
        },
        "parameter_axes": {
            "shared": ["base_model", "forge_ckpt", "adapter_ckpt", "head_ckpt", "num_samples"],
            "track_specific": ["disable_adapter", "train_steps", "lr"],
        },
        "loss_profile": {
            "primary": "discriminative_cross_entropy",
            "regularizers": [],
        },
        "cells": {},
    },
}


BRIDGE_NORMALIZED_METRICS: dict[str, str] = {
    "overall_accuracy": "held_out_accuracy",
    "validation_accuracy": "validation_accuracy",
    "seed2_accuracy": "alternate_seed_accuracy",
    "mean_answer_delta": "gold_vs_foil_margin",
    "mean_intervention_delta_gold": "intervention_effect_on_gold",
    "mean_scope": "scope_violation_score",
    "mean_operator_entropy": "operator_entropy",
    "mean_top1_op_share": "operator_top1_share",
    "mean_alignment_similarity": "alignment_similarity",
    "mean_bias_norm": "state_norm",
    "mean_return_norm": "state_norm",
    "mean_bias_gate": "state_gate",
    "mean_return_gate": "state_gate",
    "mean_return_attn_entropy": "state_attention_entropy",
    "mean_candidate_mass": "candidate_focus_mass",
    "mean_cue_mass": "cue_focus_mass",
}


def bridge_track_spec(track: str) -> dict[str, Any]:
    return deepcopy(BRIDGE_ABLATION_REGISTRY[track])


def bridge_cell_spec(track: str, cell: str) -> dict[str, Any] | None:
    spec = BRIDGE_ABLATION_REGISTRY.get(track, {})
    cell_key = str(cell).split("_")[0]
    cell_spec = spec.get("cells", {}).get(cell_key)
    if cell_spec is None:
        return None
    return deepcopy(cell_spec)


def track_cell_labels(track: str) -> dict[str, str]:
    spec = BRIDGE_ABLATION_REGISTRY[track]
    return {cell: str(cell_spec["label"]) for cell, cell_spec in spec.get("cells", {}).items()}


def build_bridge_report(
    *,
    track: str,
    script_path: str,
    args: Any,
    baseline_manifest_path: Path,
    baseline_id: str,
    checkpoint_in: str | None,
    split_meta: dict[str, Any],
    seed2_meta: dict[str, Any] | None,
    runtime_policy_source: str,
    final_metrics_source: str,
) -> dict[str, Any]:
    spec = bridge_track_spec(track)
    return {
        "timestamp": None,
        "series": None,
        "track": track,
        "baseline_manifest": str(baseline_manifest_path).replace("\\", "/"),
        "baseline_id": str(baseline_id),
        "config": {k: str(v) for k, v in vars(args).items()},
        "data_split": {
            **split_meta,
            "seed2_eval_meta": seed2_meta or {},
            "runtime_policy_source": runtime_policy_source,
            "final_metrics_source": final_metrics_source,
        },
        "ablation_contract": {
            "family_version": BRIDGE_ABLATION_FAMILY_VERSION,
            "family_name": "m_bridge_ablations",
            "runner_script": script_path,
            "dag": spec.get("dag"),
            "implementation_label": spec.get("implementation_label"),
            "tensor_flow": spec.get("tensor_flow"),
            "lora_positioning": spec.get("lora_positioning"),
            "parameter_axes": spec.get("parameter_axes"),
            "loss_profile": spec.get("loss_profile"),
            "variant_cells": spec.get("cells"),
            "normalized_metric_aliases": BRIDGE_NORMALIZED_METRICS,
            "checkpoint_in": str(checkpoint_in).replace("\\", "/") if checkpoint_in else None,
        },
        "cells": {},
    }


def finalize_bridge_report(report: dict[str, Any], track: str) -> dict[str, Any]:
    report.setdefault("ablation_contract", {})
    report["ablation_contract"].setdefault("family_version", BRIDGE_ABLATION_FAMILY_VERSION)
    report["ablation_contract"].setdefault("family_name", "m_bridge_ablations")
    report["ablation_contract"].setdefault("normalized_metric_aliases", deepcopy(BRIDGE_NORMALIZED_METRICS))
    report["ablation_contract"].setdefault("variant_cells", deepcopy(BRIDGE_ABLATION_REGISTRY[track].get("cells", {})))

    for cell_name, payload in list(report.get("cells", {}).items()):
        if not isinstance(payload, dict):
            continue
        cell_spec = bridge_cell_spec(track, str(cell_name))
        if cell_spec is not None:
            payload.setdefault("variant_spec", cell_spec)

    return report
