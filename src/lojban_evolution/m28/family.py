from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


M28_FAMILY_VERSION = "0.1"


M28_REGISTRY: dict[str, dict[str, Any]] = {
    "M28": {
        "family": "actual_logebonic_symbiote_model",
        "implementation_label": "checkpointable_prompt_to_recurrent_logebonic_trace_to_bridge_model",
        "runner_scripts": {
            "model_smoke": "scripts/m28/run_m28_logebonic_model_smoke.py",
            "model_suite": "scripts/m28/run_m28_logebonic_model_suite.py",
        },
        "output_roots": {
            "model": "artifacts/runs/telemetry/raw/ablation/hypercube/m28_logebonic_symbiote_model",
            "suite": "artifacts/runs/telemetry/raw/ablation/hypercube/m28_logebonic_symbiote_model_suite",
        },
        "report_names": {
            "model": "m28_logebonic_model_report.json",
            "suite": "m28_logebonic_model_suite_report.json",
        },
        "dataset_defaults": {
            "profile": "m25_m27_synthetic_logebonic_trace_actual_model_v1",
            "train_size": 6000,
            "eval_size": 1500,
        },
        "thesis": (
            "convert the M27 assay organism into a reusable actual model artifact: "
            "a checkpointable Logebonic Symbiote with prompt encoder, recurrent trace runtime, "
            "bridge/decoder path, inference API, and baseline-compatible reports."
        ),
        "architecture": {
            "stage_1": "load/tokenize prompts through a saved vocabulary",
            "stage_2": "encode prompts with the M26 tiny language backbone",
            "stage_3": "emit recurrent logebonic loose-bridi traces with the M27 Coconut runtime",
            "stage_4": "optionally route emitted trace slots through the M27 relevance runtime",
            "stage_5": "answer through the M26 choked trace-language bridge",
            "stage_6": "save/reload model weights, config, vocabulary, answer labels, and trace schema",
            "stage_7": "report model inference, checkpoint, and trace-causality metrics as model artifacts",
        },
        "promotion_basis": [
            "checkpoint_roundtrip_pass",
            "model_inference_api_pass",
            "strict_accuracy",
            "m27_promotion_candidate",
            "m28_trace_causality_delta",
            "m28_baseline_comparison_bundle_present",
        ],
    }
}


def m28_track_spec(track: str = "M28") -> dict[str, Any]:
    return deepcopy(M28_REGISTRY[track])


def m28_default_output_root(kind: str = "model") -> Path:
    return Path(M28_REGISTRY["M28"]["output_roots"][kind])
