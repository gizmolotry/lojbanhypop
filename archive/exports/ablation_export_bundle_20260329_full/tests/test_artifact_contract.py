from __future__ import annotations

import pytest

from lojban_evolution.artifact_contract import (
    ArtifactValidationError,
    VARIABLE_TOKEN_DISTRIBUTION_SIZE,
    validate_artifact_contract_v1,
)


def _base_artifact() -> dict:
    return {
        "artifact_contract_version": "1.0",
        "artifact_kind": "grounded_reasoner_train",
        "run": {
            "run_id": "run-001",
            "pipeline": "train_grounded_reasoner",
            "generated_utc": "2026-03-03T00:00:00+00:00",
        },
        "telemetry": {
            "system_1_topology": {
                "arity_violation_rate": 0.01,
                "surgery_trigger_count": 3,
                "variable_token_distribution": [0.0] * VARIABLE_TOKEN_DISTRIBUTION_SIZE,
            },
            "system_2_geometry": {
                "ce_loss_final": 0.5,
                "cross_attention_gain": 0.2,
            },
            "logical_accuracy": {
                "value": 0.8,
            },
        },
    }


def test_artifact_contract_accepts_distribution_length_1995() -> None:
    artifact = _base_artifact()
    validate_artifact_contract_v1(artifact)


def test_artifact_contract_missing_required_fields_raises_fatal() -> None:
    artifact = _base_artifact()
    del artifact["telemetry"]["system_1_topology"]["variable_token_distribution"]
    with pytest.raises(ArtifactValidationError, match="variable_token_distribution"):
        validate_artifact_contract_v1(artifact)


def test_artifact_contract_rejects_mismatched_kind_and_pipeline() -> None:
    artifact = _base_artifact()
    artifact["artifact_kind"] = "manifold_eval"
    with pytest.raises(ArtifactValidationError, match="incompatible with artifact_kind"):
        validate_artifact_contract_v1(artifact)


def test_artifact_contract_rejects_out_of_range_rates() -> None:
    artifact = _base_artifact()
    artifact["telemetry"]["system_1_topology"]["arity_violation_rate"] = 1.01
    with pytest.raises(ArtifactValidationError, match="arity_violation_rate"):
        validate_artifact_contract_v1(artifact)

    artifact = _base_artifact()
    artifact["telemetry"]["logical_accuracy"]["value"] = -0.01
    with pytest.raises(ArtifactValidationError, match="logical_accuracy.value"):
        validate_artifact_contract_v1(artifact)
