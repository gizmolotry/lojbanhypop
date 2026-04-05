from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


ARTIFACT_CONTRACT_VERSION = "1.0"
VARIABLE_TOKEN_DISTRIBUTION_SIZE = 1995
ALLOWED_ARTIFACT_KINDS = {"grounded_reasoner_train", "manifold_eval"}
ALLOWED_PIPELINES = {"train_grounded_reasoner", "eval_manifold"}
KIND_PIPELINE_COMPATIBILITY = {
    "grounded_reasoner_train": "train_grounded_reasoner",
    "manifold_eval": "eval_manifold",
}


class ArtifactValidationError(ValueError):
    pass


def validate_artifact_contract_v1(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError("Artifact payload must be a mapping.")

    artifact = deepcopy(dict(payload))
    _expect_exact_keys(
        artifact,
        "artifact root",
        {"artifact_contract_version", "artifact_kind", "run", "telemetry"},
    )
    _expect_string_eq(artifact, "artifact_contract_version", ARTIFACT_CONTRACT_VERSION)
    _expect_string_in(artifact, "artifact_kind", ALLOWED_ARTIFACT_KINDS)

    run = _expect_mapping(artifact, "run")
    _expect_exact_keys(run, "run", {"run_id", "pipeline", "generated_utc"})
    _expect_nonempty_string(run, "run_id")
    pipeline = _expect_string_in(run, "pipeline", ALLOWED_PIPELINES)
    _expect_nonempty_string(run, "generated_utc")
    expected_pipeline = KIND_PIPELINE_COMPATIBILITY[artifact["artifact_kind"]]
    if pipeline != expected_pipeline:
        raise ArtifactValidationError(
            "run.pipeline is incompatible with artifact_kind: "
            f"artifact_kind='{artifact['artifact_kind']}' expects pipeline='{expected_pipeline}'."
        )

    telemetry = _expect_mapping(artifact, "telemetry")
    _expect_exact_keys(
        telemetry,
        "telemetry",
        {"system_1_topology", "system_2_geometry", "logical_accuracy"},
    )

    system_1 = _expect_mapping(telemetry, "system_1_topology")
    _expect_exact_keys(
        system_1,
        "system_1_topology",
        {"arity_violation_rate", "surgery_trigger_count", "variable_token_distribution"},
    )
    arity_violation_rate = _expect_finite_number(system_1, "arity_violation_rate")
    if not (0.0 <= arity_violation_rate <= 1.0):
        raise ArtifactValidationError("system_1_topology.arity_violation_rate must be in [0.0, 1.0].")
    surgery_trigger_count = system_1["surgery_trigger_count"]
    if not isinstance(surgery_trigger_count, int) or isinstance(surgery_trigger_count, bool):
        raise ArtifactValidationError("system_1_topology.surgery_trigger_count must be an integer.")
    if surgery_trigger_count < 0:
        raise ArtifactValidationError("system_1_topology.surgery_trigger_count must be >= 0.")
    variable_token_distribution = system_1["variable_token_distribution"]
    if not isinstance(variable_token_distribution, list):
        raise ArtifactValidationError("system_1_topology.variable_token_distribution must be a list.")
    if len(variable_token_distribution) != VARIABLE_TOKEN_DISTRIBUTION_SIZE:
        raise ArtifactValidationError(
            "system_1_topology.variable_token_distribution must contain exactly "
            f"{VARIABLE_TOKEN_DISTRIBUTION_SIZE} entries."
        )
    for index, value in enumerate(variable_token_distribution):
        if not _is_finite_number(value):
            raise ArtifactValidationError(
                f"system_1_topology.variable_token_distribution[{index}] must be a finite number."
            )
        if float(value) < 0.0:
            raise ArtifactValidationError(
                f"system_1_topology.variable_token_distribution[{index}] must be >= 0.0."
            )

    system_2 = _expect_mapping(telemetry, "system_2_geometry")
    _expect_exact_keys(system_2, "system_2_geometry", {"ce_loss_final", "cross_attention_gain"})
    ce_loss_final = _expect_finite_number(system_2, "ce_loss_final")
    if ce_loss_final < 0.0:
        raise ArtifactValidationError("system_2_geometry.ce_loss_final must be >= 0.0.")
    _expect_finite_number(system_2, "cross_attention_gain")

    logical_accuracy = _expect_mapping(telemetry, "logical_accuracy")
    _expect_exact_keys(logical_accuracy, "logical_accuracy", {"value"})
    logical_accuracy_value = _expect_finite_number(logical_accuracy, "value")
    if not (0.0 <= logical_accuracy_value <= 1.0):
        raise ArtifactValidationError("logical_accuracy.value must be in [0.0, 1.0].")

    return artifact


def load_artifact(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_artifact_contract_v1(data)


def write_validated_artifact(path: str | Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    validated = validate_artifact_contract_v1(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(validated, indent=2), encoding="utf-8")
    return validated


def _expect_exact_keys(obj: Mapping[str, Any], section: str, expected: set[str]) -> None:
    keys = set(obj.keys())
    if keys != expected:
        missing = sorted(expected - keys)
        extra = sorted(keys - expected)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        detail_text = ", ".join(details) if details else "unexpected key mismatch"
        raise ArtifactValidationError(f"{section} keys mismatch: {detail_text}.")


def _expect_mapping(obj: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = obj.get(field)
    if not isinstance(value, Mapping):
        raise ArtifactValidationError(f"{field} must be an object.")
    return value


def _expect_nonempty_string(obj: Mapping[str, Any], field: str) -> str:
    value = obj.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ArtifactValidationError(f"{field} must be a non-empty string.")
    return value


def _expect_string_eq(obj: Mapping[str, Any], field: str, expected: str) -> str:
    value = _expect_nonempty_string(obj, field)
    if value != expected:
        raise ArtifactValidationError(f"{field} must equal '{expected}'.")
    return value


def _expect_string_in(obj: Mapping[str, Any], field: str, allowed: set[str]) -> str:
    value = _expect_nonempty_string(obj, field)
    if value not in allowed:
        allowed_values = ", ".join(sorted(allowed))
        raise ArtifactValidationError(f"{field} must be one of: {allowed_values}.")
    return value


def _expect_finite_number(obj: Mapping[str, Any], field: str) -> float:
    value = obj.get(field)
    if not _is_finite_number(value):
        raise ArtifactValidationError(f"{field} must be a finite number.")
    return float(value)


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
