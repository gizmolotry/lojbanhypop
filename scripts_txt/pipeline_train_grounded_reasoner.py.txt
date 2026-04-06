from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from lojban_evolution.artifact_contract import (
    ARTIFACT_CONTRACT_VERSION,
    VARIABLE_TOKEN_DISTRIBUTION_SIZE,
    write_validated_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical training pipeline artifact writer for Artifact Contract v1."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arity-violation-rate", type=float, required=True)
    parser.add_argument("--surgery-trigger-count", type=int, required=True)
    parser.add_argument("--ce-loss-final", type=float, required=True)
    parser.add_argument("--cross-attention-gain", type=float, required=True)
    parser.add_argument("--logical-accuracy", type=float, required=True)
    parser.add_argument(
        "--variable-token-distribution-json",
        type=Path,
        required=True,
        help="Path to a JSON list with exactly 1995 numeric entries.",
    )
    return parser.parse_args()


def _load_variable_distribution(path: Path) -> list[Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Variable token distribution file must contain a JSON list.")
    if len(raw) != VARIABLE_TOKEN_DISTRIBUTION_SIZE:
        raise ValueError(
            "Variable token distribution must contain exactly "
            f"{VARIABLE_TOKEN_DISTRIBUTION_SIZE} entries."
        )
    return raw


def main() -> None:
    args = parse_args()
    variable_distribution = _load_variable_distribution(args.variable_token_distribution_json)
    artifact = {
        "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
        "artifact_kind": "grounded_reasoner_train",
        "run": {
            "run_id": args.run_id,
            "pipeline": "train_grounded_reasoner",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "telemetry": {
            "system_1_topology": {
                "arity_violation_rate": args.arity_violation_rate,
                "surgery_trigger_count": args.surgery_trigger_count,
                "variable_token_distribution": variable_distribution,
            },
            "system_2_geometry": {
                "ce_loss_final": args.ce_loss_final,
                "cross_attention_gain": args.cross_attention_gain,
            },
            "logical_accuracy": {
                "value": args.logical_accuracy,
            },
        },
    }

    validated = write_validated_artifact(args.output, artifact)
    print(f"Wrote validated artifact: {args.output}")
    print(
        "Telemetry summary: "
        f"arity_violation_rate={validated['telemetry']['system_1_topology']['arity_violation_rate']}, "
        f"surgery_trigger_count={validated['telemetry']['system_1_topology']['surgery_trigger_count']}, "
        f"ce_loss_final={validated['telemetry']['system_2_geometry']['ce_loss_final']}, "
        f"cross_attention_gain={validated['telemetry']['system_2_geometry']['cross_attention_gain']}, "
        f"logical_accuracy={validated['telemetry']['logical_accuracy']['value']}"
    )


if __name__ == "__main__":
    main()
