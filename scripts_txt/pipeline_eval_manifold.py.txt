from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from lojban_evolution.artifact_contract import (
    load_artifact,
    write_validated_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Canonical manifold evaluation pipeline for Artifact Contract v1."
    )
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True, help="Run id for manifold evaluation output.")
    parser.add_argument("--logical-accuracy", type=float, required=True)
    parser.add_argument(
        "--ce-loss-final",
        type=float,
        default=None,
        help="Optional override for system_2_geometry.ce_loss_final.",
    )
    parser.add_argument(
        "--cross-attention-gain",
        type=float,
        default=None,
        help="Optional override for system_2_geometry.cross_attention_gain.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = load_artifact(args.input_artifact)
    artifact["artifact_kind"] = "manifold_eval"
    artifact["run"] = {
        "run_id": args.run_id,
        "pipeline": "eval_manifold",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    artifact["telemetry"]["logical_accuracy"]["value"] = args.logical_accuracy
    if args.ce_loss_final is not None:
        artifact["telemetry"]["system_2_geometry"]["ce_loss_final"] = args.ce_loss_final
    if args.cross_attention_gain is not None:
        artifact["telemetry"]["system_2_geometry"]["cross_attention_gain"] = args.cross_attention_gain

    validated = write_validated_artifact(args.output, artifact)

    print(f"Wrote validated manifold eval artifact: {args.output}")
    print(
        "Telemetry summary: "
        f"logical_accuracy={validated['telemetry']['logical_accuracy']['value']}, "
        f"ce_loss_final={validated['telemetry']['system_2_geometry']['ce_loss_final']}, "
        f"cross_attention_gain={validated['telemetry']['system_2_geometry']['cross_attention_gain']}"
    )


if __name__ == "__main__":
    main()
