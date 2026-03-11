from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    seed_offset: int
    extra_args: list[str]


def _specs() -> list[RunSpec]:
    return [
        RunSpec("R1", 0, []),
        RunSpec("R2", 1, ["--align-weight", "0.0"]),
        RunSpec("R3", 2, ["--margin-weight", "0.0"]),
        RunSpec("R4", 3, ["--margin", "0.4", "--margin-weight", "1.0"]),
        RunSpec("R5", 4, ["--runtime-gate-cap", "0.0"]),
        RunSpec("R6", 5, ["--align-weight", "1.5", "--max-nodes", "16"]),
        RunSpec("R7", 6, ["--runtime-enable-min-acc-gain", "0.0"]),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M3.15 rotary coconut seven-run ablation sweep (R1..R7).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven"))
    p.add_argument("--group-id", type=str, default="", help="Parent folder for this sweep. Defaults to UTC timestamp.")
    p.add_argument("--seed", type=int, default=42, help="Base seed; each run uses seed + offset.")
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--eval-size", type=int, default=500)
    p.add_argument("--pack-jsonl", type=Path, default=None)
    p.add_argument("--strict-balance", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve().parent / "run_m3_15_rotary_coconut.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    group_id = args.group_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    group_dir = args.output_root / group_id
    group_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--base-model",
        str(args.base_model),
        "--adapter",
        str(args.adapter),
        "--checkpoint",
        str(args.checkpoint),
        "--train-steps",
        str(int(args.train_steps)),
        "--eval-size",
        str(int(args.eval_size)),
        "--output-root",
        str(group_dir),
    ]
    if args.pack_jsonl is not None:
        common.extend(["--pack-jsonl", str(args.pack_jsonl)])
    if bool(args.strict_balance):
        common.append("--strict-balance")
    else:
        common.append("--no-strict-balance")
    if args.local_files_only:
        common.append("--local-files-only")

    rows: list[dict[str, Any]] = []
    for spec in _specs():
        run_seed = int(args.seed) + int(spec.seed_offset)
        cmd = [
            sys.executable,
            str(script_path),
            *common,
            "--seed",
            str(run_seed),
            "--run-id",
            spec.run_id,
            *spec.extra_args,
        ]
        rc = int(subprocess.run(cmd, check=False).returncode)
        out_dir = group_dir / spec.run_id
        rows.append(
            {
                "run_id": spec.run_id,
                "seed": run_seed,
                "status": "ok" if rc == 0 else "failed",
                "return_code": rc,
                "output_dir": str(out_dir).replace("\\", "/"),
            }
        )

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "track": "M3.15",
        "sweep": "rotary_coconut_seven",
        "group_id": group_id,
        "root_dir": str(group_dir).replace("\\", "/"),
        "runs": rows,
    }
    manifest_path = group_dir / "m3_15_rotary_coconut_seven_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    for row in rows:
        print(f"{row['run_id']}: rc={row['return_code']} out={row['output_dir']}")


if __name__ == "__main__":
    main()
