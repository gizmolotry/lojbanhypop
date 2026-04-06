from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage Phase-5 recovery run: CE warmup then Phase-5 continuation.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path("runs/phase5_two_stage_recovery"))
    p.add_argument("--epochs-stage1", type=float, default=1.0)
    p.add_argument("--epochs-stage2", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--per-device-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--logging-steps", type=int, default=1)
    p.add_argument("--save-steps", type=int, default=50)
    p.add_argument("--phase5-reward-temperature-beta", type=float, default=0.85)
    p.add_argument("--trajectory-balance-weight", type=float, default=1e-4)
    p.add_argument("--embedding-anchor-weight", type=float, default=0.02)
    p.add_argument("--compositional-anchors-file", type=Path, default=None)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def run_cmd(cmd: List[str]) -> int:
    return int(subprocess.call(cmd))


def main() -> None:
    args = parse_args()
    if args.per_device_batch_size < 2:
        print("Bumping --per-device-batch-size to 2 to satisfy Phase-5 sparsity guardrails.")
        args.per_device_batch_size = 2
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = args.output_root / ts
    root.mkdir(parents=True, exist_ok=True)
    train_script = str(Path(__file__).resolve().parent / "train_lora.py")

    stage1_out = root / "stage1_english_ce"
    stage2_out = root / "stage2_phase5"

    stage1_cmd = [
        sys.executable,
        train_script,
        "--base-model",
        args.base_model,
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(stage1_out),
        "--epochs",
        str(args.epochs_stage1),
        "--max-length",
        str(args.max_length),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--prompt-loss-multiplier",
        "0.0",
        "--trace-loss-multiplier",
        "0.0",
        "--answer-loss-multiplier",
        "1.0",
    ]
    if args.local_files_only:
        stage1_cmd.append("--local-files-only")
    if args.compositional_anchors_file is not None:
        stage1_cmd.extend(["--compositional-anchors-file", str(args.compositional_anchors_file)])

    stage2_cmd = [
        sys.executable,
        train_script,
        "--base-model",
        args.base_model,
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(stage2_out),
        "--adapter-init",
        str(stage1_out),
        "--epochs",
        str(args.epochs_stage2),
        "--max-length",
        str(args.max_length),
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--logging-steps",
        str(args.logging_steps),
        "--save-steps",
        str(args.save_steps),
        "--phase5-objectives",
        "--phase5-air-gapped-oracle",
        "--phase5-freeze-symbolic-embeddings",
        "--phase5-reward-temperature-beta",
        str(args.phase5_reward_temperature_beta),
        "--trajectory-balance-weight",
        str(args.trajectory_balance_weight),
        "--embedding-anchor-weight",
        str(args.embedding_anchor_weight),
    ]
    if args.local_files_only:
        stage2_cmd.append("--local-files-only")
    if args.compositional_anchors_file is not None:
        stage2_cmd.extend(["--compositional-anchors-file", str(args.compositional_anchors_file)])

    manifest: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "dataset": str(args.dataset),
        "stage1": {"output_dir": str(stage1_out), "command": stage1_cmd, "status": "planned", "return_code": None},
        "stage2": {"output_dir": str(stage2_out), "command": stage2_cmd, "status": "planned", "return_code": None},
    }

    if args.execute:
        rc1 = run_cmd(stage1_cmd)
        manifest["stage1"]["return_code"] = rc1
        manifest["stage1"]["status"] = "ok" if rc1 == 0 else "failed"
        if rc1 == 0:
            rc2 = run_cmd(stage2_cmd)
            manifest["stage2"]["return_code"] = rc2
            manifest["stage2"]["status"] = "ok" if rc2 == 0 else "failed"

    out = root / "two_stage_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")
    print(f"stage1: {manifest['stage1']['status']}")
    print(f"stage2: {manifest['stage2']['status']}")


if __name__ == "__main__":
    main()
