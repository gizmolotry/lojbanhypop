from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Micro-SFT DroPE recalibration (50-100 steps).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--adapter-init", type=Path, default=None, help="Optional adapter to continue from.")
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--per-device-batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script = Path(__file__).resolve().parent / "train_lora.py"
    cmd = [
        sys.executable,
        str(script),
        "--base-model",
        args.base_model,
        "--dataset",
        str(args.dataset),
        "--output-dir",
        str(args.output_dir),
        "--max-steps",
        str(args.max_steps),
        "--epochs",
        "1.0",
        "--per-device-batch-size",
        str(args.per_device_batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--lr",
        str(args.lr),
        "--max-length",
        str(args.max_length),
        "--seed",
        str(args.seed),
        "--disable-rope",
        "--trace-loss-multiplier",
        "1.0",
        "--answer-loss-multiplier",
        "1.0",
        "--prompt-loss-multiplier",
        "0.0",
    ]
    if args.adapter_init is not None:
        cmd.extend(["--adapter-init", str(args.adapter_init)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    rc = int(subprocess.call(cmd))
    if rc != 0:
        raise SystemExit(rc)
    print(f"Recalibrated DroPE adapter: {args.output_dir}")


if __name__ == "__main__":
    main()
