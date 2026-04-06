from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class Variant:
    name: str
    phase5: bool
    weights: Dict[str, float]


PHASE5_WEIGHT_DEFAULTS: Dict[str, float] = {
    "semantic_unambiguity_weight": 0.05,
    "compositional_consistency_weight": 0.05,
    "roundtrip_consistency_weight": 0.05,
    "coverage_regularization_weight": 0.01,
    "compression_regularization_weight": 0.01,
    "trajectory_balance_weight": 0.02,
    "embedding_anchor_weight": 0.02,
}


def variants(include_only: bool) -> List[Variant]:
    out = [
        Variant(name="baseline_no_phase5", phase5=False, weights={}),
        Variant(name="phase5_full", phase5=True, weights=dict(PHASE5_WEIGHT_DEFAULTS)),
    ]
    for key in PHASE5_WEIGHT_DEFAULTS:
        w = dict(PHASE5_WEIGHT_DEFAULTS)
        w[key] = 0.0
        out.append(Variant(name=f"ablate_{key}", phase5=True, weights=w))
    if include_only:
        for key in PHASE5_WEIGHT_DEFAULTS:
            w = {k: 0.0 for k in PHASE5_WEIGHT_DEFAULTS}
            w[key] = PHASE5_WEIGHT_DEFAULTS[key]
            out.append(Variant(name=f"only_{key}", phase5=True, weights=w))
    return out


def read_last_trainer_metrics(output_dir: Path) -> Optional[Dict[str, float]]:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    history = state.get("log_history")
    if not isinstance(history, list) or not history:
        return None
    keys = [
        "loss",
        "ce_loss",
        "semantic_unambiguity_loss",
        "compositional_consistency_loss",
        "roundtrip_consistency_loss",
        "coverage_regularization_loss",
        "compression_regularization_loss",
        "trajectory_balance_loss",
        "embedding_anchor_loss",
        "total_loss",
    ]
    for row in reversed(history):
        if not isinstance(row, dict):
            continue
        out: Dict[str, float] = {}
        for k in keys:
            if k in row:
                try:
                    out[k] = float(row[k])
                except Exception:
                    continue
        if out:
            return out
    return None


def write_summary_md(path: Path, manifest: Dict[str, object]) -> None:
    rows = manifest.get("variants", [])
    if not isinstance(rows, list):
        rows = []
    lines: List[str] = []
    lines.append("# Phase-5 Training Ablation Summary")
    lines.append("")
    lines.append(f"- base_model: `{manifest.get('base_model', '')}`")
    lines.append(f"- dataset: `{manifest.get('dataset', '')}`")
    lines.append(f"- variant_count: `{len(rows)}`")
    lines.append("")
    lines.append(
        "| variant | status | return_code | phase5 | semantic | compositional | roundtrip | coverage | compression | trajectory | anchor |"
    )
    lines.append("|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        m = row.get("last_metrics", {}) if isinstance(row.get("last_metrics", {}), dict) else {}
        lines.append(
            "| `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` | `{}` |".format(
                row.get("name", ""),
                row.get("status", ""),
                row.get("return_code", ""),
                str(bool(row.get("phase5", False))).lower(),
                f"{m.get('semantic_unambiguity_loss', '')}",
                f"{m.get('compositional_consistency_loss', '')}",
                f"{m.get('roundtrip_consistency_loss', '')}",
                f"{m.get('coverage_regularization_loss', '')}",
                f"{m.get('compression_regularization_loss', '')}",
                f"{m.get('trajectory_balance_loss', '')}",
                f"{m.get('embedding_anchor_loss', '')}",
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline vs Phase-5 LoRA training ablations.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("runs/phase5_train_ablation"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually run training for each variant")
    parser.add_argument("--include-only-variants", action="store_true")
    parser.add_argument("--phase5-air-gapped-oracle", action="store_true")
    parser.add_argument("--phase5-freeze-symbolic-embeddings", action="store_true")
    parser.add_argument("--phase5-reward-temperature-beta", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_device_batch_size < 2:
        print("Bumping --per-device-batch-size to 2 to satisfy Phase-5 sparsity guardrails.")
        args.per_device_batch_size = 2
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    root = args.output_root / ts
    root.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "dataset": str(args.dataset),
        "variants": [],
    }

    for v in variants(include_only=args.include_only_variants):
        out_dir = root / v.name
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "train_lora.py"),
            "--base-model",
            args.base_model,
            "--dataset",
            str(args.dataset),
            "--output-dir",
            str(out_dir),
            "--epochs",
            str(args.epochs),
            "--max-length",
            str(args.max_length),
            "--per-device-batch-size",
            str(args.per_device_batch_size),
            "--grad-accum",
            str(args.grad_accum),
            "--lr",
            str(args.lr),
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")
        if v.phase5:
            cmd.append("--phase5-objectives")
            if args.phase5_air_gapped_oracle:
                cmd.append("--phase5-air-gapped-oracle")
            if args.phase5_freeze_symbolic_embeddings:
                cmd.append("--phase5-freeze-symbolic-embeddings")
            cmd.extend(["--phase5-reward-temperature-beta", str(args.phase5_reward_temperature_beta)])
            for key, value in v.weights.items():
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        row = {
            "name": v.name,
            "phase5": v.phase5,
            "weights": v.weights,
            "output_dir": str(out_dir),
            "command": cmd,
            "status": "planned",
            "return_code": None,
            "last_metrics": None,
        }

        if args.execute:
            rc = subprocess.call(cmd)
            row["return_code"] = int(rc)
            row["status"] = "ok" if rc == 0 else "failed"
            if rc == 0:
                row["last_metrics"] = read_last_trainer_metrics(out_dir)
        manifest["variants"].append(row)

    out_file = root / "ablation_manifest.json"
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_file = root / "summary.md"
    write_summary_md(summary_file, manifest)
    print(f"Wrote: {out_file}")
    print(f"Wrote: {summary_file}")
    for r in manifest["variants"]:
        print(f"{r['name']}: {r['status']}")


if __name__ == "__main__":
    main()
