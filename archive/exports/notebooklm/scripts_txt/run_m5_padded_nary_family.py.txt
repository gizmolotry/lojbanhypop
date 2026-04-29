from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


@dataclass
class CellSpec:
    run_id: str
    name: str
    uniformity_weight: float
    grl_weight: float
    invariance_weight: float
    cpc_weight: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    return dirs[-1] if dirs else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M5 padded n-ary ablation family.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=24)
    p.add_argument("--semantic-dataset-size", type=int, default=180)
    p.add_argument("--winograd-pack-size", type=int, default=240)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--dataset-profile", type=str, default="semantic_bench_v1")
    p.add_argument("--difficulty-tier", type=str, default="all")
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--l-output-root", type=Path, default=Path("runs/m_series/m5_padded_nary"))
    p.add_argument("--report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m5_padded_nary"))
    p.add_argument("--local-files-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("M", args.l_output_root)
    assert_output_path_allowed("M", args.report_output_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    family_root = args.l_output_root / ts
    report_root = args.report_output_root / f"m5_padded_nary_{ts}"
    validate_series_outputs("M", [args.l_output_root, args.report_output_root], [family_root, report_root])
    family_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    trainer = Path(__file__).resolve().parent / "train_m5_padded_nary.py"
    cells = [
        CellSpec("M5.N0", "Padded core only", 0.0, 0.0, 0.0, 0.0),
        CellSpec("M5.N1", "Core + uniformity + GRL", 0.10, 0.10, 0.0, 0.0),
        CellSpec("M5.N2", "Add counterfactual invariance", 0.10, 0.10, 0.20, 0.0),
        CellSpec("M5.N3", "Full padded crucible", 0.10, 0.10, 0.20, 0.20),
    ]

    rows: list[dict[str, Any]] = []
    for cell in cells:
        run_root = family_root / cell.run_id.lower().replace(".", "_")
        cmd = [
            sys.executable,
            str(trainer),
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--output-root",
            str(run_root),
            "--train-steps",
            str(int(args.train_steps)),
            "--semantic-dataset-size",
            str(int(args.semantic_dataset_size)),
            "--winograd-pack-size",
            str(int(args.winograd_pack_size)),
            "--seed",
            str(int(args.seed)),
            "--dataset-profile",
            str(args.dataset_profile),
            "--difficulty-tier",
            str(args.difficulty_tier),
            "--layer-index",
            str(int(args.layer_index)),
            "--uniformity-weight",
            str(float(cell.uniformity_weight)),
            "--grl-weight",
            str(float(cell.grl_weight)),
            "--invariance-weight",
            str(float(cell.invariance_weight)),
            "--cpc-weight",
            str(float(cell.cpc_weight)),
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")

        rc = int(subprocess.run(cmd, check=False).returncode)
        latest = _latest_child_dir(run_root)
        summary_path = latest / "m5_padded_nary_summary.json" if latest is not None else None
        checkpoint_path = latest / "m5_padded_nary_checkpoint.pt" if latest is not None else None
        if rc != 0 or summary_path is None or not summary_path.exists():
            rows.append(
                {
                    "run_id": cell.run_id,
                    "name": cell.name,
                    "status": "failed",
                    "return_code": rc,
                    "run_dir": str(latest if latest is not None else run_root).replace("\\", "/"),
                }
            )
            continue
        summary = _read_json(summary_path)
        final_step = summary.get("final_step", {}) or {}
        eval_block = summary.get("eval", {}) or {}
        rows.append(
            {
                "run_id": cell.run_id,
                "name": cell.name,
                "status": "ok",
                "run_dir": str(latest).replace("\\", "/"),
                "checkpoint": str(checkpoint_path).replace("\\", "/") if checkpoint_path is not None and checkpoint_path.exists() else None,
                "final_step": final_step,
                "eval": eval_block,
            }
        )

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M5.padded_nary_family", "scripts/run_m5_padded_nary_family.py"),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "cells": rows,
    }
    (report_root / "m5_padded_nary_family_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# M5 Padded N-ary Family",
        "",
        "| Run | Name | Status | Wino Eval | Semantic Val | Top1 Share | Op Entropy |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        final_step = row.get("final_step", {}) or {}
        eval_block = row.get("eval", {}) or {}
        lines.append(
            f"| {row['run_id']} | {row['name']} | {row['status']} | "
            f"{(eval_block.get('winograd_eval', {}) or {}).get('accuracy', 'n/a')} | "
            f"{(eval_block.get('semantic_val', {}) or {}).get('accuracy', 'n/a')} | "
            f"{final_step.get('top1_op_share_batch', 'n/a')} | "
            f"{final_step.get('operator_entropy_batch', 'n/a')} |"
        )
    (report_root / "m5_padded_nary_family_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {report_root / 'm5_padded_nary_family_report.json'}")
    print(f"Wrote: {report_root / 'm5_padded_nary_family_report.md'}")


if __name__ == "__main__":
    main()
