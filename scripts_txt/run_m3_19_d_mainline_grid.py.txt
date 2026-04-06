from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


GRID = {
    "D0": {
        "supervision": "legacy_single_token",
        "d_train_objective": "margin",
        "residual_guard_threshold": 0.01,
        "hypothesis": "Anchor baseline with fluent, low-norm residual steering.",
    },
    "D1": {
        "supervision": "rich_4bucket_1to5",
        "d_train_objective": "continuation_ce",
        "residual_guard_threshold": 0.01,
        "hypothesis": "Richer gradients under a strict norm cap should improve routing without severance.",
    },
    "D2": {
        "supervision": "rich_4bucket_1to5",
        "d_train_objective": "continuation_ce",
        "residual_guard_threshold": 0.05,
        "hypothesis": "Allowing more residual volume should lift intervention and first-token accuracy.",
    },
    "D3": {
        "supervision": "rich_4bucket_1to5",
        "d_train_objective": "continuation_ce",
        "residual_guard_threshold": 0.10,
        "hypothesis": "Severance test near the physical limit of the decoder re-entry port.",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="M3.19 D-mainline ablation grid over supervision richness and residual guard threshold.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, default=Path("docs/baselines/m_series_bridge_baseline_manifest.json"))
    p.add_argument("--pack-size", type=int, default=512)
    p.add_argument("--pack-seed", type=int, default=19)
    p.add_argument("--train-steps", type=int, default=8)
    p.add_argument("--eval-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bottleneck-dim", type=int, default=64)
    p.add_argument("--num-return-tokens", type=int, default=3)
    p.add_argument("--max-logic-new-tokens", type=int, default=48)
    p.add_argument("--layer-index", type=int, default=12)
    p.add_argument("--relation-vocab", type=int, default=5)
    p.add_argument("--var-min-id", type=int, default=5)
    p.add_argument("--answer-weight", type=float, default=1.0)
    p.add_argument("--margin", type=float, default=0.2)
    p.add_argument("--return-norm-weight", type=float, default=0.01)
    p.add_argument("--continuation-target-max-tokens", type=int, default=5)
    p.add_argument("--residual-guard-weight", type=float, default=5.0)
    p.add_argument("--hybrid-token-weight", type=float, default=0.2)
    p.add_argument("--hybrid-residual-weight", type=float, default=1.0)
    p.add_argument("--hybrid-token-scale", type=float, default=0.15)
    p.add_argument("--continuation-eval-tokens", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", type=str, default="")
    p.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid"))
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return p.parse_args()


def _run(cmd: list[str], cwd: Path) -> None:
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pack_path = run_dir / "m3_19_resumption_pack.jsonl"
    pack_summary = run_dir / "m3_19_resumption_pack.summary.json"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_m3_19_resumption_pack.py"),
            "--size",
            str(int(args.pack_size)),
            "--seed",
            str(int(args.pack_seed)),
            "--output",
            str(pack_path),
            "--summary-output",
            str(pack_summary),
        ],
        cwd=repo_root,
    )

    summary: dict[str, Any] = {
        "track": "M3.19",
        "run_id": run_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pack_path": str(pack_path).replace("\\", "/"),
        "pack_summary": str(pack_summary).replace("\\", "/"),
        "cells": {},
    }

    for cell_id, cell_cfg in GRID.items():
        cell_run_id = f"{run_id}_{cell_id.lower()}"
        cell_root = run_dir / cell_id
        cell_root.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(repo_root / "scripts" / "run_m3_18_decoder_reentry_resume.py"),
            "--base-model",
            args.base_model,
            "--adapter",
            args.adapter,
            "--checkpoint",
            str(args.checkpoint),
            "--baseline-manifest",
            str(args.baseline_manifest),
            "--pack-jsonl",
            str(pack_path),
            "--train-steps",
            str(int(args.train_steps)),
            "--eval-size",
            str(int(args.eval_size)),
            "--lr",
            str(float(args.lr)),
            "--bottleneck-dim",
            str(int(args.bottleneck_dim)),
            "--num-return-tokens",
            str(int(args.num_return_tokens)),
            "--max-logic-new-tokens",
            str(int(args.max_logic_new_tokens)),
            "--layer-index",
            str(int(args.layer_index)),
            "--relation-vocab",
            str(int(args.relation_vocab)),
            "--var-min-id",
            str(int(args.var_min_id)),
            "--answer-weight",
            str(float(args.answer_weight)),
            "--margin",
            str(float(args.margin)),
            "--return-norm-weight",
            str(float(args.return_norm_weight)),
            "--d-train-objective",
            str(cell_cfg["d_train_objective"]),
            "--continuation-target-max-tokens",
            str(int(args.continuation_target_max_tokens)),
            "--residual-guard-threshold",
            str(float(cell_cfg["residual_guard_threshold"])),
            "--residual-guard-weight",
            str(float(args.residual_guard_weight)),
            "--hybrid-token-weight",
            str(float(args.hybrid_token_weight)),
            "--hybrid-residual-weight",
            str(float(args.hybrid_residual_weight)),
            "--hybrid-token-scale",
            str(float(args.hybrid_token_scale)),
            "--continuation-eval-tokens",
            str(int(args.continuation_eval_tokens)),
            "--seed",
            str(int(args.seed)),
            "--output-root",
            str(cell_root),
            "--run-id",
            cell_run_id,
        ]
        if bool(args.local_files_only):
            cmd.append("--local-files-only")
        _run(cmd, cwd=repo_root)

        report_path = cell_root / cell_run_id / "m3_18_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        d_train = report["cells"]["D"]["train"]
        d_metrics = report["cells"]["D"]["metrics"]
        summary["cells"][cell_id] = {
            **cell_cfg,
            "report_path": str(report_path).replace("\\", "/"),
            "train": d_train,
            "metrics": d_metrics,
        }

    summary_path = run_dir / "m3_19_grid_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# M3.19 D-Mainline Ablation Grid",
        "",
        "| Cell | Supervision | Guardrail | Acc | FTok | Fluency | Loop | Gold On-Off | Residual Norm | Overflow | Objective |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell_id in ("D0", "D1", "D2", "D3"):
        payload = summary["cells"][cell_id]
        metrics = payload["metrics"]
        train = payload["train"]
        md_lines.append(
            f"| {cell_id} | {payload['supervision']} | {payload['residual_guard_threshold']:.2f} | "
            f"{metrics.get('overall_accuracy', 0.0):.3f} | {metrics.get('resume_first_token_accuracy', 0.0):.3f} | "
            f"{metrics.get('english_fluency_score', 0.0):.3f} | {metrics.get('loop_rate', 0.0):.3f} | "
            f"{metrics.get('mean_intervention_delta_gold', 0.0):.4f} | {metrics.get('mean_residual_norm', 0.0):.4f} | "
            f"{metrics.get('residual_guard_overflow_rate', 0.0):.3f} | {train.get('train_objective', 'unknown')} |"
        )
    md_lines.extend(
        [
            "",
            "## Grid",
            "- D0: legacy single-token anchor.",
            "- D1: rich 4-bucket continuation CE under strict threshold 0.01.",
            "- D2: rich 4-bucket continuation CE with threshold 0.05.",
            "- D3: rich 4-bucket continuation CE with threshold 0.10.",
            "",
            f"Pack: `{pack_path.as_posix()}`",
        ]
    )
    (run_dir / "m3_19_grid_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"M3.19 grid complete: {run_dir}")


if __name__ == "__main__":
    main()
