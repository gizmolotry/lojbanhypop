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

from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)


@dataclass
class CellSpec:
    run_id: str
    name: str
    diversification_mode: str
    lexical_weight: float
    lexical_grl_scale: float
    top1_penalty: float
    shadow_align_weight: float
    shadow_separate_weight: float
    shadow_temporal_weight: float


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    return dirs[-1] if dirs else None


def _norm_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M5 corrected auto-formalization ablation family.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, default=Path("docs/baselines/m_series_baseline_manifest.json"))
    p.add_argument("--train-steps", type=int, default=60)
    p.add_argument("--dataset-size", type=int, default=600)
    p.add_argument("--dataset-profile", type=str, default="semantic_bench_v1")
    p.add_argument("--difficulty-tier", type=str, default="all")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--relation-vocab", type=int, default=16)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--l-output-root", type=Path, default=Path("runs/l_series/m5_autoformalization"))
    p.add_argument("--report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization"))
    p.add_argument("--samples-per-family", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    baseline = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    checkpoint_in = Path(str(baseline["m_base"]["checkpoint"]))
    if not checkpoint_in.exists():
        raise FileNotFoundError(f"baseline checkpoint not found: {checkpoint_in}")

    assert_output_path_allowed("M", args.l_output_root)
    assert_output_path_allowed("M", args.report_output_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    l_family_root = args.l_output_root / ts
    report_root = args.report_output_root / f"m5_{ts}"
    validate_series_outputs("M", [args.l_output_root, args.report_output_root], [l_family_root, report_root])
    l_family_root.mkdir(parents=True, exist_ok=True)
    report_root.mkdir(parents=True, exist_ok=True)

    trainer = Path(__file__).resolve().parent / "train_l_series_mvs.py"
    family_eval = Path(__file__).resolve().parent / "run_m4_operator_family_eval.py"

    cells = [
        CellSpec(
            "M5.A",
            "Reuse-oriented control",
            "domain_reuse",
            lexical_weight=0.0,
            lexical_grl_scale=1.0,
            top1_penalty=0.55,
            shadow_align_weight=0.25,
            shadow_separate_weight=0.25,
            shadow_temporal_weight=0.10,
        ),
        CellSpec(
            "M5.B",
            "Selective lexical adversary + reuse",
            "domain_reuse",
            lexical_weight=0.20,
            lexical_grl_scale=1.0,
            top1_penalty=0.55,
            shadow_align_weight=0.25,
            shadow_separate_weight=0.25,
            shadow_temporal_weight=0.10,
        ),
        CellSpec(
            "M5.C",
            "Selective lexical adversary + family clustering",
            "family_cluster",
            lexical_weight=0.20,
            lexical_grl_scale=1.2,
            top1_penalty=0.70,
            shadow_align_weight=0.30,
            shadow_separate_weight=0.30,
            shadow_temporal_weight=0.15,
        ),
    ]

    rows: list[dict[str, Any]] = []
    for cell in cells:
        run_root = l_family_root / cell.run_id.lower().replace(".", "_")
        train_cmd = [
            sys.executable,
            str(trainer),
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--resume",
            str(checkpoint_in),
            "--train-steps",
            str(int(args.train_steps)),
            "--dataset-size",
            str(int(args.dataset_size)),
            "--dataset-profile",
            str(args.dataset_profile),
            "--difficulty-tier",
            str(args.difficulty_tier),
            "--seed",
            str(int(args.seed)),
            "--output-root",
            str(run_root),
            "--relation-vocab",
            str(int(args.relation_vocab)),
            "--scope-minimal-pairs",
            "1000",
            "--scope-curriculum-ratio",
            "0.70",
            "--force-tier-b-after",
            "10",
            "--force-tier-c-after",
            "20",
            "--stage0-steps",
            "20",
            "--stage1-steps",
            str(int(args.train_steps)),
            "--tier-a-lock-eps",
            "0.30",
            "--arity-enforcement-mode",
            "crystallization",
            "--default-relation-arity",
            "2",
            "--shadow-mode",
            "rolling",
            "--shadow-align-weight",
            str(float(cell.shadow_align_weight)),
            "--shadow-separate-weight",
            str(float(cell.shadow_separate_weight)),
            "--shadow-temporal-weight",
            str(float(cell.shadow_temporal_weight)),
            "--shadow-margin",
            "0.12",
            "--diversification-mode",
            str(cell.diversification_mode),
            "--diversification-weight",
            "0.25",
            "--diversification-domain-overlap-target",
            "0.55",
            "--diversification-top1-penalty",
            str(float(cell.top1_penalty)),
            "--diversification-cluster-centroids",
            "4",
            "--diversification-cluster-margin",
            "1.10",
            "--operator-lexical-adversary-weight",
            str(float(cell.lexical_weight)),
            "--operator-lexical-grl-scale",
            str(float(cell.lexical_grl_scale)),
            "--operator-lexical-hash-buckets",
            "64",
        ]
        if args.local_files_only:
            train_cmd.append("--local-files-only")

        rc = int(subprocess.run(train_cmd, check=False).returncode)
        latest = _latest_child_dir(run_root)
        summary_path = (latest / "l_series_summary.json") if latest is not None else None
        checkpoint_path = (latest / "l_series_checkpoint.pt") if latest is not None else None
        if rc != 0 or latest is None or summary_path is None or not summary_path.exists() or checkpoint_path is None or not checkpoint_path.exists():
            rows.append(
                {
                    "run_id": cell.run_id,
                    "name": cell.name,
                    "status": "failed",
                    "return_code": rc,
                    "run_dir": _norm_path(latest if latest is not None else run_root),
                    "lineage": lineage_metadata(
                        "train",
                        checkpoint_in=_norm_path(checkpoint_in),
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                }
            )
            continue

        eval_root = report_root / cell.run_id.lower().replace(".", "_") / "operator_family_eval"
        eval_cmd = [
            sys.executable,
            str(family_eval),
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--checkpoint",
            str(checkpoint_path),
            "--output-root",
            str(eval_root),
            "--seed",
            str(int(args.seed)),
            "--samples-per-family",
            str(int(args.samples_per_family)),
            "--relation-vocab",
            str(int(args.relation_vocab)),
        ]
        if args.local_files_only:
            eval_cmd.append("--local-files-only")
        eval_rc = int(subprocess.run(eval_cmd, check=False).returncode)

        summary = _read_json(summary_path)
        final = summary.get("final_step", {})
        lexical = summary.get("lexical_adversary_metrics", {})
        family_report_path = eval_root / "operator_family_report.json"
        lexical_probe_path = eval_root / "lexical_leakage_probe.json"
        family_report = _read_json(family_report_path) if family_report_path.exists() else {}
        lexical_probe = _read_json(lexical_probe_path) if lexical_probe_path.exists() else {}

        rows.append(
            {
                "run_id": cell.run_id,
                "name": cell.name,
                "status": "ok" if eval_rc == 0 else "partial",
                "return_code": rc,
                "eval_return_code": eval_rc,
                "run_dir": _norm_path(latest),
                "summary_path": _norm_path(summary_path),
                "checkpoint_path": _norm_path(checkpoint_path),
                "relation_vocab": int(args.relation_vocab),
                "final_constraint_arity_strict": float(final.get("constraint_arity_strict", 1.0)),
                "final_constraint_scope": float(final.get("constraint_scope", 1.0)),
                "final_constraint_identity": float(final.get("constraint_identity", 1.0)),
                "operator_entropy": float(final.get("operator_entropy", 0.0)),
                "operator_top1_share": float(final.get("operator_top1_share", 1.0)),
                "diversification_loss": float(final.get("diversification_loss", 0.0)),
                "lexical_adversary_loss": float(lexical.get("final_lexical_adversary_loss", 0.0)),
                "lexical_adversary_acc": float(lexical.get("final_lexical_adversary_acc", 0.0)),
                "predicate_family_score": float(family_report.get("predicate_family_score", 0.0)),
                "active_op_count": int(family_report.get("active_op_count", 0)),
                "family_eval_top1_op_share": float(family_report.get("top1_op_share", 1.0)),
                "operator_lexical_probe_acc": float(lexical_probe.get("operator_lexical_probe_acc", 0.0)),
                "family_eval_output_root": _norm_path(eval_root),
                "lineage": lineage_metadata(
                    "train",
                    checkpoint_in=_norm_path(checkpoint_in),
                    checkpoint_out=_norm_path(checkpoint_path),
                    dataset_profile=str(args.dataset_profile),
                    difficulty_tier=str(args.difficulty_tier),
                ),
            }
        )

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m5_autoformalization",
        "series": series_metadata("M", "M5", "scripts/run_m5_autoformalization.py"),
        "baseline_manifest": _norm_path(args.baseline_manifest),
        "baseline_id": str(baseline["baseline_id"]),
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": _norm_path(args.adapter),
            "checkpoint_in": _norm_path(checkpoint_in),
            "dataset_profile": str(args.dataset_profile),
            "difficulty_tier": str(args.difficulty_tier),
            "train_steps": int(args.train_steps),
            "dataset_size": int(args.dataset_size),
            "relation_vocab": int(args.relation_vocab),
        },
        "rows": rows,
    }
    report_path = report_root / "m5_autoformalization_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# M5 Auto-Formalization",
        "",
        f"- baseline_id: `{baseline['baseline_id']}`",
        f"- checkpoint_in: `{checkpoint_in.as_posix()}`",
        f"- dataset_profile: `{args.dataset_profile}`",
        f"- difficulty_tier: `{args.difficulty_tier}`",
        f"- relation_vocab: `{int(args.relation_vocab)}`",
        "",
        "| Run | Name | Status | Op Entropy | Top1 | Family Score | Lex Probe | Lex Adv Loss | Lex Adv Acc | Scope | Identity |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        md.append(
            f"| {row['run_id']} | {row['name']} | {row['status']} | "
            f"{float(row.get('operator_entropy', 0.0)):.4f} | {float(row.get('family_eval_top1_op_share', 1.0)):.4f} | "
            f"{float(row.get('predicate_family_score', 0.0)):.4f} | {float(row.get('operator_lexical_probe_acc', 0.0)):.4f} | "
            f"{float(row.get('lexical_adversary_loss', 0.0)):.4f} | {float(row.get('lexical_adversary_acc', 0.0)):.4f} | "
            f"{float(row.get('final_constraint_scope', 1.0)):.4f} | {float(row.get('final_constraint_identity', 1.0)):.4f} |"
        )
    (report_root / "m5_autoformalization_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()
