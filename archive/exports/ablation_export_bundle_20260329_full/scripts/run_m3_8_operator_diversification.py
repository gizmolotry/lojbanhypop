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
    validate_series_outputs,
)


@dataclass
class CellSpec:
    run_id: str
    name: str
    diversification_mode: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_m_baseline_manifest(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if str(payload.get("series_id", "")).strip().upper() != "M":
        raise ValueError("baseline_manifest.series_id must be 'M'")
    if not str(payload.get("baseline_id", "")).strip():
        raise ValueError("baseline_manifest.baseline_id is required")
    return payload


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    ds = sorted([p for p in root.iterdir() if p.is_dir()])
    return ds[-1] if ds else None


def _norm_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M3.8 operator diversification family (A/B/C).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--dataset-profile", type=str, default="diverse_v2")
    p.add_argument(
        "--difficulty-buckets",
        type=str,
        default="legacy,easy,medium,hard",
        help="Comma-separated buckets to run separately (legacy,easy,medium,hard).",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--l-output-root", type=Path, default=Path("runs/l_series/m3_8_diversification"))
    p.add_argument("--report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_8_diversification"))
    p.add_argument("--arity-enforcement-mode", choices=("legacy_strict", "registry_strict", "crystallization"), default="crystallization")
    p.add_argument("--dynamic-arity-signatures", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--operator-arity-json", type=Path, default=None)
    p.add_argument("--default-relation-arity", type=int, default=2)
    p.add_argument("--shadow-margin", type=float, default=0.10)
    p.add_argument("--stage0-steps", type=int, default=30)
    p.add_argument("--stage1-steps", type=int, default=120)
    p.add_argument("--diversification-weight", type=float, default=0.05)
    p.add_argument("--diversification-domain-overlap-target", type=float, default=0.45)
    p.add_argument("--diversification-top1-penalty", type=float, default=0.25)
    p.add_argument("--diversification-cluster-centroids", type=int, default=3)
    p.add_argument("--diversification-cluster-margin", type=float, default=0.80)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline_manifest.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {args.baseline_manifest}")
    baseline_manifest = _validate_m_baseline_manifest(args.baseline_manifest)
    if str(args.arity_enforcement_mode) == "registry_strict" and args.operator_arity_json is None:
        raise ValueError("arity_enforcement_mode=registry_strict requires --operator-arity-json with provenance=observed_usage")
    assert_output_path_allowed("M", args.l_output_root)
    assert_output_path_allowed("M", args.report_output_root)

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    family_root = args.l_output_root / ts
    validate_series_outputs("M", [args.l_output_root], [family_root])
    family_root.mkdir(parents=True, exist_ok=True)

    cells = [
        CellSpec("M3.8.A", "Operator entropy regularization", "entropy"),
        CellSpec("M3.8.B", "Domain reuse reward", "domain_reuse"),
        CellSpec("M3.8.C", "Family clustering objective", "family_cluster"),
    ]

    trainer = Path(__file__).resolve().parent / "train_l_series_mvs.py"
    rows: list[dict[str, Any]] = []
    buckets = [b.strip().lower() for b in str(args.difficulty_buckets).split(",") if b.strip()]
    buckets = [b for b in buckets if b in {"legacy", "easy", "medium", "hard", "all"}]
    if not buckets:
        buckets = ["all"]

    for bucket in buckets:
        for cell in cells:
            run_root = family_root / bucket / cell.run_id.lower().replace(".", "_")
            bucket_profile = "legacy" if bucket == "legacy" else str(args.dataset_profile)
            bucket_tier = "all" if bucket == "legacy" else str(bucket)
            cmd = [
                sys.executable,
                str(trainer),
                "--base-model",
                str(args.base_model),
                "--adapter",
                str(args.adapter),
                "--train-steps",
                str(int(args.train_steps)),
                "--dataset-size",
                str(int(args.dataset_size)),
                "--dataset-profile",
                str(bucket_profile),
                "--difficulty-tier",
                str(bucket_tier),
                "--seed",
                str(int(args.seed)),
                "--output-root",
                str(run_root),
                "--scope-minimal-pairs",
                "1000",
                "--scope-curriculum-ratio",
                "0.8",
                "--force-tier-b-after",
                "20",
                "--force-tier-c-after",
                "40",
                "--stage0-steps",
                str(int(args.stage0_steps)),
                "--stage1-steps",
                str(int(args.stage1_steps)),
                "--tier-a-lock-eps",
                "0.30",
                "--shadow-mode",
                "rolling",
                "--shadow-align-weight",
                "0.35",
                "--shadow-separate-weight",
                "0.35",
                "--shadow-temporal-weight",
                "0.20",
                "--shadow-margin",
                str(float(args.shadow_margin)),
                "--diversification-mode",
                cell.diversification_mode,
                "--diversification-weight",
                str(float(args.diversification_weight)),
                "--diversification-domain-overlap-target",
                str(float(args.diversification_domain_overlap_target)),
                "--diversification-top1-penalty",
                str(float(args.diversification_top1_penalty)),
                "--diversification-cluster-centroids",
                str(int(args.diversification_cluster_centroids)),
                "--diversification-cluster-margin",
                str(float(args.diversification_cluster_margin)),
                "--arity-enforcement-mode",
                str(args.arity_enforcement_mode),
                "--default-relation-arity",
                str(int(args.default_relation_arity)),
            ]
            if args.dynamic_arity_signatures:
                cmd.append("--dynamic-arity-signatures")
            if args.operator_arity_json is not None:
                cmd.extend(["--operator-arity-json", str(args.operator_arity_json)])
            if str(args.arity_enforcement_mode) == "registry_strict":
                cmd.append("--require-observed-registry-provenance")
            if args.local_files_only:
                cmd.append("--local-files-only")

            rc = int(subprocess.run(cmd, check=False).returncode)
            latest = _latest_child_dir(run_root)
            summary_path = (latest / "l_series_summary.json") if latest is not None else None
            if rc != 0 or latest is None or summary_path is None or not summary_path.exists():
                rows.append(
                    {
                        "run_id": cell.run_id,
                        "name": cell.name,
                        "difficulty_tier": bucket,
                        "dataset_profile": str(bucket_profile),
                        "diversification_mode": cell.diversification_mode,
                        "status": "failed",
                        "return_code": rc,
                        "run_dir": _norm_path(latest if latest else run_root),
                        "lineage": lineage_metadata(
                            "train",
                            checkpoint_in=None,
                            checkpoint_out=None,
                            dataset_profile=str(bucket_profile),
                            difficulty_tier=str(bucket_tier),
                        ),
                    }
                )
                continue
            s = _read_json(summary_path)
            final = s.get("final_step", {})
            shadow_metrics = s.get("shadow_metrics", {})
            div_metrics = s.get("diversification_metrics", {})
            checkpoint = latest / "l_series_checkpoint.pt"
            if not checkpoint.exists():
                rows.append(
                    {
                        "run_id": cell.run_id,
                        "name": cell.name,
                        "difficulty_tier": bucket,
                        "dataset_profile": str(bucket_profile),
                        "diversification_mode": cell.diversification_mode,
                        "status": "failed",
                        "return_code": -3,
                        "run_dir": _norm_path(latest),
                        "summary_path": _norm_path(summary_path),
                        "lineage": lineage_metadata(
                            "train",
                            checkpoint_in=None,
                            checkpoint_out=None,
                            dataset_profile=str(bucket_profile),
                            difficulty_tier=str(bucket_tier),
                        ),
                    }
                )
                continue
            rows.append(
                {
                    "run_id": cell.run_id,
                    "name": cell.name,
                    "difficulty_tier": bucket,
                    "dataset_profile": str(bucket_profile),
                    "diversification_mode": cell.diversification_mode,
                    "status": "ok",
                    "return_code": rc,
                    "run_dir": _norm_path(latest),
                    "summary_path": _norm_path(summary_path),
                    "checkpoint_path": _norm_path(checkpoint),
                    "final_constraint_arity_strict": float(final.get("constraint_arity_strict", 1.0)),
                    "final_constraint_scope": float(final.get("constraint_scope", 1.0)),
                    "final_constraint_identity": float(final.get("constraint_identity", 1.0)),
                    "shadow_loss": float(final.get("shadow_loss", 0.0)),
                    "operator_entropy": float(final.get("operator_entropy", 0.0)),
                    "operator_top1_share": float(final.get("operator_top1_share", 1.0)),
                    "diversification_loss": float(final.get("diversification_loss", 0.0)),
                    "diversification_entropy_loss": float(final.get("diversification_entropy_loss", 0.0)),
                    "diversification_domain_reuse_loss": float(final.get("diversification_domain_reuse_loss", 0.0)),
                    "diversification_family_cluster_loss": float(final.get("diversification_family_cluster_loss", 0.0)),
                    "shadow_metrics": shadow_metrics,
                    "diversification_metrics": div_metrics,
                    "lineage": lineage_metadata(
                        "train",
                        checkpoint_in=None,
                        checkpoint_out=_norm_path(checkpoint),
                        dataset_profile=str(bucket_profile),
                        difficulty_tier=str(bucket_tier),
                    ),
                }
            )

    out_dir = args.report_output_root / f"m3_8_diversification_{ts}"
    validate_series_outputs("M", [args.report_output_root], [out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m3_8_diversification_family",
        "series": series_metadata("M", "M3.8", "scripts/run_m3_8_operator_diversification.py"),
        "declared_l_output_root": _norm_path(args.l_output_root),
        "declared_report_output_root": _norm_path(args.report_output_root),
        "lineage_defaults": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile=str(args.dataset_profile),
            difficulty_tier="bucket_specific",
        ),
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": _norm_path(args.adapter),
            "baseline_manifest": _norm_path(args.baseline_manifest),
            "baseline_id": str(baseline_manifest.get("baseline_id", "")),
            "dataset_profile": str(args.dataset_profile),
            "difficulty_buckets": buckets,
            "arity_enforcement_mode": str(args.arity_enforcement_mode),
            "diversification_weight": float(args.diversification_weight),
        },
        "rows": rows,
    }
    out_json = out_dir / "m3_8_diversification_report.json"
    out_md = out_dir / "m3_8_diversification_report.md"
    report_output_paths: list[str | Path] = [out_dir, out_json, out_md, family_root]
    for row in rows:
        for key in ("run_dir", "summary_path", "checkpoint_path"):
            value = row.get(key)
            if value:
                report_output_paths.append(value)
    validate_series_outputs("M", [args.report_output_root, args.l_output_root], report_output_paths)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# M3.8 Diversification Report",
        "",
        f"- series_id: `{report['series']['series_id']}`",
        f"- track: `{report['series']['track']}`",
        "",
        "| run_id | mode | dataset_profile | difficulty_tier | checkpoint_out | diversification_mode | status | scope | identity | op_entropy | top1_share | div_loss | run_dir |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lineage = r.get("lineage", {})
        if r.get("status") != "ok":
            md.append(
                f"| `{r['run_id']}` | `{lineage.get('mode', '')}` | `{lineage.get('dataset_profile', '')}` | `{lineage.get('difficulty_tier', '')}` | "
                f"`{lineage.get('checkpoint_out', '') or ''}` | `{r['diversification_mode']}` | `{r['status']}` |  |  |  |  |  | `{r.get('run_dir', '')}` |"
            )
            continue
        md.append(
            f"| `{r['run_id']}` | `{lineage.get('mode', '')}` | `{lineage.get('dataset_profile', '')}` | `{lineage.get('difficulty_tier', '')}` | "
            f"`{lineage.get('checkpoint_out', '') or ''}` | `{r['diversification_mode']}` | `ok` | {float(r['final_constraint_scope']):.4f} | "
            f"{float(r['final_constraint_identity']):.4f} | {float(r['operator_entropy']):.4f} | "
            f"{float(r['operator_top1_share']):.4f} | {float(r['diversification_loss']):.4f} | `{r['run_dir']}` |"
        )
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
