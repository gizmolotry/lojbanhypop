from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


@dataclass
class CellResult:
    run_id: str
    m_series_alias: str
    name: str
    status: str
    return_code: Optional[int]
    run_dir: Optional[str]
    final_constraint_arity_strict: Optional[float]
    final_constraint_scope: Optional[float]
    final_constraint_identity: Optional[float]
    tier_b_enabled: Optional[bool]
    tier_c_enabled: Optional[bool]
    summary_path: Optional[str]
    checkpoint_path: Optional[str]
    lineage: Dict[str, Any]


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _latest_child_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    ds = [p for p in root.iterdir() if p.is_dir()]
    if not ds:
        return None
    return sorted(ds)[-1]


def _norm_path(value: str | Path | None) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run L6 multi-front ablation branch for L-series phase2.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--dataset-profile", type=str, default="legacy")
    p.add_argument("--difficulty-tier", type=str, default="all")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output-root", type=Path, default=Path("runs/l_series/l6_ablation"))
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("M", args.output_root)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / ts
    validate_series_outputs("M", [args.output_root], [out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)

    l_trainer = Path(__file__).resolve().parent / "train_l_series_mvs.py"

    specs = [
        (
            "L6-A",
            "M2.A",
            "Scope Drill + TierB Force + Soft Constraint Identity Audit",
            [
                "--scope-minimal-pairs", "500",
                "--scope-curriculum-ratio", "0.7",
                "--tier-a-lock-eps", "0.35",
                "--force-tier-b-after", "40",
                "--force-tier-c-after", "80",
                "--stage0-steps", "40",
                "--stage1-steps", "120",
            ],
        ),
        (
            "L6-B",
            "M2.B",
            "Scope Drill Only",
            [
                "--scope-minimal-pairs", "500",
                "--scope-curriculum-ratio", "0.7",
                "--tier-a-lock-eps", "0.02",
                "--stage0-steps", "40",
                "--stage1-steps", "120",
            ],
        ),
        (
            "L6-C",
            "M2.C",
            "TierB Force + Soft Constraint Audit Only",
            [
                "--scope-minimal-pairs", "0",
                "--tier-a-lock-eps", "0.35",
                "--force-tier-b-after", "40",
                "--force-tier-c-after", "80",
                "--stage0-steps", "40",
                "--stage1-steps", "120",
            ],
        ),
    ]

    rows: List[CellResult] = []
    for rid, m_alias, name, extra in specs:
        run_root = out_dir / rid.lower()
        cmd = [
            sys.executable,
            str(l_trainer),
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--train-steps",
            str(args.train_steps),
            "--dataset-size",
            str(args.dataset_size),
            "--dataset-profile",
            str(args.dataset_profile),
            "--difficulty-tier",
            str(args.difficulty_tier),
            "--seed",
            str(args.seed),
            "--output-root",
            str(run_root),
            *extra,
        ]
        if args.local_files_only:
            cmd.append("--local-files-only")

        if not args.execute:
            rows.append(
                CellResult(
                    rid,
                    m_alias,
                    name,
                    "planned",
                    None,
                    _norm_path(run_root),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    lineage_metadata(
                        "train",
                        checkpoint_in=None,
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                )
            )
            continue

        rc = int(subprocess.run(cmd, check=False).returncode)
        if rc != 0:
            rows.append(
                CellResult(
                    rid,
                    m_alias,
                    name,
                    "failed",
                    rc,
                    _norm_path(run_root),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    lineage_metadata(
                        "train",
                        checkpoint_in=None,
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                )
            )
            continue

        latest = _latest_child_dir(run_root)
        payload = _read_json((latest / "l_series_summary.json") if latest is not None else Path(""))
        if latest is None or payload is None:
            rows.append(
                CellResult(
                    rid,
                    m_alias,
                    name,
                    "failed",
                    -2,
                    _norm_path(run_root),
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    lineage_metadata(
                        "train",
                        checkpoint_in=None,
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                )
            )
            continue

        checkpoint = latest / "l_series_checkpoint.pt"
        if not checkpoint.exists():
            rows.append(
                CellResult(
                    rid,
                    m_alias,
                    name,
                    "failed",
                    -3,
                    _norm_path(latest),
                    None,
                    None,
                    None,
                    None,
                    None,
                    _norm_path(latest / "l_series_summary.json"),
                    None,
                    lineage_metadata(
                        "train",
                        checkpoint_in=None,
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                )
            )
            continue

        final = payload.get("final_step", {}) if isinstance(payload, dict) else {}
        rows.append(
            CellResult(
                rid,
                m_alias,
                name,
                "ok",
                rc,
                _norm_path(latest),
                float(final.get("constraint_arity_strict", 0.0)) if isinstance(final, dict) else None,
                float(final.get("constraint_scope", 0.0)) if isinstance(final, dict) else None,
                float(final.get("constraint_identity", 0.0)) if isinstance(final, dict) else None,
                bool(payload.get("tier_b_enabled", False)) if isinstance(payload, dict) else None,
                bool(payload.get("tier_c_enabled", False)) if isinstance(payload, dict) else None,
                _norm_path(latest / "l_series_summary.json"),
                _norm_path(checkpoint),
                lineage_metadata(
                    "train",
                    checkpoint_in=None,
                    checkpoint_out=_norm_path(checkpoint),
                    dataset_profile=str(args.dataset_profile),
                    difficulty_tier=str(args.difficulty_tier),
                ),
            )
        )

    out_json = out_dir / "l6_ablation_manifest.json"
    out_md = out_dir / "l6_ablation_manifest.md"
    output_paths: List[str | Path] = [out_dir, out_json, out_md]
    for row in rows:
        if row.run_dir:
            output_paths.append(row.run_dir)
        if row.summary_path:
            output_paths.append(row.summary_path)
        if row.checkpoint_path:
            output_paths.append(row.checkpoint_path)
    validate_series_outputs("M", [args.output_root], output_paths)
    out_json.write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "series": series_metadata("M", "M2", "scripts/run_l6_ablation_branch.py"),
                "declared_output_root": _norm_path(args.output_root),
                "lineage_defaults": lineage_metadata(
                    "train",
                    checkpoint_in=None,
                    checkpoint_out=None,
                    dataset_profile=str(args.dataset_profile),
                    difficulty_tier=str(args.difficulty_tier),
                ),
                "rows": [r.__dict__ for r in rows],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# L6 Ablation Branch",
        "",
        f"- series_id: `M`",
        f"- output_root: `{_norm_path(args.output_root)}`",
        f"- dataset_profile: `{args.dataset_profile}`",
        f"- difficulty_tier: `{args.difficulty_tier}`",
        "",
        "| id | m_alias | mode | dataset_profile | difficulty_tier | checkpoint_out | status | rc | arity_strict | scope | identity | tierB | tierC | run_dir |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.run_id}` | `{r.m_series_alias}` | `{r.lineage['mode']}` | `{r.lineage['dataset_profile']}` | `{r.lineage['difficulty_tier']}` | "
            f"`{r.lineage['checkpoint_out'] or ''}` | `{r.status}` | `{r.return_code}` | "
            f"{'' if r.final_constraint_arity_strict is None else f'{r.final_constraint_arity_strict:.3f}'} | "
            f"{'' if r.final_constraint_scope is None else f'{r.final_constraint_scope:.3f}'} | "
            f"{'' if r.final_constraint_identity is None else f'{r.final_constraint_identity:.3f}'} | "
            f"{r.tier_b_enabled} | {r.tier_c_enabled} | `{r.run_dir}` |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
