from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class CellSpec:
    run_id: str
    name: str
    swap_policy: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    ds = sorted([p for p in root.iterdir() if p.is_dir()])
    return ds[-1] if ds else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M3.5 symmetry ablation family (A/B/C).")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--l-output-root", type=Path, default=Path("runs/l_series/m3_5_symmetry"))
    p.add_argument("--report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_5_symmetry"))
    p.add_argument("--arity-enforcement-mode", choices=("legacy_strict", "registry_strict", "crystallization"), default="crystallization")
    p.add_argument("--dynamic-arity-signatures", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--operator-arity-json", type=Path, default=None)
    p.add_argument("--default-relation-arity", type=int, default=2)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline_manifest.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {args.baseline_manifest}")
    if str(args.arity_enforcement_mode) == "registry_strict" and args.operator_arity_json is None:
        raise ValueError("arity_enforcement_mode=registry_strict requires --operator-arity-json with provenance=observed_usage")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    family_root = args.l_output_root / ts
    family_root.mkdir(parents=True, exist_ok=True)

    cells = [
        CellSpec("M3.5.A", "Asymmetric Baseline (forced non-commutativity)", "forced_asymmetric"),
        CellSpec("M3.5.B", "Symmetry-aware crucible (invariant vs foil)", "symmetry_aware"),
        CellSpec("M3.5.C", "Ablated anchor (swap-test disabled)", "disabled"),
    ]

    trainer = Path(__file__).resolve().parent / "train_l_series_mvs.py"
    rows: list[dict[str, Any]] = []
    resume_from: Path | None = None

    for cell in cells:
        run_root = family_root / cell.run_id.lower().replace(".", "_")
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
            "30",
            "--stage1-steps",
            "120",
            "--tier-a-lock-eps",
            "0.30",
            "--swap-policy",
            cell.swap_policy,
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
        if resume_from is not None:
            cmd.extend(["--resume", str(resume_from)])

        rc = int(subprocess.run(cmd, check=False).returncode)
        latest = _latest_child_dir(run_root)
        summary_path = (latest / "l_series_summary.json") if latest is not None else None
        if rc != 0 or latest is None or summary_path is None or not summary_path.exists():
            rows.append({"run_id": cell.run_id, "name": cell.name, "swap_policy": cell.swap_policy, "status": "failed", "return_code": rc})
            continue
        s = _read_json(summary_path)
        final = s.get("final_step", {})
        swap_metrics = s.get("swap_metrics", {})
        ckpt = latest / "l_series_checkpoint.pt"
        if ckpt.exists():
            resume_from = ckpt
        rows.append(
            {
                "run_id": cell.run_id,
                "name": cell.name,
                "swap_policy": cell.swap_policy,
                "status": "ok",
                "return_code": rc,
                "run_dir": str(latest).replace("\\", "/"),
                "summary_path": str(summary_path).replace("\\", "/"),
                "final_constraint_arity_strict": float(final.get("constraint_arity_strict", 1.0)),
                "final_constraint_scope": float(final.get("constraint_scope", 1.0)),
                "final_constraint_identity": float(final.get("constraint_identity", 1.0)),
                "arity_crystallization_rate": float(final.get("arity_crystallization_rate", 0.0)),
                "arity_mean_entropy": float(final.get("arity_mean_entropy", 0.0)),
                "arity_mean_mode_share": float(final.get("arity_mean_mode_share", 0.0)),
                "swap_active_count": int(swap_metrics.get("swap_active_count", 0)),
                "swap_counts": swap_metrics.get("swap_counts", {}),
            }
        )

    out_dir = args.report_output_root / f"m3_5_symmetry_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m3_5_symmetry_family",
        "series_id": "M",
        "track": "M3.5",
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": str(args.adapter),
            "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
            "arity_enforcement_mode": str(args.arity_enforcement_mode),
        },
        "rows": rows,
    }
    out_json = out_dir / "m3_5_symmetry_report.json"
    out_md = out_dir / "m3_5_symmetry_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# M3.5 Symmetry Report",
        "",
        "| run_id | policy | status | arity_strict | scope | identity | cryst_rate | mean_entropy | mode_share | run_dir |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            md.append(f"| `{r['run_id']}` | `{r['swap_policy']}` | `{r['status']}` |  |  |  |  |  |  |  |")
            continue
        md.append(
            f"| `{r['run_id']}` | `{r['swap_policy']}` | `ok` | {float(r['final_constraint_arity_strict']):.4f} | "
            f"{float(r['final_constraint_scope']):.4f} | {float(r['final_constraint_identity']):.4f} | "
            f"{float(r['arity_crystallization_rate']):.4f} | {float(r['arity_mean_entropy']):.4f} | "
            f"{float(r['arity_mean_mode_share']):.4f} | `{r['run_dir']}` |"
        )
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
