from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    ds = sorted([p for p in root.iterdir() if p.is_dir()])
    return ds[-1] if ds else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M4 series, optionally preceded by M3.5 symmetry family.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--baseline-manifest", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--run-m3-5-first", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--l-output-root", type=Path, default=Path("runs/l_series/m4"))
    p.add_argument("--report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m4"))
    p.add_argument("--m3-5-l-output-root", type=Path, default=Path("runs/l_series/m3_5_symmetry"))
    p.add_argument("--m3-5-report-output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m3_5_symmetry"))
    p.add_argument("--arity-enforcement-mode", choices=("legacy_strict", "registry_strict", "crystallization"), default="crystallization")
    p.add_argument("--dynamic-arity-signatures", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--operator-arity-json", type=Path, default=None)
    p.add_argument("--default-relation-arity", type=int, default=2)
    p.add_argument("--run-operator-family-instrumentation", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline_manifest.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {args.baseline_manifest}")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    m3_5_report: str = ""

    if args.run_m3_5_first:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "run_m3_5_symmetry.py"),
            "--base-model",
            str(args.base_model),
            "--adapter",
            str(args.adapter),
            "--baseline-manifest",
            str(args.baseline_manifest),
            "--train-steps",
            str(int(args.train_steps)),
            "--dataset-size",
            str(int(args.dataset_size)),
            "--seed",
            str(int(args.seed)),
            "--l-output-root",
            str(args.m3_5_l_output_root),
            "--report-output-root",
            str(args.m3_5_report_output_root),
            "--arity-enforcement-mode",
            str(args.arity_enforcement_mode),
            "--default-relation-arity",
            str(int(args.default_relation_arity)),
        ]
        if args.dynamic_arity_signatures:
            cmd.append("--dynamic-arity-signatures")
        if args.operator_arity_json is not None:
            cmd.extend(["--operator-arity-json", str(args.operator_arity_json)])
        if args.local_files_only:
            cmd.append("--local-files-only")
        subprocess.run(cmd, check=True)
        latest = _latest_child_dir(args.m3_5_report_output_root)
        if latest is not None:
            m3_5_report = str((latest / "m3_5_symmetry_report.json").as_posix())

    m4_report_root = args.report_output_root / f"manual_{ts}"
    cmd_m4 = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_m3_plus_family.py"),
        "--base-model",
        str(args.base_model),
        "--adapter",
        str(args.adapter),
        "--baseline-manifest",
        str(args.baseline_manifest),
        "--train-steps",
        str(int(args.train_steps)),
        "--dataset-size",
        str(int(args.dataset_size)),
        "--seed",
        str(int(args.seed)),
        "--l-output-root",
        str(args.l_output_root),
        "--report-output-root",
        str(m4_report_root),
        "--arity-enforcement-mode",
        str(args.arity_enforcement_mode),
        "--default-relation-arity",
        str(int(args.default_relation_arity)),
        "--track-label",
        "M4",
    ]
    if args.dynamic_arity_signatures:
        cmd_m4.append("--dynamic-arity-signatures")
    if args.operator_arity_json is not None:
        cmd_m4.extend(["--operator-arity-json", str(args.operator_arity_json)])
    if args.local_files_only:
        cmd_m4.append("--local-files-only")
    subprocess.run(cmd_m4, check=True)

    latest_m4 = _latest_child_dir(m4_report_root)
    instrumentation_dir = ""
    if bool(args.run_operator_family_instrumentation) and latest_m4 is not None:
        m4_rep = latest_m4 / "m3_plus_report.json"
        checkpoint = ""
        if m4_rep.exists():
            payload = json.loads(m4_rep.read_text(encoding="utf-8"))
            rows = payload.get("rows", [])
            if isinstance(rows, list) and rows:
                last_ok = [r for r in rows if isinstance(r, dict) and r.get("status") == "ok"]
                if last_ok:
                    checkpoint = str(last_ok[-1].get("checkpoint_path", "")).strip()
        if checkpoint:
            instr_out = latest_m4 / "m4_operator_family_eval"
            cmd_instr = [
                sys.executable,
                str(Path(__file__).resolve().parent / "run_m4_operator_family_eval.py"),
                "--base-model",
                str(args.base_model),
                "--adapter",
                str(args.adapter),
                "--checkpoint",
                checkpoint,
                "--output-root",
                str(instr_out),
                "--seed",
                str(int(args.seed)),
            ]
            if args.local_files_only:
                cmd_instr.append("--local-files-only")
            subprocess.run(cmd_instr, check=True)
            instrumentation_dir = str(instr_out.as_posix())

    payload: dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "series_id": "M",
        "track": "M4",
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "ran_m3_5_first": bool(args.run_m3_5_first),
        "m3_5_report_json": m3_5_report,
        "m4_report_json": str((latest_m4 / "m3_plus_report.json").as_posix()) if latest_m4 is not None else "",
        "m4_operator_family_eval_dir": instrumentation_dir,
    }
    out = m4_report_root / "m4_series_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
