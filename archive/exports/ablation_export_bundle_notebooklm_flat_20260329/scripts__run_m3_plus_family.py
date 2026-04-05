from __future__ import annotations

import argparse
import hashlib
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
class RunSpec:
    run_id: str
    name: str
    extra_args: list[str]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_child_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    ds = sorted([p for p in root.iterdir() if p.is_dir()])
    return ds[-1] if ds else None


def _norm_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    return str(value).replace("\\", "/")


def _validate_m_baseline_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {path}")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("baseline_manifest must be a JSON object")
    if str(payload.get("series_id", "")).strip().upper() != "M":
        raise ValueError("baseline_manifest.series_id must be 'M'")
    if not str(payload.get("baseline_id", "")).strip():
        raise ValueError("baseline_manifest.baseline_id is required")
    m_base = payload.get("m_base", {})
    if not isinstance(m_base, dict):
        raise ValueError("baseline_manifest.m_base must be an object")
    required = ("dataset", "constraints", "identity_reg", "curriculum", "optimizer")
    missing = [k for k in required if not str(m_base.get(k, "")).strip()]
    if missing:
        raise ValueError(f"baseline_manifest.m_base missing required keys: {missing}")
    upstream = payload.get("upstream_best", {})
    if not isinstance(upstream, dict):
        raise ValueError("baseline_manifest.upstream_best must be an object")
    if not str(upstream.get("j_series", "")).strip() or not str(upstream.get("l_series", "")).strip():
        raise ValueError("baseline_manifest.upstream_best must declare j_series and l_series")
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M3+ gate-driven family (M3.0..M3.4) and emit hypercube report.")
    p.add_argument("--base-model", required=True)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--train-steps", type=int, default=120)
    p.add_argument("--dataset-size", type=int, default=1000)
    p.add_argument("--dataset-profile", type=str, default="legacy")
    p.add_argument("--difficulty-tier", type=str, default="all")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--l-output-root", type=Path, default=Path("runs/l_series/m3_plus"))
    p.add_argument(
        "--report-output-root",
        type=Path,
        default=Path("runs/m_series/m3_plus"),
    )
    p.add_argument(
        "--j5-summary",
        type=Path,
        default=Path("runs/j_series/20260304_175527/j-5.json"),
        help="J-5 summary for foil metrics/polarity checks.",
    )
    p.add_argument(
        "--dynamic-arity-signatures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable dynamic n-ary signature enforcement in L-series trainer.",
    )
    p.add_argument(
        "--operator-arity-json",
        type=Path,
        default=None,
        help="Optional JSON map {relation_token_id: arity_n} passed through to L-series trainer.",
    )
    p.add_argument(
        "--default-relation-arity",
        type=int,
        default=2,
        help="Fallback arity when relation token is not present in operator arity registry.",
    )
    p.add_argument(
        "--arity-enforcement-mode",
        choices=("legacy_strict", "registry_strict", "crystallization"),
        default="crystallization",
        help="Arity policy for M-series runs. Default is crystallization (no strict registry enforcement).",
    )
    p.add_argument("--track-label", type=str, default="M3+", help="Track label written into report metadata.")
    p.add_argument(
        "--baseline-manifest",
        type=Path,
        required=True,
        help="Required M-series baseline manifest declaring inherited J/L best baselines.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_output_path_allowed("M", args.l_output_root)
    assert_output_path_allowed("M", args.report_output_root)
    baseline_manifest = _validate_m_baseline_manifest(args.baseline_manifest)
    baseline_manifest_sha1 = hashlib.sha1(args.baseline_manifest.read_bytes()).hexdigest()
    if str(args.arity_enforcement_mode) == "registry_strict" and args.operator_arity_json is None:
        raise ValueError("arity_enforcement_mode=registry_strict requires --operator-arity-json with provenance=observed_usage")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    family_root = args.l_output_root / ts
    validate_series_outputs("M", [args.l_output_root], [family_root])
    family_root.mkdir(parents=True, exist_ok=True)

    specs = [
        RunSpec(
            run_id="M3.0",
            name="Baseline Gate Run (L6-C + curriculum)",
            extra_args=[
                "--scope-minimal-pairs",
                "500",
                "--scope-curriculum-ratio",
                "0.7",
                "--tier-a-lock-eps",
                "0.35",
                "--force-tier-b-after",
                "40",
                "--force-tier-c-after",
                "80",
                "--stage0-steps",
                "40",
                "--stage1-steps",
                "120",
            ],
        ),
        RunSpec(
            run_id="M3.1",
            name="Binding Bootcamp (unbound focus)",
            extra_args=[
                "--scope-minimal-pairs",
                "1500",
                "--scope-curriculum-ratio",
                "0.95",
                "--tier-a-lock-eps",
                "0.35",
                "--force-tier-b-after",
                "30",
                "--force-tier-c-after",
                "70",
                "--stage0-steps",
                "40",
                "--stage1-steps",
                "160",
            ],
        ),
        RunSpec(
            run_id="M3.2",
            name="Depth Ramp (1->4 curriculum pressure)",
            extra_args=[
                "--scope-minimal-pairs",
                "2000",
                "--scope-curriculum-ratio",
                "0.90",
                "--tier-a-lock-eps",
                "0.30",
                "--force-tier-b-after",
                "30",
                "--force-tier-c-after",
                "60",
                "--stage0-steps",
                "30",
                "--stage1-steps",
                "140",
            ],
        ),
        RunSpec(
            run_id="M3.3",
            name="Truth Discrimination (foil-sensitive tier B)",
            extra_args=[
                "--scope-minimal-pairs",
                "1000",
                "--scope-curriculum-ratio",
                "0.70",
                "--weight-tier-b",
                "0.35",
                "--crispness-weight",
                "1.4",
                "--force-tier-b-after",
                "15",
                "--force-tier-c-after",
                "60",
                "--tier-a-lock-eps",
                "0.30",
            ],
        ),
        RunSpec(
            run_id="M3.4",
            name="Compression Activation (tier C pressure)",
            extra_args=[
                "--scope-minimal-pairs",
                "1000",
                "--scope-curriculum-ratio",
                "0.70",
                "--weight-tier-b",
                "0.30",
                "--weight-tier-c",
                "0.40",
                "--entropy-h-min",
                "0.95",
                "--force-tier-b-after",
                "10",
                "--force-tier-c-after",
                "20",
                "--tier-a-lock-eps",
                "0.30",
            ],
        ),
    ]

    trainer = Path(__file__).resolve().parent / "train_l_series_mvs.py"
    rows: list[dict[str, Any]] = []
    resume_from: Path | None = None

    for spec in specs:
        run_root = family_root / spec.run_id.lower().replace(".", "_")
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
            str(args.dataset_profile),
            "--difficulty-tier",
            str(args.difficulty_tier),
            "--seed",
            str(int(args.seed)),
            "--output-root",
            str(run_root),
            *spec.extra_args,
        ]
        if bool(args.dynamic_arity_signatures):
            cmd.append("--dynamic-arity-signatures")
        if args.operator_arity_json is not None:
            cmd.extend(["--operator-arity-json", str(args.operator_arity_json)])
        cmd.extend(["--default-relation-arity", str(int(args.default_relation_arity))])
        cmd.extend(["--arity-enforcement-mode", str(args.arity_enforcement_mode)])
        if str(args.arity_enforcement_mode) == "registry_strict":
            cmd.append("--require-observed-registry-provenance")
        if args.local_files_only:
            cmd.append("--local-files-only")
        if resume_from is not None:
            cmd.extend(["--resume", str(resume_from)])

        checkpoint_in = _norm_path(resume_from)
        rc = int(subprocess.run(cmd, check=False).returncode)
        latest = _latest_child_dir(run_root)
        summary_path = (latest / "l_series_summary.json") if latest is not None else None
        if rc != 0 or latest is None or summary_path is None or not summary_path.exists():
            rows.append(
                {
                    "run_id": spec.run_id,
                    "name": spec.name,
                    "status": "failed",
                    "return_code": rc,
                    "run_dir": _norm_path(latest if latest else run_root),
                    "lineage": lineage_metadata(
                        "train",
                        checkpoint_in=checkpoint_in,
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                }
            )
            continue

        s = _read_json(summary_path)
        final = s.get("final_step", {})
        checkpoint = latest / "l_series_checkpoint.pt"
        if not checkpoint.exists():
            rows.append(
                {
                    "run_id": spec.run_id,
                    "name": spec.name,
                    "status": "failed",
                    "return_code": -3,
                    "run_dir": _norm_path(latest),
                    "summary_path": _norm_path(summary_path),
                    "lineage": lineage_metadata(
                        "train",
                        checkpoint_in=checkpoint_in,
                        checkpoint_out=None,
                        dataset_profile=str(args.dataset_profile),
                        difficulty_tier=str(args.difficulty_tier),
                    ),
                }
            )
            continue
        resume_from = checkpoint if checkpoint.exists() else None

        rows.append(
            {
                "run_id": spec.run_id,
                "name": spec.name,
                "status": "ok",
                "return_code": rc,
                "run_dir": _norm_path(latest),
                "summary_path": _norm_path(summary_path),
                "checkpoint_path": _norm_path(checkpoint),
                "final_constraint_arity_strict": float(final.get("constraint_arity_strict", 1.0)),
                "final_constraint_scope": float(final.get("constraint_scope", 1.0)),
                "final_constraint_scope_unbound": float(final.get("constraint_scope_unbound", 1.0)),
                "final_constraint_identity": float(final.get("constraint_identity", 1.0)),
                "tier_b_enabled": bool(s.get("tier_b_enabled", False)),
                "tier_c_enabled": bool(s.get("tier_c_enabled", False)),
                "lineage": lineage_metadata(
                    "train",
                    checkpoint_in=checkpoint_in,
                    checkpoint_out=_norm_path(checkpoint),
                    dataset_profile=str(args.dataset_profile),
                    difficulty_tier=str(args.difficulty_tier),
                ),
            }
        )

    j5 = _read_json(args.j5_summary) if args.j5_summary.exists() else {}
    samples = j5.get("samples", []) if isinstance(j5, dict) else []
    true_gt_false_rate = 0.0
    if samples:
        hits = 0
        total = 0
        for x in samples:
            if not isinstance(x, dict):
                continue
            t = x.get("true_score")
            f = x.get("false_score")
            if isinstance(t, (int, float)) and isinstance(f, (int, float)):
                total += 1
                if float(t) > float(f):
                    hits += 1
        if total > 0:
            true_gt_false_rate = float(hits) / float(total)

    gate_target = next((r for r in rows if r.get("run_id") == "M3.4"), None)
    gate_row = gate_target if gate_target and gate_target.get("status") == "ok" else None
    gate_eval = {
        "gate_target_run_id": "M3.4",
        "gate_target_status": gate_target.get("status") if gate_target else "missing",
        "arity_feasible": bool(gate_row and float(gate_row["final_constraint_arity_strict"]) <= 0.01),
        "identity_guardrail_lte_0_05": bool(gate_row and float(gate_row["final_constraint_identity"]) <= 0.05),
        "unbound_solved_lt_0_02": bool(gate_row and float(gate_row["final_constraint_scope_unbound"]) < 0.02),
        "scope_gate_lt_0_10": bool(gate_row and float(gate_row["final_constraint_scope"]) < 0.10),
        "foil_metric_polarity_check_true_gt_false_rate": true_gt_false_rate,
        "accepted_foil_pair_accuracy": float(j5.get("metrics", {}).get("accepted_foil_pair_accuracy", j5.get("metrics", {}).get("foil_auc", 0.0))) if isinstance(j5, dict) else 0.0,
        "foil_auc_deprecated": float(j5.get("metrics", {}).get("foil_auc", 0.0)) if isinstance(j5, dict) else 0.0,
        "phase3_graduation_ready": bool(
            gate_row
            and float(gate_row["final_constraint_arity_strict"]) <= 0.01
            and float(gate_row["final_constraint_identity"]) <= 0.05
            and float(gate_row["final_constraint_scope"]) < 0.10
            and true_gt_false_rate >= 0.70
        ),
    }

    out_dir = args.report_output_root / f"m3_plus_{ts}"
    validate_series_outputs("M", [args.report_output_root], [out_dir])
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "m3_plus_family",
        "series": series_metadata("M", str(args.track_label), "scripts/run_m3_plus_family.py"),
        "declared_l_output_root": _norm_path(args.l_output_root),
        "declared_report_output_root": _norm_path(args.report_output_root),
        "lineage_defaults": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile=str(args.dataset_profile),
            difficulty_tier=str(args.difficulty_tier),
        ),
        "inputs": {
            "base_model": str(args.base_model),
            "adapter": _norm_path(args.adapter),
            "j5_summary": _norm_path(args.j5_summary),
            "dynamic_arity_signatures": bool(args.dynamic_arity_signatures),
            "operator_arity_json": _norm_path(args.operator_arity_json) if args.operator_arity_json else "",
            "default_relation_arity": int(args.default_relation_arity),
            "arity_enforcement_mode": str(args.arity_enforcement_mode),
            "dataset_profile": str(args.dataset_profile),
            "difficulty_tier": str(args.difficulty_tier),
            "baseline_manifest": _norm_path(args.baseline_manifest),
            "baseline_manifest_sha1": baseline_manifest_sha1,
            "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        },
        "rows": rows,
        "gate_eval": gate_eval,
    }
    out_json = out_dir / "m3_plus_report.json"
    out_md = out_dir / "m3_plus_report.md"
    report_output_paths: list[str | Path] = [out_dir, out_json, out_md, family_root]
    for row in rows:
        for key in ("run_dir", "summary_path", "checkpoint_path"):
            value = row.get(key)
            if value:
                report_output_paths.append(value)
    validate_series_outputs("M", [args.report_output_root, args.l_output_root], report_output_paths)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# M3+ Family Report",
        "",
        f"- generated_utc: `{report['generated_utc']}`",
        f"- series_id: `{report['series']['series_id']}`",
        f"- track: `{report['series']['track']}`",
        f"- base_model: `{args.base_model}`",
        f"- adapter: `{args.adapter}`",
        f"- dataset_profile: `{args.dataset_profile}`",
        f"- difficulty_tier: `{args.difficulty_tier}`",
        "",
        "| run_id | mode | dataset_profile | difficulty_tier | checkpoint_in | checkpoint_out | status | scope | scope_unbound | identity | arity_strict | tier_b | tier_c | run_dir |",
        "|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lineage = r.get("lineage", {})
            md.append(
                f"| `{r['run_id']}` | `{lineage.get('mode', '')}` | `{lineage.get('dataset_profile', '')}` | `{lineage.get('difficulty_tier', '')}` | "
                f"`{lineage.get('checkpoint_in', '') or ''}` | `{lineage.get('checkpoint_out', '') or ''}` | `{r['status']}` |  |  |  |  |  |  | `{r.get('run_dir','')}` |"
            )
            continue
        lineage = r.get("lineage", {})
        md.append(
            f"| `{r['run_id']}` | `{lineage.get('mode', '')}` | `{lineage.get('dataset_profile', '')}` | `{lineage.get('difficulty_tier', '')}` | "
            f"`{lineage.get('checkpoint_in', '') or ''}` | `{lineage.get('checkpoint_out', '') or ''}` | `ok` | "
            f"{float(r['final_constraint_scope']):.4f} | {float(r['final_constraint_scope_unbound']):.4f} | "
            f"{float(r['final_constraint_identity']):.4f} | {float(r['final_constraint_arity_strict']):.4f} | "
            f"{r['tier_b_enabled']} | {r['tier_c_enabled']} | `{r['run_dir']}` |"
        )
    md.extend(
        [
            "",
            "## Gates",
            "",
            f"- arity_feasible: `{gate_eval['arity_feasible']}`",
            f"- identity_guardrail_lte_0_05: `{gate_eval['identity_guardrail_lte_0_05']}`",
            f"- unbound_solved_lt_0_02: `{gate_eval['unbound_solved_lt_0_02']}`",
            f"- scope_gate_lt_0_10: `{gate_eval['scope_gate_lt_0_10']}`",
            f"- foil_metric_polarity_check_true_gt_false_rate: `{gate_eval['foil_metric_polarity_check_true_gt_false_rate']:.4f}`",
            f"- accepted_foil_pair_accuracy: `{gate_eval['accepted_foil_pair_accuracy']:.4f}`",
            f"- foil_auc_deprecated: `{gate_eval['foil_auc_deprecated']:.4f}`",
            f"- phase3_graduation_ready: `{gate_eval['phase3_graduation_ready']}`",
            "",
        ]
    )
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
