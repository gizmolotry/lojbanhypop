from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_match(pattern: str) -> Path:
    matches = sorted(Path(".").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return matches[-1]


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build airflow ablation hypercube report from J/L run artifacts.")
    p.add_argument(
        "--l6-manifest",
        type=Path,
        default=None,
        help="Path to l6_ablation_manifest.json. Defaults to latest under runs/l_series/l6_ablation/*/",
    )
    p.add_argument(
        "--j5-summary",
        type=Path,
        default=None,
        help="Path to j-5.json. Defaults to latest under runs/j_series/*/",
    )
    p.add_argument(
        "--m3-6-report",
        type=Path,
        default=None,
        help="Path to m3_6_symmetry_oracle_report.json (optional).",
    )
    p.add_argument(
        "--m4-operator-family-report",
        type=Path,
        default=None,
        help="Path to operator_family_report.json (optional).",
    )
    p.add_argument(
        "--m3-7-report",
        type=Path,
        default=None,
        help="Path to m3_7_shadow_report.json (optional).",
    )
    p.add_argument(
        "--m3-8-report",
        type=Path,
        default=None,
        help="Path to m3_8_diversification_report.json (optional).",
    )
    p.add_argument(
        "--m3-9-report",
        type=Path,
        default=None,
        help="Path to m3_9_primitive_probe_report.json (optional).",
    )
    p.add_argument(
        "--m3-10-report",
        type=Path,
        default=None,
        help="Path to m3_10_ood_accuracy_report.json (optional).",
    )
    p.add_argument(
        "--m3-11-report",
        type=Path,
        default=None,
        help="Path to m3_11_winograd_failure_anatomy_report.json (optional).",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/runs/telemetry/raw/ablation/hypercube"),
    )
    p.add_argument("--run-id", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    l6_manifest_path = args.l6_manifest or _latest_match("runs/l_series/l6_ablation/*/l6_ablation_manifest.json")
    j5_summary_path = args.j5_summary or _latest_match("runs/j_series/*/j-5.json")
    m3_6_report_path = args.m3_6_report
    m4_op_family_path = args.m4_operator_family_report
    m3_7_report_path = args.m3_7_report
    m3_8_report_path = args.m3_8_report
    m3_9_report_path = args.m3_9_report
    m3_10_report_path = args.m3_10_report
    m3_11_report_path = args.m3_11_report

    l6_payload = _read_json(l6_manifest_path)
    j5_payload = _read_json(j5_summary_path)
    m3_6_payload = _read_json(m3_6_report_path) if m3_6_report_path and m3_6_report_path.exists() else {}
    m4_op_payload = _read_json(m4_op_family_path) if m4_op_family_path and m4_op_family_path.exists() else {}
    m3_7_payload = _read_json(m3_7_report_path) if m3_7_report_path and m3_7_report_path.exists() else {}
    m3_8_payload = _read_json(m3_8_report_path) if m3_8_report_path and m3_8_report_path.exists() else {}
    m3_9_payload = _read_json(m3_9_report_path) if m3_9_report_path and m3_9_report_path.exists() else {}
    m3_10_payload = _read_json(m3_10_report_path) if m3_10_report_path and m3_10_report_path.exists() else {}
    m3_11_payload = _read_json(m3_11_report_path) if m3_11_report_path and m3_11_report_path.exists() else {}
    m3_7_rows = m3_7_payload.get("rows", []) if isinstance(m3_7_payload, dict) else []
    m3_7_ok = [r for r in m3_7_rows if str(r.get("status", "")).lower() == "ok"]
    m3_8_rows = m3_8_payload.get("rows", []) if isinstance(m3_8_payload, dict) else []
    m3_8_ok = [r for r in m3_8_rows if str(r.get("status", "")).lower() == "ok"]

    rows = l6_payload.get("rows", [])
    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if not ok_rows:
        raise ValueError("No successful L6 rows found in manifest.")

    best_scope = min(ok_rows, key=lambda r: _as_float(r.get("final_constraint_scope"), 1e9))
    best_identity = min(ok_rows, key=lambda r: _as_float(r.get("final_constraint_identity"), 1e9))
    arity_all_zero = all(_as_float(r.get("final_constraint_arity_strict"), 1.0) <= 1e-9 for r in ok_rows)

    j5_metrics = j5_payload.get("metrics", {})
    gate_scope = _as_float(best_scope.get("final_constraint_scope"), 1.0) < 0.10
    gate_identity = _as_float(best_identity.get("final_constraint_identity"), 1.0) <= 0.05
    gate_foil_integrity = _as_float(j5_metrics.get("foil_minimal_edit_rate"), 0.0) >= 0.95
    phase3_gate_ready = bool(gate_scope and gate_identity and gate_foil_integrity and arity_all_zero)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "report_type": "airflow_ablation_hypercube",
        "run_id": run_id,
        "series_contract_version": "1.0",
        "series_id": "M",
        "track": "hypercube_ablation",
        "source_artifacts": {
            "l6_manifest": str(l6_manifest_path).replace("\\", "/"),
            "j5_summary": str(j5_summary_path).replace("\\", "/"),
            "m3_6_report": str(m3_6_report_path).replace("\\", "/") if m3_6_report_path else "",
            "m4_operator_family_report": str(m4_op_family_path).replace("\\", "/") if m4_op_family_path else "",
            "m3_7_report": str(m3_7_report_path).replace("\\", "/") if m3_7_report_path else "",
            "m3_8_report": str(m3_8_report_path).replace("\\", "/") if m3_8_report_path else "",
            "m3_9_report": str(m3_9_report_path).replace("\\", "/") if m3_9_report_path else "",
            "m3_10_report": str(m3_10_report_path).replace("\\", "/") if m3_10_report_path else "",
            "m3_11_report": str(m3_11_report_path).replace("\\", "/") if m3_11_report_path else "",
        },
        "dimensions": {
            "m1_data_regime": ["M1.4/J-5"],
            "m2_tier_toggles": ["M2.A", "M2.B", "M2.C"],
            "gates": ["arity_strict", "scope", "identity", "foil_minimal_edit_integrity"],
        },
        "m2_cells": ok_rows,
        "j5_metrics": {
            "generator_accept_rate": _as_float(j5_metrics.get("generator_accept_rate")),
            "accepted_foil_pair_accuracy": _as_float(j5_metrics.get("accepted_foil_pair_accuracy", j5_metrics.get("foil_auc"))),
            "foil_auc_deprecated": _as_float(j5_metrics.get("foil_auc")),
            "foil_minimal_edit_rate": _as_float(j5_metrics.get("foil_minimal_edit_rate")),
            "accept_rate_by_depth": j5_metrics.get("accept_rate_by_depth", j5_metrics.get("scope_by_depth", {})),
            "scope_by_depth_deprecated": j5_metrics.get("scope_by_depth", {}),
            "scope_components_mean": j5_metrics.get("scope_components_mean", {}),
        },
        "gate_eval": {
            "arity_all_zero": arity_all_zero,
            "best_scope_cell": {
                "run_id": best_scope.get("run_id"),
                "m_series_alias": best_scope.get("m_series_alias"),
                "final_constraint_scope": _as_float(best_scope.get("final_constraint_scope")),
            },
            "best_identity_cell": {
                "run_id": best_identity.get("run_id"),
                "m_series_alias": best_identity.get("m_series_alias"),
                "final_constraint_identity": _as_float(best_identity.get("final_constraint_identity")),
            },
            "gate_scope_lt_0_10": gate_scope,
            "gate_identity_lte_0_05": gate_identity,
            "gate_foil_min_edit_gte_0_95": gate_foil_integrity,
            "phase3_gate_ready": phase3_gate_ready,
        },
        "m3_6_metrics": {
            "symmetry_false_foil_rate": _as_float(m3_6_payload.get("symmetry_false_foil_rate")),
            "m3_6_a_total": _as_float(m3_6_payload.get("cells", {}).get("M3.6.A", {}).get("symmetry_oracle_accuracy_total")),
            "m3_6_b_total": _as_float(m3_6_payload.get("cells", {}).get("M3.6.B", {}).get("symmetry_oracle_accuracy_total")),
            "m3_6_c_total": _as_float(m3_6_payload.get("cells", {}).get("M3.6.C", {}).get("symmetry_oracle_accuracy_total")),
        },
        "m4_family_metrics": {
            "active_op_count": int(_as_float(m4_op_payload.get("active_op_count"), 0.0)),
            "top1_op_share": _as_float(m4_op_payload.get("top1_op_share")),
            "predicate_family_score": _as_float(m4_op_payload.get("predicate_family_score")),
            "paraphrase_operator_consistency": _as_float(m4_op_payload.get("paraphrase_operator_consistency")),
        },
        "m3_7_shadow_metrics": {
            "cell_count_ok": int(len(m3_7_ok)),
            "best_shadow_scope": min([_as_float(r.get("final_constraint_scope"), 1e9) for r in m3_7_ok], default=None),
            "best_shadow_identity": min([_as_float(r.get("final_constraint_identity"), 1e9) for r in m3_7_ok], default=None),
            "mean_shadow_loss": (
                float(sum(_as_float(r.get("shadow_loss"), 0.0) for r in m3_7_ok) / len(m3_7_ok))
                if m3_7_ok
                else 0.0
            ),
        },
        "m3_8_diversification_metrics": {
            "cell_count_ok": int(len(m3_8_ok)),
            "mean_operator_entropy": (
                float(sum(_as_float(r.get("operator_entropy"), 0.0) for r in m3_8_ok) / len(m3_8_ok))
                if m3_8_ok
                else 0.0
            ),
            "mean_operator_top1_share": (
                float(sum(_as_float(r.get("operator_top1_share"), 0.0) for r in m3_8_ok) / len(m3_8_ok))
                if m3_8_ok
                else 0.0
            ),
            "mean_diversification_loss": (
                float(sum(_as_float(r.get("diversification_loss"), 0.0) for r in m3_8_ok) / len(m3_8_ok))
                if m3_8_ok
                else 0.0
            ),
        },
        "m3_9_primitive_metrics": {
            "active_token_count": int(_as_float(m3_9_payload.get("summary", {}).get("active_token_count"), 0.0)),
            "primitive_candidate_count": int(_as_float(m3_9_payload.get("summary", {}).get("primitive_candidate_count"), 0.0)),
            "mean_baseline_ce_loss": _as_float(m3_9_payload.get("summary", {}).get("mean_baseline_ce_loss")),
            "mean_baseline_scope": _as_float(m3_9_payload.get("summary", {}).get("mean_baseline_scope")),
        },
        "m3_10_ood_metrics": {
            "legacy_accuracy": _as_float(m3_10_payload.get("bucket_metrics", {}).get("legacy", {}).get("accuracy")),
            "easy_accuracy": _as_float(m3_10_payload.get("bucket_metrics", {}).get("easy", {}).get("accuracy")),
            "medium_accuracy": _as_float(m3_10_payload.get("bucket_metrics", {}).get("medium", {}).get("accuracy")),
            "hard_accuracy": _as_float(m3_10_payload.get("bucket_metrics", {}).get("hard", {}).get("accuracy")),
            "hard_active_token_count": _as_float(m3_10_payload.get("bucket_metrics", {}).get("hard", {}).get("mean_active_token_count")),
            "hard_scope": _as_float(m3_10_payload.get("bucket_metrics", {}).get("hard", {}).get("mean_scope")),
            "hard_corr_active_vs_correct": _as_float(m3_10_payload.get("bucket_metrics", {}).get("hard", {}).get("corr_active_token_count_vs_correct")),
        },
        "m3_11_winograd_metrics": {
            "legacy_accuracy": _as_float(m3_11_payload.get("bucket_metrics", {}).get("legacy", {}).get("accuracy")),
            "easy_accuracy": _as_float(m3_11_payload.get("bucket_metrics", {}).get("easy", {}).get("accuracy")),
            "medium_accuracy": _as_float(m3_11_payload.get("bucket_metrics", {}).get("medium", {}).get("accuracy")),
            "hard_accuracy": _as_float(m3_11_payload.get("bucket_metrics", {}).get("hard", {}).get("accuracy")),
            "overall_failure_taxonomy": m3_11_payload.get("overall_failure_taxonomy", {}),
        },
    }

    out_json = out_dir / "ablation_hypercube_report.json"
    out_md = out_dir / "ablation_hypercube_report.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Airflow Ablation Hypercube Report",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_utc: `{report['generated_utc']}`",
        f"- l6_manifest: `{report['source_artifacts']['l6_manifest']}`",
        f"- j5_summary: `{report['source_artifacts']['j5_summary']}`",
        "",
        "## M2 Cells",
        "",
        "| run_id | alias | scope | identity | arity_strict | tier_b | tier_c |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for r in ok_rows:
        lines.append(
            f"| `{r.get('run_id')}` | `{r.get('m_series_alias')}` | "
            f"{_as_float(r.get('final_constraint_scope')):.4f} | "
            f"{_as_float(r.get('final_constraint_identity')):.4f} | "
            f"{_as_float(r.get('final_constraint_arity_strict')):.4f} | "
            f"{bool(r.get('tier_b_enabled'))} | {bool(r.get('tier_c_enabled'))} |"
        )
    lines.extend(
        [
            "",
            "## J-5 Metrics",
            "",
            f"- generator_accept_rate: `{_as_float(j5_metrics.get('generator_accept_rate')):.6f}`",
            f"- accepted_foil_pair_accuracy: `{_as_float(j5_metrics.get('accepted_foil_pair_accuracy', j5_metrics.get('foil_auc'))):.6f}`",
            f"- foil_auc_deprecated: `{_as_float(j5_metrics.get('foil_auc')):.6f}`",
            f"- foil_minimal_edit_rate: `{_as_float(j5_metrics.get('foil_minimal_edit_rate')):.6f}`",
            "",
            "## M3.6 / M3.6.2 Metrics",
            "",
            f"- symmetry_false_foil_rate: `{_as_float(report['m3_6_metrics']['symmetry_false_foil_rate']):.6f}`",
            f"- m3_6_a_total: `{_as_float(report['m3_6_metrics']['m3_6_a_total']):.6f}`",
            f"- m3_6_b_total: `{_as_float(report['m3_6_metrics']['m3_6_b_total']):.6f}`",
            f"- m3_6_c_total: `{_as_float(report['m3_6_metrics']['m3_6_c_total']):.6f}`",
            f"- active_op_count: `{int(report['m4_family_metrics']['active_op_count'])}`",
            f"- top1_op_share: `{_as_float(report['m4_family_metrics']['top1_op_share']):.6f}`",
            f"- m3_7_cell_count_ok: `{int(report['m3_7_shadow_metrics']['cell_count_ok'])}`",
            f"- m3_7_best_scope: `{report['m3_7_shadow_metrics']['best_shadow_scope']}`",
            f"- m3_7_best_identity: `{report['m3_7_shadow_metrics']['best_shadow_identity']}`",
            f"- m3_7_mean_shadow_loss: `{_as_float(report['m3_7_shadow_metrics']['mean_shadow_loss']):.6f}`",
            f"- m3_8_cell_count_ok: `{int(report['m3_8_diversification_metrics']['cell_count_ok'])}`",
            f"- m3_8_mean_operator_entropy: `{_as_float(report['m3_8_diversification_metrics']['mean_operator_entropy']):.6f}`",
            f"- m3_8_mean_operator_top1_share: `{_as_float(report['m3_8_diversification_metrics']['mean_operator_top1_share']):.6f}`",
            f"- m3_8_mean_diversification_loss: `{_as_float(report['m3_8_diversification_metrics']['mean_diversification_loss']):.6f}`",
            f"- m3_9_active_token_count: `{int(report['m3_9_primitive_metrics']['active_token_count'])}`",
            f"- m3_9_primitive_candidate_count: `{int(report['m3_9_primitive_metrics']['primitive_candidate_count'])}`",
            f"- m3_9_mean_baseline_ce_loss: `{_as_float(report['m3_9_primitive_metrics']['mean_baseline_ce_loss']):.6f}`",
            f"- m3_9_mean_baseline_scope: `{_as_float(report['m3_9_primitive_metrics']['mean_baseline_scope']):.6f}`",
            f"- m3_10_legacy_accuracy: `{_as_float(report['m3_10_ood_metrics']['legacy_accuracy']):.6f}`",
            f"- m3_10_easy_accuracy: `{_as_float(report['m3_10_ood_metrics']['easy_accuracy']):.6f}`",
            f"- m3_10_medium_accuracy: `{_as_float(report['m3_10_ood_metrics']['medium_accuracy']):.6f}`",
            f"- m3_10_hard_accuracy: `{_as_float(report['m3_10_ood_metrics']['hard_accuracy']):.6f}`",
            f"- m3_10_hard_active_token_count: `{_as_float(report['m3_10_ood_metrics']['hard_active_token_count']):.6f}`",
            f"- m3_10_hard_scope: `{_as_float(report['m3_10_ood_metrics']['hard_scope']):.6f}`",
            f"- m3_10_hard_corr_active_vs_correct: `{_as_float(report['m3_10_ood_metrics']['hard_corr_active_vs_correct']):.6f}`",
            f"- m3_11_legacy_accuracy: `{_as_float(report['m3_11_winograd_metrics']['legacy_accuracy']):.6f}`",
            f"- m3_11_easy_accuracy: `{_as_float(report['m3_11_winograd_metrics']['easy_accuracy']):.6f}`",
            f"- m3_11_medium_accuracy: `{_as_float(report['m3_11_winograd_metrics']['medium_accuracy']):.6f}`",
            f"- m3_11_hard_accuracy: `{_as_float(report['m3_11_winograd_metrics']['hard_accuracy']):.6f}`",
            "",
            "## Gate Evaluation",
            "",
            f"- arity_all_zero: `{arity_all_zero}`",
            f"- best_scope_cell: `{best_scope.get('run_id')}/{best_scope.get('m_series_alias')}` (`{_as_float(best_scope.get('final_constraint_scope')):.4f}`)",
            f"- best_identity_cell: `{best_identity.get('run_id')}/{best_identity.get('m_series_alias')}` (`{_as_float(best_identity.get('final_constraint_identity')):.4f}`)",
            f"- gate_scope_lt_0_10: `{gate_scope}`",
            f"- gate_identity_lte_0_05: `{gate_identity}`",
            f"- gate_foil_min_edit_gte_0_95: `{gate_foil_integrity}`",
            f"- phase3_gate_ready: `{phase3_gate_ready}`",
            "",
        ]
    )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {out_json}")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    main()
