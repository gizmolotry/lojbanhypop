from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m22.family import M22_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402


SUITE_KEYS = (
    "strict_accuracy",
    "bridi_trace_exact_accuracy",
    "gismu_accuracy",
    "cmavo_accuracy",
    "judri_binding_accuracy",
    "cmavo_causal_delta",
    "judri_causal_delta",
)

AUDIT_KEYS = (
    "adversarial_strict_accuracy",
    "adversarial_bridi_trace_exact_accuracy",
    "adversarial_worst_surface_accuracy",
    "adversarial_judri_causal_delta",
    "adversarial_oov_synonym_accuracy",
    "adversarial_oov_synonym_trace_exact_accuracy",
    "adversarial_oov_token_rate",
    "m22_relation_ood_strict_accuracy",
    "m22_relation_ood_worst_surface_accuracy",
    "m22_relation_ood_judri_causal_delta",
    "m22_relation_ood_oov_token_rate",
    "m22_relation_ood_surface_count",
    "m22_relation_ood_surface_seed_std_max",
    "m22_relation_ood_surface_seed_min_accuracy",
)

GATE_KEYS = (
    "strict_accuracy",
    "semantic_coverage_strict_accuracy",
    "semantic_coverage_worst_surface_accuracy",
    "semantic_coverage_judri_causal_delta",
    "semantic_coverage_oov_synonym_accuracy",
    "semantic_coverage_oov_token_rate",
    "m22_relation_ood_strict_accuracy",
    "m22_relation_ood_worst_surface_accuracy",
    "m22_relation_ood_judri_causal_delta",
    "m22_hard_relation_ood_score",
    "m22_relation_ood_surface_count",
    "m22_semantic_strict_delta_vs_m21_control",
    "m22_semantic_worst_delta_vs_m21_control",
    "m22_clean_accuracy_drop_vs_m21_control",
    "m22_judri_delta_drop_vs_m21_control",
    "m22_promotion_candidate",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m22_seed_stability_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m22_seed_stability_{_timestamp()}"


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
    return {
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "count": float(len(values)),
    }


def _flatten_stats(prefix: str, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        values = [_num(row.get(key)) for row in rows if row.get(key) is not None]
        stats = _stats(values)
        for stat_key, stat_value in stats.items():
            out[f"{prefix}_{key}_{stat_key}"] = stat_value
    return out


def _suite_seed_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        run_id = str(payload.get("run_id", ""))
        cells = payload.get("cells", {})
        if not isinstance(cells, dict):
            continue
        for cell_key, cell in cells.items():
            if not isinstance(cell, dict):
                continue
            for report in cell.get("seed_reports", []):
                if not isinstance(report, dict):
                    continue
                metrics = report.get("metrics", {})
                if not isinstance(metrics, dict):
                    continue
                row = {key: _num(metrics.get(key)) for key in SUITE_KEYS}
                row.update({"run_id": run_id, "cell_key": str(cell_key), "seed": report.get("seed")})
                rows.append(row)
    return rows


def _audit_seed_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        run_id = str(payload.get("run_id", ""))
        for report in payload.get("seed_reports", []):
            if not isinstance(report, dict):
                continue
            metrics = report.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            surfaces = metrics.get("surface_metrics", {})
            row = {key: _num(metrics.get(key)) for key in AUDIT_KEYS}
            if isinstance(surfaces, dict) and row["adversarial_oov_synonym_accuracy"] == 0.0:
                oov_metrics = surfaces.get("oov_synonym", {})
                if isinstance(oov_metrics, dict):
                    row["adversarial_oov_synonym_accuracy"] = _num(oov_metrics.get("strict_accuracy"))
                    row["adversarial_oov_synonym_trace_exact_accuracy"] = _num(
                        oov_metrics.get("bridi_trace_exact_accuracy")
                    )
            row.update(
                {
                    "run_id": run_id,
                    "cell_key": str(report.get("cell_key", "")),
                    "seed": report.get("seed"),
                    "surface_metrics": surfaces,
                }
            )
            rows.append(row)
    return rows


def _gate_rows(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        metrics = payload.get("metrics", {})
        if not isinstance(metrics, dict):
            continue
        row = {key: _num(metrics.get(key)) for key in GATE_KEYS}
        gates = payload.get("promotion_gates", {})
        gates_present = isinstance(gates, dict) and bool(gates)
        gates_pass = gates_present and all(bool(value) for value in gates.values())
        row["promotion_gates_present"] = float(gates_present)
        row["promotion_gates_pass"] = float(gates_pass)
        if "m22_promotion_candidate" in row:
            row["m22_promotion_candidate"] = float(row["m22_promotion_candidate"] >= 1.0 and gates_pass)
        row["run_id"] = str(payload.get("run_id", ""))
        rows.append(row)
    return rows


def _surface_summary(audit_rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    by_surface: dict[str, list[float]] = {}
    for row in audit_rows:
        surfaces = row.get("surface_metrics", {})
        if not isinstance(surfaces, dict):
            continue
        for surface, metrics in surfaces.items():
            if isinstance(metrics, dict):
                by_surface.setdefault(str(surface), []).append(_num(metrics.get("strict_accuracy")))
    return {surface: _stats(values) for surface, values in sorted(by_surface.items())}


def _build_metrics(
    suite_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    gate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    surface_summary = _surface_summary(audit_rows)
    metrics: dict[str, Any] = {
        "m22_seed_stability_suite_seed_count": float(len(suite_rows)),
        "m22_seed_stability_audit_seed_count": float(len(audit_rows)),
        "m22_seed_stability_gate_run_count": float(len(gate_rows)),
        "m22_seed_stability_surface_accuracy": surface_summary,
        "m22_seed_stability_surface_seed_std_max": max(
            (float(row.get("std", 0.0) or 0.0) for row in surface_summary.values()),
            default=0.0,
        ),
        "m22_seed_stability_surface_seed_min_accuracy": min(
            (float(row.get("min", 0.0) or 0.0) for row in surface_summary.values()),
            default=0.0,
        ),
    }
    metrics.update(_flatten_stats("suite_seed", suite_rows, SUITE_KEYS))
    metrics.update(_flatten_stats("audit_seed", audit_rows, AUDIT_KEYS))
    metrics.update(_flatten_stats("gate_run", gate_rows, GATE_KEYS))
    promotion_values = [row["m22_promotion_candidate"] for row in gate_rows if "m22_promotion_candidate" in row]
    metrics["m22_seed_stability_promotion_rate"] = mean(promotion_values) if promotion_values else 0.0
    metrics["m22_seed_stability_all_gates_promote"] = float(bool(promotion_values) and all(value >= 1.0 for value in promotion_values))
    gate_evidence = [row.get("promotion_gates_present", 0.0) for row in gate_rows]
    metrics["m22_seed_stability_gate_evidence_rate"] = mean(gate_evidence) if gate_evidence else 0.0
    return metrics


def _render_markdown(payload: dict[str, Any]) -> str:
    metrics = payload["metrics"]
    surface_rows = metrics.get("m22_seed_stability_surface_accuracy", {})
    lines = [
        "# M22 Seed Stability Aggregate",
        "",
        f"- run_id: `{payload['run_id']}`",
        f"- suite seed count: `{metrics['m22_seed_stability_suite_seed_count']:.0f}`",
        f"- audit seed count: `{metrics['m22_seed_stability_audit_seed_count']:.0f}`",
        f"- gate run count: `{metrics['m22_seed_stability_gate_run_count']:.0f}`",
        f"- promotion rate: `{metrics['m22_seed_stability_promotion_rate']:.4f}`",
        "",
        "## Headline",
        "",
        f"- suite strict mean/std/min: `{metrics['suite_seed_strict_accuracy_mean']:.4f}` / `{metrics['suite_seed_strict_accuracy_std']:.4f}` / `{metrics['suite_seed_strict_accuracy_min']:.4f}`",
        f"- audit semantic strict mean/std/min: `{metrics['audit_seed_adversarial_strict_accuracy_mean']:.4f}` / `{metrics['audit_seed_adversarial_strict_accuracy_std']:.4f}` / `{metrics['audit_seed_adversarial_strict_accuracy_min']:.4f}`",
        f"- audit worst-surface mean/min: `{metrics['audit_seed_adversarial_worst_surface_accuracy_mean']:.4f}` / `{metrics['audit_seed_adversarial_worst_surface_accuracy_min']:.4f}`",
        f"- OOD synonym accuracy mean/min: `{metrics['audit_seed_adversarial_oov_synonym_accuracy_mean']:.4f}` / `{metrics['audit_seed_adversarial_oov_synonym_accuracy_min']:.4f}`",
        f"- OOD token rate mean: `{metrics['audit_seed_adversarial_oov_token_rate_mean']:.4f}`",
        f"- hard relation-OOD strict/worst/judri mean: `{metrics['audit_seed_m22_relation_ood_strict_accuracy_mean']:.4f}` / `{metrics['audit_seed_m22_relation_ood_worst_surface_accuracy_mean']:.4f}` / `{metrics['audit_seed_m22_relation_ood_judri_causal_delta_mean']:.4f}`",
        "",
        "## Surface Variance",
        "",
        "| surface | mean | std | min | max | seeds |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if isinstance(surface_rows, dict):
        for surface, row in surface_rows.items():
            if not isinstance(row, dict):
                continue
            lines.append(
                f"| `{surface}` | {float(row.get('mean', 0.0)):.4f} | {float(row.get('std', 0.0)):.4f} | "
                f"{float(row.get('min', 0.0)):.4f} | {float(row.get('max', 0.0)):.4f} | {float(row.get('count', 0.0)):.0f} |"
            )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Strict accuracy remains canonical; OOD synonym accuracy is a diagnostic semantic-stress metric.",
            "- OOD token rate is lexical novelty, not correctness. It is reported separately to avoid the old ambiguity.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M22_REGISTRY["M22"]
    parser = argparse.ArgumentParser(description="Aggregate M22 S-cell seed stability across suite/audit/gate reports.")
    parser.add_argument("--suite-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--adversarial-audit-reports", type=Path, nargs="+", required=True)
    parser.add_argument("--gate-reports", type=Path, nargs="*", default=[])
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["seed_stability"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


def run_seed_stability(args: argparse.Namespace) -> dict[str, Any]:
    registry = M22_REGISTRY["M22"]
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_payloads = [_read(path) for path in args.suite_reports]
    audit_payloads = [_read(path) for path in args.adversarial_audit_reports]
    gate_payloads = [_read(path) for path in args.gate_reports]
    suite_rows = _suite_seed_rows(suite_payloads)
    audit_rows = _audit_seed_rows(audit_payloads)
    gate_rows = _gate_rows(gate_payloads)
    payload = {
        "series": series_metadata("M", "M22.seed_stability", "scripts/m22/run_m22_seed_stability_aggregate.py"),
        "track": "M22",
        "family_version": "0.1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "source_reports": {
            "suite_reports": [str(path) for path in args.suite_reports],
            "adversarial_audit_reports": [str(path) for path in args.adversarial_audit_reports],
            "gate_reports": [str(path) for path in args.gate_reports],
        },
        "metrics": _build_metrics(suite_rows, audit_rows, gate_rows),
        "suite_seed_rows": suite_rows,
        "audit_seed_rows": [
            {key: value for key, value in row.items() if key != "surface_metrics"}
            for row in audit_rows
        ],
        "canonical_accuracy": "suite_seed_strict_accuracy_mean",
        "diagnostic_only": [
            "audit_seed_adversarial_oov_synonym_accuracy_mean",
            "audit_seed_adversarial_oov_token_rate_mean",
            "audit_seed_m22_relation_ood_strict_accuracy_mean",
            "audit_seed_m22_relation_ood_worst_surface_accuracy_mean",
            "audit_seed_m22_relation_ood_judri_causal_delta_mean",
            "audit_seed_m22_relation_ood_oov_token_rate_mean",
            "audit_seed_m22_relation_ood_surface_count_mean",
            "audit_seed_m22_relation_ood_surface_seed_std_max_mean",
            "audit_seed_m22_relation_ood_surface_seed_min_accuracy_mean",
        ],
        "notes": [
            "This report aggregates completed M22 runs; it does not retrain or change promotion thresholds.",
            "OOD synonym accuracy is correctness on the oov_synonym surface; OOD token rate is lexical novelty only.",
        ],
    }
    report_path = run_dir / registry["report_names"]["seed_stability"]
    markdown_path = run_dir / "m22_seed_stability_summary.md"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    markdown_path.write_text(_render_markdown(payload), encoding="utf-8")
    metrics = payload["metrics"]
    print(f"M22 seed stability report written to {report_path}")
    print(f"M22 seed stability summary written to {markdown_path}")
    print(
        "M22 stability metrics: "
        f"seeds={metrics['m22_seed_stability_suite_seed_count']:.0f} "
        f"strict={metrics['suite_seed_strict_accuracy_mean']:.4f} "
        f"audit={metrics['audit_seed_adversarial_strict_accuracy_mean']:.4f} "
        f"oov_acc={metrics['audit_seed_adversarial_oov_synonym_accuracy_mean']:.4f} "
        f"rel_ood={metrics['audit_seed_m22_relation_ood_strict_accuracy_mean']:.4f} "
        f"promote={metrics['m22_seed_stability_promotion_rate']:.4f}"
    )
    return payload


if __name__ == "__main__":
    run_seed_stability(parse_args())
