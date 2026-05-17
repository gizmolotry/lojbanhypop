from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m20.dictionary import M20_LOCKS  # noqa: E402
from lojban_evolution.m20.family import M20_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


def _safe_run_id(value: str | None) -> str:
    raw = (value or datetime.now(timezone.utc).strftime("m20_lock_suite_%Y%m%dT%H%M%SZ")).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def _read_report(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _metrics_from_reports(train_report: dict[str, Any], induction_report: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, Any] = {}
    metrics.update(induction_report.get("metrics", {}) if isinstance(induction_report, dict) else {})
    metrics.update(train_report.get("metrics", {}) if isinstance(train_report, dict) else {})
    aggregate = train_report.get("aggregate_metrics", {}) if isinstance(train_report, dict) else {}
    if aggregate:
        metrics.update(aggregate)
        metrics.setdefault("strict_accuracy", aggregate.get("mean_strict_accuracy", 0.0))
        metrics.setdefault("dictionary_coverage", aggregate.get("mean_strict_accuracy", 0.0))
        metrics.setdefault("synthetic_world_accuracy", aggregate.get("mean_strict_accuracy", 0.0))
        metrics.setdefault("factorized_exact_accuracy", aggregate.get("mean_factorized_exact_accuracy", 0.0))
        metrics.setdefault("brivi_gate_accuracy", aggregate.get("mean_brivi_gate_accuracy", 0.0))
        metrics.setdefault("predicate_identity_stability", aggregate.get("mean_predicate_identity_stability", 0.0))
        metrics.setdefault("lock_pass_rate", aggregate.get("mean_lock_pass_rate", 0.0))
    cells = train_report.get("cells", {}) if isinstance(train_report, dict) else {}
    if cells and "strict_accuracy" not in metrics:
        values = [float(cell.get("aggregate_metrics", {}).get("mean_strict_accuracy", 0.0)) for cell in cells.values()]
        metrics["strict_accuracy"] = sum(values) / max(1, len(values))
    if cells:
        seed_metric_values: dict[str, list[float]] = {}
        for cell in cells.values():
            for seed_report in cell.get("seed_reports", []):
                for key, value in seed_report.get("metrics", {}).items():
                    if isinstance(value, (int, float)):
                        seed_metric_values.setdefault(key, []).append(float(value))
        for key, values in seed_metric_values.items():
            metrics.setdefault(key, sum(values) / max(1, len(values)))
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _lock_status(metrics: dict[str, float]) -> dict[str, bool]:
    return {
        "dictionary_first_pretraining": metrics.get("dictionary_coverage", metrics.get("strict_accuracy", 0.0)) >= 0.70,
        "factorized_predicate_dictionary": metrics.get("factorized_exact_accuracy", metrics.get("mean_factorized_exact_accuracy", 0.0)) >= 0.60,
        "counterfactual_quotient_dictionary": metrics.get("predicate_identity_stability", metrics.get("counterfactual_quotient_consistency", 0.0)) >= 0.85,
        "brivi_locked_predicate_formation": metrics.get("brivi_gate_accuracy", metrics.get("mean_brivi_gate_accuracy", 0.0)) >= 0.80,
        "synthetic_world_pretraining": metrics.get("synthetic_world_accuracy", metrics.get("strict_accuracy", 0.0)) >= 0.70,
        "soft_dictionary_before_hard_dictionary": metrics.get("soft_dictionary_entropy", metrics.get("dictionary_entropy", 0.0)) > 0.0,
    }


def run_lock_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M20_REGISTRY["M20"]
    output_root = Path(args.output_root or registry["output_roots"]["lock_suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe_run_id(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.output_path) if args.output_path else run_dir / registry["report_names"]["lock_suite"]
    validate_series_outputs("M", [output_root, run_dir], [report_path])
    train_report = _read_report(args.train_report)
    induction_report = _read_report(args.induction_report)
    metrics = _metrics_from_reports(train_report, induction_report)
    statuses = _lock_status(metrics)
    payload = {
        "series": series_metadata("M", "M20.1.lock_suite", "scripts/m20/run_m20_lock_suite.py"),
        "lineage": lineage_metadata("eval_only", checkpoint_in=train_report.get("checkpoint_path") if train_report else None),
        "track": "M20.1",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "architecture_locks": M20_LOCKS,
        "lock_status": statuses,
        "metrics": {
            **metrics,
            "lock_pass_rate": sum(1.0 for ok in statuses.values() if ok) / max(1, len(statuses)),
        },
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M20 lock suite report written to {report_path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M20_REGISTRY["M20"]
    parser = argparse.ArgumentParser(description="Evaluate the six M20 dictionary-first architecture locks.")
    parser.add_argument("--train-report", type=Path, default=None)
    parser.add_argument("--induction-report", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["lock_suite"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_lock_suite(parse_args())
