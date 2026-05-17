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

from lojban_evolution.m21.bridi import M21_LOCKS  # noqa: E402
from lojban_evolution.m21.family import M21_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m21_lock_suite_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m21_lock_suite_{_timestamp()}"


def _read(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _merge_metrics(*payloads: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for payload in payloads:
        for bucket_name in ("metrics", "aggregate_metrics"):
            bucket = payload.get(bucket_name)
            if not isinstance(bucket, dict):
                continue
            for key, value in bucket.items():
                if not isinstance(value, (int, float)):
                    continue
                name = str(key)
                out[name] = float(value)
                if name.startswith("mean_"):
                    out[name[5:]] = float(value)
    return out


def _statuses(metrics: dict[str, float]) -> dict[str, bool]:
    strict = float(metrics.get("strict_accuracy", 0.0))
    return {
        "dynamic_frame_count": float(metrics.get("frame_count_mae", 999.0)) <= 0.35 and float(metrics.get("mean_active_frames", 0.0)) > 0.25,
        "bridi_trace_reconstruction": float(metrics.get("bridi_trace_exact_accuracy", 0.0)) >= 0.55,
        "cmavo_causality": float(metrics.get("cmavo_accuracy", 0.0)) >= 0.55 or float(metrics.get("cmavo_causal_delta", 0.0)) >= 0.02,
        "judri_binding_causality": float(metrics.get("judri_binding_accuracy", 0.0)) >= 0.55 or float(metrics.get("judri_causal_delta", 0.0)) >= 0.02,
        "judri_gated_bridge": float(metrics.get("judri_bridge_gate_enabled", 0.0)) >= 0.5 and float(metrics.get("judri_bridge_gate_active_mean", 0.0)) > 0.05,
        "brivi_lock": float(metrics.get("brivi_lock_violation_rate", 1.0)) <= 0.10 or float(metrics.get("brivi_gate_accuracy", 0.0)) >= 0.90,
        "actual_bridge_transfer": strict >= max(0.15, float(metrics.get("random_trace_accuracy", 0.0)) + 0.10),
    }


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M21_REGISTRY["M21"]
    output_root = Path(args.output_root or registry["output_roots"]["lock_suite"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    validate_series_outputs("M", [output_root], [run_dir])
    suite_payload = _read(args.suite_report)
    actual_payload = _read(args.actual_bridge_report)
    metrics = _merge_metrics(suite_payload, actual_payload)
    statuses = _statuses(metrics)
    metrics["lock_pass_rate"] = sum(1.0 for ok in statuses.values() if ok) / max(1, len(statuses))
    report_path = Path(args.output_path) if args.output_path else run_dir / registry["report_names"]["lock_suite"]
    validate_series_outputs("M", [registry["output_roots"]["lock_suite"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M21.1.lock_suite", "scripts/m21/run_m21_lock_suite.py"),
        "track": "M21.1",
        "registry": {
            "runner_script": registry["runner_scripts"]["lock_suite"],
            "dag": registry["dags"]["lock_suite"],
            "output_root": registry["output_roots"]["lock_suite"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "architecture_locks": M21_LOCKS,
        "lock_status": statuses,
        "metrics": metrics,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M21 lock suite report written to {report_path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M21_REGISTRY["M21"]
    parser = argparse.ArgumentParser(description="Evaluate the M21 dynamic bridi architecture locks.")
    parser.add_argument("--suite-report", type=Path, default=None)
    parser.add_argument("--actual-bridge-report", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["lock_suite"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
