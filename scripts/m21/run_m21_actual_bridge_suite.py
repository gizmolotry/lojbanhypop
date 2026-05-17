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

from lojban_evolution.m21.family import M21_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m21_actual_bridge_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m21_actual_bridge_{_timestamp()}"


def _read(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _metrics(payload: dict[str, Any]) -> dict[str, float]:
    if isinstance(payload.get("metrics"), dict):
        return {str(k): float(v) for k, v in payload["metrics"].items() if isinstance(v, (int, float))}
    if isinstance(payload.get("aggregate_metrics"), dict):
        out: dict[str, float] = {}
        for key, value in payload["aggregate_metrics"].items():
            if isinstance(value, (int, float)):
                name = str(key)
                if name.startswith("mean_"):
                    out[name[5:]] = float(value)
                out[name] = float(value)
        return out
    return {}


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    registry = M21_REGISTRY["M21"]
    output_root = Path(args.output_root or registry["output_roots"]["actual_bridge"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    validate_series_outputs("M", [output_root], [run_dir])
    source_payload = _read(args.train_report or args.suite_report)
    metrics = _metrics(source_payload)
    full = float(metrics.get("full_accuracy", metrics.get("strict_accuracy", 0.0)))
    no_cmavo = float(metrics.get("no_cmavo_accuracy", 0.0))
    no_judri = float(metrics.get("no_judri_accuracy", 0.0))
    gismu_only = float(metrics.get("gismu_only_accuracy", 0.0))
    random_trace = float(metrics.get("random_trace_accuracy", 1.0 / 18.0))
    frame_drop = float(metrics.get("frame_drop_accuracy", max(0.0, full - float(metrics.get("frame_drop_delta", 0.0)))))
    report_metrics = {
        "strict_accuracy": full,
        "full_accuracy": full,
        "no_cmavo_accuracy": no_cmavo,
        "no_judri_accuracy": no_judri,
        "gismu_only_accuracy": gismu_only,
        "random_trace_accuracy": random_trace,
        "scratchpad_only_accuracy": float(metrics.get("scratchpad_only_accuracy", 0.0)),
        "frame_drop_accuracy": frame_drop,
        "frame_drop_delta": float(metrics.get("frame_drop_delta", full - frame_drop)),
        "cmavo_causal_delta": float(metrics.get("cmavo_causal_delta", full - no_cmavo)),
        "judri_causal_delta": float(metrics.get("judri_causal_delta", full - no_judri)),
        "actual_bridge_transfer_score": max(0.0, full - max(no_cmavo, no_judri, random_trace)),
        "avg_tokens": float(metrics.get("avg_tokens", 0.0)),
        "accuracy_per_token": float(metrics.get("accuracy_per_token", 0.0)),
        "trace_tokens": float(metrics.get("trace_tokens", 0.0)),
        "accuracy_per_trace_token": float(metrics.get("accuracy_per_trace_token", 0.0)),
        "judri_bridge_gate_enabled": float(metrics.get("judri_bridge_gate_enabled", 0.0)),
        "judri_bridge_gate_mean": float(metrics.get("judri_bridge_gate_mean", 0.0)),
        "judri_bridge_gate_active_mean": float(metrics.get("judri_bridge_gate_active_mean", 0.0)),
        "judri_bridge_gate_silenced_predicate_energy_mean": float(metrics.get("judri_bridge_gate_silenced_predicate_energy_mean", 0.0)),
    }
    report_path = Path(args.output_path) if args.output_path else run_dir / registry["report_names"]["actual_bridge"]
    validate_series_outputs("M", [registry["output_roots"]["actual_bridge"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M21.1.actual_bridge", "scripts/m21/run_m21_actual_bridge_suite.py"),
        "track": "M21.1",
        "registry": {
            "runner_script": registry["runner_scripts"]["actual_bridge"],
            "dag": registry["dags"]["actual_bridge"],
            "output_root": registry["output_roots"]["actual_bridge"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "source_report": str(args.train_report or args.suite_report or ""),
        "metrics": report_metrics,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
        "notes": ["Minimal M21 actual bridge assay maps dynamic bridi traces into bridge-ablation style metrics."],
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M21 actual bridge report written to {report_path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M21_REGISTRY["M21"]
    parser = argparse.ArgumentParser(description="Run the M21 minimal actual bridge suite.")
    parser.add_argument("--train-report", type=Path, default=None)
    parser.add_argument("--suite-report", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["actual_bridge"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_suite(parse_args())
