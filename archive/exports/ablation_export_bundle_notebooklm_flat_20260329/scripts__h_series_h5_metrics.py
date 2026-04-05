from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = int(round((pct / 100.0) * (len(s) - 1)))
    i = max(0, min(i, len(s) - 1))
    return float(s[i])


def extract_h5_metrics(path: Path) -> Dict[str, float]:
    payload = _read_json(path)
    if payload is None:
        return {}
    rows = payload.get("rows", [])
    if not isinstance(rows, list):
        rows = []
    total = len(rows)

    traces = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        t = r.get("logic_trace")
        traces.append(t.strip() if isinstance(t, str) else "")
    non_empty_traces = [t for t in traces if t]
    trace_token_counts = [float(len(t.split())) for t in non_empty_traces]
    trace_char_counts = [float(len(t)) for t in non_empty_traces]

    step_lengths: List[float] = []
    step_cosines: List[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        step_trace = row.get("step_cosine")
        if not isinstance(step_trace, list) or not step_trace:
            continue
        step_lengths.append(float(len(step_trace)))
        for s in step_trace:
            if isinstance(s, dict):
                step_cosines.append(float(s.get("cos_mean", 0.0)))

    errors = float(payload.get("summary", {}).get("errors", 0.0))
    denom = float(total) if total else 1.0
    return {
        "provenance_coverage": float(len(non_empty_traces)) / denom,
        "mean_logic_trace_tokens": _safe_mean(trace_token_counts),
        "mean_logic_trace_chars": _safe_mean(trace_char_counts),
        "step_trace_coverage": float(len(step_lengths)) / denom,
        "mean_step_trace_length": _safe_mean(step_lengths),
        "trace_mean_step_cosine": _safe_mean(step_cosines),
        "trace_p90_step_cosine": _percentile(step_cosines, 90.0),
        "error_rate": errors / denom,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract H5 provenance/stress metrics from true_coconut output JSON.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = extract_h5_metrics(args.input)
    if args.output is None:
        print(json.dumps(metrics, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
