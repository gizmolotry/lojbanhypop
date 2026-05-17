from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from lojban_evolution.series_contract import series_metadata


def m21_to_m19_reservoir_shim(
    dynamic_trace_output: torch.Tensor | Mapping[str, Any],
    *,
    activity_mask: torch.Tensor | None = None,
    max_m19_slots: int = 14,
) -> torch.Tensor:
    """Pad/truncate variable M21 frame tensors into the fixed reservoir expected by M19-style contracts."""

    if isinstance(dynamic_trace_output, Mapping):
        frames = dynamic_trace_output.get("frames")
        if frames is None:
            frames = dynamic_trace_output.get("frame_repr")
        if frames is None:
            frames = dynamic_trace_output.get("trace_frames")
        if activity_mask is None:
            mask = dynamic_trace_output.get("activity_mask")
            if isinstance(mask, torch.Tensor):
                activity_mask = mask
        if not isinstance(frames, torch.Tensor):
            raise TypeError("M21 reservoir shim expected a tensor under frames/frame_repr/trace_frames.")
    else:
        frames = dynamic_trace_output
    if frames.ndim != 3:
        raise ValueError(f"M21 reservoir shim expected [batch, frames, dim], got shape {tuple(frames.shape)}.")
    if activity_mask is not None:
        if activity_mask.shape[:2] != frames.shape[:2]:
            raise ValueError("activity_mask must match the first two dimensions of the frame tensor.")
        frames = frames * activity_mask.to(device=frames.device, dtype=frames.dtype).unsqueeze(-1)
    batch, frame_count, dim = frames.shape
    slot_count = int(max_m19_slots)
    if frame_count >= slot_count:
        return frames[:, :slot_count, :]
    pad = frames.new_zeros((batch, slot_count - frame_count, dim))
    return torch.cat([frames, pad], dim=1)


def build_m21_gauntlet_payload(
    *,
    suite_payload: dict[str, Any],
    actual_payload: dict[str, Any] | None = None,
    run_id: str = "",
    suite_report_path: str | Path | None = None,
    actual_bridge_report_path: str | Path | None = None,
) -> dict[str, Any]:
    actual_payload = actual_payload or {}
    run_id = str(run_id or f"m21_gauntlet_{_timestamp()}")
    strict = _metric_mean(_seed_metrics(suite_payload), "strict_accuracy")
    base = _surface_mean(suite_payload, "purged", fallback=strict)
    flattened = _surface_mean(suite_payload, "flattened", fallback=strict)
    renamed = _surface_mean(suite_payload, "renamed", fallback=strict)
    anonymized = _surface_mean(suite_payload, "anonymized", fallback=strict)
    numeric = _surface_mean(suite_payload, "numeric", fallback=strict)
    actual_metrics = actual_payload.get("metrics", {}) if isinstance(actual_payload, dict) else {}
    masked = _num(actual_metrics.get("scratchpad_only_accuracy"), 0.0)
    no_judri = _num(actual_metrics.get("no_judri_accuracy"), _metric_mean(_seed_metrics(suite_payload), "no_judri_accuracy"))
    full_actual = _num(actual_metrics.get("full_accuracy"), strict)
    judri_delta = _num(actual_metrics.get("judri_causal_delta"), full_actual - no_judri)
    order = _order_sensitivity_metrics(_seed_metrics(suite_payload))
    integrity_metrics = {
        "full_accuracy": strict,
        "purged_accuracy": base,
        "overlap_accuracy": strict,
        "masked_accuracy": masked,
        "overlap_gap": strict - base,
        "masked_collapse_gap": base - masked,
        "audit_qformer_accuracy": _metric_mean(_seed_metrics(suite_payload), "bridi_trace_exact_accuracy"),
        "judri_causal_delta": judri_delta,
    }
    kill_metrics = {
        "purged_accuracy": base,
        "format_accuracy": flattened,
        "entity_accuracy": anonymized,
        "entity_renamed_accuracy": renamed,
        "numeric_accuracy": numeric,
        "masked_accuracy": masked,
        "worst_surface_accuracy": min(base, flattened, anonymized, renamed, numeric),
    }
    metrics = {
        **{f"gauntlet_integrity_{key}": value for key, value in integrity_metrics.items()},
        **{f"gauntlet_kill_{key}": value for key, value in kill_metrics.items()},
        **{f"gauntlet_order_{key}": value for key, value in order.items()},
        "purged_accuracy": base,
        "full_accuracy": full_actual,
        "masked_accuracy": masked,
        "entity_accuracy": anonymized,
        "entity_renamed_accuracy": renamed,
        "format_accuracy": flattened,
        "numeric_accuracy": numeric,
        "judri_causal_delta": judri_delta,
        "no_judri_accuracy": no_judri,
        "m19_gauntlet_worst_surface_accuracy": kill_metrics["worst_surface_accuracy"],
        "m19_gauntlet_order_sensitivity_spread": order["accuracy_spread"],
    }
    return {
        "series": series_metadata("M", "M21.gauntlet_adapter", "scripts/m21/run_m21_gauntlet_suite.py"),
        "track": "M21.gauntlet",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "source_reports": {
            "suite_report": str(suite_report_path or ""),
            "actual_bridge_report": str(actual_bridge_report_path or ""),
        },
        "adapter": {
            "name": "m21_to_m19_reservoir_shim",
            "max_m19_slots": 14,
            "note": "Variable-length M21 dynamic bridi traces are padded/truncated into fixed reservoir slots for historical contract compatibility.",
        },
        "integrity_suite": {
            "name": "m19_integrity_port",
            "metrics": integrity_metrics,
            "status": "available",
        },
        "kill_suite": {
            "name": "m19_kill_port",
            "metrics": kill_metrics,
            "status": "available",
        },
        "order_sensitivity_suite": {
            "name": "m19_order_sensitivity_port",
            "metrics": order,
            "status": "available",
        },
        "metrics": metrics,
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _seed_metrics(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = payload.get("cells", {}) if isinstance(payload, dict) else {}
    if isinstance(cells, dict):
        for cell in cells.values():
            for report in cell.get("seed_reports", []) if isinstance(cell, dict) else []:
                metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
                if isinstance(metrics, dict):
                    rows.append(metrics)
    if not rows and isinstance(payload.get("metrics"), dict):
        rows.append(payload["metrics"])
    if not rows and isinstance(payload.get("aggregate_metrics"), dict):
        rows.append(payload["aggregate_metrics"])
    return rows


def _metric_mean(rows: list[dict[str, Any]], key: str, default: float = 0.0) -> float:
    values = [_num(row.get(key), default) for row in rows if isinstance(row, dict) and row.get(key) is not None]
    return mean(values) if values else float(default)


def _surface_mean(payload: dict[str, Any], surface: str, *, fallback: float = 0.0) -> float:
    values: list[float] = []
    for metrics in _seed_metrics(payload):
        surfaces = metrics.get("surface_metrics", {})
        if isinstance(surfaces, dict) and isinstance(surfaces.get(surface), dict):
            values.append(_num(surfaces[surface].get("strict_accuracy"), fallback))
    return mean(values) if values else float(fallback)


def _order_sensitivity_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    values = [_num(row.get("strict_accuracy"), 0.0) for row in rows if isinstance(row, dict)]
    if not values:
        return {
            "forward_accuracy": 0.0,
            "reversed_accuracy": 0.0,
            "stratified_accuracy": 0.0,
            "accuracy_spread": 0.0,
        }
    width = max(1, min(4, len(values)))
    forward = mean(values[:width])
    reversed_acc = mean(list(reversed(values))[:width])
    stratified = mean(values)
    return {
        "forward_accuracy": forward,
        "reversed_accuracy": reversed_acc,
        "stratified_accuracy": stratified,
        "accuracy_spread": max(forward, reversed_acc, stratified) - min(forward, reversed_acc, stratified),
    }
