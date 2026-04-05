from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from lojban_evolution.m_bridge_ablation_family import BRIDGE_ABLATION_REGISTRY
from lojban_evolution.m_symbiote_scratchpad_family import SYMBIOTE_SCRATCHPAD_REGISTRY
from lojban_evolution.series_contract import (
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_baseline_manifest,
    validate_series_outputs,
)


M3_TRACKS = ("M3.15d", "M3.16", "M3.17")


def _latest_named_file(root: Path, file_name: str) -> Path | None:
    if not root.exists():
        return None
    matches = sorted(root.rglob(file_name), key=lambda p: p.stat().st_mtime)
    if not matches:
        return None
    return matches[-1]


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _best_cell(rows: list[dict[str, Any]]) -> str | None:
    if not rows:
        return None

    def key_fn(row: dict[str, Any]) -> tuple[float, float, float]:
        acc_gain = row.get("accuracy_gain_vs_control")
        intervention = row.get("intervention_delta_gold")
        answer_gain = row.get("answer_delta_gain_vs_control")
        return (
            float(acc_gain) if acc_gain is not None else 0.0,
            float(intervention) if intervention is not None else -1e18,
            float(answer_gain) if answer_gain is not None else -1e18,
        )

    return max(rows, key=key_fn).get("cell")


def _normalize_m3_track(track: str, report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", {})
    control_metrics = cells.get("A", {}).get("metrics", {})
    control_val = cells.get("A", {}).get("validation_metrics", {})

    rows: list[dict[str, Any]] = []
    for cell_name in ("B", "C", "D"):
        cell_payload = cells.get(cell_name)
        if not isinstance(cell_payload, dict):
            continue
        metrics = cell_payload.get("metrics", {})
        validation_metrics = cell_payload.get("validation_metrics", {})
        seed2_metrics = cells.get(f"{cell_name}_seed2_eval", {}).get("metrics", {})
        train_metrics = cell_payload.get("train", {})
        promotion = payload.get("promotion_gates", {}).get(cell_name, {})

        control_acc = _safe_float(control_metrics.get("overall_accuracy")) or 0.0
        control_val_acc = _safe_float(control_val.get("overall_accuracy")) or 0.0
        control_answer_delta = _safe_float(control_metrics.get("mean_answer_delta")) or 0.0
        control_scope = _safe_float(control_metrics.get("mean_scope")) or 0.0
        eval_acc = _safe_float(metrics.get("overall_accuracy"))
        val_acc = _safe_float(validation_metrics.get("overall_accuracy"))
        answer_delta = _safe_float(metrics.get("mean_answer_delta"))
        scope = _safe_float(metrics.get("mean_scope"))
        seed2_acc = _safe_float(seed2_metrics.get("overall_accuracy"))

        row = {
            "track": track,
            "cell": cell_name,
            "report_path": str(report_path).replace("\\", "/"),
            "overall_accuracy": eval_acc,
            "accuracy_gain_vs_control": (eval_acc - control_acc) if eval_acc is not None else None,
            "validation_accuracy": val_acc,
            "validation_accuracy_gain_vs_control": (val_acc - control_val_acc) if val_acc is not None else None,
            "seed2_accuracy": seed2_acc,
            "seed_stability_gap": (eval_acc - seed2_acc) if (eval_acc is not None and seed2_acc is not None) else None,
            "answer_delta": answer_delta,
            "answer_delta_gain_vs_control": (answer_delta - control_answer_delta) if answer_delta is not None else None,
            "scope": scope,
            "scope_delta_vs_control": (scope - control_scope) if scope is not None else None,
            "intervention_delta_gold": _safe_float(metrics.get("mean_intervention_delta_gold")),
            "operator_entropy": _safe_float(metrics.get("mean_operator_entropy")),
            "top1_op_share": _safe_float(metrics.get("mean_top1_op_share")),
            "active_tokens": _safe_float(metrics.get("mean_active_tokens")),
            "active_op_count": _safe_float(metrics.get("mean_active_op_count")),
            "state_norm": _safe_float(metrics.get("mean_return_norm", metrics.get("mean_bias_norm"))),
            "state_gate": _safe_float(metrics.get("mean_return_gate", metrics.get("mean_bias_gate"))),
            "state_entropy": _safe_float(metrics.get("mean_return_attn_entropy")),
            "candidate_mass": _safe_float(metrics.get("mean_candidate_mass")),
            "cue_mass": _safe_float(metrics.get("mean_cue_mass")),
            "alignment_similarity": _safe_float(metrics.get("mean_alignment_similarity")),
            "train_answer_path_loss": _safe_float(train_metrics.get("answer_path_loss")),
            "train_answer_delta": _safe_float(train_metrics.get("answer_delta")),
            "promote_to_next": bool(promotion.get("promote_to_next", False)),
        }
        rows.append(row)

    best_cell = _best_cell(rows)
    best_row = next((row for row in rows if row.get("cell") == best_cell), None)
    diagnosis: list[str] = []
    if best_row is not None and (best_row.get("accuracy_gain_vs_control") or 0.0) <= 0.0:
        diagnosis.append("No intervention in this track improved held-out accuracy over control.")
    if any((row.get("intervention_delta_gold") or 0.0) < -0.01 for row in rows):
        diagnosis.append("At least one cell harms gold-answer preference materially, indicating overexposure or miscalibrated coupling.")
    if any(bool(row.get("promote_to_next")) for row in rows):
        diagnosis.append("A promotion gate passed in this track.")
    else:
        diagnosis.append("Promotion gates blocked every intervention cell in this track.")

    return {
        "track": track,
        "family": BRIDGE_ABLATION_REGISTRY[track]["family"],
        "implementation_label": BRIDGE_ABLATION_REGISTRY[track]["implementation_label"],
        "tensor_flow": BRIDGE_ABLATION_REGISTRY[track]["tensor_flow"],
        "runner_script": BRIDGE_ABLATION_REGISTRY[track]["runner_script"],
        "dag": BRIDGE_ABLATION_REGISTRY[track]["dag"],
        "report_path": str(report_path).replace("\\", "/"),
        "control": {
            "overall_accuracy": _safe_float(control_metrics.get("overall_accuracy")),
            "validation_accuracy": _safe_float(control_val.get("overall_accuracy")),
            "answer_delta": _safe_float(control_metrics.get("mean_answer_delta")),
            "scope": _safe_float(control_metrics.get("mean_scope")),
            "operator_entropy": _safe_float(control_metrics.get("mean_operator_entropy")),
            "top1_op_share": _safe_float(control_metrics.get("mean_top1_op_share")),
        },
        "cells": rows,
        "best_cell": best_cell,
        "best_cell_snapshot": best_row,
        "diagnosis": diagnosis,
    }


def _normalize_m318(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", {})
    control = cells.get("A", {}).get("metrics", {})
    rows: list[dict[str, Any]] = []
    for cell_name in ("B", "C", "D", "E"):
        cell_payload = cells.get(cell_name, {})
        metrics = cell_payload.get("metrics", {})
        validation_metrics = cell_payload.get("validation_metrics", {})
        row = {
            "track": "M3.18",
            "cell": cell_name,
            "report_path": str(report_path).replace("\\", "/"),
            "overall_accuracy": _safe_float(metrics.get("overall_accuracy")),
            "accuracy_gain_vs_control": (_safe_float(metrics.get("overall_accuracy")) or 0.0) - (_safe_float(control.get("overall_accuracy")) or 0.0),
            "validation_accuracy": _safe_float(validation_metrics.get("overall_accuracy")),
            "validation_accuracy_gain_vs_control": (_safe_float(validation_metrics.get("overall_accuracy")) or 0.0) - (_safe_float(cells.get("A", {}).get("validation_metrics", {}).get("overall_accuracy")) or 0.0),
            "answer_delta": _safe_float(metrics.get("mean_answer_delta")),
            "answer_delta_gain_vs_control": (_safe_float(metrics.get("mean_answer_delta")) or 0.0) - (_safe_float(control.get("mean_answer_delta")) or 0.0),
            "scope": _safe_float(metrics.get("mean_scope")),
            "scope_delta_vs_control": (_safe_float(metrics.get("mean_scope")) or 0.0) - (_safe_float(control.get("mean_scope")) or 0.0),
            "intervention_delta_gold": _safe_float(metrics.get("mean_intervention_delta_gold")),
            "state_norm": _safe_float(metrics.get("mean_residual_norm")),
            "state_gate": _safe_float(metrics.get("mean_return_gate")),
            "state_entropy": _safe_float(metrics.get("mean_return_attn_entropy")),
            "train_answer_path_loss": _safe_float(cell_payload.get("train", {}).get("answer_path_loss")),
            "train_answer_delta": _safe_float(cell_payload.get("train", {}).get("answer_delta")),
            "resume_first_token_accuracy": _safe_float(metrics.get("resume_first_token_accuracy")),
            "english_fluency_score": _safe_float(metrics.get("english_fluency_score")),
            "contamination_rate": _safe_float(metrics.get("contamination_rate")),
            "loop_rate": _safe_float(metrics.get("loop_rate")),
            "scratchpad_bleed_rate": None,
            "scratchpad_attention_mass": None,
            "promote_to_next": bool(payload.get("promotion_gates", {}).get(cell_name, {}).get("promote_to_next", False)),
        }
        rows.append(row)
    return {
        "track": "M3.18",
        "family": "reentry_architecture",
        "implementation_label": "decoder_reentry_resume",
        "runner_script": "scripts/run_m3_18_decoder_reentry_resume.py",
        "dag": "airflow/dags/lojban_m3_18_decoder_reentry_resume_dag.py",
        "report_path": str(report_path).replace("\\", "/"),
        "control": {
            "overall_accuracy": _safe_float(control.get("overall_accuracy")),
            "validation_accuracy": _safe_float(cells.get("A", {}).get("validation_metrics", {}).get("overall_accuracy")),
            "answer_delta": _safe_float(control.get("mean_answer_delta")),
            "scope": _safe_float(control.get("mean_scope")),
        },
        "cells": rows,
        "best_cell": _best_cell(rows),
    }


def _normalize_m319(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    control_acc = _safe_float(payload.get("cells", {}).get("D0", {}).get("metrics", {}).get("overall_accuracy")) or 0.0
    control_answer = _safe_float(payload.get("cells", {}).get("D0", {}).get("metrics", {}).get("mean_answer_delta")) or 0.0
    for cell_name in ("D0", "D1", "D2", "D3"):
        cell_payload = payload.get("cells", {}).get(cell_name, {})
        metrics = cell_payload.get("metrics", {})
        row = {
            "track": "M3.19",
            "cell": cell_name,
            "report_path": str(report_path).replace("\\", "/"),
            "overall_accuracy": _safe_float(metrics.get("overall_accuracy")),
            "accuracy_gain_vs_control": (_safe_float(metrics.get("overall_accuracy")) or 0.0) - control_acc,
            "validation_accuracy": None,
            "validation_accuracy_gain_vs_control": None,
            "answer_delta": _safe_float(metrics.get("mean_answer_delta")),
            "answer_delta_gain_vs_control": (_safe_float(metrics.get("mean_answer_delta")) or 0.0) - control_answer,
            "scope": None,
            "scope_delta_vs_control": None,
            "intervention_delta_gold": _safe_float(metrics.get("mean_intervention_delta_gold")),
            "state_norm": _safe_float(metrics.get("mean_residual_norm")),
            "state_gate": None,
            "state_entropy": None,
            "train_answer_path_loss": _safe_float(cell_payload.get("train", {}).get("answer_path_loss")),
            "train_answer_delta": _safe_float(cell_payload.get("train", {}).get("answer_delta")),
            "resume_first_token_accuracy": _safe_float(metrics.get("resume_first_token_accuracy")),
            "english_fluency_score": _safe_float(metrics.get("english_fluency_score")),
            "contamination_rate": _safe_float(metrics.get("contamination_rate")),
            "loop_rate": _safe_float(metrics.get("loop_rate")),
            "scratchpad_bleed_rate": None,
            "scratchpad_attention_mass": None,
            "promote_to_next": False,
        }
        rows.append(row)
    return {
        "track": "M3.19",
        "family": "reentry_architecture",
        "implementation_label": "d_mainline_grid",
        "runner_script": "scripts/run_m3_19_d_mainline_grid.py",
        "dag": "airflow/dags/lojban_m3_19_d_mainline_grid_dag.py",
        "report_path": str(report_path).replace("\\", "/"),
        "control": {
            "overall_accuracy": control_acc,
            "answer_delta": control_answer,
        },
        "cells": rows,
        "best_cell": _best_cell(rows),
    }


def _normalize_m14(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    cells = payload.get("cells", {})
    control = cells.get("A", {}).get("metrics", {})
    rows: list[dict[str, Any]] = []
    for cell_name in ("B", "C", "D", "E"):
        cell_payload = cells.get(cell_name, {})
        metrics = cell_payload.get("metrics", {})
        validation_metrics = cell_payload.get("validation_metrics", {})
        row = {
            "track": "M14",
            "cell": cell_name,
            "report_path": str(report_path).replace("\\", "/"),
            "overall_accuracy": _safe_float(metrics.get("overall_accuracy")),
            "accuracy_gain_vs_control": (_safe_float(metrics.get("overall_accuracy")) or 0.0) - (_safe_float(control.get("overall_accuracy")) or 0.0),
            "validation_accuracy": _safe_float(validation_metrics.get("overall_accuracy")),
            "validation_accuracy_gain_vs_control": (_safe_float(validation_metrics.get("overall_accuracy")) or 0.0) - (_safe_float(cells.get("A", {}).get("validation_metrics", {}).get("overall_accuracy")) or 0.0),
            "answer_delta": _safe_float(metrics.get("mean_answer_delta")),
            "answer_delta_gain_vs_control": (_safe_float(metrics.get("mean_answer_delta")) or 0.0) - (_safe_float(control.get("mean_answer_delta")) or 0.0),
            "scope": _safe_float(metrics.get("mean_scope")),
            "scope_delta_vs_control": (_safe_float(metrics.get("mean_scope")) or 0.0) - (_safe_float(control.get("mean_scope")) or 0.0),
            "intervention_delta_gold": _safe_float(metrics.get("mean_intervention_delta_gold")),
            "state_norm": _safe_float(metrics.get("scratchpad_residual_norm")),
            "state_gate": _safe_float(metrics.get("scratchpad_gate_mean")),
            "state_entropy": _safe_float(metrics.get("mean_return_attn_entropy")),
            "train_answer_path_loss": _safe_float(cell_payload.get("train", {}).get("answer_path_loss")),
            "train_answer_delta": _safe_float(cell_payload.get("train", {}).get("answer_delta")),
            "resume_first_token_accuracy": _safe_float(metrics.get("resume_first_token_accuracy")),
            "english_fluency_score": _safe_float(metrics.get("english_fluency_score")),
            "contamination_rate": _safe_float(metrics.get("contamination_rate")),
            "loop_rate": _safe_float(metrics.get("loop_rate")),
            "scratchpad_bleed_rate": _safe_float(metrics.get("scratchpad_bleed_rate")),
            "scratchpad_attention_mass": _safe_float(metrics.get("scratchpad_attention_mass")),
            "promote_to_next": bool(payload.get("promotion_gates", {}).get(cell_name, {}).get("promote_to_next", False)),
        }
        rows.append(row)
    return {
        "track": "M14",
        "family": SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["family"],
        "implementation_label": SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["implementation_label"],
        "runner_script": SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["runner_script"],
        "dag": SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["dag"],
        "report_path": str(report_path).replace("\\", "/"),
        "control": {
            "overall_accuracy": _safe_float(control.get("overall_accuracy")),
            "validation_accuracy": _safe_float(cells.get("A", {}).get("validation_metrics", {}).get("overall_accuracy")),
            "answer_delta": _safe_float(control.get("mean_answer_delta")),
            "scope": _safe_float(control.get("mean_scope")),
        },
        "cells": rows,
        "best_cell": _best_cell(rows),
    }


def _history_entry_snapshot(payload: dict[str, Any], lookup_key: str) -> dict[str, Any] | None:
    lookup_index = payload.get("lookup_index", {})
    entries = payload.get("entries", [])
    canonical_id = lookup_index.get(lookup_key)
    if canonical_id is None:
        for entry in entries:
            if entry.get("canonical_id") == lookup_key:
                canonical_id = lookup_key
                break
    if canonical_id is None:
        return None
    return next((entry for entry in entries if entry.get("canonical_id") == canonical_id), None)


def _normalize_history_manifest(history_manifest_path: Path | None) -> dict[str, Any] | None:
    if history_manifest_path is None or not history_manifest_path.exists():
        return None
    payload = json.loads(history_manifest_path.read_text(encoding="utf-8"))
    representative_keys = [
        "legacy:A",
        "legacy:H3",
        "legacy:H5.2b",
        "M2.A",
        "phase5.train:phase5_full",
        "M3.18.D",
        "M14.C",
    ]
    snapshots: dict[str, Any] = {}
    for key in representative_keys:
        entry = _history_entry_snapshot(payload, key)
        if entry is None:
            continue
        metrics = entry.get("metrics_summary", {})
        snapshots[key] = {
            "canonical_id": entry.get("canonical_id"),
            "family": entry.get("family"),
            "reproducibility_status": entry.get("reproducibility_status"),
            "held_out_accuracy": metrics.get("best_held_out_accuracy"),
            "logical_accuracy": metrics.get("best_logical_accuracy"),
            "final_ce_loss": metrics.get("lowest_final_ce_loss"),
        }
    return {
        "history_manifest_path": str(history_manifest_path).replace("\\", "/"),
        "entry_count": len(payload.get("entries", [])),
        "all_evidence": payload.get("history_slices", {}).get("all", {}),
        "artifact_only": payload.get("history_slices", {}).get("artifact_only", {}),
        "runnable_only": payload.get("history_slices", {}).get("runnable_only", {}),
        "family_group_counts": payload.get("family_group_counts", {}),
        "historical_gap_count": len(payload.get("historical_gaps", [])),
        "representative_snapshots": snapshots,
    }


def _normalize_m11(manifest_path: Path | None, bridge_report_path: Path | None, floor_lock_path: Path | None, publication_path: Path | None) -> dict[str, Any]:
    manifest_payload: dict[str, Any] = {}
    bridge_payload: dict[str, Any] = {}
    floor_payload: dict[str, Any] = {}
    publication_payload: dict[str, Any] = {}
    if manifest_path and manifest_path.exists():
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bridge_report_path and bridge_report_path.exists():
        bridge_payload = json.loads(bridge_report_path.read_text(encoding="utf-8"))
    if floor_lock_path and floor_lock_path.exists():
        floor_payload = json.loads(floor_lock_path.read_text(encoding="utf-8"))
    if publication_path and publication_path.exists():
        publication_payload = json.loads(publication_path.read_text(encoding="utf-8"))

    metrics = manifest_payload.get("metrics", {})
    if "mean_accuracy" in metrics:
        headline_accuracy = _safe_float(metrics.get("mean_accuracy"))
        headline_samples = metrics.get("n_samples")
        headline_macro_f1 = _safe_float(metrics.get("macro_f1"))
        headline_seeds = metrics.get("seeds")
    else:
        headline_accuracy = _safe_float(metrics.get("accuracy"))
        headline_samples = metrics.get("num_samples")
        headline_macro_f1 = _safe_float(metrics.get("macro_f1"))
        headline_seeds = None

    diagnosis: list[str] = []
    if headline_accuracy is not None:
        diagnosis.append(f"Native M11 discriminative path reaches headline accuracy {headline_accuracy:.4f}.")
    if bridge_payload:
        diagnosis.append("Small-sample bridge audit is available for quick smoke comparison.")
    if floor_payload:
        diagnosis.append("Floor-lock audit provides the larger-sample operational reference.")

    return {
        "track": "M11.discriminative",
        "family": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["family"],
        "implementation_label": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["implementation_label"],
        "tensor_flow": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["tensor_flow"],
        "runner_script": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["runner_script"],
        "dag": BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["dag"],
        "manifest_path": str(manifest_path).replace("\\", "/") if manifest_path else None,
        "bridge_audit_path": str(bridge_report_path).replace("\\", "/") if bridge_report_path else None,
        "floor_lock_path": str(floor_lock_path).replace("\\", "/") if floor_lock_path else None,
        "publication_metrics_path": str(publication_path).replace("\\", "/") if publication_path else None,
        "headline_accuracy": headline_accuracy,
        "headline_macro_f1": headline_macro_f1,
        "headline_num_samples": headline_samples,
        "headline_seeds": headline_seeds,
        "bridge_audit_accuracy": _safe_float(bridge_payload.get("accuracy")),
        "bridge_audit_macro_f1": _safe_float(bridge_payload.get("macro_f1")),
        "bridge_audit_num_samples": bridge_payload.get("num_samples"),
        "floor_lock_accuracy": _safe_float(floor_payload.get("accuracy")),
        "floor_lock_macro_f1": _safe_float(floor_payload.get("macro_f1")),
        "floor_lock_num_samples": floor_payload.get("num_samples"),
        "publication_mean_acc": _safe_float(publication_payload.get("mean_acc")),
        "publication_std_acc": _safe_float(publication_payload.get("std_acc")),
        "publication_accuracies": publication_payload.get("accuracies"),
        "diagnosis": diagnosis,
    }


def _suite_diagnosis(m3_tracks: list[dict[str, Any]], reentry_tracks: list[dict[str, Any]], m11_track: dict[str, Any] | None) -> list[str]:
    lines: list[str] = []
    harmful_cells: list[str] = []
    neutral_cells: list[str] = []
    val_only_spikes: list[str] = []

    for track in [*m3_tracks, *reentry_tracks]:
        for row in track.get("cells", []):
            track_cell = f"{track.get('track')}/{row.get('cell')}"
            intervention = row.get("intervention_delta_gold")
            acc_gain = row.get("accuracy_gain_vs_control")
            val_gain = row.get("validation_accuracy_gain_vs_control")
            if intervention is not None and intervention < -0.01:
                harmful_cells.append(track_cell)
            if intervention is not None and acc_gain is not None and abs(float(intervention)) <= 0.01 and abs(float(acc_gain)) <= 1e-9:
                neutral_cells.append(track_cell)
            if val_gain is not None and acc_gain is not None and float(val_gain) >= 0.5 and float(acc_gain) <= 0.0:
                val_only_spikes.append(track_cell)

    if harmful_cells:
        lines.append("Harmful cells are concentrated where the sidecar stays too exposed: " + ", ".join(harmful_cells) + ".")
    if neutral_cells:
        lines.append("Near-neutral cells preserve control behavior but do not create measurable gains: " + ", ".join(neutral_cells[:6]) + ".")
    if val_only_spikes:
        lines.append("Validation-only spikes without held-out lift suggest overfitting or split-specific coupling: " + ", ".join(val_only_spikes) + ".")

    best_rows: list[tuple[str, dict[str, Any]]] = []
    for track in [*m3_tracks, *reentry_tracks]:
        best_cell = track.get("best_cell")
        row = next((item for item in track.get("cells", []) if item.get("cell") == best_cell), None)
        if row is not None:
            best_rows.append((str(track.get("track")), row))
    if best_rows:
        best_line = "; ".join(
            f"{track} best={row.get('cell')} acc_gain={_round(row.get('accuracy_gain_vs_control'))} intervention={_round(row.get('intervention_delta_gold'))}"
            for track, row in best_rows
        )
        lines.append("Best-per-track snapshot: " + best_line + ".")

    if m11_track is not None and m11_track.get("headline_accuracy") is not None:
        lines.append(
            "The M11 native discriminative branch is materially stronger than the generative bridge ablations,"
            f" with headline accuracy {_round(m11_track.get('headline_accuracy'))}, floor-lock accuracy {_round(m11_track.get('floor_lock_accuracy'))},"
            f" and publication mean {_round(m11_track.get('publication_mean_acc'))}."
        )
    m14_track = next((track for track in reentry_tracks if str(track.get("track")) == "M14"), None)
    if m14_track is not None:
        best_cell = m14_track.get("best_cell")
        best_row = next((item for item in m14_track.get("cells", []) if item.get("cell") == best_cell), None)
        if best_row is not None:
            lines.append(
                "M14 scratchpad best snapshot: "
                f"cell {best_cell} intervention={_round(best_row.get('intervention_delta_gold'))} "
                f"first_token={_round(best_row.get('resume_first_token_accuracy'))} "
                f"fluency={_round(best_row.get('english_fluency_score'))} "
                f"scratchpad_bleed={_round(best_row.get('scratchpad_bleed_rate'))}."
            )
    return lines


def _write_csv(csv_path: Path, track_groups: list[dict[str, Any]]) -> None:
    fieldnames = [
        "track",
        "cell",
        "overall_accuracy",
        "accuracy_gain_vs_control",
        "validation_accuracy",
        "validation_accuracy_gain_vs_control",
        "seed2_accuracy",
        "seed_stability_gap",
        "answer_delta",
        "answer_delta_gain_vs_control",
        "scope",
        "scope_delta_vs_control",
        "intervention_delta_gold",
        "operator_entropy",
        "top1_op_share",
        "active_tokens",
        "active_op_count",
        "state_norm",
        "state_gate",
        "state_entropy",
        "candidate_mass",
        "cue_mass",
        "alignment_similarity",
        "resume_first_token_accuracy",
        "english_fluency_score",
        "contamination_rate",
        "loop_rate",
        "scratchpad_bleed_rate",
        "scratchpad_attention_mass",
        "train_answer_path_loss",
        "train_answer_delta",
        "promote_to_next",
        "report_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for track in track_groups:
            for row in track.get("cells", []):
                writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the latest bridge-series ablation reports into a unified DAG/ledger diagnosis suite.")
    parser.add_argument("--baseline-manifest", type=Path, default=Path("docs/baselines/m_series_bridge_baseline_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite"))
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--history-manifest", type=Path, default=None)
    parser.add_argument("--m3-15d-report", type=Path, default=None)
    parser.add_argument("--m3-16-report", type=Path, default=None)
    parser.add_argument("--m3-17-report", type=Path, default=None)
    parser.add_argument("--m3-18-report", type=Path, default=None)
    parser.add_argument("--m3-19-report", type=Path, default=None)
    parser.add_argument("--m14-report", type=Path, default=None)
    parser.add_argument("--m11-manifest", type=Path, default=None)
    parser.add_argument("--m11-bridge-audit", type=Path, default=Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_bridge_audit.json"))
    parser.add_argument("--m11-floor-lock", type=Path, default=Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_floor_lock.json"))
    parser.add_argument("--m11-publication-metrics", type=Path, default=Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/final_publication_metrics.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_manifest = validate_baseline_manifest(args.baseline_manifest, series_id="M")
    assert_output_path_allowed("M", args.output_root)

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root / run_id
    validate_series_outputs("M", [args.output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    report_paths: dict[str, Path] = {}
    for track, override in {
        "M3.15d": args.m3_15d_report,
        "M3.16": args.m3_16_report,
        "M3.17": args.m3_17_report,
    }.items():
        if override is not None:
            report_paths[track] = override
            continue
        meta = BRIDGE_ABLATION_REGISTRY[track]
        latest = _latest_named_file(Path(meta["search_root"]), str(meta["report_name"]))
        if latest is None:
            raise FileNotFoundError(f"Could not find latest report for {track} under {meta['search_root']}")
        report_paths[track] = latest

    m3_tracks = [_normalize_m3_track(track, report_paths[track]) for track in M3_TRACKS]

    reentry_paths: dict[str, Path] = {}
    reentry_overrides = {
        "M3.18": args.m3_18_report,
        "M3.19": args.m3_19_report,
        "M14": args.m14_report,
    }
    reentry_search = {
        "M3.18": ("artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume", "m3_18_report.json"),
        "M3.19": ("artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid", "m3_19_grid_report.json"),
        "M14": (SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["output_root"], SYMBIOTE_SCRATCHPAD_REGISTRY["M14"]["report_name"]),
    }
    for track, override in reentry_overrides.items():
        if override is not None:
            reentry_paths[track] = override
            continue
        search_root, report_name = reentry_search[track]
        latest = _latest_named_file(Path(search_root), str(report_name))
        if latest is not None:
            reentry_paths[track] = latest

    reentry_tracks: list[dict[str, Any]] = []
    if "M3.18" in reentry_paths:
        reentry_tracks.append(_normalize_m318(reentry_paths["M3.18"]))
    if "M3.19" in reentry_paths:
        reentry_tracks.append(_normalize_m319(reentry_paths["M3.19"]))
    if "M14" in reentry_paths:
        reentry_tracks.append(_normalize_m14(reentry_paths["M14"]))

    m11_manifest = args.m11_manifest
    if m11_manifest is None:
        suite_manifest = _latest_named_file(Path(BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["search_root"]), str(BRIDGE_ABLATION_REGISTRY["M11.discriminative"]["manifest_name"]))
        fallback_manifest = Path("archive/results/m10/active/RESULTS_M10_FINAL_AUDIT/m11_discriminative_manifest.json")
        m11_manifest = suite_manifest if suite_manifest is not None else (fallback_manifest if fallback_manifest.exists() else None)

    m11_track = _normalize_m11(
        m11_manifest,
        args.m11_bridge_audit if args.m11_bridge_audit.exists() else None,
        args.m11_floor_lock if args.m11_floor_lock.exists() else None,
        args.m11_publication_metrics if args.m11_publication_metrics.exists() else None,
    )

    history_manifest = args.history_manifest
    if history_manifest is None:
        history_manifest = _latest_named_file(
            Path("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill"),
            "ablation_history_manifest.json",
        )
    history_summary = _normalize_history_manifest(history_manifest)

    suite_payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M.bridge_suite", "scripts/run_m_bridge_ablation_test_suite.py"),
        "track": "M.bridge_suite",
        "lineage": lineage_metadata(
            "eval_only",
            checkpoint_in=None,
            checkpoint_out=None,
            dataset_profile="bridge_ablation_reports",
            difficulty_tier="mixed",
        ),
        "baseline_manifest": str(args.baseline_manifest).replace("\\", "/"),
        "baseline_id": str(baseline_manifest.get("baseline_id", "")),
        "config": {
            "output_root": str(args.output_root).replace("\\", "/"),
            "run_id": run_id,
        },
        "implementation_registry": BRIDGE_ABLATION_REGISTRY,
        "m3_tracks": m3_tracks,
        "reentry_tracks": reentry_tracks,
        "m11_track": m11_track,
        "history_summary": history_summary,
        "diagnosis": _suite_diagnosis(m3_tracks, reentry_tracks, m11_track),
    }

    manifest_path = run_dir / "m_bridge_ablation_suite_manifest.json"
    manifest_path.write_text(json.dumps(suite_payload, indent=2), encoding="utf-8")

    csv_path = run_dir / "m_bridge_ablation_cell_metrics.csv"
    _write_csv(csv_path, [*m3_tracks, *reentry_tracks])

    manifest_str = str(manifest_path).replace("\\", "/")
    csv_str = str(csv_path).replace("\\", "/")
    md_lines = [
        "# M Bridge Ablation Test Suite",
        "",
        f"- manifest: `{manifest_str}`",
        f"- cell_metrics_csv: `{csv_str}`",
        "",
        "## M3 Track Summary",
        "",
    ]
    for track in m3_tracks:
        control = track.get("control", {})
        md_lines.append(f"### {track.get('track')}")
        md_lines.append(f"- control_accuracy: `{_round(control.get('overall_accuracy'))}`")
        md_lines.append(f"- control_answer_delta: `{_round(control.get('answer_delta'))}`")
        md_lines.append(f"- best_cell: `{track.get('best_cell')}`")
        for row in track.get("cells", []):
            md_lines.append(
                f"- {row.get('cell')}: acc=`{_round(row.get('overall_accuracy'))}` acc_gain=`{_round(row.get('accuracy_gain_vs_control'))}` answer_delta_gain=`{_round(row.get('answer_delta_gain_vs_control'))}` intervention_gold=`{_round(row.get('intervention_delta_gold'))}` promote=`{row.get('promote_to_next')}`"
            )
        md_lines.append("")

    if reentry_tracks:
        md_lines.extend([
            "## Reentry Track Summary",
            "",
        ])
        for track in reentry_tracks:
            control = track.get("control", {})
            md_lines.append(f"### {track.get('track')}")
            md_lines.append(f"- control_accuracy: `{_round(control.get('overall_accuracy'))}`")
            md_lines.append(f"- control_answer_delta: `{_round(control.get('answer_delta'))}`")
            md_lines.append(f"- best_cell: `{track.get('best_cell')}`")
            for row in track.get("cells", []):
                md_lines.append(
                    f"- {row.get('cell')}: acc=`{_round(row.get('overall_accuracy'))}` acc_gain=`{_round(row.get('accuracy_gain_vs_control'))}` "
                    f"ftok=`{_round(row.get('resume_first_token_accuracy'))}` fluency=`{_round(row.get('english_fluency_score'))}` "
                    f"loop=`{_round(row.get('loop_rate'))}` intervention_gold=`{_round(row.get('intervention_delta_gold'))}` "
                    f"s_bleed=`{_round(row.get('scratchpad_bleed_rate'))}` promote=`{row.get('promote_to_next')}`"
                )
            md_lines.append("")

    md_lines.extend([
        "## M11 Summary",
        "",
        f"- headline_accuracy: `{_round(m11_track.get('headline_accuracy'))}`",
        f"- headline_macro_f1: `{_round(m11_track.get('headline_macro_f1'))}`",
        f"- floor_lock_accuracy: `{_round(m11_track.get('floor_lock_accuracy'))}`",
        f"- publication_mean_acc: `{_round(m11_track.get('publication_mean_acc'))}`",
        "",
    ])

    if history_summary is not None:
        md_lines.extend([
            "## History Backfill",
            "",
            f"- history_manifest: `{history_summary.get('history_manifest_path')}`",
            f"- all_evidence_entries: `{history_summary.get('all_evidence', {}).get('entry_count')}`",
            f"- artifact_only_entries: `{history_summary.get('artifact_only', {}).get('entry_count')}`",
            f"- runnable_only_entries: `{history_summary.get('runnable_only', {}).get('entry_count')}`",
            f"- l_series_entries: `{history_summary.get('family_group_counts', {}).get('l_series')}`",
            f"- historical_gap_count: `{history_summary.get('historical_gap_count')}`",
        ])
        for key, snapshot in history_summary.get("representative_snapshots", {}).items():
            md_lines.append(
                f"- {key}: canonical=`{snapshot.get('canonical_id')}` repro=`{snapshot.get('reproducibility_status')}` "
                f"acc=`{_round(snapshot.get('held_out_accuracy'))}` logical=`{_round(snapshot.get('logical_accuracy'))}` "
                f"ce=`{_round(snapshot.get('final_ce_loss'))}`"
            )
        md_lines.extend([
            "",
        ])

    md_lines.extend([
        "## Diagnosis",
        "",
    ])
    for line in suite_payload["diagnosis"]:
        md_lines.append(f"- {line}")

    summary_path = run_dir / "m_bridge_ablation_suite_summary.md"
    summary_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"M bridge ablation suite complete: {run_dir}")


if __name__ == "__main__":
    main()


