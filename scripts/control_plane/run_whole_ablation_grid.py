from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.control_plane.artifact_io import (
    latest_named_manifest as _artifact_latest_named_manifest,
    read_json_optional as _artifact_read_json_optional,
    repo_relative_or_string as _artifact_repo_relative_or_string,
)
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "whole_ablation_grid"
DEFAULT_DOC_OUTPUT = REPO_ROOT / "docs" / "history" / "reports" / "WHOLE_ABLATION_GRID_LATEST.md"
DEFAULT_HISTORY_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_history_backfill"
DEFAULT_PROGRAM_SPINE_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_program_spine"
DEFAULT_LEGACY_GRID_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "legacy_grid"
DEFAULT_M_BRIDGE_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m_bridge_ablation_test_suite"
DEFAULT_M14_REPORT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m14_5_decompressor" / "m14_5_report.json"
DEFAULT_M18_REPORT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m18_controller_family" / "m18_frontier_audits_20260409" / "m18_family_report.json"
DEFAULT_DIRECT_UNIFIED_EVAL_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "direct_unified_eval"
DEFAULT_M19_MAINLINE_REPORT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_mainline_suite" / "m19_mainline_20260409_v2" / "m19_mainline_report.json"
DEFAULT_M19_4_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_dynamic_pacing_suite"
DEFAULT_M19_4_BENCH_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_dynamic_pacing_benchmark"
DEFAULT_M19_ISOLATION_REPORT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_isolation_grid" / "m19_isolation_grid_20260409_v2" / "m19_isolation_grid_report.json"
DEFAULT_M19_ZH_SUMMARY = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_zh_comparison" / "m19_zh_comparison_20260409" / "m19_zh_comparison_summary.json"
DEFAULT_M19_REPLICATION_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_replication_suite"
DEFAULT_M19_KILL_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m19_kill_test_suite"
DEFAULT_M20_SUITE_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m20_dictionary_first_suite"
DEFAULT_M20_LOCK_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m20_lock_suite"
DEFAULT_M20_INDUCTION_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m20_predicate_induction"
DEFAULT_M21_SUITE_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_dynamic_bridi_suite"
DEFAULT_M21_SYNTHETIC_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_synthetic_assay_suite"
DEFAULT_M21_ACTUAL_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_actual_bridge_suite"
DEFAULT_M21_LOCK_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m21_lock_suite"
DEFAULT_M22_GENERALIZATION_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m22_semantic_generalization"
DEFAULT_M23_RELEVANCE_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m23_relevance_suite"
DEFAULT_M24_COMPRESSION_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m24_substrate_compression"
DEFAULT_M25_EMERGENT_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "m25_emergent_bridi"

STAGE_ORDER = [
    "A-G",
    "H",
    "H5",
    "J",
    "L",
    "J/L Hypercube",
    "Phase Eval",
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "M10",
    "M11",
    "M14",
    "M18",
    "M19",
    "M20",
    "M21",
    "M22",
    "M23",
    "M24",
    "M25",
    "Control Plane",
]

PREFERRED_REPORT_GLOBS = ["*report.json", "*manifest.json", "*summary.json", "*.json"]
KEY_METRICS = [
    "headline_accuracy",
    "headline_macro_f1",
    "overall_accuracy",
    "overall_phrase_accuracy",
    "held_out_accuracy",
    "logical_accuracy",
    "mean_intervention_delta_gold",
    "resume_first_token_accuracy",
    "english_fluency_score",
    "loop_rate",
    "contamination_rate",
    "lift_vs_en_cot",
    "lift_vs_random",
    "avg_tokens",
    "audit_qformer_accuracy",
    "typed_family_accuracy",
    "masked_pointer_zero_rate",
    "family_slot_entropy",
    "symbolic_trace_alignment",
    "strict_accuracy",
    "lock_pass_rate",
    "dictionary_coverage",
    "factorized_exact_accuracy",
    "predicate_identity_stability",
    "brivi_gate_accuracy",
    "argument_binding_accuracy",
    "accuracy_per_token",
    "predicate_pointer_radial_gap",
    "family_radius_violation_rate",
    "hyperbolic_geodesic_margin",
    "hyperbolic_projection_clip_rate",
    "constraint_scope",
    "constraint_identity",
    "full_total_regularizer",
    "bridi_trace_exact_accuracy",
    "gismu_accuracy",
    "cmavo_accuracy",
    "judri_binding_accuracy",
    "frame_count_mae",
    "stop_accuracy",
    "mean_active_frames",
    "active_gismu_count",
    "active_cmavo_count",
    "active_code_fraction_reachable",
    "active_code_fraction_total",
    "actual_bridge_transfer_score",
    "no_cmavo_accuracy",
    "no_judri_accuracy",
    "gismu_only_accuracy",
    "random_trace_accuracy",
    "scratchpad_only_accuracy",
    "frame_drop_delta",
    "cmavo_causal_delta",
    "judri_causal_delta",
    "trace_tokens",
    "accuracy_per_trace_token",
    "decoy_relation_ood_accuracy",
    "worst_surface_accuracy",
    "relevance_top1_accuracy",
    "relevance_margin",
    "loss_trace_exact_surrogate",
    "trace_exact_surrogate_weight",
    "oracle_relevance_accuracy",
    "random_relevance_accuracy",
    "no_relevance_accuracy",
    "decoy_only_accuracy",
    "m23_router_decoy_lift_vs_scale",
    "m23_router_worst_surface_lift_vs_scale",
    "m23_oracle_relevance_lift",
    "m23_trace_punish_trace_exact_lift_vs_scale",
    "m23_trace_punish_decoy_delta_vs_scale",
    "m23_trace_punish_strict_delta_vs_scale",
    "predicted_trace_accuracy",
    "oracle_trace_accuracy",
    "random_trace_accuracy",
    "shuffled_trace_accuracy",
    "zero_trace_accuracy",
    "prompt_only_accuracy",
    "advisor_vs_prompt_delta",
    "m24_strict_delta_vs_prompt_only",
    "predicted_vs_random_delta",
    "predicted_vs_shuffled_delta",
    "oracle_trained_oracle_trace_accuracy",
    "oracle_trained_predicted_trace_accuracy",
    "oracle_trained_random_trace_accuracy",
    "oracle_trained_trace_delta",
    "predicted_trace_gap_to_oracle_upper_bound",
    "cross_advisor_oracle_gap",
    "substrate_claim_score",
    "m24_promotion_gate_pass_rate",
    "m24_promotion_candidate",
    "generator_parameter_max_delta_after_advisor",
    "generator_parameters_unchanged_after_advisor",
    "substrate_token_count",
    "mean_substrate_token_count",
    "avg_substrate_token_count",
    "substrate_tokens",
    "mean_substrate_tokens",
    "avg_substrate_tokens",
    "reference_token_count",
    "mean_reference_token_count",
    "reference_tokens",
    "mean_reference_tokens",
    "baseline_token_count",
    "mean_baseline_token_count",
    "baseline_tokens",
    "mean_baseline_tokens",
    "compression_ratio",
    "mean_compression_ratio",
    "packed_symbol_to_prompt_ratio",
    "prompt_to_packed_symbol_ratio",
    "packed_to_prompt_ratio",
    "prompt_to_packed_ratio",
    "packed_symbol_compression_ratio",
    "mdl_weight",
    "token_reduction_ratio",
    "mean_token_reduction_ratio",
    "token_ratio_vs_m23",
    "compression_lift_vs_m23",
    "compression_adjusted_strict_accuracy",
    "strict_accuracy_per_substrate_token",
    "m24_gate_packed_trace_shorter_than_prompt",
    "m24_gate_trace_beats_random",
    "m24_gate_trace_beats_zero",
    "m24_gate_trace_beats_shuffled",
    "m24_gate_trace_matches_oracle_upper_bound",
    "m24_gate_trace_beats_prompt_only",
    "m24_gate_nonzero_exact_trace_reconstruction",
    "m24_gate_token_reduction_positive",
    "m24_2_hard_bottleneck_strict_accuracy",
    "m24_2_hard_bottleneck_trace_exact_accuracy",
    "m24_2_hard_bottleneck_token_count",
    "m24_2_hard_bottleneck_compression_ratio",
    "m24_2_hard_bottleneck_accuracy_per_token",
    "m24_2_hard_bottleneck_delta_vs_m24_1",
    "m24_2_hard_bottleneck_delta_vs_prompt_only",
    "m24_2_hard_bottleneck_symbol_error_rate",
    "m24_2_hard_bottleneck_score",
    "m24_2_promotion_gate_pass_rate",
    "m24_2_promotion_candidate",
    "m24_2_gate_hard_bottleneck_configured",
    "m24_2_gate_strict_accuracy_retained",
    "m24_2_gate_trace_beats_shuffled_strong",
    "m24_2_gate_trace_beats_random_strong",
    "m24_2_gate_trace_exact_floor",
    "m24_2_gate_symbol_budget_respected",
    "m24_2_gate_hard_trace_beats_random",
    "m24_2_gate_hard_trace_beats_prompt_only",
    "m24_2_gate_token_reduction_positive",
    "predicted_stream_accuracy",
    "oracle_stream_accuracy",
    "shuffled_stream_accuracy",
    "random_stream_accuracy",
    "zero_stream_accuracy",
    "m25_strict_delta_vs_prompt_only",
    "oracle_stream_delta",
    "stream_advisor_delta",
    "loose_stream_exact_accuracy",
    "stream_type_accuracy",
    "stream_value_accuracy",
    "stream_aux_accuracy",
    "loose_symbol_to_prompt_ratio",
    "prompt_to_loose_symbol_ratio",
    "loose_symbol_budget",
    "accuracy_per_loose_symbol",
    "accuracy_per_prompt_token",
    "m25_promotion_gate_pass_rate",
    "m25_promotion_candidate",
    "m25_gate_strict_accuracy_retained",
    "m25_gate_stream_beats_shuffled",
    "m25_gate_stream_beats_random",
    "m25_gate_token_reduction_positive",
    "m25_gate_nonzero_stream_reconstruction",
    "m25_gate_symbolic_trace_only",
    "phrase_accuracy",
    "phrase_exact_accuracy",
    "semantic_coverage_strict_accuracy",
    "semantic_coverage_worst_surface_accuracy",
    "semantic_coverage_judri_causal_delta",
    "semantic_coverage_training_exposure_rate",
    "semantic_coverage_surface_count",
    "semantic_isolation_cell_count",
    "semantic_coverage_lexical_shift_effect_strict_accuracy_delta",
    "semantic_coverage_role_binding_effect_strict_accuracy_delta",
    "semantic_coverage_combined_effect_strict_accuracy_delta",
    "semantic_coverage_fraction_effect_strict_accuracy_delta",
    "semantic_coverage_role_curriculum_effect_strict_accuracy_delta",
    "semantic_coverage_role_swap_effect_strict_accuracy_delta",
    "semantic_coverage_role_curriculum_fraction_effect_strict_accuracy_delta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one canonical whole-grid ablation report.")
    parser.add_argument("--history-manifest", type=Path, default=None)
    parser.add_argument("--program-spine-manifest", type=Path, default=None)
    parser.add_argument("--legacy-grid-manifest", type=Path, default=None)
    parser.add_argument("--m-bridge-manifest", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--doc-output", type=Path, default=DEFAULT_DOC_OUTPUT)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--refresh-legacy-grid", action="store_true")
    parser.add_argument("--legacy-grid-run-id", type=str, default="")
    parser.add_argument("--legacy-grid-execute", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = _validated_output_root(args.output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("whole_ablation_grid_%Y%m%d_%H%M%S")
    output_dir = output_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    history_manifest = args.history_manifest or _latest_named_manifest(DEFAULT_HISTORY_ROOT, "ablation_history_manifest.json")
    program_spine_manifest = args.program_spine_manifest or _latest_named_manifest(
        DEFAULT_PROGRAM_SPINE_ROOT, "ablation_program_spine_manifest.json"
    )
    legacy_grid_manifest = args.legacy_grid_manifest or _latest_named_manifest(
        DEFAULT_LEGACY_GRID_ROOT, "legacy_ablation_grid_manifest.json"
    )
    if args.refresh_legacy_grid:
        legacy_grid_manifest = _refresh_legacy_grid(args, legacy_grid_manifest)
    m_bridge_manifest = args.m_bridge_manifest or _latest_named_manifest(
        DEFAULT_M_BRIDGE_ROOT, "m_bridge_ablation_suite_manifest.json"
    )

    history = _read_json_required(history_manifest)
    spine = _read_json_required(program_spine_manifest)
    legacy_grid = _read_json_optional(legacy_grid_manifest)
    m_bridge = _read_json_optional(m_bridge_manifest)

    stages_by_key = {str(stage.get("stage_key", "")): stage for stage in spine.get("stages", []) if isinstance(stage, dict)}
    rows: list[dict[str, Any]] = []
    for stage_key in STAGE_ORDER:
        stage = stages_by_key.get(stage_key)
        if stage is None:
            continue
        if stage_key in {"A-G", "H", "H5", "J", "L", "J/L Hypercube", "Phase Eval"}:
            row = _legacy_row(stage, legacy_grid)
        elif stage_key == "M3":
            row = _m3_row(stage, m_bridge)
        elif stage_key == "M11":
            row = _m11_row(stage, m_bridge)
        elif stage_key in {"M14", "M18", "M19", "M20", "M21", "M22", "M23", "M24", "M25"}:
            row = _special_stage_row(stage)
        elif stage_key == "Control Plane":
            row = _control_plane_row(stage, history_manifest, program_spine_manifest, legacy_grid_manifest, m_bridge_manifest)
        else:
            row = _generic_row(stage)
        rows.append(row)

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M.whole_grid", "scripts/control_plane/run_whole_ablation_grid.py"),
        "run_id": run_id,
        "source_manifests": {
            "history_manifest": _repo_relative(history_manifest),
            "program_spine_manifest": _repo_relative(program_spine_manifest),
            "legacy_grid_manifest": _repo_relative(legacy_grid_manifest) if legacy_grid_manifest else None,
            "m_bridge_manifest": _repo_relative(m_bridge_manifest) if m_bridge_manifest else None,
        },
        "legacy_grid_status": _legacy_grid_status(legacy_grid),
        "coverage_summary": _coverage_summary(rows),
        "stage_rows": rows,
        "notes": [
            "Fresh reruns and artifact-backed anchors are intentionally separated.",
            "Legacy H and H5 remain represented through the recovered shared H/H5/J surface.",
            "This control-plane report is the whole grid view, not a claim that every historical family was retrained in this pass.",
        ],
    }

    manifest_path = output_dir / "whole_ablation_grid_manifest.json"
    summary_path = output_dir / "whole_ablation_grid_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary = _render_markdown(manifest)
    summary_path.write_text(summary, encoding="utf-8")
    args.doc_output.parent.mkdir(parents=True, exist_ok=True)
    args.doc_output.write_text(summary, encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {args.doc_output}")


def _refresh_legacy_grid(args: argparse.Namespace, current_manifest: Path | None) -> Path:
    run_id = args.legacy_grid_run_id.strip()
    if not run_id and current_manifest is not None:
        run_id = current_manifest.parent.name
    if not run_id:
        raise ValueError("Need --legacy-grid-run-id or an existing legacy manifest to refresh the legacy grid.")
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "legacy" / "run_legacy_ablation_grid.py"),
        "--run-id",
        run_id,
        "--execute" if args.legacy_grid_execute else "--aggregate-only",
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return DEFAULT_LEGACY_GRID_ROOT / run_id / "legacy_ablation_grid_manifest.json"


def _validated_output_root(path: Path) -> Path:
    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        relative = str(candidate).replace("\\", "/")
    validated = assert_output_path_allowed("M", relative)
    return REPO_ROOT / validated


def _legacy_row(stage: dict[str, Any], legacy_grid: dict[str, Any] | None) -> dict[str, Any]:
    lane_map = {
        "A-G": "a_to_g",
        "H": "hj",
        "H5": "hj",
        "J": "hj",
        "L": "l6",
        "Phase Eval": "phase5_objective",
    }
    stage_key = str(stage.get("stage_key", ""))
    lane_name = lane_map.get(stage_key)
    lane = _legacy_lane(legacy_grid, lane_name) if lane_name else None
    metrics: dict[str, float] = {}
    notes: list[str] = []
    supplemental: list[str] = []
    surface_kind = "history_only"
    anchor_path = None

    if lane is not None:
        surface_kind = "fresh_legacy_lane"
        anchor_path = _string_or_none(lane.get("artifact_path"))
        if isinstance(lane.get("metrics_digest"), dict):
            metrics.update({str(k): _float_or_zero(v) for k, v in lane["metrics_digest"].items()})
        notes.append(f"fresh legacy surface status: {lane.get('status', 'unknown')}")
        if stage_key in {"H", "H5", "J"}:
            notes.append("represented inside the recovered shared H/H5/J runnable lane")
        if stage_key == "Phase Eval":
            for extra_lane_name in ("phase5_train", "english_cot_duel"):
                extra_lane = _legacy_lane(legacy_grid, extra_lane_name)
                if extra_lane is not None and extra_lane.get("artifact_path"):
                    supplemental.append(str(extra_lane["artifact_path"]))
            metrics.update(_phase_eval_metrics(legacy_grid))
    elif stage_key == "J/L Hypercube":
        notes.append("historical orchestration layer linking late J/L aggregation into the M families")

    return _row(stage, surface_kind, anchor_path, supplemental, _limit_metrics(metrics), notes)


def _m3_row(stage: dict[str, Any], m_bridge: dict[str, Any] | None) -> dict[str, Any]:
    metrics: dict[str, float] = {}
    notes: list[str] = []
    anchor_path = _repo_relative(_latest_named_manifest(DEFAULT_M_BRIDGE_ROOT, "m_bridge_ablation_suite_manifest.json"))
    if isinstance(m_bridge, dict):
        tracks = m_bridge.get("m3_tracks", [])
        if isinstance(tracks, list):
            metrics["bridge_track_count"] = float(len(tracks))
            harmful = 0
            for track in tracks:
                if isinstance(track, dict) and any("harm" in str(line).lower() for line in track.get("diagnosis", [])):
                    harmful += 1
            metrics["harmful_track_count"] = float(harmful)
        for line in m_bridge.get("diagnosis", [])[:4]:
            notes.append(str(line))
    return _row(stage, "artifact_anchor", anchor_path, [], _limit_metrics(metrics), notes)


def _m11_row(stage: dict[str, Any], m_bridge: dict[str, Any] | None) -> dict[str, Any]:
    m11 = (m_bridge or {}).get("m11_track", {}) if isinstance(m_bridge, dict) else {}
    metrics = {
        "headline_accuracy": _float_or_zero(m11.get("headline_accuracy")),
        "headline_macro_f1": _float_or_zero(m11.get("headline_macro_f1")),
        "bridge_audit_accuracy": _float_or_zero(m11.get("bridge_audit_accuracy")),
        "floor_lock_accuracy": _float_or_zero(m11.get("floor_lock_accuracy")),
        "publication_mean_acc": _float_or_zero(m11.get("publication_mean_acc")),
    }
    notes = [str(line) for line in m11.get("diagnosis", [])[:4]] if isinstance(m11, dict) else []
    anchor_path = _string_or_none(m11.get("manifest_path"))
    supplemental = [
        path
        for path in [
            _string_or_none(m11.get("bridge_audit_path")),
            _string_or_none(m11.get("floor_lock_path")),
            _string_or_none(m11.get("publication_metrics_path")),
        ]
        if path
    ]
    return _row(stage, "artifact_anchor" if anchor_path else "runnable_no_anchor", anchor_path, supplemental, _limit_metrics(metrics), notes)


def _special_stage_row(stage: dict[str, Any]) -> dict[str, Any]:
    stage_key = str(stage.get("stage_key"))
    anchor = _explicit_stage_anchor(stage_key) or _resolve_generic_anchor(stage)
    payload = _read_json_optional(anchor) if anchor else None
    metrics = _special_stage_metrics(stage_key, payload)
    stage_for_row = _stage_with_direct_contract(stage, payload) if stage_key in {"M19", "M20", "M21", "M22", "M23", "M24", "M25"} else stage
    supplemental: list[str] = []
    notes: list[str] = []
    if stage_key == "M14":
        notes.append("anchored M14 artifact surface is still sparse and should not be over-read")
    if stage_key == "M18":
        notes.append("M18 anchor includes English, Chinese, typed, and hybrid comparisons")
    if stage_key == "M19":
        notes.append("M19 anchor captures runway mainline and isolation-era artifacts")
        dynamic_anchor = _latest_named_manifest(DEFAULT_M19_4_ROOT, "m19_4_mainline_report.json")
        if dynamic_anchor and dynamic_anchor.exists():
            supplemental.append(_repo_relative(dynamic_anchor) or "")
        dynamic_benchmark = _latest_named_manifest(DEFAULT_M19_4_BENCH_ROOT, "m19_4_benchmark_report.json")
        if dynamic_benchmark and dynamic_benchmark.exists():
            supplemental.append(_repo_relative(dynamic_benchmark) or "")
        if DEFAULT_M19_ISOLATION_REPORT.exists():
            supplemental.append(_repo_relative(DEFAULT_M19_ISOLATION_REPORT) or "")
        if DEFAULT_M19_ZH_SUMMARY.exists():
            supplemental.append(_repo_relative(DEFAULT_M19_ZH_SUMMARY) or "")
    if stage_key == "M20":
        notes.append("M20 anchor captures dictionary-first substrate metrics; it is not a downstream English bridge claim")
        lock_anchor = _latest_named_manifest(DEFAULT_M20_LOCK_ROOT, "m20_lock_suite_report.json")
        if lock_anchor and lock_anchor.exists():
            supplemental.append(_repo_relative(lock_anchor) or "")
        induction_anchor = _latest_named_manifest(DEFAULT_M20_INDUCTION_ROOT, "m20_predicate_induction_report.json")
        if induction_anchor and induction_anchor.exists():
            supplemental.append(_repo_relative(induction_anchor) or "")
    if stage_key == "M21":
        notes.append("M21 anchor captures dynamic bridi Q-former metrics; promotion still requires actual bridge causality, not synthetic trace accuracy alone")
        synthetic_anchor = _latest_named_manifest(DEFAULT_M21_SYNTHETIC_ROOT, "m21_synthetic_assay_report.json")
        if synthetic_anchor and synthetic_anchor.exists():
            supplemental.append(_repo_relative(synthetic_anchor) or "")
        actual_anchor = _latest_named_manifest(DEFAULT_M21_ACTUAL_ROOT, "m21_actual_bridge_report.json")
        if actual_anchor and actual_anchor.exists():
            supplemental.append(_repo_relative(actual_anchor) or "")
        lock_anchor = _latest_named_manifest(DEFAULT_M21_LOCK_ROOT, "m21_lock_suite_report.json")
        if lock_anchor and lock_anchor.exists():
            supplemental.append(_repo_relative(lock_anchor) or "")
    if stage_key == "M22":
        notes.append("M22 is a semantic coverage generalization gate over M21 controls; failed promotion remains visible as evidence, not success")
    if stage_key == "M23":
        notes.append("M23 is the causal relevance-router fork over the M21/M22 bridi substrate; promotion depends on decoy OOD lift, not clean accuracy alone")
    if stage_key == "M24":
        notes.append(
            "M24 is the M24.1 matched trace corruption and compression-pressure fork; "
            "strict accuracy is canonical, phrase accuracy is diagnostic only, and promotion requires m24_promotion_candidate=1.0"
        )
    if stage_key == "M25":
        notes.append(
            "M25 is the emergent loose bridi grammar-action stream fork over M24.2; "
            "strict accuracy is canonical, phrase accuracy is diagnostic only, and promotion requires m25_promotion_candidate=1.0"
        )
    return _row(
        stage_for_row,
        "artifact_anchor" if anchor else "runnable_no_anchor",
        _repo_relative(anchor) if anchor else None,
        [path for path in supplemental if path],
        _limit_metrics(metrics, limit=18 if stage_key in {"M21", "M22", "M23", "M24", "M25"} else 8),
        notes,
    )


def _control_plane_row(
    stage: dict[str, Any],
    history_manifest: Path,
    program_spine_manifest: Path,
    legacy_grid_manifest: Path | None,
    m_bridge_manifest: Path | None,
) -> dict[str, Any]:
    supplemental = [
        _repo_relative(program_spine_manifest),
        _repo_relative(legacy_grid_manifest) if legacy_grid_manifest else None,
        _repo_relative(m_bridge_manifest) if m_bridge_manifest else None,
    ]
    return _row(
        stage,
        "control_plane_manifest",
        _repo_relative(history_manifest),
        [path for path in supplemental if path],
        {},
        ["canonical history, spine, legacy grid, and bridge suite are linked from this row"],
    )


def _stage_with_direct_contract(stage: dict[str, Any], payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return stage
    contract = payload.get("comparison_contract")
    if not isinstance(contract, dict):
        return stage

    updated = dict(stage)
    required = contract.get("required_test_contract_ids")
    if isinstance(required, list):
        updated["required_test_contracts"] = [str(item) for item in required]
    targets = contract.get("comparison_targets")
    if isinstance(targets, list):
        updated["comparison_targets"] = [target for target in targets if isinstance(target, dict)]
    historical = contract.get("historical_comparison_families")
    if isinstance(historical, list):
        updated["historical_comparison_families"] = [str(item) for item in historical]
    return updated


def _generic_row(stage: dict[str, Any]) -> dict[str, Any]:
    anchor = _resolve_generic_anchor(stage)
    payload = _read_json_optional(anchor) if anchor else None
    metrics = _generic_metrics(payload) if isinstance(payload, dict) else {}
    if anchor is None:
        surface_kind = "runnable_no_anchor" if int(stage.get("runnable_count") or 0) > 0 else "history_only"
        notes = ["no canonical anchor report auto-resolved for this stage"]
    else:
        surface_kind = "artifact_anchor"
        notes = []
    return _row(stage, surface_kind, _repo_relative(anchor) if anchor else None, [], _limit_metrics(metrics), notes)


def _resolve_generic_anchor(stage: dict[str, Any]) -> Path | None:
    candidates: list[Path] = []
    for raw in stage.get("artifact_roots", []):
        path = _repo_path(raw)
        if path.is_file() and path.suffix.lower() == ".json":
            candidates.append(path)
        elif path.is_dir():
            for pattern in PREFERRED_REPORT_GLOBS:
                candidates.extend(match for match in path.rglob(pattern) if match.is_file())
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def _generic_metrics(payload: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if isinstance(payload.get("metrics"), dict):
        metrics.update(_pick_metrics(payload["metrics"]))
    metrics.update(_pick_metrics(payload))
    if not metrics and isinstance(payload.get("summary"), dict):
        metrics.update(_pick_metrics(payload["summary"]))
    if isinstance(payload.get("aggregate_metrics"), dict):
        metrics.update(_pick_metrics(payload["aggregate_metrics"]))
    if not metrics and isinstance(payload.get("cells"), dict):
        best_id, best_cell = _best_cell(payload["cells"])
        if best_id and isinstance(best_cell, dict):
            metrics["best_cell_accuracy"] = _float_or_zero(_best_cell_metric(best_cell))
    if not metrics:
        metrics.update(_pick_metrics(_numeric_leaves(payload)))
    return metrics


def _explicit_stage_anchor(stage_key: str) -> Path | None:
    if stage_key == "M14" and DEFAULT_M14_REPORT.exists():
        return DEFAULT_M14_REPORT
    if stage_key == "M18" and DEFAULT_M18_REPORT.exists():
        return DEFAULT_M18_REPORT
    if stage_key == "M19":
        direct_anchor = _latest_direct_unified_eval_anchor("M19")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        dynamic_anchor = _latest_named_manifest(DEFAULT_M19_4_ROOT, "m19_4_mainline_report.json")
        if dynamic_anchor and dynamic_anchor.exists():
            return dynamic_anchor
        dynamic_benchmark = _latest_named_manifest(DEFAULT_M19_4_BENCH_ROOT, "m19_4_benchmark_report.json")
        if dynamic_benchmark and dynamic_benchmark.exists():
            return dynamic_benchmark
    if stage_key == "M19" and DEFAULT_M19_MAINLINE_REPORT.exists():
        return DEFAULT_M19_MAINLINE_REPORT
    if stage_key == "M20":
        direct_anchor = _latest_direct_unified_eval_anchor("M20")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        suite_anchor = _latest_named_manifest(DEFAULT_M20_SUITE_ROOT, "m20_dictionary_first_suite_report.json")
        if suite_anchor and suite_anchor.exists():
            return suite_anchor
    if stage_key == "M21":
        direct_anchor = _latest_direct_unified_eval_anchor("M21")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        suite_anchor = _latest_named_manifest(DEFAULT_M21_SUITE_ROOT, "m21_dynamic_bridi_suite_report.json")
        if suite_anchor and suite_anchor.exists():
            return suite_anchor
    if stage_key == "M22":
        direct_anchor = _latest_direct_unified_eval_anchor("M22")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        generalization_anchor = _latest_named_manifest(DEFAULT_M22_GENERALIZATION_ROOT, "m22_semantic_generalization_report.json")
        if generalization_anchor and generalization_anchor.exists():
            return generalization_anchor
    if stage_key == "M23":
        direct_anchor = _latest_direct_unified_eval_anchor("M23")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        relevance_anchor = _latest_named_manifest(DEFAULT_M23_RELEVANCE_ROOT, "m23_relevance_suite_report.json")
        if relevance_anchor and relevance_anchor.exists():
            return relevance_anchor
    if stage_key == "M24":
        direct_anchor = _latest_direct_unified_eval_anchor("M24")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        compression_anchor = _latest_named_manifest(DEFAULT_M24_COMPRESSION_ROOT, "m24_substrate_compression_report.json")
        if compression_anchor and compression_anchor.exists():
            return compression_anchor
    if stage_key == "M25":
        direct_anchor = _latest_direct_unified_eval_anchor("M25")
        if direct_anchor and direct_anchor.exists():
            return direct_anchor
        emergent_anchor = _latest_named_manifest(DEFAULT_M25_EMERGENT_ROOT, "m25_emergent_bridi_report.json")
        if emergent_anchor and emergent_anchor.exists():
            return emergent_anchor
    return None


def _special_stage_metrics(stage_key: str, payload: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}
    if stage_key == "M14":
        cells = payload.get("cells", {})
        if isinstance(cells, dict):
            best_id, best_cell = _best_cell(cells)
            return {
                "cell_count": float(len(cells)),
                "best_cell_accuracy": _float_or_zero(best_cell.get("accuracy")) if isinstance(best_cell, dict) else 0.0,
                "all_cells_zero": 1.0
                if all(isinstance(cell, dict) and _float_or_zero(cell.get("accuracy")) == 0.0 for cell in cells.values())
                else 0.0,
            }
        return {}
    if stage_key == "M18":
        top = payload.get("metrics", {})
        if isinstance(top, dict):
            return {
                "sapir_english_accuracy": _float_or_zero(top.get("sapir_english_accuracy")),
                "sapir_chinese_accuracy": _float_or_zero(top.get("sapir_chinese_accuracy")),
                "harmonized_en_concise_accuracy": _float_or_zero(top.get("harmonized_en_concise_accuracy")),
                "harmonized_l_typed_accuracy": _float_or_zero(top.get("harmonized_l_typed_accuracy")),
                "hybrid_en_cot_accuracy": _float_or_zero(top.get("hybrid_en_cot_accuracy")),
            }
        return {}
    if stage_key == "M19":
        metrics = {}
        top = payload.get("metrics", {})
        headline = payload.get("headline_metrics", {})
        if isinstance(headline, dict):
            _merge_existing_metrics(
                metrics,
                headline,
                {
                    "mainline_overall_accuracy": "overall_accuracy",
                    "mainline_avg_tokens": "avg_tokens",
                    "mainline_lift_vs_en_cot": "lift_vs_en_cot",
                    "mainline_lift_vs_random": "lift_vs_random",
                    "mainline_audit_qformer_accuracy": "audit_qformer_accuracy",
                    "purged_accuracy": "purged_accuracy",
                    "masked_accuracy": "masked_accuracy",
                    "replication_mean_accuracy": ("mean_accuracy", "replication_mean_accuracy"),
                    "replication_std_accuracy": ("std_accuracy", "replication_std_accuracy"),
                    "kill_entity_accuracy": ("entity_accuracy", "kill_entity_accuracy"),
                    "kill_entity_renamed_accuracy": ("entity_renamed_accuracy", "kill_entity_renamed_accuracy"),
                    "kill_format_accuracy": ("format_accuracy", "kill_format_accuracy"),
                    "kill_numeric_accuracy": ("numeric_accuracy", "kill_numeric_accuracy"),
                    "typed_family_accuracy": "typed_family_accuracy",
                    "masked_pointer_zero_rate": "masked_pointer_zero_rate",
                    "family_slot_entropy": "family_slot_entropy",
                    "symbolic_trace_alignment": "symbolic_trace_alignment",
                    "predicate_pointer_radial_gap": "predicate_pointer_radial_gap",
                    "family_radius_violation_rate": "family_radius_violation_rate",
                    "hyperbolic_geodesic_margin": "hyperbolic_geodesic_margin",
                    "hyperbolic_projection_clip_rate": "hyperbolic_projection_clip_rate",
                },
            )
        if isinstance(top, dict):
            _merge_existing_metrics(
                metrics,
                top,
                {
                    "mainline_overall_accuracy": "overall_accuracy",
                    "mainline_avg_tokens": "avg_tokens",
                    "mainline_lift_vs_en_cot": "lift_vs_en_cot",
                    "mainline_lift_vs_random": "lift_vs_random",
                    "mainline_audit_qformer_accuracy": "audit_qformer_accuracy",
                    "mainline_premature_stop_rate": "premature_stop_rate",
                    "mainline_max_cap_hit_rate": "max_cap_hit_rate",
                    "mainline_caa_entanglement": "caa_manifold_entanglement_score",
                    "typed_family_accuracy": "typed_family_accuracy",
                    "masked_pointer_zero_rate": "masked_pointer_zero_rate",
                    "family_slot_entropy": "family_slot_entropy",
                    "symbolic_trace_alignment": "symbolic_trace_alignment",
                    "predicate_pointer_radial_gap": "predicate_pointer_radial_gap",
                    "family_radius_violation_rate": "family_radius_violation_rate",
                    "hyperbolic_geodesic_margin": "hyperbolic_geodesic_margin",
                    "hyperbolic_projection_clip_rate": "hyperbolic_projection_clip_rate",
                },
            )
        isolation = _read_json_optional(DEFAULT_M19_ISOLATION_REPORT)
        if isinstance(isolation, dict) and isinstance(isolation.get("cells"), dict):
            best_id, best_cell = _best_cell(isolation["cells"])
            metrics["best_isolation_accuracy"] = _best_cell_metric(best_cell) if isinstance(best_cell, dict) else 0.0
            if isinstance(best_cell, dict) and isinstance(best_cell.get("metrics"), dict):
                metrics["best_isolation_avg_tokens"] = _float_or_zero(best_cell["metrics"].get("avg_tokens"))
        zh = _read_json_optional(DEFAULT_M19_ZH_SUMMARY)
        if isinstance(zh, dict) and isinstance(zh.get("mainline"), dict):
            metrics["zh_cot_accuracy"] = _float_or_zero(zh["mainline"].get("zh_cot_accuracy"))
            metrics["lift_vs_zh"] = _float_or_zero(zh["mainline"].get("lift_vs_zh_cot"))
        replication = _latest_named_manifest(DEFAULT_M19_REPLICATION_ROOT, "m19_replication_report.json")
        replication_payload = _read_json_optional(replication)
        if isinstance(replication_payload, dict) and isinstance(replication_payload.get("metrics"), dict):
            metrics["replication_mean_accuracy"] = _float_or_zero(replication_payload["metrics"].get("mean_accuracy"))
            metrics["replication_std_accuracy"] = _float_or_zero(replication_payload["metrics"].get("std_accuracy"))
        kill_report = _latest_named_manifest(DEFAULT_M19_KILL_ROOT, "m19_kill_test_report.json")
        kill_payload = _read_json_optional(kill_report)
        if isinstance(kill_payload, dict) and isinstance(kill_payload.get("metrics"), dict):
            metrics["kill_entity_accuracy"] = _float_or_zero(kill_payload["metrics"].get("entity_accuracy"))
            metrics["kill_format_accuracy"] = _float_or_zero(kill_payload["metrics"].get("format_accuracy"))
        return metrics
    if stage_key == "M20":
        metrics: dict[str, float] = {}
        headline = payload.get("headline_metrics", {})
        aggregate = payload.get("aggregate_metrics", {})
        top = payload.get("metrics", {})
        for source in (headline, aggregate, top, payload):
            if not isinstance(source, dict):
                continue
            _merge_existing_metrics(
                metrics,
                source,
                {
                    "strict_accuracy": ("strict_accuracy", "mean_strict_accuracy", "synthetic_world_accuracy"),
                    "synthetic_world_accuracy": ("synthetic_world_accuracy", "mean_synthetic_world_accuracy"),
                    "dictionary_coverage": ("dictionary_coverage", "mean_dictionary_coverage"),
                    "factorized_exact_accuracy": ("factorized_exact_accuracy", "mean_factorized_exact_accuracy"),
                    "brivi_gate_accuracy": ("brivi_gate_accuracy", "mean_brivi_gate_accuracy"),
                    "predicate_identity_stability": (
                        "predicate_identity_stability",
                        "mean_predicate_identity_stability",
                    ),
                    "lock_pass_rate": ("lock_pass_rate", "mean_lock_pass_rate"),
                    "masked_accuracy": ("masked_accuracy", "mean_masked_accuracy"),
                    "entity_leakage_proxy": ("entity_leakage_proxy", "mean_entity_leakage_proxy"),
                    "active_code_fraction": ("active_code_fraction", "mean_active_code_fraction"),
                    "hard_code_utilization_count": (
                        "hard_code_utilization_count",
                        "mean_hard_code_utilization_count",
                    ),
                    "soft_hard_dictionary_agreement": (
                        "soft_hard_dictionary_agreement",
                        "mean_soft_hard_dictionary_agreement",
                    ),
                    "stable_seed_rate": "stable_seed_rate",
                },
            )
        if isinstance(payload.get("cells"), dict):
            best_id, best_cell = _best_cell(payload["cells"])
            if best_id and isinstance(best_cell, dict):
                metrics["best_cell_accuracy"] = _best_cell_metric(best_cell)
        return metrics
    if stage_key == "M21":
        metrics: dict[str, float] = {}
        headline = payload.get("headline_metrics", {})
        aggregate = payload.get("aggregate_metrics", {})
        top = payload.get("metrics", {})
        for source in (headline, aggregate, top, payload):
            if not isinstance(source, dict):
                continue
            _merge_existing_metrics(
                metrics,
                source,
                {
                    "strict_accuracy": ("strict_accuracy", "mean_strict_accuracy"),
                    "bridi_trace_exact_accuracy": ("bridi_trace_exact_accuracy", "mean_bridi_trace_exact_accuracy"),
                    "gismu_accuracy": ("gismu_accuracy", "mean_gismu_accuracy"),
                    "cmavo_accuracy": ("cmavo_accuracy", "mean_cmavo_accuracy"),
                    "judri_binding_accuracy": ("judri_binding_accuracy", "mean_judri_binding_accuracy"),
                    "frame_count_mae": ("frame_count_mae", "mean_frame_count_mae"),
                    "lock_pass_rate": ("lock_pass_rate", "mean_lock_pass_rate"),
                    "actual_bridge_transfer_score": "actual_bridge_transfer_score",
                    "full_accuracy": ("full_accuracy", "mean_full_accuracy"),
                    "no_cmavo_accuracy": ("no_cmavo_accuracy", "mean_no_cmavo_accuracy"),
                    "no_judri_accuracy": ("no_judri_accuracy", "mean_no_judri_accuracy"),
                    "gismu_only_accuracy": ("gismu_only_accuracy", "mean_gismu_only_accuracy"),
                    "random_trace_accuracy": "random_trace_accuracy",
                    "scratchpad_only_accuracy": "scratchpad_only_accuracy",
                    "frame_drop_delta": ("frame_drop_delta", "mean_frame_drop_delta"),
                    "cmavo_causal_delta": ("cmavo_causal_delta", "mean_cmavo_causal_delta"),
                    "judri_causal_delta": ("judri_causal_delta", "mean_judri_causal_delta"),
                    "mean_active_frames": "mean_active_frames",
                    "active_code_fraction_reachable": ("active_code_fraction_reachable", "mean_active_code_fraction_reachable"),
                    "avg_tokens": "avg_tokens",
                    "accuracy_per_token": "accuracy_per_token",
                    "trace_tokens": "trace_tokens",
                    "accuracy_per_trace_token": "accuracy_per_trace_token",
                    "semantic_coverage_strict_accuracy": "semantic_coverage_strict_accuracy",
                    "semantic_coverage_worst_surface_accuracy": "semantic_coverage_worst_surface_accuracy",
                    "semantic_coverage_judri_causal_delta": "semantic_coverage_judri_causal_delta",
                    "semantic_coverage_training_exposure_rate": "semantic_coverage_training_exposure_rate",
                    "semantic_coverage_surface_count": "semantic_coverage_surface_count",
                    "semantic_isolation_cell_count": "semantic_isolation_cell_count",
                    "semantic_coverage_lexical_shift_effect_strict_accuracy_delta": "semantic_coverage_lexical_shift_effect_strict_accuracy_delta",
                    "semantic_coverage_role_binding_effect_strict_accuracy_delta": "semantic_coverage_role_binding_effect_strict_accuracy_delta",
                    "semantic_coverage_combined_effect_strict_accuracy_delta": "semantic_coverage_combined_effect_strict_accuracy_delta",
                    "semantic_coverage_fraction_effect_strict_accuracy_delta": "semantic_coverage_fraction_effect_strict_accuracy_delta",
                    "semantic_coverage_role_curriculum_effect_strict_accuracy_delta": "semantic_coverage_role_curriculum_effect_strict_accuracy_delta",
                    "semantic_coverage_role_swap_effect_strict_accuracy_delta": "semantic_coverage_role_swap_effect_strict_accuracy_delta",
                    "semantic_coverage_role_curriculum_fraction_effect_strict_accuracy_delta": "semantic_coverage_role_curriculum_fraction_effect_strict_accuracy_delta",
                },
            )
        actual_anchor = _latest_named_manifest(DEFAULT_M21_ACTUAL_ROOT, "m21_actual_bridge_report.json")
        actual_payload = _read_json_optional(actual_anchor)
        if isinstance(actual_payload, dict) and isinstance(actual_payload.get("metrics"), dict):
            _merge_existing_metrics(
                metrics,
                actual_payload["metrics"],
                {
                    "actual_bridge_transfer_score": "actual_bridge_transfer_score",
                    "strict_accuracy": "strict_accuracy",
                    "random_trace_accuracy": "random_trace_accuracy",
                    "scratchpad_only_accuracy": "scratchpad_only_accuracy",
                    "cmavo_causal_delta": "cmavo_causal_delta",
                    "judri_causal_delta": "judri_causal_delta",
                },
            )
        lock_anchor = _latest_named_manifest(DEFAULT_M21_LOCK_ROOT, "m21_lock_suite_report.json")
        lock_payload = _read_json_optional(lock_anchor)
        if isinstance(lock_payload, dict) and isinstance(lock_payload.get("metrics"), dict):
            _merge_existing_metrics(metrics, lock_payload["metrics"], {"lock_pass_rate": "lock_pass_rate", "brivi_lock_violation_rate": "brivi_lock_violation_rate"})
        if isinstance(payload.get("cells"), dict):
            best_id, best_cell = _best_cell(payload["cells"])
            if best_id and isinstance(best_cell, dict):
                metrics["best_cell_accuracy"] = _best_cell_metric(best_cell)
        return metrics
    if stage_key == "M22":
        metrics: dict[str, float] = {}
        headline = payload.get("headline_metrics", {})
        top = payload.get("metrics", {})
        for source in (headline, top, payload):
            if not isinstance(source, dict):
                continue
            _merge_existing_metrics(
                metrics,
                source,
                {
                    "m22_promotion_candidate": "m22_promotion_candidate",
                    "m22_promotion_gate_pass_rate": "m22_promotion_gate_pass_rate",
                    "strict_accuracy": "strict_accuracy",
                    "m22_candidate_cell_count": "m22_candidate_cell_count",
                    "m22_candidate_cells_present": "m22_candidate_cells_present",
                    "m22_semantic_generalization_score": "m22_semantic_generalization_score",
                    "semantic_coverage_strict_accuracy": "semantic_coverage_strict_accuracy",
                    "semantic_coverage_worst_surface_accuracy": "semantic_coverage_worst_surface_accuracy",
                    "semantic_coverage_judri_causal_delta": "semantic_coverage_judri_causal_delta",
                    "m22_semantic_strict_delta_vs_m21_control": "m22_semantic_strict_delta_vs_m21_control",
                    "m22_semantic_worst_delta_vs_m21_control": "m22_semantic_worst_delta_vs_m21_control",
                    "m22_clean_accuracy_drop_vs_m21_control": "m22_clean_accuracy_drop_vs_m21_control",
                    "m22_judri_delta_drop_vs_m21_control": "m22_judri_delta_drop_vs_m21_control",
                },
            )
        return metrics
    if stage_key == "M23":
        metrics: dict[str, float] = {}
        headline = payload.get("headline_metrics", {})
        aggregate = payload.get("aggregate_metrics", {})
        top = payload.get("metrics", {})
        for source in (headline, aggregate, top, payload):
            if not isinstance(source, dict):
                continue
            _merge_existing_metrics(
                metrics,
                source,
                {
                    "strict_accuracy": ("strict_accuracy", "mean_strict_accuracy"),
                    "decoy_relation_ood_accuracy": ("decoy_relation_ood_accuracy", "mean_decoy_relation_ood_accuracy"),
                    "worst_surface_accuracy": ("worst_surface_accuracy", "mean_worst_surface_accuracy"),
                    "bridi_trace_exact_accuracy": ("bridi_trace_exact_accuracy", "mean_bridi_trace_exact_accuracy"),
                    "relevance_top1_accuracy": ("relevance_top1_accuracy", "mean_relevance_top1_accuracy"),
                    "relevance_margin": ("relevance_margin", "mean_relevance_margin"),
                    "loss_trace_exact_surrogate": ("loss_trace_exact_surrogate", "mean_loss_trace_exact_surrogate"),
                    "trace_exact_surrogate_weight": ("trace_exact_surrogate_weight", "mean_trace_exact_surrogate_weight"),
                    "oracle_relevance_accuracy": ("oracle_relevance_accuracy", "mean_oracle_relevance_accuracy"),
                    "random_relevance_accuracy": ("random_relevance_accuracy", "mean_random_relevance_accuracy"),
                    "no_relevance_accuracy": ("no_relevance_accuracy", "mean_no_relevance_accuracy"),
                    "decoy_only_accuracy": ("decoy_only_accuracy", "mean_decoy_only_accuracy"),
                    "m23_router_decoy_lift_vs_scale": "m23_router_decoy_lift_vs_scale",
                    "m23_router_worst_surface_lift_vs_scale": "m23_router_worst_surface_lift_vs_scale",
                    "m23_oracle_relevance_lift": "m23_oracle_relevance_lift",
                    "m23_trace_punish_trace_exact_lift_vs_scale": "m23_trace_punish_trace_exact_lift_vs_scale",
                    "m23_trace_punish_decoy_delta_vs_scale": "m23_trace_punish_decoy_delta_vs_scale",
                    "m23_trace_punish_strict_delta_vs_scale": "m23_trace_punish_strict_delta_vs_scale",
                    "accuracy_per_token": "accuracy_per_token",
                    "accuracy_per_trace_token": "accuracy_per_trace_token",
                },
            )
        if isinstance(payload.get("cells"), dict):
            best_id, best_cell = _best_cell(payload["cells"])
            if best_id and isinstance(best_cell, dict):
                metrics["best_cell_accuracy"] = _best_cell_metric(best_cell)
        return metrics
    if stage_key == "M24":
        metrics: dict[str, float] = {}
        headline = payload.get("headline_metrics", {})
        aggregate = payload.get("aggregate_metrics", {})
        top = payload.get("metrics", {})
        for source in (headline, aggregate, top, payload):
            if not isinstance(source, dict):
                continue
            _merge_existing_metrics(
                metrics,
                source,
                {
                    "strict_accuracy": ("strict_accuracy", "mean_strict_accuracy"),
                    "m24_promotion_candidate": ("m24_promotion_candidate", "mean_m24_promotion_candidate"),
                    "m24_promotion_gate_pass_rate": ("m24_promotion_gate_pass_rate", "mean_m24_promotion_gate_pass_rate"),
                    "predicted_vs_shuffled_delta": (
                        "predicted_vs_shuffled_delta",
                        "mean_predicted_vs_shuffled_delta",
                    ),
                    "shuffled_trace_accuracy": ("shuffled_trace_accuracy", "mean_shuffled_trace_accuracy"),
                    "predicted_vs_random_delta": ("predicted_vs_random_delta", "mean_predicted_vs_random_delta"),
                    "advisor_vs_prompt_delta": ("advisor_vs_prompt_delta", "mean_advisor_vs_prompt_delta"),
                    "packed_symbol_to_prompt_ratio": ("packed_symbol_to_prompt_ratio", "mean_packed_symbol_to_prompt_ratio"),
                    "token_reduction_ratio": ("token_reduction_ratio", "mean_token_reduction_ratio"),
                    "mdl_weight": ("mdl_weight", "mean_mdl_weight"),
                    "m24_gate_packed_trace_shorter_than_prompt": (
                        "m24_gate_packed_trace_shorter_than_prompt",
                        "mean_m24_gate_packed_trace_shorter_than_prompt",
                    ),
                    "m24_gate_trace_beats_shuffled": (
                        "m24_gate_trace_beats_shuffled",
                        "mean_m24_gate_trace_beats_shuffled",
                    ),
                    "m24_gate_trace_beats_random": (
                        "m24_gate_trace_beats_random",
                        "mean_m24_gate_trace_beats_random",
                    ),
                    "m24_gate_trace_beats_zero": (
                        "m24_gate_trace_beats_zero",
                        "mean_m24_gate_trace_beats_zero",
                    ),
                    "m24_gate_trace_matches_oracle_upper_bound": (
                        "m24_gate_trace_matches_oracle_upper_bound",
                        "mean_m24_gate_trace_matches_oracle_upper_bound",
                    ),
                    "m24_gate_trace_beats_prompt_only": (
                        "m24_gate_trace_beats_prompt_only",
                        "mean_m24_gate_trace_beats_prompt_only",
                    ),
                    "m24_gate_nonzero_exact_trace_reconstruction": (
                        "m24_gate_nonzero_exact_trace_reconstruction",
                        "mean_m24_gate_nonzero_exact_trace_reconstruction",
                    ),
                    "m24_gate_token_reduction_positive": (
                        "m24_gate_token_reduction_positive",
                        "mean_m24_gate_token_reduction_positive",
                    ),
                    "m24_2_hard_bottleneck_strict_accuracy": (
                        "m24_2_hard_bottleneck_strict_accuracy",
                        "mean_m24_2_hard_bottleneck_strict_accuracy",
                    ),
                    "m24_2_hard_bottleneck_trace_exact_accuracy": (
                        "m24_2_hard_bottleneck_trace_exact_accuracy",
                        "mean_m24_2_hard_bottleneck_trace_exact_accuracy",
                    ),
                    "m24_2_hard_bottleneck_token_count": (
                        "m24_2_hard_bottleneck_token_count",
                        "mean_m24_2_hard_bottleneck_token_count",
                    ),
                    "m24_2_hard_bottleneck_compression_ratio": (
                        "m24_2_hard_bottleneck_compression_ratio",
                        "mean_m24_2_hard_bottleneck_compression_ratio",
                    ),
                    "m24_2_hard_bottleneck_accuracy_per_token": (
                        "m24_2_hard_bottleneck_accuracy_per_token",
                        "mean_m24_2_hard_bottleneck_accuracy_per_token",
                    ),
                    "m24_2_hard_bottleneck_delta_vs_m24_1": (
                        "m24_2_hard_bottleneck_delta_vs_m24_1",
                        "mean_m24_2_hard_bottleneck_delta_vs_m24_1",
                    ),
                    "m24_2_hard_bottleneck_delta_vs_prompt_only": (
                        "m24_2_hard_bottleneck_delta_vs_prompt_only",
                        "mean_m24_2_hard_bottleneck_delta_vs_prompt_only",
                    ),
                    "m24_2_hard_bottleneck_symbol_error_rate": (
                        "m24_2_hard_bottleneck_symbol_error_rate",
                        "mean_m24_2_hard_bottleneck_symbol_error_rate",
                    ),
                    "m24_2_hard_bottleneck_score": (
                        "m24_2_hard_bottleneck_score",
                        "mean_m24_2_hard_bottleneck_score",
                    ),
                    "m24_2_promotion_gate_pass_rate": (
                        "m24_2_promotion_gate_pass_rate",
                        "mean_m24_2_promotion_gate_pass_rate",
                    ),
                    "m24_2_promotion_candidate": (
                        "m24_2_promotion_candidate",
                        "mean_m24_2_promotion_candidate",
                    ),
                    "m24_2_gate_hard_bottleneck_configured": (
                        "m24_2_gate_hard_bottleneck_configured",
                        "mean_m24_2_gate_hard_bottleneck_configured",
                    ),
                    "m24_2_gate_strict_accuracy_retained": (
                        "m24_2_gate_strict_accuracy_retained",
                        "mean_m24_2_gate_strict_accuracy_retained",
                    ),
                    "m24_2_gate_trace_beats_shuffled_strong": (
                        "m24_2_gate_trace_beats_shuffled_strong",
                        "mean_m24_2_gate_trace_beats_shuffled_strong",
                    ),
                    "m24_2_gate_trace_beats_random_strong": (
                        "m24_2_gate_trace_beats_random_strong",
                        "mean_m24_2_gate_trace_beats_random_strong",
                    ),
                    "m24_2_gate_trace_exact_floor": (
                        "m24_2_gate_trace_exact_floor",
                        "mean_m24_2_gate_trace_exact_floor",
                    ),
                    "m24_2_gate_symbol_budget_respected": (
                        "m24_2_gate_symbol_budget_respected",
                        "mean_m24_2_gate_symbol_budget_respected",
                    ),
                    "m24_2_gate_hard_trace_beats_random": (
                        "m24_2_gate_hard_trace_beats_random",
                        "mean_m24_2_gate_hard_trace_beats_random",
                    ),
                    "m24_2_gate_hard_trace_beats_prompt_only": (
                        "m24_2_gate_hard_trace_beats_prompt_only",
                        "mean_m24_2_gate_hard_trace_beats_prompt_only",
                    ),
                    "m24_2_gate_token_reduction_positive": (
                        "m24_2_gate_token_reduction_positive",
                        "mean_m24_2_gate_token_reduction_positive",
                    ),
                    "generator_parameter_max_delta_after_advisor": (
                        "generator_parameter_max_delta_after_advisor",
                        "mean_generator_parameter_max_delta_after_advisor",
                    ),
                    "generator_parameters_unchanged_after_advisor": (
                        "generator_parameters_unchanged_after_advisor",
                        "mean_generator_parameters_unchanged_after_advisor",
                    ),
                    "overall_phrase_accuracy": ("overall_phrase_accuracy", "mean_overall_phrase_accuracy"),
                    "phrase_accuracy": ("phrase_accuracy", "mean_phrase_accuracy"),
                    "phrase_exact_accuracy": ("phrase_exact_accuracy", "mean_phrase_exact_accuracy"),
                    "judri_binding_accuracy": (
                        "judri_binding_accuracy",
                        "mean_judri_binding_accuracy",
                        "mean_judri_accuracy",
                    ),
                    "substrate_token_count": (
                        "substrate_token_count",
                        "mean_substrate_token_count",
                        "avg_substrate_token_count",
                        "substrate_tokens",
                        "mean_substrate_tokens",
                        "avg_substrate_tokens",
                    ),
                    "reference_token_count": (
                        "reference_token_count",
                        "mean_reference_token_count",
                        "reference_tokens",
                        "mean_reference_tokens",
                        "baseline_token_count",
                        "mean_baseline_token_count",
                        "baseline_tokens",
                        "mean_baseline_tokens",
                    ),
                    "compression_ratio": ("compression_ratio", "mean_compression_ratio"),
                    "prompt_to_packed_symbol_ratio": ("prompt_to_packed_symbol_ratio", "mean_prompt_to_packed_symbol_ratio"),
                    "packed_to_prompt_ratio": ("packed_to_prompt_ratio", "mean_packed_to_prompt_ratio"),
                    "prompt_to_packed_ratio": ("prompt_to_packed_ratio", "mean_prompt_to_packed_ratio"),
                    "packed_symbol_compression_ratio": ("packed_symbol_compression_ratio", "mean_packed_symbol_compression_ratio"),
                    "token_ratio_vs_m23": "token_ratio_vs_m23",
                    "compression_lift_vs_m23": "compression_lift_vs_m23",
                    "avg_tokens": ("avg_tokens", "mean_avg_tokens"),
                    "trace_tokens": ("trace_tokens", "mean_trace_tokens"),
                    "accuracy_per_token": ("accuracy_per_token", "mean_accuracy_per_token"),
                    "accuracy_per_trace_token": ("accuracy_per_trace_token", "mean_accuracy_per_trace_token"),
                    "compression_adjusted_strict_accuracy": "compression_adjusted_strict_accuracy",
                    "strict_accuracy_per_substrate_token": "strict_accuracy_per_substrate_token",
                    "predicted_trace_accuracy": ("predicted_trace_accuracy", "mean_predicted_trace_accuracy"),
                    "oracle_trace_accuracy": ("oracle_trace_accuracy", "mean_oracle_trace_accuracy"),
                    "random_trace_accuracy": ("random_trace_accuracy", "mean_random_trace_accuracy"),
                    "zero_trace_accuracy": ("zero_trace_accuracy", "mean_zero_trace_accuracy"),
                    "prompt_only_accuracy": ("prompt_only_accuracy", "mean_prompt_only_accuracy"),
                    "m24_strict_delta_vs_prompt_only": (
                        "m24_strict_delta_vs_prompt_only",
                        "mean_m24_strict_delta_vs_prompt_only",
                    ),
                    "oracle_trained_oracle_trace_accuracy": (
                        "oracle_trained_oracle_trace_accuracy",
                        "mean_oracle_trained_oracle_trace_accuracy",
                    ),
                    "oracle_trained_predicted_trace_accuracy": (
                        "oracle_trained_predicted_trace_accuracy",
                        "mean_oracle_trained_predicted_trace_accuracy",
                    ),
                    "oracle_trained_random_trace_accuracy": (
                        "oracle_trained_random_trace_accuracy",
                        "mean_oracle_trained_random_trace_accuracy",
                    ),
                    "oracle_trained_trace_delta": ("oracle_trained_trace_delta", "mean_oracle_trained_trace_delta"),
                    "predicted_trace_gap_to_oracle_upper_bound": (
                        "predicted_trace_gap_to_oracle_upper_bound",
                        "mean_predicted_trace_gap_to_oracle_upper_bound",
                    ),
                    "cross_advisor_oracle_gap": ("cross_advisor_oracle_gap", "mean_cross_advisor_oracle_gap"),
                    "substrate_claim_score": ("substrate_claim_score", "mean_substrate_claim_score"),
                },
            )
        if isinstance(payload.get("cells"), dict):
            best_id, best_cell = _best_cell(payload["cells"])
            if best_id and isinstance(best_cell, dict):
                metrics["best_cell_accuracy"] = _best_cell_metric(best_cell)
        priority = (
            "strict_accuracy",
            "m24_2_promotion_candidate",
            "m24_2_promotion_gate_pass_rate",
            "m24_2_hard_bottleneck_compression_ratio",
            "m24_2_hard_bottleneck_token_count",
            "m24_2_gate_strict_accuracy_retained",
            "m24_2_gate_trace_beats_shuffled_strong",
            "m24_2_gate_trace_exact_floor",
            "m24_promotion_candidate",
            "m24_promotion_gate_pass_rate",
            "predicted_vs_shuffled_delta",
            "shuffled_trace_accuracy",
        )
        ordered_metrics: dict[str, float] = {}
        for key in priority:
            if key in metrics:
                ordered_metrics[key] = metrics[key]
        for key, value in metrics.items():
            ordered_metrics.setdefault(key, value)
        return ordered_metrics
    if stage_key == "M25":
        metrics: dict[str, float] = {}
        headline = payload.get("headline_metrics", {})
        aggregate = payload.get("aggregate_metrics", {})
        top = payload.get("metrics", {})
        for source in (headline, aggregate, top, payload):
            if not isinstance(source, dict):
                continue
            _merge_existing_metrics(
                metrics,
                source,
                {
                    "strict_accuracy": ("strict_accuracy", "mean_strict_accuracy"),
                    "predicted_stream_accuracy": ("predicted_stream_accuracy", "mean_predicted_stream_accuracy"),
                    "oracle_stream_accuracy": ("oracle_stream_accuracy", "mean_oracle_stream_accuracy"),
                    "shuffled_stream_accuracy": ("shuffled_stream_accuracy", "mean_shuffled_stream_accuracy"),
                    "random_stream_accuracy": ("random_stream_accuracy", "mean_random_stream_accuracy"),
                    "zero_stream_accuracy": ("zero_stream_accuracy", "mean_zero_stream_accuracy"),
                    "prompt_only_accuracy": ("prompt_only_accuracy", "mean_prompt_only_accuracy"),
                    "m25_strict_delta_vs_prompt_only": ("m25_strict_delta_vs_prompt_only", "mean_m25_strict_delta_vs_prompt_only"),
                    "predicted_vs_shuffled_delta": ("predicted_vs_shuffled_delta", "mean_predicted_vs_shuffled_delta"),
                    "predicted_vs_random_delta": ("predicted_vs_random_delta", "mean_predicted_vs_random_delta"),
                    "oracle_stream_delta": ("oracle_stream_delta", "mean_oracle_stream_delta"),
                    "stream_advisor_delta": ("stream_advisor_delta", "mean_stream_advisor_delta"),
                    "loose_stream_exact_accuracy": ("loose_stream_exact_accuracy", "mean_loose_stream_exact_accuracy"),
                    "stream_type_accuracy": ("stream_type_accuracy", "mean_stream_type_accuracy"),
                    "stream_value_accuracy": ("stream_value_accuracy", "mean_stream_value_accuracy"),
                    "stream_aux_accuracy": ("stream_aux_accuracy", "mean_stream_aux_accuracy"),
                    "loose_symbol_to_prompt_ratio": ("loose_symbol_to_prompt_ratio", "mean_loose_symbol_to_prompt_ratio"),
                    "prompt_to_loose_symbol_ratio": ("prompt_to_loose_symbol_ratio", "mean_prompt_to_loose_symbol_ratio"),
                    "token_reduction_ratio": ("token_reduction_ratio", "mean_token_reduction_ratio"),
                    "accuracy_per_loose_symbol": ("accuracy_per_loose_symbol", "mean_accuracy_per_loose_symbol"),
                    "accuracy_per_prompt_token": ("accuracy_per_prompt_token", "mean_accuracy_per_prompt_token"),
                    "m25_promotion_gate_pass_rate": ("m25_promotion_gate_pass_rate", "mean_m25_promotion_gate_pass_rate"),
                    "m25_promotion_candidate": ("m25_promotion_candidate", "mean_m25_promotion_candidate"),
                    "m25_gate_strict_accuracy_retained": ("m25_gate_strict_accuracy_retained", "mean_m25_gate_strict_accuracy_retained"),
                    "m25_gate_stream_beats_shuffled": ("m25_gate_stream_beats_shuffled", "mean_m25_gate_stream_beats_shuffled"),
                    "m25_gate_stream_beats_random": ("m25_gate_stream_beats_random", "mean_m25_gate_stream_beats_random"),
                    "m25_gate_token_reduction_positive": ("m25_gate_token_reduction_positive", "mean_m25_gate_token_reduction_positive"),
                    "m25_gate_nonzero_stream_reconstruction": ("m25_gate_nonzero_stream_reconstruction", "mean_m25_gate_nonzero_stream_reconstruction"),
                    "m25_gate_symbolic_trace_only": ("m25_gate_symbolic_trace_only", "mean_m25_gate_symbolic_trace_only"),
                    "generator_parameters_unchanged_after_advisor": ("generator_parameters_unchanged_after_advisor", "mean_generator_parameters_unchanged_after_advisor"),
                },
            )
        priority = (
            "strict_accuracy",
            "m25_promotion_candidate",
            "m25_promotion_gate_pass_rate",
            "loose_stream_exact_accuracy",
            "token_reduction_ratio",
            "predicted_vs_shuffled_delta",
            "predicted_vs_random_delta",
        )
        ordered_metrics: dict[str, float] = {}
        for key in priority:
            if key in metrics:
                ordered_metrics[key] = metrics[key]
        for key, value in metrics.items():
            ordered_metrics.setdefault(key, value)
        return ordered_metrics
    return _generic_metrics(payload)


def _merge_existing_metrics(
    target: dict[str, float],
    source: dict[str, Any],
    mapping: dict[str, str | tuple[str, ...]],
) -> None:
    for out_key, raw_keys in mapping.items():
        candidates = (raw_keys,) if isinstance(raw_keys, str) else raw_keys
        for raw_key in candidates:
            if raw_key in source:
                target[out_key] = _float_or_zero(source.get(raw_key))
                break


def _best_cell(cells: dict[str, Any]) -> tuple[str | None, dict[str, Any] | None]:
    best_id = None
    best_cell = None
    best_value = float("-inf")
    for cell_id, cell in cells.items():
        if not isinstance(cell, dict):
            continue
        value = _best_cell_metric(cell)
        if value > best_value:
            best_id = str(cell_id)
            best_cell = cell
            best_value = value
    return best_id, best_cell


def _best_cell_metric(cell: dict[str, Any]) -> float:
    metrics = cell.get("metrics", {})
    if isinstance(metrics, dict):
        for key in ("strict_accuracy", "mean_strict_accuracy", "overall_accuracy", "held_out_accuracy", "accuracy"):
            if key in metrics:
                return _float_or_zero(metrics[key])
    aggregate = cell.get("aggregate_metrics", {})
    if isinstance(aggregate, dict):
            for key in ("strict_accuracy", "mean_strict_accuracy", "synthetic_world_accuracy", "factorized_exact_accuracy", "bridi_trace_exact_accuracy", "mean_bridi_trace_exact_accuracy"):
                if key in aggregate:
                    return _float_or_zero(aggregate[key])
    for key in ("accuracy", "overall_accuracy", "held_out_accuracy"):
        if key in cell:
            return _float_or_zero(cell[key])
    return float("-inf")


def _pick_metrics(values: dict[str, Any]) -> dict[str, float]:
    picked: dict[str, float] = {}
    for wanted in KEY_METRICS:
        for key, value in values.items():
            if isinstance(key, str) and (key == wanted or key.endswith(f".{wanted}")) and isinstance(value, (int, float)):
                picked[wanted] = float(value)
                break
    return picked


def _numeric_leaves(payload: Any, prefix: str = "") -> dict[str, float]:
    leaves: dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            leaves.update(_numeric_leaves(value, child))
    elif isinstance(payload, list):
        for idx, value in enumerate(payload):
            leaves.update(_numeric_leaves(value, f"{prefix}[{idx}]"))
    elif isinstance(payload, (int, float)):
        leaves[prefix] = float(payload)
    return leaves


def _phase_eval_metrics(legacy_grid: dict[str, Any] | None) -> dict[str, float]:
    metrics: dict[str, float] = {}
    phase5_train = _legacy_lane(legacy_grid, "phase5_train")
    if isinstance(phase5_train, dict) and isinstance(phase5_train.get("metrics_digest"), dict):
        digest = phase5_train["metrics_digest"]
        metrics["phase5_train_executed_variants"] = _float_or_zero(digest.get("executed_variants"))
        metrics["phase5_train_mean_total_loss"] = _float_or_zero(digest.get("mean_total_loss"))
    english_duel = _legacy_lane(legacy_grid, "english_cot_duel")
    if isinstance(english_duel, dict) and isinstance(english_duel.get("metrics_digest"), dict):
        digest = english_duel["metrics_digest"]
        metrics["english_cot_base_acc"] = _float_or_zero(digest.get("base_acc"))
        metrics["english_cot_adapter_acc"] = _float_or_zero(digest.get("english_cot_adapter_acc"))
        metrics["english_cot_lojban_adapter_acc"] = _float_or_zero(digest.get("lojban_adapter_acc"))
    return metrics


def _legacy_lane(legacy_grid: dict[str, Any] | None, lane_name: str | None) -> dict[str, Any] | None:
    if not lane_name or not isinstance(legacy_grid, dict):
        return None
    lanes = legacy_grid.get("lanes", [])
    if not isinstance(lanes, list):
        return None
    for lane in lanes:
        if isinstance(lane, dict) and str(lane.get("lane", "")) == lane_name:
            return lane
    return None


def _legacy_grid_status(legacy_grid: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(legacy_grid, dict):
        return {"available": False}
    statuses: dict[str, str] = {}
    for lane in legacy_grid.get("lanes", []):
        if isinstance(lane, dict):
            statuses[str(lane.get("lane", ""))] = str(lane.get("status", ""))
    return {"available": True, "run_id": legacy_grid.get("run_id"), "statuses": statuses}


def _coverage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    surface_counts: dict[str, int] = {}
    for row in rows:
        surface = str(row.get("surface_kind", "unknown"))
        surface_counts[surface] = surface_counts.get(surface, 0) + 1
    return {
        "stage_count": len(rows),
        "surface_counts": surface_counts,
        "fresh_stage_count": surface_counts.get("fresh_legacy_lane", 0),
        "artifact_anchor_count": surface_counts.get("artifact_anchor", 0),
        "history_only_count": surface_counts.get("history_only", 0),
    }


def _row(
    stage: dict[str, Any],
    surface_kind: str,
    anchor_path: str | None,
    supplemental_paths: list[str],
    headline_metrics: dict[str, float],
    notes: list[str],
) -> dict[str, Any]:
    return {
        "stage_key": stage.get("stage_key"),
        "title": stage.get("title"),
        "stage_kind": stage.get("stage_kind"),
        "program_layer": stage.get("program_layer"),
        "entry_count": int(stage.get("entry_count") or 0),
        "runnable_count": int(stage.get("runnable_count") or 0),
        "artifact_only_count": int(stage.get("artifact_only_count") or 0),
        "doc_only_count": int(stage.get("doc_only_count") or 0),
        "legacy_origin": stage.get("legacy_origin"),
        "selected_upstream": stage.get("selected_upstream"),
        "question_boundary": stage.get("question_boundary"),
        "architectural_thesis": stage.get("architectural_thesis") or stage.get("objective"),
        "historical_comparison_families": list(stage.get("historical_comparison_families", [])),
        "required_test_contracts": list(stage.get("required_test_contracts", [])),
        "comparison_targets": list(stage.get("comparison_targets", [])),
        "surface_kind": surface_kind,
        "anchor_path": anchor_path,
        "supplemental_paths": supplemental_paths,
        "headline_metrics": headline_metrics,
        "notes": notes,
    }


def _render_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Whole Ablation Grid",
        "",
        f"- run_id: `{manifest.get('run_id', '')}`",
        f"- generated: `{manifest.get('timestamp', '')}`",
        f"- history manifest: `{manifest.get('source_manifests', {}).get('history_manifest', '')}`",
        f"- program spine manifest: `{manifest.get('source_manifests', {}).get('program_spine_manifest', '')}`",
        "",
        "## Coverage",
        "",
        f"- stages: `{manifest.get('coverage_summary', {}).get('stage_count', 0)}`",
        f"- fresh legacy surfaces: `{manifest.get('coverage_summary', {}).get('fresh_stage_count', 0)}`",
        f"- artifact anchors: `{manifest.get('coverage_summary', {}).get('artifact_anchor_count', 0)}`",
        f"- history-only stages: `{manifest.get('coverage_summary', {}).get('history_only_count', 0)}`",
        "",
        "## Legacy Grid Status",
        "",
    ]
    legacy_status = manifest.get("legacy_grid_status", {})
    if legacy_status.get("available"):
        lines.append(f"- legacy run_id: `{legacy_status.get('run_id', '')}`")
        for lane, status in sorted((legacy_status.get("statuses") or {}).items()):
            lines.append(f"- `{lane}`: `{status}`")
    else:
        lines.append("- no legacy grid manifest available")

    lines.extend(["", "## Stage Table", "", "| stage | surface | counts | anchor | headline |", "|---|---|---|---|---|"])
    for row in manifest.get("stage_rows", []):
        if not isinstance(row, dict):
            continue
        counts = f"e={row.get('entry_count', 0)} r={row.get('runnable_count', 0)} a={row.get('artifact_only_count', 0)} d={row.get('doc_only_count', 0)}"
        anchor = str(row.get("anchor_path") or "")
        headline = ", ".join(f"{k}={_fmt_metric(v)}" for k, v in list((row.get("headline_metrics") or {}).items())[:4])
        lines.append(f"| `{row.get('stage_key', '')}` | `{row.get('surface_kind', '')}` | `{counts}` | `{anchor}` | {headline} |")

    lines.extend(["", "## Comparison Policy", ""])
    for row in manifest.get("stage_rows", []):
        if not isinstance(row, dict):
            continue
        compare_targets = [
            str(target.get("target", ""))
            for target in row.get("comparison_targets", [])
            if isinstance(target, dict) and str(target.get("target", "")).strip()
        ]
        required_tests = row.get("required_test_contracts", [])
        if not compare_targets and not required_tests:
            continue
        lines.append(f"### {row.get('stage_key', '')}")
        lines.append("")
        if compare_targets:
            lines.append(f"- automatic compare-against: `{', '.join(compare_targets)}`")
        if row.get("historical_comparison_families"):
            lines.append(f"- historical families carried forward: `{', '.join(row.get('historical_comparison_families', []))}`")
        if required_tests:
            lines.append(f"- required test contracts: `{', '.join(required_tests)}`")
        lines.append("")

    lines.extend(["", "## Read", ""])
    lines.append("- The fresh part of the whole grid is now the recovered legacy runnable surface: A-G, H/H5/J, L6, and the phase-eval lanes under one manifest.")
    lines.append("- The modern M rows are represented through artifact-backed anchors and the control-plane lineage manifests, so the whole program is visible without pretending every stage was freshly retrained.")
    lines.append("- M3 remains the generative bridge archaeology block, M11 the discriminative oracle, M18 the controller-era comparison family, M19 the bounded runway mainline, M20 the dictionary-first substrate branch, M21 the dynamic bridi substrate branch, M22 the semantic-coverage generalization gate, M23 the causal relevance-router fork, M24 the substrate-first compression fork, and M25 the emergent loose bridi grammar stream fork.")
    return "\n".join(lines) + "\n"


def _limit_metrics(metrics: dict[str, float], limit: int = 8) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in metrics.items():
        if len(out) >= limit:
            break
        out[key] = value
    return out


def _read_json_required(path: Path | None) -> dict[str, Any]:
    payload = _read_json_optional(path)
    if payload is None:
        raise FileNotFoundError(f"Required JSON not found: {path}")
    return payload


def _read_json_optional(path: Path | None) -> dict[str, Any] | None:
    return _artifact_read_json_optional(path, swallow_errors=True)


def _latest_named_manifest(root: Path, filename: str) -> Path | None:
    return _artifact_latest_named_manifest(root, filename, recursive=False, newest_first=True, path_filter=None)


def _latest_direct_unified_eval_anchor(family_key: str) -> Path | None:
    if not DEFAULT_DIRECT_UNIFIED_EVAL_ROOT.exists():
        return None
    matches = sorted(
        DEFAULT_DIRECT_UNIFIED_EVAL_ROOT.glob("*/direct_unified_eval_manifest.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in matches:
        payload = _read_json_optional(path)
        if isinstance(payload, dict) and str(payload.get("family_key") or "").upper() == family_key.upper():
            return path
    return None


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


def _repo_relative(path: Path | None) -> str | None:
    return _artifact_repo_relative_or_string(path, REPO_ROOT)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _fmt_metric(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    main()
