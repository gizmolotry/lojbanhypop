from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata


CORE_LANES = ("a_to_g", "hj", "l6", "phase5_objective")
OPTIONAL_LANES = ("phase5_train", "english_cot_duel")
ALL_LANES = CORE_LANES + OPTIONAL_LANES


@dataclass
class LaneResult:
    lane: str
    title: str
    status: str
    return_code: Optional[int]
    command: List[str]
    output_root: str
    artifact_path: Optional[str]
    artifact_kind: Optional[str]
    metrics_digest: Optional[Dict[str, object]]
    notes: str


def _norm_path(value: str | Path | None) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("\\", "/")


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_mean(values: Iterable[float]) -> float:
    xs = [float(v) for v in values]
    return sum(xs) / len(xs) if xs else 0.0


def _read_json(path: Path | None) -> Optional[dict[str, Any]]:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _run(cmd: Sequence[str], execute: bool) -> tuple[str, Optional[int]]:
    if not execute:
        return "planned", None
    try:
        rc = int(subprocess.run(list(cmd), check=False).returncode)
    except Exception:
        return "failed", -999
    return ("ok", rc) if rc == 0 else ("failed", rc)


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def _newest_match(root: Path, pattern: str) -> Optional[Path]:
    if not root.exists():
        return None
    matches = [path for path in root.glob(pattern) if path.is_file()]
    if not matches:
        return None
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def _normalize_lane_list(raw: Optional[Sequence[str]]) -> list[str]:
    if not raw:
        return list(CORE_LANES)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw:
        lane = str(item).strip().lower()
        if lane not in ALL_LANES:
            raise ValueError(f"Unknown lane '{item}'. Allowed: {', '.join(ALL_LANES)}")
        if lane not in seen:
            normalized.append(lane)
            seen.add(lane)
    return normalized


def _resolve_selected_lanes(args: argparse.Namespace) -> list[str]:
    selected = _normalize_lane_list(args.only_lanes)
    if args.include_phase5_train and "phase5_train" not in selected:
        selected.append("phase5_train")
    if args.include_english_cot_duel and "english_cot_duel" not in selected:
        selected.append("english_cot_duel")
    return selected


def _lane_roots(args: argparse.Namespace, run_id: str) -> dict[str, Path]:
    return {
        "master": Path(args.master_output_root) / run_id,
        "a_to_g": Path(args.a_to_g_output_root) / run_id,
        "hj": Path(args.hj_output_root) / run_id,
        "l6": Path(args.l6_output_root) / run_id,
        "phase5_objective": Path(args.phase5_output_root) / run_id,
        "phase5_train": Path(args.phase5_train_output_root) / run_id,
        "english_cot_duel": Path(args.english_cot_duel_output_root) / run_id,
    }


def _discover_default_assets() -> dict[str, Optional[Path]]:
    return {
        "base_model": _first_existing([Path(r"C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct")]),
        "adapter": _first_existing([Path("runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5")]),
        "h3_bridge": _first_existing([Path("runs/projections/swiglu_midlayer_bridge_h3.pt")]),
        "h4_bridge": _first_existing([Path("runs/projections/swiglu_midlayer_bridge_h3_exp4.pt")]),
        "h5_checkpoint": _first_existing([Path("runs/i_series/20260302_172603/h5_checkpoint.pt")]),
        "h5_slice1_checkpoint": _first_existing([Path("runs/i_series/20260302_053614/h5_checkpoint.pt")]),
        "a_to_g_b2_adapter": _first_existing([Path("runs/adapter_qwen")]),
        "handoff_projection": _first_existing(
            [Path("artifacts/projections/bridge_projection.pt"), Path("runs/projections/bridge_projection.pt")]
        ),
    }


def _a_to_g_digest(payload: dict[str, Any]) -> Dict[str, object]:
    runs = payload.get("runs", [])
    by_id = {
        str(row.get("run_id", "")).strip(): row
        for row in runs
        if isinstance(row, dict) and str(row.get("run_id", "")).strip()
    }
    return {
        "executed_runs": float(sum(1 for row in runs if isinstance(row, dict) and row.get("status") == "ok")),
        "control_base_final_acc": _safe_float(by_id.get("A", {}).get("metrics", {}).get("control_base_final_acc", 0.0)),
        "coconut_handoff_final_acc": _safe_float(
            by_id.get("C", {}).get("metrics", {}).get("coconut_handoff_final_acc", 0.0)
        ),
        "nope_handoff_lift": _safe_float(by_id.get("D", {}).get("metrics", {}).get("handoff_lift", 0.0)),
    }


def _hj_digest(payload: dict[str, Any]) -> Dict[str, object]:
    runs = payload.get("runs", [])
    by_id = {
        str(row.get("run_id", "")).strip(): row
        for row in runs
        if isinstance(row, dict) and str(row.get("run_id", "")).strip()
    }
    return {
        "executed_runs": float(sum(1 for row in runs if isinstance(row, dict) and row.get("status") == "ok")),
        "h1_handoff_lift": _safe_float(by_id.get("H1", {}).get("metrics", {}).get("handoff_lift", 0.0)),
        "h5_ood_accuracy": _safe_float(by_id.get("H5-OOD", {}).get("metrics", {}).get("ood_accuracy", 0.0)),
        "j1_schema_valid_rate": _safe_float(by_id.get("J-1", {}).get("metrics", {}).get("schema_valid_rate", 0.0)),
        "j5_accepted_foil_pair_accuracy": _safe_float(
            by_id.get("J-5", {}).get("metrics", {}).get("accepted_foil_pair_accuracy", 0.0)
        ),
    }


def _l6_digest(payload: dict[str, Any]) -> Dict[str, object]:
    rows = payload.get("rows", [])
    scopes = [
        _safe_float(row.get("final_constraint_scope", 0.0))
        for row in rows
        if isinstance(row, dict) and row.get("status") == "ok"
    ]
    return {
        "executed_rows": float(sum(1 for row in rows if isinstance(row, dict) and row.get("status") == "ok")),
        "mean_scope_constraint": _safe_mean(scopes),
        "best_scope_constraint": max(scopes) if scopes else 0.0,
    }


def _phase5_objective_digest(payload: dict[str, Any]) -> Dict[str, object]:
    terms = payload.get("terms", {})
    dominant_name = ""
    dominant_value = 0.0
    if isinstance(terms, dict) and terms:
        dominant_name, dominant_value = max(
            ((str(key), _safe_float(value)) for key, value in terms.items()),
            key=lambda item: item[1],
            default=("", 0.0),
        )
    return {
        "full_total_regularizer": _safe_float(payload.get("full_total_regularizer", 0.0)),
        "dead_term_count": float(len(payload.get("dead_terms", []))) if isinstance(payload.get("dead_terms"), list) else 0.0,
        "dominant_term": dominant_name,
        "dominant_term_value": dominant_value,
    }


def _phase5_train_digest(payload: dict[str, Any]) -> Dict[str, object]:
    rows = payload.get("variants", [])
    ok_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "ok"]
    failed_rows = [row for row in rows if isinstance(row, dict) and row.get("status") == "failed"]
    total_losses = [
        _safe_float(row.get("last_metrics", {}).get("total_loss", 0.0))
        for row in ok_rows
        if isinstance(row.get("last_metrics"), dict)
    ]
    return {
        "variant_count": float(len(rows)),
        "executed_variants": float(len(ok_rows)),
        "failed_variants": float(len(failed_rows)),
        "mean_total_loss": _safe_mean(total_losses),
    }


def _english_duel_digest(payload: dict[str, Any]) -> Dict[str, object]:
    comparison = payload.get("comparison", {})
    if not isinstance(comparison, dict):
        comparison = {}
    return {
        "base_acc": _safe_float(comparison.get("base_acc", 0.0)),
        "english_cot_adapter_acc": _safe_float(comparison.get("english_cot_adapter_acc", 0.0)),
        "english_lift_vs_base": _safe_float(comparison.get("english_lift_vs_base", 0.0)),
        "lojban_adapter_acc": _safe_float(comparison.get("lojban_adapter_acc", 0.0)),
        "english_train_failed": 1.0 if str(payload.get("train_english_cot", {}).get("status", "")) == "failed" else 0.0,
    }


def _lane_title(lane: str) -> str:
    return {
        "a_to_g": "A-G Coconut Matrix",
        "hj": "H/H5/J Legacy Surface",
        "l6": "L6 Constraint Branch",
        "phase5_objective": "Phase-5 Objective Ablation",
        "phase5_train": "Phase-5 Training Ablation",
        "english_cot_duel": "English CoT Control Duel",
    }[lane]


def _write_summary_md(path: Path, manifest: dict[str, Any]) -> None:
    lane_rows = manifest.get("lanes", [])
    if not isinstance(lane_rows, list):
        lane_rows = []
    lines: list[str] = []
    lines.append("# Legacy Ablation Grid")
    lines.append("")
    lines.append(f"- run_id: `{manifest.get('run_id', '')}`")
    lines.append(f"- timestamp: `{manifest.get('timestamp', '')}`")
    lines.append(f"- execute: `{str(bool(manifest.get('execute', False))).lower()}`")
    lines.append(f"- aggregate_only: `{str(bool(manifest.get('aggregate_only', False))).lower()}`")
    lines.append("")
    lines.append("| lane | title | status | artifact | headline digest |")
    lines.append("|---|---|---|---|---|")
    for row in lane_rows:
        if not isinstance(row, dict):
            continue
        digest = row.get("metrics_digest")
        digest_str = ""
        if isinstance(digest, dict):
            head = []
            for key in list(digest.keys())[:4]:
                head.append(f"{key}={digest[key]}")
            digest_str = ", ".join(head)
        lines.append(
            f"| `{row.get('lane', '')}` | `{row.get('title', '')}` | `{row.get('status', '')}` | "
            f"`{row.get('artifact_path', '') or ''}` | {digest_str} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _aggregate_lane(lane: str, lane_root: Path) -> LaneResult:
    title = _lane_title(lane)
    if lane == "a_to_g":
        artifact = _newest_match(lane_root, "**/ablation_matrix.json")
        payload = _read_json(artifact)
        return LaneResult(lane, title, "ok" if payload else "missing", None, [], _norm_path(lane_root) or "", _norm_path(artifact), "ablation_matrix", _a_to_g_digest(payload) if payload else None, "Recovered A-G matrix lane.")
    if lane == "hj":
        artifact = _newest_match(lane_root, "**/run_h_series.json")
        payload = _read_json(artifact)
        return LaneResult(lane, title, "ok" if payload else "missing", None, [], _norm_path(lane_root) or "", _norm_path(artifact), "legacy_hj_manifest", _hj_digest(payload) if payload else None, "Unified H/H5/J legacy surface.")
    if lane == "l6":
        artifact = _newest_match(lane_root, "**/l6_ablation_manifest.json")
        payload = _read_json(artifact)
        return LaneResult(lane, title, "ok" if payload else "missing", None, [], _norm_path(lane_root) or "", _norm_path(artifact), "l6_manifest", _l6_digest(payload) if payload else None, "Recovered L6 branch.")
    if lane == "phase5_objective":
        artifact = _newest_match(lane_root, "**/phase5_objective_ablation.json")
        payload = _read_json(artifact)
        return LaneResult(lane, title, "ok" if payload else "missing", None, [], _norm_path(lane_root) or "", _norm_path(artifact), "phase5_objective_manifest", _phase5_objective_digest(payload) if payload else None, "Closed-form objective surface.")
    if lane == "phase5_train":
        artifact = _newest_match(lane_root, "**/ablation_manifest.json")
        payload = _read_json(artifact)
        status = "missing"
        if payload:
            rows = payload.get("variants", [])
            if isinstance(rows, list) and rows:
                statuses = {str(row.get("status", "")) for row in rows if isinstance(row, dict)}
                if statuses == {"ok"}:
                    status = "ok"
                elif "ok" in statuses and "failed" in statuses:
                    status = "partial"
                elif "failed" in statuses:
                    status = "failed"
                else:
                    status = "planned"
            else:
                status = "ok"
        return LaneResult(lane, title, status, None, [], _norm_path(lane_root) or "", _norm_path(artifact), "phase5_train_manifest", _phase5_train_digest(payload) if payload else None, "Full training-ablation lane.")
    if lane == "english_cot_duel":
        artifact = _newest_match(lane_root, "**/english_cot_control_manifest.json")
        payload = _read_json(artifact)
        status = "missing"
        if payload:
            train_status = str(payload.get("train_english_cot", {}).get("status", ""))
            eval_status = str(payload.get("eval_english_cot", {}).get("status", ""))
            if train_status == "ok" and eval_status == "ok":
                status = "ok"
            elif train_status == "failed":
                status = "failed"
            elif train_status == "ok" and eval_status in {"planned", ""}:
                status = "partial"
            else:
                status = "planned"
        return LaneResult(lane, title, status, None, [], _norm_path(lane_root) or "", _norm_path(artifact), "english_duel_manifest", _english_duel_digest(payload) if payload else None, "English control branch lane.")
    raise ValueError(f"Unsupported lane '{lane}'")


def _run_a_to_g(args: argparse.Namespace, lane_root: Path) -> LaneResult:
    assert_output_path_allowed("A-G", lane_root)
    lane_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_coconut_ablation_matrix.py"),
        "--base-model",
        args.base_model,
        "--adapter",
        str(args.adapter),
        "--sample-size",
        str(args.sample_size),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--dataset-size",
        str(args.dataset_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--output-root",
        str(lane_root),
    ]
    if args.a_to_g_b2_adapter:
        cmd.extend(["--b2-adapter", str(args.a_to_g_b2_adapter)])
    if args.drope_adapter:
        cmd.extend(["--drope-adapter", str(args.drope_adapter)])
    if args.handoff_projection:
        cmd.extend(["--handoff-projection", str(args.handoff_projection)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.execute:
        cmd.append("--execute")
    status, rc = _run(cmd, args.execute)
    aggregated = _aggregate_lane("a_to_g", lane_root)
    aggregated.status = status if status != "ok" else aggregated.status
    aggregated.return_code = rc
    aggregated.command = cmd
    if status == "failed":
        aggregated.notes = "A-G matrix execution failed."
    return aggregated


def _run_hj(args: argparse.Namespace, lane_root: Path) -> LaneResult:
    assert_output_path_allowed("M", lane_root)
    lane_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_true_coconut_h_series.py"),
        "--base-model",
        args.base_model,
        "--adapter",
        str(args.adapter),
        "--sample-size",
        str(args.sample_size),
        "--seeds",
        *[str(seed) for seed in args.seeds],
        "--dataset-size",
        str(args.dataset_size),
        "--max-logic-new-tokens",
        str(args.max_logic_new_tokens),
        "--max-final-new-tokens",
        str(args.max_final_new_tokens),
        "--output-root",
        str(lane_root),
    ]
    if args.h5_checkpoint:
        cmd.extend(["--h5-checkpoint", str(args.h5_checkpoint)])
    if args.h5_slice1_checkpoint:
        cmd.extend(["--h5-prov-slice1-checkpoint", str(args.h5_slice1_checkpoint)])
    if args.h3_bridge:
        cmd.extend(["--h3-adapter", str(args.adapter), "--h3-bridge", str(args.h3_bridge)])
    if args.h4_bridge:
        cmd.extend(["--h4-bridge", str(args.h4_bridge)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.execute:
        cmd.append("--execute")
    status, rc = _run(cmd, args.execute)
    aggregated = _aggregate_lane("hj", lane_root)
    aggregated.status = status if status != "ok" else aggregated.status
    aggregated.return_code = rc
    aggregated.command = cmd
    if status == "failed":
        aggregated.notes = "H/H5/J surface execution failed."
    return aggregated


def _run_l6(args: argparse.Namespace, lane_root: Path) -> LaneResult:
    assert_output_path_allowed("M", lane_root)
    lane_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_l6_ablation_branch.py"),
        "--base-model",
        args.base_model,
        "--adapter",
        str(args.adapter),
        "--train-steps",
        str(args.l6_train_steps),
        "--dataset-size",
        str(args.l6_dataset_size),
        "--dataset-profile",
        args.l6_dataset_profile,
        "--difficulty-tier",
        args.l6_difficulty_tier,
        "--seed",
        str(args.l6_seed),
        "--output-root",
        str(lane_root),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.execute:
        cmd.append("--execute")
    status, rc = _run(cmd, args.execute)
    aggregated = _aggregate_lane("l6", lane_root)
    aggregated.status = status if status != "ok" else aggregated.status
    aggregated.return_code = rc
    aggregated.command = cmd
    if status == "failed":
        aggregated.notes = "L6 branch execution failed."
    return aggregated


def _run_phase5_objective(args: argparse.Namespace, lane_root: Path) -> LaneResult:
    lane_root.mkdir(parents=True, exist_ok=True)
    output = lane_root / "phase5_objective_ablation.json"
    summary_md = lane_root / "phase5_objective_ablation.md"
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_phase5_objective_ablation.py"),
        "--output",
        str(output),
        "--summary-md",
        str(summary_md),
        "--batch-size",
        str(args.phase5_batch_size),
        "--seq-len",
        str(args.phase5_seq_len),
        "--hidden-dim",
        str(args.phase5_hidden_dim),
        "--vocab-size",
        str(args.phase5_vocab_size),
        "--seed",
        str(args.phase5_seed),
    ]
    status, rc = _run(cmd, args.execute)
    aggregated = _aggregate_lane("phase5_objective", lane_root)
    aggregated.status = status if status != "ok" else aggregated.status
    aggregated.return_code = rc
    aggregated.command = cmd
    if status == "failed":
        aggregated.notes = "Phase-5 objective ablation execution failed."
    return aggregated


def _run_phase5_train(args: argparse.Namespace, lane_root: Path) -> LaneResult:
    if not args.phase5_train_dataset:
        return LaneResult("phase5_train", _lane_title("phase5_train"), "skipped", None, [], _norm_path(lane_root) or "", None, None, None, "Skipped: provide --phase5-train-dataset to execute this lane.")
    lane_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_phase5_train_ablation.py"),
        "--base-model",
        args.base_model,
        "--dataset",
        str(args.phase5_train_dataset),
        "--output-root",
        str(lane_root),
        "--epochs",
        str(args.phase5_train_epochs),
        "--max-length",
        str(args.phase5_train_max_length),
        "--per-device-batch-size",
        str(args.phase5_train_batch_size),
        "--grad-accum",
        str(args.phase5_train_grad_accum),
        "--lr",
        str(args.phase5_train_lr),
    ]
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.phase5_include_only_variants:
        cmd.append("--include-only-variants")
    if args.execute:
        cmd.append("--execute")
    status, rc = _run(cmd, args.execute)
    aggregated = _aggregate_lane("phase5_train", lane_root)
    aggregated.status = status if status != "ok" else aggregated.status
    aggregated.return_code = rc
    aggregated.command = cmd
    if status == "failed":
        aggregated.notes = "Phase-5 training ablation execution failed."
    return aggregated


def _run_english_cot_duel(args: argparse.Namespace, lane_root: Path) -> LaneResult:
    lane_root.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_english_cot_control_duel.py"),
        "--base-model",
        args.base_model,
        "--output-root",
        str(lane_root),
        "--dataset-size",
        str(args.english_duel_dataset_size),
        "--seeds",
        *[str(seed) for seed in args.english_duel_seeds],
        "--epochs",
        str(args.english_duel_epochs),
        "--max-length",
        str(args.english_duel_max_length),
        "--per-device-batch-size",
        str(args.english_duel_batch_size),
        "--grad-accum",
        str(args.english_duel_grad_accum),
        "--lr",
        str(args.english_duel_lr),
        "--eval-sample-size",
        str(args.english_duel_eval_sample_size),
        "--eval-seed",
        str(args.english_duel_eval_seed),
        "--eval-dataset-size",
        str(args.english_duel_eval_dataset_size),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.english_duel_lojban_adapter:
        cmd.extend(["--lojban-adapter", str(args.english_duel_lojban_adapter)])
    if args.local_files_only:
        cmd.append("--local-files-only")
    if args.execute:
        cmd.append("--execute")
    status, rc = _run(cmd, args.execute)
    aggregated = _aggregate_lane("english_cot_duel", lane_root)
    aggregated.status = status if status != "ok" else aggregated.status
    aggregated.return_code = rc
    aggregated.command = cmd
    if status == "failed":
        aggregated.notes = "English CoT control duel execution failed."
    return aggregated


def _build_manifest(args: argparse.Namespace, run_id: str, selected_lanes: Sequence[str], roots: dict[str, Path], lane_results: Sequence[LaneResult]) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "legacy_grid", "scripts/legacy/run_legacy_ablation_grid.py"),
        "run_id": run_id,
        "execute": bool(args.execute),
        "aggregate_only": bool(args.aggregate_only),
        "selected_lanes": list(selected_lanes),
        "base_model": args.base_model,
        "adapter": _norm_path(args.adapter),
        "roots": {key: _norm_path(value) for key, value in roots.items()},
        "asset_defaults": {
            "h3_bridge": _norm_path(args.h3_bridge),
            "h4_bridge": _norm_path(args.h4_bridge),
            "h5_checkpoint": _norm_path(args.h5_checkpoint),
            "h5_slice1_checkpoint": _norm_path(args.h5_slice1_checkpoint),
            "a_to_g_b2_adapter": _norm_path(args.a_to_g_b2_adapter),
            "handoff_projection": _norm_path(args.handoff_projection),
        },
        "lanes": [asdict(row) for row in lane_results],
    }


def parse_args() -> argparse.Namespace:
    defaults = _discover_default_assets()
    parser = argparse.ArgumentParser(description="Unified legacy ablation grid runner/aggregator.")
    parser.add_argument("--base-model", default=_norm_path(defaults["base_model"]), required=defaults["base_model"] is None)
    parser.add_argument("--adapter", default=_norm_path(defaults["adapter"]), required=defaults["adapter"] is None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--only-lanes", nargs="+", choices=ALL_LANES, default=None)
    parser.add_argument("--include-phase5-train", action="store_true")
    parser.add_argument("--include-english-cot-duel", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--master-output-root", default="artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid")
    parser.add_argument("--a-to-g-output-root", default="artifacts/runs/telemetry/raw/ablation/a_to_g/legacy_grid")
    parser.add_argument("--hj-output-root", default="artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid")
    parser.add_argument("--l6-output-root", default="runs/l_series/l6_ablation/legacy_grid")
    parser.add_argument("--phase5-output-root", default="artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid")
    parser.add_argument("--phase5-train-output-root", default="runs/phase5_train_ablation/legacy_grid")
    parser.add_argument("--english-cot-duel-output-root", default="runs/english_cot_control_duel/legacy_grid")
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--dataset-size", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-logic-new-tokens", type=int, default=16)
    parser.add_argument("--max-final-new-tokens", type=int, default=12)
    parser.add_argument("--h3-bridge", type=Path, default=defaults["h3_bridge"])
    parser.add_argument("--h4-bridge", type=Path, default=defaults["h4_bridge"])
    parser.add_argument("--h5-checkpoint", type=Path, default=defaults["h5_checkpoint"])
    parser.add_argument("--h5-slice1-checkpoint", type=Path, default=defaults["h5_slice1_checkpoint"])
    parser.add_argument("--a-to-g-b2-adapter", type=Path, default=defaults["a_to_g_b2_adapter"])
    parser.add_argument("--drope-adapter", type=Path, default=None)
    parser.add_argument("--handoff-projection", type=Path, default=defaults["handoff_projection"])
    parser.add_argument("--l6-train-steps", type=int, default=4)
    parser.add_argument("--l6-dataset-size", type=int, default=64)
    parser.add_argument("--l6-dataset-profile", default="legacy")
    parser.add_argument("--l6-difficulty-tier", default="all")
    parser.add_argument("--l6-seed", type=int, default=7)
    parser.add_argument("--phase5-batch-size", type=int, default=8)
    parser.add_argument("--phase5-seq-len", type=int, default=32)
    parser.add_argument("--phase5-hidden-dim", type=int, default=64)
    parser.add_argument("--phase5-vocab-size", type=int, default=128)
    parser.add_argument("--phase5-seed", type=int, default=7)
    parser.add_argument("--phase5-train-dataset", type=Path, default=None)
    parser.add_argument("--phase5-train-epochs", type=float, default=0.2)
    parser.add_argument("--phase5-train-max-length", type=int, default=256)
    parser.add_argument("--phase5-train-batch-size", type=int, default=2)
    parser.add_argument("--phase5-train-grad-accum", type=int, default=4)
    parser.add_argument("--phase5-train-lr", type=float, default=2e-4)
    parser.add_argument("--phase5-include-only-variants", action="store_true")
    parser.add_argument("--english-duel-dataset-size", type=int, default=256)
    parser.add_argument("--english-duel-seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--english-duel-epochs", type=float, default=0.2)
    parser.add_argument("--english-duel-max-length", type=int, default=256)
    parser.add_argument("--english-duel-batch-size", type=int, default=2)
    parser.add_argument("--english-duel-grad-accum", type=int, default=4)
    parser.add_argument("--english-duel-lr", type=float, default=2e-4)
    parser.add_argument("--english-duel-eval-sample-size", type=int, default=12)
    parser.add_argument("--english-duel-eval-seed", type=int, default=7)
    parser.add_argument("--english-duel-eval-dataset-size", type=int, default=256)
    parser.add_argument("--english-duel-lojban-adapter", type=Path, default=defaults["adapter"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = str(args.run_id).strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    selected_lanes = _resolve_selected_lanes(args)
    roots = _lane_roots(args, run_id)
    assert_output_path_allowed("M", roots["master"])
    assert_output_path_allowed("A-G", roots["a_to_g"])
    assert_output_path_allowed("M", roots["hj"])
    assert_output_path_allowed("M", roots["l6"])
    roots["master"].mkdir(parents=True, exist_ok=True)

    lane_results: list[LaneResult] = []
    if args.aggregate_only:
        for lane in selected_lanes:
            lane_results.append(_aggregate_lane(lane, roots[lane]))
    else:
        for lane in selected_lanes:
            if lane == "a_to_g":
                lane_results.append(_run_a_to_g(args, roots["a_to_g"]))
            elif lane == "hj":
                lane_results.append(_run_hj(args, roots["hj"]))
            elif lane == "l6":
                lane_results.append(_run_l6(args, roots["l6"]))
            elif lane == "phase5_objective":
                lane_results.append(_run_phase5_objective(args, roots["phase5_objective"]))
            elif lane == "phase5_train":
                lane_results.append(_run_phase5_train(args, roots["phase5_train"]))
            elif lane == "english_cot_duel":
                lane_results.append(_run_english_cot_duel(args, roots["english_cot_duel"]))
            else:
                raise ValueError(f"Unsupported lane '{lane}'")

    manifest = _build_manifest(args, run_id, selected_lanes, roots, lane_results)
    manifest_path = roots["master"] / "legacy_ablation_grid_manifest.json"
    summary_path = roots["master"] / "legacy_ablation_grid_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_summary_md(summary_path, manifest)
    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")
    for lane in lane_results:
        print(f"{lane.lane}: {lane.status} -> {lane.artifact_path or '<none>'}")


if __name__ == "__main__":
    main()
