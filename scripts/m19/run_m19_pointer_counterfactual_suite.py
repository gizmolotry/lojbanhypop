from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.m19.artifact_contract import run_if_needed
from lojban_evolution.m19.family import M19_HIDDEN_SIZE, M19_REGISTRY
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs


POINTER_MODE_ALIASES = {
    "none": "scratchpad_only",
    "no_bridge": "scratchpad_only",
    "scratchpad": "scratchpad_only",
    "random": "random_shape",
    "drop_pointer": "drop_judri",
    "drop_judri_slots": "drop_judri",
    "shuffle_judri": "reverse_judri",
}


def _track_key(track: str) -> str:
    candidate = str(track or "").strip()
    return candidate if candidate in M19_REGISTRY else "M19"


def _apply_track_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = M19_REGISTRY.get(_track_key(args.track), {}).get("defaults", {})
    if defaults:
        if not str(args.typed_slot_layout).strip() and defaults.get("typed_slot_layout"):
            args.typed_slot_layout = str(defaults["typed_slot_layout"])
        if str(args.arity_router_mode).strip() == "soft" and defaults.get("arity_router_mode"):
            args.arity_router_mode = str(defaults["arity_router_mode"])
        if str(args.geometry_mode).strip() == "euclidean" and defaults.get("geometry_mode"):
            args.geometry_mode = str(defaults["geometry_mode"])
        if not args.gumbel_hard and str(defaults.get("arity_router_mode", "")).strip() == "gumbel_hard":
            args.gumbel_hard = True
    return args


def parse_args() -> argparse.Namespace:
    registry = M19_REGISTRY["M19.31"]
    parser = argparse.ArgumentParser(description="Run eval-only M19 judri pointer counterfactual interventions.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", type=Path, required=True)
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument(
        "--mode-list",
        type=str,
        default="full,drop_judri,swap_judri,reverse_judri,rotate_judri,judri_only,gismu_only",
    )
    parser.add_argument("--pointer-causality-threshold", type=float, default=0.05)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=M19_HIDDEN_SIZE)
    parser.add_argument("--tap-layer", type=int, default=12)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
    parser.add_argument("--arity-override-mode", type=str, default="predicted", choices=["predicted", "oracle", "random", "force", "no_mask"])
    parser.add_argument("--force-arity", type=int, default=1)
    parser.add_argument("--gumbel-hard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--gumbel-temp-end", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--track", type=str, default="M19.31")
    parser.add_argument("--cell-id", type=str, default="M19.3_8Q_128D_8S")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(registry["output_roots"]["benchmark"]) / "pointer_counterfactual",
    )
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return _apply_track_defaults(parser.parse_args())


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output_root)
    assert_output_path_allowed("M", output_root)
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / run_id
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mode_label in _parse_modes(args.mode_list):
        mode_dir = run_dir / mode_label
        mode_dir.mkdir(parents=True, exist_ok=True)
        benchmark_path = mode_dir / "benchmark_report.json"
        regime_id = _regime_for_mode(mode_label, str(args.cell_id))
        _run_mode_if_needed(repo_root=repo_root, args=args, mode_label=mode_label, output_path=benchmark_path)
        payload = _read_json(benchmark_path)
        rows.append(_summarize_mode(mode_label, regime_id, benchmark_path, payload))

    interpretation = _interpret_pointer_counterfactuals(rows, float(args.pointer_causality_threshold))
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.pointer_counterfactual", "scripts/m19/run_m19_pointer_counterfactual_suite.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "mode_list": [row["mode"] for row in rows],
            "pointer_causality_threshold": float(args.pointer_causality_threshold),
            "arity_override_mode": str(args.arity_override_mode),
            "force_arity": int(args.force_arity),
            "cell_id": str(args.cell_id),
            "strict_accuracy_canonical": True,
            "phrase_accuracy_diagnostic_only": True,
        },
        "mode_rows": rows,
        "interpretation": interpretation,
        "headline": {
            "full_accuracy": _mode_metric(rows, "full", "strict_accuracy"),
            "drop_judri_accuracy": _mode_metric(rows, "drop_judri", "strict_accuracy"),
            "no_judri_accuracy": _mode_metric(rows, "no_judri", "strict_accuracy"),
            "swap_judri_accuracy": _mode_metric(rows, "swap_judri", "strict_accuracy"),
            "reverse_judri_accuracy": _mode_metric(rows, "reverse_judri", "strict_accuracy"),
            "rotate_judri_accuracy": _mode_metric(rows, "rotate_judri", "strict_accuracy"),
            "pointer_binding_causal": interpretation.get("pointer_binding_causal"),
            "pointer_counterfactual_shortcut_warning": interpretation.get("pointer_counterfactual_shortcut_warning"),
        },
    }

    report_path = Path(args.output_path) if args.output_path else run_dir / "m19_pointer_counterfactual_report.json"
    assert_output_path_allowed("M", report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote pointer counterfactual report: {report_path}")


def _run_mode_if_needed(*, repo_root: Path, args: argparse.Namespace, mode_label: str, output_path: Path) -> None:
    channel_mode = _bridge_channel_mode(mode_label)
    regime_id = _regime_for_mode(mode_label, str(args.cell_id))
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
        "--base-model",
        str(args.base_model),
        "--bridge-path",
        str(args.bridge_path),
        "--eval-data-path",
        str(args.eval_data_path),
        "--eval-size",
        str(int(args.eval_size)),
        "--num-queries",
        str(int(args.num_queries)),
        "--bottleneck-dim",
        str(int(args.bottleneck_dim)),
        "--scratchpad-length",
        str(int(args.scratchpad_length)),
        "--max-latent-steps",
        str(int(args.max_latent_steps)),
        "--hidden-size",
        str(int(args.hidden_size)),
        "--tap-layer",
        str(int(args.tap_layer)),
        "--random-scale",
        str(float(args.random_scale)),
        "--typed-slot-layout",
        str(args.typed_slot_layout),
        "--arity-router-mode",
        str(args.arity_router_mode),
        "--arity-override-mode",
        str(args.arity_override_mode),
        "--force-arity",
        str(int(args.force_arity)),
        "--gumbel-temp-end",
        str(float(args.gumbel_temp_end)),
        "--geometry-mode",
        str(args.geometry_mode),
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
        "--bridge-channel-mode",
        channel_mode,
        "--seed",
        str(int(args.seed)),
        "--track",
        str(args.track),
        "--cell-id",
        str(args.cell_id),
        "--regimes",
        regime_id,
        "--output-path",
        str(output_path),
    ]
    if args.gumbel_hard:
        cmd.append("--gumbel-hard")
    run_if_needed(Path(output_path), cmd, Path(repo_root))


def _summarize_mode(mode_label: str, regime_id: str, benchmark_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    results = payload.get("results", {}) if isinstance(payload.get("results"), dict) else {}
    row = results.get(regime_id, {}) if isinstance(results.get(regime_id), dict) else {}
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    prediction_summaries = payload.get("prediction_summaries", {}) if isinstance(payload.get("prediction_summaries"), dict) else {}
    top_predictions = []
    summary = prediction_summaries.get(regime_id, {})
    if isinstance(summary, dict) and isinstance(summary.get("top_predictions"), list):
        top_predictions = summary.get("top_predictions", [])[:5]
    strict_accuracy = _metric(row, "accuracy")
    avg_tokens = _metric(row, "avg_tokens")
    avg_runway_tokens = _metric(row, "avg_runway_tokens")
    return {
        "mode": mode_label,
        "observed_bridge_channel_mode": config.get("bridge_channel_mode"),
        "regime_id": regime_id,
        "report_path": str(benchmark_path).replace("\\", "/"),
        "strict_accuracy": strict_accuracy,
        "phrase_accuracy": _metric(row, "phrase_accuracy"),
        "avg_tokens": avg_tokens,
        "accuracy_per_token": _safe_div(strict_accuracy, avg_tokens),
        "avg_runway_tokens": avg_runway_tokens,
        "accuracy_per_runway_token": _safe_div(strict_accuracy, avg_runway_tokens),
        "typed_family_accuracy": _metric(row, "typed_family_accuracy"),
        "arity_violation_rate": _metric(row, "arity_violation_rate"),
        "masked_pointer_zero_rate": _metric(row, "masked_pointer_zero_rate"),
        "bridge_channel_retained_slot_fraction": _metric(row, "bridge_channel_retained_slot_fraction"),
        "top_predictions": top_predictions,
    }


def _interpret_pointer_counterfactuals(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    full = _mode_metric(rows, "full", "strict_accuracy")
    if full is None:
        return {"status": "inconclusive", "reason": "missing full mode"}
    corrupt_modes = ["drop_judri", "no_judri", "swap_judri", "reverse_judri", "rotate_judri"]
    corrupt_scores = {
        mode: score
        for mode in corrupt_modes
        if (score := _mode_metric(rows, mode, "strict_accuracy")) is not None
    }
    best_corrupt_mode = None
    best_corrupt_accuracy = None
    worst_corrupt_mode = None
    worst_corrupt_accuracy = None
    if corrupt_scores:
        best_corrupt_mode, best_corrupt_accuracy = max(corrupt_scores.items(), key=lambda item: item[1])
        worst_corrupt_mode, worst_corrupt_accuracy = min(corrupt_scores.items(), key=lambda item: item[1])
    min_corrupt_drop = (float(full) - float(best_corrupt_accuracy)) if best_corrupt_accuracy is not None else None
    max_corrupt_retention = (
        float(best_corrupt_accuracy) / float(full)
        if best_corrupt_accuracy is not None and float(full) != 0.0
        else None
    )
    return {
        "status": "complete",
        "threshold": float(threshold),
        "full_accuracy": float(full),
        "corrupt_scores": corrupt_scores,
        "best_corrupt_mode": best_corrupt_mode,
        "best_corrupt_accuracy": float(best_corrupt_accuracy) if best_corrupt_accuracy is not None else None,
        "worst_corrupt_mode": worst_corrupt_mode,
        "worst_corrupt_accuracy": float(worst_corrupt_accuracy) if worst_corrupt_accuracy is not None else None,
        "max_corrupt_retention": max_corrupt_retention,
        "min_corrupt_drop": min_corrupt_drop,
        "drop_judri_delta": _delta_vs_full(rows, "drop_judri", full),
        "no_judri_delta": _delta_vs_full(rows, "no_judri", full),
        "swap_judri_delta": _delta_vs_full(rows, "swap_judri", full),
        "reverse_judri_delta": _delta_vs_full(rows, "reverse_judri", full),
        "rotate_judri_delta": _delta_vs_full(rows, "rotate_judri", full),
        "pointer_binding_causal": bool(min_corrupt_drop is not None and min_corrupt_drop >= float(threshold)),
        "pointer_counterfactual_shortcut_warning": bool(
            best_corrupt_accuracy is not None and float(best_corrupt_accuracy) >= float(full) - float(threshold)
        ),
    }


def _canonical_mode(mode_label: str) -> str:
    label = str(mode_label).strip().lower()
    return POINTER_MODE_ALIASES.get(label, label)


def _bridge_channel_mode(mode_label: str) -> str:
    label = _canonical_mode(mode_label)
    if label in {"scratchpad_only", "random_shape"}:
        return "full"
    return label


def _regime_for_mode(mode_label: str, cell_id: str) -> str:
    label = _canonical_mode(mode_label)
    if label == "scratchpad_only":
        return "SCRATCHPAD-ONLY"
    if label == "random_shape":
        return "RANDOM-SHAPE"
    return str(cell_id)


def _parse_modes(text: str) -> list[str]:
    modes = [_canonical_mode(part) for part in str(text).split(",") if part.strip()]
    if not modes:
        raise ValueError("mode-list must include at least one channel mode")
    return modes


def _metric(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _mode_metric(rows: list[dict[str, Any]], mode: str, metric: str) -> float | None:
    for row in rows:
        if str(row.get("mode")) == str(mode):
            value = row.get(metric)
            return float(value) if value is not None else None
    return None


def _delta_vs_full(rows: list[dict[str, Any]], mode: str, full_accuracy: float) -> float | None:
    value = _mode_metric(rows, mode, "strict_accuracy")
    return (float(full_accuracy) - float(value)) if value is not None else None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
