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
    parser = argparse.ArgumentParser(description="Run eval-only M19 arity causal interventions.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", type=Path, required=True)
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--mode-list", type=str, default="predicted,oracle,random,force_1,force_8,no_mask")
    parser.add_argument("--benchmark-regimes", type=str, default="")
    parser.add_argument("--oracle-help-threshold", type=float, default=0.05)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--bottleneck-dim", type=int, default=128)
    parser.add_argument("--scratchpad-length", type=int, default=8)
    parser.add_argument("--max-latent-steps", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=M19_HIDDEN_SIZE)
    parser.add_argument("--tap-layer", type=int, default=12)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--typed-slot-layout", type=str, default="")
    parser.add_argument("--arity-router-mode", type=str, default="soft")
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
        default=Path(registry["output_roots"]["benchmark"]) / "arity_causal",
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
        _run_mode_if_needed(repo_root=repo_root, args=args, mode_label=mode_label, output_path=benchmark_path)
        payload = _read_json(benchmark_path)
        rows.append(_summarize_mode(mode_label, benchmark_path, payload, str(args.cell_id)))

    interpretation = _interpret_arity(rows, float(args.oracle_help_threshold))
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.arity_causal", "scripts/m19/run_m19_arity_causal_suite.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "mode_list": [row["mode"] for row in rows],
            "benchmark_regimes": str(args.benchmark_regimes),
            "oracle_help_threshold": float(args.oracle_help_threshold),
            "cell_id": str(args.cell_id),
            "strict_accuracy_canonical": True,
            "phrase_accuracy_diagnostic_only": True,
        },
        "mode_rows": rows,
        "interpretation": interpretation,
        "headline": {
            "predicted_accuracy": _mode_metric(rows, "predicted", "strict_accuracy"),
            "oracle_accuracy": _mode_metric(rows, "oracle", "strict_accuracy"),
            "oracle_delta_vs_predicted": interpretation.get("oracle_delta_vs_predicted"),
            "arity_router_bottleneck": interpretation.get("arity_router_bottleneck"),
        },
    }

    report_path = Path(args.output_path) if args.output_path else run_dir / "m19_arity_causal_report.json"
    assert_output_path_allowed("M", report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote arity causal report: {report_path}")


def _run_mode_if_needed(*, repo_root: Path, args: argparse.Namespace, mode_label: str, output_path: Path) -> None:
    override_mode, force_arity = _mode_to_override(mode_label)
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
        override_mode,
        "--force-arity",
        str(int(force_arity)),
        "--gumbel-temp-end",
        str(float(args.gumbel_temp_end)),
        "--geometry-mode",
        str(args.geometry_mode),
        "--poincare-curvature",
        str(float(args.poincare_curvature)),
        "--seed",
        str(int(args.seed)),
        "--track",
        str(args.track),
        "--cell-id",
        str(args.cell_id),
        "--regimes",
        str(args.benchmark_regimes).strip() or f"BASE,RANDOM-SHAPE,SCRATCHPAD-ONLY,{args.cell_id}",
        "--output-path",
        str(output_path),
    ]
    if args.gumbel_hard:
        cmd.append("--gumbel-hard")
    run_if_needed(Path(output_path), cmd, Path(repo_root))


def _summarize_mode(mode_label: str, benchmark_path: Path, payload: dict[str, Any], cell_id: str) -> dict[str, Any]:
    results = payload.get("results", {}) if isinstance(payload.get("results"), dict) else {}
    cell = results.get(cell_id, {}) if isinstance(results.get(cell_id), dict) else {}
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    return {
        "mode": mode_label,
        "report_path": str(benchmark_path).replace("\\", "/"),
        "strict_accuracy": _metric(metrics, "strict_accuracy", cell, "accuracy"),
        "phrase_accuracy": _metric(metrics, "overall_phrase_accuracy", cell, "phrase_accuracy"),
        "avg_tokens": _metric(metrics, "avg_tokens", cell, "avg_tokens"),
        "accuracy_per_token": _metric(metrics, "accuracy_per_token"),
        "arity_violation_rate": _metric(metrics, "arity_violation_rate", cell, "arity_violation_rate"),
        "masked_pointer_zero_rate": _metric(metrics, "masked_pointer_zero_rate", cell, "masked_pointer_zero_rate"),
        "typed_family_accuracy": _metric(metrics, "typed_family_accuracy", cell, "typed_family_accuracy"),
    }


def _interpret_arity(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    predicted = _mode_metric(rows, "predicted", "strict_accuracy")
    oracle = _mode_metric(rows, "oracle", "strict_accuracy")
    if predicted is None or oracle is None:
        return {
            "status": "inconclusive",
            "oracle_delta_vs_predicted": None,
            "arity_router_bottleneck": None,
        }
    delta = float(oracle) - float(predicted)
    return {
        "status": "oracle_helps" if delta >= float(threshold) else "oracle_does_not_help",
        "oracle_delta_vs_predicted": delta,
        "arity_router_bottleneck": bool(delta >= float(threshold)),
        "threshold": float(threshold),
    }


def _mode_to_override(mode_label: str) -> tuple[str, int]:
    label = str(mode_label).strip().lower()
    if label in {"predicted", "oracle", "random", "no_mask"}:
        return label, 1
    if label.startswith("force_"):
        return "force", int(label.split("_", 1)[1])
    raise ValueError(f"unsupported arity causal mode: {mode_label}")


def _parse_modes(text: str) -> list[str]:
    modes = [part.strip() for part in str(text).split(",") if part.strip()]
    if not modes:
        raise ValueError("mode-list must include at least one arity mode")
    return modes


def _metric(primary: dict[str, Any], primary_key: str, fallback: dict[str, Any] | None = None, fallback_key: str | None = None) -> float | None:
    value = primary.get(primary_key)
    if value is None and fallback is not None and fallback_key is not None:
        value = fallback.get(fallback_key)
    try:
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
