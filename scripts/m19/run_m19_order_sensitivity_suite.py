from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict, deque
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
    parser = argparse.ArgumentParser(description="Run no-retrain M19 order-sensitivity benchmark slices.")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--bridge-path", type=Path, required=True)
    parser.add_argument("--eval-data-path", type=Path, default=Path(registry["dataset_defaults"]["benchmark"]))
    parser.add_argument("--eval-size", type=int, default=100)
    parser.add_argument("--slice-list", type=str, default="first,reversed,shuffled,stratified")
    parser.add_argument("--benchmark-regimes", type=str, default="")
    parser.add_argument("--order-sensitivity-threshold", type=float, default=0.05)
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
        default=Path(registry["output_roots"]["benchmark"]) / "order_sensitivity",
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

    source_rows = _load_jsonl(Path(args.eval_data_path))
    slice_specs = _build_slices(
        source_rows,
        slice_names=_parse_csv(args.slice_list),
        eval_size=int(args.eval_size),
        seed=int(args.seed),
    )
    slice_manifest_rows = []
    mode_rows = []
    for slice_name, rows in slice_specs.items():
        slice_dir = run_dir / slice_name
        slice_dir.mkdir(parents=True, exist_ok=True)
        slice_path = slice_dir / "eval_slice.jsonl"
        _write_jsonl(slice_path, rows)
        benchmark_path = slice_dir / "benchmark_report.json"
        _run_slice_if_needed(
            repo_root=repo_root,
            args=args,
            slice_path=slice_path,
            slice_size=len(rows),
            output_path=benchmark_path,
        )
        payload = _read_json(benchmark_path)
        mode_rows.append(_summarize_slice(slice_name, benchmark_path, payload, str(args.cell_id)))
        slice_manifest_rows.append(
            {
                "slice": slice_name,
                "row_count": len(rows),
                "eval_slice_path": str(slice_path).replace("\\", "/"),
                "first_prompt": str(rows[0].get("prompt", rows[0].get("text", ""))) if rows else None,
            }
        )

    interpretation = _interpret_order(mode_rows, float(args.order_sensitivity_threshold))
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "series": series_metadata("M", "M19.order_sensitivity", "scripts/m19/run_m19_order_sensitivity_suite.py"),
        "track": str(args.track),
        "config": {
            "base_model": str(args.base_model),
            "bridge_path": str(args.bridge_path).replace("\\", "/"),
            "eval_data_path": str(args.eval_data_path).replace("\\", "/"),
            "eval_size": int(args.eval_size),
            "slice_list": list(slice_specs.keys()),
            "benchmark_regimes": str(args.benchmark_regimes),
            "order_sensitivity_threshold": float(args.order_sensitivity_threshold),
            "cell_id": str(args.cell_id),
            "strict_accuracy_canonical": True,
            "phrase_accuracy_diagnostic_only": True,
        },
        "slice_manifest": slice_manifest_rows,
        "slice_rows": mode_rows,
        "interpretation": interpretation,
        "headline": {
            "accuracy_spread": interpretation.get("accuracy_spread"),
            "order_sensitive": interpretation.get("order_sensitive"),
            "best_slice": interpretation.get("best_slice"),
            "worst_slice": interpretation.get("worst_slice"),
            "first_accuracy": _slice_metric(mode_rows, "first", "strict_accuracy"),
            "reversed_accuracy": _slice_metric(mode_rows, "reversed", "strict_accuracy"),
            "shuffled_accuracy": _slice_metric(mode_rows, "shuffled", "strict_accuracy"),
            "stratified_accuracy": _slice_metric(mode_rows, "stratified", "strict_accuracy"),
        },
    }

    report_path = Path(args.output_path) if args.output_path else run_dir / "m19_order_sensitivity_report.json"
    assert_output_path_allowed("M", report_path)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote order-sensitivity report: {report_path}")


def _run_slice_if_needed(
    *,
    repo_root: Path,
    args: argparse.Namespace,
    slice_path: Path,
    slice_size: int,
    output_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "m19" / "run_m19_godtier_benchmark.py"),
        "--base-model",
        str(args.base_model),
        "--bridge-path",
        str(args.bridge_path),
        "--eval-data-path",
        str(slice_path),
        "--eval-size",
        str(int(slice_size)),
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
    run_if_needed(output_path, cmd, repo_root)


def _build_slices(
    rows: list[dict[str, Any]],
    *,
    slice_names: list[str],
    eval_size: int,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for name in slice_names:
        normalized = name.strip().lower()
        if normalized == "first":
            result[normalized] = rows[:eval_size]
        elif normalized == "reversed":
            result[normalized] = list(reversed(rows))[:eval_size]
        elif normalized == "shuffled":
            shuffled = list(rows)
            random.Random(seed).shuffle(shuffled)
            result[normalized] = shuffled[:eval_size]
        elif normalized == "stratified":
            result[normalized] = _stratified_sample(rows, eval_size)
        else:
            raise ValueError(f"unsupported slice mode: {name}")
    return result


def _stratified_sample(rows: list[dict[str, Any]], eval_size: int) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        buckets[_strat_key(row)].append(row)
    selected: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(selected) < eval_size and keys:
        next_keys: list[str] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < eval_size:
                selected.append(bucket.popleft())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def _strat_key(row: dict[str, Any]) -> str:
    for key in ("mode", "task_family", "answer"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"{key}:{value.lower()}"
    prompt = str(row.get("prompt", row.get("text", "")))
    return f"length:{len(prompt) // 80}"


def _summarize_slice(slice_name: str, benchmark_path: Path, payload: dict[str, Any], cell_id: str) -> dict[str, Any]:
    results = payload.get("results", {}) if isinstance(payload.get("results"), dict) else {}
    row = results.get(cell_id, {}) if isinstance(results.get(cell_id), dict) else {}
    prediction_summaries = payload.get("prediction_summaries", {}) if isinstance(payload.get("prediction_summaries"), dict) else {}
    prediction_summary = prediction_summaries.get(cell_id, {}) if isinstance(prediction_summaries.get(cell_id), dict) else {}
    strict_accuracy = _metric(row, "accuracy")
    avg_tokens = _metric(row, "avg_tokens")
    avg_runway_tokens = _metric(row, "avg_runway_tokens")
    return {
        "slice": slice_name,
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
        "unique_prediction_count": prediction_summary.get("unique_prediction_count"),
        "empty_prediction_rate": prediction_summary.get("empty_prediction_rate"),
        "top_predictions": prediction_summary.get("top_predictions", [])[:5]
        if isinstance(prediction_summary.get("top_predictions"), list)
        else [],
    }


def _interpret_order(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    scored = [
        (str(row.get("slice")), float(row["strict_accuracy"]))
        for row in rows
        if isinstance(row.get("strict_accuracy"), (int, float))
    ]
    if not scored:
        return {"order_sensitive": None, "accuracy_spread": None}
    best_slice, best_accuracy = max(scored, key=lambda item: item[1])
    worst_slice, worst_accuracy = min(scored, key=lambda item: item[1])
    spread = best_accuracy - worst_accuracy
    return {
        "order_sensitive": spread > float(threshold),
        "accuracy_spread": spread,
        "best_slice": best_slice,
        "best_accuracy": best_accuracy,
        "worst_slice": worst_slice,
        "worst_accuracy": worst_accuracy,
        "threshold": float(threshold),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _slice_metric(rows: list[dict[str, Any]], slice_name: str, key: str) -> float | None:
    for row in rows:
        if str(row.get("slice")) == slice_name:
            return _metric(row, key)
    return None


def _metric(row: dict[str, Any], key: str) -> float | None:
    try:
        value = row.get(key)
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    main()
