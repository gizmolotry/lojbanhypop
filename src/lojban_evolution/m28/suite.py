from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from lojban_evolution.m28.family import M28_REGISTRY
from lojban_evolution.m28.model import LogebonicSymbioteTrainingResult, _safe_run_id, train_logebonic_symbiote_model


M28_SUITE_METRIC_KEYS = (
    "strict_accuracy",
    "phrase_accuracy",
    "m28_actual_model_artifact",
    "checkpoint_roundtrip_pass",
    "model_inference_api_pass",
    "trace_schema_saved",
    "m28_baseline_comparison_bundle_present",
    "m28_learned_logebonic_accuracy",
    "m28_best_non_logebonic_baseline_accuracy",
    "m28_learned_vs_best_baseline_delta",
    "m28_learned_vs_no_cot_delta",
    "m28_trace_causality_delta",
    "m27_full_organism_gate_pass_rate",
    "m27_promotion_candidate",
    "answer_loss_reaches_generator",
    "answer_loss_reaches_coconut_cell",
    "answer_loss_reaches_recurrent_bridi_feedback",
    "answer_loss_reaches_language_backbone",
    "answer_loss_reaches_bridge",
)


@dataclass(frozen=True)
class M28SuiteResult:
    report_path: Path
    metrics: dict[str, float]
    best_report_path: Path
    best_checkpoint_path: Path
    run_reports: tuple[Path, ...]


def parse_seed_list(seed_list: str | Sequence[int]) -> list[int]:
    if isinstance(seed_list, str):
        values = [part.strip() for part in seed_list.split(",") if part.strip()]
        if not values:
            raise ValueError("seed_list may not be empty")
        return [int(value) for value in values]
    seeds = [int(value) for value in seed_list]
    if not seeds:
        raise ValueError("seed_list may not be empty")
    return seeds


def aggregate_m28_suite_metrics(
    runs: Sequence[dict[str, Any]],
    *,
    stable_accuracy_threshold: float = 0.5,
) -> dict[str, float]:
    metrics: dict[str, float] = {"m28_suite_run_count": float(len(runs))}
    if not runs:
        return metrics
    for key in M28_SUITE_METRIC_KEYS:
        values = [_float_or_none(row.get("metrics", {}).get(key)) for row in runs]
        present = [float(value) for value in values if value is not None]
        if not present:
            continue
        metrics[f"mean_{key}"] = mean(present)
        metrics[f"min_{key}"] = min(present)
        metrics[f"max_{key}"] = max(present)
        metrics[f"std_{key}"] = pstdev(present) if len(present) > 1 else 0.0

    strict_values = [
        float(value)
        for value in (_float_or_none(row.get("metrics", {}).get("strict_accuracy")) for row in runs)
        if value is not None
    ]
    if strict_values:
        threshold = float(stable_accuracy_threshold)
        metrics["m28_suite_stable_seed_rate"] = sum(1.0 for value in strict_values if value >= threshold) / len(strict_values)
        metrics["m28_suite_worst_seed_accuracy"] = min(strict_values)

    gate_keys = (
        "m28_actual_model_artifact",
        "checkpoint_roundtrip_pass",
        "model_inference_api_pass",
        "trace_schema_saved",
        "m28_baseline_comparison_bundle_present",
    )
    gate_rows = []
    for row in runs:
        row_metrics = row.get("metrics", {})
        if not isinstance(row_metrics, dict):
            continue
        gate_rows.append(all(float(row_metrics.get(key, 0.0) or 0.0) >= 1.0 for key in gate_keys))
    if gate_rows:
        metrics["m28_suite_artifact_gate_pass_rate"] = sum(1.0 for value in gate_rows if value) / len(gate_rows)

    best = select_best_m28_run(runs)
    if best:
        best_metrics = best.get("metrics", {})
        if isinstance(best_metrics, dict):
            for key, value in best_metrics.items():
                numeric = _float_or_none(value)
                if numeric is not None:
                    metrics[f"best_{key}"] = float(numeric)
        metrics["m28_suite_best_seed"] = float(best.get("seed", -1))
        metrics["m28_suite_best_score"] = _m28_model_selection_score(best_metrics if isinstance(best_metrics, dict) else {})
    return metrics


def select_best_m28_run(runs: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    if not runs:
        return None
    return max(
        runs,
        key=lambda row: (
            _m28_model_selection_score(row.get("metrics", {}) if isinstance(row.get("metrics"), dict) else {}),
            -int(row.get("seed", 0) or 0),
        ),
    )


def run_m28_logebonic_symbiote_suite(
    *,
    seed_list: str | Sequence[int] = (23, 29),
    train_size: int = 6000,
    eval_size: int = 1500,
    epochs: int = 8,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    max_frames: int = 6,
    max_symbols: int = 32,
    embedding_dim: int = 64,
    hidden_dim: int = 128,
    advisor_hidden_dim: int = 64,
    symbol_budget: int = 0,
    enable_relevance_runtime: bool = True,
    relevance_rank_weight: float = 0.25,
    use_relevance_answer: bool = True,
    run_baselines: bool = True,
    baseline_epochs: int = 2,
    checkpoint_every_epochs: int = 0,
    use_amp: bool = False,
    device: str = "cpu",
    output_root: str | Path = M28_REGISTRY["M28"]["output_roots"]["suite"],
    run_id: str | None = None,
    stable_accuracy_threshold: float = 0.5,
) -> M28SuiteResult:
    seeds = parse_seed_list(seed_list)
    suite_dir = Path(output_root) / _safe_run_id(run_id or "m28_logebonic_model_suite")
    model_root = suite_dir / "model_runs"
    suite_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for seed in seeds:
        model_run_id = f"seed_{int(seed)}"
        result = train_logebonic_symbiote_model(
            train_size=int(train_size),
            eval_size=int(eval_size),
            epochs=int(epochs),
            batch_size=int(batch_size),
            learning_rate=float(learning_rate),
            seed=int(seed),
            max_frames=int(max_frames),
            max_symbols=int(max_symbols),
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            advisor_hidden_dim=int(advisor_hidden_dim),
            symbol_budget=int(symbol_budget),
            enable_relevance_runtime=bool(enable_relevance_runtime),
            relevance_rank_weight=float(relevance_rank_weight),
            use_relevance_answer=bool(use_relevance_answer),
            run_baselines=bool(run_baselines),
            baseline_epochs=int(baseline_epochs),
            checkpoint_every_epochs=int(checkpoint_every_epochs),
            use_amp=bool(use_amp),
            device=str(device),
            output_root=model_root,
            run_id=model_run_id,
        )
        runs.append(_run_record(seed=seed, result=result))

    aggregate = aggregate_m28_suite_metrics(runs, stable_accuracy_threshold=float(stable_accuracy_threshold))
    best = select_best_m28_run(runs)
    if not best:
        raise RuntimeError("M28 suite produced no runs")
    best_report_path = Path(str(best["report_path"]))
    best_checkpoint_path = Path(str(best["checkpoint_path"]))
    direct_manifest_path, direct_summary_path = _write_embedded_direct_eval(suite_dir, best_report_path)
    aggregate["m28_suite_direct_eval_embedded"] = 1.0 if direct_manifest_path.exists() else 0.0
    aggregate["m28_suite_best_report_available"] = 1.0 if best_report_path.exists() else 0.0
    aggregate["m28_suite_best_checkpoint_available"] = 1.0 if best_checkpoint_path.exists() else 0.0

    report = {
        "track": "M28",
        "suite": "m28_logebonic_symbiote_model_suite",
        "run_id": suite_dir.name,
        "config": {
            "seed_list": seeds,
            "train_size": int(train_size),
            "eval_size": int(eval_size),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "max_frames": int(max_frames),
            "max_symbols": int(max_symbols),
            "embedding_dim": int(embedding_dim),
            "hidden_dim": int(hidden_dim),
            "advisor_hidden_dim": int(advisor_hidden_dim),
            "symbol_budget": int(symbol_budget),
            "enable_relevance_runtime": bool(enable_relevance_runtime),
            "relevance_rank_weight": float(relevance_rank_weight),
            "use_relevance_answer": bool(use_relevance_answer),
            "run_baselines": bool(run_baselines),
            "baseline_epochs": int(baseline_epochs),
            "checkpoint_every_epochs": int(checkpoint_every_epochs),
            "use_amp": bool(use_amp),
            "device": str(device),
            "stable_accuracy_threshold": float(stable_accuracy_threshold),
        },
        "metrics": aggregate,
        "aggregate_metrics": aggregate,
        "runs": runs,
        "best_run": best,
        "best_report_path": str(best_report_path),
        "best_checkpoint_path": str(best_checkpoint_path),
        "direct_unified_eval": {
            "manifest_path": str(direct_manifest_path),
            "summary_path": str(direct_summary_path),
        },
    }
    report_path = suite_dir / "m28_logebonic_model_suite_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return M28SuiteResult(
        report_path=report_path,
        metrics={key: float(value) for key, value in aggregate.items() if isinstance(value, (int, float))},
        best_report_path=best_report_path,
        best_checkpoint_path=best_checkpoint_path,
        run_reports=tuple(Path(str(row["report_path"])) for row in runs),
    )


def _run_record(*, seed: int, result: LogebonicSymbioteTrainingResult) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "report_path": str(result.report_path),
        "checkpoint_path": str(result.checkpoint_path),
        "vocab_size": int(result.vocab_size),
        "config": asdict(result.config),
        "metrics": dict(result.metrics),
    }


def _write_embedded_direct_eval(suite_dir: Path, model_report_path: Path) -> tuple[Path, Path]:
    from lojban_evolution.direct_unified_eval import build_direct_unified_eval_manifest, render_direct_unified_eval_markdown

    manifest = build_direct_unified_eval_manifest(
        family_key="M28",
        track="M28",
        m28_model_report_path=model_report_path,
    )
    manifest["run_id"] = f"{suite_dir.name}_embedded_direct_eval"
    out_dir = suite_dir / "direct_unified_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "direct_unified_eval_manifest.json"
    summary_path = out_dir / "direct_unified_eval_summary.md"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(render_direct_unified_eval_markdown(manifest), encoding="utf-8")
    return manifest_path, summary_path


def _m28_model_selection_score(metrics: dict[str, Any]) -> float:
    strict = float(metrics.get("strict_accuracy", 0.0) or 0.0)
    baseline_delta = float(metrics.get("m28_learned_vs_best_baseline_delta", 0.0) or 0.0)
    trace_delta = float(metrics.get("m28_trace_causality_delta", 0.0) or 0.0)
    gate = min(
        float(metrics.get("m28_actual_model_artifact", 0.0) or 0.0),
        float(metrics.get("checkpoint_roundtrip_pass", 0.0) or 0.0),
        float(metrics.get("model_inference_api_pass", 0.0) or 0.0),
    )
    return strict + 0.25 * baseline_delta + 0.25 * trace_delta + 0.1 * gate


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None
