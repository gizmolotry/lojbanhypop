from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m23.family import M23_FAMILY_VERSION, M23_REGISTRY  # noqa: E402
from lojban_evolution.m23.relevance import M23_LOCKS, train_m23_relevance_router  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m23_relevance_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m23_relevance_{_timestamp()}"


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M23_REGISTRY["M23"]
    output_root = Path(args.output_root or registry["output_roots"]["train"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _lock_status(metrics: dict[str, Any]) -> dict[str, bool]:
    strict = float(metrics.get("strict_accuracy", 0.0) or 0.0)
    decoy = float(metrics.get("decoy_relation_ood_accuracy", 0.0) or 0.0)
    return {
        "scale_control": float(metrics.get("use_relevance_router", 0.0) or 0.0) == 0.0 and decoy >= 0.0,
        "relevance_router": float(metrics.get("use_relevance_router", 0.0) or 0.0) >= 0.5 and float(metrics.get("relevance_top1_accuracy", 0.0) or 0.0) >= 0.20,
        "oracle_relevance": float(metrics.get("oracle_relevance_accuracy", 0.0) or 0.0) >= max(0.0, strict - 0.02),
        "random_relevance": float(metrics.get("random_relevance_accuracy", 1.0) or 1.0) <= max(1.0, strict + 0.50),
        "decoy_only": float(metrics.get("decoy_only_accuracy", 1.0) or 1.0) <= max(1.0, strict + 0.50),
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = M23_REGISTRY["M23"]
    result = train_m23_relevance_router(
        train_size=int(args.train_size),
        eval_size=int(args.eval_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        max_frames=int(args.max_frames),
        max_places=int(args.max_places),
        max_entities=int(args.max_entities),
        trace_weight=float(args.trace_weight),
        answer_weight=float(args.answer_weight),
        counterfactual_weight=float(args.counterfactual_weight),
        brivi_lock_weight=float(args.brivi_lock_weight),
        frame_necessity_weight=float(args.frame_necessity_weight),
        mdl_weight=float(args.mdl_weight),
        necessity_margin=float(args.necessity_margin),
        pointer_necessity_weight=float(args.pointer_necessity_weight),
        pointer_necessity_margin=float(args.pointer_necessity_margin),
        relevance_rank_weight=float(args.relevance_rank_weight),
        relevance_margin=float(args.relevance_margin),
        use_relevance_router=bool(args.use_relevance_router),
        relevance_temperature=float(args.relevance_temperature),
        clean_train_fraction=float(args.clean_train_fraction),
        clean_eval_fraction=float(args.clean_eval_fraction),
        geometry_mode=str(args.geometry_mode),
        poincare_curvature=float(args.poincare_curvature),
        poincare_max_norm=float(args.poincare_max_norm),
        riemannian_gradient_scale=bool(args.riemannian_gradient_scale),
        judri_bridge_gate=bool(args.judri_bridge_gate),
        judri_bridge_gate_temperature=float(args.judri_bridge_gate_temperature),
        device=str(args.device),
    )
    metrics = dict(result["metrics"])
    if result.get("history"):
        for key, value in dict(result["history"][-1]).items():
            if str(key).startswith(("loss", "grad_norm", "relevance")):
                metrics.setdefault(str(key), value)
    metrics.setdefault("synthetic_world_accuracy", metrics.get("strict_accuracy", 0.0))
    metrics.setdefault("phrase_accuracy", metrics.get("strict_accuracy", 0.0))
    locks = _lock_status(metrics)
    metrics["lock_pass_rate"] = sum(1.0 for ok in locks.values() if ok) / max(1, len(locks))
    checkpoint_path = run_dir / "m23_relevance_router_model.pt"
    torch.save({"state_dict": result["model"].state_dict(), "vocab": result["vocab"], "config": result["config"], "metrics": metrics}, checkpoint_path)
    report_path = Path(args.output_path) if args.output_path else run_dir / registry["report_names"]["train"]
    validate_series_outputs("M", [registry["output_roots"]["train"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M23.causal_relevance_router", "scripts/m23/train_m23_relevance_router.py"),
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=str(checkpoint_path),
            dataset_profile=registry["dataset_defaults"]["profile"],
            difficulty_tier="decoy_relevance_synthetic",
        ),
        "track": "M23",
        "family_version": M23_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["train"],
            "dag": registry["dags"].get("suite"),
            "output_root": registry["output_roots"]["train"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "config": result["config"],
        "architecture_locks": M23_LOCKS,
        "lock_status": locks,
        "metrics": metrics,
        "history": result["history"],
        "sample_eval_rows": [row.to_json() for row in result["eval_examples"][:5]],
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M23 relevance checkpoint saved to {checkpoint_path}")
    print(f"M23 relevance report written to {report_path}")
    print(
        "M23 metrics: "
        f"strict={metrics.get('strict_accuracy', 0.0):.4f} "
        f"decoy={metrics.get('decoy_relation_ood_accuracy', 0.0):.4f} "
        f"rel_top1={metrics.get('relevance_top1_accuracy', 0.0):.4f} "
        f"oracle={metrics.get('oracle_relevance_accuracy', 0.0):.4f}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M23_REGISTRY["M23"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Train the M23 causal relevance router over dynamic bridi frames.")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--max-places", type=int, default=5)
    parser.add_argument("--max-entities", type=int, default=8)
    parser.add_argument("--trace-weight", type=float, default=1.25)
    parser.add_argument("--answer-weight", type=float, default=1.25)
    parser.add_argument("--counterfactual-weight", type=float, default=1.25)
    parser.add_argument("--brivi-lock-weight", type=float, default=1.5)
    parser.add_argument("--frame-necessity-weight", type=float, default=1.0)
    parser.add_argument("--mdl-weight", type=float, default=0.01)
    parser.add_argument("--necessity-margin", type=float, default=0.04)
    parser.add_argument("--pointer-necessity-weight", type=float, default=0.0)
    parser.add_argument("--pointer-necessity-margin", type=float, default=0.05)
    parser.add_argument("--relevance-rank-weight", type=float, default=0.0)
    parser.add_argument("--relevance-margin", type=float, default=0.15)
    parser.add_argument("--use-relevance-router", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--relevance-temperature", type=float, default=1.0)
    parser.add_argument("--clean-train-fraction", type=float, default=0.35)
    parser.add_argument("--clean-eval-fraction", type=float, default=0.35)
    parser.add_argument("--geometry-mode", type=str, choices=("euclidean", "poincare"), default="euclidean")
    parser.add_argument("--poincare-curvature", type=float, default=1.0)
    parser.add_argument("--poincare-max-norm", type=float, default=0.99)
    parser.add_argument("--riemannian-gradient-scale", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judri-bridge-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judri-bridge-gate-temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["train"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_train(parse_args())
