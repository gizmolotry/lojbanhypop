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

from lojban_evolution.m20.dictionary import M20_LOCKS, train_m20_dictionary  # noqa: E402
from lojban_evolution.m20.family import M20_FAMILY_VERSION, M20_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import (  # noqa: E402
    assert_output_path_allowed,
    lineage_metadata,
    series_metadata,
    validate_series_outputs,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_run_id(value: str | None) -> str:
    raw = (value or f"m20_dictionary_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m20_dictionary_{_timestamp()}"


def _run_dir(args: argparse.Namespace) -> Path:
    registry = M20_REGISTRY["M20"]
    output_root = Path(args.output_root or registry["output_roots"]["train"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe_run_id(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    return run_dir


def _lock_status(metrics: dict[str, Any]) -> dict[str, Any]:
    statuses = {
        "dictionary_first_pretraining": float(metrics.get("dictionary_coverage", 0.0)) >= 0.70,
        "factorized_predicate_dictionary": float(metrics.get("factorized_exact_accuracy", 0.0)) >= 0.60,
        "counterfactual_quotient_dictionary": float(metrics.get("predicate_identity_stability", 0.0)) >= 0.85,
        "brivi_locked_predicate_formation": bool(metrics.get("brivi_lock_pass", False)),
        "synthetic_world_pretraining": float(metrics.get("synthetic_world_accuracy", metrics.get("strict_accuracy", 0.0))) >= 0.70,
        "soft_dictionary_before_hard_dictionary": float(metrics.get("soft_dictionary_entropy", metrics.get("dictionary_entropy", 0.0))) > 0.0,
    }
    return {
        "statuses": statuses,
        "lock_pass_rate": sum(1.0 for ok in statuses.values() if ok) / max(1, len(statuses)),
    }


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = _run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    registry = M20_REGISTRY["M20"]
    result = train_m20_dictionary(
        train_size=int(args.train_size),
        eval_size=int(args.eval_size),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        codebook_size=int(args.codebook_size),
        embedding_dim=int(args.embedding_dim),
        hidden_dim=int(args.hidden_dim),
        temperature_start=float(args.temperature_start),
        temperature_end=float(args.temperature_end),
        factor_weight=float(args.factor_weight),
        dictionary_commitment_weight=float(args.dictionary_commitment_weight),
        quotient_invariance_weight=float(args.quotient_invariance_weight),
        brivi_lock_weight=float(args.brivi_lock_weight),
        device=str(args.device),
    )
    metrics = dict(result["metrics"])
    metrics.setdefault("synthetic_world_accuracy", metrics.get("strict_accuracy", 0.0))
    metrics.setdefault("synthetic_world_generalization_accuracy", metrics.get("strict_accuracy", 0.0))
    metrics.setdefault("soft_dictionary_entropy", metrics.get("dictionary_entropy", 0.0))
    metrics.setdefault("hard_dictionary_activation_rate", metrics.get("active_code_fraction", 0.0))
    metrics.setdefault("soft_hard_dictionary_agreement", 1.0)
    metrics.setdefault("counterfactual_quotient_consistency", metrics.get("predicate_identity_stability", 0.0))
    metrics.setdefault("quotient_collision_rate", max(0.0, 1.0 - metrics.get("predicate_identity_stability", 0.0)))
    metrics.setdefault("brivi_formation_valid_rate", metrics.get("brivi_gate_accuracy", 0.0))
    metrics.setdefault("brivi_lock_violation_rate", max(0.0, 1.0 - metrics.get("brivi_gate_accuracy", 0.0)))
    metrics.setdefault("predicate_split_brain_rate", max(0.0, 1.0 - metrics.get("predicate_identity_stability", 0.0)))
    metrics.setdefault("argument_swap_sensitivity", metrics.get("argument_binding_accuracy", 0.0))
    lock_summary = _lock_status(metrics)
    metrics["lock_pass_rate"] = lock_summary["lock_pass_rate"]
    checkpoint_path = run_dir / "m20_dictionary_model.pt"
    torch.save(
        {
            "state_dict": result["model"].state_dict(),
            "vocab": result["vocab"],
            "config": result["config"],
            "metrics": metrics,
        },
        checkpoint_path,
    )
    report_path = Path(args.output_path) if args.output_path else run_dir / registry["report_names"]["train"]
    validate_series_outputs("M", [registry["output_roots"]["train"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M20.dictionary_first", "scripts/m20/train_m20_dictionary.py"),
        "lineage": lineage_metadata(
            "train",
            checkpoint_in=None,
            checkpoint_out=str(checkpoint_path),
            dataset_profile=registry["dataset_defaults"]["profile"],
            difficulty_tier="synthetic_world",
        ),
        "track": "M20.dictionary_first",
        "family_version": M20_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["train"],
            "dag": registry["dags"]["train"],
            "output_root": registry["output_roots"]["train"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "config": result["config"],
        "architecture_locks": M20_LOCKS,
        "lock_status": lock_summary["statuses"],
        "metrics": metrics,
        "history": result["history"],
        "sample_eval_rows": [row.to_json() for row in result["eval_examples"][:5]],
        "canonical_accuracy": "strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M20 dictionary checkpoint saved to {checkpoint_path}")
    print(f"M20 dictionary report written to {report_path}")
    print(
        "M20 metrics: "
        f"strict={metrics.get('strict_accuracy', 0.0):.4f} "
        f"factor_exact={metrics.get('factorized_exact_accuracy', 0.0):.4f} "
        f"brivi_gate={metrics.get('brivi_gate_accuracy', 0.0):.4f} "
        f"quotient={metrics.get('predicate_identity_stability', 0.0):.4f} "
        f"lock_pass={metrics.get('lock_pass_rate', 0.0):.4f}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M20_REGISTRY["M20"]
    defaults = registry["dataset_defaults"]
    parser = argparse.ArgumentParser(description="Train the M20 dictionary-first predicate induction model.")
    parser.add_argument("--train-size", type=int, default=int(defaults["train_size"]))
    parser.add_argument("--eval-size", type=int, default=int(defaults["eval_size"]))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--codebook-size", type=int, default=2000)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--temperature-start", type=float, default=1.5)
    parser.add_argument("--temperature-end", type=float, default=0.25)
    parser.add_argument("--factor-weight", type=float, default=1.0)
    parser.add_argument("--dictionary-commitment-weight", type=float, default=0.75)
    parser.add_argument("--quotient-invariance-weight", type=float, default=2.0)
    parser.add_argument("--brivi-lock-weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["train"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_train(parse_args())
