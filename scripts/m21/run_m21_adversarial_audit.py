from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m21.bridi import (  # noqa: E402
    M21DynamicBridiQFormer,
    evaluate_model,
    generate_dynamic_bridi_adversarial_examples,
    semantic_training_surface_names,
    tokenize,
)
from lojban_evolution.m21.family import M21_FAMILY_VERSION, M21_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, series_metadata, validate_series_outputs  # noqa: E402

SEMANTIC_ISOLATION_CELLS = ("H", "I", "J", "K", "L", "M", "N", "O")
SEMANTIC_ISOLATION_EFFECTS = {
    "lexical_shift": ("J", "H"),
    "role_binding": ("K", "H"),
    "combined": ("I", "H"),
    "fraction": ("L", "I"),
    "role_curriculum": ("M", "K"),
    "role_swap": ("N", "H"),
    "role_curriculum_fraction": ("O", "M"),
}
SEMANTIC_ISOLATION_METRICS = {
    "strict_accuracy": "adversarial_strict_accuracy",
    "worst_surface_accuracy": "adversarial_worst_surface_accuracy",
    "judri_causal_delta": "adversarial_judri_causal_delta",
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe(value: str | None) -> str:
    raw = (value or f"m21_adversarial_audit_{_timestamp()}").strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw) or f"m21_adversarial_audit_{_timestamp()}"


def _read(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _parse_cell_list(value: str) -> set[str]:
    return {item.strip().upper() for item in str(value).split(",") if item.strip()}


def _checkpoint_rows(payload: dict[str, Any], selected_cells: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cells = payload.get("cells", {}) if isinstance(payload, dict) else {}
    if isinstance(cells, dict):
        for cell_key, cell in cells.items():
            if selected_cells and str(cell_key).upper() not in selected_cells:
                continue
            for report in cell.get("seed_reports", []) if isinstance(cell, dict) else []:
                checkpoint_path = report.get("checkpoint_path") if isinstance(report, dict) else None
                if checkpoint_path:
                    rows.append(
                        {
                            "cell_key": str(cell_key),
                            "cell_id": cell.get("cell_id", ""),
                            "seed": report.get("seed"),
                            "checkpoint_path": str(checkpoint_path),
                            "source_run_dir": report.get("run_dir", ""),
                        }
                    )
    if not rows and payload.get("checkpoint_path"):
        rows.append(
            {
                "cell_key": str(payload.get("cell_key", "")),
                "cell_id": str(payload.get("cell_id", "")),
                "seed": payload.get("config", {}).get("seed"),
                "checkpoint_path": str(payload["checkpoint_path"]),
                "source_run_dir": payload.get("run_dir", ""),
            }
        )
    return rows


def _load_checkpoint_model(checkpoint_path: Path, device: str) -> tuple[M21DynamicBridiQFormer, dict[str, int], dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = dict(checkpoint.get("config", {}))
    vocab = dict(checkpoint.get("vocab", {}))
    model = M21DynamicBridiQFormer(
        vocab_size=len(vocab),
        embedding_dim=int(config.get("embedding_dim", 64)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        max_frames=int(config.get("max_frames", 6)),
        max_places=int(config.get("max_places", 5)),
        max_entities=int(config.get("max_entities", 8)),
        geometry_mode=str(config.get("geometry_mode", "euclidean")),
        poincare_curvature=float(config.get("poincare_curvature", 1.0)),
        poincare_max_norm=float(config.get("poincare_max_norm", 0.99)),
        riemannian_gradient_scale=bool(config.get("riemannian_gradient_scale", True)),
        judri_bridge_gate=bool(config.get("judri_bridge_gate", False)),
        judri_bridge_gate_temperature=float(config.get("judri_bridge_gate_temperature", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, vocab, config


def _oov_metrics(examples: list[Any], vocab: dict[str, int]) -> dict[str, Any]:
    total = 0
    unknown = 0
    by_surface: dict[str, list[float]] = {}
    for row in examples:
        tokens = tokenize(row.prompt)
        row_total = max(1, len(tokens))
        row_unknown = sum(1 for token in tokens if token not in vocab)
        total += len(tokens)
        unknown += row_unknown
        by_surface.setdefault(row.surface, []).append(row_unknown / row_total)
    return {
        "adversarial_oov_token_rate": float(unknown / max(1, total)),
        "adversarial_surface_oov_rate": {surface: float(mean(values)) for surface, values in sorted(by_surface.items())},
    }


def _surface_accuracy(metrics: dict[str, Any]) -> dict[str, float]:
    surfaces = metrics.get("surface_metrics", {})
    if not isinstance(surfaces, dict):
        return {}
    return {str(key): float(value.get("strict_accuracy", 0.0)) for key, value in surfaces.items() if isinstance(value, dict)}


def _with_adversarial_prefix(metrics: dict[str, Any], oov: dict[str, Any]) -> dict[str, Any]:
    surface_acc = _surface_accuracy(metrics)
    out = {
        "strict_accuracy": float(metrics.get("strict_accuracy", 0.0)),
        "adversarial_strict_accuracy": float(metrics.get("strict_accuracy", 0.0)),
        "adversarial_bridi_trace_exact_accuracy": float(metrics.get("bridi_trace_exact_accuracy", 0.0)),
        "adversarial_gismu_accuracy": float(metrics.get("gismu_accuracy", 0.0)),
        "adversarial_cmavo_accuracy": float(metrics.get("cmavo_accuracy", 0.0)),
        "adversarial_judri_binding_accuracy": float(metrics.get("judri_binding_accuracy", 0.0)),
        "adversarial_no_cmavo_accuracy": float(metrics.get("no_cmavo_accuracy", 0.0)),
        "adversarial_no_judri_accuracy": float(metrics.get("no_judri_accuracy", 0.0)),
        "adversarial_judri_causal_delta": float(metrics.get("judri_causal_delta", 0.0)),
        "adversarial_cmavo_causal_delta": float(metrics.get("cmavo_causal_delta", 0.0)),
        "adversarial_worst_surface_accuracy": min(surface_acc.values()) if surface_acc else float(metrics.get("strict_accuracy", 0.0)),
        "adversarial_oov_token_rate": float(oov.get("adversarial_oov_token_rate", 0.0)),
        "judri_bridge_gate_enabled": float(metrics.get("judri_bridge_gate_enabled", 0.0)),
        "judri_bridge_gate_active_mean": float(metrics.get("judri_bridge_gate_active_mean", 0.0)),
        "judri_bridge_gate_silenced_predicate_energy_mean": float(metrics.get("judri_bridge_gate_silenced_predicate_energy_mean", 0.0)),
        "surface_metrics": metrics.get("surface_metrics", {}),
        "adversarial_surface_oov_rate": oov.get("adversarial_surface_oov_rate", {}),
    }
    return out


def _summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    def collect(key: str) -> list[float]:
        return [float(row["metrics"].get(key, 0.0) or 0.0) for row in rows]

    strict = collect("adversarial_strict_accuracy")
    train_fractions = [float(row.get("config", {}).get("adversarial_train_fraction", 0.0) or 0.0) for row in rows]
    aggregate = {
        "mean_adversarial_strict_accuracy": mean(strict) if strict else 0.0,
        "std_adversarial_strict_accuracy": pstdev(strict) if len(strict) > 1 else 0.0,
        "mean_adversarial_bridi_trace_exact_accuracy": mean(collect("adversarial_bridi_trace_exact_accuracy")) if rows else 0.0,
        "mean_adversarial_gismu_accuracy": mean(collect("adversarial_gismu_accuracy")) if rows else 0.0,
        "mean_adversarial_cmavo_accuracy": mean(collect("adversarial_cmavo_accuracy")) if rows else 0.0,
        "mean_adversarial_judri_binding_accuracy": mean(collect("adversarial_judri_binding_accuracy")) if rows else 0.0,
        "mean_adversarial_no_judri_accuracy": mean(collect("adversarial_no_judri_accuracy")) if rows else 0.0,
        "mean_adversarial_judri_causal_delta": mean(collect("adversarial_judri_causal_delta")) if rows else 0.0,
        "mean_adversarial_cmavo_causal_delta": mean(collect("adversarial_cmavo_causal_delta")) if rows else 0.0,
        "mean_adversarial_worst_surface_accuracy": mean(collect("adversarial_worst_surface_accuracy")) if rows else 0.0,
        "mean_adversarial_oov_token_rate": mean(collect("adversarial_oov_token_rate")) if rows else 0.0,
        "mean_adversarial_train_fraction": mean(train_fractions) if train_fractions else 0.0,
        "adversarial_training_exposure_rate": sum(1.0 for value in train_fractions if value > 0.0) / max(1, len(train_fractions)),
    }
    semantic_rows = [row for row in rows if _semantic_coverage_surface_count(row.get("config", {})) > 0]
    if semantic_rows:
        semantic_train_fractions = [
            float(row.get("config", {}).get("adversarial_train_fraction", 0.0) or 0.0) for row in semantic_rows
        ]

        def semantic_collect(key: str) -> list[float]:
            return [float(row["metrics"].get(key, 0.0) or 0.0) for row in semantic_rows]

        aggregate.update(
            {
                "semantic_coverage_strict_accuracy": mean(semantic_collect("adversarial_strict_accuracy")),
                "semantic_coverage_worst_surface_accuracy": mean(semantic_collect("adversarial_worst_surface_accuracy")),
                "semantic_coverage_judri_causal_delta": mean(semantic_collect("adversarial_judri_causal_delta")),
                "semantic_coverage_oov_token_rate": mean(semantic_collect("adversarial_oov_token_rate")),
                "semantic_coverage_training_exposure_rate": sum(1.0 for value in semantic_train_fractions if value > 0.0)
                / max(1, len(semantic_train_fractions)),
                "semantic_coverage_train_fraction": mean(semantic_train_fractions),
                "semantic_coverage_surface_count": float(
                    mean([_semantic_coverage_surface_count(row.get("config", {})) for row in semantic_rows])
                ),
            }
        )
    aggregate.update(_semantic_isolation_metrics(rows))
    return aggregate


def _semantic_isolation_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    cell_metrics: dict[str, dict[str, float]] = {}
    out: dict[str, float] = {}
    for cell in SEMANTIC_ISOLATION_CELLS:
        cell_rows = [row for row in rows if str(row.get("cell_key", "")).upper() == cell]
        if not cell_rows:
            continue
        cell_metrics[cell] = {}
        for short_name, metric_key in SEMANTIC_ISOLATION_METRICS.items():
            values = [float(row.get("metrics", {}).get(metric_key, 0.0) or 0.0) for row in cell_rows]
            value = mean(values) if values else 0.0
            cell_metrics[cell][short_name] = value
            out[f"semantic_isolation_{cell.lower()}_{short_name}"] = value
    out["semantic_isolation_cell_count"] = float(len(cell_metrics))
    if all(cell in cell_metrics for cell in SEMANTIC_ISOLATION_CELLS):
        for effect_name, (target_cell, baseline_cell) in SEMANTIC_ISOLATION_EFFECTS.items():
            for short_name in SEMANTIC_ISOLATION_METRICS:
                target = cell_metrics[target_cell][short_name]
                baseline = cell_metrics[baseline_cell][short_name]
                out[f"semantic_coverage_{effect_name}_effect_{short_name}_delta"] = target - baseline
    return out


def _semantic_coverage_surface_count(config: dict[str, Any]) -> int:
    surfaces = config.get("adversarial_train_surfaces", [])
    if isinstance(surfaces, str):
        selected = {item.strip() for item in surfaces.split(",") if item.strip()}
    elif isinstance(surfaces, (list, tuple, set)):
        selected = {str(item).strip() for item in surfaces if str(item).strip()}
    else:
        selected = set()
    return len(selected.intersection(set(semantic_training_surface_names())))


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    registry = M21_REGISTRY["M21"]
    output_root = Path(args.output_root or registry["output_roots"]["adversarial_audit"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe(args.run_id)
    validate_series_outputs("M", [output_root], [run_dir])
    run_dir.mkdir(parents=True, exist_ok=True)
    suite_payload = _read(args.suite_report)
    rows = _checkpoint_rows(suite_payload, _parse_cell_list(args.cell_list))
    if not rows:
        raise FileNotFoundError("No M21 checkpoints found in --suite-report for the requested cells.")
    seed_reports: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        checkpoint_path = Path(row["checkpoint_path"])
        if not checkpoint_path.is_absolute():
            checkpoint_path = REPO_ROOT / checkpoint_path
        model, vocab, config = _load_checkpoint_model(checkpoint_path, str(args.device))
        seed = int(row.get("seed") or 0)
        examples = generate_dynamic_bridi_adversarial_examples(
            int(args.eval_size),
            seed=int(args.audit_seed) + seed + index * 1009,
            surfaces=tuple(args.surfaces.split(",")),
        )
        metrics = evaluate_model(model, examples, vocab, batch_size=int(args.batch_size), device=str(args.device))
        audit_metrics = _with_adversarial_prefix(metrics, _oov_metrics(examples, vocab))
        audit_metrics["adversarial_train_fraction"] = float(config.get("adversarial_train_fraction", 0.0) or 0.0)
        seed_reports.append(
            {
                **row,
                "checkpoint_path": str(checkpoint_path),
                "config": config,
                "metrics": audit_metrics,
                "sample_eval_rows": [example.to_json() for example in examples[:5]],
            }
        )
    aggregate = _summarize(seed_reports)
    report_path = run_dir / registry["report_names"]["adversarial_audit"]
    validate_series_outputs("M", [registry["output_roots"]["adversarial_audit"], str(run_dir)], [report_path])
    payload = {
        "series": series_metadata("M", "M21.1.adversarial_audit", "scripts/m21/run_m21_adversarial_audit.py"),
        "track": "M21.1",
        "family_version": M21_FAMILY_VERSION,
        "registry": {
            "runner_script": registry["runner_scripts"]["adversarial_audit"],
            "output_root": registry["output_roots"]["adversarial_audit"],
        },
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "source_suite_report": str(args.suite_report or ""),
        "config": {
            "cell_list": sorted(_parse_cell_list(args.cell_list)),
            "eval_size": int(args.eval_size),
            "batch_size": int(args.batch_size),
            "audit_seed": int(args.audit_seed),
            "surfaces": [surface.strip() for surface in args.surfaces.split(",") if surface.strip()],
            "device": str(args.device),
        },
        "aggregate_metrics": aggregate,
        "seed_reports": seed_reports,
        "canonical_accuracy": "adversarial_strict_accuracy",
        "diagnostic_only": ["phrase_accuracy"],
        "notes": [
            "Eval-only audit loads existing M21 checkpoints and tests held-out prompts outside the training generator templates.",
            "OOV rate is reported because lexical novelty is a separate stressor from bridi structural generalization.",
        ],
    }
    report_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"M21 adversarial audit report written to {report_path}")
    print(
        "M21 adversarial audit metrics: "
        f"strict={aggregate['mean_adversarial_strict_accuracy']:.4f} "
        f"trace={aggregate['mean_adversarial_bridi_trace_exact_accuracy']:.4f} "
        f"judri_delta={aggregate['mean_adversarial_judri_causal_delta']:.4f} "
        f"oov={aggregate['mean_adversarial_oov_token_rate']:.4f}"
    )
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M21_REGISTRY["M21"]
    parser = argparse.ArgumentParser(description="Run M21 held-out adversarial prompt audit on saved dynamic bridi checkpoints.")
    parser.add_argument("--suite-report", type=Path, required=True)
    parser.add_argument("--cell-list", type=str, default="H,I,J,K,L,M,N,O")
    parser.add_argument("--eval-size", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--audit-seed", type=int, default=21017)
    parser.add_argument("--surfaces", type=str, default="heldout_paraphrase,role_distractor,clausal_permutation,oov_synonym")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["adversarial_audit"]))
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_audit(parse_args())
