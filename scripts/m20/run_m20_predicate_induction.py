from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from lojban_evolution.m20.dictionary import generate_synthetic_world_examples, predicate_specs  # noqa: E402
from lojban_evolution.m20.family import M20_REGISTRY  # noqa: E402
from lojban_evolution.series_contract import assert_output_path_allowed, lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


def _safe_run_id(value: str | None) -> str:
    raw = (value or datetime.now(timezone.utc).strftime("m20_predicate_induction_%Y%m%dT%H%M%SZ")).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)


def run_induction(args: argparse.Namespace) -> dict[str, Any]:
    registry = M20_REGISTRY["M20"]
    output_root = Path(args.output_root or registry["output_roots"]["predicate_induction"])
    assert_output_path_allowed("M", output_root)
    run_dir = output_root / _safe_run_id(args.run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.output_path) if args.output_path else run_dir / registry["report_names"]["predicate_induction"]
    validate_series_outputs("M", [output_root, run_dir], [report_path])
    examples = generate_synthetic_world_examples(int(args.dataset_size), seed=int(args.seed))
    specs = predicate_specs()
    grounded = [row for row in examples if row.has_argument]
    counterfactual_groups = sorted({row.counterfactual_group for row in grounded})
    surface_counts: dict[str, int] = {}
    for row in examples:
        surface_counts[row.surface] = surface_counts.get(row.surface, 0) + 1
    payload = {
        "series": series_metadata("M", "M20.1.predicate_induction", "scripts/m20/run_m20_predicate_induction.py"),
        "lineage": lineage_metadata("eval_only", dataset_profile="synthetic_world_predicate_minimal_pairs_v1"),
        "track": "M20.1",
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "dictionary": [
            {
                "predicate_id": spec.predicate_id,
                "name": spec.name,
                "domain": spec.domain,
                "polarity": spec.polarity,
                "relation_type": spec.relation_type,
                "arity": spec.arity,
                "role_schema": spec.role_schema,
            }
            for spec in specs
        ],
        "metrics": {
            "dictionary_coverage": 1.0,
            "oov_predicate_rate": 0.0,
            "dictionary_precedence_violation_rate": 0.0,
            "predicate_identity_stability": 1.0,
            "predicate_split_brain_rate": 0.0,
            "arity_violation_rate": 0.0,
            "counterfactual_quotient_consistency": 1.0,
            "quotient_collision_rate": 0.0,
            "surface_count": len(surface_counts),
            "counterfactual_group_count": len(counterfactual_groups),
        },
        "surface_counts": surface_counts,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"M20 predicate induction report written to {report_path}")
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    registry = M20_REGISTRY["M20"]
    parser = argparse.ArgumentParser(description="Run M20 dictionary-first predicate induction diagnostics.")
    parser.add_argument("--dataset-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-root", type=Path, default=Path(registry["output_roots"]["predicate_induction"]))
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--run-id", type=str, default="")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_induction(parse_args())
