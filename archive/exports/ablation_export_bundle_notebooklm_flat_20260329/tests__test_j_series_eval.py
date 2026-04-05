from __future__ import annotations

import json
from pathlib import Path
import uuid

from lojban_evolution.j_series_eval import run_j5_adversarial_synthesis


def test_j5_adversarial_synthesis_outputs_contract() -> None:
    out_dir = Path("runs/j_series") / f"test_j5_eval_{uuid.uuid4().hex}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "j5.json"
    ds = out_dir / "j5.jsonl"
    payload = run_j5_adversarial_synthesis(
        output=out,
        dataset_output=ds,
        sample_count=32,
        seed=7,
        novelty_threshold=0.2,
    )
    assert out.exists()
    assert ds.exists()
    assert payload["summary"]["run_id"] == "J-5"
    assert payload["summary"]["series_id"] == "J"
    assert payload["summary"]["track"] == "invariance_data"
    metrics = payload["metrics"]
    assert "generator_accept_rate" in metrics
    assert "accept_rate_by_depth" in metrics
    assert "accepted_foil_pair_accuracy" in metrics
    assert "scope_by_depth" in metrics
    assert "foil_auc" in metrics
    assert 0.0 <= float(metrics["generator_accept_rate"]) <= 1.0
    assert 0.0 <= float(metrics["accepted_foil_pair_accuracy"]) <= 1.0
    assert 0.0 <= float(metrics["foil_auc"]) <= 1.0
    scope = metrics["accept_rate_by_depth"]
    assert set(scope.keys()) == {"1", "2", "3", "4"}
    assert metrics["scope_by_depth"] == scope
    assert metrics["foil_auc"] == metrics["accepted_foil_pair_accuracy"]

    with ds.open("r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    assert rows, "Expected at least one accepted adversarial row."
