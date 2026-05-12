from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_benchmark_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "run_m19_godtier_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_m19_godtier_benchmark", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_benchmark = _load_benchmark_module()


def test_prediction_record_keeps_generated_and_runway_token_counts_separate() -> None:
    row = _benchmark._prediction_record(
        {"prompt": "Who moved?", "answer": "Alice"},
        "M19.3_8Q_128D_8S",
        "Alice",
        token_count=2,
        extra={"runway_token_count": 10},
    )

    assert row["token_count"] == 2
    assert row["runway_token_count"] == 10
    assert row["correct"] is True


def test_efficiency_row_reports_runway_adjusted_efficiency() -> None:
    row = _benchmark._efficiency_row(
        "M19.3_8Q_128D_8S",
        {
            "M19.3_8Q_128D_8S": {
                "accuracy": 0.5,
                "phrase_accuracy": 0.5,
                "avg_tokens": 2.0,
                "avg_runway_tokens": 10.0,
            }
        },
        en_cot_accuracy=1.0,
        en_cot_tokens=20.0,
    )

    assert row["accuracy_per_token"] == 0.25
    assert row["accuracy_per_runway_token"] == 0.05
    assert row["token_ratio_vs_en_cot"] == 0.1
    assert row["runway_token_ratio_vs_en_cot"] == 0.5
