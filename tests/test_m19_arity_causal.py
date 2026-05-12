from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_arity_causal_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "run_m19_arity_causal_suite.py"
    spec = importlib.util.spec_from_file_location("run_m19_arity_causal_suite", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_arity_causal = _load_arity_causal_module()
_interpret_arity = _arity_causal._interpret_arity
_mode_to_override = _arity_causal._mode_to_override
_summarize_mode = _arity_causal._summarize_mode


def test_arity_causal_modes_are_accepted() -> None:
    assert _mode_to_override("predicted") == ("predicted", 1)
    assert _mode_to_override("oracle") == ("oracle", 1)
    assert _mode_to_override("random") == ("random", 1)
    assert _mode_to_override("no_mask") == ("no_mask", 1)
    assert _mode_to_override("force_8") == ("force", 8)


def test_arity_interpretation_marks_router_bottleneck_only_when_oracle_helps() -> None:
    rows = [
        {"mode": "predicted", "strict_accuracy": 0.30},
        {"mode": "oracle", "strict_accuracy": 0.38},
    ]
    report = _interpret_arity(rows, threshold=0.05)

    assert report["arity_router_bottleneck"] is True
    assert abs(report["oracle_delta_vs_predicted"] - 0.08) < 1e-9


def test_arity_summary_keeps_strict_accuracy_canonical() -> None:
    payload = {
        "metrics": {
            "strict_accuracy": 0.25,
            "overall_phrase_accuracy": 0.90,
            "avg_tokens": 31.0,
            "arity_violation_rate": 0.5,
            "masked_pointer_zero_rate": 1.0,
            "typed_family_accuracy": 0.75,
        },
        "results": {
            "M19.3_8Q_128D_8S": {
                "accuracy": 0.1,
                "phrase_accuracy": 1.0,
            }
        },
    }

    row = _summarize_mode("predicted", Path("benchmark.json"), payload, "M19.3_8Q_128D_8S")

    assert row["strict_accuracy"] == 0.25
    assert row["phrase_accuracy"] == 0.90
    assert row["strict_accuracy"] != row["phrase_accuracy"]
