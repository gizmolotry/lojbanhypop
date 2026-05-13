from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from lojban_evolution.m19 import artifact_contract


def _load_pointer_counterfactual_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "run_m19_pointer_counterfactual_suite.py"
    spec = importlib.util.spec_from_file_location("run_m19_pointer_counterfactual_suite", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pointer_counterfactual = _load_pointer_counterfactual_module()
_bridge_channel_mode = _pointer_counterfactual._bridge_channel_mode
_interpret_pointer_counterfactuals = _pointer_counterfactual._interpret_pointer_counterfactuals
_run_mode_if_needed = _pointer_counterfactual._run_mode_if_needed
_summarize_mode = _pointer_counterfactual._summarize_mode


def _pointer_args(**overrides):
    values = {
        "base_model": "base-model",
        "bridge_path": Path("bridge.pt"),
        "eval_data_path": Path("eval.jsonl"),
        "eval_size": 11,
        "num_queries": 8,
        "bottleneck_dim": 128,
        "scratchpad_length": 8,
        "max_latent_steps": 64,
        "hidden_size": 768,
        "tap_layer": 12,
        "random_scale": 0.05,
        "typed_slot_layout": "",
        "arity_router_mode": "soft",
        "arity_override_mode": "predicted",
        "force_arity": 1,
        "gumbel_hard": False,
        "gumbel_temp_end": 0.35,
        "geometry_mode": "euclidean",
        "poincare_curvature": 1.0,
        "seed": 29,
        "track": "M19.31",
        "cell_id": "M19.3_8Q_128D_8S",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_pointer_counterfactual_modes_route_to_bridge_modes() -> None:
    assert _bridge_channel_mode("full") == "full"
    assert _bridge_channel_mode("drop_judri") == "drop_judri"
    assert _bridge_channel_mode("drop_pointer") == "drop_judri"
    assert _bridge_channel_mode("swap_judri") == "swap_judri"
    assert _bridge_channel_mode("reverse_judri") == "reverse_judri"
    assert _bridge_channel_mode("shuffle_judri") == "reverse_judri"
    assert _bridge_channel_mode("rotate_judri") == "rotate_judri"


def test_pointer_counterfactual_interpretation_flags_shortcut_when_corruption_retains_full() -> None:
    rows = [
        {"mode": "full", "strict_accuracy": 0.60},
        {"mode": "drop_judri", "strict_accuracy": 0.30},
        {"mode": "swap_judri", "strict_accuracy": 0.57},
        {"mode": "reverse_judri", "strict_accuracy": 0.45},
        {"mode": "rotate_judri", "strict_accuracy": 0.40},
    ]

    report = _interpret_pointer_counterfactuals(rows, threshold=0.05)

    assert report["best_corrupt_mode"] == "swap_judri"
    assert report["pointer_binding_causal"] is False
    assert report["pointer_counterfactual_shortcut_warning"] is True
    assert abs(report["min_corrupt_drop"] - 0.03) < 1e-9


def test_pointer_counterfactual_interpretation_marks_binding_causal_when_all_corruptions_drop() -> None:
    rows = [
        {"mode": "full", "strict_accuracy": 0.60},
        {"mode": "drop_judri", "strict_accuracy": 0.40},
        {"mode": "swap_judri", "strict_accuracy": 0.44},
        {"mode": "reverse_judri", "strict_accuracy": 0.42},
        {"mode": "rotate_judri", "strict_accuracy": 0.39},
    ]

    report = _interpret_pointer_counterfactuals(rows, threshold=0.05)

    assert report["best_corrupt_mode"] == "swap_judri"
    assert report["pointer_binding_causal"] is True
    assert report["pointer_counterfactual_shortcut_warning"] is False
    assert abs(report["max_corrupt_retention"] - (0.44 / 0.60)) < 1e-9


def test_pointer_counterfactual_summary_keeps_strict_accuracy_canonical() -> None:
    payload = {
        "results": {
            "M19.3_8Q_128D_8S": {
                "accuracy": 0.25,
                "phrase_accuracy": 0.90,
                "avg_tokens": 31.0,
                "avg_runway_tokens": 39.0,
                "bridge_channel_retained_slot_fraction": 1.0,
            }
        },
        "config": {"bridge_channel_mode": "swap_judri"},
        "prediction_summaries": {
            "M19.3_8Q_128D_8S": {
                "top_predictions": [{"prediction": "riley", "count": 3, "rate": 0.6}],
            }
        },
    }

    row = _summarize_mode("swap_judri", "M19.3_8Q_128D_8S", Path("benchmark.json"), payload)

    assert row["strict_accuracy"] == 0.25
    assert row["phrase_accuracy"] == 0.90
    assert row["strict_accuracy"] != row["phrase_accuracy"]
    assert row["observed_bridge_channel_mode"] == "swap_judri"
    assert row["accuracy_per_runway_token"] == 0.25 / 39.0
    assert row["top_predictions"][0]["prediction"] == "riley"


def test_pointer_counterfactual_mode_report_uses_command_contract(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_text('{"results": {}}', encoding="utf-8")
    calls: list[tuple[list[str], str, bool]] = []

    def fake_run(cmd: list[str], cwd: str, check: bool) -> None:
        calls.append((cmd, cwd, check))
        report_path.write_text('{"results": {"M19.3_8Q_128D_8S": {"accuracy": 0.1}}}', encoding="utf-8")

    monkeypatch.setattr(artifact_contract.subprocess, "run", fake_run)
    args = _pointer_args()

    _run_mode_if_needed(repo_root=tmp_path, args=args, mode_label="swap_judri", output_path=report_path)

    assert len(calls) == 1
    assert calls[0][1] == str(tmp_path)
    assert calls[0][2] is True
    assert "--bridge-channel-mode" in calls[0][0]
    assert "swap_judri" in calls[0][0]

    _run_mode_if_needed(repo_root=tmp_path, args=args, mode_label="swap_judri", output_path=report_path)

    assert len(calls) == 1

    args.eval_size = 12
    _run_mode_if_needed(repo_root=tmp_path, args=args, mode_label="swap_judri", output_path=report_path)

    assert len(calls) == 2
