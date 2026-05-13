from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from lojban_evolution.m19 import artifact_contract


def _load_bridge_channel_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "run_m19_bridge_channel_suite.py"
    spec = importlib.util.spec_from_file_location("run_m19_bridge_channel_suite", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_bridge_channel = _load_bridge_channel_module()
_bridge_channel_mode = _bridge_channel._bridge_channel_mode
_interpret_channels = _bridge_channel._interpret_channels
_regime_for_mode = _bridge_channel._regime_for_mode
_run_mode_if_needed = _bridge_channel._run_mode_if_needed
_summarize_mode = _bridge_channel._summarize_mode


def _bridge_args(**overrides):
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


def test_bridge_channel_modes_route_to_expected_benchmark_regimes() -> None:
    assert _bridge_channel_mode("full") == "full"
    assert _bridge_channel_mode("gismu_only") == "gismu_only"
    assert _bridge_channel_mode("drop_judri") == "drop_judri"
    assert _bridge_channel_mode("swap_judri") == "swap_judri"
    assert _bridge_channel_mode("reverse_judri") == "reverse_judri"
    assert _bridge_channel_mode("rotate_judri") == "rotate_judri"
    assert _bridge_channel_mode("none") == "full"
    assert _regime_for_mode("none", "M19.3_8Q_128D_8S") == "SCRATCHPAD-ONLY"
    assert _regime_for_mode("random", "M19.3_8Q_128D_8S") == "RANDOM-SHAPE"
    assert _regime_for_mode("no_judri", "M19.3_8Q_128D_8S") == "M19.3_8Q_128D_8S"


def test_bridge_channel_interpretation_flags_predicate_only_shortcut() -> None:
    rows = [
        {"mode": "full", "strict_accuracy": 0.60},
        {"mode": "scratchpad_only", "strict_accuracy": 0.20},
        {"mode": "random_shape", "strict_accuracy": 0.10},
        {"mode": "no_gismu", "strict_accuracy": 0.35},
        {"mode": "gismu_only", "strict_accuracy": 0.58},
        {"mode": "no_judri", "strict_accuracy": 0.57},
        {"mode": "judri_only", "strict_accuracy": 0.25},
    ]

    report = _interpret_channels(rows, threshold=0.05)

    assert report["bridge_carries_answer_signal"] is True
    assert report["gismu_channel_causal"] is True
    assert report["judri_channel_causal"] is False
    assert report["predicate_only_shortcut_warning"] is True
    assert report["pointer_only_shortcut_warning"] is False


def test_bridge_channel_summary_keeps_strict_accuracy_canonical() -> None:
    payload = {
        "results": {
            "M19.3_8Q_128D_8S": {
                "accuracy": 0.25,
                "phrase_accuracy": 0.90,
                "avg_tokens": 31.0,
                "avg_runway_tokens": 39.0,
                "bridge_channel_retained_slot_fraction": 0.25,
            }
        },
        "prediction_summaries": {
            "M19.3_8Q_128D_8S": {
                "top_predictions": [{"prediction": "no", "count": 3, "rate": 0.6}],
            }
        },
    }

    row = _summarize_mode("gismu_only", "M19.3_8Q_128D_8S", Path("benchmark.json"), payload)

    assert row["strict_accuracy"] == 0.25
    assert row["phrase_accuracy"] == 0.90
    assert row["accuracy_per_runway_token"] == 0.25 / 39.0
    assert row["strict_accuracy"] != row["phrase_accuracy"]
    assert row["bridge_channel_retained_slot_fraction"] == 0.25
    assert row["top_predictions"][0]["prediction"] == "no"


def test_bridge_channel_mode_report_uses_command_contract(tmp_path: Path, monkeypatch) -> None:
    report_path = tmp_path / "benchmark_report.json"
    report_path.write_text('{"results": {}}', encoding="utf-8")
    calls: list[tuple[list[str], str, bool]] = []

    def fake_run(cmd: list[str], cwd: str, check: bool) -> None:
        calls.append((cmd, cwd, check))
        report_path.write_text('{"results": {"M19.3_8Q_128D_8S": {"accuracy": 0.1}}}', encoding="utf-8")

    monkeypatch.setattr(artifact_contract.subprocess, "run", fake_run)
    args = _bridge_args()

    _run_mode_if_needed(repo_root=tmp_path, args=args, mode_label="full", output_path=report_path)

    assert len(calls) == 1
    assert calls[0][1] == str(tmp_path)
    assert calls[0][2] is True

    _run_mode_if_needed(repo_root=tmp_path, args=args, mode_label="full", output_path=report_path)

    assert len(calls) == 1

    args.eval_size = 12
    _run_mode_if_needed(repo_root=tmp_path, args=args, mode_label="full", output_path=report_path)

    assert len(calls) == 2
