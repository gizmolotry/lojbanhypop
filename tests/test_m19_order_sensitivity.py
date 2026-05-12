from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_order_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "m19" / "run_m19_order_sensitivity_suite.py"
    spec = importlib.util.spec_from_file_location("run_m19_order_sensitivity_suite", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_order = _load_order_module()


def test_order_slices_cover_first_reversed_shuffled_and_stratified() -> None:
    rows = [
        {"prompt": f"p{idx}", "answer": "A" if idx % 2 == 0 else "B", "mode": "m1" if idx < 4 else "m2"}
        for idx in range(8)
    ]

    slices = _order._build_slices(
        rows,
        slice_names=["first", "reversed", "shuffled", "stratified"],
        eval_size=4,
        seed=3,
    )

    assert [row["prompt"] for row in slices["first"]] == ["p0", "p1", "p2", "p3"]
    assert [row["prompt"] for row in slices["reversed"]] == ["p7", "p6", "p5", "p4"]
    assert len(slices["shuffled"]) == 4
    assert [row["prompt"] for row in slices["shuffled"]] != ["p0", "p1", "p2", "p3"]
    assert {row["mode"] for row in slices["stratified"]} == {"m1", "m2"}


def test_order_summary_keeps_strict_accuracy_canonical() -> None:
    payload = {
        "results": {
            "M19.3_8Q_128D_8S": {
                "accuracy": 0.25,
                "phrase_accuracy": 0.90,
                "avg_tokens": 20.0,
                "avg_runway_tokens": 28.0,
            }
        },
        "prediction_summaries": {
            "M19.3_8Q_128D_8S": {
                "unique_prediction_count": 3,
                "top_predictions": [{"prediction": "yes", "count": 2}],
            }
        },
    }

    row = _order._summarize_slice("first", Path("benchmark.json"), payload, "M19.3_8Q_128D_8S")

    assert row["strict_accuracy"] == 0.25
    assert row["phrase_accuracy"] == 0.90
    assert row["accuracy_per_token"] == 0.0125
    assert row["accuracy_per_runway_token"] == 0.25 / 28.0
    assert row["strict_accuracy"] != row["phrase_accuracy"]
    assert row["top_predictions"][0]["prediction"] == "yes"


def test_order_interpretation_flags_accuracy_spread() -> None:
    rows = [
        {"slice": "first", "strict_accuracy": 0.60},
        {"slice": "reversed", "strict_accuracy": 0.52},
        {"slice": "shuffled", "strict_accuracy": 0.58},
    ]

    interpretation = _order._interpret_order(rows, threshold=0.05)

    assert interpretation["order_sensitive"] is True
    assert interpretation["accuracy_spread"] == 0.07999999999999996
    assert interpretation["best_slice"] == "first"
    assert interpretation["worst_slice"] == "reversed"


def test_order_slice_runner_uses_command_contract(tmp_path: Path, monkeypatch) -> None:
    calls: list[tuple[list[str], str, bool]] = []
    output_path = tmp_path / "benchmark_report.json"
    slice_path = tmp_path / "slice.jsonl"
    slice_path.write_text(json.dumps({"prompt": "p", "answer": "a"}) + "\n", encoding="utf-8")

    def fake_run(cmd: list[str], cwd: str, check: bool) -> None:
        calls.append((cmd, cwd, check))
        output_path.write_text('{"results": {}}', encoding="utf-8")

    monkeypatch.setattr(_order.run_if_needed.__globals__["subprocess"], "run", fake_run)
    args = _order.argparse.Namespace(
        base_model="base",
        bridge_path=Path("bridge.pt"),
        num_queries=8,
        bottleneck_dim=128,
        scratchpad_length=8,
        max_latent_steps=64,
        hidden_size=768,
        tap_layer=12,
        random_scale=0.05,
        typed_slot_layout="",
        arity_router_mode="soft",
        gumbel_temp_end=0.35,
        geometry_mode="euclidean",
        poincare_curvature=1.0,
        seed=29,
        track="M19.31",
        cell_id="M19.3_8Q_128D_8S",
        benchmark_regimes="",
        gumbel_hard=False,
    )

    _order._run_slice_if_needed(
        repo_root=tmp_path,
        args=args,
        slice_path=slice_path,
        slice_size=1,
        output_path=output_path,
    )
    _order._run_slice_if_needed(
        repo_root=tmp_path,
        args=args,
        slice_path=slice_path,
        slice_size=1,
        output_path=output_path,
    )

    assert len(calls) == 1
