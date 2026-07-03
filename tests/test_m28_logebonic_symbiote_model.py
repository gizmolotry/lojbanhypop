from __future__ import annotations

import json
from pathlib import Path

import torch

from lojban_evolution.m25.emergent_bridi import generate_m25_emergent_bridi_examples
from lojban_evolution.m28.model import (
    LogebonicSymbioteModel,
    load_logebonic_symbiote_checkpoint,
    train_logebonic_symbiote_model,
)
from lojban_evolution.m28.baselines import ALL_BASELINES, build_m28_baseline_examples
from lojban_evolution.m28.suite import aggregate_m28_suite_metrics, run_m28_logebonic_symbiote_suite


def test_m28_model_exposes_inference_and_trace_schema() -> None:
    examples = generate_m25_emergent_bridi_examples(12, seed=280, max_symbols=8)
    model = LogebonicSymbioteModel.from_examples(
        examples,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
    )

    output = model.predict(examples[0].prompt)
    schema = model.trace_schema()

    assert output["answer_label"] in model.answer_labels
    assert output["trace"]
    assert output["trace"][0]["type"] in schema["type_vocabulary"].values()
    assert schema["format"] == "loose_bridi_triple_stream"


def test_m28_checkpoint_roundtrip_preserves_inference_contract(tmp_path: Path) -> None:
    examples = generate_m25_emergent_bridi_examples(12, seed=281, max_symbols=8)
    model = LogebonicSymbioteModel.from_examples(
        examples,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
    )
    checkpoint = model.save_checkpoint(tmp_path / "model.pt", metrics={"strict_accuracy": 0.25})

    reloaded = load_logebonic_symbiote_checkpoint(checkpoint)
    prediction = reloaded.predict(examples[0].prompt)

    assert reloaded.config == model.config
    assert reloaded.vocab == model.vocab
    assert prediction["trace"]
    assert prediction["logits_source"] in {"answer_logits", "relevance_answer_logits"}


def test_m28_tiny_training_writes_checkpoint_and_report(tmp_path: Path) -> None:
    result = train_logebonic_symbiote_model(
        train_size=18,
        eval_size=8,
        epochs=1,
        batch_size=4,
        seed=282,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        output_root=tmp_path,
        run_id="tiny_m28",
    )

    assert result.checkpoint_path.exists()
    assert result.report_path.exists()
    assert result.metrics["m28_actual_model_artifact"] == 1.0
    assert result.metrics["checkpoint_roundtrip_pass"] == 1.0
    assert result.metrics["model_inference_api_pass"] == 1.0
    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["track"] == "M28"
    assert payload["checkpoint_path"].endswith("final_logebonic_symbiote.pt")
    assert payload["trace_schema"]["format"] == "loose_bridi_triple_stream"
    state = torch.load(result.checkpoint_path, map_location="cpu")
    assert state["format"] == "logebonic_symbiote_checkpoint_v1"


def test_m28_baseline_bundle_reports_required_comparators(tmp_path: Path) -> None:
    examples = generate_m25_emergent_bridi_examples(4, seed=283, max_symbols=8)
    random_rows = build_m28_baseline_examples(examples, baseline="random_discrete_code", seed=1, max_symbols=8)

    assert random_rows[0].trace_token_count > 0

    result = train_logebonic_symbiote_model(
        train_size=12,
        eval_size=6,
        epochs=1,
        baseline_epochs=1,
        batch_size=3,
        seed=284,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        run_baselines=True,
        output_root=tmp_path,
        run_id="baseline_bundle",
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    baseline_results = payload["baseline_comparison"]["baseline_results"]
    assert result.metrics["m28_baseline_comparison_bundle_present"] == 1.0
    assert set(ALL_BASELINES) <= set(baseline_results)
    for name in ALL_BASELINES:
        assert "strict_accuracy" in baseline_results[name]
        assert "accuracy_per_token" in baseline_results[name]


def test_m28_training_can_resume_from_checkpoint(tmp_path: Path) -> None:
    first = train_logebonic_symbiote_model(
        train_size=12,
        eval_size=6,
        epochs=1,
        batch_size=3,
        seed=285,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        checkpoint_every_epochs=1,
        output_root=tmp_path,
        run_id="resume_first",
    )

    resumed = train_logebonic_symbiote_model(
        train_size=12,
        eval_size=6,
        epochs=2,
        batch_size=3,
        seed=285,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        resume_checkpoint=first.checkpoint_path,
        output_root=tmp_path,
        run_id="resume_second",
    )

    payload = json.loads(resumed.report_path.read_text(encoding="utf-8"))
    assert payload["training_config"]["start_epoch"] == 1
    assert payload["training_config"]["resume_checkpoint"] == str(first.checkpoint_path)
    assert len(payload["history"]) == 2
    assert payload["metrics"]["checkpoint_roundtrip_pass"] == 1.0


def test_m28_suite_aggregates_runs_and_embeds_direct_eval(tmp_path: Path) -> None:
    result = run_m28_logebonic_symbiote_suite(
        seed_list=[286, 287],
        train_size=10,
        eval_size=4,
        epochs=1,
        baseline_epochs=1,
        batch_size=2,
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        output_root=tmp_path,
        run_id="suite_tiny",
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    direct_manifest = Path(payload["direct_unified_eval"]["manifest_path"])

    assert result.report_path.exists()
    assert result.best_report_path.exists()
    assert result.best_checkpoint_path.exists()
    assert len(result.run_reports) == 2
    assert payload["track"] == "M28"
    assert payload["metrics"]["m28_suite_run_count"] == 2.0
    assert payload["metrics"]["m28_suite_best_report_available"] == 1.0
    assert payload["metrics"]["m28_suite_direct_eval_embedded"] == 1.0
    assert direct_manifest.exists()
    assert json.loads(direct_manifest.read_text(encoding="utf-8"))["family_key"] == "M28"


def test_m28_suite_aggregate_prefers_baseline_and_causality_visible() -> None:
    runs = [
        {
            "seed": 1,
            "metrics": {
                "strict_accuracy": 0.40,
                "m28_learned_vs_best_baseline_delta": -0.10,
                "m28_trace_causality_delta": 0.00,
                "m28_actual_model_artifact": 1.0,
                "checkpoint_roundtrip_pass": 1.0,
                "model_inference_api_pass": 1.0,
                "trace_schema_saved": 1.0,
                "m28_baseline_comparison_bundle_present": 1.0,
            },
        },
        {
            "seed": 2,
            "metrics": {
                "strict_accuracy": 0.41,
                "m28_learned_vs_best_baseline_delta": 0.08,
                "m28_trace_causality_delta": 0.12,
                "m28_actual_model_artifact": 1.0,
                "checkpoint_roundtrip_pass": 1.0,
                "model_inference_api_pass": 1.0,
                "trace_schema_saved": 1.0,
                "m28_baseline_comparison_bundle_present": 1.0,
            },
        },
    ]

    metrics = aggregate_m28_suite_metrics(runs, stable_accuracy_threshold=0.4)

    assert metrics["m28_suite_run_count"] == 2.0
    assert metrics["mean_strict_accuracy"] == 0.405
    assert metrics["m28_suite_stable_seed_rate"] == 1.0
    assert metrics["m28_suite_artifact_gate_pass_rate"] == 1.0
    assert metrics["best_m28_learned_vs_best_baseline_delta"] == 0.08
    assert metrics["best_m28_trace_causality_delta"] == 0.12
