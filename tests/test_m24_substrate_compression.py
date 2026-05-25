from __future__ import annotations

import inspect

import torch

from lojban_evolution.bridi_substrate import (
    assert_symbolic_trace_contract,
    pack_symbolic_trace_from_batch,
    pack_symbolic_trace_from_outputs,
    packed_trace_component_accuracy,
    shuffled_packed_trace_like,
)
from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m23.relevance import M23RelevanceDataset, generate_m23_relevance_examples, m23_collate
from lojban_evolution.m24.compression import PackedTraceAdvisor, train_m24_substrate_compression


def test_packed_symbolic_trace_helper_uses_integer_contract() -> None:
    examples = generate_m23_relevance_examples(12, seed=24, clean_fraction=0.0)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    batch = m23_collate([M23RelevanceDataset(examples, vocab)[i] for i in range(6)])
    oracle = pack_symbolic_trace_from_batch(batch)

    assert oracle.dtype == torch.long
    assert oracle.ndim == 3
    assert_symbolic_trace_contract(oracle)
    metrics = packed_trace_component_accuracy(oracle, oracle)
    assert metrics["bridi_trace_exact_accuracy"] == 1.0
    assert metrics["gismu_accuracy"] == 1.0
    assert metrics["cmavo_accuracy"] == 1.0
    assert metrics["judri_accuracy"] == 1.0
    shuffled = shuffled_packed_trace_like(oracle, seed=24)
    assert shuffled.dtype == torch.long
    assert shuffled.shape == oracle.shape
    assert_symbolic_trace_contract(shuffled)
    assert sorted(shuffled.flatten().tolist()) == sorted(oracle.flatten().tolist())


def test_trace_advisor_contract_has_no_continuous_primary_inputs() -> None:
    advisor = PackedTraceAdvisor(max_frames=6, hidden_dim=16)
    params = inspect.signature(advisor.forward).parameters

    assert list(params) == ["packed_trace"]
    assert advisor.primary_trace_input == "packed_symbolic_trace"
    for forbidden in ("frame_repr", "trace_state", "prompt_state"):
        assert forbidden not in params
        assert forbidden in advisor.disallowed_primary_inputs

    with torch.no_grad():
        try:
            advisor(torch.randn(2, 6, advisor.spec.width))
        except TypeError as exc:
            assert "integer packed symbols" in str(exc)
        else:  # pragma: no cover - failure path
            raise AssertionError("PackedTraceAdvisor accepted a continuous float trace.")


def test_m24_tiny_cpu_training_freezes_generator_and_reports_metrics() -> None:
    result = train_m24_substrate_compression(
        train_size=36,
        eval_size=18,
        generator_epochs=1,
        advisor_epochs=1,
        prompt_epochs=1,
        batch_size=18,
        seed=24,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        mdl_weight=0.1,
        device="cpu",
    )
    metrics = result["metrics"]

    assert result["config"]["mdl_weight"] == 0.1
    assert result["stage1_config"]["mdl_weight"] == 0.1
    assert result["stage1_metrics"]["trace_exact_surrogate_weight"] == result["config"]["trace_exact_surrogate_weight"]
    assert metrics["mdl_weight"] == 0.1
    assert metrics["generator_trainable_parameter_count_after_freeze"] == 0.0
    assert metrics["generator_parameter_max_delta_after_advisor"] == 0.0
    assert metrics["generator_parameters_unchanged_after_advisor"] == 1.0
    assert metrics["advisor_primary_trace_is_symbolic"] == 1.0
    assert metrics["continuous_trace_smuggling_detected"] == 0.0
    assert metrics["predicted_trace_gap_to_oracle_upper_bound"] == (
        metrics["oracle_trained_oracle_trace_accuracy"] - metrics["oracle_trained_predicted_trace_accuracy"]
    )
    assert metrics["m24_promotion_candidate"] in {0.0, 1.0}
    assert 0.0 <= metrics["m24_promotion_gate_pass_rate"] <= 1.0
    for key in (
        "strict_accuracy",
        "predicted_trace_accuracy",
        "oracle_trace_accuracy",
        "oracle_trained_oracle_trace_accuracy",
        "oracle_trained_predicted_trace_accuracy",
        "oracle_trained_shuffled_trace_accuracy",
        "oracle_trained_random_trace_accuracy",
        "oracle_trained_trace_delta",
        "predicted_trace_gap_to_oracle_upper_bound",
        "cross_advisor_oracle_gap",
        "m24_strict_delta_vs_prompt_only",
        "shuffled_trace_accuracy",
        "predicted_vs_shuffled_delta",
        "random_trace_accuracy",
        "zero_trace_accuracy",
        "prompt_only_accuracy",
        "trace_advisor_delta",
        "bridi_trace_exact_accuracy",
        "gismu_accuracy",
        "cmavo_accuracy",
        "judri_accuracy",
        "packed_symbol_compression_ratio",
        "packed_symbol_to_prompt_ratio",
        "packed_to_prompt_ratio",
        "prompt_to_packed_ratio",
        "token_reduction_ratio",
        "accuracy_per_packed_symbol",
        "substrate_claim_score",
        "m24_promotion_gate_pass_rate",
        "m24_promotion_candidate",
        "m24_gate_trace_beats_shuffled",
        "m24_gate_token_reduction_positive",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert metrics["predicted_vs_shuffled_delta"] == metrics["predicted_trace_accuracy"] - metrics["shuffled_trace_accuracy"]
    assert metrics["token_reduction_ratio"] == 1.0 - metrics["packed_symbol_to_prompt_ratio"]

    examples = result["eval_examples"][:4]
    vocab = result["vocab"]
    batch = m23_collate([M23RelevanceDataset(examples, vocab)[i] for i in range(len(examples))])
    outputs = result["generator"](batch["input_ids"])
    predicted = pack_symbolic_trace_from_outputs(outputs)
    assert predicted.dtype == torch.long
