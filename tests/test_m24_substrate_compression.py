from __future__ import annotations

import inspect

import pytest
import torch

from lojban_evolution.bridi_substrate import (
    assert_symbolic_trace_contract,
    budget_packed_trace_symbols,
    pack_symbolic_trace_from_batch,
    pack_symbolic_trace_from_outputs,
    packed_trace_component_accuracy,
    packed_trace_spec,
    packed_trace_symbol_counts,
    shuffled_packed_trace_like,
    truncate_packed_trace_active_frames,
)
from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m23.relevance import M23RelevanceDataset, generate_m23_relevance_examples, m23_collate
from lojban_evolution.m24.compression import PackedTraceAdvisor, m24_2_promotion_gate_metrics, train_m24_substrate_compression


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


def test_m24_2_low_strict_accuracy_cannot_promote_from_compression_only() -> None:
    metrics = m24_2_promotion_gate_metrics(
        hard_bottleneck_configured=True,
        strict_accuracy=0.44,
        predicted_vs_shuffled_delta=0.62,
        predicted_vs_random_delta=0.64,
        hard_bottleneck_trace_accuracy=0.06,
        effective_packed_symbol_to_prompt_ratio=0.45,
        symbol_budget_respected=True,
        advisor_vs_prompt_delta=0.01,
    )

    assert metrics["m24_2_gate_token_reduction_positive"] == 1.0
    assert metrics["m24_2_gate_trace_exact_floor"] == 1.0
    assert metrics["m24_2_gate_strict_accuracy_retained"] == 0.0
    assert metrics["m24_2_promotion_candidate"] == 0.0
    assert metrics["m24_2_promotion_gate_pass_rate"] == pytest.approx(6.0 / 7.0)


def test_hard_symbolic_trace_bottleneck_helpers_preserve_integer_contract() -> None:
    spec = packed_trace_spec(max_frames=3)
    packed = torch.zeros(1, 3, spec.width, dtype=torch.int32)
    packed[0, 0, spec.active_col] = 1
    packed[0, 0, spec.stop_col] = 1
    packed[0, 0, spec.gismu_col] = 3
    packed[0, 0, spec.cmavo_start] = 1
    packed[0, 0, spec.cmavo_start + 2] = 1
    packed[0, 0, spec.judri_start] = 4
    packed[0, 1, spec.stop_col] = 1
    packed[0, 1, spec.cmavo_start] = 1
    packed[0, 2, spec.active_col] = 1
    packed[0, 2, spec.gismu_col] = 0
    packed[0, 2, spec.cmavo_start + 1] = 1
    packed[0, 2, spec.judri_start] = 2
    packed[0, 2, spec.judri_start + 1] = 5

    assert packed_trace_symbol_counts(packed).tolist() == [11]

    truncated = truncate_packed_trace_active_frames(packed, active_frame_budget=1)
    assert truncated.dtype == packed.dtype
    assert truncated.shape == packed.shape
    assert truncated[0, 0].eq(packed[0, 0]).all()
    assert truncated[0, 1].eq(0).all()
    assert truncated[0, 2].eq(0).all()
    assert packed_trace_symbol_counts(truncated).tolist() == [6]

    budgeted = budget_packed_trace_symbols(packed, symbol_budget=8)
    assert budgeted.dtype == packed.dtype
    assert budgeted.shape == packed.shape
    assert packed_trace_symbol_counts(budgeted).tolist() == [8]
    assert budgeted[0, 0, spec.stop_col] == 1
    assert budgeted[0, 2, spec.active_col] == 1
    assert budgeted[0, 2, spec.gismu_col] == 0
    assert budgeted[0, 2, spec.cmavo_start + 1] == 0

    combined = budget_packed_trace_symbols(packed, symbol_budget=100, active_frame_budget=1)
    assert packed_trace_symbol_counts(combined).tolist() == [6]


def test_packed_symbol_counts_include_zero_gismu_on_active_frames() -> None:
    spec = packed_trace_spec(max_frames=1)
    packed = torch.zeros(1, 1, spec.width, dtype=torch.long)
    packed[0, 0, spec.active_col] = 1
    packed[0, 0, spec.gismu_col] = 0

    assert int(packed.ne(0).sum().item()) == 1
    assert packed_trace_symbol_counts(packed).tolist() == [2]


def test_trace_advisor_contract_has_no_continuous_primary_inputs() -> None:
    advisor = PackedTraceAdvisor(max_frames=6, hidden_dim=16, active_frame_budget=2, trace_symbol_budget=8)
    params = inspect.signature(advisor.forward).parameters

    assert list(params) == ["packed_trace"]
    assert advisor.primary_trace_input == "packed_symbolic_trace"
    assert advisor.active_frame_budget == 2
    assert advisor.trace_symbol_budget == 8
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


def test_trace_advisor_budget_zero_is_disabled_and_negative_raises() -> None:
    advisor = PackedTraceAdvisor(max_frames=6, hidden_dim=16, active_frame_budget=0, trace_symbol_budget=0)

    assert advisor.active_frame_budget is None
    assert advisor.trace_symbol_budget is None

    with pytest.raises(ValueError, match="active_frame_budget"):
        PackedTraceAdvisor(max_frames=6, hidden_dim=16, active_frame_budget=-1)
    with pytest.raises(ValueError, match="trace_symbol_budget"):
        PackedTraceAdvisor(max_frames=6, hidden_dim=16, trace_symbol_budget=-1)


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
        active_frame_budget=2,
        trace_symbol_budget=8,
        device="cpu",
    )
    metrics = result["metrics"]

    assert result["config"]["mdl_weight"] == 0.1
    assert result["config"]["active_frame_budget"] == 2
    assert result["config"]["trace_symbol_budget"] == 8
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
        "active_frame_budget",
        "trace_symbol_budget",
        "hard_trace_length_bottleneck_active",
        "hard_symbol_budget_active",
        "mean_predicted_emitted_symbols_before_bottleneck",
        "mean_predicted_emitted_symbols_after_bottleneck",
        "mean_oracle_emitted_symbols_before_bottleneck",
        "mean_oracle_emitted_symbols_after_bottleneck",
        "diagnostic_mean_predicted_raw_nonzero_entries",
        "diagnostic_mean_oracle_raw_nonzero_entries",
        "predicted_symbol_budget_overflow_rate",
        "oracle_symbol_budget_overflow_rate",
        "predicted_bottleneck_symbol_drop_rate",
        "oracle_bottleneck_symbol_drop_rate",
        "effective_packed_symbol_to_prompt_ratio",
        "effective_token_reduction_ratio",
        "hard_bottleneck_trace_accuracy",
        "hard_bottleneck_vs_shuffled_delta",
        "hard_bottleneck_vs_random_delta",
        "m24_2_hard_bottleneck_strict_accuracy",
        "m24_2_hard_bottleneck_trace_exact_accuracy",
        "m24_2_hard_bottleneck_token_count",
        "m24_2_hard_bottleneck_compression_ratio",
        "m24_2_hard_bottleneck_accuracy_per_token",
        "m24_2_hard_bottleneck_delta_vs_prompt_only",
        "m24_2_hard_bottleneck_symbol_error_rate",
        "m24_2_hard_bottleneck_score",
        "m24_2_promotion_gate_pass_rate",
        "m24_2_promotion_candidate",
        "m24_2_gate_hard_bottleneck_configured",
        "m24_2_gate_strict_accuracy_retained",
        "m24_2_gate_trace_beats_shuffled_strong",
        "m24_2_gate_trace_beats_random_strong",
        "m24_2_gate_trace_exact_floor",
        "m24_2_gate_symbol_budget_respected",
        "m24_2_gate_token_reduction_positive",
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
    assert metrics["substrate_token_count"] == metrics["mean_predicted_emitted_symbols_after_bottleneck"]
    assert metrics["substrate_tokens"] == metrics["mean_predicted_emitted_symbols_after_bottleneck"]
    assert metrics["accuracy_per_packed_symbol"] == metrics["predicted_trace_accuracy"] / max(
        1.0, metrics["mean_predicted_emitted_symbols_after_bottleneck"]
    )
    assert metrics["active_frame_budget"] == 2.0
    assert metrics["trace_symbol_budget"] == 8.0
    assert metrics["hard_trace_length_bottleneck_active"] == 1.0
    assert metrics["hard_symbol_budget_active"] == 1.0
    assert metrics["m24_2_hard_bottleneck_strict_accuracy"] == metrics["strict_accuracy"]
    assert metrics["m24_2_hard_bottleneck_trace_exact_accuracy"] == metrics["hard_bottleneck_trace_accuracy"]
    assert metrics["m24_2_hard_bottleneck_token_count"] == metrics["substrate_token_count"]
    assert metrics["m24_2_hard_bottleneck_compression_ratio"] == metrics["effective_packed_symbol_to_prompt_ratio"]
    assert metrics["m24_2_hard_bottleneck_accuracy_per_token"] == metrics["accuracy_per_packed_symbol"]
    assert metrics["m24_2_hard_bottleneck_delta_vs_prompt_only"] == metrics["advisor_vs_prompt_delta"]
    assert metrics["m24_2_hard_bottleneck_symbol_error_rate"] == 1.0 - metrics["hard_bottleneck_trace_accuracy"]
    assert metrics["m24_2_hard_bottleneck_score"] == metrics["m24_2_promotion_gate_pass_rate"]
    assert metrics["mean_predicted_emitted_symbols_after_bottleneck"] <= 8.0
    assert metrics["mean_oracle_emitted_symbols_after_bottleneck"] <= 8.0
    assert metrics["hard_bottleneck_vs_shuffled_delta"] == metrics["predicted_vs_shuffled_delta"]
    assert 0.0 <= metrics["m24_2_promotion_gate_pass_rate"] <= 1.0

    examples = result["eval_examples"][:4]
    vocab = result["vocab"]
    batch = m23_collate([M23RelevanceDataset(examples, vocab)[i] for i in range(len(examples))])
    outputs = result["generator"](batch["input_ids"])
    predicted = pack_symbolic_trace_from_outputs(outputs)
    assert predicted.dtype == torch.long
