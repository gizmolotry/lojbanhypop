from __future__ import annotations

import inspect

import pytest
import torch

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m25.emergent_bridi import (
    LOOSE_ARG,
    LOOSE_OPEN,
    LOOSE_PAD,
    LOOSE_PRED,
    LOOSE_STOP,
    LooseStreamAdvisor,
    M25EmergentBridiQFormer,
    M25LooseBridiDataset,
    budget_loose_stream_symbols,
    budget_prompt_tokens,
    generate_m25_emergent_bridi_examples,
    loose_stream_symbol_counts,
    m25_collate,
    pack_loose_stream_from_outputs,
    random_loose_stream_like,
    shuffled_loose_stream_like,
    train_m25_emergent_bridi,
)


def test_m25_generator_emits_variable_loose_bridi_streams() -> None:
    examples = generate_m25_emergent_bridi_examples(24, seed=25, clean_fraction=0.0, max_symbols=32)
    lengths = {len([symbol for symbol in row.loose_symbols if symbol.type_id != LOOSE_PAD]) for row in examples}
    first_types = [symbol.type_id for symbol in examples[0].loose_symbols]

    assert len(lengths) > 1
    assert LOOSE_OPEN in first_types
    assert LOOSE_PRED in first_types
    assert LOOSE_ARG in first_types
    assert first_types[-1] == LOOSE_STOP or len(first_types) == 32


def test_m25_integer_stream_budget_and_controls_preserve_shape() -> None:
    examples = generate_m25_emergent_bridi_examples(8, seed=26, clean_fraction=0.0, max_symbols=16)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    batch = m25_collate([M25LooseBridiDataset(examples, vocab, max_symbols=16)[i] for i in range(8)])
    oracle = batch["stream_targets"]

    assert oracle.dtype == torch.long
    budgeted = budget_loose_stream_symbols(oracle, symbol_budget=5)
    shuffled = shuffled_loose_stream_like(oracle, seed=26)
    random_stream = random_loose_stream_like(oracle, seed=26)

    assert budgeted.dtype == torch.long
    assert budgeted.shape == oracle.shape
    assert shuffled.shape == oracle.shape
    assert random_stream.shape == oracle.shape
    assert loose_stream_symbol_counts(budgeted).max().item() <= 5
    assert sorted(shuffled.flatten().tolist()) == sorted(oracle.flatten().tolist())


def test_m25_matched_prompt_budget_masks_prompt_tail() -> None:
    input_ids = torch.tensor([[4, 5, 6, 7, 0], [8, 9, 0, 0, 0]], dtype=torch.long)

    budgeted = budget_prompt_tokens(input_ids, token_budget=2)
    disabled = budget_prompt_tokens(input_ids, token_budget=0)

    assert budgeted.tolist() == [[4, 5, 0, 0, 0], [8, 9, 0, 0, 0]]
    assert disabled.tolist() == input_ids.tolist()


def test_m25_qformer_pack_and_advisor_rejects_continuous_smuggling() -> None:
    examples = generate_m25_emergent_bridi_examples(4, seed=27, max_symbols=12)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    batch = m25_collate([M25LooseBridiDataset(examples, vocab, max_symbols=12)[i] for i in range(4)])
    model = M25EmergentBridiQFormer(vocab_size=len(vocab), max_symbols=12, embedding_dim=8, hidden_dim=16)
    advisor = LooseStreamAdvisor(max_symbols=12, hidden_dim=16, symbol_budget=6)

    packed = pack_loose_stream_from_outputs(model(batch["input_ids"]))
    assert packed.dtype == torch.long
    assert packed.shape == (4, 12, 3)
    assert list(inspect.signature(advisor.forward).parameters) == ["loose_stream"]
    assert advisor.primary_trace_input == "loose_integer_bridi_stream"
    with pytest.raises(TypeError, match="integer loose bridi streams"):
        advisor(torch.randn(4, 12, 3))


def test_m25_tiny_cpu_training_reports_symbolic_metrics() -> None:
    result = train_m25_emergent_bridi(
        train_size=24,
        eval_size=12,
        generator_epochs=1,
        advisor_epochs=1,
        prompt_epochs=1,
        batch_size=12,
        seed=25,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        max_symbols=16,
        symbol_budget=8,
        matched_prompt_budget=8,
        mdl_weight=0.1,
        device="cpu",
    )
    metrics = result["metrics"]

    assert result["config"]["symbol_budget"] == 8
    assert result["config"]["matched_prompt_budget"] == 8
    assert result["config"]["mdl_weight"] == 0.1
    assert metrics["generator_trainable_parameter_count_after_freeze"] == 0.0
    assert metrics["generator_parameters_unchanged_after_advisor"] == 1.0
    assert metrics["advisor_primary_trace_is_symbolic"] == 1.0
    assert metrics["continuous_trace_smuggling_detected"] == 0.0
    for key in (
        "strict_accuracy",
        "predicted_stream_accuracy",
        "oracle_stream_accuracy",
        "shuffled_stream_accuracy",
        "random_stream_accuracy",
        "zero_stream_accuracy",
        "prompt_only_accuracy",
        "matched_prompt_accuracy",
        "m25_strict_delta_vs_matched_prompt",
        "loose_stream_exact_accuracy",
        "stream_type_accuracy",
        "stream_value_accuracy",
        "stream_aux_accuracy",
        "token_reduction_ratio",
        "accuracy_per_loose_symbol",
        "matched_prompt_accuracy_per_token",
        "m25_accuracy_per_symbol_delta_vs_matched_prompt",
        "m25_gate_beats_matched_prompt",
        "m25_promotion_gate_pass_rate",
        "m25_promotion_candidate",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert metrics["predicted_vs_shuffled_delta"] == metrics["predicted_stream_accuracy"] - metrics["shuffled_stream_accuracy"]
