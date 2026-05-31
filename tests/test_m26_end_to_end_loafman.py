from __future__ import annotations

import torch

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m25.emergent_bridi import (
    M25LooseBridiDataset,
    generate_m25_emergent_bridi_examples,
    m25_collate,
)
from lojban_evolution.m26.end_to_end import (
    DifferentiableLooseStreamAdvisor,
    M26EndToEndLoafman,
    m26_promotion_gate_metrics,
    probe_m26_answer_gradient_flow,
    train_m26_end_to_end_loafman,
)


def _tiny_batch(size: int = 8) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    examples = generate_m25_emergent_bridi_examples(size, seed=260, max_symbols=16)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=16)
    batch = m25_collate([dataset[i] for i in range(size)])
    return batch, vocab


def test_m26_answer_loss_backprops_into_bridi_generator() -> None:
    batch, vocab = _tiny_batch(8)
    model = M26EndToEndLoafman(
        vocab_size=len(vocab),
        max_symbols=16,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
    )

    probe = probe_m26_answer_gradient_flow(model, batch)

    assert probe.answer_loss_reaches_generator == 1.0
    assert probe.answer_loss_reaches_symbol_heads == 1.0
    assert probe.answer_loss_generator_grad_norm > 0.0
    assert probe.answer_loss_symbol_head_grad_norm > 0.0
    assert probe.answer_loss_advisor_grad_norm > 0.0


def test_m26_soft_trace_handoff_remains_differentiable() -> None:
    batch, vocab = _tiny_batch(4)
    model = M26EndToEndLoafman(
        vocab_size=len(vocab),
        max_symbols=16,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
    )

    outputs = model(batch["input_ids"])

    assert outputs["trace_state"].dtype.is_floating_point
    assert outputs["trace_state"].requires_grad
    assert outputs["answer_logits"].requires_grad
    assert model.advisor.primary_trace_input == "soft_differentiable_loose_bridi_stream"
    assert "raw_prompt_tokens" in model.advisor.disallowed_primary_inputs


def test_m26_advisor_rejects_positional_hard_stream_args() -> None:
    advisor = DifferentiableLooseStreamAdvisor(max_symbols=4, hidden_dim=8)
    hard_stream = torch.zeros(2, 4, 3, dtype=torch.long)

    try:
        advisor(hard_stream)
    except TypeError as exc:
        assert "logits by keyword" in str(exc)
    else:  # pragma: no cover - defensive assertion branch
        raise AssertionError("advisor accepted a hard positional stream argument")


def test_m26_tiny_cpu_training_reports_spinal_cord_metrics() -> None:
    result = train_m26_end_to_end_loafman(
        train_size=24,
        eval_size=12,
        epochs=1,
        batch_size=12,
        seed=26,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        max_symbols=16,
        symbol_budget=8,
        mdl_weight=0.1,
        device="cpu",
    )
    metrics = result["metrics"]

    assert result["config"]["symbol_budget"] == 8
    assert metrics["single_optimizer_end_to_end_training"] == 1.0
    assert metrics["hard_argmax_training_cut_detected"] == 0.0
    assert metrics["torch_no_grad_training_cut_detected"] == 0.0
    assert metrics["advisor_primary_trace_is_differentiable"] == 1.0
    assert metrics["answer_loss_reaches_generator"] == 1.0
    assert metrics["answer_loss_reaches_symbol_heads"] == 1.0
    for key in (
        "strict_accuracy",
        "end_to_end_answer_accuracy",
        "shuffled_trace_accuracy",
        "random_trace_accuracy",
        "zero_trace_accuracy",
        "prompt_only_accuracy",
        "matched_prompt_accuracy",
        "m26_strict_delta_vs_matched_prompt",
        "predicted_vs_zero_delta",
        "loose_stream_exact_accuracy",
        "accuracy_per_loose_symbol",
        "m26_spinal_cord_gate_pass_rate",
        "m26_promotion_candidate",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)
    assert metrics["mean_matched_prompt_tokens"] <= result["config"]["matched_prompt_budget"]


def test_m26_prompt_comparability_gate_requires_matched_prompt_win() -> None:
    metrics = {
        "answer_loss_reaches_generator": 1.0,
        "answer_loss_reaches_symbol_heads": 1.0,
        "single_optimizer_end_to_end_training": 1.0,
        "hard_argmax_training_cut_detected": 0.0,
        "predicted_vs_zero_delta": 0.9,
        "m26_strict_delta_vs_matched_prompt": -0.001,
    }

    gates = m26_promotion_gate_metrics(metrics)

    assert gates["m26_spinal_cord_candidate"] == 1.0
    assert gates["m26_gate_beats_matched_prompt"] == 0.0
    assert gates["m26_prompt_comparable_candidate"] == 0.0
    assert gates["m26_promotion_candidate"] == 0.0
