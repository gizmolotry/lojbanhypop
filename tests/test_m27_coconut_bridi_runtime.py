from __future__ import annotations

import torch

from lojban_evolution.m21.bridi import build_vocab
from lojban_evolution.m25.emergent_bridi import M25LooseBridiDataset, generate_m25_emergent_bridi_examples, m25_collate
from lojban_evolution.m27.runtime import (
    M27CoconutBridiRuntime,
    compute_m27_loss,
    evaluate_m27_coconut_bridi_runtime,
    m27_promotion_gate_metrics,
    probe_m27_answer_gradient_flow,
    train_m27_coconut_bridi_runtime,
)


def _tiny_batch(size: int = 8) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    examples = generate_m25_emergent_bridi_examples(size, seed=270, max_symbols=8)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    dataset = M25LooseBridiDataset(examples, vocab, max_symbols=8)
    return m25_collate([dataset[i] for i in range(size)]), vocab


def _tiny_model(vocab: dict[str, int]) -> M27CoconutBridiRuntime:
    return M27CoconutBridiRuntime(
        vocab_size=len(vocab),
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
    )


def test_m27_dataset_exports_relevance_and_decoy_masks() -> None:
    batch, _ = _tiny_batch(8)

    assert batch["relevance_targets"].shape == (8, 8)
    assert batch["decoy_targets"].shape == (8, 8)
    assert float(batch["relevance_targets"].sum().item()) > 0.0
    assert float(batch["decoy_targets"].sum().item()) > 0.0


def test_m27_forward_emits_autoregressive_trace_shapes() -> None:
    batch, vocab = _tiny_batch(4)
    model = _tiny_model(vocab)

    outputs = model(batch["input_ids"], teacher_trace=batch["stream_targets"], max_steps=6)

    assert outputs["active_logits"].shape == (4, 8)
    assert outputs["type_logits"].shape[:2] == (4, 8)
    assert outputs["value_logits"].shape[:2] == (4, 8)
    assert outputs["aux_logits"].shape[:2] == (4, 8)
    assert outputs["coconut_states"].shape == (4, 8, 16)
    assert outputs["prompt_attention"].shape[:2] == (4, 8)
    assert outputs["hard_trace_tokens"].shape == (4, 8, 3)
    assert outputs["soft_trace_embeddings"].requires_grad
    assert outputs["answer_logits"].requires_grad
    assert model.trace_runtime_mode == "autoregressive_coconut_loose_bridi"


def test_m27_relevance_runtime_scores_existing_trace_slots() -> None:
    batch, vocab = _tiny_batch(4)
    model = M27CoconutBridiRuntime(
        vocab_size=len(vocab),
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        enable_relevance_runtime=True,
    )

    outputs = model(batch["input_ids"], teacher_trace=batch["stream_targets"], max_steps=6)

    assert outputs["m27_relevance_runtime_enabled"].item() == 1.0
    assert outputs["relevance_logits"].shape == (4, 8)
    assert outputs["relevance_weights"].shape == (4, 8)
    assert outputs["relevance_answer_logits"].shape == outputs["answer_logits"].shape
    assert torch.isfinite(outputs["relevance_weights"]).all()


def test_m27_relevance_rank_loss_is_finite() -> None:
    batch, vocab = _tiny_batch(6)
    model = M27CoconutBridiRuntime(
        vocab_size=len(vocab),
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        enable_relevance_runtime=True,
    )

    teacher = model(batch["input_ids"], teacher_trace=batch["stream_targets"])
    soft = model(batch["input_ids"])
    loss, metrics = compute_m27_loss(
        teacher,
        batch,
        answer_outputs=soft,
        relevance_rank_weight=0.5,
        use_relevance_answer=True,
    )

    assert torch.isfinite(loss)
    assert "loss_relevance_rank" in metrics
    assert metrics["m27_relevance_rank_valid_fraction"] > 0.0
    assert metrics["answer_loss_uses_relevance_runtime_trace"] == 1.0


def test_m27_eval_reports_relevance_controls() -> None:
    examples = generate_m25_emergent_bridi_examples(12, seed=271, max_symbols=8)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    model = M27CoconutBridiRuntime(
        vocab_size=len(vocab),
        max_symbols=8,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        symbol_budget=8,
        enable_relevance_runtime=True,
    )

    payload = evaluate_m27_coconut_bridi_runtime(
        model=model,
        examples=examples,
        vocab=vocab,
        batch_size=6,
        device="cpu",
        seed=27,
    )
    metrics = payload["metrics"]

    assert metrics["m27_relevance_runtime_enabled"] == 1.0
    assert metrics["m27_relevance_runtime_active"] == 1.0
    for key in (
        "m27_relevance_top1_accuracy",
        "m27_relevance_margin",
        "m27_relevance_full_accuracy",
        "m27_relevance_oracle_accuracy",
        "m27_relevance_random_accuracy",
        "m27_relevance_decoy_only_accuracy",
        "m27_relevance_full_vs_random_delta",
        "m27_relevance_oracle_lift",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_m27_answer_loss_backprops_through_recurrent_bridi_loop() -> None:
    batch, vocab = _tiny_batch(8)
    model = _tiny_model(vocab)

    probe = probe_m27_answer_gradient_flow(model, batch)

    assert probe.answer_loss_reaches_generator == 1.0
    assert probe.answer_loss_reaches_coconut_cell == 1.0
    assert probe.answer_loss_reaches_symbol_heads == 1.0
    assert probe.answer_loss_reaches_recurrent_bridi_feedback == 1.0
    assert probe.answer_loss_reaches_trace_slot_advisor == 1.0
    assert probe.answer_loss_reaches_advisor_classifier == 0.0
    assert probe.answer_loss_reaches_language_backbone == 1.0
    assert probe.answer_loss_reaches_bridge == 1.0
    assert probe.hard_argmax_training_cut_detected == 0.0
    assert probe.torch_no_grad_training_cut_detected == 0.0


def test_m27_teacher_feedback_changes_later_coconut_states() -> None:
    batch, vocab = _tiny_batch(4)
    model = _tiny_model(vocab)
    teacher = batch["stream_targets"].clone()
    edited = teacher.clone()
    edited[:, 0, 1] = (edited[:, 0, 1] + 1).clamp_max(model.generator.value_vocab_size - 1)

    base = model(batch["input_ids"], teacher_trace=teacher)["coconut_states"][:, 1:]
    changed = model(batch["input_ids"], teacher_trace=edited)["coconut_states"][:, 1:]

    assert (base - changed).abs().mean().item() > 0.0


def test_m27_hard_free_run_is_real_autoregressive_runtime() -> None:
    batch, vocab = _tiny_batch(4)
    model = _tiny_model(vocab)

    soft = model(batch["input_ids"], teacher_trace=batch["stream_targets"], mode="soft_train", max_steps=6)
    hard = model(batch["input_ids"], mode="hard_free_run", max_steps=6)

    assert soft["soft_trace_embeddings"].requires_grad
    assert hard["hard_trace_tokens"].dtype == torch.long
    assert hard["hard_trace_tokens"].shape[:2] == hard["stop_logits"].shape[:2]
    assert hard["coconut_states"].shape[0] == batch["input_ids"].shape[0]
    assert hard["coconut_states"].shape[1] == hard["hard_trace_tokens"].shape[1]
    assert hard["answer_logits"].shape[0] == batch["input_ids"].shape[0]


def test_m27_tiny_cpu_training_reports_runtime_metrics() -> None:
    result = train_m27_coconut_bridi_runtime(
        train_size=24,
        eval_size=12,
        epochs=1,
        prompt_epochs=1,
        batch_size=12,
        seed=27,
        embedding_dim=8,
        hidden_dim=16,
        advisor_hidden_dim=16,
        max_symbols=8,
        max_steps=8,
        symbol_budget=8,
        mdl_weight=0.1,
        enable_relevance_runtime=True,
        relevance_rank_weight=0.25,
        use_relevance_answer=True,
        device="cpu",
    )
    metrics = result["metrics"]

    assert result["config"]["organism_mode"] == "coconut_autoregressive_lm_hidden_bridi_bridge"
    assert metrics["single_optimizer_end_to_end_training"] == 1.0
    assert metrics["hard_argmax_training_cut_detected"] == 0.0
    assert metrics["autoregressive_coconut_runtime_active"] == 1.0
    assert metrics["soft_train_and_hard_free_run_both_available"] == 1.0
    assert metrics["answer_loss_reaches_generator"] == 1.0
    assert metrics["answer_loss_reaches_coconut_cell"] == 1.0
    assert metrics["answer_loss_reaches_recurrent_bridi_feedback"] == 1.0
    assert metrics["answer_loss_reaches_language_backbone"] == 1.0
    assert metrics["answer_loss_reaches_bridge"] == 1.0
    assert metrics["m27_training_answer_loss_uses_soft_free_run_trace"] == 1.0
    assert metrics["m27_relevance_runtime_enabled"] == 1.0
    assert metrics["m27_training_answer_loss_uses_relevance_runtime_trace"] == 1.0
    assert metrics["m27_inherited_contract_bundle_present"] == 1.0
    assert metrics["m27_gate_autoregressive_step_dependency"] == 1.0
    for key in (
        "strict_accuracy",
        "hard_free_run_strict_accuracy",
        "no_recurrence_accuracy",
        "multi_step_delta_vs_no_recurrence",
        "soft_hard_accuracy_gap",
        "predicted_vs_zero_delta",
        "loose_stream_exact_accuracy",
        "m27_full_organism_gate_pass_rate",
        "m27_promotion_candidate",
    ):
        assert key in metrics
        assert isinstance(metrics[key], float)


def test_m27_prompt_comparability_gate_requires_matched_prompt_win() -> None:
    gates = m27_promotion_gate_metrics(
        {
            "answer_loss_reaches_generator": 1.0,
            "answer_loss_reaches_coconut_cell": 1.0,
            "answer_loss_reaches_symbol_heads": 1.0,
            "answer_loss_reaches_language_backbone": 1.0,
            "answer_loss_reaches_bridge": 1.0,
            "answer_loss_reaches_recurrent_bridi_feedback": 1.0,
            "m27_training_answer_loss_uses_soft_free_run_trace": 1.0,
            "hard_argmax_training_cut_detected": 0.0,
            "m27_step_dependency_delta": 0.1,
            "soft_train_and_hard_free_run_both_available": 1.0,
            "raw_prompt_bypass_blocked": 1.0,
            "m27_strict_delta_vs_matched_prompt": -0.01,
            "predicted_vs_zero_delta": 0.5,
        }
    )

    assert gates["m27_full_organism_candidate"] == 1.0
    assert gates["m27_gate_beats_matched_prompt"] == 0.0
    assert gates["m27_prompt_comparable_candidate"] == 0.0
    assert gates["m27_promotion_candidate"] == 0.0
