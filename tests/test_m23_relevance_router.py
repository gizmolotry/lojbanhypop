from __future__ import annotations

import torch

from lojban_evolution.m23.relevance import (
    M23CausalRelevanceQFormer,
    M23RelevanceDataset,
    compute_m23_loss,
    generate_m23_relevance_examples,
    m23_collate,
    train_m23_relevance_router,
)
from lojban_evolution.m21.bridi import build_vocab


def test_m23_generator_emits_relevant_and_decoy_frames() -> None:
    examples = generate_m23_relevance_examples(64, seed=23, clean_fraction=0.0)
    assert examples
    assert {row.relevance_surface for row in examples} == {"decoy_relation_ood"}
    assert all(row.relevant_frame_indices for row in examples)
    assert all(row.decoy_frame_indices for row in examples)
    assert all(set(row.relevant_frame_indices).isdisjoint(row.decoy_frame_indices) for row in examples)
    assert all(row.answer_label != "" for row in examples)


def test_m23_dataset_collates_relevance_masks() -> None:
    examples = generate_m23_relevance_examples(12, seed=29, clean_fraction=0.0)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    dataset = M23RelevanceDataset(examples, vocab)
    batch = m23_collate([dataset[i] for i in range(4)])
    assert batch["relevance_targets"].shape == batch["active_targets"].shape
    assert batch["decoy_targets"].shape == batch["active_targets"].shape
    assert torch.all(batch["relevance_targets"].sum(dim=-1) >= 1.0)
    assert torch.all(batch["decoy_targets"].sum(dim=-1) >= 1.0)


def test_m23_model_forward_and_rank_loss_are_finite() -> None:
    examples = generate_m23_relevance_examples(16, seed=31, clean_fraction=0.0)
    vocab = build_vocab(examples)  # type: ignore[arg-type]
    dataset = M23RelevanceDataset(examples, vocab)
    batch = m23_collate([dataset[i] for i in range(8)])
    model = M23CausalRelevanceQFormer(vocab_size=len(vocab), embedding_dim=16, hidden_dim=32, judri_bridge_gate=True)
    outputs = model(batch["input_ids"])
    assert outputs["relevance_logits"].shape == batch["active_targets"].shape
    assert outputs["relevance_answer_logits"].shape[0] == batch["answer_id"].shape[0]
    loss, pieces = compute_m23_loss(
        outputs,
        batch,
        use_relevance_answer=True,
        relevance_rank_weight=1.0,
        trace_weight=1.0,
        answer_weight=1.0,
        counterfactual_weight=0.0,
        brivi_lock_weight=0.0,
        frame_necessity_weight=0.0,
        mdl_weight=0.0,
    )
    assert torch.isfinite(loss)
    assert pieces["loss_relevance_rank"] >= 0.0
    loss_punished, punished_pieces = compute_m23_loss(
        outputs,
        batch,
        use_relevance_answer=True,
        relevance_rank_weight=1.0,
        trace_exact_surrogate_weight=0.5,
        trace_weight=1.0,
        answer_weight=1.0,
        counterfactual_weight=0.0,
        brivi_lock_weight=0.0,
        frame_necessity_weight=0.0,
        mdl_weight=0.0,
    )
    assert torch.isfinite(loss_punished)
    assert punished_pieces["loss_trace_exact_surrogate"] >= 0.0
    assert loss_punished >= loss


def test_m23_tiny_train_runs_scale_and_router() -> None:
    common = dict(train_size=64, eval_size=32, epochs=1, batch_size=16, embedding_dim=16, hidden_dim=32, device="cpu")
    scale = train_m23_relevance_router(seed=23, use_relevance_router=False, relevance_rank_weight=0.0, **common)
    router = train_m23_relevance_router(seed=23, use_relevance_router=True, relevance_rank_weight=0.5, **common)
    assert "decoy_relation_ood_accuracy" in scale["metrics"]
    assert "relevance_top1_accuracy" in router["metrics"]
    assert router["metrics"]["oracle_relevance_accuracy"] >= 0.0
    assert router["metrics"]["decoy_only_accuracy"] >= 0.0
