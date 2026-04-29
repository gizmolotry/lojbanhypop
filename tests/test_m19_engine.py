from __future__ import annotations

import torch

from lojban_evolution.m19.engine import (
    batched_pairwise_cosine_stats,
    compute_query_repulsion_loss,
    operator_distribution_stats,
    pairwise_cosine_stats,
)


def test_pairwise_cosine_stats_detects_identical_vectors() -> None:
    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    stats = pairwise_cosine_stats(vectors)

    assert stats["pairwise_cosine_mean"] == 1.0
    assert stats["pairwise_cosine_max"] == 1.0
    assert stats["anisotropy"] == 1.0


def test_batched_pairwise_cosine_stats_respects_lengths() -> None:
    trace = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ]
    )
    lengths = torch.tensor([2, 2])
    stats = batched_pairwise_cosine_stats(trace, lengths=lengths)

    assert abs(stats["pairwise_cosine_mean"] - 0.5) < 1e-6
    assert stats["pairwise_cosine_max"] == 1.0


def test_query_repulsion_penalizes_collapsed_queries() -> None:
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    collapsed_loss = compute_query_repulsion_loss(collapsed, margin=0.15)
    separated_loss = compute_query_repulsion_loss(separated, margin=0.15)

    assert float(collapsed_loss.item()) > 0.8
    assert float(separated_loss.item()) == 0.0


def test_operator_distribution_stats_reports_entropy_and_top1_share() -> None:
    logits = torch.tensor(
        [
            [[10.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        ]
    )
    stats = operator_distribution_stats(logits)

    assert stats["operator_top1_share_mean"] > 0.99
    assert stats["operator_entropy_ratio_mean"] < 0.05
