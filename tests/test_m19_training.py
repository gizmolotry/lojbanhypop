from __future__ import annotations

import random

from lojban_evolution.m19.training import (
    checkpoint_selection_score,
    maybe_apply_surface_augmentations,
    select_best_checkpoint,
)


def test_surface_augmentations_can_compose_entity_and_format_changes() -> None:
    rng = random.Random(0)
    question, answer, flags = maybe_apply_surface_augmentations(
        "Alice asked Bob: Where is the key?",
        "Bob",
        entity_rename_probability=1.0,
        format_flatten_probability=1.0,
        rng=rng,
    )

    assert question == "avery asked blake where is the key?"
    assert answer == "blake"
    assert flags == {"entity_renamed": True, "format_flattened": True}


def test_checkpoint_selection_prefers_purged_and_audit_under_audit_purged_policy() -> None:
    candidates = [
        {
            "epoch": 1,
            "checkpoint_path": "epoch1.pt",
            "purged_accuracy": 0.30,
            "audit_qformer_accuracy": 0.90,
            "final_mean_loss": 0.70,
        },
        {
            "epoch": 2,
            "checkpoint_path": "epoch2.pt",
            "purged_accuracy": 0.42,
            "audit_qformer_accuracy": 0.60,
            "final_mean_loss": 0.80,
        },
    ]

    best = select_best_checkpoint(candidates, "audit_purged")

    assert best is not None
    assert best["checkpoint_path"] == "epoch2.pt"
    assert checkpoint_selection_score(
        purged_accuracy=0.42,
        audit_qformer_accuracy=0.60,
        final_mean_loss=0.80,
        policy="audit_purged",
    ) > checkpoint_selection_score(
        purged_accuracy=0.30,
        audit_qformer_accuracy=0.90,
        final_mean_loss=0.70,
        policy="audit_purged",
    )


def test_checkpoint_selection_uses_loss_for_final_only_policy() -> None:
    score_low_loss = checkpoint_selection_score(
        purged_accuracy=None,
        audit_qformer_accuracy=None,
        final_mean_loss=0.4,
        policy="final_only",
    )
    score_high_loss = checkpoint_selection_score(
        purged_accuracy=None,
        audit_qformer_accuracy=None,
        final_mean_loss=0.9,
        policy="final_only",
    )

    assert score_low_loss is not None
    assert score_high_loss is not None
    assert score_low_loss > score_high_loss


def test_checkpoint_selection_can_prefer_format_robust_epoch() -> None:
    candidates = [
        {
            "epoch": 1,
            "checkpoint_path": "epoch1.pt",
            "purged_accuracy": 0.40,
            "format_accuracy": 0.18,
            "audit_qformer_accuracy": 0.90,
            "final_mean_loss": 0.70,
        },
        {
            "epoch": 2,
            "checkpoint_path": "epoch2.pt",
            "purged_accuracy": 0.38,
            "format_accuracy": 0.30,
            "audit_qformer_accuracy": 0.80,
            "final_mean_loss": 0.72,
        },
    ]

    best = select_best_checkpoint(candidates, "audit_purged_format")

    assert best is not None
    assert best["checkpoint_path"] == "epoch2.pt"
    assert checkpoint_selection_score(
        purged_accuracy=0.38,
        format_accuracy=0.30,
        audit_qformer_accuracy=0.80,
        final_mean_loss=0.72,
        policy="audit_purged_format",
    ) > checkpoint_selection_score(
        purged_accuracy=0.40,
        format_accuracy=0.18,
        audit_qformer_accuracy=0.90,
        final_mean_loss=0.70,
        policy="audit_purged_format",
    )
