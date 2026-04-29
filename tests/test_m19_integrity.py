from __future__ import annotations

from pytest import approx

from lojban_evolution.m19.integrity import (
    build_masked_eval_rows,
    build_train_pair_index,
    classify_integrity_status,
    compute_integrity_metrics,
    extract_training_pair,
    mask_prompt_text,
    split_eval_rows_by_overlap,
)


def test_extract_training_pair_handles_fluid_and_crystal_rows() -> None:
    fluid = {
        "text": "[MODE=FLUID]\nSolve the logic question.\n\nQuestion: Where is the ball?\nFinal answer: desk"
    }
    crystal = {
        "text": "[MODE=CRYSTAL]\nQUESTION: Who moved it?\nTRACE: A B C\nANSWER: bob"
    }

    assert extract_training_pair(fluid) == ("Where is the ball?", "desk")
    assert extract_training_pair(crystal) == ("Who moved it?", "bob")


def test_overlap_split_and_masking_controls() -> None:
    train_rows = [
        {"text": "[MODE=FLUID]\nQuestion: Where is the ball?\nFinal answer: desk"},
        {"text": "[MODE=CRYSTAL]\nQUESTION: Who moved it?\nTRACE: X\nANSWER: bob"},
    ]
    eval_rows = [
        {"prompt": "Where is the ball?", "answer": "desk"},
        {"prompt": "Who moved it?", "answer": "bob"},
        {"prompt": "Is the box left of the crate?", "answer": "no"},
    ]

    train_pairs = build_train_pair_index(train_rows)
    overlap, purged = split_eval_rows_by_overlap(eval_rows, train_pairs)
    masked = build_masked_eval_rows(purged)

    assert len(overlap) == 2
    assert len(purged) == 1
    assert masked[0]["prompt"] != purged[0]["prompt"]
    assert mask_prompt_text("Alice has 2 boxes.") == "x x 0 x."


def test_compute_integrity_metrics_and_status() -> None:
    full = {"metrics": {"strict_accuracy": 0.38, "avg_tokens": 32.0}}
    purged = {
        "metrics": {
            "strict_accuracy": 0.37,
            "overall_phrase_accuracy": 0.37,
            "avg_tokens": 32.0,
            "lift_vs_base": 0.35,
            "lift_vs_en_cot": 0.36,
            "lift_vs_random": 0.34,
            "random_accuracy": 0.03,
        },
        "results": {"SCRATCHPAD-ONLY": {"accuracy": 0.02}},
    }
    overlap = {"metrics": {"strict_accuracy": 0.39, "overall_phrase_accuracy": 0.39, "avg_tokens": 32.0}}
    masked = {
        "metrics": {
            "strict_accuracy": 0.04,
            "overall_phrase_accuracy": 0.05,
            "avg_tokens": 32.0,
            "lift_vs_base": 0.01,
            "lift_vs_random": 0.01,
        }
    }
    audit = {"headline": {"qformer_accuracy": 1.0, "qformer_phrase_accuracy": 1.0, "lift_vs_base": 1.0, "lift_vs_random": 1.0}}

    metrics = compute_integrity_metrics(
        full_report=full,
        purged_report=purged,
        overlap_report=overlap,
        masked_report=masked,
        audit_report=audit,
        overlap_size=89,
        purged_size=311,
        eval_size=400,
    )

    assert metrics["purged_accuracy"] == 0.37
    assert metrics["overlap_gap"] == 0.020000000000000018
    assert metrics["masked_collapse_gap"] == approx(0.33)
    assert metrics["integrity_audit_flag"] is False
    assert classify_integrity_status(metrics) == "pass"
