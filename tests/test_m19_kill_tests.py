from __future__ import annotations

from lojban_evolution.m19.kill_tests import (
    anonymize_entities_text,
    build_entity_anonymized_rows,
    build_entity_renamed_rows,
    build_format_flattened_rows,
    build_numeric_normalized_rows,
    classify_kill_test_status,
    compute_kill_test_metrics,
)


def test_kill_test_row_transforms_are_semantic_shape_preserving() -> None:
    rows = [{"prompt": "Alice moved 2 boxes to Bob.", "answer": "Alice"}]
    entity = build_entity_anonymized_rows(rows)
    renamed = build_entity_renamed_rows(rows)
    flattened = build_format_flattened_rows(rows)
    numeric = build_numeric_normalized_rows(rows)

    assert entity[0]["prompt"] != rows[0]["prompt"]
    assert entity[0]["answer"] != rows[0]["answer"]
    assert "entity1" in entity[0]["prompt"]
    assert renamed[0]["prompt"] == "Avery moved 2 boxes to Blake."
    assert renamed[0]["answer"] == "Avery"
    assert flattened[0]["prompt"] == "alice moved 2 boxes to bob."
    assert numeric[0]["prompt"] == "Alice moved 0 boxes to Bob."
    assert anonymize_entities_text("Alice told Bob.") == "entity1 told entity2."


def test_entity_anonymization_uses_shared_prompt_answer_mapping() -> None:
    rows = [{"prompt": "Alice told Bob that Bob won.", "answer": "Bob"}]
    entity = build_entity_anonymized_rows(rows)

    assert entity[0]["prompt"] == "entity1 told entity2 that entity2 won."
    assert entity[0]["answer"] == "entity2"


def test_entity_transforms_do_not_mangle_question_function_words() -> None:
    rows = [{"prompt": "The lawyer asked Riley. Who answered?", "answer": "Riley"}]
    entity = build_entity_anonymized_rows(rows)
    renamed = build_entity_renamed_rows(rows)

    assert entity[0]["prompt"] == "The lawyer asked entity1. Who answered?"
    assert entity[0]["answer"] == "entity1"
    assert renamed[0]["prompt"] == "The lawyer asked Avery. Who answered?"
    assert renamed[0]["answer"] == "Avery"


def test_compute_kill_metrics_and_status() -> None:
    purged = {"metrics": {"strict_accuracy": 0.37, "lift_vs_random": 0.34}}
    entity = {"metrics": {"strict_accuracy": 0.34, "lift_vs_random": 0.31}}
    entity_renamed = {"metrics": {"strict_accuracy": 0.36, "lift_vs_random": 0.33}}
    formatted = {"metrics": {"strict_accuracy": 0.35, "lift_vs_random": 0.32}}
    numeric = {"metrics": {"strict_accuracy": 0.33, "lift_vs_random": 0.30}}
    masked = {"metrics": {"strict_accuracy": 0.0, "lift_vs_random": -0.02}}

    metrics = compute_kill_test_metrics(
        purged_report=purged,
        masked_report=masked,
        entity_report=entity,
        entity_renamed_report=entity_renamed,
        format_report=formatted,
        numeric_report=numeric,
    )

    assert metrics["entity_accuracy"] == 0.34
    assert metrics["entity_renamed_accuracy"] == 0.36
    assert metrics["format_accuracy"] == 0.35
    assert metrics["numeric_accuracy"] == 0.33
    assert metrics["masked_accuracy"] == 0.0
    assert classify_kill_test_status(metrics) == "pass"
