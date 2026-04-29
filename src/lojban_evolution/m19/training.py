from __future__ import annotations

import math
import random
from typing import Any

from .kill_tests import flatten_format_text, rename_entities_text


def maybe_apply_entity_rename_augmentation(
    question: str,
    answer: str,
    probability: float,
    rng: random.Random | None = None,
) -> tuple[str, str, bool]:
    active_rng = rng or random
    if float(probability) <= 0.0 or active_rng.random() >= float(probability):
        return question, answer, False
    mapping: dict[str, str] = {}
    return (
        rename_entities_text(question, mapping=mapping),
        rename_entities_text(answer, mapping=mapping),
        True,
    )


def maybe_apply_format_flatten_augmentation(
    question: str,
    answer: str,
    probability: float,
    rng: random.Random | None = None,
) -> tuple[str, str, bool]:
    active_rng = rng or random
    if float(probability) <= 0.0 or active_rng.random() >= float(probability):
        return question, answer, False
    return flatten_format_text(question), flatten_format_text(answer), True


def maybe_apply_surface_augmentations(
    question: str,
    answer: str,
    *,
    entity_rename_probability: float = 0.0,
    format_flatten_probability: float = 0.0,
    rng: random.Random | None = None,
) -> tuple[str, str, dict[str, bool]]:
    active_rng = rng or random
    updated_question, updated_answer, entity_applied = maybe_apply_entity_rename_augmentation(
        question,
        answer,
        probability=entity_rename_probability,
        rng=active_rng,
    )
    updated_question, updated_answer, format_applied = maybe_apply_format_flatten_augmentation(
        updated_question,
        updated_answer,
        probability=format_flatten_probability,
        rng=active_rng,
    )
    return (
        updated_question,
        updated_answer,
        {
            "entity_renamed": bool(entity_applied),
            "format_flattened": bool(format_applied),
        },
    )


def checkpoint_selection_score(
    *,
    purged_accuracy: float | None,
    audit_qformer_accuracy: float | None,
    format_accuracy: float | None = None,
    final_mean_loss: float | None = None,
    policy: str,
) -> float | None:
    if policy == "final_only":
        if final_mean_loss is None:
            return None
        return -float(final_mean_loss)
    if policy == "audit_purged":
        if purged_accuracy is None and audit_qformer_accuracy is None:
            return None
        purged_term = float(purged_accuracy) if purged_accuracy is not None else 0.0
        audit_term = float(audit_qformer_accuracy) if audit_qformer_accuracy is not None else 0.0
        loss_term = 0.0 if final_mean_loss is None else min(float(final_mean_loss), 10.0) * 0.01
        return purged_term + (0.25 * audit_term) - loss_term
    if policy == "audit_purged_format":
        if purged_accuracy is None and audit_qformer_accuracy is None and format_accuracy is None:
            return None
        purged_term = float(purged_accuracy) if purged_accuracy is not None else 0.0
        audit_term = float(audit_qformer_accuracy) if audit_qformer_accuracy is not None else 0.0
        format_term = float(format_accuracy) if format_accuracy is not None else 0.0
        loss_term = 0.0 if final_mean_loss is None else min(float(final_mean_loss), 10.0) * 0.01
        return purged_term + (0.35 * format_term) + (0.20 * audit_term) - loss_term
    raise ValueError(f"unsupported checkpoint selection policy: {policy}")


def checkpoint_selection_key(record: dict[str, Any], policy: str) -> tuple[float, ...]:
    purged_accuracy = _safe_float(record.get("purged_accuracy"))
    audit_qformer_accuracy = _safe_float(record.get("audit_qformer_accuracy"))
    format_accuracy = _safe_float(record.get("format_accuracy"))
    final_mean_loss = _safe_float(record.get("final_mean_loss"))
    score = checkpoint_selection_score(
        purged_accuracy=purged_accuracy,
        audit_qformer_accuracy=audit_qformer_accuracy,
        format_accuracy=format_accuracy,
        final_mean_loss=final_mean_loss,
        policy=policy,
    )
    if policy == "final_only":
        return (
            _neg_inf_if_none(score),
            _neg_inf_if_none(-final_mean_loss if final_mean_loss is not None else None),
        )
    if policy == "audit_purged":
        return (
            _neg_inf_if_none(score),
            _neg_inf_if_none(purged_accuracy),
            _neg_inf_if_none(audit_qformer_accuracy),
            _neg_inf_if_none(-final_mean_loss if final_mean_loss is not None else None),
        )
    if policy == "audit_purged_format":
        return (
            _neg_inf_if_none(score),
            _neg_inf_if_none(purged_accuracy),
            _neg_inf_if_none(format_accuracy),
            _neg_inf_if_none(audit_qformer_accuracy),
            _neg_inf_if_none(-final_mean_loss if final_mean_loss is not None else None),
        )
    raise ValueError(f"unsupported checkpoint selection policy: {policy}")


def select_best_checkpoint(records: list[dict[str, Any]], policy: str) -> dict[str, Any] | None:
    best_record: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for record in records:
        current_key = checkpoint_selection_key(record, policy)
        if best_key is None or current_key > best_key:
            best_record = record
            best_key = current_key
    return best_record


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _neg_inf_if_none(value: float | None) -> float:
    if value is None:
        return -math.inf
    return float(value)
