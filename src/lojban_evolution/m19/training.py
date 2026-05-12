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


def build_surface_consistency_variants(
    question: str,
    answer: str,
    *,
    include_entity_rename: bool = False,
    include_format_flatten: bool = False,
    include_combined: bool = False,
) -> list[dict[str, str]]:
    variants: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add_variant(kind: str, variant_question: str, variant_answer: str) -> None:
        key = (variant_question, variant_answer)
        if key == (question, answer) or key in seen:
            return
        seen.add(key)
        variants.append(
            {
                "kind": kind,
                "question": variant_question,
                "answer": variant_answer,
            }
        )

    if include_entity_rename:
        mapping: dict[str, str] = {}
        add_variant(
            "entity_renamed",
            rename_entities_text(question, mapping=mapping),
            rename_entities_text(answer, mapping=mapping),
        )
    if include_format_flatten:
        add_variant(
            "format_flattened",
            flatten_format_text(question),
            flatten_format_text(answer),
        )
    if include_combined:
        mapping = {}
        renamed_question = rename_entities_text(question, mapping=mapping)
        renamed_answer = rename_entities_text(answer, mapping=mapping)
        add_variant(
            "entity_renamed_format_flattened",
            flatten_format_text(renamed_question),
            flatten_format_text(renamed_answer),
        )

    return variants


def checkpoint_selection_score(
    *,
    purged_accuracy: float | None,
    audit_qformer_accuracy: float | None,
    format_accuracy: float | None = None,
    entity_accuracy: float | None = None,
    entity_renamed_accuracy: float | None = None,
    numeric_accuracy: float | None = None,
    arity_violation_rate: float | None = None,
    masked_pointer_zero_rate: float | None = None,
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
    if policy == "audit_purged_format_arity":
        if (
            purged_accuracy is None
            and audit_qformer_accuracy is None
            and format_accuracy is None
            and entity_accuracy is None
            and entity_renamed_accuracy is None
            and numeric_accuracy is None
            and arity_violation_rate is None
            and masked_pointer_zero_rate is None
        ):
            return None
        purged_term = float(purged_accuracy) if purged_accuracy is not None else 0.0
        audit_term = float(audit_qformer_accuracy) if audit_qformer_accuracy is not None else 0.0
        format_term = float(format_accuracy) if format_accuracy is not None else 0.0
        entity_accuracy_term = float(entity_accuracy) if entity_accuracy is not None else 0.0
        entity_term = float(entity_renamed_accuracy) if entity_renamed_accuracy is not None else 0.0
        numeric_term = float(numeric_accuracy) if numeric_accuracy is not None else 0.0
        arity_term = 1.0 - float(arity_violation_rate) if arity_violation_rate is not None else 0.0
        pointer_term = float(masked_pointer_zero_rate) if masked_pointer_zero_rate is not None else 0.0
        loss_term = 0.0 if final_mean_loss is None else min(float(final_mean_loss), 10.0) * 0.01
        return purged_term + (0.16 * format_term) + (0.10 * entity_accuracy_term) + (0.10 * entity_term) + (0.10 * numeric_term) + (0.15 * audit_term) + (0.20 * arity_term) + (0.10 * pointer_term) - loss_term
    if policy == "audit_purged_surface_arity_weakseed":
        if (
            purged_accuracy is None
            and audit_qformer_accuracy is None
            and format_accuracy is None
            and entity_accuracy is None
            and entity_renamed_accuracy is None
            and numeric_accuracy is None
            and arity_violation_rate is None
            and masked_pointer_zero_rate is None
        ):
            return None
        purged_term = float(purged_accuracy) if purged_accuracy is not None else 0.0
        audit_term = float(audit_qformer_accuracy) if audit_qformer_accuracy is not None else 0.0
        format_term = float(format_accuracy) if format_accuracy is not None else 0.0
        entity_accuracy_term = float(entity_accuracy) if entity_accuracy is not None else 0.0
        entity_term = float(entity_renamed_accuracy) if entity_renamed_accuracy is not None else 0.0
        numeric_term = float(numeric_accuracy) if numeric_accuracy is not None else 0.0
        arity_term = 1.0 - float(arity_violation_rate) if arity_violation_rate is not None else 0.0
        pointer_term = float(masked_pointer_zero_rate) if masked_pointer_zero_rate is not None else 0.0
        loss_term = 0.0 if final_mean_loss is None else min(float(final_mean_loss), 10.0) * 0.01
        surfaces = [
            value
            for value in (
                _safe_float(format_accuracy),
                _safe_float(entity_accuracy),
                _safe_float(entity_renamed_accuracy),
                _safe_float(numeric_accuracy),
            )
            if value is not None
        ]
        weakest_surface = min(surfaces) if surfaces else 0.0
        robustness_mean = (sum(surfaces) / len(surfaces)) if surfaces else 0.0
        purged_to_worst_gap = max(0.0, purged_term - weakest_surface)
        weak_seed_pressure = _weak_seed_pressure(
            purged_accuracy=purged_accuracy,
            audit_qformer_accuracy=audit_qformer_accuracy,
        )
        surface_weight = 0.20 + (0.35 * weak_seed_pressure)
        weakest_surface_weight = 0.10 + (0.30 * weak_seed_pressure)
        worst_gap_penalty_weight = 0.10 + (0.30 * weak_seed_pressure)
        arity_weight = 0.12 + (0.12 * weak_seed_pressure)
        pointer_weight = 0.08 + (0.04 * weak_seed_pressure)
        audit_weight = 0.18 - (0.06 * weak_seed_pressure)
        return (
            purged_term
            + (surface_weight * robustness_mean)
            + (weakest_surface_weight * weakest_surface)
            + (audit_weight * audit_term)
            + (arity_weight * arity_term)
            + (pointer_weight * pointer_term)
            - (worst_gap_penalty_weight * purged_to_worst_gap)
            - loss_term
        )
    raise ValueError(f"unsupported checkpoint selection policy: {policy}")


def checkpoint_selection_key(record: dict[str, Any], policy: str) -> tuple[float, ...]:
    purged_accuracy = _safe_float(record.get("purged_accuracy"))
    audit_qformer_accuracy = _safe_float(record.get("audit_qformer_accuracy"))
    format_accuracy = _safe_float(record.get("format_accuracy"))
    entity_accuracy = _safe_float(record.get("entity_accuracy"))
    entity_renamed_accuracy = _safe_float(record.get("entity_renamed_accuracy"))
    numeric_accuracy = _safe_float(record.get("numeric_accuracy"))
    arity_violation_rate = _safe_float(record.get("arity_violation_rate"))
    masked_pointer_zero_rate = _safe_float(record.get("masked_pointer_zero_rate"))
    final_mean_loss = _safe_float(record.get("final_mean_loss"))
    score = checkpoint_selection_score(
        purged_accuracy=purged_accuracy,
        audit_qformer_accuracy=audit_qformer_accuracy,
        format_accuracy=format_accuracy,
        entity_accuracy=entity_accuracy,
        entity_renamed_accuracy=entity_renamed_accuracy,
        numeric_accuracy=numeric_accuracy,
        arity_violation_rate=arity_violation_rate,
        masked_pointer_zero_rate=masked_pointer_zero_rate,
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
    if policy == "audit_purged_format_arity":
        return (
            _neg_inf_if_none(score),
            _neg_inf_if_none(purged_accuracy),
            _neg_inf_if_none(entity_accuracy),
            _neg_inf_if_none(entity_renamed_accuracy),
            _neg_inf_if_none(numeric_accuracy),
            _neg_inf_if_none(1.0 - arity_violation_rate if arity_violation_rate is not None else None),
            _neg_inf_if_none(masked_pointer_zero_rate),
            _neg_inf_if_none(format_accuracy),
            _neg_inf_if_none(audit_qformer_accuracy),
            _neg_inf_if_none(-final_mean_loss if final_mean_loss is not None else None),
        )
    if policy == "audit_purged_surface_arity_weakseed":
        weakest_surface = None
        surfaces = [
            value
            for value in (format_accuracy, entity_accuracy, entity_renamed_accuracy, numeric_accuracy)
            if value is not None
        ]
        if surfaces:
            weakest_surface = min(surfaces)
        return (
            _neg_inf_if_none(score),
            _neg_inf_if_none(weakest_surface),
            _neg_inf_if_none(entity_accuracy),
            _neg_inf_if_none(entity_renamed_accuracy),
            _neg_inf_if_none(format_accuracy),
            _neg_inf_if_none(numeric_accuracy),
            _neg_inf_if_none(purged_accuracy),
            _neg_inf_if_none(1.0 - arity_violation_rate if arity_violation_rate is not None else None),
            _neg_inf_if_none(masked_pointer_zero_rate),
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


def _weak_seed_pressure(
    *,
    purged_accuracy: float | None,
    audit_qformer_accuracy: float | None,
) -> float:
    purged_term = 0.0
    audit_term = 0.0
    if purged_accuracy is not None:
        purged_term = max(0.0, min(1.0, (0.55 - float(purged_accuracy)) / 0.20))
    if audit_qformer_accuracy is not None:
        audit_term = max(0.0, min(1.0, (0.75 - float(audit_qformer_accuracy)) / 0.35))
    return max(purged_term, audit_term)
