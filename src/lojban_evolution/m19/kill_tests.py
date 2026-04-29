from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .integrity import (
    build_train_pair_index,
    load_jsonl_rows,
    split_eval_rows_by_overlap,
    write_jsonl,
)


_TITLE_TOKEN = re.compile(r"\b[A-Z][a-z]+\b")
_NUMBER_TOKEN = re.compile(r"\b\d+\b")
_ENTITY_EXCLUDE = {
    "A",
    "An",
    "Answer",
    "Can",
    "Could",
    "Did",
    "Do",
    "Does",
    "Final",
    "If",
    "In",
    "Is",
    "No",
    "Question",
    "Should",
    "Solve",
    "The",
    "Then",
    "True",
    "What",
    "When",
    "Where",
    "Which",
    "Who",
    "Would",
    "Yes",
}
_RENAME_POOL = [
    "Avery",
    "Blake",
    "Casey",
    "Devon",
    "Emery",
    "Finley",
    "Gray",
    "Harper",
    "Indigo",
    "Jordan",
    "Kai",
    "Logan",
    "Morgan",
    "Nova",
    "Parker",
    "Quinn",
    "Reese",
    "Sage",
    "Taylor",
    "Vale",
]


def anonymize_entities_text(text: str, mapping: dict[str, str] | None = None, prefix: str = "entity") -> str:
    active_mapping = mapping if mapping is not None else {}
    counter = len(active_mapping) + 1

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        token = match.group(0)
        if token in _ENTITY_EXCLUDE:
            return token
        if token not in active_mapping:
            active_mapping[token] = f"{prefix}{counter}"
            counter += 1
        return active_mapping[token]

    return _TITLE_TOKEN.sub(repl, str(text or ""))


def rename_entities_text(text: str, mapping: dict[str, str] | None = None) -> str:
    active_mapping = mapping if mapping is not None else {}

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        if token in _ENTITY_EXCLUDE:
            return token
        if token not in active_mapping:
            active_mapping[token] = _RENAME_POOL[len(active_mapping) % len(_RENAME_POOL)]
        return active_mapping[token]

    return _TITLE_TOKEN.sub(repl, str(text or ""))


def flatten_format_text(text: str) -> str:
    flattened = str(text or "").lower()
    flattened = re.sub(r"[“”\"']", "", flattened)
    flattened = re.sub(r"[,:;]", " ", flattened)
    flattened = re.sub(r"\s+", " ", flattened)
    return flattened.strip()


def normalize_numeric_surface(text: str) -> str:
    return _NUMBER_TOKEN.sub("0", str(text or ""))


def build_entity_anonymized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        mapping: dict[str, str] = {}
        item["prompt"] = anonymize_entities_text(str(row.get("prompt") or ""), mapping=mapping)
        item["answer"] = anonymize_entities_text(str(row.get("answer") or ""), mapping=mapping)
        item["entity_mapping"] = mapping
        output.append(item)
    return output


def build_entity_renamed_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        mapping: dict[str, str] = {}
        item["prompt"] = rename_entities_text(str(row.get("prompt") or ""), mapping=mapping)
        item["answer"] = rename_entities_text(str(row.get("answer") or ""), mapping=mapping)
        item["entity_mapping"] = mapping
        output.append(item)
    return output


def build_format_flattened_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["prompt"] = flatten_format_text(str(row.get("prompt") or ""))
        item["answer"] = flatten_format_text(str(row.get("answer") or ""))
        output.append(item)
    return output


def build_numeric_normalized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["prompt"] = normalize_numeric_surface(str(row.get("prompt") or ""))
        item["answer"] = normalize_numeric_surface(str(row.get("answer") or ""))
        output.append(item)
    return output


def build_purged_eval_rows(train_path: Path, eval_path: Path, eval_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = load_jsonl_rows(Path(train_path))
    eval_rows = load_jsonl_rows(Path(eval_path))[: int(eval_size)]
    train_pairs = build_train_pair_index(train_rows)
    overlap_rows, purged_rows = split_eval_rows_by_overlap(eval_rows, train_pairs)
    return overlap_rows, purged_rows


def compute_kill_test_metrics(
    *,
    purged_report: dict[str, Any] | None,
    masked_report: dict[str, Any] | None,
    entity_report: dict[str, Any] | None,
    entity_renamed_report: dict[str, Any] | None = None,
    format_report: dict[str, Any] | None,
    numeric_report: dict[str, Any] | None,
) -> dict[str, Any]:
    purged_acc = _strict_accuracy(purged_report)
    entity_acc = _strict_accuracy(entity_report)
    entity_renamed_acc = _strict_accuracy(entity_renamed_report)
    format_acc = _strict_accuracy(format_report)
    numeric_acc = _strict_accuracy(numeric_report)
    masked_acc = _strict_accuracy(masked_report)

    return {
        "purged_accuracy": purged_acc,
        "entity_accuracy": entity_acc,
        "entity_lift_vs_random": _lift_vs_random(entity_report),
        "entity_drop_vs_purged": _safe_delta(purged_acc, entity_acc),
        "entity_renamed_accuracy": entity_renamed_acc,
        "entity_renamed_lift_vs_random": _lift_vs_random(entity_renamed_report),
        "entity_renamed_drop_vs_purged": _safe_delta(purged_acc, entity_renamed_acc),
        "format_accuracy": format_acc,
        "format_lift_vs_random": _lift_vs_random(format_report),
        "format_drop_vs_purged": _safe_delta(purged_acc, format_acc),
        "numeric_accuracy": numeric_acc,
        "numeric_lift_vs_random": _lift_vs_random(numeric_report),
        "numeric_drop_vs_purged": _safe_delta(purged_acc, numeric_acc),
        "masked_accuracy": masked_acc,
        "masked_lift_vs_random": _lift_vs_random(masked_report),
        "masked_drop_vs_purged": _safe_delta(purged_acc, masked_acc),
        "kill_entity_flag": bool(entity_acc is not None and purged_acc is not None and entity_acc < purged_acc - 0.15),
        "kill_entity_renamed_flag": bool(
            entity_renamed_acc is not None and purged_acc is not None and entity_renamed_acc < purged_acc - 0.15
        ),
        "kill_format_flag": bool(format_acc is not None and purged_acc is not None and format_acc < purged_acc - 0.10),
        "kill_numeric_flag": bool(numeric_acc is not None and purged_acc is not None and numeric_acc < purged_acc - 0.10),
        "kill_mask_flag": bool(masked_acc is not None and purged_acc is not None and masked_acc > purged_acc * 0.5),
    }


def classify_kill_test_status(metrics: dict[str, Any]) -> str:
    flags = [
        bool(metrics.get("kill_entity_flag")),
        bool(metrics.get("kill_entity_renamed_flag")),
        bool(metrics.get("kill_format_flag")),
        bool(metrics.get("kill_numeric_flag")),
        bool(metrics.get("kill_mask_flag")),
    ]
    if any(flags):
        return "mixed"
    return "pass"


def _strict_accuracy(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    metrics = report.get("metrics", {})
    if isinstance(metrics, dict) and "strict_accuracy" in metrics:
        return _safe_float(metrics.get("strict_accuracy"))
    return None


def _lift_vs_random(report: dict[str, Any] | None) -> float | None:
    if not isinstance(report, dict):
        return None
    metrics = report.get("metrics", {})
    if isinstance(metrics, dict) and "lift_vs_random" in metrics:
        return _safe_float(metrics.get("lift_vs_random"))
    return None


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
