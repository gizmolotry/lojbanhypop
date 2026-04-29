from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


QUESTION_PATTERNS = (
    re.compile(r"Question:\s*(.*?)\nFinal answer:", re.IGNORECASE | re.DOTALL),
    re.compile(r"QUESTION:\s*(.*?)\nTRACE:", re.IGNORECASE | re.DOTALL),
)
ANSWER_PATTERNS = (
    re.compile(r"Final answer:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL),
    re.compile(r"ANSWER:\s*(.+?)\s*$", re.IGNORECASE | re.DOTALL),
)


def normalize_overlap_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def extract_training_pair(row: dict[str, Any]) -> tuple[str, str] | None:
    prompt = row.get("prompt")
    answer = row.get("answer")
    if prompt is not None and answer is not None:
        return normalize_overlap_text(str(prompt)), normalize_overlap_text(str(answer))

    text = str(row.get("text") or "")
    if not text:
        return None

    question = None
    for pattern in QUESTION_PATTERNS:
        match = pattern.search(text)
        if match:
            question = match.group(1)
            break

    answer_text = None
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(text)
        if match:
            answer_text = match.group(1)
            break

    if question is None or answer_text is None:
        return None
    return normalize_overlap_text(question), normalize_overlap_text(answer_text)


def load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_train_pair_index(rows: list[dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for row in rows:
        pair = extract_training_pair(row)
        if pair is not None:
            pairs.add(pair)
    return pairs


def split_eval_rows_by_overlap(
    eval_rows: list[dict[str, Any]],
    train_pairs: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overlap_rows: list[dict[str, Any]] = []
    purged_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        pair = (
            normalize_overlap_text(str(row.get("prompt") or "")),
            normalize_overlap_text(str(row.get("answer") or "")),
        )
        if pair in train_pairs:
            overlap_rows.append(row)
        else:
            purged_rows.append(row)
    return overlap_rows, purged_rows


def mask_prompt_text(text: str) -> str:
    text = re.sub(r"\d+", "0", str(text))
    return re.sub(r"[A-Za-z]+", "x", text)


def build_masked_eval_rows(eval_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    masked_rows: list[dict[str, Any]] = []
    for row in eval_rows:
        masked = dict(row)
        masked["prompt"] = mask_prompt_text(str(row.get("prompt") or ""))
        masked_rows.append(masked)
    return masked_rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")
    return path


def compute_integrity_metrics(
    *,
    full_report: dict[str, Any],
    purged_report: dict[str, Any],
    overlap_report: dict[str, Any],
    masked_report: dict[str, Any],
    audit_report: dict[str, Any],
    overlap_size: int,
    purged_size: int,
    eval_size: int,
) -> dict[str, Any]:
    full_metrics = dict(full_report.get("metrics", {}))
    purged_metrics = dict(purged_report.get("metrics", {}))
    overlap_metrics = dict(overlap_report.get("metrics", {}))
    masked_metrics = dict(masked_report.get("metrics", {}))
    audit_headline = dict(audit_report.get("headline", {}))

    purged_accuracy = purged_metrics.get("strict_accuracy")
    overlap_accuracy = overlap_metrics.get("strict_accuracy")
    masked_accuracy = masked_metrics.get("strict_accuracy")
    random_accuracy = purged_metrics.get("random_accuracy")
    scratchpad_accuracy = purged_report.get("results", {}).get("SCRATCHPAD-ONLY", {}).get("accuracy")

    return {
        "full_accuracy": full_metrics.get("strict_accuracy"),
        "full_avg_tokens": full_metrics.get("avg_tokens"),
        "purged_accuracy": purged_accuracy,
        "purged_phrase_accuracy": purged_metrics.get("overall_phrase_accuracy"),
        "purged_avg_tokens": purged_metrics.get("avg_tokens"),
        "purged_lift_vs_base": purged_metrics.get("lift_vs_base"),
        "purged_lift_vs_en_cot": purged_metrics.get("lift_vs_en_cot"),
        "purged_lift_vs_random": purged_metrics.get("lift_vs_random"),
        "purged_lift_vs_scratchpad_only": _safe_delta(purged_accuracy, scratchpad_accuracy),
        "overlap_accuracy": overlap_accuracy,
        "overlap_phrase_accuracy": overlap_metrics.get("overall_phrase_accuracy"),
        "overlap_avg_tokens": overlap_metrics.get("avg_tokens"),
        "masked_accuracy": masked_accuracy,
        "masked_phrase_accuracy": masked_metrics.get("overall_phrase_accuracy"),
        "masked_avg_tokens": masked_metrics.get("avg_tokens"),
        "masked_lift_vs_base": masked_metrics.get("lift_vs_base"),
        "masked_lift_vs_random": masked_metrics.get("lift_vs_random"),
        "overlap_gap": _safe_delta(overlap_accuracy, purged_accuracy),
        "masked_collapse_gap": _safe_delta(purged_accuracy, masked_accuracy),
        "masked_random_gap": _safe_delta(masked_accuracy, random_accuracy),
        "masked_scratchpad_gap": _safe_delta(masked_accuracy, scratchpad_accuracy),
        "leakage_overlap_rate": _safe_ratio(overlap_size, eval_size),
        "purged_coverage_rate": _safe_ratio(purged_size, eval_size),
        "audit_qformer_accuracy": audit_headline.get("qformer_accuracy"),
        "audit_qformer_phrase_accuracy": audit_headline.get("qformer_phrase_accuracy"),
        "audit_lift_vs_base": audit_headline.get("lift_vs_base"),
        "audit_lift_vs_random": audit_headline.get("lift_vs_random"),
        "integrity_overlap_flag": bool(_safe_delta(overlap_accuracy, purged_accuracy) is not None and _safe_delta(overlap_accuracy, purged_accuracy) > 0.05),
        "integrity_mask_flag": bool(masked_accuracy is not None and random_accuracy is not None and masked_accuracy > random_accuracy + 0.05),
        "integrity_audit_flag": bool(audit_headline.get("qformer_accuracy") is not None and audit_headline.get("qformer_accuracy") < 0.5),
    }


def classify_integrity_status(metrics: dict[str, Any]) -> str:
    flags = [
        bool(metrics.get("integrity_overlap_flag")),
        bool(metrics.get("integrity_mask_flag")),
        bool(metrics.get("integrity_audit_flag")),
    ]
    if any(flags):
        return "mixed" if metrics.get("audit_qformer_accuracy", 0.0) >= 0.5 else "fail"
    return "pass"


def _safe_ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator)


def _safe_delta(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)
