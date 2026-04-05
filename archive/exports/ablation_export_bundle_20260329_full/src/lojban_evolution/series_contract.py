from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SERIES_CONTRACT_VERSION = "1.0"


@dataclass(frozen=True)
class SeriesRule:
    series_id: str
    allowed_prefixes: tuple[str, ...]
    description: str


SERIES_RULES: dict[str, SeriesRule] = {
    "J": SeriesRule(
        series_id="J",
        allowed_prefixes=(
            "runs/j_series",
            "artifacts/runs/telemetry/raw",
        ),
        description="J-series invariance/data-generation artifacts.",
    ),
    "L": SeriesRule(
        series_id="L",
        allowed_prefixes=(
            "runs/l_series",
            "artifacts/runs/models/frozen_manifolds",
        ),
        description="L-series lagrangian training artifacts.",
    ),
    "A-G": SeriesRule(
        series_id="A-G",
        allowed_prefixes=(
            "runs/ablation/a_to_g",
            "artifacts/runs/telemetry/raw/ablation/a_to_g",
        ),
        description="A-G benchmark ablation artifacts.",
    ),
    "M": SeriesRule(
        series_id="M",
        allowed_prefixes=(
            "runs/m_series",
            "runs/j_series",
            "runs/l_series/l6_ablation",
            "runs/l_series/m3_plus",
            "runs/l_series/m3_8_diversification",
            "artifacts/runs/telemetry/raw/ablation/hypercube",
        ),
        description="Merged modern-stack artifacts (J/L coupled ablation reporting).",
    ),
}


def _normalize_path(path: str | Path) -> str:
    value = str(path).replace("\\", "/").strip()
    if not value:
        raise ValueError("Output path may not be empty.")
    if ".." in Path(value).parts:
        raise ValueError(f"Output path may not include '..': {path}")
    return value


def _path_matches_prefix(value: str, prefix: str) -> bool:
    return value == prefix or value.startswith(f"{prefix}/")


def assert_output_path_allowed(series_id: str, path: str | Path) -> str:
    sid = str(series_id).strip().upper()
    if sid not in SERIES_RULES:
        raise ValueError(f"Unknown series id '{series_id}'.")
    rule = SERIES_RULES[sid]
    value = _normalize_path(path)
    if value.startswith("s3://"):
        # S3 validation is partition-based in DAG wrappers; local path checks do not apply.
        return value
    if not any(_path_matches_prefix(value, prefix) for prefix in rule.allowed_prefixes):
        raise ValueError(
            f"Output path '{path}' is outside allowed {series_id} prefixes: {list(rule.allowed_prefixes)}"
        )
    return value


def assert_output_path_within_declared_root(path: str | Path, declared_root: str | Path) -> str:
    value = _normalize_path(path)
    root_value = _normalize_path(declared_root)
    if value.startswith("s3://") or root_value.startswith("s3://"):
        return value
    if not _path_matches_prefix(value, root_value):
        raise ValueError(f"Output path '{path}' is outside declared output root '{declared_root}'.")
    return value


def validate_series_outputs(
    series_id: str,
    declared_output_roots: Sequence[str | Path],
    output_paths: Iterable[str | Path],
) -> None:
    roots = [assert_output_path_allowed(series_id, root) for root in declared_output_roots]
    if not roots:
        raise ValueError("At least one declared output root is required.")
    for path in output_paths:
        value = assert_output_path_allowed(series_id, path)
        if value.startswith("s3://"):
            continue
        if not any(_path_matches_prefix(value, root) for root in roots):
            raise ValueError(
                f"Output path '{path}' is allowed for series {series_id} but outside declared roots {roots}."
            )


def series_metadata(series_id: str, track: str, script: str) -> dict[str, str]:
    sid = str(series_id).strip().upper()
    if sid not in SERIES_RULES:
        raise ValueError(f"Unknown series id '{series_id}'.")
    return {
        "series_contract_version": SERIES_CONTRACT_VERSION,
        "series_id": sid,
        "track": str(track),
        "script": str(script),
    }


def lineage_metadata(
    mode: str,
    *,
    checkpoint_in: Any = None,
    checkpoint_out: Any = None,
    dataset_profile: str | None = None,
    difficulty_tier: str | None = None,
) -> dict[str, Any]:
    lineage_mode = str(mode).strip().lower()
    if lineage_mode not in {"train", "eval_only"}:
        raise ValueError(f"Unsupported lineage mode '{mode}'.")
    return {
        "mode": lineage_mode,
        "checkpoint_in": checkpoint_in,
        "checkpoint_out": checkpoint_out,
        "dataset_profile": None if dataset_profile is None else str(dataset_profile),
        "difficulty_tier": None if difficulty_tier is None else str(difficulty_tier),
    }


def validate_manifest_series(series_id: str, output_paths: Iterable[str | Path]) -> None:
    sid = str(series_id).strip().upper()
    if sid not in SERIES_RULES:
        raise ValueError(f"Unknown series id '{series_id}'.")
    validate_series_outputs(series_id, SERIES_RULES[sid].allowed_prefixes, output_paths)


def validate_baseline_manifest(
    path: str | Path,
    *,
    series_id: str,
    require_upstream_best: bool = True,
    require_m_base: bool = True,
) -> dict[str, Any]:
    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"baseline_manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline_manifest must be a JSON object")

    expected_sid = str(series_id).strip().upper()
    actual_sid = str(payload.get("series_id", "")).strip().upper()
    if actual_sid != expected_sid:
        raise ValueError(f"baseline_manifest.series_id must be '{expected_sid}'")
    if not str(payload.get("baseline_id", "")).strip():
        raise ValueError("baseline_manifest.baseline_id is required")

    if require_upstream_best:
        upstream = payload.get("upstream_best", {})
        if not isinstance(upstream, dict):
            raise ValueError("baseline_manifest.upstream_best must be an object")
        if not str(upstream.get("j_series", "")).strip() or not str(upstream.get("l_series", "")).strip():
            raise ValueError("baseline_manifest.upstream_best must declare j_series and l_series")

    if require_m_base:
        m_base = payload.get("m_base", {})
        if not isinstance(m_base, dict):
            raise ValueError("baseline_manifest.m_base must be an object")
        required = ("dataset", "constraints", "identity_reg", "curriculum", "optimizer")
        missing = [key for key in required if not str(m_base.get(key, "")).strip()]
        if missing:
            raise ValueError(f"baseline_manifest.m_base missing required keys: {missing}")

    return payload
