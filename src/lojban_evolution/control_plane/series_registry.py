from __future__ import annotations

import fnmatch
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REGISTRY_PATH = REPO_ROOT / "configs" / "script_surface_registry.json"


@dataclass(frozen=True)
class SurfaceClassification:
    path: str
    series: str
    status: str
    kind: str
    bucket: str
    owner: str
    replacement: str
    note: str


def load_script_surface_registry(path: str | Path | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    payload = json.loads(registry_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Script surface registry must be a JSON object: {registry_path}")
    return payload


def _as_posix(path: str | Path) -> str:
    return str(path).replace("\\", "/").lstrip("./")


def classify_surface_path(path: str | Path, registry: dict[str, Any] | None = None) -> SurfaceClassification:
    payload = registry or load_script_surface_registry()
    rel = _as_posix(path)
    for rule in payload.get("path_rules", []):
        patterns = rule.get("globs", [])
        if isinstance(patterns, str):
            patterns = [patterns]
        if any(fnmatch.fnmatch(rel, str(pattern)) for pattern in patterns):
            return SurfaceClassification(
                path=rel,
                series=str(rule.get("series", "shared")),
                status=str(rule.get("status", "tracked")),
                kind=str(rule.get("kind", "unknown")),
                bucket=str(rule.get("bucket", "")),
                owner=str(rule.get("owner", "unowned")),
                replacement=str(rule.get("replacement", "")),
                note=str(rule.get("note", "")),
            )
    return SurfaceClassification(
        path=rel,
        series="unclassified",
        status="unclassified",
        kind="unknown",
        bucket="",
        owner="unowned",
        replacement="",
        note="No script surface registry rule matched this path.",
    )


def series_order(registry: dict[str, Any] | None = None) -> list[str]:
    payload = registry or load_script_surface_registry()
    return [str(item) for item in payload.get("series_order", [])]


def known_series(registry: dict[str, Any] | None = None) -> set[str]:
    payload = registry or load_script_surface_registry()
    return set(series_order(payload)) | set(payload.get("series", {}).keys())
