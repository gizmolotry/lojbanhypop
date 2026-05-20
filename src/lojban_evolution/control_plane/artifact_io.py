from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


def path_allowed_for_discovery(path: Path) -> bool:
    bad_parts = {"__pycache__"}
    for part in path.parts:
        lowered = part.lower()
        if lowered in bad_parts:
            return False
        if lowered.startswith("test_"):
            return False
    return True


def read_json_optional(path: Path | None, *, swallow_errors: bool = False) -> dict[str, Any] | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        if swallow_errors:
            return None
        raise
    return payload if isinstance(payload, dict) else None


def latest_named_manifest(
    root: Path,
    filename: str,
    *,
    recursive: bool = True,
    newest_first: bool = True,
    path_filter: Callable[[Path], bool] | None = path_allowed_for_discovery,
) -> Path | None:
    if not root.exists():
        return None
    iterator = root.rglob(filename) if recursive else root.glob(f"*/{filename}")
    matches = [path for path in iterator if path_filter is None or path_filter(path)]
    if not matches:
        return None
    matches.sort(key=lambda item: item.stat().st_mtime, reverse=bool(newest_first))
    return matches[0]


def latest_json(
    root: Path,
    preferred_names: list[str] | None = None,
    *,
    path_filter: Callable[[Path], bool] | None = path_allowed_for_discovery,
) -> Path | None:
    if not root.exists():
        return None
    preferred_names = preferred_names or []
    candidates = [path for path in root.rglob("*.json") if path_filter is None or path_filter(path)]
    if not candidates:
        return None
    ranked: list[tuple[int, float, Path]] = []
    for path in candidates:
        score = 0
        if path.name in preferred_names:
            score += 100
        if "manifest" in path.name:
            score += 20
        if "report" in path.name:
            score += 20
        if "summary" in path.name:
            score += 10
        ranked.append((score, path.stat().st_mtime, path))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return ranked[0][2]


def repo_relative_or_string(path: Path | None, repo_root: Path) -> str | None:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def repo_string(path: Path | None, repo_root: Path) -> str | None:
    return repo_relative_or_string(path, repo_root)
