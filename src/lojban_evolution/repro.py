from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .experiment import Problem


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_git_commit(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None
    return out or None


def dataset_fingerprint(problems: Sequence[Problem]) -> str:
    h = hashlib.sha256()
    for problem in problems:
        row = {
            "problem_id": problem.problem_id,
            "prompt": problem.prompt,
            "answer": problem.answer,
            "trace": list(problem.trace),
        }
        h.update(json.dumps(row, sort_keys=True, ensure_ascii=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def write_run_manifest(path: Path, payload: Mapping[str, Any]) -> None:
    base = {
        "schema_version": "1.0",
        "generated_utc": utc_now_iso(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    out = dict(base)
    out.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
