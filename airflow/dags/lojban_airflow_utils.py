from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse


def _repo_root() -> Path:
    allow_overrides = os.environ.get("LOJBAN_ALLOW_ENV_OVERRIDES", "0") == "1"
    override = os.environ.get("LOJBAN_REPO_ROOT")
    if allow_overrides and override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def run_repo_script(script_relpath: str, cli_args: list[str]) -> None:
    repo_root = _repo_root()
    script_path = repo_root / script_relpath
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")

    allow_overrides = os.environ.get("LOJBAN_ALLOW_ENV_OVERRIDES", "0") == "1"
    python_bin = os.environ.get("LOJBAN_PYTHON_BIN", sys.executable) if allow_overrides else sys.executable
    env = os.environ.copy()

    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    repo_src = str(repo_root / "src")
    env["PYTHONPATH"] = (
        f"{repo_src}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else repo_src
    )

    cmd = [python_bin, script_relpath, *cli_args]
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def validate_output_dir(output_dir: str) -> str:
    value = str(output_dir).strip()
    if not value:
        raise ValueError("output_dir cannot be empty")

    if value.startswith("s3://"):
        allowed = [x.strip() for x in os.environ.get("LOJBAN_ALLOWED_S3_PREFIXES", "").split(",") if x.strip()]
        if allowed and not any(value.startswith(prefix) for prefix in allowed):
            raise ValueError(f"output_dir '{value}' is not under allowed S3 prefixes: {allowed}")
        parsed = urlparse(value)
        if not parsed.netloc:
            raise ValueError(f"output_dir '{value}' must include an S3 bucket")
        return value

    path_obj = Path(value)
    if path_obj.is_absolute() and os.environ.get("LOJBAN_ALLOW_ABSOLUTE_OUTPUT_DIR", "0") != "1":
        raise ValueError("absolute output_dir paths are not allowed unless LOJBAN_ALLOW_ABSOLUTE_OUTPUT_DIR=1")
    if ".." in path_obj.parts:
        raise ValueError("output_dir may not include path traversal ('..')")

    allowed_local = [
        x.strip().replace("\\", "/")
        for x in os.environ.get("LOJBAN_ALLOWED_LOCAL_OUTPUT_ROOTS", "artifacts/runs,runs").split(",")
        if x.strip()
    ]
    normalized = value.replace("\\", "/")
    if not any(normalized == root or normalized.startswith(f"{root}/") for root in allowed_local):
        raise ValueError(f"output_dir '{value}' is not under allowed local roots: {allowed_local}")
    return value


def merge_conf(defaults: Mapping[str, object], dag_conf: Mapping[str, object] | None) -> dict[str, object]:
    merged = dict(defaults)
    if dag_conf:
        for key, value in dag_conf.items():
            if key in merged and value is not None:
                merged[key] = value
    return merged
