from __future__ import annotations

import os
import re
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


def validate_output_partition(output_dir: str, required_partition: str) -> str:
    value = validate_output_dir(output_dir)
    partition = required_partition.strip().replace("\\", "/").strip("/")
    if not partition:
        raise ValueError("required_partition cannot be empty")

    normalized = value.replace("\\", "/").rstrip("/")
    if normalized == partition or normalized.endswith(f"/{partition}") or f"/{partition}/" in f"{normalized}/":
        return value

    raise ValueError(
        f"output_dir '{value}' must target the '{partition}' artifact partition "
        "(for example s3://<bucket>/<prefix>/{partition}/...)"
    )


def sanitize_run_id(run_id: str, fallback: str = "manual") -> str:
    raw = str(run_id).strip() or fallback
    normalized = raw.replace(":", "_")
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", normalized):
        raise ValueError(
            "run_id must match ^[A-Za-z0-9._-]{1,64}$ and may not contain path separators or traversal tokens."
        )
    return normalized


def validate_input_artifact(input_artifact: str, required_partition: str) -> str:
    value = str(input_artifact).strip()
    if not value:
        raise ValueError("input_artifact cannot be empty")
    partition = required_partition.strip().replace("\\", "/").strip("/")
    normalized = value.replace("\\", "/").rstrip("/")
    in_partition = (
        normalized == partition or normalized.endswith(f"/{partition}") or f"/{partition}/" in f"{normalized}/"
    )
    if normalized.startswith("s3://"):
        if not in_partition:
            raise ValueError(f"input_artifact '{value}' must be under partition '{partition}'")
        return value

    path_obj = Path(value)
    if path_obj.is_absolute() and os.environ.get("LOJBAN_ALLOW_ABSOLUTE_INPUT_ARTIFACT", "0") != "1":
        raise ValueError("absolute input_artifact paths are not allowed unless LOJBAN_ALLOW_ABSOLUTE_INPUT_ARTIFACT=1")
    if ".." in path_obj.parts:
        raise ValueError("input_artifact may not include path traversal ('..')")
    if not in_partition:
        raise ValueError(f"input_artifact '{value}' must be under partition '{partition}'")
    return value


def validate_distribution_json_path(path_value: str) -> str:
    value = str(path_value).strip()
    if not value:
        raise ValueError("variable_token_distribution_json cannot be empty")
    if value.startswith("s3://"):
        raise ValueError("variable_token_distribution_json must be a local repo path under docs/")
    normalized = value.replace("\\", "/")
    if not normalized.startswith("docs/"):
        raise ValueError("variable_token_distribution_json must be under docs/")
    if not normalized.endswith(".json"):
        raise ValueError("variable_token_distribution_json must point to a .json file")
    if ".." in Path(value).parts:
        raise ValueError("variable_token_distribution_json may not include path traversal ('..')")
    return value


def validate_baseline_manifest_path(path_value: str) -> str:
    value = str(path_value).strip()
    if not value:
        raise ValueError("baseline_manifest cannot be empty")
    if value.startswith("s3://"):
        raise ValueError("baseline_manifest must be a local repo path under docs/")
    normalized = value.replace("\\", "/")
    if not normalized.startswith("docs/"):
        raise ValueError("baseline_manifest must be under docs/")
    if not normalized.endswith(".json"):
        raise ValueError("baseline_manifest must point to a .json file")
    if ".." in Path(value).parts:
        raise ValueError("baseline_manifest may not include path traversal ('..')")
    manifest_path = _repo_root() / Path(normalized)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"baseline_manifest does not exist: {manifest_path}")
    return value


def merge_conf(defaults: Mapping[str, object], dag_conf: Mapping[str, object] | None) -> dict[str, object]:
    merged = dict(defaults)
    if dag_conf is None:
        return merged
    if not isinstance(dag_conf, Mapping):
        raise TypeError("dag_run.conf must be a mapping of config keys to values")

    unknown_keys = sorted(str(key) for key in dag_conf.keys() if key not in merged)
    if unknown_keys:
        allowed_keys = ", ".join(sorted(str(key) for key in merged))
        raise ValueError(
            "Unknown dag_run.conf keys: "
            f"{unknown_keys}. Allowed keys: [{allowed_keys}]"
        )

    for key, value in dag_conf.items():
        if value is not None:
            merged[key] = value
    return merged
