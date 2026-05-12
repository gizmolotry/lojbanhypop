from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Sequence


CONTRACT_SCHEMA_VERSION = 2
HASH_SIZE_LIMIT_BYTES = 512 * 1024 * 1024
INPUT_PATH_FLAGS = {
    "--audit-data-path",
    "--bridge-path",
    "--data-path",
    "--dataset-path",
    "--eval-data-path",
    "--typed-physics-config",
}


def command_contract(cmd: Sequence[object], repo_root: Path | None = None) -> dict[str, object]:
    """Build a stable fingerprint for a generated report command."""
    command = [str(part).replace("\\", "/") for part in cmd]
    payload: dict[str, object] = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "command": command,
        "inputs": _command_inputs(command, repo_root),
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def command_contract_path(report_path: Path) -> Path:
    return Path(str(report_path) + ".cmd.json")


def command_contract_matches(report_path: Path, cmd: Sequence[object], repo_root: Path | None = None) -> bool:
    report_path = Path(report_path)
    if not report_path.exists():
        return False
    sidecar_path = command_contract_path(report_path)
    if not sidecar_path.exists():
        return False
    try:
        existing = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected = command_contract(cmd, repo_root)
    return (
        existing.get("schema_version") == expected["schema_version"]
        and existing.get("sha256") == expected["sha256"]
        and existing.get("command") == expected["command"]
    )


def write_command_contract(report_path: Path, cmd: Sequence[object], repo_root: Path | None = None) -> None:
    sidecar_path = command_contract_path(Path(report_path))
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(command_contract(cmd, repo_root), indent=2), encoding="utf-8")


def run_if_needed(report_path: Path, cmd: Sequence[object], repo_root: Path) -> bool:
    """Run a command unless the report and its command contract already match."""
    report_path = Path(report_path)
    repo_root = Path(repo_root)
    if command_contract_matches(report_path, cmd, repo_root):
        return False
    if report_path.exists():
        print(f"Report command contract missing or stale; rerunning: {report_path}")
    subprocess.run([str(part) for part in cmd], cwd=str(repo_root), check=True)
    if report_path.exists():
        write_command_contract(report_path, cmd, repo_root)
    return True


def _command_inputs(command: Sequence[str], repo_root: Path | None) -> list[dict[str, object]]:
    base_dir = Path(repo_root) if repo_root is not None else Path.cwd()
    inputs: list[dict[str, object]] = []
    if len(command) > 1 and str(command[1]).endswith(".py"):
        inputs.append({"role": "script", **_fingerprint_path(command[1], base_dir)})
    idx = 0
    while idx < len(command) - 1:
        flag = str(command[idx])
        if flag in INPUT_PATH_FLAGS:
            inputs.append({"role": flag.removeprefix("--"), **_fingerprint_path(command[idx + 1], base_dir)})
            idx += 2
            continue
        idx += 1
    return inputs


def _fingerprint_path(raw_path: str, base_dir: Path) -> dict[str, object]:
    path = Path(raw_path)
    resolved = path if path.is_absolute() else base_dir / path
    normalized = str(path).replace("\\", "/")
    if not resolved.exists():
        return {"path": normalized, "exists": False}
    if resolved.is_dir():
        return {
            "path": normalized,
            "exists": True,
            "kind": "directory",
        }
    stat = resolved.stat()
    row: dict[str, object] = {
        "path": normalized,
        "exists": True,
        "kind": "file",
        "size": int(stat.st_size),
    }
    if stat.st_size <= HASH_SIZE_LIMIT_BYTES:
        row["sha256"] = _sha256_file(resolved)
    else:
        row["sha256"] = None
        row["hash_skipped_reason"] = "file_too_large"
    return row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
