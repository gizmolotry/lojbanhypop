from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any


sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from lojban_evolution.series_contract import lineage_metadata, series_metadata, validate_series_outputs  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "ablation_test_matrix.json"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "runs" / "telemetry" / "raw" / "ablation" / "hypercube" / "ablation_test_matrix"
REPORT_NAME = "ablation_test_matrix_manifest.json"
SUMMARY_NAME = "ablation_test_matrix_summary.md"


class TestMatrixError(ValueError):
    pass


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    raw = str(value or "").strip()
    slug = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return slug.strip("._-") or f"ablation_test_matrix_{_timestamp()}"


def _repo_rel(path: str | Path) -> str:
    resolved = Path(path)
    try:
        return resolved.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path).replace("\\", "/")


def _resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return REPO_ROOT / candidate


def _parse_csv(values: list[str] | None) -> list[str]:
    if not values:
        return []
    out: list[str] = []
    for value in values:
        out.extend(item.strip() for item in str(value).split(",") if item.strip())
    return out


def _validate_flat_matrix_config(config: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> None:
    errors: list[str] = []
    for section_name in ("lanes", "families"):
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            errors.append(f"config.{section_name} must be an object")
            continue
        for key, value in section.items():
            if not isinstance(value, dict):
                errors.append(f"{section_name}.{key} must be an object")
                continue
            for raw_path in value.get("pytest_files", []):
                rel_path = str(raw_path).replace("\\", "/")
                if not (repo_root / rel_path).exists():
                    errors.append(f"{section_name}.{key} references missing pytest file: {rel_path}")
    if errors:
        raise TestMatrixError("; ".join(errors))


def _config_is_group_matrix(config: dict[str, Any]) -> bool:
    return isinstance(config.get("test_groups"), list)


def load_test_matrix_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config_path = _resolve_repo_path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TestMatrixError("ablation test matrix config must be a JSON object")
    if _config_is_group_matrix(payload):
        validate_test_matrix_config(payload)
    else:
        _validate_flat_matrix_config(payload)
    return payload


load_ablation_test_matrix_config = load_test_matrix_config
load_matrix_config = load_test_matrix_config
load_config = load_test_matrix_config


def _test_groups(config: dict[str, Any]) -> list[dict[str, Any]]:
    groups = config.get("test_groups", [])
    if not isinstance(groups, list):
        raise TestMatrixError("config.test_groups must be a list")
    normalized: list[dict[str, Any]] = []
    for idx, group in enumerate(groups):
        if not isinstance(group, dict):
            raise TestMatrixError(f"test_groups[{idx}] must be an object")
        group_id = str(group.get("group_id", "")).strip()
        if not group_id:
            raise TestMatrixError(f"test_groups[{idx}].group_id is required")
        paths = group.get("pytest_paths", [])
        if not isinstance(paths, list) or not all(isinstance(path, str) and path.strip() for path in paths):
            raise TestMatrixError(f"{group_id}.pytest_paths must be a non-empty list of paths")
        lanes = group.get("lanes", [])
        if not isinstance(lanes, list) or not all(isinstance(lane, str) and lane.strip() for lane in lanes):
            raise TestMatrixError(f"{group_id}.lanes must be a non-empty list of lane names")
        normalized.append(group)
    return normalized


def validate_test_matrix_config(config: dict[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    if not _config_is_group_matrix(config):
        _validate_flat_matrix_config(config, repo_root=repo_root)
        listed = sorted(set(_collect_flat_pytest_paths(config)))
        return {
            "group_count": len(config.get("families", {})) if isinstance(config.get("families"), dict) else 0,
            "listed_test_count": len(listed),
            "unique_test_count": len(listed),
            "discovered_test_count": len(list(repo_root.glob("tests/test_*.py"))),
            "unlisted_tests": [],
            "extra_listed_tests": [],
            "declared_lanes": sorted(str(key) for key in (config.get("lanes") or {}).keys()),
        }
    groups = _test_groups(config)
    lane_config = config.get("lanes") or {}
    if not isinstance(lane_config, dict):
        raise TestMatrixError("config.lanes must be an object")
    declared_lanes = set(str(key) for key in lane_config.keys())
    errors: list[str] = []
    listed_paths: list[str] = []
    duplicate_paths: list[str] = []
    seen_paths: set[str] = set()
    for group in groups:
        group_id = str(group["group_id"])
        for lane in group.get("lanes", []):
            if declared_lanes and str(lane) not in declared_lanes:
                errors.append(f"{group_id} references undeclared lane '{lane}'")
        for raw_path in group.get("pytest_paths", []):
            rel_path = str(raw_path).replace("\\", "/")
            full_path = repo_root / rel_path
            if not full_path.exists():
                errors.append(f"{group_id} references missing pytest file: {rel_path}")
            if rel_path in seen_paths:
                duplicate_paths.append(rel_path)
            seen_paths.add(rel_path)
            listed_paths.append(rel_path)

    coverage = config.get("coverage_policy", {})
    discover_glob = str(coverage.get("discover_glob", "tests/test_*.py")) if isinstance(coverage, dict) else "tests/test_*.py"
    discovered = sorted(path.relative_to(repo_root).as_posix() for path in repo_root.glob(discover_glob))
    unlisted = sorted(set(discovered) - set(listed_paths))
    extra = sorted(set(listed_paths) - set(discovered))
    if isinstance(coverage, dict) and bool(coverage.get("all_pytest_files_listed", False)) and unlisted:
        errors.append(f"unlisted pytest files: {unlisted}")
    if duplicate_paths:
        errors.append(f"duplicate pytest file ownership: {sorted(set(duplicate_paths))}")
    if errors:
        raise TestMatrixError("; ".join(errors))
    return {
        "group_count": len(groups),
        "listed_test_count": len(listed_paths),
        "unique_test_count": len(set(listed_paths)),
        "discovered_test_count": len(discovered),
        "unlisted_tests": unlisted,
        "extra_listed_tests": extra,
        "declared_lanes": sorted(declared_lanes),
    }


def _collect_flat_pytest_paths(config: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for section_name in ("lanes", "families"):
        section = config.get(section_name, {})
        if not isinstance(section, dict):
            continue
        for value in section.values():
            if not isinstance(value, dict):
                continue
            paths.extend(str(path).replace("\\", "/") for path in value.get("pytest_files", []))
    return paths


def select_test_groups(
    config: dict[str, Any],
    *,
    families: list[str] | None = None,
    lanes: list[str] | None = None,
) -> list[dict[str, Any]]:
    groups = _test_groups(config)
    selected_families = {value.strip().upper() for value in (families or []) if value.strip()}
    selected_lanes = {value.strip() for value in (lanes or []) if value.strip()}
    if not selected_lanes:
        default_lane = str(config.get("default_lane", "")).strip()
        if default_lane:
            selected_lanes = {default_lane}

    out: list[dict[str, Any]] = []
    for group in groups:
        group_family = str(group.get("family", "")).strip().upper()
        group_series = str(group.get("series", "")).strip().upper()
        group_id = str(group.get("group_id", "")).strip().upper()
        family_match = not selected_families or bool({group_family, group_series, group_id} & selected_families)
        lane_match = not selected_lanes or bool(set(str(lane) for lane in group.get("lanes", [])) & selected_lanes)
        if family_match and lane_match:
            out.append(group)
    return out


def _select_flat_tests(
    config: dict[str, Any],
    *,
    families: list[str] | None = None,
    lanes: list[str] | None = None,
    series: str | None = None,
) -> list[str]:
    selected_families = {value.strip().upper() for value in (families or []) if value and value.strip()}
    selected_lanes = {value.strip() for value in (lanes or []) if value and value.strip()}
    selected_series = str(series or "").strip().upper()
    candidates: list[set[str]] = []
    lanes_obj = config.get("lanes", {})
    families_obj = config.get("families", {})
    if selected_lanes and isinstance(lanes_obj, dict):
        lane_paths: set[str] = set()
        for lane in selected_lanes:
            lane_entry = lanes_obj.get(lane)
            if isinstance(lane_entry, dict):
                lane_paths.update(str(path).replace("\\", "/") for path in lane_entry.get("pytest_files", []))
        candidates.append(lane_paths)
    if (selected_families or selected_series) and isinstance(families_obj, dict):
        family_paths: set[str] = set()
        for key, entry in families_obj.items():
            if not isinstance(entry, dict):
                continue
            key_match = str(key).strip().upper() in selected_families
            entry_series = {str(item).strip().upper() for item in entry.get("series", [])}
            series_match = bool(selected_series and selected_series in entry_series)
            if key_match or series_match:
                family_paths.update(str(path).replace("\\", "/") for path in entry.get("pytest_files", []))
        candidates.append(family_paths)
    if not candidates:
        candidates.append(set(_collect_flat_pytest_paths(config)))
    selected = set.intersection(*candidates) if candidates else set()
    return sorted(selected)


def resolve_selected_tests(
    *,
    config: dict[str, Any],
    lane: str | None = None,
    lanes: list[str] | None = None,
    family: str | None = None,
    families: list[str] | None = None,
    series: str | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    del repo_root
    lane_values = [value for value in ([lane] if lane else []) + (lanes or []) if value]
    family_values = [value for value in ([family] if family else []) + (families or []) if value]
    if not _config_is_group_matrix(config):
        paths = _select_flat_tests(config, families=family_values, lanes=lane_values, series=series)
        return {"selected_tests": paths, "pytest_files": paths, "tests": paths, "files": paths}
    groups = select_test_groups(config, families=family_values or ([series] if series else []), lanes=lane_values)
    paths = selected_pytest_paths(groups)
    return {"selected_tests": paths, "pytest_files": paths, "tests": paths, "files": paths, "selected_groups": groups}


resolve_matrix_tests = resolve_selected_tests
resolve_tests = resolve_selected_tests
select_tests = resolve_selected_tests


def selected_pytest_paths(groups: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for raw_path in group.get("pytest_paths", []):
            rel_path = str(raw_path).replace("\\", "/")
            if rel_path in seen:
                continue
            seen.add(rel_path)
            paths.append(rel_path)
    return paths


def build_pytest_command(
    pytest_paths: list[str],
    *,
    run_id: str,
    extra_args: list[str] | None = None,
) -> list[str]:
    basetemp = f"artifacts/tmp/pytest_ablation_test_matrix/{_safe_slug(run_id)}"
    (REPO_ROOT / basetemp).parent.mkdir(parents=True, exist_ok=True)
    return [sys.executable, "-m", "pytest", "-q", *pytest_paths, "--basetemp", basetemp, *(extra_args or [])]


def build_test_matrix_manifest(
    *,
    config: dict[str, Any],
    config_path: Path,
    run_id: str,
    selected_groups: list[dict[str, Any]],
    lanes: list[str],
    families: list[str],
    execute: bool,
    command: list[str],
    validation: dict[str, Any],
    returncode: int | None = None,
    duration_seconds: float | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    pytest_paths = selected_pytest_paths(selected_groups)
    status = "dry_run"
    if execute:
        status = "passed" if returncode == 0 else "failed"
    metrics = {
        "selected_group_count": float(len(selected_groups)),
        "selected_test_count": float(len(pytest_paths)),
        "matrix_unique_test_count": float(validation.get("unique_test_count", 0.0) or 0.0),
        "matrix_discovered_test_count": float(validation.get("discovered_test_count", 0.0) or 0.0),
        "matrix_unlisted_test_count": float(len(validation.get("unlisted_tests", []) or [])),
        "matrix_extra_listed_test_count": float(len(validation.get("extra_listed_tests", []) or [])),
        "pytest_returncode": float(returncode) if returncode is not None else 0.0,
        "pytest_passed": 1.0 if execute and returncode == 0 else 0.0,
        "pytest_executed": 1.0 if execute else 0.0,
    }
    return {
        "series": series_metadata("M", "ablation_test_matrix", "scripts/control_plane/run_ablation_test_matrix.py"),
        "lineage": lineage_metadata("eval_only", dataset_profile="pytest_ablation_test_matrix", difficulty_tier="control_plane"),
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "execute": bool(execute),
        "config_path": _repo_rel(config_path),
        "schema_version": config.get("schema_version"),
        "selected_lanes": lanes,
        "selected_families": families,
        "selected_group_count": len(selected_groups),
        "selected_test_count": len(pytest_paths),
        "metrics": metrics,
        "selected_groups": [
            {
                "group_id": group.get("group_id"),
                "series": group.get("series"),
                "family": group.get("family"),
                "lanes": group.get("lanes", []),
                "pytest_paths": group.get("pytest_paths", []),
            }
            for group in selected_groups
        ],
        "pytest_paths": pytest_paths,
        "command": command,
        "commands": [command],
        "validation": validation,
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "stdout_tail": stdout[-8000:],
        "stderr_tail": stderr[-8000:],
    }


def build_dry_run_manifest(
    *,
    config: dict[str, Any],
    selection: Any | None = None,
    selected_tests: list[str] | None = None,
    lane: str | None = None,
    family: str | None = None,
    run_id: str = "ablation_test_matrix_dry_run",
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dry_run: bool = True,
    report_only: bool = True,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    del dry_run, report_only, repo_root
    paths = selected_tests or []
    if not paths and isinstance(selection, dict):
        for key in ("selected_tests", "pytest_files", "tests", "files"):
            if key in selection:
                paths = [str(path).replace("\\", "/") for path in selection[key]]
                break
    if not paths and selection is not None and not isinstance(selection, dict):
        paths = [str(path).replace("\\", "/") for path in selection]
    command = build_pytest_command(paths, run_id=run_id)
    manifest = {
        "series": series_metadata("M", "ablation_test_matrix", "scripts/control_plane/run_ablation_test_matrix.py"),
        "lineage": lineage_metadata("eval_only", dataset_profile="pytest_ablation_test_matrix", difficulty_tier="control_plane"),
        "run_id": run_id,
        "status": "dry_run",
        "execute": False,
        "output_root": _repo_rel(output_root),
        "schema_version": config.get("schema_version"),
        "selected_lanes": [lane] if lane else [],
        "selected_families": [family] if family else [],
        "selected_tests": paths,
        "pytest_files": paths,
        "tests": paths,
        "files": paths,
        "commands": [command],
        "command": command,
    }
    return manifest


dry_run_manifest = build_dry_run_manifest
plan_test_matrix_run = build_dry_run_manifest


def default_ablation_test_matrix_config(*, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    config = load_test_matrix_config(DEFAULT_CONFIG_PATH)
    families: dict[str, dict[str, Any]] = {}
    for group in _test_groups(config):
        key = str(group.get("family") or group.get("series") or group.get("group_id"))
        entry = families.setdefault(
            key,
            {
                "series": [],
                "lanes": [],
                "pytest_files": [],
            },
        )
        series = str(group.get("series", "")).strip()
        if series and series not in entry["series"]:
            entry["series"].append(series)
        for lane in group.get("lanes", []):
            if lane not in entry["lanes"]:
                entry["lanes"].append(lane)
        for path in group.get("pytest_paths", []):
            rel_path = Path(path).as_posix()
            if not (repo_root / rel_path).exists():
                raise FileNotFoundError(rel_path)
            if rel_path not in entry["pytest_files"]:
                entry["pytest_files"].append(rel_path)
    return {
        "schema_version": config.get("schema_version"),
        "purpose": config.get("purpose"),
        "lanes": config.get("lanes", {}),
        "families": families,
        "output_root": config.get("output_root"),
    }


build_default_matrix_config = default_ablation_test_matrix_config
default_matrix_config = default_ablation_test_matrix_config


def render_test_matrix_markdown(manifest: dict[str, Any]) -> str:
    lines = [
        "# Ablation Test Matrix",
        "",
        f"- run_id: `{manifest.get('run_id')}`",
        f"- status: `{manifest.get('status')}`",
        f"- execute: `{manifest.get('execute')}`",
        f"- selected lanes: `{', '.join(manifest.get('selected_lanes', [])) or 'default'}`",
        f"- selected families: `{', '.join(manifest.get('selected_families', [])) or 'all'}`",
        f"- selected groups: `{manifest.get('selected_group_count')}`",
        f"- selected tests: `{manifest.get('selected_test_count')}`",
        "",
        "## Command",
        "",
        "```text",
        " ".join(str(part) for part in manifest.get("command", [])),
        "```",
        "",
        "## Groups",
        "",
        "| group | series | family | tests |",
        "|---|---|---|---:|",
    ]
    for group in manifest.get("selected_groups", []):
        lines.append(
            f"| `{group.get('group_id')}` | `{group.get('series')}` | `{group.get('family')}` | `{len(group.get('pytest_paths', []))}` |"
        )
    return "\n".join(lines) + "\n"


def run_ablation_test_matrix(args: argparse.Namespace) -> dict[str, Any]:
    config_path = _resolve_repo_path(args.config)
    config = load_test_matrix_config(config_path)
    validation = validate_test_matrix_config(config)
    families = _parse_csv(args.family)
    lanes = _parse_csv(args.lane)
    if not lanes:
        lanes = [str(config.get("default_lane", "smoke"))]
    selected_groups = select_test_groups(config, families=families, lanes=lanes)
    if not selected_groups:
        raise TestMatrixError(f"No test groups selected for families={families or ['all']} lanes={lanes or ['default']}")
    pytest_paths = selected_pytest_paths(selected_groups)
    run_id = _safe_slug(args.run_id or f"ablation_test_matrix_{_timestamp()}")
    command = build_pytest_command(pytest_paths, run_id=run_id, extra_args=args.pytest_arg or [])

    output_root = _resolve_repo_path(args.output_root or config.get("output_root") or DEFAULT_OUTPUT_ROOT)
    output_dir = output_root / run_id
    output_root_rel = _repo_rel(output_root)
    output_dir_rel = _repo_rel(output_dir)
    validate_series_outputs("M", [output_root_rel], [output_dir_rel])
    output_dir.mkdir(parents=True, exist_ok=True)

    returncode: int | None = None
    duration_seconds: float | None = None
    stdout = ""
    stderr = ""
    if bool(args.execute):
        started = perf_counter()
        proc = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
        duration_seconds = perf_counter() - started
        returncode = int(proc.returncode)
        stdout = proc.stdout
        stderr = proc.stderr

    manifest = build_test_matrix_manifest(
        config=config,
        config_path=config_path,
        run_id=run_id,
        selected_groups=selected_groups,
        lanes=lanes,
        families=families,
        execute=bool(args.execute),
        command=command,
        validation=validation,
        returncode=returncode,
        duration_seconds=duration_seconds,
        stdout=stdout,
        stderr=stderr,
    )
    manifest["output_root"] = output_root_rel
    manifest_path = output_dir / REPORT_NAME
    summary_path = output_dir / SUMMARY_NAME
    validate_series_outputs("M", [output_root_rel], [output_dir_rel, _repo_rel(manifest_path), _repo_rel(summary_path)])
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summary_path.write_text(render_test_matrix_markdown(manifest), encoding="utf-8")

    print(f"Wrote: {manifest_path}")
    print(f"Wrote: {summary_path}")
    if stdout:
        print(stdout[-4000:])
    if stderr:
        print(stderr[-4000:], file=sys.stderr)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or dry-run the series-aware ablation pytest matrix.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--family", action="append", default=[], help="Family/series selector; may be repeated or comma-separated.")
    parser.add_argument("--lane", action="append", default=[], help="Lane selector such as smoke, ledger, architecture, full.")
    parser.add_argument("--pytest-arg", action="append", default=[], help="Extra argument appended to the pytest command.")
    parser.add_argument("--execute", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_ablation_test_matrix(args)
    return int(manifest.get("returncode") or 0)


if __name__ == "__main__":
    raise SystemExit(main())
