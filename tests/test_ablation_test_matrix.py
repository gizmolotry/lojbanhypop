from __future__ import annotations

import inspect
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import pytest

from conftest import REPO_ROOT, load_script_module


RUNNER_PATH = "scripts/control_plane/run_ablation_test_matrix.py"
EXPECTED_OUTPUT_ROOT = (
    "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_test_matrix"
)


def _load_runner():
    return load_script_module("run_ablation_test_matrix_test", RUNNER_PATH)


def _call_supported(fn, /, *positional: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(fn)
    parameters = signature.parameters
    positional_parameters = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    used_names = {parameter.name for parameter in positional_parameters[: len(positional)]}
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    supported_kwargs = (
        dict(kwargs)
        if accepts_var_kwargs
        else {
            key: value
            for key, value in kwargs.items()
            if key in parameters and key not in used_names
        }
    )
    return fn(*positional[: len(positional_parameters)], **supported_kwargs)


def _call_first(module: Any, names: Iterable[str], /, *args: Any, **kwargs: Any) -> Any:
    attempted: list[str] = []
    for name in names:
        candidate = getattr(module, name, None)
        if callable(candidate):
            return _call_supported(candidate, *args, **kwargs)
        attempted.append(name)
    pytest.fail(
        f"{RUNNER_PATH} must expose one of {attempted} for the ablation test matrix contract."
    )


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        converted = value.to_dict()
        if isinstance(converted, dict):
            return converted
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    pytest.fail(f"Expected mapping-shaped value, got {type(value).__name__}.")


def _relative_test_path(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("path", "pytest_file", "test_file", "file"):
            if key in value:
                value = value[key]
                break
    path = Path(str(value))
    if path.is_absolute():
        path = path.relative_to(REPO_ROOT)
    return path.as_posix()


def _selected_paths(selection: Any) -> list[str]:
    if isinstance(selection, (str, Path)):
        return [_relative_test_path(selection)]
    if isinstance(selection, dict):
        for key in ("selected_tests", "pytest_files", "tests", "files"):
            if key in selection:
                return [_relative_test_path(item) for item in selection[key]]
    for attr in ("selected_tests", "pytest_files", "tests", "files"):
        if hasattr(selection, attr):
            return [_relative_test_path(item) for item in getattr(selection, attr)]
    if isinstance(selection, Iterable):
        return [_relative_test_path(item) for item in selection]
    pytest.fail(f"Could not read selected tests from {type(selection).__name__}.")


def _collect_pytest_files(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = Path(value).as_posix()
        return [normalized] if normalized.startswith("tests/") and normalized.endswith(".py") else []
    if isinstance(value, dict):
        files: list[str] = []
        for nested in value.values():
            files.extend(_collect_pytest_files(nested))
        return files
    if isinstance(value, Iterable):
        files: list[str] = []
        for nested in value:
            files.extend(_collect_pytest_files(nested))
        return files
    return []


def _assert_selected(selection: Any, expected: set[str]) -> None:
    paths = _selected_paths(selection)
    assert set(paths) == expected
    assert len(paths) == len(set(paths))


def _write_matrix_config(tmp_path: Path, *, missing_file: bool = False) -> Path:
    m26_smoke = "tests/test_m26_control_plane_wiring.py"
    control_smoke = "tests/test_control_plane_scaffolding.py"
    missing = "tests/does_not_exist_for_ablation_matrix.py"
    payload = {
        "schema_version": "1.0",
        "lanes": {
            "smoke": {
                "pytest_files": [
                    m26_smoke,
                    missing if missing_file else control_smoke,
                ],
            },
            "full": {
                "pytest_files": [
                    m26_smoke,
                    "tests/test_m26_end_to_end_loafman.py",
                    control_smoke,
                    "tests/test_whole_ablation_grid.py",
                ],
            },
        },
        "families": {
            "M26": {
                "series": ["M26"],
                "pytest_files": [
                    m26_smoke,
                    "tests/test_m26_end_to_end_loafman.py",
                ],
            },
            "control-plane": {
                "series": ["Control Plane", "control_plane"],
                "pytest_files": [
                    control_smoke,
                    "tests/test_whole_ablation_grid.py",
                ],
            },
        },
    }
    path = tmp_path / "ablation_test_matrix.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _load_matrix_config(module: Any, path: Path) -> Any:
    return _call_first(
        module,
        (
            "load_ablation_test_matrix_config",
            "load_matrix_config",
            "load_config",
        ),
        path,
        config_path=path,
        repo_root=REPO_ROOT,
    )


def _resolve_tests(
    module: Any,
    config: Any,
    *,
    lane: str | None = None,
    family: str | None = None,
    series: str | None = None,
) -> Any:
    return _call_first(
        module,
        (
            "resolve_selected_tests",
            "resolve_matrix_tests",
            "resolve_tests",
            "select_tests",
        ),
        config=config,
        lane=lane,
        lanes=[lane] if lane else None,
        family=family,
        families=[family] if family else None,
        series=series,
        repo_root=REPO_ROOT,
    )


def _dry_run_manifest(
    module: Any,
    config: Any,
    selected_tests: Any,
    tmp_path: Path,
) -> dict[str, Any]:
    run_id = "pytest_matrix_dry_run"
    output_root = tmp_path / "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_test_matrix"
    names = (
        "build_dry_run_manifest",
        "dry_run_manifest",
        "plan_test_matrix_run",
        "build_run_manifest",
        "run_ablation_test_matrix",
    )
    for name in names:
        candidate = getattr(module, name, None)
        if not callable(candidate):
            continue
        parameters = inspect.signature(candidate).parameters
        if name.startswith("run_") and not (
            "dry_run" in parameters or "report_only" in parameters
        ):
            continue
        manifest = _call_supported(
            candidate,
            config=config,
            selection=selected_tests,
            selected_tests=_selected_paths(selected_tests),
            lane="smoke",
            family="M26",
            run_id=run_id,
            output_root=output_root,
            dry_run=True,
            report_only=True,
            repo_root=REPO_ROOT,
        )
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if manifest_path.exists():
                return json.loads(manifest_path.read_text(encoding="utf-8"))
        return _as_mapping(manifest)
    pytest.fail(
        f"{RUNNER_PATH} must expose a dry-run/report manifest builder or dry-run runner."
    )


def _default_config_payload(module: Any) -> dict[str, Any]:
    for name in (
        "DEFAULT_ABLATION_TEST_MATRIX_CONFIG",
        "DEFAULT_MATRIX_CONFIG",
        "DEFAULT_CONFIG",
        "ABLATION_TEST_MATRIX_CONFIG",
    ):
        if hasattr(module, name):
            return _as_mapping(getattr(module, name))
    return _as_mapping(
        _call_first(
            module,
            (
                "default_ablation_test_matrix_config",
                "build_default_matrix_config",
                "default_matrix_config",
            ),
            repo_root=REPO_ROOT,
        )
    )


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _family_entry(config: dict[str, Any], wanted: str) -> Any:
    families = config.get("families") or config.get("family_entries") or config.get("series")
    assert isinstance(families, dict)
    wanted_key = _normalized_key(wanted)
    for key, value in families.items():
        if _normalized_key(str(key)) == wanted_key:
            return value
    pytest.fail(f"Default ablation test matrix config is missing {wanted!r} coverage.")


def test_matrix_loader_accepts_lanes_and_rejects_missing_pytest_files(tmp_path: Path) -> None:
    runner = _load_runner()
    config_path = _write_matrix_config(tmp_path)

    loaded = _load_matrix_config(runner, config_path)
    payload = _as_mapping(loaded)

    assert set(payload["lanes"]) == {"smoke", "full"}
    assert {"M26", "control-plane"} <= set(payload["families"])

    bad_config_path = _write_matrix_config(tmp_path, missing_file=True)
    with pytest.raises((FileNotFoundError, ValueError), match="does_not_exist_for_ablation_matrix"):
        _load_matrix_config(runner, bad_config_path)


def test_group_matrix_loader_rejects_malformed_lane_schema(tmp_path: Path) -> None:
    runner = _load_runner()
    bad_config = {
        "schema_version": "bad",
        "lanes": ["smoke"],
        "test_groups": [
            {
                "group_id": "bad.group",
                "series": "bad",
                "family": "bad",
                "lanes": ["smoke"],
                "pytest_paths": ["tests/test_ablation_test_matrix.py"],
            }
        ],
    }
    config_path = tmp_path / "bad_group_matrix.json"
    config_path.write_text(json.dumps(bad_config), encoding="utf-8")

    with pytest.raises(ValueError, match="config.lanes must be an object"):
        _load_matrix_config(runner, config_path)


def test_matrix_resolver_selects_by_lane_family_and_series(tmp_path: Path) -> None:
    runner = _load_runner()
    config = _load_matrix_config(runner, _write_matrix_config(tmp_path))

    _assert_selected(
        _resolve_tests(runner, config, lane="smoke"),
        {
            "tests/test_m26_control_plane_wiring.py",
            "tests/test_control_plane_scaffolding.py",
        },
    )
    _assert_selected(
        _resolve_tests(runner, config, family="M26"),
        {
            "tests/test_m26_control_plane_wiring.py",
            "tests/test_m26_end_to_end_loafman.py",
        },
    )
    _assert_selected(
        _resolve_tests(runner, config, lane="smoke", family="M26"),
        {"tests/test_m26_control_plane_wiring.py"},
    )
    _assert_selected(
        _resolve_tests(runner, config, series="control_plane"),
        {
            "tests/test_control_plane_scaffolding.py",
            "tests/test_whole_ablation_grid.py",
        },
    )


def test_matrix_dry_run_manifest_is_ledger_shaped(tmp_path: Path) -> None:
    runner = _load_runner()
    config = _load_matrix_config(runner, _write_matrix_config(tmp_path))
    selected = _resolve_tests(runner, config, lane="smoke", family="M26")

    manifest = _dry_run_manifest(runner, config, selected, tmp_path)

    assert manifest["run_id"] == "pytest_matrix_dry_run"
    assert set(_selected_paths(manifest)) == {"tests/test_m26_control_plane_wiring.py"}
    assert manifest["status"] in {"dry_run", "planned", "not_run", "skipped"}
    assert EXPECTED_OUTPUT_ROOT in str(manifest["output_root"]).replace("\\", "/")
    assert isinstance(manifest["commands"], list)
    assert manifest["commands"]
    command_text = "\n".join(
        " ".join(str(part) for part in command)
        if isinstance(command, list)
        else json.dumps(command, sort_keys=True)
        if isinstance(command, dict)
        else str(command)
        for command in manifest["commands"]
    )
    assert "pytest" in command_text
    assert "tests/test_m26_control_plane_wiring.py" in command_text


def test_default_matrix_config_covers_m26_and_control_plane() -> None:
    runner = _load_runner()
    config = _default_config_payload(runner)
    group_config = _load_matrix_config(
        runner,
        REPO_ROOT / "configs" / "ablation_test_matrix.json",
    )

    assert set(config["lanes"]) == {"smoke", "ledger", "architecture", "full"}
    for lane in config["lanes"]:
        assert _selected_paths(_resolve_tests(runner, group_config, lane=lane))
    m26_files = _collect_pytest_files(_family_entry(config, "M26"))
    control_plane_files = _collect_pytest_files(_family_entry(config, "control-plane"))
    all_pytest_files = _collect_pytest_files(config)
    discovered_pytest_files = {
        path.relative_to(REPO_ROOT).as_posix()
        for path in REPO_ROOT.glob("tests/test_*.py")
    }
    full_lane_files = set(_selected_paths(_resolve_tests(runner, group_config, lane="full")))

    assert any("test_m26" in path for path in m26_files)
    assert any(
        "test_control_plane" in path or "test_whole_ablation_grid.py" in path
        for path in control_plane_files
    )
    assert sorted(set(all_pytest_files)) == sorted(all_pytest_files)
    assert full_lane_files == discovered_pytest_files
    missing = [path for path in all_pytest_files if not (REPO_ROOT / path).exists()]
    assert missing == []
