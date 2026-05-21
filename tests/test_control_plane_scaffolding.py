from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lojban_evolution.control_plane.artifact_io import (
    latest_json,
    latest_named_manifest,
    load_manifest_with_schema,
    path_allowed_for_discovery,
    read_json_optional,
    read_json_required,
    repo_relative_or_string,
    write_json,
    write_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_airflow_utils_module():
    module_path = REPO_ROOT / "airflow" / "dags" / "lojban_airflow_utils.py"
    spec = importlib.util.spec_from_file_location("lojban_airflow_utils", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_artifact_io_latest_manifest_supports_recursive_and_one_level_search(tmp_path: Path) -> None:
    shallow = tmp_path / "run_a" / "manifest.json"
    deep = tmp_path / "run_b" / "nested" / "manifest.json"
    shallow.parent.mkdir(parents=True)
    deep.parent.mkdir(parents=True)
    shallow.write_text(json.dumps({"name": "shallow"}), encoding="utf-8")
    deep.write_text(json.dumps({"name": "deep"}), encoding="utf-8")

    nonrecursive = latest_named_manifest(tmp_path, "manifest.json", recursive=False, path_filter=None)
    recursive = latest_named_manifest(tmp_path, "manifest.json", recursive=True, path_filter=None)

    assert nonrecursive == shallow
    assert recursive in {shallow, deep}


def test_artifact_io_json_and_path_helpers_are_tolerant(tmp_path: Path) -> None:
    good = tmp_path / "report.json"
    bad = tmp_path / "bad.json"
    good.write_text(json.dumps({"ok": True}), encoding="utf-8")
    bad.write_text("{", encoding="utf-8")

    assert read_json_optional(good) == {"ok": True}
    assert read_json_optional(tmp_path / "missing.json") is None
    assert read_json_optional(bad, swallow_errors=True) is None
    assert latest_json(tmp_path, preferred_names=["report.json"], path_filter=None) == good
    assert path_allowed_for_discovery(Path("runs") / "real" / "report.json")
    assert not path_allowed_for_discovery(Path("runs") / "__pycache__" / "report.json")
    assert not path_allowed_for_discovery(Path("runs") / "test_generated" / "report.json")
    assert repo_relative_or_string(REPO_ROOT / "docs" / "PROJECT_INDEX.md", REPO_ROOT) == "docs/PROJECT_INDEX.md"


def test_artifact_io_required_json_schema_and_writers(tmp_path: Path) -> None:
    manifest = write_json(tmp_path / "nested" / "manifest.json", {"schema_version": "1.0", "ok": True})
    text = write_text(tmp_path / "nested" / "note.txt", "kept")
    list_payload = tmp_path / "list.json"
    missing_schema = tmp_path / "missing_schema.json"
    list_payload.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    missing_schema.write_text(json.dumps({"ok": True}), encoding="utf-8")

    assert manifest.exists()
    assert text.read_text(encoding="utf-8") == "kept"
    assert read_json_required(manifest) == {"schema_version": "1.0", "ok": True}
    assert load_manifest_with_schema(manifest, expected_schema_version="1.0")["ok"] is True

    try:
        read_json_required(tmp_path / "missing.json")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing required JSON should raise FileNotFoundError")

    try:
        read_json_required(list_payload)
    except TypeError:
        pass
    else:
        raise AssertionError("non-dict JSON should raise TypeError by default")

    try:
        load_manifest_with_schema(missing_schema)
    except KeyError:
        pass
    else:
        raise AssertionError("manifest without schema_version should raise KeyError")

    try:
        load_manifest_with_schema(manifest, expected_schema_version="2.0")
    except ValueError:
        pass
    else:
        raise AssertionError("wrong schema_version should raise ValueError")


def test_airflow_series_factory_cli_helpers_and_conf_resolution() -> None:
    airflow_utils = _load_airflow_utils_module()
    dag_run = SimpleNamespace(conf={"epochs": 4, "enabled": False, "label": "smoke"})

    cfg = airflow_utils.resolve_dag_conf(
        {"epochs": 1, "enabled": True, "label": "default"},
        {"dag_run": dag_run},
    )

    assert cfg == {"epochs": 4, "enabled": False, "label": "smoke"}
    assert airflow_utils.bool_cli_flag("use-cuda", True) == "--use-cuda"
    assert airflow_utils.bool_cli_flag("--use-cuda", False) == "--no-use-cuda"
    assert airflow_utils.scalar_cli_args(
        [
            ("epochs", 4),
            ("enabled", False),
            ("skip", None),
            ("label", "smoke"),
        ]
    ) == ["--epochs", "4", "--no-enabled", "--label", "smoke"]
    assert airflow_utils.optional_cli_args(
        {"epochs": 4, "enabled": False},
        [
            airflow_utils.CliArgSpec("epochs", "epochs", int),
            airflow_utils.CliArgSpec("enabled", "enabled", bool),
            airflow_utils.CliArgSpec("label", "label", optional=True),
        ],
    ) == ["--epochs", "4", "--no-enabled"]
    assert airflow_utils.params_from_defaults({"epochs": 4, "enabled": True, "label": "smoke"}) == {
        "epochs": 4,
        "enabled": True,
        "label": "smoke",
    }


def test_airflow_series_callable_builds_validated_script_invocation(monkeypatch) -> None:
    airflow_utils = _load_airflow_utils_module()
    calls: list[tuple[str, list[str]]] = []

    monkeypatch.setattr(
        airflow_utils,
        "run_repo_script",
        lambda script_relpath, cli_args: calls.append((script_relpath, cli_args)),
    )
    task_spec = airflow_utils.SeriesTaskSpec(
        task_id="train",
        script_relpath="scripts/m21/run_m21_dynamic_bridi_suite.py",
        output_partition="telemetry/raw",
        argv=lambda cfg, output_dir, run_id: airflow_utils.scalar_cli_args(
            [
                ("output-dir", output_dir),
                ("run-id", run_id),
                ("epochs", cfg["epochs"]),
            ]
        ),
    )
    callable_ = airflow_utils.build_series_python_callable(
        task_spec,
        {"output_dir": "artifacts/runs/telemetry/raw/m21", "epochs": 2, "run_id": "default"},
    )

    callable_(dag_run=SimpleNamespace(conf={"epochs": 3, "run_id": "manual__2026-05-20"}))

    assert calls == [
        (
            "scripts/m21/run_m21_dynamic_bridi_suite.py",
            [
                "--output-dir",
                "artifacts/runs/telemetry/raw/m21",
                "--run-id",
                "manual__2026-05-20",
                "--epochs",
                "3",
            ],
        )
    ]


def test_airflow_run_repo_script_canonicalizes_legacy_paths(monkeypatch) -> None:
    airflow_utils = _load_airflow_utils_module()
    calls = []

    monkeypatch.setattr(
        airflow_utils.subprocess,
        "run",
        lambda cmd, cwd, env, check: calls.append((cmd, cwd, check)),
    )

    airflow_utils.run_repo_script("scripts/run_m18_controller_family.py", ["--help"])

    assert calls
    assert calls[0][0][1] == "scripts/m18/run_m18_controller_family.py"
    assert calls[0][2] is True
