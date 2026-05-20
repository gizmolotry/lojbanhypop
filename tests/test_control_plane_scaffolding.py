from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from lojban_evolution.control_plane.artifact_io import (
    latest_json,
    latest_named_manifest,
    path_allowed_for_discovery,
    read_json_optional,
    repo_relative_or_string,
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
