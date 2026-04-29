from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import (
    merge_conf,
    run_repo_script,
    sanitize_run_id,
    validate_baseline_manifest_path,
    validate_output_dir,
    validate_output_partition,
)


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "train_steps": 120,
    "dataset_size": 1000,
    "seed": 7,
    "local_files_only": False,
    "l_output_root": "runs/l_series/m3_plus",
    "report_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_plus",
    "j5_summary": "",
    "dynamic_arity_signatures": False,
    "operator_arity_json": "",
    "default_relation_arity": 2,
    "arity_enforcement_mode": "crystallization",
    "baseline_manifest": "docs/baselines/m_series_baseline_manifest.json",
    "run_id": "",
}


def _repo_root() -> Path:
    allow_overrides = os.environ.get("LOJBAN_ALLOW_ENV_OVERRIDES", "0") == "1"
    override = os.environ.get("LOJBAN_REPO_ROOT")
    if allow_overrides and override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[2]


def _validate_optional_local_input_file(path_value: str, label: str) -> str:
    value = str(path_value).strip()
    if not value:
        return ""
    if value.startswith("s3://"):
        raise ValueError(f"{label} must be a local file path")
    path = Path(value)
    if not path.is_absolute():
        path = _repo_root() / path
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return value


def _run_m3_plus(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")
    baseline_manifest = validate_baseline_manifest_path(str(cfg.get("baseline_manifest", "")).strip())

    l_output_root = validate_output_dir(str(cfg["l_output_root"]))
    report_output_dir = validate_output_partition(str(cfg["report_output_dir"]), "telemetry/raw")
    if l_output_root.startswith("s3://") or report_output_dir.startswith("s3://"):
        raise ValueError("M3+ runner writes local artifacts only. Use local output roots and sync to S3 downstream.")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    report_output_root = f"{report_output_dir.rstrip('/')}/{run_id}"

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--train-steps",
        str(int(cfg["train_steps"])),
        "--dataset-size",
        str(int(cfg["dataset_size"])),
        "--seed",
        str(int(cfg["seed"])),
        "--l-output-root",
        l_output_root,
        "--report-output-root",
        report_output_root,
        "--baseline-manifest",
        baseline_manifest,
    ]
    if bool(cfg.get("local_files_only")):
        args.append("--local-files-only")
    j5_summary = _validate_optional_local_input_file(str(cfg.get("j5_summary", "")).strip(), "j5_summary")
    if j5_summary:
        args.extend(["--j5-summary", j5_summary])
    if bool(cfg.get("dynamic_arity_signatures")):
        args.append("--dynamic-arity-signatures")
    operator_arity_json = _validate_optional_local_input_file(
        str(cfg.get("operator_arity_json", "")).strip(),
        "operator_arity_json",
    )
    if operator_arity_json:
        args.extend(["--operator-arity-json", operator_arity_json])
    args.extend(["--default-relation-arity", str(int(cfg.get("default_relation_arity", 2)))])
    args.extend(["--arity-enforcement-mode", str(cfg.get("arity_enforcement_mode", "crystallization"))])

    run_repo_script("scripts/run_m3_plus_family.py", args)


with DAG(
    dag_id="lojban_m3_plus_family",
    description="M3+ gate-driven family wrapper (M3.0..M3.4) for scripts/run_m3_plus_family.py",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-plus", "ablation"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "train_steps": Param(120, type="integer", minimum=1),
        "dataset_size": Param(1000, type="integer", minimum=1),
        "seed": Param(7, type="integer", minimum=0),
        "local_files_only": Param(False, type="boolean"),
        "l_output_root": Param("runs/l_series/m3_plus", type="string", minLength=1),
        "report_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_plus", type="string", minLength=1),
        "j5_summary": Param("", type="string"),
        "dynamic_arity_signatures": Param(False, type="boolean"),
        "operator_arity_json": Param("", type="string"),
        "default_relation_arity": Param(2, type="integer", minimum=1),
        "arity_enforcement_mode": Param("crystallization", enum=["legacy_strict", "registry_strict", "crystallization"]),
        "baseline_manifest": Param("docs/baselines/m_series_baseline_manifest.json", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_plus_family",
        python_callable=_run_m3_plus,
    )
