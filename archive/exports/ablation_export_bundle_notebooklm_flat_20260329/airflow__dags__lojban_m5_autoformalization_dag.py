from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import (
    merge_conf,
    run_repo_script,
    sanitize_run_id,
    validate_baseline_manifest_path,
    validate_output_partition,
)


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "baseline_manifest": "docs/baselines/m_series_baseline_manifest.json",
    "train_steps": 60,
    "dataset_size": 600,
    "dataset_profile": "semantic_bench_v1",
    "difficulty_tier": "all",
    "seed": 7,
    "relation_vocab": 16,
    "samples_per_family": 12,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization",
    "local_files_only": False,
    "run_id": "",
}


def _run_m5(**context: object) -> None:
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
    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model", base_model,
        "--adapter", adapter,
        "--baseline-manifest", baseline_manifest,
        "--train-steps", str(int(cfg.get("train_steps", 60))),
        "--dataset-size", str(int(cfg.get("dataset_size", 600))),
        "--dataset-profile", str(cfg.get("dataset_profile", "semantic_bench_v1")),
        "--difficulty-tier", str(cfg.get("difficulty_tier", "all")),
        "--seed", str(int(cfg.get("seed", 7))),
        "--relation-vocab", str(int(cfg.get("relation_vocab", 16))),
        "--samples-per-family", str(int(cfg.get("samples_per_family", 12))),
        "--report-output-root", output_dir,
    ]
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")
    # run_id is used to create distinct airflow run metadata; script timestamps directories internally.
    _ = run_id
    run_repo_script("scripts/run_m5_autoformalization.py", args)


with DAG(
    dag_id="lojban_m5_autoformalization",
    description="M5 corrected auto-formalization family with selective lexical adversary and predicate-family evaluation.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m5", "autoformalization", "predicate"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "baseline_manifest": Param("docs/baselines/m_series_baseline_manifest.json", type="string", minLength=1),
        "train_steps": Param(60, type="integer", minimum=1),
        "dataset_size": Param(600, type="integer", minimum=32),
        "dataset_profile": Param("semantic_bench_v1", type="string", minLength=1),
        "difficulty_tier": Param("all", type="string", minLength=1),
        "seed": Param(7, type="integer", minimum=0),
        "relation_vocab": Param(16, type="integer", minimum=2),
        "samples_per_family": Param(12, type="integer", minimum=1),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m5_autoformalization", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m5_autoformalization",
        python_callable=_run_m5,
    )
