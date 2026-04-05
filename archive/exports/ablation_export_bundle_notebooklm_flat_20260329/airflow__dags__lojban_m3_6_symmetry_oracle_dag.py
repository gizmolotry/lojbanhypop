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
    "samples_per_operator": 20,
    "seed": 7,
    "baseline_manifest": "docs/baselines/m_series_baseline_manifest.json",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle",
    "run_id": "",
}


def _run_m3_6_oracle(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    baseline_manifest = validate_baseline_manifest_path(str(cfg.get("baseline_manifest", "")).strip())
    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--samples-per-operator",
        str(int(cfg.get("samples_per_operator", 20))),
        "--seed",
        str(int(cfg.get("seed", 7))),
        "--baseline-manifest",
        baseline_manifest,
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    run_repo_script("scripts/run_m3_6_symmetry_oracle.py", args)


with DAG(
    dag_id="lojban_m3_6_symmetry_oracle_suite",
    description="M3.6 evaluation-only symmetry oracle validation suite (A/B/C).",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-6", "oracle", "eval"],
    params={
        "samples_per_operator": Param(20, type="integer", minimum=1),
        "seed": Param(7, type="integer", minimum=0),
        "baseline_manifest": Param("docs/baselines/m_series_baseline_manifest.json", type="string", minLength=1),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m3_6_symmetry_oracle", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_6_oracle",
        python_callable=_run_m3_6_oracle,
    )
