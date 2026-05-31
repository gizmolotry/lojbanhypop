from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_test_matrix",
    "run_id": "",
    "lane": "smoke",
    "family": "",
    "execute": False,
}


def _run_ablation_test_matrix(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    if output_dir.startswith("s3://"):
        raise ValueError("lojban_ablation_test_matrix writes local pytest artifacts; use a local output_dir, not S3.")
    args = [
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
        "--lane",
        str(cfg.get("lane", DEFAULTS["lane"])),
    ]
    family = str(cfg.get("family", "")).strip()
    if family:
        args.extend(["--family", family])
    args.append("--execute" if bool(cfg.get("execute", False)) else "--no-execute")
    run_repo_script("scripts/control_plane/run_ablation_test_matrix.py", args)


with DAG(
    dag_id="lojban_ablation_test_matrix",
    description="Series-aware pytest matrix over ablation families, emitted as a ledger manifest.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "pytest", "control-plane"],
    params={
        "output_dir": Param(DEFAULTS["output_dir"], type="string", minLength=1),
        "run_id": Param("", type="string"),
        "lane": Param(DEFAULTS["lane"], type="string", minLength=1),
        "family": Param("", type="string"),
        "execute": Param(False, type="boolean"),
    },
) as dag:
    run_test_matrix = PythonOperator(
        task_id="run_test_matrix",
        python_callable=_run_ablation_test_matrix,
    )
