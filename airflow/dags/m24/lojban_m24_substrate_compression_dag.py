from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m24_substrate_compression",
    "run_id": "",
    "seed_list": "24,30,36,42,48,54",
    "train_size": 6000,
    "eval_size": 1500,
    "generator_epochs": 16,
    "advisor_epochs": 16,
    "prompt_epochs": 16,
    "batch_size": 128,
    "device": "cuda",
}


def _run_m24_substrate_compression_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    run_repo_script(
        "scripts/m24/run_m24_substrate_compression_suite.py",
        [
            "--output-root", output_dir,
            "--run-id", run_id,
            "--seed-list", str(cfg.get("seed_list")),
            "--train-size", str(int(cfg.get("train_size", 6000))),
            "--eval-size", str(int(cfg.get("eval_size", 1500))),
            "--generator-epochs", str(int(cfg.get("generator_epochs", 16))),
            "--advisor-epochs", str(int(cfg.get("advisor_epochs", 16))),
            "--prompt-epochs", str(int(cfg.get("prompt_epochs", 16))),
            "--batch-size", str(int(cfg.get("batch_size", 128))),
            "--device", str(cfg.get("device")),
        ],
    )


with DAG(
    dag_id="lojban_m24_substrate_compression",
    description="M24 substrate compression suite over modern M-series direct evaluation surfaces.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m24", "substrate-compression"],
    params={
        "output_dir": Param(DEFAULTS["output_dir"], type="string", minLength=1),
        "run_id": Param("", type="string"),
        "seed_list": Param(DEFAULTS["seed_list"], type="string", minLength=1),
        "train_size": Param(DEFAULTS["train_size"], type="integer", minimum=1),
        "eval_size": Param(DEFAULTS["eval_size"], type="integer", minimum=1),
        "generator_epochs": Param(DEFAULTS["generator_epochs"], type="integer", minimum=1),
        "advisor_epochs": Param(DEFAULTS["advisor_epochs"], type="integer", minimum=1),
        "prompt_epochs": Param(DEFAULTS["prompt_epochs"], type="integer", minimum=1),
        "batch_size": Param(DEFAULTS["batch_size"], type="integer", minimum=1),
        "device": Param(DEFAULTS["device"], type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_m24_substrate_compression_suite",
        python_callable=_run_m24_substrate_compression_suite,
    )
