from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m26_end_to_end_loafman",
    "run_id": "",
    "seed_list": "23,29,31,37,41,43",
    "train_size": 24000,
    "eval_size": 6000,
    "epochs": 32,
    "prompt_epochs": 32,
    "batch_size": 512,
    "max_symbols": 32,
    "symbol_budget": 8,
    "matched_prompt_budget": 8,
    "device": "cuda",
}


def _run_m26_end_to_end_loafman_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    run_repo_script(
        "scripts/m26/run_m26_end_to_end_loafman_suite.py",
        [
            "--output-root", output_dir,
            "--run-id", run_id,
            "--seed-list", str(cfg.get("seed_list")),
            "--train-size", str(int(cfg.get("train_size", 24000))),
            "--eval-size", str(int(cfg.get("eval_size", 6000))),
            "--epochs", str(int(cfg.get("epochs", 32))),
            "--prompt-epochs", str(int(cfg.get("prompt_epochs", 32))),
            "--batch-size", str(int(cfg.get("batch_size", 512))),
            "--max-symbols", str(int(cfg.get("max_symbols", 32))),
            "--symbol-budget", str(int(cfg.get("symbol_budget", 8))),
            "--matched-prompt-budget", str(int(cfg.get("matched_prompt_budget", 8))),
            "--device", str(cfg.get("device")),
        ],
    )


with DAG(
    dag_id="lojban_m26_end_to_end_loafman",
    description="M26 end-to-end Lojban symbiote spinal-cord suite.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m26", "end-to-end-loafman"],
    params={
        "output_dir": Param(DEFAULTS["output_dir"], type="string", minLength=1),
        "run_id": Param("", type="string"),
        "seed_list": Param(DEFAULTS["seed_list"], type="string", minLength=1),
        "train_size": Param(DEFAULTS["train_size"], type="integer", minimum=1),
        "eval_size": Param(DEFAULTS["eval_size"], type="integer", minimum=1),
        "epochs": Param(DEFAULTS["epochs"], type="integer", minimum=1),
        "prompt_epochs": Param(DEFAULTS["prompt_epochs"], type="integer", minimum=1),
        "batch_size": Param(DEFAULTS["batch_size"], type="integer", minimum=1),
        "max_symbols": Param(DEFAULTS["max_symbols"], type="integer", minimum=1),
        "symbol_budget": Param(DEFAULTS["symbol_budget"], type="integer", minimum=0),
        "matched_prompt_budget": Param(DEFAULTS["matched_prompt_budget"], type="integer", minimum=0),
        "device": Param(DEFAULTS["device"], type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_m26_end_to_end_loafman_suite",
        python_callable=_run_m26_end_to_end_loafman_suite,
    )
