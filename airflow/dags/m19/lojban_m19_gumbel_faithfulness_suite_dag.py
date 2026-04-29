from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id


DEFAULTS = {
    "base_model": "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
    "seed": 23,
    "epochs": 1,
    "eval_size": 64,
    "audit_eval_size": 10,
    "dictionary_eval_size": 64,
    "run_id": "",
}


def _run_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model",
        str(cfg.get("base_model")),
        "--seed",
        str(int(cfg.get("seed", 23))),
        "--epochs",
        str(int(cfg.get("epochs", 1))),
        "--eval-size",
        str(int(cfg.get("eval_size", 64))),
        "--audit-eval-size",
        str(int(cfg.get("audit_eval_size", 10))),
        "--dictionary-eval-size",
        str(int(cfg.get("dictionary_eval_size", 64))),
        "--run-id",
        run_id,
    ]
    run_repo_script("scripts/m19/run_m19_gumbel_faithfulness_suite.py", args)


with DAG(
    dag_id="lojban_m19_gumbel_faithfulness_suite",
    description="Typed Euclidean + Gumbel hard-routing M19.31 faithfulness suite.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "gumbel", "faithfulness"],
    params={
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "seed": Param(23, type="integer", minimum=0),
        "epochs": Param(1, type="integer", minimum=1),
        "eval_size": Param(64, type="integer", minimum=1),
        "audit_eval_size": Param(10, type="integer", minimum=1),
        "dictionary_eval_size": Param(64, type="integer", minimum=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m19_gumbel_faithfulness_suite",
        python_callable=_run_suite,
    )
