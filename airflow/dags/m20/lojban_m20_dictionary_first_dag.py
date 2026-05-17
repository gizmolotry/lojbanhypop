from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "seed_list": "23,29",
    "cell_list": "A,B,C,D,E,F",
    "train_size": 2400,
    "eval_size": 600,
    "epochs": 8,
    "batch_size": 96,
    "learning_rate": 0.003,
    "codebook_size": 2000,
    "embedding_dim": 64,
    "hidden_dim": 96,
    "temperature_start": 1.5,
    "temperature_end": 0.25,
    "factor_weight": 1.0,
    "dictionary_commitment_weight": 0.75,
    "quotient_invariance_weight": 2.0,
    "brivi_lock_weight": 1.0,
    "stable_threshold": 0.70,
    "device": "cpu",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite",
    "run_id": "",
}


def _run_m20_dictionary_first(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--seed-list",
        str(cfg.get("seed_list", "23,29")),
        "--cell-list",
        str(cfg.get("cell_list", "A,B,C,D,E,F")),
        "--train-size",
        str(int(cfg.get("train_size", 2400))),
        "--eval-size",
        str(int(cfg.get("eval_size", 600))),
        "--epochs",
        str(int(cfg.get("epochs", 8))),
        "--batch-size",
        str(int(cfg.get("batch_size", 96))),
        "--learning-rate",
        str(float(cfg.get("learning_rate", 0.003))),
        "--codebook-size",
        str(int(cfg.get("codebook_size", 2000))),
        "--embedding-dim",
        str(int(cfg.get("embedding_dim", 64))),
        "--hidden-dim",
        str(int(cfg.get("hidden_dim", 96))),
        "--temperature-start",
        str(float(cfg.get("temperature_start", 1.5))),
        "--temperature-end",
        str(float(cfg.get("temperature_end", 0.25))),
        "--factor-weight",
        str(float(cfg.get("factor_weight", 1.0))),
        "--dictionary-commitment-weight",
        str(float(cfg.get("dictionary_commitment_weight", 0.75))),
        "--quotient-invariance-weight",
        str(float(cfg.get("quotient_invariance_weight", 2.0))),
        "--brivi-lock-weight",
        str(float(cfg.get("brivi_lock_weight", 1.0))),
        "--stable-threshold",
        str(float(cfg.get("stable_threshold", 0.70))),
        "--device",
        str(cfg.get("device", "cpu")),
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    run_repo_script("scripts/m20/run_m20_dictionary_first_suite.py", args)


with DAG(
    dag_id="lojban_m20_dictionary_first",
    description="M20 dictionary-first predicate induction retraining suite.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m20", "dictionary"],
    params={
        "seed_list": Param("23,29", type="string", minLength=1),
        "cell_list": Param("A,B,C,D,E,F", type="string", minLength=1),
        "train_size": Param(2400, type="integer", minimum=1),
        "eval_size": Param(600, type="integer", minimum=1),
        "epochs": Param(8, type="integer", minimum=1),
        "batch_size": Param(96, type="integer", minimum=1),
        "learning_rate": Param(0.003, type="number", minimum=0.0),
        "codebook_size": Param(2000, type="integer", minimum=1),
        "embedding_dim": Param(64, type="integer", minimum=1),
        "hidden_dim": Param(96, type="integer", minimum=1),
        "temperature_start": Param(1.5, type="number", minimum=0.0),
        "temperature_end": Param(0.25, type="number", minimum=0.0),
        "factor_weight": Param(1.0, type="number", minimum=0.0),
        "dictionary_commitment_weight": Param(0.75, type="number", minimum=0.0),
        "quotient_invariance_weight": Param(2.0, type="number", minimum=0.0),
        "brivi_lock_weight": Param(1.0, type="number", minimum=0.0),
        "stable_threshold": Param(0.70, type="number", minimum=0.0),
        "device": Param("cpu", type="string", minLength=1),
        "output_dir": Param(
            "artifacts/runs/telemetry/raw/ablation/hypercube/m20_dictionary_first_suite",
            type="string",
            minLength=1,
        ),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(task_id="run_m20_dictionary_first", python_callable=_run_m20_dictionary_first)
