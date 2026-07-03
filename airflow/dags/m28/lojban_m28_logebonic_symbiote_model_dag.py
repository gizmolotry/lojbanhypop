from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m28_logebonic_symbiote_model_suite",
    "run_id": "",
    "seed_list": "23,29,31,37,41,43",
    "train_size": 24000,
    "eval_size": 6000,
    "epochs": 48,
    "baseline_epochs": 8,
    "batch_size": 512,
    "learning_rate": 0.001,
    "seed": 23,
    "max_frames": 4,
    "max_symbols": 32,
    "embedding_dim": 64,
    "hidden_dim": 128,
    "advisor_hidden_dim": 128,
    "symbol_budget": 8,
    "relevance_rank_weight": 0.25,
    "checkpoint_every_epochs": 8,
    "device": "cuda",
    "use_amp": True,
}


def _run_m28_logebonic_symbiote_model(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--output-root", output_dir,
        "--run-id", run_id,
        "--seed-list", str(cfg.get("seed_list")),
        "--train-size", str(int(cfg.get("train_size", 24000))),
        "--eval-size", str(int(cfg.get("eval_size", 6000))),
        "--epochs", str(int(cfg.get("epochs", 48))),
        "--baseline-epochs", str(int(cfg.get("baseline_epochs", 8))),
        "--batch-size", str(int(cfg.get("batch_size", 512))),
        "--learning-rate", str(float(cfg.get("learning_rate", 0.001))),
        "--seed", str(int(cfg.get("seed", 23))),
        "--max-frames", str(int(cfg.get("max_frames", 4))),
        "--max-symbols", str(int(cfg.get("max_symbols", 32))),
        "--embedding-dim", str(int(cfg.get("embedding_dim", 64))),
        "--hidden-dim", str(int(cfg.get("hidden_dim", 128))),
        "--advisor-hidden-dim", str(int(cfg.get("advisor_hidden_dim", 128))),
        "--symbol-budget", str(int(cfg.get("symbol_budget", 8))),
        "--relevance-rank-weight", str(float(cfg.get("relevance_rank_weight", 0.25))),
        "--checkpoint-every-epochs", str(int(cfg.get("checkpoint_every_epochs", 8))),
        "--device", str(cfg.get("device")),
    ]
    if bool(cfg.get("use_amp", True)):
        args.append("--use-amp")
    run_repo_script("scripts/m28/run_m28_logebonic_model_suite.py", args)


with DAG(
    dag_id="lojban_m28_logebonic_symbiote_model",
    description="M28 actual checkpointable Logebonic Symbiote model training and baseline suite.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m28", "logebonic-symbiote-model"],
    params={
        "output_dir": Param(DEFAULTS["output_dir"], type="string", minLength=1),
        "run_id": Param("", type="string"),
        "seed_list": Param(DEFAULTS["seed_list"], type="string", minLength=1),
        "train_size": Param(DEFAULTS["train_size"], type="integer", minimum=1),
        "eval_size": Param(DEFAULTS["eval_size"], type="integer", minimum=1),
        "epochs": Param(DEFAULTS["epochs"], type="integer", minimum=1),
        "baseline_epochs": Param(DEFAULTS["baseline_epochs"], type="integer", minimum=1),
        "batch_size": Param(DEFAULTS["batch_size"], type="integer", minimum=1),
        "learning_rate": Param(DEFAULTS["learning_rate"], type="number", minimum=0.0),
        "seed": Param(DEFAULTS["seed"], type="integer"),
        "max_frames": Param(DEFAULTS["max_frames"], type="integer", minimum=1),
        "max_symbols": Param(DEFAULTS["max_symbols"], type="integer", minimum=1),
        "embedding_dim": Param(DEFAULTS["embedding_dim"], type="integer", minimum=1),
        "hidden_dim": Param(DEFAULTS["hidden_dim"], type="integer", minimum=1),
        "advisor_hidden_dim": Param(DEFAULTS["advisor_hidden_dim"], type="integer", minimum=1),
        "symbol_budget": Param(DEFAULTS["symbol_budget"], type="integer", minimum=0),
        "relevance_rank_weight": Param(DEFAULTS["relevance_rank_weight"], type="number", minimum=0.0),
        "checkpoint_every_epochs": Param(DEFAULTS["checkpoint_every_epochs"], type="integer", minimum=0),
        "device": Param(DEFAULTS["device"], type="string", minLength=1),
        "use_amp": Param(DEFAULTS["use_amp"], type="boolean"),
    },
) as dag:
    PythonOperator(
        task_id="run_m28_logebonic_symbiote_model",
        python_callable=_run_m28_logebonic_symbiote_model,
    )
