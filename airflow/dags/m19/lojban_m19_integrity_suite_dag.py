from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
    "bridge_path": "artifacts/models/m19/retrain_current_bundle/m19_3_b_retrain_20260421/M19.3_8Q_128D_8S.pt",
    "train_data_path": "artifacts/datasets/m19_mixed_curriculum_v1.jsonl",
    "eval_data_path": "artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl",
    "audit_data_path": "artifacts/datasets/sanity_check_v1.jsonl",
    "eval_size": 400,
    "audit_eval_size": 10,
    "track": "M19",
    "cell_id": "M19.3_8Q_128D_8S",
    "num_queries": 8,
    "bottleneck_dim": 128,
    "scratchpad_length": 8,
    "max_latent_steps": 64,
    "random_scale": 0.05,
    "seed": 19,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_integrity_suite",
    "run_id": "",
}


def _run_m19_integrity(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model",
        str(cfg.get("base_model")),
        "--bridge-path",
        str(cfg.get("bridge_path")),
        "--train-data-path",
        str(cfg.get("train_data_path")),
        "--eval-data-path",
        str(cfg.get("eval_data_path")),
        "--audit-data-path",
        str(cfg.get("audit_data_path")),
        "--eval-size",
        str(int(cfg.get("eval_size", 400))),
        "--audit-eval-size",
        str(int(cfg.get("audit_eval_size", 10))),
        "--track",
        str(cfg.get("track", "M19")),
        "--cell-id",
        str(cfg.get("cell_id", "M19.3_8Q_128D_8S")),
        "--num-queries",
        str(int(cfg.get("num_queries", 8))),
        "--bottleneck-dim",
        str(int(cfg.get("bottleneck_dim", 128))),
        "--scratchpad-length",
        str(int(cfg.get("scratchpad_length", 8))),
        "--max-latent-steps",
        str(int(cfg.get("max_latent_steps", 64))),
        "--random-scale",
        str(float(cfg.get("random_scale", 0.05))),
        "--seed",
        str(int(cfg.get("seed", 19))),
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    run_repo_script("scripts/m19/run_m19_integrity_suite.py", args)


with DAG(
    dag_id="lojban_m19_integrity_suite",
    description="M19 integrity suite with overlap-aware, purged, masked, and audit controls.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "integrity", "controls"],
    params={
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "bridge_path": Param("artifacts/models/m19/retrain_current_bundle/m19_3_b_retrain_20260421/M19.3_8Q_128D_8S.pt", type="string", minLength=1),
        "train_data_path": Param("artifacts/datasets/m19_mixed_curriculum_v1.jsonl", type="string", minLength=1),
        "eval_data_path": Param("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl", type="string", minLength=1),
        "audit_data_path": Param("artifacts/datasets/sanity_check_v1.jsonl", type="string", minLength=1),
        "eval_size": Param(400, type="integer", minimum=1),
        "audit_eval_size": Param(10, type="integer", minimum=1),
        "track": Param("M19", type="string", minLength=1),
        "cell_id": Param("M19.3_8Q_128D_8S", type="string", minLength=1),
        "num_queries": Param(8, type="integer", minimum=1),
        "bottleneck_dim": Param(128, type="integer", minimum=1),
        "scratchpad_length": Param(8, type="integer", minimum=1),
        "max_latent_steps": Param(64, type="integer", minimum=1),
        "random_scale": Param(0.05, type="number", minimum=0.0),
        "seed": Param(19, type="integer", minimum=0),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_integrity_suite", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m19_integrity_suite",
        python_callable=_run_m19_integrity,
    )
