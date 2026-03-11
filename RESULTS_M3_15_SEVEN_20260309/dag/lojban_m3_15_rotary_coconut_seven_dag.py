from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "checkpoint": "",
    "train_steps": 120,
    "eval_size": 500,
    "seed": 42,
    "pack_jsonl": "",
    "strict_balance": True,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven",
    "local_files_only": False,
    "group_id": "",
}


def _run_m3_15_rotary_coconut_seven(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    checkpoint = str(cfg.get("checkpoint", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")
    if not checkpoint:
        raise ValueError("checkpoint is required")

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    group_id = sanitize_run_id(str(cfg.get("group_id") or getattr(dag_run, "run_id", "manual")))
    output_root = output_dir

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--checkpoint",
        checkpoint,
        "--train-steps",
        str(int(cfg.get("train_steps", 120))),
        "--eval-size",
        str(int(cfg.get("eval_size", 500))),
        "--seed",
        str(int(cfg.get("seed", 42))),
        "--output-root",
        output_root,
        "--group-id",
        group_id,
    ]
    if bool(cfg.get("strict_balance", True)):
        args.append("--strict-balance")
    else:
        args.append("--no-strict-balance")
    pack_jsonl = str(cfg.get("pack_jsonl", "")).strip()
    if pack_jsonl:
        args.extend(["--pack-jsonl", pack_jsonl])
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")

    run_repo_script("scripts/run_m3_15_rotary_coconut_seven.py", args)


with DAG(
    dag_id="lojban_m3_15_rotary_coconut_seven",
    description="M3.15 rotary coconut seven-run harness wrapper.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m3-15", "rotary", "coconut", "ablation", "seven-run"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "checkpoint": Param("", type="string", minLength=1),
        "train_steps": Param(120, type="integer", minimum=0),
        "eval_size": Param(500, type="integer", minimum=2),
        "seed": Param(42, type="integer", minimum=0),
        "pack_jsonl": Param("", type="string"),
        "strict_balance": Param(True, type="boolean"),
        "output_dir": Param(
            "artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven",
            type="string",
            minLength=1,
        ),
        "local_files_only": Param(False, type="boolean"),
        "group_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m3_15_rotary_coconut_seven",
        python_callable=_run_m3_15_rotary_coconut_seven,
    )
