from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
    "bridge_specs": [
        "seed23=artifacts/runs/telemetry/raw/ablation/hypercube/m19_replication_suite/m19_3_b_fmt05_select_format_20260425/seed_23/M19.3_8Q_128D_8S.pt",
        "seed29=artifacts/runs/telemetry/raw/ablation/hypercube/m19_replication_suite/m19_3_b_fmt05_select_format_20260425/seed_29/M19.3_8Q_128D_8S.pt",
    ],
    "dataset_path": "artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl",
    "eval_size": 128,
    "scratchpad_length": 8,
    "num_queries": 8,
    "bottleneck_dim": 128,
    "max_latent_steps": 64,
    "tap_layer": 12,
    "seed": 19,
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_dictionary_audit",
    "run_id": "",
}


def _run_m19_dictionary_audit(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--base-model",
        str(cfg.get("base_model")),
        "--dataset-path",
        str(cfg.get("dataset_path")),
        "--eval-size",
        str(int(cfg.get("eval_size", 128))),
        "--scratchpad-length",
        str(int(cfg.get("scratchpad_length", 8))),
        "--num-queries",
        str(int(cfg.get("num_queries", 8))),
        "--bottleneck-dim",
        str(int(cfg.get("bottleneck_dim", 128))),
        "--max-latent-steps",
        str(int(cfg.get("max_latent_steps", 64))),
        "--tap-layer",
        str(int(cfg.get("tap_layer", 12))),
        "--seed",
        str(int(cfg.get("seed", 19))),
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    for bridge_spec in list(cfg.get("bridge_specs", []) or []):
        args.extend(["--bridge-spec", str(bridge_spec)])
    run_repo_script("scripts/m19/run_m19_dictionary_audit.py", args)


with DAG(
    dag_id="lojban_m19_dictionary_audit",
    description="Compare dictionary health across M19 checkpoints.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "dictionary", "audit"],
    params={
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "bridge_specs": Param(
            [
                "seed23=artifacts/runs/telemetry/raw/ablation/hypercube/m19_replication_suite/m19_3_b_fmt05_select_format_20260425/seed_23/M19.3_8Q_128D_8S.pt",
                "seed29=artifacts/runs/telemetry/raw/ablation/hypercube/m19_replication_suite/m19_3_b_fmt05_select_format_20260425/seed_29/M19.3_8Q_128D_8S.pt",
            ],
            type="array",
        ),
        "dataset_path": Param("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl", type="string", minLength=1),
        "eval_size": Param(128, type="integer", minimum=1),
        "scratchpad_length": Param(8, type="integer", minimum=1),
        "num_queries": Param(8, type="integer", minimum=1),
        "bottleneck_dim": Param(128, type="integer", minimum=1),
        "max_latent_steps": Param(64, type="integer", minimum=1),
        "tap_layer": Param(12, type="integer", minimum=0),
        "seed": Param(19, type="integer", minimum=0),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_dictionary_audit", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="run_m19_dictionary_audit",
        python_callable=_run_m19_dictionary_audit,
    )
