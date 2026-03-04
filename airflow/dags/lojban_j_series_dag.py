from __future__ import annotations

from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import (
    merge_conf,
    run_repo_script,
    sanitize_run_id,
    validate_input_artifact,
    validate_output_partition,
)


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw",
    "run_id": "",
    "h5_ood_artifact": "",
    "j2_variants_per_graph": 1000,
    "j4_per_operator": 256,
}


def _compose_output(output_dir: str, run_id: str, file_name: str) -> str:
    if output_dir.startswith("s3://"):
        return f"{output_dir.rstrip('/')}/{run_id}_{file_name}"
    return str(Path(output_dir) / f"{run_id}_{file_name}")


def _run_j_series(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    if output_dir.startswith("s3://"):
        raise ValueError(
            "lojban_j_series_invariance currently writes local artifacts only. "
            "Use a local telemetry/raw path and sync to S3 downstream."
        )
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    j1_out = _compose_output(output_dir, run_id, "j1_graph_target.json")
    j2_out = _compose_output(output_dir, run_id, "j2_paraphrase_explosion.json")
    j3_out = _compose_output(output_dir, run_id, "j3_stopgrad_gate.json")
    j4_out = _compose_output(output_dir, run_id, "j4_operator_curriculum.json")

    j1_args = ["--output", j1_out]
    h5_ood = str(cfg.get("h5_ood_artifact") or "").strip()
    if h5_ood:
        validated = validate_input_artifact(h5_ood, "telemetry/raw")
        j1_args.extend(["--input-artifact", validated])
    run_repo_script("scripts/eval_j_1.py", j1_args)

    run_repo_script(
        "scripts/eval_j_2.py",
        [
            "--j1-artifact",
            j1_out,
            "--variants-per-graph",
            str(int(cfg["j2_variants_per_graph"])),
            "--output",
            j2_out,
        ],
    )

    run_repo_script("scripts/eval_j_3.py", ["--output", j3_out])

    curriculum_path = _compose_output(output_dir, run_id, "j4_operator_curriculum.jsonl")
    run_repo_script(
        "scripts/eval_j_4.py",
        [
            "--per-operator",
            str(int(cfg["j4_per_operator"])),
            "--output",
            j4_out,
            "--dataset-output",
            curriculum_path,
        ],
    )


with DAG(
    dag_id="lojban_j_series_invariance",
    description="Phase 7 J-series (graph target, paraphrase invariance, stop-grad gate, operator curriculum).",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "phase7", "j-series", "invariance"],
    params={
        "output_dir": Param("artifacts/runs/telemetry/raw", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "h5_ood_artifact": Param("", type="string"),
        "j2_variants_per_graph": Param(1000, type="integer", minimum=1),
        "j4_per_operator": Param(256, type="integer", minimum=1),
    },
) as dag:
    PythonOperator(
        task_id="run_j_series",
        python_callable=_run_j_series,
    )
