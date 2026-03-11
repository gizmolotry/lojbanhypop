from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": "",
    "adapter": "",
    "b2_adapter": "",
    "drope_adapter": "",
    "handoff_projection": "",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/a_to_g",
    "run_id": "",
    "sample_size": 24,
    "dataset_size": 1000,
    "max_new_tokens": 48,
    "seeds_csv": "7,11",
    "local_files_only": False,
    "execute": True,
}


def _parse_seeds_csv(value: str) -> list[int]:
    parts = [p.strip() for p in str(value).split(",") if p.strip()]
    if not parts:
        raise ValueError("seeds_csv must contain at least one integer seed.")
    out: list[int] = []
    for p in parts:
        seed = int(p)
        if seed < 0:
            raise ValueError("seeds must be non-negative integers.")
        out.append(seed)
    return out


def _run_ablation_matrix(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    base_model = str(cfg.get("base_model", "")).strip()
    adapter = str(cfg.get("adapter", "")).strip()
    if not base_model:
        raise ValueError("base_model is required")
    if not adapter:
        raise ValueError("adapter is required")

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_root = f"{output_dir.rstrip('/')}/{run_id}"
    seeds = _parse_seeds_csv(str(cfg["seeds_csv"]))

    args = [
        "--base-model",
        base_model,
        "--adapter",
        adapter,
        "--sample-size",
        str(int(cfg["sample_size"])),
        "--seeds",
        *[str(s) for s in seeds],
        "--dataset-size",
        str(int(cfg["dataset_size"])),
        "--max-new-tokens",
        str(int(cfg["max_new_tokens"])),
        "--output-root",
        output_root,
    ]
    b2_adapter = str(cfg.get("b2_adapter", "")).strip()
    if b2_adapter:
        args.extend(["--b2-adapter", b2_adapter])

    drope_adapter = str(cfg.get("drope_adapter", "")).strip()
    if drope_adapter:
        args.extend(["--drope-adapter", drope_adapter])

    handoff_projection = str(cfg.get("handoff_projection", "")).strip()
    if handoff_projection:
        args.extend(["--handoff-projection", handoff_projection])

    if bool(cfg.get("local_files_only")):
        args.append("--local-files-only")
    if bool(cfg.get("execute")):
        args.append("--execute")

    run_repo_script("scripts/run_coconut_ablation_matrix.py", args)


with DAG(
    dag_id="lojban_ablation_a_to_g",
    description="A-G coconut ablation matrix wrapper for scripts/run_coconut_ablation_matrix.py",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "a-to-g"],
    params={
        "base_model": Param("", type="string", minLength=1),
        "adapter": Param("", type="string", minLength=1),
        "b2_adapter": Param("", type="string"),
        "drope_adapter": Param("", type="string"),
        "handoff_projection": Param("", type="string"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/a_to_g", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "sample_size": Param(24, type="integer", minimum=1),
        "dataset_size": Param(1000, type="integer", minimum=1),
        "max_new_tokens": Param(48, type="integer", minimum=1),
        "seeds_csv": Param("7,11", type="string", minLength=1),
        "local_files_only": Param(False, type="boolean"),
        "execute": Param(True, type="boolean"),
    },
) as dag:
    PythonOperator(
        task_id="run_ablation_a_to_g",
        python_callable=_run_ablation_matrix,
    )
