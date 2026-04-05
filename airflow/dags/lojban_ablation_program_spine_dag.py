from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "history_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill",
    "program_map_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_program_map",
    "program_spine_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/ablation_program_spine",
    "suite_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite",
    "run_id": "",
    "include_git": False,
    "build_suite": True,
}


def _history_run_id(base_run_id: str) -> str:
    return f"{base_run_id}_history"


def _program_map_run_id(base_run_id: str) -> str:
    return f"{base_run_id}_program_map"


def _program_spine_run_id(base_run_id: str) -> str:
    return f"{base_run_id}_program_spine"


def _suite_run_id(base_run_id: str) -> str:
    return f"{base_run_id}_suite"


def _run_history_backfill(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("history_output_dir", "")), "telemetry/raw")

    args = [
        "--output-root", output_dir,
        "--run-id", _history_run_id(run_id),
    ]
    if bool(cfg.get("include_git", False)):
        args.append("--include-git")
    run_repo_script("scripts/run_ablation_history_backfill.py", args)


def _run_program_map(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("program_map_output_dir", "")), "telemetry/raw")

    args = [
        "--output-root", output_dir,
        "--run-id", _program_map_run_id(run_id),
    ]
    run_repo_script("scripts/build_ablation_program_map.py", args)


def _run_program_spine(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("program_spine_output_dir", "")), "telemetry/raw")

    args = [
        "--output-root", output_dir,
        "--run-id", _program_spine_run_id(run_id),
    ]
    run_repo_script("scripts/build_ablation_program_spine.py", args)


def _run_suite(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    if not bool(cfg.get("build_suite", True)):
        return
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("suite_output_dir", "")), "telemetry/raw")

    args = [
        "--output-root", output_dir,
        "--run-id", _suite_run_id(run_id),
    ]
    run_repo_script("scripts/run_m_bridge_ablation_test_suite.py", args)


with DAG(
    dag_id="lojban_ablation_program_spine",
    description="Master control-plane DAG that backfills the full ablation history, renders the concentrated program map, renders the ordered program spine, and refreshes the unified M-series suite.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "program", "spine", "history", "ledger"],
    params={
        "history_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill", type="string", minLength=1),
        "program_map_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_program_map", type="string", minLength=1),
        "program_spine_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/ablation_program_spine", type="string", minLength=1),
        "suite_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m_bridge_ablation_test_suite", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "include_git": Param(False, type="boolean"),
        "build_suite": Param(True, type="boolean"),
    },
) as dag:
    history_backfill = PythonOperator(
        task_id="history_backfill",
        python_callable=_run_history_backfill,
    )

    build_program_map = PythonOperator(
        task_id="build_program_map",
        python_callable=_run_program_map,
    )

    build_program_spine = PythonOperator(
        task_id="build_program_spine",
        python_callable=_run_program_spine,
    )

    refresh_unified_suite = PythonOperator(
        task_id="refresh_unified_suite",
        python_callable=_run_suite,
    )

    history_backfill >> build_program_map >> build_program_spine >> refresh_unified_suite
