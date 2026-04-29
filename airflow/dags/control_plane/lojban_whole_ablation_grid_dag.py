from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "whole_grid_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/whole_ablation_grid",
    "doc_output": "docs/history/reports/WHOLE_ABLATION_GRID_LATEST.md",
    "run_id": "",
    "refresh_legacy_grid": False,
    "legacy_grid_run_id": "",
    "legacy_grid_execute": False,
    "local_files_only": False,
}


def _run_whole_grid(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    output_dir = validate_output_partition(str(cfg.get("whole_grid_output_dir", "")), "telemetry/raw")
    args = [
        "--output-root",
        output_dir,
        "--doc-output",
        str(cfg.get("doc_output", DEFAULTS["doc_output"])),
        "--run-id",
        run_id,
    ]
    if bool(cfg.get("refresh_legacy_grid", False)):
        args.append("--refresh-legacy-grid")
    legacy_grid_run_id = str(cfg.get("legacy_grid_run_id", "")).strip()
    if legacy_grid_run_id:
        args.extend(["--legacy-grid-run-id", legacy_grid_run_id])
    if bool(cfg.get("legacy_grid_execute", False)):
        args.append("--legacy-grid-execute")
    if bool(cfg.get("local_files_only", False)):
        args.append("--local-files-only")
    run_repo_script("scripts/control_plane/run_whole_ablation_grid.py", args)


with DAG(
    dag_id="lojban_whole_ablation_grid",
    description="Canonical whole-grid control-plane DAG: legacy runnable surface, normalized M-series anchors, and lineage manifests in one auditable report.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "whole-grid", "history", "control-plane"],
    params={
        "whole_grid_output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/whole_ablation_grid", type="string", minLength=1),
        "doc_output": Param("docs/history/reports/WHOLE_ABLATION_GRID_LATEST.md", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "refresh_legacy_grid": Param(False, type="boolean"),
        "legacy_grid_run_id": Param("", type="string"),
        "legacy_grid_execute": Param(False, type="boolean"),
        "local_files_only": Param(False, type="boolean"),
    },
) as dag:
    render_whole_grid = PythonOperator(
        task_id="render_whole_grid",
        python_callable=_run_whole_grid,
    )

