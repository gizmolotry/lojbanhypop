from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "direct_unified_eval": "artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m19_3_b_integrity_20260422/direct_unified_eval_manifest.json",
    "replication_report": "",
    "kill_test_report": "",
    "whole_grid_manifest": "",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m19_paper_package",
    "doc_output": "docs/history/reports/M19_PAPER_PACKAGE_LATEST.md",
    "run_id": "",
}


def _run_m19_paper_package(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--direct-unified-eval",
        str(cfg.get("direct_unified_eval")),
        "--output-root",
        output_dir,
        "--doc-output",
        str(cfg.get("doc_output")),
        "--run-id",
        run_id,
    ]
    if str(cfg.get("replication_report") or "").strip():
        args.extend(["--replication-report", str(cfg.get("replication_report"))])
    if str(cfg.get("kill_test_report") or "").strip():
        args.extend(["--kill-test-report", str(cfg.get("kill_test_report"))])
    if str(cfg.get("whole_grid_manifest") or "").strip():
        args.extend(["--whole-grid-manifest", str(cfg.get("whole_grid_manifest"))])
    run_repo_script("scripts/control_plane/render_m19_paper_package.py", args)


with DAG(
    dag_id="lojban_m19_paper_package",
    description="Render the current M19 paper-style evidence package.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "m-series", "m19", "paper"],
    params={
        "direct_unified_eval": Param("artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m19_3_b_integrity_20260422/direct_unified_eval_manifest.json", type="string", minLength=1),
        "replication_report": Param("", type="string"),
        "kill_test_report": Param("", type="string"),
        "whole_grid_manifest": Param("", type="string"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/m19_paper_package", type="string", minLength=1),
        "doc_output": Param("docs/history/reports/M19_PAPER_PACKAGE_LATEST.md", type="string", minLength=1),
        "run_id": Param("", type="string"),
    },
) as dag:
    PythonOperator(task_id="render_m19_paper_package", python_callable=_run_m19_paper_package)
