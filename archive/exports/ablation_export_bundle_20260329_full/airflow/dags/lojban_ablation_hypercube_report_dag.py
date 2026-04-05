from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube",
    "run_id": "",
    "l6_manifest": "",
    "j5_summary": "",
    "m3_6_report": "",
    "m4_operator_family_report": "",
    "m3_7_report": "",
    "m3_8_report": "",
    "m3_9_report": "",
    "m3_10_report": "",
    "m3_11_report": "",
}


def _build_hypercube_report(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    output_dir = validate_output_partition(str(cfg["output_dir"]), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))

    args = [
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    l6_manifest = str(cfg.get("l6_manifest", "")).strip()
    j5_summary = str(cfg.get("j5_summary", "")).strip()
    m3_6_report = str(cfg.get("m3_6_report", "")).strip()
    m4_operator_family_report = str(cfg.get("m4_operator_family_report", "")).strip()
    m3_7_report = str(cfg.get("m3_7_report", "")).strip()
    m3_8_report = str(cfg.get("m3_8_report", "")).strip()
    m3_9_report = str(cfg.get("m3_9_report", "")).strip()
    m3_10_report = str(cfg.get("m3_10_report", "")).strip()
    m3_11_report = str(cfg.get("m3_11_report", "")).strip()
    if l6_manifest:
        args.extend(["--l6-manifest", l6_manifest])
    if j5_summary:
        args.extend(["--j5-summary", j5_summary])
    if m3_6_report:
        args.extend(["--m3-6-report", m3_6_report])
    if m4_operator_family_report:
        args.extend(["--m4-operator-family-report", m4_operator_family_report])
    if m3_7_report:
        args.extend(["--m3-7-report", m3_7_report])
    if m3_8_report:
        args.extend(["--m3-8-report", m3_8_report])
    if m3_9_report:
        args.extend(["--m3-9-report", m3_9_report])
    if m3_10_report:
        args.extend(["--m3-10-report", m3_10_report])
    if m3_11_report:
        args.extend(["--m3-11-report", m3_11_report])

    run_repo_script("scripts/build_airflow_ablation_hypercube_report.py", args)


with DAG(
    dag_id="lojban_ablation_hypercube_report",
    description="Build consolidated M-series ablation hypercube report from J/L artifacts.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "ablation", "hypercube", "report"],
    params={
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "l6_manifest": Param("", type="string"),
        "j5_summary": Param("", type="string"),
        "m3_6_report": Param("", type="string"),
        "m4_operator_family_report": Param("", type="string"),
        "m3_7_report": Param("", type="string"),
        "m3_8_report": Param("", type="string"),
        "m3_9_report": Param("", type="string"),
        "m3_10_report": Param("", type="string"),
        "m3_11_report": Param("", type="string"),
    },
) as dag:
    PythonOperator(
        task_id="build_hypercube_report",
        python_callable=_build_hypercube_report,
    )
