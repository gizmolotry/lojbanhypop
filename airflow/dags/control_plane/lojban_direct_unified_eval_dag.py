from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "family": "M19",
    "track": "",
    "output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval",
    "run_id": "",
    "history_manifest": "",
    "benchmark_report": "",
    "audit_report": "",
    "m20_suite_report": "",
    "m20_lock_report": "",
    "m20_induction_report": "",
    "m21_suite_report": "",
    "m21_synthetic_assay_report": "",
    "m21_actual_bridge_report": "",
    "m21_lock_report": "",
    "m21_pointer_microgrid_report": "",
    "m21_gauntlet_report": "",
    "m21_adversarial_audit_report": "",
    "m22_generalization_report": "",
    "m23_relevance_report": "",
    "m24_substrate_compression_report": "",
    "m25_emergent_report": "",
    "m26_end_to_end_report": "",
    "execute_m19_direct": False,
    "base_model": "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
    "bridge_path": "artifacts/models/m19/grid/m19_isolation_grid_20260409_v3/M19.3_8Q_128D_8S.pt",
    "eval_data_path": "artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl",
    "audit_data_path": "artifacts/datasets/sanity_check_v1.jsonl",
    "eval_size": 100,
    "audit_eval_size": 10,
    "num_queries": 8,
    "bottleneck_dim": 128,
    "scratchpad_length": 8,
    "min_latent_steps": 4,
    "max_latent_steps": 64,
    "random_scale": 0.05,
    "seed": 19,
    "cell_id": "M19.3_8Q_128D_8S",
}


def _run_direct_unified_eval(**context: object) -> None:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)
    output_dir = validate_output_partition(str(cfg.get("output_dir", "")), "telemetry/raw")
    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    args = [
        "--family",
        str(cfg.get("family", "M19")),
        "--track",
        str(cfg.get("track", "")),
        "--output-root",
        output_dir,
        "--run-id",
        run_id,
    ]
    if str(cfg.get("history_manifest", "")).strip():
        args.extend(["--history-manifest", str(cfg.get("history_manifest"))])
    if str(cfg.get("benchmark_report", "")).strip():
        args.extend(["--benchmark-report", str(cfg.get("benchmark_report"))])
    if str(cfg.get("audit_report", "")).strip():
        args.extend(["--audit-report", str(cfg.get("audit_report"))])
    if str(cfg.get("m20_suite_report", "")).strip():
        args.extend(["--m20-suite-report", str(cfg.get("m20_suite_report"))])
    if str(cfg.get("m20_lock_report", "")).strip():
        args.extend(["--m20-lock-report", str(cfg.get("m20_lock_report"))])
    if str(cfg.get("m20_induction_report", "")).strip():
        args.extend(["--m20-induction-report", str(cfg.get("m20_induction_report"))])
    if str(cfg.get("m21_suite_report", "")).strip():
        args.extend(["--m21-suite-report", str(cfg.get("m21_suite_report"))])
    if str(cfg.get("m21_synthetic_assay_report", "")).strip():
        args.extend(["--m21-synthetic-assay-report", str(cfg.get("m21_synthetic_assay_report"))])
    if str(cfg.get("m21_actual_bridge_report", "")).strip():
        args.extend(["--m21-actual-bridge-report", str(cfg.get("m21_actual_bridge_report"))])
    if str(cfg.get("m21_lock_report", "")).strip():
        args.extend(["--m21-lock-report", str(cfg.get("m21_lock_report"))])
    if str(cfg.get("m21_pointer_microgrid_report", "")).strip():
        args.extend(["--m21-pointer-microgrid-report", str(cfg.get("m21_pointer_microgrid_report"))])
    if str(cfg.get("m21_gauntlet_report", "")).strip():
        args.extend(["--m21-gauntlet-report", str(cfg.get("m21_gauntlet_report"))])
    if str(cfg.get("m21_adversarial_audit_report", "")).strip():
        args.extend(["--m21-adversarial-audit-report", str(cfg.get("m21_adversarial_audit_report"))])
    if str(cfg.get("m22_generalization_report", "")).strip():
        args.extend(["--m22-generalization-report", str(cfg.get("m22_generalization_report"))])
    if str(cfg.get("m23_relevance_report", "")).strip():
        args.extend(["--m23-relevance-report", str(cfg.get("m23_relevance_report"))])
    if str(cfg.get("m24_substrate_compression_report", "")).strip():
        args.extend(["--m24-compression-report", str(cfg.get("m24_substrate_compression_report"))])
    if str(cfg.get("m25_emergent_report", "")).strip():
        args.extend(["--m25-emergent-report", str(cfg.get("m25_emergent_report"))])
    if str(cfg.get("m26_end_to_end_report", "")).strip():
        args.extend(["--m26-end-to-end-report", str(cfg.get("m26_end_to_end_report"))])
    if bool(cfg.get("execute_m19_direct", False)):
        args.extend(
            [
                "--execute-m19-direct",
                "--base-model",
                str(cfg.get("base_model")),
                "--bridge-path",
                str(cfg.get("bridge_path")),
                "--eval-data-path",
                str(cfg.get("eval_data_path")),
                "--audit-data-path",
                str(cfg.get("audit_data_path")),
                "--eval-size",
                str(int(cfg.get("eval_size", 100))),
                "--audit-eval-size",
                str(int(cfg.get("audit_eval_size", 10))),
                "--num-queries",
                str(int(cfg.get("num_queries", 8))),
                "--bottleneck-dim",
                str(int(cfg.get("bottleneck_dim", 128))),
                "--scratchpad-length",
                str(int(cfg.get("scratchpad_length", 8))),
                "--min-latent-steps",
                str(int(cfg.get("min_latent_steps", 4))),
                "--max-latent-steps",
                str(int(cfg.get("max_latent_steps", 64))),
                "--random-scale",
                str(float(cfg.get("random_scale", 0.05))),
                "--seed",
                str(int(cfg.get("seed", 19))),
                "--cell-id",
                str(cfg.get("cell_id", "M19.3_8Q_128D_8S")),
            ]
        )
    run_repo_script("scripts/control_plane/run_direct_unified_eval.py", args)


with DAG(
    dag_id="lojban_direct_unified_eval",
    description="Contract-aware direct unified eval surface for modern families.",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "control-plane", "direct-eval", "m19", "m20", "m21", "m22", "m23", "m24", "m25", "m26", "lineage"],
    params={
        "family": Param("M19", type="string", minLength=1),
        "track": Param("", type="string"),
        "output_dir": Param("artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "history_manifest": Param("", type="string"),
        "benchmark_report": Param("", type="string"),
        "audit_report": Param("", type="string"),
        "m20_suite_report": Param("", type="string"),
        "m20_lock_report": Param("", type="string"),
        "m20_induction_report": Param("", type="string"),
        "m21_suite_report": Param("", type="string"),
        "m21_synthetic_assay_report": Param("", type="string"),
        "m21_actual_bridge_report": Param("", type="string"),
        "m21_lock_report": Param("", type="string"),
        "m21_pointer_microgrid_report": Param("", type="string"),
        "m21_gauntlet_report": Param("", type="string"),
        "m21_adversarial_audit_report": Param("", type="string"),
        "m22_generalization_report": Param("", type="string"),
        "m23_relevance_report": Param("", type="string"),
        "m24_substrate_compression_report": Param("", type="string"),
        "m25_emergent_report": Param("", type="string"),
        "m26_end_to_end_report": Param("", type="string"),
        "execute_m19_direct": Param(False, type="boolean"),
        "base_model": Param("C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "bridge_path": Param("artifacts/models/m19/grid/m19_isolation_grid_20260409_v3/M19.3_8Q_128D_8S.pt", type="string", minLength=1),
        "eval_data_path": Param("artifacts/datasets/m14_5_unified/m14_5_unified_test.jsonl", type="string", minLength=1),
        "audit_data_path": Param("artifacts/datasets/sanity_check_v1.jsonl", type="string", minLength=1),
        "eval_size": Param(100, type="integer", minimum=1),
        "audit_eval_size": Param(10, type="integer", minimum=1),
        "num_queries": Param(8, type="integer", minimum=1),
        "bottleneck_dim": Param(128, type="integer", minimum=1),
        "scratchpad_length": Param(8, type="integer", minimum=1),
        "min_latent_steps": Param(4, type="integer", minimum=1),
        "max_latent_steps": Param(64, type="integer", minimum=1),
        "random_scale": Param(0.05, type="number", minimum=0.0),
        "seed": Param(19, type="integer", minimum=0),
        "cell_id": Param("M19.3_8Q_128D_8S", type="string", minLength=1),
    },
) as dag:
    PythonOperator(
        task_id="run_direct_unified_eval",
        python_callable=_run_direct_unified_eval,
    )
