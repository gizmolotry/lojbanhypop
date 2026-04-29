from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.models.param import Param
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from lojban_airflow_utils import merge_conf, run_repo_script, sanitize_run_id, validate_output_partition


DEFAULTS = {
    "base_model": r"C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct",
    "adapter": "runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5",
    "run_id": "",
    "execute": True,
    "local_files_only": True,
    "run_a_to_g": True,
    "run_hj": True,
    "run_l6": True,
    "run_phase5_objective": True,
    "run_phase5_train": False,
    "run_english_cot_duel": False,
    "master_output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid",
    "a_to_g_output_root": "artifacts/runs/telemetry/raw/ablation/a_to_g/legacy_grid",
    "hj_output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid",
    "l6_output_root": "runs/l_series/l6_ablation/legacy_grid",
    "phase5_output_root": "artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid",
    "phase5_train_output_root": "runs/phase5_train_ablation/legacy_grid",
    "english_cot_duel_output_root": "runs/english_cot_control_duel/legacy_grid",
    "sample_size": 12,
    "seeds": [7],
    "dataset_size": 256,
    "max_new_tokens": 32,
    "max_logic_new_tokens": 16,
    "max_final_new_tokens": 12,
    "l6_train_steps": 4,
    "l6_dataset_size": 64,
    "phase5_train_dataset": "",
}


def _resolve_contract(context: dict[str, object]) -> dict[str, object]:
    dag_run = context.get("dag_run")
    conf = getattr(dag_run, "conf", None)
    cfg = merge_conf(DEFAULTS, conf)

    run_id = sanitize_run_id(str(cfg.get("run_id") or getattr(dag_run, "run_id", "manual")))
    master_output_root = validate_output_partition(str(cfg["master_output_root"]), "ablation/hypercube")
    a_to_g_output_root = validate_output_partition(str(cfg["a_to_g_output_root"]), "ablation/a_to_g")
    hj_output_root = validate_output_partition(str(cfg["hj_output_root"]), "ablation/hypercube")
    l6_output_root = validate_output_partition(str(cfg["l6_output_root"]), "l_series/l6_ablation")
    phase5_output_root = validate_output_partition(str(cfg["phase5_output_root"]), "ablation/hypercube")
    phase5_train_output_root = validate_output_partition(
        str(cfg["phase5_train_output_root"]), "phase5_train_ablation"
    )
    english_cot_duel_output_root = validate_output_partition(
        str(cfg["english_cot_duel_output_root"]), "english_cot_control_duel"
    )

    seeds = cfg["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list of integers")
    seed_list = [int(seed) for seed in seeds]

    return {
        "base_model": str(cfg["base_model"]),
        "adapter": str(cfg["adapter"]),
        "run_id": run_id,
        "execute": bool(cfg["execute"]),
        "local_files_only": bool(cfg["local_files_only"]),
        "run_a_to_g": bool(cfg["run_a_to_g"]),
        "run_hj": bool(cfg["run_hj"]),
        "run_l6": bool(cfg["run_l6"]),
        "run_phase5_objective": bool(cfg["run_phase5_objective"]),
        "run_phase5_train": bool(cfg["run_phase5_train"]),
        "run_english_cot_duel": bool(cfg["run_english_cot_duel"]),
        "master_output_root": master_output_root,
        "a_to_g_output_root": a_to_g_output_root,
        "hj_output_root": hj_output_root,
        "l6_output_root": l6_output_root,
        "phase5_output_root": phase5_output_root,
        "phase5_train_output_root": phase5_train_output_root,
        "english_cot_duel_output_root": english_cot_duel_output_root,
        "sample_size": int(cfg["sample_size"]),
        "seeds": seed_list,
        "dataset_size": int(cfg["dataset_size"]),
        "max_new_tokens": int(cfg["max_new_tokens"]),
        "max_logic_new_tokens": int(cfg["max_logic_new_tokens"]),
        "max_final_new_tokens": int(cfg["max_final_new_tokens"]),
        "l6_train_steps": int(cfg["l6_train_steps"]),
        "l6_dataset_size": int(cfg["l6_dataset_size"]),
        "phase5_train_dataset": str(cfg["phase5_train_dataset"]).strip(),
    }


def _common_args(contract: dict[str, object]) -> list[str]:
    args = [
        "--base-model",
        str(contract["base_model"]),
        "--adapter",
        str(contract["adapter"]),
        "--run-id",
        str(contract["run_id"]),
        "--master-output-root",
        str(contract["master_output_root"]),
        "--a-to-g-output-root",
        str(contract["a_to_g_output_root"]),
        "--hj-output-root",
        str(contract["hj_output_root"]),
        "--l6-output-root",
        str(contract["l6_output_root"]),
        "--phase5-output-root",
        str(contract["phase5_output_root"]),
        "--phase5-train-output-root",
        str(contract["phase5_train_output_root"]),
        "--english-cot-duel-output-root",
        str(contract["english_cot_duel_output_root"]),
        "--sample-size",
        str(contract["sample_size"]),
        "--seeds",
        *[str(seed) for seed in contract["seeds"]],
        "--dataset-size",
        str(contract["dataset_size"]),
        "--max-new-tokens",
        str(contract["max_new_tokens"]),
        "--max-logic-new-tokens",
        str(contract["max_logic_new_tokens"]),
        "--max-final-new-tokens",
        str(contract["max_final_new_tokens"]),
        "--l6-train-steps",
        str(contract["l6_train_steps"]),
        "--l6-dataset-size",
        str(contract["l6_dataset_size"]),
    ]
    if contract["phase5_train_dataset"]:
        args.extend(["--phase5-train-dataset", str(contract["phase5_train_dataset"])])
    if contract["local_files_only"]:
        args.append("--local-files-only")
    if contract["execute"]:
        args.append("--execute")
    return args


def _run_lane(flag_key: str, lane: str, **context: object) -> None:
    contract = _resolve_contract(context)
    if not bool(contract[flag_key]):
        return
    cli_args = _common_args(contract) + ["--only-lanes", lane]
    if lane == "phase5_train":
        cli_args.append("--include-phase5-train")
    if lane == "english_cot_duel":
        cli_args.append("--include-english-cot-duel")
    run_repo_script("scripts/legacy/run_legacy_ablation_grid.py", cli_args)


def _aggregate_grid(**context: object) -> None:
    contract = _resolve_contract(context)
    lanes: list[str] = []
    if contract["run_a_to_g"]:
        lanes.append("a_to_g")
    if contract["run_hj"]:
        lanes.append("hj")
    if contract["run_l6"]:
        lanes.append("l6")
    if contract["run_phase5_objective"]:
        lanes.append("phase5_objective")
    if contract["run_phase5_train"]:
        lanes.append("phase5_train")
    if contract["run_english_cot_duel"]:
        lanes.append("english_cot_duel")

    cli_args = _common_args(contract) + ["--aggregate-only"]
    if lanes:
        cli_args.extend(["--only-lanes", *lanes])
    if contract["run_phase5_train"]:
        cli_args.append("--include-phase5-train")
    if contract["run_english_cot_duel"]:
        cli_args.append("--include-english-cot-duel")
    run_repo_script("scripts/legacy/run_legacy_ablation_grid.py", cli_args)


with DAG(
    dag_id="lojban_legacy_ablation_grid",
    description="Unified DAG for the widest runnable legacy ablation surface (A-G, H/H5/J, L6, Phase-5).",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    max_active_runs=1,
    tags=["lojban", "legacy", "ablation", "grid"],
    params={
        "base_model": Param(r"C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct", type="string", minLength=1),
        "adapter": Param("runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5", type="string", minLength=1),
        "run_id": Param("", type="string"),
        "execute": Param(True, type="boolean"),
        "local_files_only": Param(True, type="boolean"),
        "run_a_to_g": Param(True, type="boolean"),
        "run_hj": Param(True, type="boolean"),
        "run_l6": Param(True, type="boolean"),
        "run_phase5_objective": Param(True, type="boolean"),
        "run_phase5_train": Param(False, type="boolean"),
        "run_english_cot_duel": Param(False, type="boolean"),
        "master_output_root": Param("artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid", type="string", minLength=1),
        "a_to_g_output_root": Param("artifacts/runs/telemetry/raw/ablation/a_to_g/legacy_grid", type="string", minLength=1),
        "hj_output_root": Param("artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid", type="string", minLength=1),
        "l6_output_root": Param("runs/l_series/l6_ablation/legacy_grid", type="string", minLength=1),
        "phase5_output_root": Param("artifacts/runs/telemetry/raw/ablation/hypercube/legacy_grid", type="string", minLength=1),
        "phase5_train_output_root": Param("runs/phase5_train_ablation/legacy_grid", type="string", minLength=1),
        "english_cot_duel_output_root": Param("runs/english_cot_control_duel/legacy_grid", type="string", minLength=1),
        "sample_size": Param(12, type="integer", minimum=1),
        "seeds": Param([7], type="array"),
        "dataset_size": Param(256, type="integer", minimum=1),
        "max_new_tokens": Param(32, type="integer", minimum=1),
        "max_logic_new_tokens": Param(16, type="integer", minimum=1),
        "max_final_new_tokens": Param(12, type="integer", minimum=1),
        "l6_train_steps": Param(4, type="integer", minimum=1),
        "l6_dataset_size": Param(64, type="integer", minimum=1),
        "phase5_train_dataset": Param("", type="string"),
    },
) as dag:
    start = EmptyOperator(task_id="start")
    finish = EmptyOperator(task_id="finish")

    a_to_g = PythonOperator(
        task_id="run_a_to_g",
        python_callable=_run_lane,
        op_kwargs={"flag_key": "run_a_to_g", "lane": "a_to_g"},
    )
    hj = PythonOperator(
        task_id="run_hj",
        python_callable=_run_lane,
        op_kwargs={"flag_key": "run_hj", "lane": "hj"},
    )
    l6 = PythonOperator(
        task_id="run_l6",
        python_callable=_run_lane,
        op_kwargs={"flag_key": "run_l6", "lane": "l6"},
    )
    phase5_objective = PythonOperator(
        task_id="run_phase5_objective",
        python_callable=_run_lane,
        op_kwargs={"flag_key": "run_phase5_objective", "lane": "phase5_objective"},
    )
    phase5_train = PythonOperator(
        task_id="run_phase5_train",
        python_callable=_run_lane,
        op_kwargs={"flag_key": "run_phase5_train", "lane": "phase5_train"},
    )
    english_cot_duel = PythonOperator(
        task_id="run_english_cot_duel",
        python_callable=_run_lane,
        op_kwargs={"flag_key": "run_english_cot_duel", "lane": "english_cot_duel"},
    )
    aggregate = PythonOperator(
        task_id="aggregate_legacy_grid",
        python_callable=_aggregate_grid,
    )

    start >> [a_to_g, hj, l6, phase5_objective, phase5_train, english_cot_duel] >> aggregate >> finish
