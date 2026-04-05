# Airflow Orchestration

Airflow DAGs in this repository are thin wrappers around canonical scripts only:

- `airflow/dags/lojban_experiment_dag.py` -> `scripts/pipeline_train_grounded_reasoner.py`
- `airflow/dags/lojban_phase_ablation_dag.py` -> `scripts/pipeline_eval_manifold.py`
- `airflow/dags/lojban_ablation_matrix_dag.py` -> `scripts/run_coconut_ablation_matrix.py`
- `airflow/dags/lojban_j_series_dag.py` -> `scripts/eval_j_1.py`, `scripts/eval_j_2.py`, `scripts/eval_j_3.py`, `scripts/eval_j_4.py`, `scripts/eval_j_5.py`
- `airflow/dags/lojban_l_series_dag.py` -> `scripts/train_l_series_mvs.py`
- `airflow/dags/lojban_ablation_hypercube_report_dag.py` -> `scripts/build_airflow_ablation_hypercube_report.py`
- `airflow/dags/lojban_m3_plus_dag.py` -> `scripts/run_m3_plus_family.py`
- `airflow/dags/lojban_m3_5_symmetry_dag.py` -> `scripts/run_m3_5_symmetry.py`
- `airflow/dags/lojban_m3_6_symmetry_oracle_dag.py` -> `scripts/run_m3_6_symmetry_oracle.py`
- `airflow/dags/lojban_m4_series_dag.py` -> `scripts/run_m4_series.py`

No training/eval business logic is implemented in DAG code.
Series semantics are governed by `docs/SERIES_CHARTER.md` and enforced in script runtime via `series_contract.py`.

## Contract-First Artifact Flow

This orchestration follows an artifact-contract-first model:

1. Train pipeline writes frozen manifold artifacts.
2. Eval pipeline reads model artifacts and writes raw telemetry artifacts.
3. Downstream jobs consume artifacts by partition contract, not by ad-hoc paths.

Canonical S3 partitions:

- `models/frozen_manifolds` (train outputs)
- `telemetry/raw` (eval outputs)

DAG runtime validation enforces that `output_dir` is both:

- valid per shared Airflow safety checks (`LOJBAN_ALLOWED_LOCAL_OUTPUT_ROOTS`, `LOJBAN_ALLOWED_S3_PREFIXES`)
- targeted to the expected partition for that DAG

All wrappers also reject unknown `dag_run.conf` keys. Only documented config keys are accepted; omitted keys fall back to the DAG defaults.

## Prerequisites

1. Install optional orchestration dependencies:

```bash
pip install -e ".[orchestration]"
```

2. Initialize Airflow metadata DB (example for local SQLite):

```bash
airflow db init
```

3. Ensure DAG files are visible to Airflow:

- Option A: run Airflow with `D:\lojbanhypop\airflow\dags` as the DAG folder.
- Option B: copy/symlink `airflow/dags/*.py` into your existing `$AIRFLOW_HOME/dags`.

## Environment Variables

- `LOJBAN_ALLOW_ENV_OVERRIDES` (optional, default `0`): set to `1` to allow the two overrides below.
- `LOJBAN_REPO_ROOT` (optional): absolute path to repository root (only used when `LOJBAN_ALLOW_ENV_OVERRIDES=1`).
- `LOJBAN_PYTHON_BIN` (optional): Python executable used for script execution (only used when `LOJBAN_ALLOW_ENV_OVERRIDES=1`).
- `PYTHONPATH` is managed automatically by DAG utility to include `<repo>/src`.
- `LOJBAN_ALLOWED_LOCAL_OUTPUT_ROOTS` (optional, default `artifacts/runs,runs`): comma-separated allowlist for local `output_dir`.
- `LOJBAN_ALLOWED_S3_PREFIXES` (optional): comma-separated allowlist for S3 prefixes (for example `s3://my-bucket/lojban-prod`).
- `LOJBAN_ALLOW_ABSOLUTE_OUTPUT_DIR` (optional, default `0`): set to `1` to allow absolute local output paths.
- `LOJBAN_ALLOW_ABSOLUTE_INPUT_ARTIFACT` (optional, default `0`): set to `1` to allow absolute local `input_artifact` paths.

## DAGs

### `lojban_train_grounded_reasoner`

- Schedule: `@daily`
- Task: `run_train_grounded_reasoner`
- Script: `scripts/pipeline_train_grounded_reasoner.py`
- Required partition: `models/frozen_manifolds`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `output_dir` (str, default `artifacts/runs/models/frozen_manifolds`)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `arity_violation_rate` (float, default `0.05`)
- `surgery_trigger_count` (int, default `1`)
- `ce_loss_final` (float, default `1.0`)
- `cross_attention_gain` (float, default `0.1`)
- `logical_accuracy` (float in `[0,1]`, default `0.6`)
- `variable_token_distribution_json` (str path, default `docs/variable_token_distribution_1995.example.json`)

Wrapper behavior:

- Validates `output_dir` and required partition (`models/frozen_manifolds`).
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Validates `variable_token_distribution_json` as a local `docs/*.json` path.
- Composes script `--output` as `<output_dir>/<run_id>_grounded_reasoner_train.json`.

Trigger example:

```bash
airflow dags trigger lojban_train_grounded_reasoner \
  --conf '{
    "output_dir": "s3://my-bucket/lojban-prod/models/frozen_manifolds/2026-03-03",
    "run_id": "train_2026_03_03",
    "arity_violation_rate": 0.04,
    "surgery_trigger_count": 3,
    "ce_loss_final": 0.92,
    "cross_attention_gain": 0.18,
    "logical_accuracy": 0.81,
    "variable_token_distribution_json": "docs/variable_token_distribution_1995.example.json"
  }'
```

### `lojban_eval_manifold`

- Schedule: manual (`schedule=None`)
- Task: `run_eval_manifold`
- Script: `scripts/pipeline_eval_manifold.py`
- Required partition: `telemetry/raw`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `output_dir` (str, default `artifacts/runs/telemetry/raw`)
- `input_artifact` (str path, required)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `logical_accuracy` (float in `[0,1]`, default `0.6`)
- `ce_loss_final` (float, optional)
- `cross_attention_gain` (float, optional)

Wrapper behavior:

- Validates `output_dir` and required partition (`telemetry/raw`).
- Validates `input_artifact` is under `models/frozen_manifolds`.
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Composes script `--output` as `<output_dir>/<run_id>_manifold_eval.json`.

Trigger example:

```bash
airflow dags trigger lojban_eval_manifold \
  --conf '{
    "output_dir": "s3://my-bucket/lojban-prod/telemetry/raw/2026-03-03",
    "input_artifact": "s3://my-bucket/lojban-prod/models/frozen_manifolds/2026-03-03/train_2026_03_03_grounded_reasoner_train.json",
    "run_id": "eval_2026_03_03",
    "logical_accuracy": 0.79,
    "ce_loss_final": 0.95,
    "cross_attention_gain": 0.16
  }'
```

### `lojban_ablation_a_to_g`

- Schedule: manual (`schedule=None`)
- Task: `run_ablation_a_to_g`
- Script: `scripts/run_coconut_ablation_matrix.py`
- Required partition: `telemetry/raw` (recommended subpath: `telemetry/raw/ablation/a_to_g`)
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `base_model` (str, required)
- `adapter` (str, required)
- `b2_adapter` (str, optional; enhanced-constraint text-to-text adapter for `B.2`)
- `drope_adapter` (str, optional)
- `handoff_projection` (str, optional)
- `output_dir` (str, default `artifacts/runs/telemetry/raw/ablation/a_to_g`)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `sample_size` (int, default `24`)
- `dataset_size` (int, default `1000`)
- `max_new_tokens` (int, default `48`)
- `seeds_csv` (str CSV, default `"7,11"`)
- `local_files_only` (bool, default `false`)
- `execute` (bool, default `true`)

Wrapper behavior:

- Validates `output_dir` under `telemetry/raw`.
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Routes script output to `<output_dir>/<run_id>/<timestamp>/...`.
- Emits `A`, `B.1`, `B.2`, `C`, `D`, `E` records (with `B.2` skipped unless `b2_adapter` is provided).

Trigger example:

```bash
airflow dags trigger lojban_ablation_a_to_g \
  --conf '{
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "adapter": "runs/i_series/20260303_031401/h5_checkpoint.pt",
    "output_dir": "s3://my-bucket/lojban-prod/telemetry/raw/ablation/a_to_g/2026-03-03",
    "run_id": "ablation_ag_2026_03_03",
    "sample_size": 24,
    "dataset_size": 1000,
    "max_new_tokens": 48,
    "seeds_csv": "7,11",
    "execute": true
  }'
```

### `lojban_ablation_hypercube_report`

- Schedule: manual (`schedule=None`)
- Task: `build_hypercube_report`
- Script: `scripts/build_airflow_ablation_hypercube_report.py`
- Required partition: `telemetry/raw` (recommended subpath: `telemetry/raw/ablation/hypercube`)
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `output_dir` (str, default `artifacts/runs/telemetry/raw/ablation/hypercube`)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `l6_manifest` (str path, optional; defaults to latest local L6 manifest)
- `j5_summary` (str path, optional; defaults to latest local J-5 summary)

Wrapper behavior:

- Validates `output_dir` under `telemetry/raw`.
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Writes `<output_dir>/<run_id>/ablation_hypercube_report.{json,md}`.

Trigger example:

```bash
airflow dags trigger lojban_ablation_hypercube_report \
  --conf '{
    "output_dir": "s3://my-bucket/lojban-prod/telemetry/raw/ablation/hypercube/2026-03-04",
    "run_id": "m2_hypercube_2026_03_04"
  }'
```

### `lojban_j_series_invariance`

- Schedule: manual (`schedule=None`)
- Tasks: `validate_j_series_contract`, `run_j1_graph_target`, `run_j2_paraphrase_explosion`, `run_j3_stopgrad_gate`, `run_j4_operator_curriculum`, `run_j5_adversarial_synthesis`
- Scripts: `scripts/eval_j_1.py`, `scripts/eval_j_2.py`, `scripts/eval_j_3.py`, `scripts/eval_j_4.py`, `scripts/eval_j_5.py`
- Required partition: `telemetry/raw`
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `output_dir` (str, default `artifacts/runs/telemetry/raw`)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)
- `h5_ood_artifact` (str path, optional)
- `j2_variants_per_graph` (int, default `1000`)
- `j4_per_operator` (int, default `256`)
- `j5_sample_count` (int, default `256`)
- `j5_novelty_threshold` (float in `[0,1]`, default `0.30`)

Wrapper behavior:

- Validates `output_dir` under `telemetry/raw` and rejects S3 outputs because the J-series scripts write local artifacts only.
- Validates `h5_ood_artifact` under `telemetry/raw` and requires it to exist locally when provided.
- Splits the DAG into explicit stage tasks with `validate_j_series_contract -> run_j1_graph_target -> run_j2_paraphrase_explosion`; `J3`, `J4`, and `J5` run as separate validated tasks after contract resolution.
- Validates stage outputs after each script run: J1 must emit a non-empty `graphs` list, J2/J3/J4/J5 must emit required metric keys, and J4/J5 must also materialize their declared dataset sidecars.
- Fails immediately on missing or malformed upstream stage artifacts instead of passing paths through blindly.

Trigger example:

```bash
airflow dags trigger lojban_j_series_invariance \
  --conf '{
    "output_dir": "artifacts/runs/telemetry/raw",
    "run_id": "j_series_2026_03_10",
    "h5_ood_artifact": "artifacts/runs/telemetry/raw/h5_ood_eval.json",
    "j2_variants_per_graph": 512,
    "j4_per_operator": 128,
    "j5_sample_count": 128,
    "j5_novelty_threshold": 0.30
  }'
```

### `lojban_m3_plus_family`

- Schedule: manual (`schedule=None`)
- Task: `run_m3_plus_family`
- Script: `scripts/run_m3_plus_family.py`
- Required partition: `telemetry/raw` for report outputs (recommended subpath: `telemetry/raw/ablation/hypercube/m3_plus`)
- Backfill: disabled (`catchup=False`)

Supported runtime config keys (`dag_run.conf`):

- `base_model` (str, required)
- `adapter` (str, required)
- `train_steps` (int, default `120`)
- `dataset_size` (int, default `1000`)
- `seed` (int, default `7`)
- `local_files_only` (bool, default `false`)
- `l_output_root` (str, default `runs/l_series/m3_plus`)
- `report_output_dir` (str, default `artifacts/runs/telemetry/raw/ablation/hypercube/m3_plus`)
- `j5_summary` (str path, optional)
- `dynamic_arity_signatures` (bool, default `false`)
- `operator_arity_json` (str path, optional)
- `default_relation_arity` (int, default `2`)
- `arity_enforcement_mode` (enum: `legacy_strict`, `registry_strict`, `crystallization`; default `crystallization`)
- `baseline_manifest` (str path, required by policy; default `docs/baselines/m_series_baseline_manifest.json`)
- `run_id` (str, optional; falls back to Airflow `dag_run.run_id`)

Wrapper behavior:

- Validates `l_output_root` under allowed local output roots.
- Validates `report_output_dir` under `telemetry/raw`.
- Rejects S3 paths for M3+ runner outputs (local-only execution contract).
- Validates `baseline_manifest` as an existing local `docs/*.json` file and passes it to the M-series runner.
- Validates `j5_summary` and `operator_arity_json` as existing local files when provided.
- Fails fast if the M-series baseline manifest is missing required `M_BASE` fields.
- Uses crystallization-first arity policy by default (`arity_enforcement_mode=crystallization`) for M-series discovery runs.
- Validates `run_id` against `^[A-Za-z0-9._-]{1,64}$`.
- Writes report output to `<report_output_dir>/<run_id>/m3_plus_<timestamp>/...`.

Trigger example:

```bash
airflow dags trigger lojban_m3_plus_family \
  --conf '{
    "base_model": "C:/Users/Andrew/hf_models/Qwen2.5-0.5B-Instruct",
    "adapter": "runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5",
    "report_output_dir": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_plus",
    "baseline_manifest": "docs/baselines/m_series_baseline_manifest.json",
    "run_id": "m3_plus_2026_03_04",
    "local_files_only": true
  }'
```

## Operations Notes

- Keep `max_active_runs=1` per DAG to avoid overlapping heavyweight runs.
- For production workers, keep `LOJBAN_ALLOW_ENV_OVERRIDES=0` unless explicit path overrides are required.
- Keep pipeline compatibility at the script contract boundary (`--output` artifact + `output_dir` partitioning) to simplify upgrades.
