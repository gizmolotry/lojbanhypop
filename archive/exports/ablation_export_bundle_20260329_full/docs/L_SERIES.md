# L-Series (Lagrangian Series)

The L-Series implements a lexicographic augmented Lagrangian controller to replace static weighted-loss blending.

## Core Objective

For Tier A constraints (arity, scope, identity), the trainer uses:

`L = L_task + sum_i (lambda_i * c_i) + (rho / 2) * sum_i (c_i^2) + sum_j w_j L_j`

Multiplier update each step:

`lambda_i <- max(0, lambda_i + rho * c_i)`

## MVS Entrypoint

- Script: `scripts/train_l_series_mvs.py`
- Output: `runs/l_series/<timestamp>/`
  - `l_series_checkpoint.pt`
  - `l_series_summary.json`

Example:

```powershell
$env:PYTHONPATH="src"
python scripts/train_l_series_mvs.py \
  --base-model <model_or_path> \
  --adapter <adapter_path> \
  --train-steps 200 \
  --rho 0.2 \
  --tier-a-lock-eps 0.02 \
  --tier-a-lock-window 16
```

## Tier Mapping

- Tier A: Arity, Scope, Identity via augmented Lagrangian multipliers.
- Tier B: 3-valued soft valves (Lukasiewicz) + crispness pressure.
- Tier C: Entropy floor + explicit register overflow penalty.

## Airflow

DAG wrapper:

- `airflow/dags/lojban_l_series_dag.py`

The DAG follows existing utility patterns (`merge_conf`, `sanitize_run_id`, `validate_output_partition`, `run_repo_script`) and currently writes to local output paths.
