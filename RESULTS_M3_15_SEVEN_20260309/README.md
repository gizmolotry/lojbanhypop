# RESULTS_M3_15_SEVEN_20260309

## Contents
- `m3_15_rotary_coconut_seven_manifest.json`: sweep status/paths for R1-R7.
- `m3_15_rotary_coconut_seven_aggregate.md`: compact metric table.
- `run_m3_15_rotary_coconut.py`: core M3.15 runner used for each run.
- `run_m3_15_rotary_coconut_seven.py`: seven-run harness.
- `dag/`: airflow DAG wrappers for single and seven-run execution.
- `runs/R*/`: per-run raw artifacts and reports.

## Seven-Run Matrix
| Run | Delta | Status | Output |
|---|---|---|---|
| R1 | baseline defaults | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R1` |
| R2 | align_weight=0.0 | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R2` |
| R3 | margin_weight=0.0 | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R3` |
| R4 | margin=0.4, margin_weight=1.0 | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R4` |
| R5 | runtime_gate_cap=0.0 | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R5` |
| R6 | align_weight=1.5, max_nodes=16 | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R6` |
| R7 | runtime_enable_min_acc_gain=0.0 | ok | `D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_15_rotary_coconut_seven/m3_15_seven_20260309/R7` |

## Notes
- This bundle uses the repaired M3.15 core (relation-bias dim guard, dtype-safe advisor hook path, fixed eval alignment metrics).
- Seven-run sweep completed via mixed execution: R1-R5 from harness batch; R6-R7 completed directly after timeout recovery.