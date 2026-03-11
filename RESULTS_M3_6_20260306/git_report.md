# Git Report

## status --short

```
 M airflow/dags/lojban_airflow_utils.py
 M airflow/dags/lojban_j_series_dag.py
 M docs/AIRFLOW_ORCHESTRATION.md
 M docs/H5_ABLATION_EXTENSION.md
 D runs/20260219_213448/history.json
 D runs/20260219_213448/summary.md
 D runs/coconut_ablation_matrix/20260226_090029/ablation_matrix.md
 D runs/swiglu_bridge_report_h3.json
 D runs/true_coconut_h_series/20260228_190640/h3_seed11.json
 D runs/true_coconut_h_series/20260228_190640/h3_seed7.json
 D runs/true_coconut_h_series/20260228_190640/run_h_series.json
 D runs/true_coconut_h_series/20260228_190640/run_h_series.md
 M scripts/eval_hf_dual_mode_gate.py
 M scripts/run_coconut_ablation_matrix.py
 M scripts/run_true_coconut_h_series.py
 M scripts/train_h5_persistent_vq_advisor.py
 M scripts/train_l_series_mvs.py
 M scripts/train_swiglu_bridge.py
 M src/lojban_evolution/j_series_eval.py
 M src/lojban_evolution/l_series.py
 M tests/test_h5_ablation_extension_contract.py
?? .env.example
?? .github/
?? AUDIT_REPORT.md
?? CANONICAL_LEDGER.json
?? CANONICAL_LEDGER.md
?? "Core Concept_ Self-Optimizing Language Evolution.txt"
?? FINAL_THESIS_REPORT.md
?? GRANULAR_DATAPACK.json
?? H5_ABLATION_REPORT.md
?? H5_DATA_MANIFEST.json
?? H5_SUMMARY.md
?? NUMERICAL_AUDIT.md
?? REPORTS_RELEVANT/
?? RESULTS_FULL_GRID_20260305/
?? RESULTS_M3_6_20260306/
?? RESULTS_M_POLICY_20260305/
?? RESULTS_NARY_AIRFLOW_20260305/
?? RESULTS_NEW_20260305/
?? TECHNICAL_APPENDIX.md
?? airflow/dags/lojban_ablation_hypercube_report_dag.py
?? airflow/dags/lojban_ablation_matrix_dag.py
?? airflow/dags/lojban_m3_5_symmetry_dag.py
?? airflow/dags/lojban_m3_6_symmetry_oracle_dag.py
?? airflow/dags/lojban_m3_plus_dag.py
?? airflow/dags/lojban_m4_series_dag.py
?? configs/
?? docs/CAUSAL_PROBE_PROTOCOL.md
?? docs/DECISIONS.md
?? docs/SERIES_CHARTER.md
?? docs/baselines/
?? runs_archive_20260301_185218.zip
?? scripts/build_airflow_ablation_hypercube_report.py
?? scripts/build_english_cot_control_dataset.py
?? scripts/build_full_coconut_report.py
?? scripts/build_gguf_pack.py
?? scripts/build_lora_dataset.py
?? scripts/build_mixed_curriculum_dataset.py
?? scripts/build_science_metrics_pack.py
?? scripts/build_synthetic_lora_dataset.py
?? scripts/eval_hf_adapter.py
?? scripts/eval_hf_dual_mode_handoff.py
?? scripts/eval_j_5.py
?? scripts/eval_with_lms.py
?? scripts/latent_handoff_eval.py
?? scripts/mine_compositional_anchors.py
?? scripts/multi_stage_cot.py
?? scripts/reconstruct_lora_dataset.py
?? scripts/run_causal_probe_matrix.py
?? scripts/run_drope_recalibration.py
?? scripts/run_english_cot_control_duel.py
?? scripts/run_l6_ablation_branch.py
?? scripts/run_m3_5_symmetry.py
?? scripts/run_m3_6_symmetry_oracle.py
?? scripts/run_m3_plus_family.py
?? scripts/run_m4_series.py
?? scripts/run_phase5_objective_ablation.py
?? scripts/run_phase5_train_ablation.py
?? scripts/run_phase5_two_stage_recovery.py
?? scripts/run_three_engine_comparison.py
?? scripts/tasks.ps1
?? scripts/train_babel.py
?? scripts/train_cpu_hf.ps1
?? scripts/train_h5_slice2_bridge.py
?? scripts/train_mixed_curriculum_cpu.ps1
?? scripts/train_vq_reasoning_pilot.py
?? scripts/verify_h5_ablation.py
?? scripts/visualize_shock.py
?? src/lojban_evolution/series_contract.py
?? tests/test_j_series_eval.py
?? tests/test_series_contract.py
```

## diff --stat

```
 airflow/dags/lojban_airflow_utils.py               |   16 +
 airflow/dags/lojban_j_series_dag.py                |   20 +
 docs/AIRFLOW_ORCHESTRATION.md                      |  140 +-
 docs/H5_ABLATION_EXTENSION.md                      |   16 +-
 runs/20260219_213448/history.json                  |  283 -
 runs/20260219_213448/summary.md                    |   31 -
 .../20260226_090029/ablation_matrix.md             |   22 -
 runs/swiglu_bridge_report_h3.json                  |   35 -
 .../20260228_190640/h3_seed11.json                 | 7296 --------------------
 .../20260228_190640/h3_seed7.json                  | 7296 --------------------
 .../20260228_190640/run_h_series.json              |   59 -
 .../20260228_190640/run_h_series.md                |    7 -
 scripts/eval_hf_dual_mode_gate.py                  |    6 +-
 scripts/run_coconut_ablation_matrix.py             |   85 +-
 scripts/run_true_coconut_h_series.py               |   48 +-
 scripts/train_h5_persistent_vq_advisor.py          |  584 +-
 scripts/train_l_series_mvs.py                      |  486 +-
 scripts/train_swiglu_bridge.py                     |    2 +-
 src/lojban_evolution/j_series_eval.py              |  588 +-
 src/lojban_evolution/l_series.py                   |  190 +-
 tests/test_h5_ablation_extension_contract.py       |   18 +-
 21 files changed, 1695 insertions(+), 15533 deletions(-)
```
