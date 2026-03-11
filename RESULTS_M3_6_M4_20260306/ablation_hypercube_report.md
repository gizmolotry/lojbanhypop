# Airflow Ablation Hypercube Report

- run_id: `hypercube_m3_6_1_rerun_m4_rerun_20260306`
- generated_utc: `2026-03-06T13:55:10.758490+00:00`
- l6_manifest: `runs/l_series/l6_ablation/20260305_170454/l6_ablation_manifest.json`
- j5_summary: `runs/j_series/audit_tmp/j-5.json`

## M2 Cells

| run_id | alias | scope | identity | arity_strict | tier_b | tier_c |
|---|---|---:|---:|---:|---|---|
| `L6-A` | `M2.A` | 0.2824 | 0.0000 | 0.0000 | True | True |
| `L6-B` | `M2.B` | 0.3478 | 0.0000 | 0.0000 | False | False |
| `L6-C` | `M2.C` | 0.2892 | 0.0000 | 0.0000 | True | True |

## J-5 Metrics

- generator_accept_rate: `0.087500`
- foil_auc: `1.000000`
- foil_minimal_edit_rate: `1.000000`

## M3.6 / M3.6.2 Metrics

- symmetry_false_foil_rate: `0.000000`
- m3_6_a_total: `1.000000`
- m3_6_b_total: `1.000000`
- m3_6_c_total: `1.000000`
- active_op_count: `1`
- top1_op_share: `1.000000`

## Gate Evaluation

- arity_all_zero: `True`
- best_scope_cell: `L6-A/M2.A` (`0.2824`)
- best_identity_cell: `L6-A/M2.A` (`0.0000`)
- gate_scope_lt_0_10: `False`
- gate_identity_lte_0_05: `True`
- gate_foil_min_edit_gte_0_95: `True`
- phase3_gate_ready: `False`
