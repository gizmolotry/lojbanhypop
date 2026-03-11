# Airflow Ablation Hypercube Report

- run_id: `hypercube_m3_8_split_20260307`
- generated_utc: `2026-03-07T01:36:22.839592+00:00`
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
- m3_7_cell_count_ok: `3`
- m3_7_best_scope: `0.0`
- m3_7_best_identity: `0.0`
- m3_7_mean_shadow_loss: `0.032760`
- m3_8_cell_count_ok: `12`
- m3_8_mean_operator_entropy: `1.609288`
- m3_8_mean_operator_top1_share: `0.204683`
- m3_8_mean_diversification_loss: `-0.283944`
- m3_9_active_token_count: `55`
- m3_9_primitive_candidate_count: `10`
- m3_9_mean_baseline_ce_loss: `13.086168`
- m3_9_mean_baseline_scope: `0.371933`

## Gate Evaluation

- arity_all_zero: `True`
- best_scope_cell: `L6-A/M2.A` (`0.2824`)
- best_identity_cell: `L6-A/M2.A` (`0.0000`)
- gate_scope_lt_0_10: `False`
- gate_identity_lte_0_05: `True`
- gate_foil_min_edit_gte_0_95: `True`
- phase3_gate_ready: `False`
