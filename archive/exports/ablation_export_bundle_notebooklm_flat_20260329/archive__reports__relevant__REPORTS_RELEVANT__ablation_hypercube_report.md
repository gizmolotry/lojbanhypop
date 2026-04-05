# Airflow Ablation Hypercube Report

- run_id: `m2_hypercube_20260304`
- generated_utc: `2026-03-04T19:50:16.037113+00:00`
- l6_manifest: `runs/l_series/l6_ablation/20260304_175726/l6_ablation_manifest.json`
- j5_summary: `runs/j_series/20260304_175527/j-5.json`

## M2 Cells

| run_id | alias | scope | identity | arity_strict | tier_b | tier_c |
|---|---|---:|---:|---:|---|---|
| `L6-A` | `M2.A` | 0.3200 | 0.1119 | 0.0000 | True | True |
| `L6-B` | `M2.B` | 0.3380 | 0.0038 | 0.0000 | False | False |
| `L6-C` | `M2.C` | 0.2759 | 0.0000 | 0.0000 | True | True |

## J-5 Metrics

- generator_accept_rate: `0.214844`
- foil_auc: `0.310547`
- foil_minimal_edit_rate: `1.000000`

## Gate Evaluation

- arity_all_zero: `True`
- best_scope_cell: `L6-C/M2.C` (`0.2759`)
- best_identity_cell: `L6-C/M2.C` (`0.0000`)
- gate_scope_lt_0_10: `False`
- gate_identity_lte_0_05: `True`
- gate_foil_min_edit_gte_0_95: `True`
- phase3_gate_ready: `False`
