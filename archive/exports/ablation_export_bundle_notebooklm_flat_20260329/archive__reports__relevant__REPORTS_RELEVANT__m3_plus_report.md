# M3+ Family Report

- generated_utc: `2026-03-04T22:29:41.475989+00:00`
- base_model: `C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct`
- adapter: `runs\phase5_two_stage_recovery_anchors\20260302_030738\stage2_phase5`

| run_id | status | scope | scope_unbound | identity | arity_strict | tier_b | tier_c | run_dir |
|---|---|---:|---:|---:|---:|---|---|---|
| `M3.0` | `ok` | 0.3288 | 0.0890 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/20260304_220300/m3_0/20260304_220308` |
| `M3.1` | `ok` | 0.3478 | 0.0725 | 0.0892 | 0.0000 | True | True | `runs/l_series/m3_plus/20260304_220300/m3_1/20260304_220830` |
| `M3.2` | `ok` | 0.3478 | 0.0725 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/20260304_220300/m3_2/20260304_221344` |
| `M3.3` | `ok` | 0.3478 | 0.0725 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/20260304_220300/m3_3/20260304_221900` |
| `M3.4` | `ok` | 0.3582 | 0.0672 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/20260304_220300/m3_4/20260304_222423` |

## Gates

- arity_feasible: `True`
- identity_guardrail_lte_0_05: `True`
- unbound_solved_lt_0_02: `False`
- scope_gate_lt_0_10: `False`
- foil_metric_polarity_check_true_gt_false_rate: `1.0000`
- foil_auc_reported: `0.3105`
- phase3_graduation_ready: `False`
