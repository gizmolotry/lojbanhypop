# M3+ Family Report

- generated_utc: `2026-03-10T23:01:25.503498+00:00`
- series_id: `M`
- track: `M3+`
- base_model: `C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct`
- adapter: `runs\phase5_two_stage_recovery_anchors\20260302_030738\stage2_phase5`
- dataset_profile: `legacy`
- difficulty_tier: `all`

| run_id | mode | dataset_profile | difficulty_tier | checkpoint_in | checkpoint_out | status | scope | scope_unbound | identity | arity_strict | tier_b | tier_c | run_dir |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `M3.0` | `train` | `legacy` | `all` | `` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_0/20260310_222920/l_series_checkpoint.pt` | `ok` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_0/20260310_222920` |
| `M3.1` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_0/20260310_222920/l_series_checkpoint.pt` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_1/20260310_223434/l_series_checkpoint.pt` | `ok` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_1/20260310_223434` |
| `M3.2` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_1/20260310_223434/l_series_checkpoint.pt` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_2/20260310_223944/l_series_checkpoint.pt` | `ok` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_2/20260310_223944` |
| `M3.3` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_2/20260310_223944/l_series_checkpoint.pt` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_3/20260310_224555/l_series_checkpoint.pt` | `ok` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_3/20260310_224555` |
| `M3.4` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_3/20260310_224555/l_series_checkpoint.pt` | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_4/20260310_225323/l_series_checkpoint.pt` | `ok` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/refactor_clean/20260310_222914/m3_4/20260310_225323` |

## Gates

- arity_feasible: `True`
- identity_guardrail_lte_0_05: `True`
- unbound_solved_lt_0_02: `True`
- scope_gate_lt_0_10: `True`
- foil_metric_polarity_check_true_gt_false_rate: `1.0000`
- accepted_foil_pair_accuracy: `1.0000`
- foil_auc_deprecated: `1.0000`
- phase3_graduation_ready: `True`
