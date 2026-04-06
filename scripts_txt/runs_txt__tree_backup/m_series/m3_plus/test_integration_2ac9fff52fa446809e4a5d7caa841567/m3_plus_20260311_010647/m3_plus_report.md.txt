# M3+ Family Report

- generated_utc: `2026-03-11T01:06:47.733277+00:00`
- series_id: `M`
- track: `M3+`
- base_model: `dummy-base`
- adapter: `runs\phase5_two_stage_recovery_anchors\test_adapter_2ac9fff52fa446809e4a5d7caa841567`
- dataset_profile: `legacy`
- difficulty_tier: `all`

| run_id | mode | dataset_profile | difficulty_tier | checkpoint_in | checkpoint_out | status | scope | scope_unbound | identity | arity_strict | tier_b | tier_c | run_dir |
|---|---|---|---|---|---|---|---:|---:|---:|---:|---|---|---|
| `M3.0` | `train` | `legacy` | `all` | `` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_0/20260310_000000/l_series_checkpoint.pt` | `ok` | 0.2500 | 0.0800 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_0/20260310_000000` |
| `M3.1` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_0/20260310_000000/l_series_checkpoint.pt` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_1/20260310_000000/l_series_checkpoint.pt` | `ok` | 0.2500 | 0.0800 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_1/20260310_000000` |
| `M3.2` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_1/20260310_000000/l_series_checkpoint.pt` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_2/20260310_000000/l_series_checkpoint.pt` | `ok` | 0.2500 | 0.0800 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_2/20260310_000000` |
| `M3.3` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_2/20260310_000000/l_series_checkpoint.pt` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_3/20260310_000000/l_series_checkpoint.pt` | `ok` | 0.2500 | 0.0800 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_3/20260310_000000` |
| `M3.4` | `train` | `legacy` | `all` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_3/20260310_000000/l_series_checkpoint.pt` | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_4/20260310_000000/l_series_checkpoint.pt` | `ok` | 0.0900 | 0.0100 | 0.0000 | 0.0000 | True | True | `runs/l_series/m3_plus/test_integration_2ac9fff52fa446809e4a5d7caa841567/20260311_010647/m3_4/20260310_000000` |

## Gates

- arity_feasible: `True`
- identity_guardrail_lte_0_05: `True`
- unbound_solved_lt_0_02: `True`
- scope_gate_lt_0_10: `True`
- foil_metric_polarity_check_true_gt_false_rate: `1.0000`
- accepted_foil_pair_accuracy: `0.8125`
- foil_auc_deprecated: `0.8125`
- phase3_graduation_ready: `True`
