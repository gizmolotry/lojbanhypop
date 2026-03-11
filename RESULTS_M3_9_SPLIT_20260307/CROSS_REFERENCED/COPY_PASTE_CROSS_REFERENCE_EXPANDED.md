# Expanded Cross-Referenced Results (M3.8 -> M3.9, Split by Bucket)

Generated UTC: 2026-03-07T02:14:36.9319080Z

## Executive Matrix

| bucket | profile | tier | active_tokens | primitive_candidates | mean_scope | mean_ce | m3.8 op_entropy | m3.8 top1_share | m3.8 div_loss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| legacy | legacy | all | 61 | 10 | 0.347891 | 13.137092 | 1.609222 | 0.20669 | 0.551854 |
| easy | diverse_v2 | easy | 33 | 10 | 0.376876 | 3.33385 | 1.609181 | 0.206381 | 0.557218 |
| medium | diverse_v2 | medium | 79 | 10 | 0.340161 | 8.185414 | 1.609383 | 0.203747 | 0.605804 |
| hard | diverse_v2 | hard | 87 | 10 | 0.280296 | 2.984679 | 1.609332 | 0.202934 | 0.560377 |

## Bucket: legacy

### Upstream (M3.8.C)
- run_dir: runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110
- summary_path: runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110/l_series_summary.json
- checkpoint_path: runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110/l_series_checkpoint.pt
- diversification_mode: family_cluster
- operator_entropy: 1.609222173690796
- operator_top1_share: 0.2066899538040161
- diversification_loss: 0.5518538951873779
- shadow_loss: 0.1187596395611763

### Downstream (M3.9)
- run_dir: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307
- report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/m3_9_primitive_probe_report.json
- report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/m3_9_primitive_probe_report.md
- primitive_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/primitive_candidate_list.json
- clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/grammatical_cluster_report.json
- token_role_matrix_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/token_role_matrix.json
- token_position_entropy_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/token_position_entropy.json
- primitive_token_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/primitive_token_report.json
- active_token_count: 61
- primitive_candidate_count: 10
- mean_baseline_scope: 0.34789143967811476
- mean_baseline_ce_loss: 13.137092158198357

### Top Primitive Candidates (Top 20)

| token_id | score | specialization | causal_score | position_entropy | samples_with_token | del_ce | del_scope | sub_ce | swap_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 499 | 0.605418 | 1 | 0.210835 | 0 | 8 | -0.000268 | -0.262263 | 3.3E-05 | 0.000103 |
| 274 | 0.605416 | 1 | 0.210832 | 0 | 8 | -0.0003 | -0.262263 | 7.5E-05 | 1.4E-05 |
| 914 | 0.604044 | 1 | 0.208088 | 0 | 3 | 0 | -0.268938 | 0 | 0 |
| 3 | 0.593559 | 1 | 0.187117 | 0 | 8 | 0 | -0.147783 | 0 | 0 |
| 729 | 0.585497 | 1 | 0.170994 | 0 | 5 | 0 | -0.049263 | 0 | 0 |
| 0 | 0.584078 | 0.930415 | 0.23774 | 0.111992 | 8 | -0.003581 | -0.254234 | 0.002941 | -0.000465 |
| 1511 | 0.574704 | 1 | 0.149407 | 0 | 8 | 0 | -0.118927 | 0 | 0 |
| 1054 | 0.555143 | 1 | 0.110287 | 0 | 3 | 0 | -0.179339 | 0 | 0 |
| 504 | 0.548124 | 0.894044 | 0.202205 | 0.17053 | 5 | 0 | -0.114583 | 0 | 0 |
| 505 | 0.470235 | 0.830184 | 0.110287 | 0.273308 | 3 | 0 | -0.179339 | 0 | 0 |

### Grammatical Clusters

| cluster_id | label | member_count | dominant_role | member_tokens |
|---|---|---:|---|---|
| 0 | predicate_heads | 1 | predicate_head | 0 |
| 1 | predicate_heads | 1 | predicate_head | 3 |
| 2 | referential_markers | 14 | arg_slot_1 | 61,359,499,729,914,944,1032,1054,1268,1511,1585,1695,1895,1919 |
| 3 | referential_markers | 28 | arg_slot_2 | 75,274,302,304,315,504,505,533,551,555,608,704,772,793,830,853,920,975,989,1194,1232,1333,1396,1458,1483,1544,1638,1998 |
| 4 | referential_markers | 17 | arg_slot_1 | 182,453,564,639,659,734,865,908,915,990,999,1022,1248,1261,1866,1910,1971 |

### Notes
- Candidate scores are composite (causal intervention magnitude + positional specialization).
- Compare medium/hard against legacy for emergence of broader primitive inventories.

## Bucket: easy

### Upstream (M3.8.C)
- run_dir: runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238
- summary_path: runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238/l_series_summary.json
- checkpoint_path: runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238/l_series_checkpoint.pt
- diversification_mode: family_cluster
- operator_entropy: 1.6091806888580322
- operator_top1_share: 0.20638099312782288
- diversification_loss: 0.5572179555892944
- shadow_loss: 0.07978913187980652

### Downstream (M3.9)
- run_dir: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307
- report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/m3_9_primitive_probe_report.json
- report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/m3_9_primitive_probe_report.md
- primitive_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/primitive_candidate_list.json
- clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/grammatical_cluster_report.json
- token_role_matrix_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/token_role_matrix.json
- token_position_entropy_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/token_position_entropy.json
- primitive_token_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/primitive_token_report.json
- active_token_count: 33
- primitive_candidate_count: 10
- mean_baseline_scope: 0.3768763429490443
- mean_baseline_ce_loss: 3.3338495176285505

### Top Primitive Candidates (Top 20)

| token_id | score | specialization | causal_score | position_entropy | samples_with_token | del_ce | del_scope | sub_ce | swap_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1625 | 0.605533 | 1 | 0.211066 | 0 | 3 | 0 | -0.278076 | 0 | 0 |
| 1 | 0.605456 | 1 | 0.210913 | 0 | 8 | 0 | -0.231953 | 0 | 0 |
| 1650 | 0.601077 | 1 | 0.202154 | 0 | 5 | 0 | -0.203194 | 0 | 0 |
| 597 | 0.592991 | 1 | 0.185982 | 0 | 5 | 0 | -0.267236 | 0 | 0 |
| 1146 | 0.592991 | 1 | 0.185982 | 0 | 5 | 0 | -0.267236 | 0 | 0 |
| 21 | 0.585782 | 1 | 0.171564 | 0 | 5 | 0 | -0.236255 | 0 | 0 |
| 1811 | 0.577265 | 0.952375 | 0.202154 | 0.076649 | 5 | 0 | -0.203194 | 0 | 0 |
| 3 | 0.573623 | 0.918252 | 0.228994 | 0.131568 | 8 | 0.000553 | -0.298708 | 0.000502 | -5E-06 |
| 1900 | 0.515678 | 0.859793 | 0.171564 | 0.225655 | 5 | 0 | -0.236255 | 0 | 0 |
| 579 | 0.41695 | 0.642944 | 0.190956 | 0.57466 | 8 | 0 | -0.195222 | 0 | 0 |

### Grammatical Clusters

| cluster_id | label | member_count | dominant_role | member_tokens |
|---|---|---:|---|---|
| 0 | predicate_heads | 1 | predicate_head | 1 |
| 1 | predicate_heads | 1 | predicate_head | 3 |
| 2 | referential_markers | 15 | arg_slot_2 | 7,190,218,890,938,953,1146,1206,1404,1601,1625,1778,1782,1811,1900 |
| 3 | referential_markers | 9 | arg_slot_1 | 21,46,494,597,828,1088,1178,1346,1650 |
| 4 | referential_markers | 7 | arg_slot_1 | 194,316,579,1139,1292,1641,1737 |

### Notes
- Candidate scores are composite (causal intervention magnitude + positional specialization).
- Compare medium/hard against legacy for emergence of broader primitive inventories.

## Bucket: medium

### Upstream (M3.8.C)
- run_dir: runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404
- summary_path: runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404/l_series_summary.json
- checkpoint_path: runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404/l_series_checkpoint.pt
- diversification_mode: family_cluster
- operator_entropy: 1.6093828678131104
- operator_top1_share: 0.2037467360496521
- diversification_loss: 0.6058041453361511
- shadow_loss: 0.04740815982222557

### Downstream (M3.9)
- run_dir: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307
- report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/m3_9_primitive_probe_report.json
- report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/m3_9_primitive_probe_report.md
- primitive_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/primitive_candidate_list.json
- clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/grammatical_cluster_report.json
- token_role_matrix_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/token_role_matrix.json
- token_position_entropy_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/token_position_entropy.json
- primitive_token_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/primitive_token_report.json
- active_token_count: 79
- primitive_candidate_count: 10
- mean_baseline_scope: 0.3401606514678994
- mean_baseline_ce_loss: 8.185414001345634

### Top Primitive Candidates (Top 20)

| token_id | score | specialization | causal_score | position_entropy | samples_with_token | del_ce | del_scope | sub_ce | swap_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.610532 | 1 | 0.221063 | 0 | 8 | 0.006135 | -0.187206 | 0.007738 | 0.004888 |
| 882 | 0.597251 | 1 | 0.194501 | 0 | 4 | 0.000215 | -0.289599 | 0.000377 | -3E-06 |
| 97 | 0.597246 | 1 | 0.194492 | 0 | 4 | 0.00017 | -0.289599 | -4.6E-05 | 0.000332 |
| 984 | 0.590117 | 1 | 0.180235 | 0 | 5 | 0.000175 | -0.263878 | 0.000176 | 0.000174 |
| 4 | 0.570064 | 1 | 0.140129 | 0 | 5 | 0 | -0.154324 | 0 | 0 |
| 875 | 0.563911 | 0.904915 | 0.222907 | 0.153033 | 4 | -1E-06 | -0.329205 | -2E-06 | -1E-06 |
| 256 | 0.561998 | 1 | 0.123995 | 0 | 5 | 0 | -0.14684 | 0 | 0 |
| 1423 | 0.561998 | 1 | 0.123995 | 0 | 5 | 0 | -0.14684 | 0 | 0 |
| 0 | 0.524902 | 0.85265 | 0.197153 | 0.23715 | 8 | -0.001613 | -0.107344 | -0.002185 | 0.002261 |
| 131 | 0.521686 | 1 | 0.043372 | 0 | 4 | 0 | -0.04842 | 0 | 0 |

### Grammatical Clusters

| cluster_id | label | member_count | dominant_role | member_tokens |
|---|---|---:|---|---|
| 1 | predicate_heads | 4 | predicate_head | 0,2,3,4 |
| 4 | referential_markers | 37 | arg_slot_2 | 5,16,47,72,182,189,244,289,363,383,499,516,559,825,858,875,882,990,1058,1079,1100,1122,1184,1263,1264,1283,1303,1304,1338,1423,1433,1544,1602,1711,1824,1960,1979 |
| 3 | referential_markers | 24 | arg_slot_1 | 82,86,103,131,219,231,270,631,643,781,932,1042,1053,1059,1072,1111,1207,1386,1393,1467,1623,1717,1750,1921 |
| 2 | referential_markers | 2 | arg_slot_1 | 97,984 |
| 0 | referential_markers | 12 | arg_slot_1 | 256,385,425,544,672,773,1273,1306,1368,1395,1406,1888 |

### Notes
- Candidate scores are composite (causal intervention magnitude + positional specialization).
- Compare medium/hard against legacy for emergence of broader primitive inventories.

## Bucket: hard

### Upstream (M3.8.C)
- run_dir: runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545
- summary_path: runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545/l_series_summary.json
- checkpoint_path: runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545/l_series_checkpoint.pt
- diversification_mode: family_cluster
- operator_entropy: 1.6093316078186035
- operator_top1_share: 0.2029338926076889
- diversification_loss: 0.5603771209716797
- shadow_loss: 0.07288209348917007

### Downstream (M3.9)
- run_dir: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307
- report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/m3_9_primitive_probe_report.json
- report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/m3_9_primitive_probe_report.md
- primitive_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/primitive_candidate_list.json
- clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/grammatical_cluster_report.json
- token_role_matrix_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/token_role_matrix.json
- token_position_entropy_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/token_position_entropy.json
- primitive_token_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/primitive_token_report.json
- active_token_count: 87
- primitive_candidate_count: 10
- mean_baseline_scope: 0.2802961778447583
- mean_baseline_ce_loss: 2.984679162502289

### Top Primitive Candidates (Top 20)

| token_id | score | specialization | causal_score | position_entropy | samples_with_token | del_ce | del_scope | sub_ce | swap_ce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1184 | 0.60057 | 1 | 0.20114 | 0 | 8 | -0.00224 | -0.067069 | 0.000818 | -0.00224 |
| 1226 | 0.600426 | 1 | 0.200851 | 0 | 8 | -0.003047 | -0.067069 | -0.000806 | -2E-06 |
| 1285 | 0.593474 | 1 | 0.186949 | 0 | 8 | 0 | -0.179593 | 0 | 0 |
| 1888 | 0.592094 | 1 | 0.184189 | 0 | 8 | 0 | -0.061092 | 0 | 0 |
| 2 | 0.577833 | 1 | 0.155666 | 0 | 8 | 0 | -0.162042 | 0 | 0 |
| 1729 | 0.572857 | 1 | 0.145713 | 0 | 8 | 0 | -0.164673 | 0 | 0 |
| 4 | 0.570469 | 1 | 0.140938 | 0 | 8 | 0 | -0.168673 | 0 | 0 |
| 399 | 0.559633 | 1 | 0.119265 | 0 | 8 | 0 | -0.138395 | 0 | 0 |
| 1 | 0.537462 | 0.882094 | 0.19283 | 0.189762 | 8 | 0.003639 | -0.179526 | 0.003136 | 9E-06 |
| 1130 | 0.491528 | 0.854737 | 0.128319 | 0.233792 | 8 | 0 | -0.112824 | 0 | 0 |

### Grammatical Clusters

| cluster_id | label | member_count | dominant_role | member_tokens |
|---|---|---:|---|---|
| 1 | predicate_heads | 1 | predicate_head | 1 |
| 3 | predicate_heads | 2 | predicate_head | 2,4 |
| 2 | predicate_heads | 1 | predicate_head | 3 |
| 4 | referential_markers | 42 | arg_slot_1 | 28,55,74,157,160,188,238,249,272,399,437,451,473,517,567,595,602,703,732,733,807,864,915,1076,1096,1126,1187,1226,1285,1378,1460,1521,1540,1710,1786,1838,1882,1919,1925,1938,1964,1972 |
| 0 | referential_markers | 41 | arg_slot_2 | 38,177,393,405,432,457,534,536,579,584,646,667,672,674,700,705,729,839,840,855,860,901,1007,1100,1130,1184,1315,1458,1517,1525,1543,1582,1605,1729,1783,1809,1810,1878,1879,1888,1928 |

### Notes
- Candidate scores are composite (causal intervention magnitude + positional specialization).
- Compare medium/hard against legacy for emergence of broader primitive inventories.

