# Cross-Referenced M3.8 -> M3.9 Split Results

Copy/paste table:

| bucket | dataset_profile | difficulty_tier | active_tokens | primitive_candidates | mean_scope | mean_ce | m3.8.c checkpoint | m3.9 report json |
|---|---|---|---:|---:|---:|---:|---|---|
| legacy | legacy | all | 61 | 10 | 0.347891 | 13.137092 | runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110/l_series_checkpoint.pt | artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/m3_9_primitive_probe_report.json |
| easy | diverse_v2 | easy | 33 | 10 | 0.376876 | 3.33385 | runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238/l_series_checkpoint.pt | artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/m3_9_primitive_probe_report.json |
| medium | diverse_v2 | medium | 79 | 10 | 0.340161 | 8.185414 | runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404/l_series_checkpoint.pt | artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/m3_9_primitive_probe_report.json |
| hard | diverse_v2 | hard | 87 | 10 | 0.280296 | 2.984679 | runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545/l_series_checkpoint.pt | artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/m3_9_primitive_probe_report.json |

Detailed paths by bucket:

## legacy
- m3_8_c_run_dir: runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110
- m3_8_c_checkpoint: runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110/l_series_checkpoint.pt
- m3_8_c_summary: runs/l_series/m3_8_diversification/20260307_013004/legacy/m3_8_c/20260307_013110/l_series_summary.json
- m3_9_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/m3_9_primitive_probe_report.json
- m3_9_report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/m3_9_primitive_probe_report.md
- m3_9_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/primitive_candidate_list.json
- m3_9_clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_legacy_20260307/grammatical_cluster_report.json

## easy
- m3_8_c_run_dir: runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238
- m3_8_c_checkpoint: runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238/l_series_checkpoint.pt
- m3_8_c_summary: runs/l_series/m3_8_diversification/20260307_013004/easy/m3_8_c/20260307_013238/l_series_summary.json
- m3_9_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/m3_9_primitive_probe_report.json
- m3_9_report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/m3_9_primitive_probe_report.md
- m3_9_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/primitive_candidate_list.json
- m3_9_clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_easy_20260307/grammatical_cluster_report.json

## medium
- m3_8_c_run_dir: runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404
- m3_8_c_checkpoint: runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404/l_series_checkpoint.pt
- m3_8_c_summary: runs/l_series/m3_8_diversification/20260307_013004/medium/m3_8_c/20260307_013404/l_series_summary.json
- m3_9_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/m3_9_primitive_probe_report.json
- m3_9_report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/m3_9_primitive_probe_report.md
- m3_9_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/primitive_candidate_list.json
- m3_9_clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_medium_20260307/grammatical_cluster_report.json

## hard
- m3_8_c_run_dir: runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545
- m3_8_c_checkpoint: runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545/l_series_checkpoint.pt
- m3_8_c_summary: runs/l_series/m3_8_diversification/20260307_013004/hard/m3_8_c/20260307_013545/l_series_summary.json
- m3_9_report_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/m3_9_primitive_probe_report.json
- m3_9_report_md: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/m3_9_primitive_probe_report.md
- m3_9_candidates_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/primitive_candidate_list.json
- m3_9_clusters_json: artifacts/runs/telemetry/raw/ablation/hypercube/m3_9_primitive_probe_split/m3_9_hard_20260307/grammatical_cluster_report.json

