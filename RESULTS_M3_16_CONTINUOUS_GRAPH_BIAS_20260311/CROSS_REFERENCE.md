# Cross Reference

## Primary run
- Report JSON: [m3_16_report.json](/D:/lojbanhypop/RESULTS_M3_16_CONTINUOUS_GRAPH_BIAS_20260311/m3_16_report.json)
- Report MD: [m3_16_report.md](/D:/lojbanhypop/RESULTS_M3_16_CONTINUOUS_GRAPH_BIAS_20260311/m3_16_report.md)
- Source run dir: [m3_16_20260311_bridge_base_v1_small](/D:/lojbanhypop/artifacts/runs/telemetry/raw/ablation/hypercube/m3_16_continuous_graph_bias/bridge_base_v1_small/m3_16_20260311_bridge_base_v1_small)

## Inputs
- Baseline manifest: [m_series_bridge_baseline_manifest.json](/D:/lojbanhypop/docs/baselines/m_series_bridge_baseline_manifest.json)
- Frozen bridge checkpoint: [l_series_checkpoint.pt](/D:/lojbanhypop/runs/l_series/m3_8_diversification/decollapse_small/20260311_103934/hard/m3_8_b/20260311_104736/l_series_checkpoint.pt)
- Adapter path: `runs/phase5_two_stage_recovery_anchors/20260302_030738/stage2_phase5`
- Base model: `C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct`

## Related prior runs
- M3.15b guarded bridge base bundle: [RESULTS_M3_15B_BRIDGE_BASE_20260311](/D:/lojbanhypop/RESULTS_M3_15B_BRIDGE_BASE_20260311)
- M3.15d answer-path forcing bundle: [RESULTS_M3_15D_ANSWER_PATH_20260311](/D:/lojbanhypop/RESULTS_M3_15D_ANSWER_PATH_20260311)
- Root recent M discoveries: [ROOT_RECENT_M_DISCOVERIES_20260311.md](/D:/lojbanhypop/ROOT_RECENT_M_DISCOVERIES_20260311.md)

## Notes
- A larger `bridge_base_v1` run was started first and timed out after producing partial artifacts (`A`, `B`, `B_seed2`).
- The completed, comparable run used `train_steps=40` and `eval_size=32`.
