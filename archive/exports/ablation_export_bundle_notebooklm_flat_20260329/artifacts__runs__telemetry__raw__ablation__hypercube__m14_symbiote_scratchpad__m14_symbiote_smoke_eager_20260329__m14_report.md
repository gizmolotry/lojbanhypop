# M14 Symbiote Scratchpad

| Cell | Regime | Acc | FTok | Fluency | Contam | S-Bleed | Loop | Answer Delta | Gold On-Off | S-Attn | S-Norm | Scope |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | scratchpad-only control | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.3247 | 0.0000 | 0.0433 | 0.0000 | 0.0000 |
| B | strict residual scratchpad | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.3245 | -0.0001 | 0.0433 | 0.0063 | 0.4068 |
| C | relaxed residual scratchpad | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.3251 | 0.0003 | 0.0433 | 0.0072 | 0.4068 |
| D | severance-threshold residual scratchpad | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.3247 | 0.0003 | 0.0433 | 0.0064 | 0.4068 |
| E | token-only scratchpad baseline | 1.000 | 0.000 | 1.000 | 0.000 | 0.000 | 0.000 | 4.3225 | -0.0003 | 0.0433 | 0.0711 | 0.0000 |

## Comparison References
- M3.18.D: `{"report": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_18_decoder_reentry_resume/m3_18_d_sft_smoke_20260328/m3_18_report.json", "overall_accuracy": 0.5, "resume_first_token_accuracy": 0.0, "english_fluency_score": 1.0, "loop_rate": 0.0, "mean_intervention_delta_gold": -4.251301288604736e-05}`
- M3.19: `{"report": "artifacts/runs/telemetry/raw/ablation/hypercube/m3_19_d_mainline_grid/m3_19_grid_smoke_v2_20260329/m3_19_grid_report.json", "cells": {"D0": {"overall_accuracy": 0.0, "resume_first_token_accuracy": 0.0, "english_fluency_score": 1.0, "loop_rate": 0.0, "mean_intervention_delta_gold": 1.0251998901367188e-05}, "D1": {"overall_accuracy": 0.0, "resume_first_token_accuracy": 0.0, "english_fluency_score": 1.0, "loop_rate": 0.0, "mean_intervention_delta_gold": -2.1457672119140625e-06}, "D2": {"overall_accuracy": 0.0, "resume_first_token_accuracy": 0.0, "english_fluency_score": 1.0, "loop_rate": 0.0, "mean_intervention_delta_gold": -2.1457672119140625e-06}, "D3": {"overall_accuracy": 0.0, "resume_first_token_accuracy": 0.0, "english_fluency_score": 1.0, "loop_rate": 0.0, "mean_intervention_delta_gold": -2.1457672119140625e-06}}}`
- M11.discriminative: `{"manifest": "RESULTS_M10_FINAL_AUDIT/m11_discriminative_manifest.json", "headline_accuracy": 0.85916, "headline_macro_f1": 0.6287, "num_samples": 1200}`

## Promotion Gates
- B:
  - positive_intervention_delta: `False`
  - resume_first_token_gain: `True`
  - fluency_preserved: `True`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `True`
  - scratchpad_bleed_below_threshold: `True`
  - seed_stability: `False`
  - promote_to_next: `False`
- C:
  - positive_intervention_delta: `False`
  - resume_first_token_gain: `True`
  - fluency_preserved: `True`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `True`
  - scratchpad_bleed_below_threshold: `True`
  - seed_stability: `False`
  - promote_to_next: `False`
- D:
  - positive_intervention_delta: `False`
  - resume_first_token_gain: `True`
  - fluency_preserved: `True`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `True`
  - scratchpad_bleed_below_threshold: `True`
  - seed_stability: `False`
  - promote_to_next: `False`
- E:
  - positive_intervention_delta: `False`
  - resume_first_token_gain: `True`
  - fluency_preserved: `True`
  - contamination_below_threshold: `True`
  - loop_rate_below_threshold: `True`
  - scratchpad_bleed_below_threshold: `True`
  - seed_stability: `False`
  - promote_to_next: `False`