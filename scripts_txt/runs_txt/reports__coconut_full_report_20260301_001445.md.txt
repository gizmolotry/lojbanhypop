# Coconut Comprehensive Report

- generated_at: `2026-03-01T23:47:34.174943+00:00`

## Core Results
- Run A base final: `0.167`
- Run B rigid final: `0.396`
- Run C kv handoff final: `0.104`
- Run E babel final: `0.167`
- Run F self-correct final: `0.312`
- Run G true-coconut final: `0.104`

## H-Series
- H1: handoff=`0.000`, lift=`-0.167`, mean_step_cos=`0.456`
- H2: handoff=`0.042`, lift=`-0.125`, mean_step_cos=`0.923`
- H3: handoff=`0.000`, lift=`-0.167`, mean_step_cos=`0.934`
- H4: handoff=`0.083`, lift=`-0.083`, mean_step_cos=`0.925`

## H3 Sweep
| rank | exp | scale | bridge | handoff | lift | step_cos |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 4 | 0.25 | `runs\projections\swiglu_midlayer_bridge_h3_exp4.pt` | 0.083 | -0.083 | 0.924 |
| 2 | 4 | 0.5 | `runs\projections\swiglu_midlayer_bridge_h3_exp4.pt` | 0.083 | -0.083 | 0.925 |
| 3 | 4 | 1.0 | `runs\projections\swiglu_midlayer_bridge_h3_exp4.pt` | 0.083 | -0.083 | 0.925 |
| 4 | 4 | 1.5 | `runs\projections\swiglu_midlayer_bridge_h3_exp4.pt` | 0.083 | -0.083 | 0.925 |
| 5 | 2 | 0.25 | `runs\projections\swiglu_midlayer_bridge_h3.pt` | 0.000 | -0.167 | 0.932 |
| 6 | 2 | 0.5 | `runs\projections\swiglu_midlayer_bridge_h3.pt` | 0.000 | -0.167 | 0.934 |
| 7 | 2 | 1.0 | `runs\projections\swiglu_midlayer_bridge_h3.pt` | 0.000 | -0.167 | 0.934 |
| 8 | 2 | 1.5 | `runs\projections\swiglu_midlayer_bridge_h3.pt` | 0.000 | -0.167 | 0.934 |

## English Control Duel
- base_acc: `0.250`
- english_cot_acc: `0.000`
- lojban_adapter_acc: `0.417`
- english_minus_lojban: `-0.417`

## VQ Pilot
- codebook_size: `200`
- codes_used: `9` (ratio `0.045`)
- train_steps: `500`
- loss start/end: `135.016` -> `108.622`

## SwiGLU Train
| exp | train_examples | train_steps | loss start -> end | bridge |
|---:|---:|---:|---|---|
| 2 | 19 | 500 | `11.135 -> 1.695` | `runs\projections\swiglu_midlayer_bridge_h3.pt` |
| 4 | 19 | 500 | `16.298 -> 1.564` | `runs\projections\swiglu_midlayer_bridge_h3_exp4.pt` |

## Verdict
- Best final accuracy remains Run B (`0.396`) and Run F (`0.312`) among recovery paths.
- Mid-layer transport preserves geometry (high step-cos), but semantic alignment remains unresolved in final-answer decoding.
