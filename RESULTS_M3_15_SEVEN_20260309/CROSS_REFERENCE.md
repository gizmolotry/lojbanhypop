# CROSS_REFERENCE

## DAGs
- `lojban_m3_15_rotary_coconut` -> `dag/lojban_m3_15_rotary_coconut_dag.py`
- `lojban_m3_15_rotary_coconut_seven` -> `dag/lojban_m3_15_rotary_coconut_seven_dag.py`

## Run-to-Artifact Map
### R1
- Summary JSON: `runs/R1/m3_15_report.json`
- Summary MD: `runs/R1/m3_15_report.md`
- Eval rows A/B/C: `runs/R1/m3_15_A_eval.json`, `runs/R1/m3_15_B_eval.json`, `runs/R1/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R1/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R1/m3_15_eval_pack_preview.json`

### R2
- Summary JSON: `runs/R2/m3_15_report.json`
- Summary MD: `runs/R2/m3_15_report.md`
- Eval rows A/B/C: `runs/R2/m3_15_A_eval.json`, `runs/R2/m3_15_B_eval.json`, `runs/R2/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R2/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R2/m3_15_eval_pack_preview.json`

### R3
- Summary JSON: `runs/R3/m3_15_report.json`
- Summary MD: `runs/R3/m3_15_report.md`
- Eval rows A/B/C: `runs/R3/m3_15_A_eval.json`, `runs/R3/m3_15_B_eval.json`, `runs/R3/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R3/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R3/m3_15_eval_pack_preview.json`

### R4
- Summary JSON: `runs/R4/m3_15_report.json`
- Summary MD: `runs/R4/m3_15_report.md`
- Eval rows A/B/C: `runs/R4/m3_15_A_eval.json`, `runs/R4/m3_15_B_eval.json`, `runs/R4/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R4/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R4/m3_15_eval_pack_preview.json`

### R5
- Summary JSON: `runs/R5/m3_15_report.json`
- Summary MD: `runs/R5/m3_15_report.md`
- Eval rows A/B/C: `runs/R5/m3_15_A_eval.json`, `runs/R5/m3_15_B_eval.json`, `runs/R5/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R5/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R5/m3_15_eval_pack_preview.json`

### R6
- Summary JSON: `runs/R6/m3_15_report.json`
- Summary MD: `runs/R6/m3_15_report.md`
- Eval rows A/B/C: `runs/R6/m3_15_A_eval.json`, `runs/R6/m3_15_B_eval.json`, `runs/R6/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R6/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R6/m3_15_eval_pack_preview.json`

### R7
- Summary JSON: `runs/R7/m3_15_report.json`
- Summary MD: `runs/R7/m3_15_report.md`
- Eval rows A/B/C: `runs/R7/m3_15_A_eval.json`, `runs/R7/m3_15_B_eval.json`, `runs/R7/m3_15_C_eval.json`
- Seed2 stability eval: `runs/R7/m3_15_B_seed2_eval.json`
- Eval pack preview: `runs/R7/m3_15_eval_pack_preview.json`

## Config Deltas (relative to R1)
| Run | Delta |
|---|---|
| R1 | baseline defaults |
| R2 | align_weight=0.0 |
| R3 | margin_weight=0.0 |
| R4 | margin=0.4, margin_weight=1.0 |
| R5 | runtime_gate_cap=0.0 |
| R6 | align_weight=1.5, max_nodes=16 |
| R7 | runtime_enable_min_acc_gain=0.0 |