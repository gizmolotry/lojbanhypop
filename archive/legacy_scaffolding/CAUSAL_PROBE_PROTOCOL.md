# Causal Probe Protocol (RoPE/NoPE x Rigidity k)

## Goal
Measure whether increasing symbolic rigidity (`k`) causally improves identity consistency while preserving final-answer performance.

## Fixed Matrix
- Regimes: `rope`, `nope` (`--disable-rope`)
- Rigidity levels: `k in {0, 0.25, 0.5, 1.0, 2.0}`
- Evaluation seeds: `7, 11`
- Sample size: `24`

## Command (Dry Run First)
```powershell
$env:PYTHONPATH="src"
python scripts/run_causal_probe_matrix.py `
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct `
  --dataset runs/lora_sft_dataset.jsonl `
  --local-files-only
```

## Command (Execute Full Matrix)
```powershell
$env:PYTHONPATH="src"
python scripts/run_causal_probe_matrix.py `
  --base-model C:\Users\Andrew\hf_models\Qwen2.5-0.5B-Instruct `
  --dataset runs/lora_sft_dataset.jsonl `
  --local-files-only `
  --execute
```

## Outputs
Under `artifacts/causal_probe/<timestamp>/`:
- `gate_<regime>_k*.json` per matrix cell
- `causal_probe_matrix.json` (combined analysis)
- `causal_probe_matrix.md` (human summary)

## Interpretation
- Primary signal: `slope(symbolic_lift ~ k)` per regime
- Guardrail: `max_final_drop_from_k0`
- Preferred outcome:
  1. Positive symbolic slope in both regimes.
  2. Stronger symbolic slope in `nope` vs `rope`.
  3. Final-answer drop bounded relative to `k=0`.
