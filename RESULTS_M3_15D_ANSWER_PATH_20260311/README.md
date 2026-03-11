# M3.15d Answer-Path Forcing

This bundle contains the full output of the `M3.15d` ablation run:

- `A`: control from the de-collapsed bridge base
- `B`: direct-drive differentiable answer-path margin
- `C`: late-stage causal blindfold
- `D`: candidate pointer head

Primary report:

- `m3_15d_report.json`
- `m3_15d_report.md`

Key outcome:

- all cells stayed at `0.50` eval accuracy
- `B` and `D` preserved operator diversity and maintained positive internal answer deltas
- `C` pushed the answer delta negative under blindfolding
- no cell produced a positive `gold on-off` intervention delta

Interpretation:

The answer path can be forced locally without re-collapsing the operator inventory, but that forcing still does not change the final decision boundary in the current stack.
