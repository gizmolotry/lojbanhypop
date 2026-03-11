# M3.15b Bridge Base V2

## Files
- `bridge_base_v2/`: full report bundle for the guarded anti-collapse run.
- `baselines/m_series_bridge_baseline_manifest.json`: promoted bridge comparison base.

## Headline
- The de-collapsed checkpoint is now the explicit bridge baseline.
- `B` and `C` no longer collapse the operator inventory.
- Accuracy did not improve beyond the de-collapsed control.

## Metrics
### Control A
- overall_accuracy: `0.5`
- mean_active_op_count: `3.0`
- mean_operator_entropy: `0.93130153729747`
- mean_top1_op_share: `0.4554036458333333`

### Guarded B
- overall_accuracy: `0.5`
- mean_active_op_count: `3.0`
- mean_operator_entropy: `0.93130153729747`
- mean_top1_op_share: `0.4554036458333333`
- train anti_collapse_loss: `6.307556607983618e-07`

### Guarded C
- overall_accuracy: `0.5`
- mean_active_op_count: `3.0`
- mean_operator_entropy: `0.93130153729747`
- mean_top1_op_share: `0.4554036458333333`
- train anti_collapse_loss: `1.9453839922789484e-05`

## Interpretation
The bridge fix solved the mechanical collapse problem. It did not solve task performance. That narrows the remaining problem substantially: bridge training is no longer destroying the predicate inventory, so the next bottleneck is whether the bridge signal is actually informative enough to move the answer decision.
