# M3.16 Continuous Graph Bias

Artifacts in this folder are copied from the completed small-profile run:
- Source run dir: `artifacts/runs/telemetry/raw/ablation/hypercube/m3_16_continuous_graph_bias/bridge_base_v1_small/m3_16_20260311_bridge_base_v1_small`
- Baseline id: `M_BRIDGE_BASE_20260311_DEC`

## Headline
- `A/B/C/D` all finished.
- Final accuracy stayed flat at `0.500` for every cell.
- The graph-bias translator stayed local and structurally clean.
- Candidate-only (`B`) produced the largest positive intervention delta, but it was still too small to clear promotion.

## Cells
- `A`: control frozen bridge base
- `B`: candidate-only soft graph bias
- `C`: candidate+cue soft graph bias
- `D`: global prompt soft graph bias

## Main read
This ablation supports the current bottleneck diagnosis:
- preserving operator diversity is no longer the problem
- injecting a soft structural bias into System 2 attention is mechanically feasible
- but the injected bias is still not causally strong enough to move final answer accuracy on this Winograd slice

## Files
- `m3_16_report.json`
- `m3_16_report.md`
- `m3_16_A_eval.json`
- `m3_16_B_eval.json`
- `m3_16_C_eval.json`
- `m3_16_D_eval.json`
