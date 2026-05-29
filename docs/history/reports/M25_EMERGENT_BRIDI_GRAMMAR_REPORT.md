# M25 Emergent Bridi Grammar Report

Date: 2026-05-29

## Status

M25 is an active successor branch to M24.2. It is not a chatbot-quality result, because prompt-only still wins on clean synthetic accuracy. It is a substrate result: a compressed, loose integer bridi stream is causally active under shuffled/random/zero stream controls.

## Architecture

M25 replaces fixed bridi frame rows with a variable typed grammar-action stream:

`OPEN / PRED / MOD / ARG / LINK / CLOSE / STOP`

The advisor reads only the integer stream. It does not receive continuous prompt states, frame representations, or hidden trace states. The generator is frozen before advisor training, preserving the M24 trace-only discipline.

## Key Artifact

Run: `m25_medium_grok_probe_20260529`

Report:
`artifacts/runs/telemetry/raw/ablation/hypercube/m25_emergent_bridi/m25_medium_grok_probe_20260529/m25_emergent_bridi_report.json`

Direct unified eval:
`artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m25_medium_direct_20260529/direct_unified_eval_manifest.json`

Whole-grid refresh:
`artifacts/runs/telemetry/raw/ablation/hypercube/whole_ablation_grid/m25_medium_whole_grid_20260529/whole_ablation_grid_manifest.json`

## Metrics

| Metric | Value |
| --- | ---: |
| mean strict accuracy | 0.7403 |
| strict std | 0.0190 |
| prompt-only accuracy | 0.9987 |
| loose stream exact accuracy | 0.4973 |
| predicted vs shuffled delta | 0.6910 |
| predicted vs random delta | 0.6817 |
| token reduction ratio | 0.4123 |
| mean emitted stream symbols | 6.2987 |
| mean prompt tokens | 10.7173 |
| promotion gate pass rate | 1.0000 |
| promotion candidate flag | 1.0000 |

## Interpretation

M25 shows that the looser bridi grammar stream can become a meaningful symbolic substrate. The important evidence is not raw clean accuracy. The important evidence is that predicted streams beat shuffled and random streams by roughly 0.68 to 0.69 while using about 41% fewer symbols than the prompt.

Prompt-only remains near perfect on this synthetic surface, so M25 is not a final assistant architecture. It is a stronger internal-language compression result than the initial M25 smoke runs.

## Stress Attempt

Attempted run:

`m25_sixseed_48e_grok_probe_20260529`

Config: six seeds, train size 12000, eval size 3000, 48 epochs, CUDA, batch size 1024.

Outcome: failed before report write. CUDA returned `cudaErrorUnknown` during the oracle-advisor phase, and `nvidia-smi` subsequently reported: `GPU is lost. Reboot the system to recover this GPU`.

This is recorded as a hardware/driver stress failure, not a model promotion failure. The successful 32-epoch two-seed run remains the current M25 evidence anchor.

## Next Tests

1. Reboot before any further CUDA runs.
2. Rerun the six-seed M25 pass with batch size 512 and periodic per-seed report writes.
3. Add checkpoint/report flushing after each seed so a late CUDA failure does not erase completed seeds.
4. Add a matched-token prompt baseline to test whether the symbolic stream beats a text-only baseline with the same token budget.
