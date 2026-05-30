# M25 Emergent Bridi Grammar Report

Date: 2026-05-30

## Status

M25 is an active successor branch to M24.2. It is not promoted under the fair matched-token comparison. It is still a substrate result: a compressed, loose integer bridi stream is causally active under shuffled/random/zero stream controls, but same-budget English remains much stronger on the current synthetic surface.

## Architecture

M25 replaces fixed bridi frame rows with a variable typed grammar-action stream:

`OPEN / PRED / MOD / ARG / LINK / CLOSE / STOP`

The advisor reads only the integer stream. It does not receive continuous prompt states, frame representations, or hidden trace states. The generator is frozen before advisor training, preserving the M24 trace-only discipline.

## Key Artifact

Current large matched-token run:

`artifacts/runs/telemetry/raw/ablation/hypercube/m25_emergent_bridi/m25_large_48e_matched_20260530/m25_emergent_bridi_report.json`

Direct unified eval:

`artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m25_large_48e_matched_direct_20260530/direct_unified_eval_manifest.json`

Whole-grid refresh:

`artifacts/runs/telemetry/raw/ablation/hypercube/whole_ablation_grid/m25_large_48e_matched_whole_grid_20260530/whole_ablation_grid_manifest.json`

Previous medium run:

Run: `m25_medium_grok_probe_20260529`

Report:
`artifacts/runs/telemetry/raw/ablation/hypercube/m25_emergent_bridi/m25_medium_grok_probe_20260529/m25_emergent_bridi_report.json`

Direct unified eval:
`artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m25_medium_direct_20260529/direct_unified_eval_manifest.json`

Whole-grid refresh:
`artifacts/runs/telemetry/raw/ablation/hypercube/whole_ablation_grid/m25_medium_whole_grid_20260529/whole_ablation_grid_manifest.json`

## Metrics

Large six-seed matched-token run:

| Metric | Value |
| --- | ---: |
| mean strict accuracy | 0.7104 |
| strict std | 0.0235 |
| full prompt-only accuracy | 0.9999 |
| matched-token prompt accuracy | 0.9974 |
| M25 delta vs matched prompt | -0.2870 |
| loose stream exact accuracy | 0.5105 |
| predicted vs shuffled delta | 0.6533 |
| predicted vs random delta | 0.6555 |
| token reduction ratio | 0.4062 |
| mean emitted stream symbols | 6.3517 |
| mean matched prompt tokens | 7.9157 |
| matched prompt gate | 0.0000 |
| promotion gate pass rate | 0.8571 |
| promotion candidate flag | 0.0000 |

Previous medium run, before matched-token promotion gating:

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

The large matched-token rerun sharpens the interpretation. M25 is causally real as a symbolic stream, but it does not beat same-budget English on this benchmark. The correct status is therefore `active_non_promoted`: useful evidence for internal-language compression, not a winning prompt-compression architecture yet.

## Matched-Token Prompt Control

The original prompt-only control is an intentionally unfair upper bound: it reads the whole English prompt. M25 now also trains and reports a matched-token prompt control. That classifier receives only the first `N` prompt tokens, where `N` defaults to the same hard symbol budget used by the loose bridi stream.

New ledger metrics include:

| Metric | Meaning |
| --- | --- |
| `matched_prompt_accuracy` | strict accuracy of the same-budget prompt-only classifier |
| `m25_strict_delta_vs_matched_prompt` | bridi stream strict accuracy minus matched prompt strict accuracy |
| `matched_prompt_token_budget` | text budget used by the matched prompt control |
| `matched_prompt_accuracy_per_token` | matched prompt token-efficiency diagnostic |
| `m25_accuracy_per_symbol_delta_vs_matched_prompt` | stream efficiency minus matched prompt efficiency |

CPU plumbing smoke:

`artifacts/runs/telemetry/raw/ablation/hypercube/m25_emergent_bridi/m25_matched_prompt_cpu_smoke_20260529/m25_emergent_bridi_report.json`

The CPU smoke confirmed the reporting path, not the scientific claim. The six-seed CUDA run above is the current matched-token evidence anchor.

The promotion gate now requires `m25_gate_beats_matched_prompt=1.0`. The large run has this gate at `0.0`, so `m25_promotion_candidate=0.0` even though the older stream-corruption gates pass.

## Stress Attempt

Attempted run:

`m25_sixseed_48e_grok_probe_20260529`

Config: six seeds, train size 12000, eval size 3000, 48 epochs, CUDA, batch size 1024.

Outcome: failed before report write. CUDA returned `cudaErrorUnknown` during the oracle-advisor phase, and `nvidia-smi` subsequently reported: `GPU is lost. Reboot the system to recover this GPU`.

This is recorded as a hardware/driver stress failure. The May 30 rerun used batch size 512 and per-seed jobs to avoid losing completed evidence.

## Next Tests

1. Make the matched-token baseline harder by preventing lexical prefix leakage, not by weakening the baseline unfairly.
2. Test semantic-code controls where the text baseline receives the same token budget but not the same surface lexical order.
3. Add an explicit report assembler or per-seed flush mode to the M25 runner so large runs do not require manual aggregation.
4. Decide whether M26 should target a harder benchmark where symbolic compression has room to matter, instead of clean synthetic templates that prompt-only solves almost perfectly.
