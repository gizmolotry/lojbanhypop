# M26 End-To-End Lojban Symbiote Spinal-Cord Report

Date: 2026-05-31

## Claim Boundary

M26 is not the final Lojbanic chatbot.

M26 tests the narrower missing claim: whether the Lojban symbiote path exists as one trainable PyTorch organism where final answer loss backpropagates through the bridi scratchpad generator instead of stopping at a frozen, hard-packed symbolic stream.

## What Changed

- Added `M26EndToEndLoafman`: prompt encoder plus loose bridi stream emitter plus differentiable trace-only advisor under one optimizer.
- Replaced M25's training-time hard `argmax`/integer stream cut with a soft type/value/aux distribution handoff.
- Added gradient telemetry proving answer loss reaches the generator and symbol heads.
- Added prompt-only and matched-token controls to the M26 suite so prompt comparability is measured on the same splits.
- Added M26 family registry, runner, Airflow DAG, taxonomy contract, direct unified eval support, and whole-grid visibility.

## Smoke Result

Run: `m26_spinal_cord_smoke_20260530`

- strict accuracy: `0.0625`
- answer loss reaches generator: `1.0`
- answer loss reaches symbol heads: `1.0`
- hard argmax training cut detected: `0.0`
- torch no-grad training cut detected: `0.0`
- spinal-cord gate pass rate: `0.8`
- promotion candidate: `0.0`

## Scaled Result

Run: `m26_e2e_large_64e_20260531`

Configuration:

- seeds: `23,29,31,37,41,43`
- train size: `24000`
- eval size: `6000`
- epochs: `64`
- batch size: `512`
- device: `cuda`
- max symbols: `32`
- symbol budget: `16`

Aggregate metrics:

- mean strict accuracy: `0.9984722236792246`
- std strict accuracy: `0.0009048023566035344`
- shuffled trace accuracy: `0.05763888917863369`
- random trace accuracy: `0.05699999940892061`
- zero trace accuracy: `0.055916666984558105`
- predicted vs shuffled delta: `0.9408333345005909`
- predicted vs random delta: `0.9414722242703041`
- predicted vs zero delta: `0.9425555566946665`
- loose stream exact accuracy: `0.5289166818062464`
- mean emitted symbols after bottleneck: `8.428666591644287`
- answer loss reaches generator: `1.0`
- answer loss reaches symbol heads: `1.0`
- hard argmax training cut detected: `0.0`
- torch no-grad training cut detected: `0.0`
- spinal-cord gate pass rate: `1.0`
- promotion candidate: `1.0`

## Compression And Necessity Diagnostics

Runs: `m26_budget4_32e_20260531`, `m26_budget8_32e_20260531`, `m26_budget16_32e_20260531`, `m26_answer0_32e_20260531`

Common configuration:

- seeds: `23,29,31`
- train size: `12000`
- eval size: `3000`
- epochs: `32`
- batch size: `512`
- device: `cuda`

Budget grid:

| run | symbol budget | strict accuracy | predicted vs zero | spinal-cord candidate |
|---|---:|---:|---:|---:|
| `m26_budget4_32e_20260531` | `4` | `0.9956666827201843` | `0.9410000157852968` | `1.0` |
| `m26_budget8_32e_20260531` | `8` | `0.9961111148198446` | `0.9400000038246313` | `1.0` |
| `m26_budget16_32e_20260531` | `16` | `0.994777778784434` | `0.9372222237288952` | `1.0` |

Answer-loss ablation:

- run: `m26_answer0_32e_20260531`
- answer weight: `0.0`
- strict accuracy: `0.0409999992698431`
- predicted vs zero delta: `-0.013666667665044466`
- loose stream exact accuracy: `0.5308888951937357`
- spinal-cord candidate: `0.0`

Interpretation: trace reconstruction alone is not enough. The end-to-end answer loss is the causal pressure that makes the bridi trace useful to the advisor.

## Prompt Comparability Diagnostic

Run: `m26_matched_prompt_b8_32e_20260531_r2`

Configuration:

- seeds: `23,29,31`
- train size: `12000`
- eval size: `3000`
- epochs: `32`
- prompt epochs: `32`
- symbol budget: `8`
- matched prompt budget: `8`

Metrics:

- strict accuracy: `0.9961111148198446`
- prompt-only accuracy: `1.0`
- matched-token prompt accuracy: `0.9973333477973938`
- delta vs prompt-only: `-0.003888885180155436`
- delta vs matched prompt: `-0.001222232977549235`
- matched prompt accuracy per token: `0.12593305800367893`
- M26 accuracy per loose symbol: `0.11620068531437672`
- spinal-cord candidate: `1.0`
- prompt-comparable candidate: `0.3333333333333333`

Interpretation: M26 is now strongly validated as an end-to-end bridi symbiote, but it does not yet beat the same-budget matched-token prompt control. The gap is tiny in accuracy and larger in accuracy-per-token.

## Interpretation

The smoke result established gradient topology. The scaled result adds causal utility: the learned soft bridi trace strongly beats shuffled, random, and zero-trace controls while preserving end-to-end gradient flow.

The exact stream reconstruction is only partial, which is scientifically useful: M26 does not need perfect symbolic imitation to solve the synthetic task, but the corruption controls show that the trace is carrying answer-causal information.

## Current Status

`M26` is promoted for the narrow spinal-cord claim: the bridi scratchpad and advisor now exist as one end-to-end trainable organism.

`M26` is not promoted for prompt comparability. Matched-token English still edges it on the same M26 diagnostic split.

It is not promoted as a final chatbot. The next architectural step should connect this end-to-end differentiable bridi path to an actual base-LLM bridge/decoder path, with prompt-only and matched-token controls kept intact.
