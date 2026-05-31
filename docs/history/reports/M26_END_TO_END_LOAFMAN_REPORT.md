# M26 End-To-End Lojban Symbiote Full-Organism Report

Date: 2026-05-31

## Claim Boundary

M26 is not the final Lojbanic chatbot.

M26 now tests the full hidden-state bridge organism claim: a tiny language backbone emits prompt hidden states, the bridi generator reads that language stream, a trace-language cross-attention bridge reads prompt hidden states and bridi trace state, and a choked answer head reads the fused language/trace state with raw prompt bypass blocked.

The original spinal-cord result remains valid as a narrower subclaim: final answer loss must still backpropagate through the bridi scratchpad generator and symbol heads instead of stopping at a frozen, hard-packed symbolic stream.

## What Changed

- Added `M26EndToEndLoafman`: tiny language backbone plus loose bridi stream emitter plus trace-language bridge plus fused-state answer head under one optimizer.
- Replaced M25's training-time hard `argmax`/integer stream cut with a soft type/value/aux distribution handoff.
- Added gradient telemetry proving answer loss reaches the language backbone, bridge, generator, and symbol heads.
- Added full-organism topology gates: `lm_hidden_state_stream_active`, `bridi_generator_reads_lm_hidden_states`, `trace_bridge_reads_prompt_hidden_states`, `answer_head_reads_fused_lm_trace_state`, `raw_prompt_bypass_blocked`, `m26_full_organism_gate_pass_rate`, and `m26_full_organism_candidate`.
- Added prompt-only and matched-token controls to the M26 suite so prompt comparability is measured on the same splits.
- Added M26 family registry, runner, Airflow DAG, taxonomy contract, direct unified eval support, and whole-grid visibility.

## Historical Spinal-Cord Smoke Result

Run: `m26_spinal_cord_smoke_20260530`

- strict accuracy: `0.0625`
- answer loss reaches generator: `1.0`
- answer loss reaches symbol heads: `1.0`
- hard argmax training cut detected: `0.0`
- torch no-grad training cut detected: `0.0`
- spinal-cord gate pass rate: `0.8`
- promotion candidate: `0.0`

## Current Full-Organism Smoke Verification

Run: `pytest_m26_full_organism_smoke2`

- strict accuracy: `0.0833333358168602`
- answer loss reaches generator: `1.0`
- answer loss reaches symbol heads: `1.0`
- answer loss reaches language backbone: `1.0`
- answer loss reaches bridge: `1.0`
- LM hidden-state stream active: `1.0`
- bridi generator reads LM hidden states: `1.0`
- trace bridge reads prompt hidden states: `1.0`
- answer head reads fused LM/trace state: `1.0`
- raw prompt bypass blocked: `1.0`
- full-organism gate pass rate: `1.0`
- full-organism candidate: `0.0`

Interpretation: the assembled organism now exists and the answer loss reaches all major organs. This smoke is intentionally tiny, so it verifies topology and reporting, not scaled accuracy.

## Historical Scaled Spinal-Cord Result

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

Important: this scaled artifact predates the full hidden-state bridge gates. It remains valid evidence for the narrower spinal-cord gradient-through-trace subclaim, but it must be rerun before being cited as full-organism evidence.

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

Interpretation: trace reconstruction alone is not enough. The end-to-end answer loss is the causal pressure that makes the bridi trace useful to the fused language/trace answer path. Under the full-organism interpretation, this spinal-cord evidence must be paired with language-backbone, bridge, fused-head, and raw-bypass gates.

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

Interpretation: the historical spinal-cord path was strongly validated as an end-to-end bridi symbiote, but it did not beat the same-budget matched-token prompt control. The gap was tiny in accuracy and larger in accuracy-per-token. This control must be rerun under the full-organism bridge before promotion.

## Interpretation

The current full-organism smoke establishes assembled-organism gradient topology. The historical scaled result adds causal utility for the earlier spinal-cord path: the learned soft bridi trace strongly beat shuffled, random, and zero-trace controls while preserving end-to-end gradient flow.

The exact stream reconstruction is only partial, which is scientifically useful: M26 does not need perfect symbolic imitation to solve the synthetic task, but the corruption controls show that the trace is carrying answer-causal information.

## Current Status

`M26` is promoted for the historical narrow spinal-cord subclaim: the bridi scratchpad path and answer head now exist as one end-to-end trainable gradient path.

`M26` full-organism promotion is governed by `m26_full_organism_candidate` and the hidden-state bridge gates, including language-backbone gradient flow, bridge gradient flow, bridi-generator reads of LM hidden states, trace-bridge reads of prompt hidden states, fused-state answer-head reads, and raw prompt bypass blocking.

`M26` is not promoted for prompt comparability. Matched-token English edged the historical spinal-cord diagnostic split, and the full-organism bridge still needs a scaled matched-token rerun.

It is not promoted as a final chatbot. The next architectural step should keep the full hidden-state bridge organism gates intact while scaling from the tiny language backbone toward an actual base-LLM bridge/decoder path, with prompt-only and matched-token controls kept intact.
