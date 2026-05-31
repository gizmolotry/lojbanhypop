# M26 End-To-End Lojban Symbiote Spinal-Cord Report

Date: 2026-05-31

## Claim Boundary

M26 is not the final Lojbanic chatbot.

M26 tests the narrower missing claim: whether the Lojban symbiote path exists as one trainable PyTorch organism where final answer loss backpropagates through the bridi scratchpad generator instead of stopping at a frozen, hard-packed symbolic stream.

## What Changed

- Added `M26EndToEndLoafman`: prompt encoder plus loose bridi stream emitter plus differentiable trace-only advisor under one optimizer.
- Replaced M25's training-time hard `argmax`/integer stream cut with a soft type/value/aux distribution handoff.
- Added gradient telemetry proving answer loss reaches the generator and symbol heads.
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

## Interpretation

The smoke result established gradient topology. The scaled result adds causal utility: the learned soft bridi trace strongly beats shuffled, random, and zero-trace controls while preserving end-to-end gradient flow.

The exact stream reconstruction is only partial, which is scientifically useful: M26 does not need perfect symbolic imitation to solve the synthetic task, but the corruption controls show that the trace is carrying answer-causal information.

## Current Status

`M26` is promoted for the narrow spinal-cord claim: the bridi scratchpad and advisor now exist as one end-to-end trainable organism.

It is not promoted as a final chatbot. The next architectural step should connect this end-to-end differentiable bridi path to an actual base-LLM bridge/decoder path, with prompt-only and matched-token controls kept intact.
