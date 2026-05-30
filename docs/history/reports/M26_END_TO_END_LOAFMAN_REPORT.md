# M26 End-To-End Loafman Spinal-Cord Report

Date: 2026-05-30

## Claim Boundary

M26 is not the final Lojbanic chatbot.

M26 tests the narrower missing claim: whether the Loafman path exists as one trainable PyTorch organism where final answer loss backpropagates through the bridi scratchpad generator instead of stopping at a frozen, hard-packed symbolic stream.

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

## Interpretation

The important positive result is gradient topology, not task accuracy. The final answer objective now reaches the symbolic emitter through a differentiable bridi advisor path.

The important negative result is that the toy smoke run does not yet prove causal trace utility: predicted trace accuracy did not beat zero/shuffled/random controls at this tiny training scale.

## Current Status

`M26` is a valid scaffold for the first true Loafman organism, but it is not promoted.

Next work should train M26 at comparable M25 scale and then add the later base-LLM bridge coupling as a separate cell, not quietly confuse this spinal-cord test with chatbot success.
