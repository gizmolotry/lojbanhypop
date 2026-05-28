# M24.2 Hard Trace Bottleneck Report

**Date:** 2026-05-28  
**Branch:** `codex/m24-2-hard-trace-bottleneck-20260525`  
**Status:** `stable_compressed_substrate_anchor`  
**Canonical anchor:** `m24_2_b11_g32_a48_train12000_eval6000_tenseed_20260528`

## Executive Finding

M24.2 is the first M-series result that cleanly satisfies the hard trace bottleneck claim across a ten-seed validation surface.

The best current claim is not that the compressed trace beats the full prompt. It does not. The defensible claim is narrower and stronger:

> A frozen dynamic bridi trace, passed through a hard integer-only packed-symbol bottleneck, carries causal answer information under shuffled/random/zero-trace controls while using fewer symbols than the source prompt.

The current stability anchor is `b11`: active frame budget `2`, trace symbol budget `11`, generator epochs `32`, advisor epochs `48`, train size `12000`, eval size `6000`, seeds `23,29,31,37,41,43,47,53,59,61`.

## Anchor Metrics

| Metric | Value |
| --- | ---: |
| strict accuracy | `0.7682` |
| strict std | `0.0162` |
| prompt-only accuracy | `0.9960` |
| delta vs prompt-only | `-0.2278` |
| bridi trace exact accuracy | `0.3951` |
| predicted vs shuffled delta | `0.7124` |
| predicted vs random delta | `0.7130` |
| packed/prompt ratio | `0.9446` |
| token reduction | `5.54%` |
| hard bottleneck token count | `10.1296` |
| M24.2 candidate rate | `1.0000` |
| M24.2 gate pass rate | `1.0000` |
| substrate claim score | `0.4520` |

## Frontier Bracket

| Run | Seeds | Budget | Strict | Trace Exact | Pred vs Random | Packed/Prompt | Token Reduction | Candidate | Verdict |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `m24_2_b12_g32_a48_train12000_eval6000_20260527` | 6 | `12` | `0.7597` | `0.3916` | `0.7049` | `0.9869` | `1.31%` | `1.0000` | conservative pass, weak compression |
| `m24_2_b11_g32_a48_train12000_eval6000_20260527` | 6 | `11` | `0.7700` | `0.3931` | `0.7156` | `0.9427` | `5.73%` | `1.0000` | best six-seed frontier |
| `m24_2_b11_g32_a48_train12000_eval6000_tenseed_20260528` | 10 | `11` | `0.7682` | `0.3951` | `0.7130` | `0.9446` | `5.54%` | `1.0000` | stability anchor |
| `m24_2_b10_g32_a48_train12000_eval6000_20260527` | 6 | `10` | `0.7343` | `0.3984` | `0.6811` | `0.8755` | `12.45%` | `1.0000` | compression tradeoff |
| `m24_2_b9_g32_a48_train12000_eval6000_20260528` | 6 | `9` | `0.7184` | `0.4067` | `0.6621` | `0.8029` | `19.71%` | `1.0000` | aggressive tradeoff |

## What Changed Scientifically

### 1. Advisor capacity was the missing piece

Earlier `b12 g32/a32` missed a six-seed clean pass by one seed. Extending the advisor from `32` to `48` epochs fixed the miss without changing the trace generator. That indicates the hard trace was already carrying useful structure; the readout was undertrained.

### 2. More training data moved the knee

At `train_size=6000`, the useful frontier was around `b12`, with only about `1.5%` token reduction. At `train_size=12000`, the knee moved to `b11`, giving about `5.5%` token reduction while preserving or improving strict accuracy and causal deltas.

### 3. The hard trace remains causal under controls

The anchor's predicted trace beats shuffled and random traces by more than `0.71` accuracy. This matters more than trace exact alone: the compressed bridi trace is not merely decorative, because destroying trace alignment collapses performance.

### 4. Compression has a visible accuracy frontier

`b10` and `b9` still pass M24.2 gates, but they trade away strict accuracy. This is useful evidence: the system is not faking compression with an unconstrained shortcut. Reducing symbolic budget produces the expected rate-distortion curve.

## Defensible Claim

M24.2 supports this claim:

> A learned Lojban-inspired bridi trace can serve as a compressed, causally active symbolic substrate when the advisor is forced to consume only packed integer trace symbols under hard active-frame and symbol-budget constraints.

It does not yet support this stronger claim:

> The compressed bridi trace is competitive with or superior to full prompt access.

Prompt-only remains much higher (`0.9960` on the ten-seed anchor). The current contribution is compression-under-causal-control, not full-prompt replacement.

## Limits And Risks

- The best stability anchor reduces tokens by only `5.54%`; the compression is real but modest.
- Prompt-only remains the ceiling, so the advisor still loses substantial information relative to full English context.
- M24.2 gates pass for `b9` and `b10`, but strict accuracy degrades, so gate pass alone is not enough to choose a promotion anchor.
- The report depends on synthetic M24/M23-style bridi tasks; transfer to broader natural language remains unproven.
- Whole-grid M24 anchoring is mtime-sensitive through direct unified eval; the report should always cite explicit artifacts.

## Artifact Links

- Compression report: `artifacts/runs/telemetry/raw/ablation/hypercube/m24_substrate_compression/m24_2_b11_g32_a48_train12000_eval6000_tenseed_20260528/m24_substrate_compression_report.json`
- Direct unified eval: `artifacts/runs/telemetry/raw/ablation/hypercube/direct_unified_eval/m24_2_b11_g32_a48_train12000_eval6000_tenseed_direct_20260528/direct_unified_eval_manifest.json`
- Whole-grid anchor: `docs/history/reports/WHOLE_ABLATION_GRID_LATEST.md`

## Recommended Next Tests

1. Run a `b11` transfer slice outside the synthetic M24 substrate generator to test whether the trace survives less controlled prompts.
2. Run a matched prompt-token control where the baseline receives a prompt truncated to the same token budget as the packed bridi trace.
3. Add a report-level rate-distortion chart over `b9`, `b10`, `b11`, and `b12` so the frontier is visible without reading raw artifacts.
4. Keep `b11` as the M24.2 stability anchor unless a new run improves both strict accuracy and token reduction under ten-seed validation.
