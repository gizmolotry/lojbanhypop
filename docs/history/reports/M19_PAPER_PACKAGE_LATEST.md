# M19 Paper Package

- narrative verdict: `promising_but_unstable`

## Narrative

M19 currently supports a narrower and more defensible claim than earlier optimistic summaries: a compact symbiote runway can retain substantial lift with strong token efficiency and survive initial anti-shortcut checks.
The current evidence does not justify a claim of general reasoning closure, nor does it erase the negative lessons from earlier bridge families.
The May 8 weak-seed surface-contract run is treated as a negative robustness result: stronger surface consistency did not recover seed stability and should not be promoted.
The bridge-channel isolation suite is now the active diagnostic path: it tests whether answer lift survives removing typed predicate or pointer channels.
The training response is an opt-in pointer-necessity contrast loss: full bridge answer loss must beat a `no_judri` ablated bridge by a margin, forcing answer lift to depend on judri pointer channels instead of predicate-only shortcuts.
Token efficiency is now split into generated answer tokens and runway-adjusted tokens so static scratchpad carriers are counted against the compression thesis rather than hidden outside the denominator.
The order-sensitivity suite is now part of the required M19 diagnostic surface: first-N benchmark scores must be checked against reversed, shuffled, and stratified slices before they are treated as stable claims.
Strict accuracy remains canonical; phrase accuracy and token efficiency are diagnostic context only.

## Ablation Table

# M19 Ablation Table

| Regime | Accuracy | Avg Tokens | Acc/Token | CoT Token Ratio | Retained CoT Acc/Token | Notes |
|---|---:|---:|---:|---:|---:|---|
| M19.3 mainline | 0.6000 | 30.8400 | 0.0195 |  |  | current direct unified eval headline |
| Purged | 0.6559 |  |  |  |  | overlap-purged benchmark slice |
| Replication mean | 0.3700 | 24.4415 | 0.0151 |  |  | multi-seed mean |
| Entity kill | 0.4084 |  |  |  |  | entity anonymization on purged slice |
| Format kill | 0.3569 |  |  |  |  | format flattening on purged slice |
| Masked blindfold | 0.0000 |  |  |  |  | lexical blindfold carryover from integrity suite |
| May 8 surface contract | 0.4350 | 31.3200 | 0.0139 |  |  | negative robustness result; not promoted |
| Bridge channel full | 0.7500 | 30.5500 | 0.0245 |  |  | eval-only channel isolation; no_judri=0.7000 gismu_only=0.3000 |

Latest smoke-only order-sensitivity diagnostic on the completed weak-seed pointer baseline (`ptr=0.0`, seed 29) produced first=`0.5000`, reversed=`0.7500`, shuffled=`0.5000`, spread=`0.2500` at `eval_size=4`. This is not a result claim, but it validates that the new suite catches first-N brittleness.

## Methodology

# Methodology

M19 is evaluated as a bounded continuous scratchpad runway rather than a claim of general reasoning closure.
The mainline hypothesis is that compact symbiote runway states can retain useful reasoning lift at far lower token cost than explicit natural-language CoT.
All active claims are expected to route through the ledger-backed surfaces: benchmark, audit, integrity, replication, and kill tests.

Active track: `M19.31`

## Limitations

# Limitations

- M19 does not prove general reasoning solved; it shows a compact runway can retain lift and survive initial anti-shortcut checks.
- Legacy H/J/L families are largely comparison anchors and obligations, not fully rerunnable peers on the exact same benchmark surface.
- Replication count is `5`; more seeds would still strengthen stability claims.
- Entity/format kill tests are stronger than lexical masking alone, but they are still perturbation checks rather than theorem-proving guarantees.
- Current masked blindfold accuracy is `0.0000`.
- Current bridge-channel isolation remains aggregate family masking, not per-example pointer counterfactual proof; pointer binding is still an open causal question.
- Any token-efficiency claim must report runway-adjusted tokens alongside generated tokens.

## Appendix

# Appendix

## Direct Unified Eval Contracts

- `m19.runway_efficiency`: available (artifact)
- `m19.zh_branch_control`: available (artifact)
- `m19.dynamic_pacing_guardrails`: not_applicable (artifact)
- `m19.integrity_controls`: available (artifact)
- `m19.replication_stability`: available (artifact)
- `m19.kill_test_suite`: available (artifact)
- `m19.typed_faithfulness`: available (artifact)
- `m19.hyperbolic_geometry`: available (artifact)
- `m18.kill_random_gap`: reference_only (reference)
- `m18.language_tax_compactness`: reference_only (reference)
- `m14.scratchpad_bleed`: reference_only (reference)
- `m11.native_discriminative_oracle`: reference_only (reference)
- `j.accepted_foil_pair_accuracy`: reference_only (reference)
- `l.constraint_scope`: reference_only (reference)
- `m5.chain_serialization`: reference_only (reference)
- `m4.operator_family_consistency`: reference_only (reference)
- `m3.reentry_intervention`: reference_only (reference)
- `m3.reentry_fluency`: reference_only (reference)
- `j.invariance_rate`: reference_only (reference)
- `l.constraint_identity`: reference_only (reference)
- `l.constraint_arity_strict`: reference_only (reference)
- `j.accept_rate_by_depth`: reference_only (reference)
- `j.schema_validity`: reference_only (reference)

## Replication Headlines

- mean accuracy: `0.3700`
- std accuracy: `0.1050`

## Kill-Test Headlines

- entity accuracy: `0.4084`
- format accuracy: `0.3569`
- numeric accuracy: `0.5080`

## Typed Faithfulness Headlines

- typed family accuracy: `0.2500`
- arity violation rate: `0.0000`
- masked pointer zero rate: `1.0000`
- symbolic trace alignment: `0.0000`

## Bridge Channel Isolation

- full accuracy: `0.7500`
- scratchpad-only accuracy: `0.0500`
- random-shape accuracy: `0.0500`
- no-gismu accuracy: `0.1500`
- gismu-only accuracy: `0.3000`
- no-judri accuracy: `0.7000`
- judri-only accuracy: `0.0500`
- bridge carries answer signal: `True`
- predicate-only shortcut warning: `False`
- pointer-only shortcut warning: `False`

## Benchmark Protocol

# Stable Benchmark Protocol

1. Train the active M19 mainline cell under the current contract.
2. Benchmark on the shared M14.5 unified surface with BASE, EN-COT, ZH-COT, RANDOM-SHAPE, and SCRATCHPAD-ONLY controls.
3. Run the sanity audit with BASE-NO-SCRATCHPAD, SCRATCHPAD-ONLY, RANDOM, and Q-FORMER.
4. Run integrity on full, purged, overlap, and masked slices.
5. Run multi-seed replications and broader kill tests before promoting a claim.

Current direct family: `M19` / `M19.31`

## Whole-Program Context

- whole-grid stage count: `22`
- direct unified eval track: `M19.31`
