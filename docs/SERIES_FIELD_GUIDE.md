# Series Field Guide

This is the human-readable map of the repo's experiment history.

If you want one document that answers "what does this series mean, what question was it asking, what did it inherit, what changed, and what did it turn into?" this is that document.

This guide is the translation layer over the heavier control-plane documents:

- [ABLATION_HISTORY_FULL.md](/D:/lojbanhypop/docs/ABLATION_HISTORY_FULL.md)
- [ABLATION_PROGRAM_SPINE.md](/D:/lojbanhypop/docs/ABLATION_PROGRAM_SPINE.md)
- [SERIES_CHARTER.md](/D:/lojbanhypop/docs/SERIES_CHARTER.md)
- [EXPERIMENT_TAXONOMY.md](/D:/lojbanhypop/docs/EXPERIMENT_TAXONOMY.md)
- [LINEAGE_POLICY.md](/D:/lojbanhypop/docs/LINEAGE_POLICY.md)

## How To Read This Repo

There are two different histories living in this project at the same time:

1. The real chronological history.
   This is the messy path the project actually took through lettered eras, partial reboots, architecture pivots, and inconsistent naming.

2. The normalized methodological history.
   This is the newer control-plane view where experiments are described by explicit architectural questions, inheritance manifests, and stable canonical IDs.

Both matter.

The lettered families are not fake. The newer `M<major>.<minor>` taxonomy is not a rewrite of history either. It is a control-plane layer that makes the history auditable.

## The Core Naming Rule

Going forward, the intended public taxonomy is:

- `M<major>` = one architectural question boundary
- `M<major>.<minor>` = one ablation or variant inside that boundary
- `M<major>.<minor>.<cell>` or `M<major>.<minor>.<suffix>` = a tightly scoped matrix cell inside that ablation

Examples:

- `M1.1`
- `M2.3`
- `M3.18.D`
- `M14.C`
- `M19.4`

Legacy names remain valid as aliases:

- `J-3`
- `L6-A`
- `H5.2b`
- `Iron Collar`
- `Gearbox Control`

Those aliases are preserved for lookup and historical honesty, but they are not supposed to drive the structure of future work.

## The Most Important Lineage Rule

Inheritance is explicit, not implicit.

That means:

- later series do not automatically "contain" the architecture of earlier ones
- re-evaluating an older family in a modern harness does not mean that old family retroactively inherited later components
- promotion from one major family to another is a methodological claim that should be declared, not assumed

This matters especially for `H`, `H5`, `J`, and `L`, because those older families are often re-read through modern `M`-series comparisons.

## The Big Picture In One Paragraph

The project starts with early benchmark/control matrices (`A-G`), pushes into direct bridge and advisor-handoff experiments (`H`, `H5`), then splits into diagnostics and data-generation discipline (`J`) and constraint-optimized training (`L`). The normalized `M` era turns those lessons into explicit question-boundary families: `M1` for J-style invariance and adversarial diagnostics, `M2` for L-style constraint optimization, `M3` for bridge exposure and re-entry, `M4-M10` for grounding/serialization/bridge/composition/manifold/translation work, `M11` for discriminative proof that the manifold actually carries cognition, `M14` for scratchpad-mediated re-entry, `M18` for controller-style steering and language-tax comparisons, and `M19` for bounded continuous scratchpad runway and token-efficiency-oriented reasoning.

## Chronology At A Glance

This is the practical spine of the program as it exists in the repo today:

- `A-G`: early benchmark/control matrix
- `H`: direct bridge and handoff experiments
- `H5`: persistent advisor / surgery / extension era
- `J`: advisor-side data, invariance, curriculum, adversarial structure
- `L`: constraint-optimized training and lexicographic/Lagrangian control
- `J/L Hypercube`: orchestration and aggregation surface over older families
- `Phase Eval`: phase-5 objective and evaluation ablations
- `M0`: normalized H5 extension aliases
- `M1`: normalized J family
- `M2`: normalized L family
- `M3`: bridge exposure and return-channel shaping
- `M4`: semantic grounding and predicate family structure
- `M5`: formalization and chain structuring
- `M6`: direct logic-engine bridge
- `M7`: interleaved coprocessor
- `M8`: council-of-oracles composition
- `M9`: provenance manifold and contrastive NLI engine
- `M10`: English translation and return-path adaptation
- `M11`: native discriminative manifold readout
- `M14`: symbiote scratchpad re-entry
- `M18`: two-pass salience controller steering and language-tax audits
- `M19`: bounded continuous scratchpad runway

## Quick Lookup Table

| Series | What It Means | Main Question | What It Led To |
| --- | --- | --- | --- |
| `A-G` | early control matrix | do simple substrate/control variants move benchmark behavior at all? | `H`, later control comparisons |
| `H` | bridge era | can a structured latent bridge hand useful state into the decoder? | `H5`, later bridge skepticism |
| `H5` | persistent advisor / surgery extensions | can the bridge be stabilized with stronger advisor persistence, surgery, or grounding? | `J`, H5 extension aliases, modern legacy comparisons |
| `J` | diagnostics/data generation | is the substrate structurally valid, invariant, adversarially meaningful, and curriculum-stable? | `M1`, later test-contract obligations |
| `L` | constraint-optimized training | can explicit lexicographic/Lagrangian constraints keep structure intact during training? | `M2`, later constraint obligations |
| `Phase Eval` | objective/eval stress tests | which losses and semantic/compression objectives help versus fake lift? | informed later family guardrails |
| `M1` | normalized J | lock invariance and foil discipline before architecture coupling | `M2+` test-contract inheritance |
| `M2` | normalized L | optimize structure with explicit constraints | `M3+` baseline for constraint compliance |
| `M3` | bridge archaeology | how should advisor state re-enter generation without contaminating English? | `M11`, `M14`, much of the re-entry doctrine |
| `M4` | grounding | improve operator/predicate grounding before longer chains | `M5+` |
| `M5` | chain structuring | which reasoning serialization format is viable? | `M6+` |
| `M6` | logic-engine bridge | can direct logic-engine coupling work? | `M7+` |
| `M7` | interleaved coprocessor | can structured steps be interleaved with decoding? | `M8+` |
| `M8` | council composition | can multiple oracle channels outperform one monolith? | `M9+` |
| `M9` | provenance manifold | can reasoning move to a synchronized provenance-aware manifold? | `M10`, `M11` |
| `M10` | translation/return path | can manifold state be translated back into English-compatible continuation? | `M11`, `M14` |
| `M11` | discriminative oracle | does the manifold carry real cognition even if generation still fails? | `M14`, later oracle checks |
| `M14` | scratchpad re-entry | can bounded scratchpad tokens create a safer compute horizon for re-entry? | `M18`, `M19` |
| `M18` | controller steering | can compact salience/controller steering beat raw post-M14 injection? | `M19` |
| `M19` | bounded runway mainline | can a compact scratchpad substrate retain reasoning lift while reducing token cost? | current mainline runway family |

## Detailed Series Guide

## `A-G`: Early Core Matrix

What it was:

- the first serious benchmark/control matrix
- a broad early attempt to compare substrate and control variants before the program had a stable methodology
- the place where "English versus Lojban control duel" style comparisons first mattered as a framing device

What question it was asking:

- does any structured substrate or control variation move benchmark outcomes in a way that is not just noise?

What was inside it:

- base/control comparisons
- projected handoff variants
- coconut-style branches
- English-vs-Lojban control comparisons
- run-level deltas and accuracy-lift style reporting

Why it matters now:

- it established the habit of ablation grids, but not yet the discipline
- it is the earliest source of "there is maybe a signal here, but the interface is unstable"
- many later control and comparator patterns trace back to this era

What it turned into:

- the bridge-focused `H` era

Current status:

- historically important
- mostly legacy/prose/artifact-backed depending on row
- useful as ancestry and baseline context, not as a modern runnable family

## `H`: Direct Bridge Era

What it was:

- the first serious attempt to hand structured advisor state directly into a language model decoder
- linear and SwiGLU-style latent handoff experiments

What question it was asking:

- can the model consume a foreign structured latent state during generation and use it causally without destroying normal English behavior?

What the repo learned:

- bridge exposure is fragile
- preserving some geometry is not enough if the decoder cannot actually read the interface
- apparent gains can coexist with fluency damage or causal ambiguity

Why it matters now:

- `H` is where the "foreign-body" problem starts to become undeniable
- a lot of later skepticism about continuous handoff comes from repeated H-era pain, not just later M-era philosophy

What it turned into:

- `H5`, which tried to stabilize or strengthen the advisor/bridge path

Current status:

- mostly not a clean modern grid
- still relevant conceptually as the beginning of bridge archaeology

## `H5`: Persistent Advisor / Surgery / Extension Era

What it was:

- the late H-era attempt to salvage or extend the bridge idea with stronger advisor persistence, surgery, and grounding discipline

Key subrows and aliases:

- `H5.2a` = `Gearbox Control`
- `H5.2b` = `True Neuro-Symbolic`
- `H5.4` = `Iron Collar`
- `H5.5` = `Grounded Fine-Tune`
- `H5-PROV`
- `H5-OOD`
- `H5-DPTR`

What question it was asking:

- can a more persistent or better-constrained advisor pathway avoid the brittleness seen in earlier bridge work?

What it contributed:

- stronger provenance and OOD stress framing
- dynamic pointer and persistent-advisor style extensions
- early versions of the idea that the substrate needs structural discipline, not just higher scores

What it did not prove:

- it did not prove that direct bridge exposure was fundamentally solved
- it did not cleanly settle the chatbot handoff problem

What it turned into:

- `J`, where the program became much more diagnostic and adversarial
- normalized extension aliases in `M0`

Current status:

- historically important and partially artifact-backed
- can still be re-evaluated in modern harnesses
- should not be described as having inherited later M-series components

## `J`: Advisor-Side Structural Diagnostics

What it was:

- the family that asked whether the substrate and data discipline were structurally real before layering on more architecture

Normalized mapping:

- `J-1` -> `M1.1`
- `J-2` -> `M1.2`
- `J-3` -> `M1.3`
- `J-4` -> `M1.4`
- `J-5` -> `M1.5`

What question it was asking:

- is the structured substrate invariant, adversarially meaningful, curriculum-stable, and resistant to foil acceptance?

Typical themes:

- schema graphs
- invariance
- stopgrad/isolation
- curriculum
- adversarial scope and foil controls

Why it matters more than people sometimes admit:

- a huge amount of later M-series work still depends on J-style obligations
- many modern families carry `j.*` test contracts specifically because architecture work without J-style falsification kept producing fake or shortcut wins

What it turned into:

- normalized `M1`
- permanent downstream test-contract obligations across `M2+`

Current status:

- conceptually alive even when specific legacy scripts are brittle
- more diagnostic than headline-generative

## `L`: Constraint-Optimized Training Family

What it was:

- the family that tried to keep structure intact through explicit lexicographic or Lagrangian constraint control

Normalized mapping:

- `L6-A` -> `M2.1`
- `L6-B` -> `M2.2`
- `L6-C` -> `M2.3`

What question it was asking:

- can careful constraint optimization keep scope, identity, and arity from dissolving during training?

Why it matters:

- L is where "structural validity must be enforced, not hoped for" becomes programmatically explicit
- later families inherit L-style guardrails even when they are no longer doing L-series optimization directly

What it turned into:

- normalized `M2`
- later `l.*` constraint contracts used throughout the M era

Current status:

- historically central but not fully runnable in the way recent M-series families are
- still a methodological ancestor, not dead history

## `J/L Hypercube`: Aggregation Layer, Not A Model Family

What it was:

- an orchestration and aggregation surface over legacy `J` and `L` style runs

What it was not:

- not a standalone scientific architecture family
- not a new substrate or decoder interface by itself

Why it matters:

- it was an early control-plane instinct
- it helped concentrate older experiments into something more comparable

## `Phase Eval`: Objective And Evaluation Stress Tests

What it was:

- the phase-5 train/objective ablation block
- a place where semantic/compression objectives and eval framing were stressed more directly

What question it was asking:

- which objectives improve the intended reasoning behavior versus merely producing prettier metrics?

Why it matters:

- it contributed to the later insistence on causal kill tests and multi-surface evaluation
- it helped expose that metric improvements can come from shortcuts instead of genuine structured use

## `M0`: Normalized H5 Extension Aliases

What it is:

- a normalization layer for H5 extension rows

Known mappings:

- `M0.1` = `H5-PROV`
- `M0.2` = `H5-OOD`
- `M0.3` = `H5-DPTR`

Why it exists:

- to let the control plane talk about those rows using the same canonical machinery as the rest of the M taxonomy

## `M1`: Normalized J Family

What it means:

- the first explicitly normalized major family
- the program says: before doing architecture coupling, lock invariance, foil resistance, and diagnostic discipline

Question boundary:

- data invariance and adversarial synthesis diagnostics

Architectural thesis:

- do not let later architecture claims rest on unstable or contaminated substrate/data assumptions

Typical minors:

- `M1.1` to `M1.5`, mapping from the `J` rows

What it feeds:

- virtually every later family through required J-style contracts

## `M2`: Normalized L Family

What it means:

- the normalized constraint-optimization family

Question boundary:

- constraint-optimized Lagrangian training

Architectural thesis:

- preserve structure with explicit lexicographic augmented-Lagrangian control rather than loose objective mixing

Typical minors:

- `M2.1`, `M2.2`, `M2.3`
- legacy aliases `M2.A`, `M2.B`, `M2.C` and `L6-A/B/C`

What it feeds:

- later families inherit L-style scope, identity, and arity contracts even when their main question moves elsewhere

## `M3`: Bridge Archaeology And Re-Entry Doctrine

What it means:

- this is the generative bridge archaeology block
- it is the family most directly concerned with the question: how should structured advisor state couple back into generation?

Question boundary:

- bridge exposure and return-channel shaping

Architectural thesis:

- test how structured advisor state should couple back into generation without collapsing the decoder's English continuation manifold

Why it is so important:

- M3 contains the project's most explicit evidence that raw symbolic or continuous sidecar exposure can damage English continuation
- it is where the program starts distinguishing "signal exists" from "the decoder can fluently use that signal"

Important rows:

- `M3.15*`
- `M3.17`
- `M3.18`
- `M3.19`

Practical interpretation:

- `M3` is where the repo learned that direct re-entry is an interface problem, not just a tuning problem
- later families like `M11`, `M14`, and `M19` only make sense if you understand the scars from `M3`

## `M4`: Semantic Grounding And Predicate Family Structure

What it means:

- after bridge pain, the program tests whether better operator/predicate grounding is needed before longer structured serialization

Question boundary:

- semantic grounding and predicate family structure

Why it exists:

- because if predicate/operator families are mushy, later chain or manifold work becomes hard to interpret

## `M5`: Formalization And Chain Structuring

What it means:

- a serialization family
- the repo asks which chain formats are viable for structured reasoning traces

Question boundary:

- formalization and chain structuring

Typical themes:

- autoregressive chain formats
- masked-pair formats
- padded n-ary serialization

Why it matters:

- M5 is where "the model wants sequence" becomes operationalized in serialization design, not just theorized

## `M6`: Direct Logic-Engine Bridge

What it means:

- an attempt to bridge directly from a logic engine into model behavior

Question boundary:

- logic engine bridge

Why it matters:

- it tests a cleaner and more explicit bridge thesis than some earlier latent-handoff attempts
- it also helps show the limits of direct bridge optimism

## `M7`: Interleaved Coprocessor

What it means:

- the family where structured computation is interleaved with base decoding rather than only prepended or injected as one-shot foreign state

Question boundary:

- interleaved coprocessor

Why it matters:

- it broadens the search space beyond simple bridge/no-bridge framing

## `M8`: Council Of Oracles

What it means:

- a multi-oracle composition family

Question boundary:

- council-of-oracles composition

What it asks:

- can composed oracle channels outperform a monolithic English baseline or single-path structured rollout?

Why it matters:

- it is part of the repo's decomposition/composition branch, not just bridge surgery

## `M9`: Provenance Manifold And Contrastive NLI Engine

What it means:

- the program pivots toward a provenance-aware manifold and synchronized tokenizer-backed reasoning substrate

Question boundary:

- provenance manifold and contrastive NLI engine

Why it matters:

- M9 is a major conceptual turn
- the project is no longer just asking how to hand latent bridge state into English
- it is building a more explicit reasoning substrate with provenance and contrastive structure

## `M10`: English Translation And Return-Path Adaptation

What it means:

- the family explicitly about translating structured manifold state back into something English-compatible

Question boundary:

- English translation and return-path adaptation

Why it matters:

- this is where the "the missing piece is a return path, not more raw sidecar exposure" intuition becomes first-class

## `M11`: Native Discriminative Manifold Readout

What it means:

- the discriminative oracle family

Question boundary:

- native discriminative manifold readout

Architectural thesis:

- prove that the manifold carries useful cognition even when generative re-entry remains unresolved

Why it matters:

- M11 is not the chatbot end state
- it is the proof-of-cognition oracle
- it tells the repo: there is meaningful signal in the structured/manifold state even if fluent generative handoff is still failing

What it feeds:

- `M14`, because scratchpad re-entry only makes sense once the project believes the advisor/manifold signal is real

## `M14`: Symbiote Scratchpad Re-Entry

What it means:

- bounded scratchpad tokens as a compute horizon before English resumption

Question boundary:

- symbiote scratchpad re-entry

Architectural thesis:

- inject continuous advisor math only into scratchpad states before the model resumes English, rather than forcing direct raw exposure during answer generation

Why it matters:

- this is the repo's most explicit attempt to give the decoder time to absorb structured state over normal transformer hops
- it is a response to M3-style contamination and M11's evidence that useful advisor signal exists

What it contributed:

- scratchpad bleed analysis
- re-entry metrics focused on fluency and contamination
- a safer compute-horizon framing

## `M18`: Two-Pass Salience Controller Steering

What it means:

- a controller-era family

Question boundary:

- two-pass salience controller steering

Architectural thesis:

- compile salient hidden-state structure into a compact steering signal for a second decoder pass, instead of relying on raw foreign-state injection

What this family is famous for:

- harmonized audits
- kill/random controls
- language-tax compactness comparisons
- branch-style comparator surfaces like `EN-CONCISE`, `EN-COT`, `ZH-COT`, `L-TYPED`, `U-TYPED`

Why it matters:

- M18 makes the evaluation culture much healthier
- it stops treating raw accuracy as the only story
- it brings token-cost and causal-control comparisons closer to the center

## `M19`: Bounded Continuous Scratchpad Runway

What it means:

- the current bounded runway mainline

Question boundary:

- bounded continuous scratchpad runway

Architectural thesis:

- compress tapped hidden state into a learned continuous runway over repeated `<symbiote>` positions so the decoder gets multiple normal attention hops before final-answer English resumption

Important sub-era inside M19:

- `M19.3`: static-cell and isolation-grid era
- `M19.4`: dynamic pacing attempt with `<symbiote_end>` and variable-length runway logic

What question it is really asking:

- can a compact scratchpad substrate retain enough reasoning lift while using far fewer generated tokens than explicit English CoT?

Why it matters:

- M19 is where token efficiency becomes part of the core scientific objective, not just a side metric
- the family is directly comparable against `EN-CONCISE`, `EN-COT`, and `ZH-COT` branch-style comparators

Current honest reading:

- M19 is promising when viewed as a compression-efficiency program
- it is not yet the final chatbot symbiote solution
- it is the cleanest current expression of the compact-runway idea

## What Changed Across The Whole Program

The simplest honest summary is this:

- `A-G` asked whether there was any signal at all.
- `H/H5` tried to directly bridge structured state into generation.
- `J/L` forced the project to care about validity, invariance, scope, and constraint discipline.
- `M1/M2` normalized those obligations into explicit major families.
- `M3` showed that direct re-entry is an interface problem.
- `M4-M10` explored grounding, serialization, bridge variants, manifold structure, and translation.
- `M11` proved the structured/manifold state could carry useful cognition in discriminative form.
- `M14` tried to give that cognition a safer scratchpad-mediated re-entry path.
- `M18` improved evaluation discipline and controller-style steering.
- `M19` turned compact runway and token efficiency into the current mainline problem.

## What The Repo Tends To Confuse

These are the misunderstandings this guide is trying to prevent:

`Older families did not retroactively inherit newer components.`

If `H5` is benchmarked today next to `M19`, that does not mean `H5` secretly contains M19 architecture. It means the control plane can compare them under one reporting surface.

`The M taxonomy is a control layer, not a deletion of legacy history.`

When `J-3` becomes `M1.3` as a normalized canonical ID, the old label is not being erased. It is being made auditable.

`Diagnostic families and generative families are not the same kind of success.`

`J` and `M11` are incredibly important, but they are not identical to a fluent chatbot-success family. They prove substrate or signal properties that later families try to exploit.

`A good score is not enough.`

This repo has repeatedly produced results where metrics looked good and causal usage was weak, shortcut-driven, or contaminated. That is why later families inherit J-style kill tests, L-style constraint checks, and M3/M14-style fluency/bleed checks.

## The Best Mental Model For A New Engineer

If a strong senior engineer opens this repo, the most useful mental model is:

- `J` tells you whether the substrate and data discipline are real.
- `L` tells you whether structure survives optimization.
- `M3` tells you why naive re-entry into language fails.
- `M11` tells you whether the structured state contains useful cognition at all.
- `M14` tells you whether bounded scratchpads help re-entry.
- `M18` tells you whether the evaluation culture is honest.
- `M19` tells you where the current compact-runway mainline actually stands.

Everything else is either a precursor, a specialization, or a bridge between those ideas.

## If You Need One Sentence Per Series

- `A-G`: early benchmark matrix that proved there might be signal, but not yet discipline.
- `H`: first serious bridge era trying to inject structured state into generation.
- `H5`: stronger advisor/surgery/extensions trying to rescue the bridge idea.
- `J`: structural diagnostics and adversarial discipline for the substrate itself.
- `L`: constraint-optimized training to preserve scope/identity/arity.
- `Phase Eval`: objective stress tests that separated real lift from metric theater.
- `M0`: normalized aliases for H5 extension rows.
- `M1`: normalized J-style invariance and adversarial synthesis family.
- `M2`: normalized L-style constraint optimization family.
- `M3`: bridge archaeology and return-channel shaping.
- `M4`: grounding operator and predicate family structure.
- `M5`: formalizing reasoning as sequence/chain structure.
- `M6`: direct logic-engine bridge testing.
- `M7`: interleaved coprocessor experiments.
- `M8`: council-of-oracles composition.
- `M9`: provenance-aware manifold and contrastive substrate.
- `M10`: translation of manifold state back toward English compatibility.
- `M11`: discriminative proof that the manifold carries useful cognition.
- `M14`: scratchpad-mediated re-entry using bounded compute horizon.
- `M18`: controller-style steering plus honest branch/control audits.
- `M19`: current compact runway family optimizing reasoning retention versus token cost.

## Where To Go Next

If you want raw exhaustive row-by-row history:

- [ABLATION_HISTORY_FULL.md](/D:/lojbanhypop/docs/ABLATION_HISTORY_FULL.md)

If you want the canonical lineage chain:

- [ABLATION_PROGRAM_SPINE.md](/D:/lojbanhypop/docs/ABLATION_PROGRAM_SPINE.md)

If you want the naming and inheritance rules:

- [EXPERIMENT_TAXONOMY.md](/D:/lojbanhypop/docs/EXPERIMENT_TAXONOMY.md)
- [LINEAGE_POLICY.md](/D:/lojbanhypop/docs/LINEAGE_POLICY.md)

If you want the charter-style map of series boundaries and aliases:

- [SERIES_CHARTER.md](/D:/lojbanhypop/docs/SERIES_CHARTER.md)
