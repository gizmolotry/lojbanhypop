# M3.18 Re-entry Reboot

## Thesis

The active bridge ablations indicate that the project should stop treating raw symbolic or quasi-Lojban sidecar state as a thing that should be directly exposed to the English generator. The repeated failure mode is interface-level, not just optimization-level.

Observed pattern:
- when sidecar state becomes behaviorally strong, English generation degrades
- when English fluency remains strong, the sidecar becomes weak or decorative
- when answer-selection metrics rise, generative kill tests expose shortcut texture and leakage
- when rollout is strongly conditioned, the model loops, contaminates, or reflects sidecar residue into language

The architectural redirection is therefore:

`English context -> advisor reasoning space -> compressed re-entry -> English continuation`

not

`sidecar -> direct participation in generation`

## New Ontology

The system should be decomposed into four explicit objects:

1. English context encoder
- the base decoder builds a fluent English prompt state

2. Advisor reasoning state
- a separate reasoning substrate tracks roles, bindings, ambiguity, and relational structure
- this state can be discrete, continuous, or hybrid
- Lojban remains inspiration for structure and typing, not a mandatory literal manifold interface

3. Re-entry encoder
- a learned compressor translates advisor state into decoder-native continuation material
- candidate forms:
  - one latent return token
  - a short bundle of return tokens
  - one residual continuation vector
  - a hybrid token-plus-residual translator

4. English resumption
- the main decoder resumes from the compressed re-entry state
- raw advisor state should not remain exposed throughout rollout

## Why M11 Matters

The current M11 discriminative path should be read as evidence, not as the final destination.

It shows that:
- the sidecar/advisor can carry useful cognition
- direct generative re-entry remains unsolved

So the project is not redirecting toward better classification. It is redirecting toward better re-entry into language after sidecar reasoning.

## M3.18 Family

Track name:
- `M3.18 Decoder Re-entry Resume`

Core question:
- what compressed return channel best preserves advisor reasoning while restoring fluent decoder-native continuation?

Cells:
- `A`: control, no advisor and no re-entry
- `B`: frozen single return token
- `C`: frozen multi-return token bundle
- `D`: learned residual continuation vector
- `E`: hybrid token plus residual translator

Mandatory comparison axes:
- advisor state type
- return channel type
- coupling locus in the decoder
- rollout exposure policy
- loss penalties
- fluency and contamination metrics

## Success Criteria

The new family should be judged on more than answer accuracy.

Primary metrics:
- role resolution accuracy
- ambiguity handling accuracy
- multi-hop consistency
- English fluency preservation
- contamination rate
- loop rate
- kill-test performance on structure-dependent examples

Secondary metrics:
- first-token accuracy
- answer delta versus foil
- intervention delta on gold
- state norm / gate / attention entropy
- seed stability

Promotion should require both:
- structural gain
- preserved English continuation quality

## Design Rules

- raw sidecar state must not stay directly exposed during rollout
- re-entry must be compact and decoder-native
- M3.15d/M3.16/M3.17 remain the failed direct-interface family
- M11 discriminative remains the useful-cognition-without-generative-re-entry family
- M3.18 begins the re-entry architecture family

## Repo Implications

- keep a single family registry for tensor-flow semantics, coupling locus, loss profiles, and cell specs
- emit machine-readable ablation contracts in every report manifest
- keep DAG entrypoints aligned with the registry
- make future changes by editing family specs rather than rewriting per-script metadata by hand
