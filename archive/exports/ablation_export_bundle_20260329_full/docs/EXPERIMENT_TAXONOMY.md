# Experiment Taxonomy

## Naming Rule

The canonical public taxonomy is numeric:

- `M<major>.<minor>` for a series variant inside one architectural question boundary
- `M<major>.<minor>.<cell>` for tightly scoped matrix cells inside that variant

Examples:

- `M1.1`
- `M2.3`
- `M3.18.D`
- `M14.C`

Legacy labels such as `J-3`, `L6-A`, and `H5.2b` remain valid as aliases only.

## Semantic Rule

- `major` = architectural question or stage boundary
- `minor` = ablation or variant within that boundary
- `cell` = within-run matrix or bucket cell

Moving from one major family to the next is never implicit. It requires an inheritance manifest.

## Current Normalized Lineage

- `J-1` -> `M1.1`
- `J-2` -> `M1.2`
- `J-3` -> `M1.3`
- `J-4` -> `M1.4`
- `J-5` -> `M1.5`
- `L6-A` -> `M2.1`
- `L6-B` -> `M2.2`
- `L6-C` -> `M2.3`

Existing `M3.*`, `M4.*`, `M5.*`, and later families keep their numeric identity, with cell suffixes preserved where needed.

## Methodological Boundary

Every major family must declare:

- the architectural question
- the allowed ablation axes
- the forbidden drift axes
- the promotion basis
- the primary and guardrail metrics
- the baseline manifest

These machine-readable definitions live in `configs/experiment_taxonomy.json`.
