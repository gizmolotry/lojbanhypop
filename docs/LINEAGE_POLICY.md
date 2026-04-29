# Lineage Policy

## Inheritance Is Explicit

Promotion from one major family to the next must declare:

- selected upstream variant
- inherited components
- reopened components
- rejected components
- promotion basis

No component is assumed to flow forward just because it happened to be locally best.

## Comparison Is Explicit Too

A new branch is not scientifically complete just because it has a normalized ID.
It must also inherit an automatic comparison contract that declares:

- which peer rows inside the same major family it must be compared against
- which ancestor major families it must still be compared against
- which historical families such as `J` or `L` remain mandatory comparators
- which family-specific test contracts must still run

This is how we stop losing J-series obligations when later branches look modern.

## Required Entry Metadata

Every canonical experiment row should expose:

- `canonical_id`
- `normalized_canonical_id`
- `series_major`
- `series_minor`
- `question_boundary`
- `architectural_thesis`
- `inherits_from`
- `inherits_components`
- `frozen_components`
- `changed_components`
- `dropped_components`
- `promotion_basis`
- `metrics_primary`
- `metrics_guardrail`
- `baseline_manifest`

## Historical Backfill Rule

For historical work, inheritance may be marked as:

- `documented`
- `legacy_inferred`
- `documented_redirection`

This allows older families to be mapped honestly without pretending the original repo always enforced modern lineage discipline.

## Component Accounting

Component inventory should be tracked in machine-readable form whenever known, including:

- prompt format
- tokenizer additions
- adapter type
- residual gate type
- injection layer
- loss objective
- data pack
- kill-test suite
- evaluation harness
- discriminator head
- archive/reporting path

## Canonical Source

The canonical machine-readable sources are:

- `configs/experiment_taxonomy.json`
- the generated ablation history manifest under `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/...`

The human-readable companions are:

- `docs/EXPERIMENT_TAXONOMY.md`
- `docs/ABLATION_HISTORY_FULL.md`
