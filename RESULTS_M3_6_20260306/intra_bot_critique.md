# Intra-Bot Critique

## Agent Franklin (risk audit)
- Swap generation can swap header tokens instead of semantic entities in some prompt styles.
- infer_swap_semantics is lexicon/format brittle and under-detects symmetric cases.
- Foil target fallback can introduce label noise on non-binary answers.
- Identity penalty applies only to invariant path; routing errors change optimization regime.

## Agent Nash (metrics schema)
- Recommended explicit per-cell schema with total/symmetric/asymmetric oracle accuracy.
- Include confusion matrix and secondary truth/canonical rates.
- Keep per-cell and aggregated report artifacts versioned and diff-friendly.
