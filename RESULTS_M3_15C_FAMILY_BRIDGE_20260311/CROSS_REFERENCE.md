# M3.15c Family-Conditioned Bridge

## Headline
Family-conditioning did not improve usefulness.

## Metrics
- A overall_accuracy: `0.5`
- B overall_accuracy: `0.5`
- C overall_accuracy: `0.5`
- B family_classification_accuracy: `0.5`
- C family_classification_accuracy: `0.5`
- C mean_operator_entropy: `0.5179627930614239`
- C mean_top1_op_share: `0.7887369791666666`

## Interpretation
- The family auxiliary loss did not produce real family separation on eval.
- B preserved operator diversity but did not move answer quality.
- C partially re-collapsed despite anti-collapse controls.
- This suggests the current family supervision path is too weak or too indirect to become a useful decision signal.
