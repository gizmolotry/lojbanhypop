# Ablation Program Spine

- Generated UTC: `2026-05-21T20:35:54.602633+00:00`
- Source history manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/m22_contract_tightening_history_20260521/ablation_history_manifest.json`
- Source taxonomy config: `configs/experiment_taxonomy.json`
- Stage count: `25`

This is the ordered research spine of the project: legacy letter-series families, normalized M-major families, and the control plane that keeps the program auditable.

## 1. A-G

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: early benchmark matrix covering base control, projected handoff, coconut variants, and English-vs-Lojban control comparisons.
- Entry count: `14`
- Runnable rows: `0`
- Artifact-only rows: `5`
- Doc-only rows: `9`
- IDs: `a_to_g.a, a_to_g.b_1, a_to_g.b_2, a_to_g.c, a_to_g.d, a_to_g.e, legacy.core.a, legacy.core.b, legacy.core.c, legacy.core.e, legacy.core.f, legacy.core.g, legacy.duel.english_cot, legacy.duel.lojban_topology`
- Aliases: `A, A-G/A, A-G/B.1, A-G/B.2, A-G/C, A-G/D, A-G/E, B, B.1, B.2, C, Control Duel/English, Control Duel/Lojban, Core/A, Core/B, Core/C, Core/E, Core/F, Core/G, D, E, English CoT, F, G, Lojban Dual, Run B`
- Docs: `docs/ledger/CANONICAL_LEDGER.md, docs/history/reports/AUDIT_REPORT.md`
- Scripts: `scripts/legacy/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/control_plane/lojban_ablation_matrix_dag.py`

## 2. H

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: mid-layer bridge experiments testing linear and SwiGLU geometric handoff into the host decoder.
- Entry count: `5`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Doc-only rows: `4`
- IDs: `h.series.h3, legacy.h.h1, legacy.h.h2, legacy.h.h3, legacy.h.h4`
- Aliases: `H1, H2, H3, H4`
- Docs: `docs/ledger/CANONICAL_LEDGER.md, docs/history/reports/NUMERICAL_AUDIT.md`
- Scripts: `scripts/legacy/true_coconut.py`

## 3. H5

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: boolean-surgery, persistent-advisor, and bridge-extension experiments that fed into the later J and L stacks.
- Entry count: `7`
- Runnable rows: `0`
- Artifact-only rows: `3`
- Doc-only rows: `4`
- IDs: `h.series.h5_dptr, h.series.h5_ood, h.series.h5_prov, legacy.h5.h5_2a, legacy.h5.h5_2b, legacy.h5.h5_4, legacy.h5.h5_5`
- Aliases: `Gearbox Control, H5-DPTR, H5-OOD, H5-PROV, H5.2a, H5.2b, H5.4, H5.5, Iron Collar, True Neuro-Symbolic`
- Docs: `docs/history/reports/H5_ABLATION_REPORT.md, docs/history/reports/H5_SUMMARY.md`
- Scripts: `archive/legacy_scaffolding/trace_h5_provenance.py, scripts/legacy/eval_h5_dynamic_pointer_refactor.py, scripts/legacy/eval_h5_ood_stress.py, scripts/legacy/train_h5_persistent_vq_advisor.py`

## 4. J

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: advisor-side data generation and adversarial synthesis family that seeded the later numeric M-line.
- Entry count: `5`
- Runnable rows: `5`
- IDs: `M1.1, M1.2, M1.3, M1.4, M1.5`
- Aliases: `J-1, J-2, J-3, J-4, J-5, M1.1, M1.2, M1.3, M1.4, M1.5`
- Docs: `docs/SERIES_CHARTER.md, docs/ledger/CANONICAL_LEDGER.md`
- Scripts: `archive/legacy_scaffolding/eval_j_1.py, archive/legacy_scaffolding/eval_j_2.py, archive/legacy_scaffolding/eval_j_3.py, archive/legacy_scaffolding/eval_j_4.py, archive/legacy_scaffolding/eval_j_5.py, scripts/legacy/train_h5_persistent_vq_advisor.py`
- DAGs: `airflow/dags/legacy/lojban_j_series_dag.py`

## 5. L

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: lagrangian constrained-manifold family and its branch lineages before the later M unification.
- Entry count: `21`
- Runnable rows: `3`
- Artifact-only rows: `17`
- Doc-only rows: `1`
- IDs: `M2.1, M2.2, M2.3, M3.0, M3.1, M3.2, M3.3, M3.4, l.branch.m3_5.m3_5_a, l.branch.m3_5.m3_5_b, l.branch.m3_5.m3_5_c, l.branch.m3_6.m3_6_a, l.branch.m3_6.m3_6_b, l.branch.m3_6.m3_6_c, l.branch.m3_7.m3_7_a, l.branch.m3_7.m3_7_b, l.branch.m3_7.m3_7_c, l.branch.m3_8.m3_8_a, l.branch.m3_8.m3_8_b, l.branch.m3_8.m3_8_c, l.series.charter`
- Aliases: `L-Series, L6-A, L6-B, L6-C, Lagrangian Series, M2.1, M2.2, M2.3, M2.A, M2.B, M2.C, M3.0, M3.1, M3.2, M3.3, M3.4, M3.5.A, M3.5.B, M3.5.C, M3.6.A, M3.6.B, M3.6.C, M3.6.M3.6.A, M3.6.M3.6.B, M3.6.M3.6.C, M3.7.A, M3.7.B, M3.7.C, M3.8.A, M3.8.B, M3.8.C`
- Docs: `docs/SERIES_CHARTER.md, archive/reports/relevant/REPORTS_RELEVANT/l6_ablation_manifest.md`
- Scripts: `scripts/legacy/run_l6_ablation_branch.py, scripts/legacy/train_l_series_mvs.py`
- DAGs: `airflow/dags/legacy/lojban_l_series_dag.py`

## 6. J/L Hypercube

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: cross-family orchestration and hypercube reporting layer that consolidated J/L-era runs before the modern M suite.
- Entry count: `0`
- Runnable rows: `0`
- Docs: `archive/reports/relevant/REPORTS_RELEVANT/ablation_hypercube_report.md, archive/results/legacy_misc/20260305/RESULTS_FULL_GRID_20260305/ablation_hypercube_report.md`
- DAGs: `airflow/dags/control_plane/lojban_ablation_hypercube_report_dag.py`

## 7. Phase Eval

- Kind: `legacy_series`
- Layer: `legacy_orchestration`
- Objective: phase-5 train/objective ablations used to stress semantic and compression loss surfaces before later M-series serialization work.
- Entry count: `14`
- Runnable rows: `14`
- IDs: `phase5.objective.ablate_compositional_consistency_loss, phase5.objective.ablate_compression_regularization_loss, phase5.objective.ablate_coverage_regularization_loss, phase5.objective.ablate_roundtrip_consistency_loss, phase5.objective.ablate_semantic_unambiguity_loss, phase5.objective.baseline_no_phase5, phase5.objective.phase5_full, phase5.train.ablate_compositional_consistency_weight, phase5.train.ablate_compression_regularization_weight, phase5.train.ablate_coverage_regularization_weight, phase5.train.ablate_roundtrip_consistency_weight, phase5.train.ablate_semantic_unambiguity_weight, phase5.train.baseline_no_phase5, phase5.train.phase5_full`
- Aliases: `ablate_compositional_consistency_loss, ablate_compositional_consistency_weight, ablate_compression_regularization_loss, ablate_compression_regularization_weight, ablate_coverage_regularization_loss, ablate_coverage_regularization_weight, ablate_roundtrip_consistency_loss, ablate_roundtrip_consistency_weight, ablate_semantic_unambiguity_loss, ablate_semantic_unambiguity_weight, baseline_no_phase5, phase5_full`
- Docs: `docs/SERIES_CHARTER.md`
- Scripts: `scripts/legacy/run_phase5_objective_ablation.py, scripts/legacy/run_phase5_train_ablation.py`
- DAGs: `airflow/dags/control_plane/lojban_phase_ablation_dag.py`

## 8. M1

- Kind: `major_series`
- Layer: `legacy_orchestration`
- Objective: lock dataset invariance, paraphrase robustness, and foil acceptance before architectural coupling work.
- Legacy origin: `J`
- Entry count: `5`
- Runnable rows: `5`
- Question boundary: data invariance and adversarial synthesis diagnostics
- Thesis: lock dataset invariance, paraphrase robustness, and foil acceptance before architectural coupling work.
- IDs: `M1.1, M1.2, M1.3, M1.4, M1.5`
- Aliases: `J-1, J-2, J-3, J-4, J-5, M1.1, M1.2, M1.3, M1.4, M1.5`
- Allowed axes: `generator depth, foil acceptance thresholds, invariance synthesis, scope diagnostics`
- Frozen/forbidden drift: `bridge architecture changes, decoder residual injection, scratchpad token mediation`
- Promotion basis: `accept_rate_by_depth, accepted_foil_pair_accuracy, invariance rate`
- Primary metrics: `accepted_foil_pair_accuracy, invariance rate`
- Guardrail metrics: `accept_rate_by_depth`
- Historical comparison families: `J`
- Required test contracts: `j.accept_rate_by_depth, j.accepted_foil_pair_accuracy, j.invariance_rate, j.schema_validity`
- Automatic comparison targets: `M1, J, M1.1, M1.5`
- Baseline manifest: `docs/baselines/m1_series_baseline_manifest.json`
- Scripts: `scripts/legacy/eval_j_1.py, scripts/legacy/eval_j_2.py, scripts/legacy/eval_j_3.py, scripts/legacy/eval_j_4.py, scripts/legacy/eval_j_5.py, scripts/legacy/train_h5_persistent_vq_advisor.py`

## 9. M2

- Kind: `major_series`
- Layer: `legacy_orchestration`
- Objective: replace static objective blending with lexicographic augmented-Lagrangian control while preserving scoped identity and arity constraints.
- Legacy origin: `L`
- Entry count: `3`
- Runnable rows: `3`
- Question boundary: constraint-optimized Lagrangian training
- Thesis: replace static objective blending with lexicographic augmented-Lagrangian control while preserving scoped identity and arity constraints.
- Selected upstream: `M1.5`
- Inherits: `generator-derived diagnostic dataset discipline, scope and foil stress awareness`
- Reopens: `training objective regime, constraint controller`
- Rejects: `diagnostic-only success criterion`
- IDs: `M2.1, M2.2, M2.3`
- Aliases: `L6-A, L6-B, L6-C, M2.1, M2.2, M2.3, M2.A, M2.B, M2.C`
- Allowed axes: `tier-a lock regime, scope drill pressure, tier-b force valves, soft identity audit`
- Frozen/forbidden drift: `bridge exposure strategy, scratchpad tokenizer changes, decoder continuation losses`
- Promotion basis: `constraint_arity_strict, constraint_scope, constraint_identity`
- Primary metrics: `constraint_scope, constraint_identity`
- Guardrail metrics: `constraint_arity_strict`
- Historical comparison families: `J, L`
- Required test contracts: `j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M2, M1, J, L, M2.1, M2.3`
- Baseline manifest: `docs/baselines/m2_series_baseline_manifest.json`
- Scripts: `scripts/legacy/run_l6_ablation_branch.py, scripts/legacy/train_l_series_mvs.py`

## 10. M3

- Kind: `major_series`
- Layer: `bridge_and_serialization`
- Objective: test how structured advisor state should couple back into generation without collapsing the decoder's English continuation manifold.
- Entry count: `48`
- Runnable rows: `12`
- Artifact-only rows: `36`
- Question boundary: bridge exposure and return-channel shaping
- Thesis: test how structured advisor state should couple back into generation without collapsing the decoder's English continuation manifold.
- Selected upstream: `M2.1`
- Inherits: `constraint-aware training stack, scope and identity discipline`
- Reopens: `bridge exposure strategy, return-path geometry`
- Rejects: `static objective-only optimization as sole lever`
- IDs: `M3.0, M3.1, M3.10, M3.11, M3.12.A, M3.12.B, M3.12.C, M3.13.A, M3.13.B, M3.13.C, M3.13.D, M3.14.A, M3.14.B, M3.14.C, M3.15.A, M3.15.B, M3.15.C, M3.15b.A, M3.15b.B, M3.15b.C, M3.15c.A, M3.15c.B, M3.15c.C, M3.15d.A, M3.15d.B, M3.15d.C, M3.15d.D, M3.16.A, M3.16.B, M3.16.C, M3.16.D, M3.17.A, M3.17.B, M3.17.C, M3.17.D, M3.18.A, M3.18.B, M3.18.C, M3.18.D, M3.18.E`
- Aliases: `A, B, C, D, D0, D1, D2, D3, E, M3.0, M3.1, M3.10, M3.11, M3.12.A, M3.12.B, M3.12.C, M3.13.A, M3.13.B, M3.13.C, M3.13.D, M3.14.A, M3.14.B, M3.14.C, M3.15.A, M3.15.B, M3.15.C, M3.15b.A, M3.15b.B, M3.15b.C, M3.15c.A, M3.15c.B, M3.15c.C, M3.15d.A, M3.15d.B, M3.15d.C, M3.15d.D, M3.16.A, M3.16.B, M3.16.C, M3.16.D`
- Allowed axes: `LoRA positioning, bias or return-state geometry, answer path forcing, residual re-entry compression, continuation supervision`
- Frozen/forbidden drift: `base dataset family replacement, discriminator-only evaluation as sole success criterion`
- Promotion basis: `mean_intervention_delta_gold, resume_first_token_accuracy, english_fluency_score`
- Primary metrics: `overall_accuracy, mean_intervention_delta_gold, resume_first_token_accuracy`
- Guardrail metrics: `english_fluency_score, loop_rate, contamination_rate`
- Historical comparison families: `J, L`
- Required test contracts: `m3.reentry_intervention, m3.reentry_fluency, j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M3, M2, M1, J, L, M3.15d, M3.17, M3.18, M3.19`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m3/run_m3_10_ood_accuracy_probe.py, scripts/m3/run_m3_11_winograd_failure_anatomy.py, scripts/m3/run_m3_12_geometric_return_stream.py, scripts/m3/run_m3_13_geometric_ablation_grid.py, scripts/m3/run_m3_14_structural_alignment_bridge.py, scripts/m3/run_m3_15_rotary_coconut.py, scripts/m3/run_m3_15b_relation_local_rotary.py, scripts/m3/run_m3_15c_family_conditioned_bridge.py, scripts/m3/run_m3_15d_answer_path_forcing.py, scripts/m3/run_m3_16_continuous_graph_bias.py, scripts/m3/run_m3_17_advisor_reentry_bridge.py, scripts/m3/run_m3_18_decoder_reentry_resume.py`
- DAGs: `airflow/dags/m3/lojban_m3_18_decoder_reentry_resume_dag.py`

## 11. M4

- Kind: `major_series`
- Layer: `bridge_and_serialization`
- Objective: improve predicate and operator grounding before later chain-style reasoning expansions.
- Entry count: `2`
- Runnable rows: `2`
- Question boundary: semantic grounding and predicate family structure
- Thesis: improve predicate and operator grounding before later chain-style reasoning expansions.
- Selected upstream: `M3.8.C`
- Inherits: `constraint-aware branch training stack, early bridge geometry probes, operator diversification pressure`
- Reopens: `semantic probing, predicate grounding, operator family structure`
- Rejects: `bridge exposure as the only progress axis`
- IDs: `M4.0, M4.2`
- Aliases: `M4.0, M4.2`
- Allowed axes: `semantic probe family, predicate grounding, operator family evaluation`
- Frozen/forbidden drift: `scratchpad token injection, M11 discriminative readout substitution`
- Promotion basis: `held_out_accuracy, operator family consistency`
- Primary metrics: `held_out_accuracy`
- Guardrail metrics: `logical_accuracy`
- Historical comparison families: `J, L`
- Required test contracts: `m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M4, M3, M2, M1, J, L, M4.0, M4.2`
- Baseline manifest: `docs/baselines/m_series_baseline_manifest.json`
- Scripts: `scripts/m4/run_m4_0_semantic_probe.py, scripts/m4/run_m4_2_predicate_grounding.py`

## 12. M5

- Kind: `major_series`
- Layer: `bridge_and_serialization`
- Objective: test autoregressive, masked-pair, and padded-n-ary chain formats for structured reasoning serialization.
- Entry count: `9`
- Runnable rows: `2`
- Artifact-only rows: `7`
- Question boundary: formalization and chain structuring
- Thesis: test autoregressive, masked-pair, and padded-n-ary chain formats for structured reasoning serialization.
- Selected upstream: `M4.2`
- Inherits: `predicate-family grounding signal, operator family evaluation harness`
- Reopens: `serialization format, chain supervision objective, formalization depth`
- Rejects: `probe-only evaluation as sufficient end state`
- IDs: `M5.0.A, M5.0.B, M5.0.C, M5.1.N0, M5.1.N1, M5.1.N2, M5.1.N3, M5.2, M5.3`
- Aliases: `M5.2.autoregressive_chain.run, M5.3.masked_pair_chain.run, M5.A, M5.B, M5.C, M5.N0, M5.N1, M5.N2, M5.N3`
- Allowed axes: `autoregressive chain, masked pair chain, padded n-ary serialization`
- Frozen/forbidden drift: `M9 manifold replacement, M14 scratchpad mediation`
- Promotion basis: `held_out_accuracy, logical_accuracy`
- Primary metrics: `held_out_accuracy, logical_accuracy`
- Guardrail metrics: `final_ce_loss`
- Historical comparison families: `J, L`
- Required test contracts: `m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.accepted_foil_pair_accuracy, j.invariance_rate, l.constraint_scope, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M5, M4, M3, M2, M1, J, L, M5.1, M5.2, M5.3`
- Baseline manifest: `docs/baselines/m_series_baseline_manifest.json`
- Scripts: `scripts/m5/run_m5_2_autoregressive_chain.py, scripts/m5/run_m5_3_masked_pair_chain.py, scripts/m5/run_m5_autoformalization.py`

## 13. M6

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: test a direct logic-engine bridge and its calibration limits before interleaved and council-style decompositions.
- Entry count: `5`
- Runnable rows: `0`
- Artifact-only rows: `5`
- Question boundary: logic engine bridge
- Thesis: test a direct logic-engine bridge and its calibration limits before interleaved and council-style decompositions.
- Selected upstream: `M5.3`
- Inherits: `chain-style structured supervision, formalized reasoning serialization`
- Reopens: `logic-engine bridge, alignment and scratchpad exposure`
- Rejects: `serialization-only progress without stronger bridge tests`
- IDs: `M6.0, M6.1, M6.2, M6.3, M6.6`
- Aliases: `M6.0, M6.1, M6.2, M6.3, M6.6, RESULTS_M6_1_ALIGNMENT_70ACC, RESULTS_M6_2_ALIGNED_30ACC, RESULTS_M6_3_SCRATCHPAD_35ACC, RESULTS_M6_6_DIRECTED_AST_FINAL, RESULTS_M6_SEVERED_BRIDGE_20260314`
- Allowed axes: `logic engine bridge scale, alignment regime, scratchpad direct bridge`
- Frozen/forbidden drift: `council routing, native manifold discriminative swap`
- Promotion basis: `held_out_accuracy, english_fluency_score`
- Primary metrics: `held_out_accuracy`
- Guardrail metrics: `english_fluency_score`
- Historical comparison families: `J, L`
- Required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M6, M5, M4, M3, M2, M1, J, L`
- Baseline manifest: `docs/baselines/m_series_baseline_manifest.json`
- Scripts: `scripts/m6/eval_m6_logic_engine.py, scripts/m6/train_m6_logic_engine.py`

## 14. M7

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: interleave structured coprocessor steps with base decoding while preserving operator semantics.
- Entry count: `1`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Question boundary: interleaved coprocessor
- Thesis: interleave structured coprocessor steps with base decoding while preserving operator semantics.
- Selected upstream: `M6.3`
- Inherits: `logic-engine bridge calibration lessons, scratchpad-directed bridge experience`
- Reopens: `interleaving schedule, coprocessor timing`
- Rejects: `single-pass bridge injection as the only coupling mode`
- IDs: `M7.0`
- Aliases: `M7, M7.0, RESULTS_M7_INTERLEAVED_COPROCESSOR`
- Allowed axes: `interleaving schedule, semantic probe alignment`
- Frozen/forbidden drift: `multi-oracle council voting`
- Promotion basis: `held_out_accuracy, semantic probe stability`
- Primary metrics: `held_out_accuracy`
- Guardrail metrics: `logical_accuracy`
- Historical comparison families: `J, L`
- Required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M7, M6, M5, M4, M3, M2, M1, J, L`
- Baseline manifest: `docs/baselines/m_series_baseline_manifest.json`
- Scripts: `scripts/m7/eval_m7_interleaved.py, scripts/m7/train_m7_interleaved.py`

## 15. M8

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: test multi-oracle composition against a monolithic English baseline and direct council rollout.
- Entry count: `1`
- Runnable rows: `0`
- Artifact-only rows: `1`
- Question boundary: council-of-oracles composition
- Thesis: test multi-oracle composition against a monolithic English baseline and direct council rollout.
- Selected upstream: `M7.0`
- Inherits: `interleaved structured rollout, coprocessor timing discipline`
- Reopens: `multi-oracle composition, council aggregation`
- Rejects: `single-coprocessor rollout as final architecture`
- IDs: `M8.0`
- Aliases: `M8, M8.0, RESULTS_M8_COUNCIL_OF_ORACLES`
- Allowed axes: `oracle composition, baseline comparison`
- Frozen/forbidden drift: `M9 provenance manifold substitution`
- Promotion basis: `held_out_accuracy, final_answer_lift`
- Primary metrics: `held_out_accuracy, final_answer_lift`
- Guardrail metrics: `logical_accuracy`
- Historical comparison families: `J, L`
- Required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- Baseline manifest: `docs/baselines/m_series_baseline_manifest.json`
- Scripts: `scripts/m8/eval_m8_council.py, scripts/m8/train_m8_council.py`

## 16. M9

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: replace monolithic vocabulary state with a provenance-aware manifold and synchronized tokenizer-backed reasoning engine.
- Entry count: `2`
- Runnable rows: `0`
- Artifact-only rows: `2`
- Question boundary: provenance manifold and contrastive NLI engine
- Thesis: replace monolithic vocabulary state with a provenance-aware manifold and synchronized tokenizer-backed reasoning engine.
- Selected upstream: `M8.0`
- Inherits: `structured multi-agent reasoning lessons, English baseline comparison harness`
- Reopens: `native provenance manifold, phase curriculum, tokenizer synchronization`
- Rejects: `oracle composition as the terminal substrate`
- IDs: `M9.0, M9.1`
- Aliases: `M9.0, M9.1, M9.audit, M9.hypercube, RESULTS_M9_AUDIT, RESULTS_M9_HYPERCUBE`
- Allowed axes: `phase curriculum, mov gate, phase forge progression, synchronized tokenizer path`
- Frozen/forbidden drift: `M10 translator/head swap during base forge phases`
- Promotion basis: `held_out_accuracy, logical_accuracy, codebook health`
- Primary metrics: `held_out_accuracy, logical_accuracy`
- Guardrail metrics: `geometry_retention`
- Historical comparison families: `J, L`
- Required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- Baseline manifest: `docs/baselines/m_series_baseline_manifest.json`
- Scripts: `scripts/m9/eval_m9.py`

## 17. M10

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: translate structured manifold state into English-compatible continuation or discriminative readout heads.
- Entry count: `4`
- Runnable rows: `0`
- Artifact-only rows: `4`
- Question boundary: english translation and return-path adaptation
- Thesis: translate structured manifold state into English-compatible continuation or discriminative readout heads.
- Selected upstream: `M9.0`
- Inherits: `provenance manifold engine, synchronized reasoning state, audit harnesses`
- Reopens: `translator depth, English head design, joint alignment`
- Rejects: `direct manifold outputs without translation or readout heads`
- IDs: `M10.0, M10.1, M10.2, M10.3`
- Aliases: `M10.0, M10.1, M10.2, M10.3, M10.audit, M10.final_bridge, M10.floor_lock, M10.publication, RESULTS_M10_AUDIT, final_bridge_audit, final_floor_lock, final_publication_metrics`
- Allowed axes: `translator depth, english head structure, joint alignment`
- Frozen/forbidden drift: `re-entry scratchpad mediation without explicit family reboot`
- Promotion basis: `macro_f1, held_out_accuracy, final_ce_loss`
- Primary metrics: `held_out_accuracy, macro_f1`
- Guardrail metrics: `final_ce_loss`
- Historical comparison families: `J, L`
- Required test contracts: `j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m10/final_audit.py`

## 18. M11

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: prove that the manifold carries useful cognition through native discriminative heads even when generative re-entry remains unresolved.
- Entry count: `0`
- Runnable rows: `0`
- Question boundary: native discriminative manifold readout
- Thesis: prove that the manifold carries useful cognition through native discriminative heads even when generative re-entry remains unresolved.
- Selected upstream: `M10`
- Inherits: `translator and English head infrastructure, final audit harness`
- Reopens: `native manifold structure, discriminative bridge training`
- Rejects: `monolithic vocabulary embeddings`
- Allowed axes: `discriminative adapter training, head readout, adapter disable ablation`
- Frozen/forbidden drift: `raw generative rollout claims without re-entry metrics`
- Promotion basis: `mean_accuracy, macro_f1`
- Primary metrics: `mean_accuracy, macro_f1`
- Guardrail metrics: `held_out_accuracy`
- Historical comparison families: `J, L`
- Required test contracts: `m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`

## 19. M14

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: use bounded scratchpad tokens as a compute horizon while injecting continuous advisor math only into scratchpad states before English resumption.
- Entry count: `0`
- Runnable rows: `0`
- Question boundary: symbiote scratchpad re-entry
- Thesis: use bounded scratchpad tokens as a compute horizon while injecting continuous advisor math only into scratchpad states before English resumption.
- Selected upstream: `M11`
- Inherits: `manifold oracle signal, M11 discriminative head as cognition proof`
- Reopens: `generative return path, scratchpad compute horizon, residual injection locality`
- Rejects: `raw direct sidecar exposure during answer rollout`
- Allowed axes: `scratchpad length, scratchpad gate threshold, injection layer, residual scratchpad alpha`
- Frozen/forbidden drift: `direct raw sidecar exposure to answer token, token-only return path as mainline`
- Promotion basis: `mean_intervention_delta_gold, resume_first_token_accuracy, scratchpad_bleed_rate`
- Primary metrics: `mean_intervention_delta_gold, resume_first_token_accuracy`
- Guardrail metrics: `english_fluency_score, loop_rate, scratchpad_bleed_rate`
- Historical comparison families: `J, L`
- Required test contracts: `m14.scratchpad_bleed, m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M14, M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L, M3.18.D, M3.19.D0, M14.C`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`

## 20. M18

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: extract salient hidden-state structure, induce a compact relation graph, and steer a second decoder pass with compiled attention bias instead of raw foreign-state injection.
- Entry count: `14`
- Runnable rows: `14`
- Question boundary: two-pass salience controller steering
- Thesis: extract salient hidden-state structure, induce a compact relation graph, and steer a second decoder pass with compiled attention bias instead of raw foreign-state injection.
- Selected upstream: `M14.D`
- Inherits: `bounded-intervention discipline, scratchpad-era causal skepticism, fluency and loop guardrails`
- Reopens: `controller extraction path, salience typing, attention bias steering`
- Rejects: `direct residual scratchpad injection as the only post-M14 strategy`
- IDs: `M18, M18.1.A, M18.1.B, M18.1.C, M18.1.D, M18.1.E, M18.1.F, M18.1.G, M18.2.A, M18.2.B, M18.2.C, M18.2.D, M18.2.E, M18.2.F`
- Aliases: `EN-CONCISE, EN-COT, EN-COT+KILL-RANDOM, EN-COT+L-TYPED, EN-COT+U-TYPED, KILL-LABEL, KILL-RANDOM, L-TYPED, M18, M18.harmonized_audit.EN-CONCISE, M18.harmonized_audit.EN-COT, M18.harmonized_audit.KILL-LABEL, M18.harmonized_audit.KILL-RANDOM, M18.harmonized_audit.L-TYPED, M18.harmonized_audit.U-TYPED, M18.harmonized_audit.ZH-COT, M18.hybrid_cot_audit.EN-CONCISE, M18.hybrid_cot_audit.EN-COT, M18.hybrid_cot_audit.EN-COT+KILL-RANDOM, M18.hybrid_cot_audit.EN-COT+L-TYPED, M18.hybrid_cot_audit.EN-COT+U-TYPED, M18.hybrid_cot_audit.ZH-COT, U-TYPED, ZH-COT`
- Allowed axes: `salience selector training, ontology choice, interpreter warm start, bias compiler and intervention layers, hybrid cot comparison`
- Frozen/forbidden drift: `raw scratchpad runway injection as the main intervention surface, headline accuracy claims without kill or random controls`
- Promotion basis: `u_typed_accuracy, l_typed_accuracy, kill_random_gap`
- Primary metrics: `u_typed_accuracy, l_typed_accuracy`
- Guardrail metrics: `kill_random_accuracy, language_tax_tokens`
- Historical comparison families: `J, L`
- Required test contracts: `m18.kill_random_gap, m18.language_tax_compactness, m14.scratchpad_bleed, m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M18, M14, M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L, M14.D`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m18/run_harmonized_audit.py, scripts/m18/run_hybrid_cot_audit.py, scripts/m18/run_m18_controller_family.py, scripts/m18/run_sapir_whorf_audit.py`
- DAGs: `airflow/dags/m18/lojban_m18_controller_family_dag.py`

## 21. M19

- Kind: `major_series`
- Layer: `manifold_and_return_path`
- Objective: compress tapped hidden state into a learned continuous runway over repeated <symbiote> token positions so the decoder gets multiple normal attention hops before final-answer English resumption.
- Entry count: `14`
- Runnable rows: `14`
- Question boundary: bounded continuous scratchpad runway
- Thesis: compress tapped hidden state into a learned continuous runway over repeated <symbiote> token positions so the decoder gets multiple normal attention hops before final-answer English resumption.
- Selected upstream: `M18`
- Inherits: `two-pass discipline, explicit random and kill controls, bounded intervention locality`
- Reopens: `continuous runway carrier, query bottleneck width, scratchpad runway length`
- Rejects: `global attention biasing as the only coupling surface`
- IDs: `M19, M19.3.A, M19.3.B, M19.3.C, M19.3.D, M19.3.E, M19.3.F, M19.3.R11, M19.3.R12, M19.3.R21, M19.3.R22, M19.31, M19.32, M19.4`
- Aliases: `A, B, C, D, E, F, M19, M19.3.A, M19.3.B, M19.3.C, M19.3.D, M19.3.E, M19.3.F, M19.3.R11, M19.3.R12, M19.3.R21, M19.3.R22, M19.31, M19.32, M19.4, R11, R12, R21, R22`
- Allowed axes: `query count, bottleneck width, scratchpad runway length, random-control comparison, mainline benchmark replication`
- Frozen/forbidden drift: `generic checkpoint renaming after train time, console-only reporting without per-cell json artifacts`
- Promotion basis: `overall_accuracy, lift_vs_en_cot, lift_vs_random`
- Primary metrics: `overall_accuracy, lift_vs_en_cot`
- Guardrail metrics: `lift_vs_random, avg_tokens, audit_qformer_accuracy`
- Historical comparison families: `J, L`
- Required test contracts: `m19.runway_efficiency, m19.zh_branch_control, m19.dynamic_pacing_guardrails, m19.integrity_controls, m19.replication_stability, m19.kill_test_suite, m19.typed_faithfulness, m19.hyperbolic_geometry, m19.bridge_channel_isolation, m19.arity_causal, m19.pointer_counterfactual, m19.order_sensitivity, m18.kill_random_gap, m18.language_tax_compactness, m14.scratchpad_bleed, m11.native_discriminative_oracle, j.accepted_foil_pair_accuracy, l.constraint_scope, m5.chain_serialization, m4.operator_family_consistency, m3.reentry_intervention, m3.reentry_fluency, j.invariance_rate, l.constraint_identity, l.constraint_arity_strict, j.accept_rate_by_depth, j.schema_validity`
- Automatic comparison targets: `M19, M18, M14, M11, M10, M9, M8, M7, M6, M5, M4, M3, M2, M1, J, L, M19.3.A, M19.3.B, M19.31, M19.32`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m19/run_isolation_grid.py, scripts/m19/run_m19_arity_causal_suite.py, scripts/m19/run_m19_bridge_channel_suite.py, scripts/m19/run_m19_godtier_benchmark.py, scripts/m19/run_m19_integrity_suite.py, scripts/m19/run_m19_kill_test_suite.py, scripts/m19/run_m19_mainline_suite.py, scripts/m19/run_m19_order_sensitivity_suite.py, scripts/m19/run_m19_pointer_counterfactual_suite.py, scripts/m19/run_m19_replication_suite.py, scripts/m19/run_m19_stability_microgrid.py`
- DAGs: `airflow/dags/m19/lojban_m19_family_dag.py, airflow/dags/m19/lojban_m19_isolation_grid_dag.py, airflow/dags/m19/lojban_m19_mainline_suite_dag.py`

## 22. M20

- Kind: `major_series`
- Layer: `dictionary_first_substrate`
- Objective: pretrain a factorized predicate dictionary before bridge work so code tokens carry causal properties, stay invariant under entity swaps, and remain silent without grounded brivi arguments.
- Entry count: `7`
- Runnable rows: `7`
- Question boundary: dictionary-first Lojbanic predicate induction
- Thesis: pretrain a factorized predicate dictionary before bridge work so code tokens carry causal properties, stay invariant under entity swaps, and remain silent without grounded brivi arguments.
- IDs: `M20.1, M20.1.A, M20.1.B, M20.1.C, M20.1.D, M20.1.E, M20.1.F`
- Aliases: `A, B, C, D, E, F, M20.1, M20.1.A, M20.1.B, M20.1.C, M20.1.D, M20.1.E, M20.1.F`
- Allowed axes: `dictionary-first pretraining, factorized predicate heads, counterfactual quotient invariance, brivi grounding gate, synthetic world curriculum, soft-to-hard dictionary annealing`
- Frozen/forbidden drift: `M19 runway patching as the main intervention, phrase accuracy as canonical accuracy, entity-name dictionary memorization as a success criterion`
- Promotion basis: `strict_accuracy, lock_pass_rate, predicate_identity_stability, brivi_gate_accuracy`
- Primary metrics: `strict_accuracy, lock_pass_rate, factorized_exact_accuracy, predicate_identity_stability`
- Guardrail metrics: `brivi_gate_accuracy, masked_accuracy, entity_leakage_proxy, accuracy_per_token`
- Historical comparison families: `M19`
- Required test contracts: `m20.dictionary_precedence, m20.factorized_predicate_dictionary, m20.counterfactual_quotient, m20.brivi_lock, m20.synthetic_world_pretraining, m20.soft_dictionary_annealing`
- Automatic comparison targets: `M20, M19, M19.31, M20.1.A, M20.1.B, M20.1.C, M20.1.D, M20.1.E, M20.1.F`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m20/run_m20_dictionary_first_suite.py, scripts/m20/run_m20_lock_suite.py, scripts/m20/run_m20_predicate_induction.py`
- DAGs: `airflow/dags/m20/lojban_m20_dictionary_first_dag.py`

## 23. M21

- Kind: `major_series`
- Layer: `dynamic_bridi_substrate`
- Objective: replace fixed typed slots with variable-length bridi traces so the model learns how many gismu predicates, cmavo modifiers, and judri bindings are needed, then proves those frames are causally useful.
- Entry count: `16`
- Runnable rows: `16`
- Question boundary: dynamic Lojbanic bridi Q-former
- Thesis: replace fixed typed slots with variable-length bridi traces so the model learns how many gismu predicates, cmavo modifiers, and judri bindings are needed, then proves those frames are causally useful.
- Selected upstream: `M20.1.F`
- Inherits: `dictionary-first causal predicate discipline, counterfactual quotient invariance, brivi lock skepticism`
- Reopens: `fixed typed slot count, fixed predicate target set, static dictionary trace shape`
- Rejects: `decorative slot occupancy as semantic evidence, dictionary spread without causal utility`
- IDs: `M21.1, M21.1.A, M21.1.B, M21.1.C, M21.1.D, M21.1.E, M21.1.F, M21.1.G, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L, M21.1.M, M21.1.N, M21.1.O`
- Aliases: `A, B, C, D, E, F, G, H, I, J, K, L, M, M21.1, M21.1.A, M21.1.B, M21.1.C, M21.1.D, M21.1.E, M21.1.F, M21.1.G, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L, M21.1.M, M21.1.N, M21.1.O, N, O`
- Allowed axes: `dynamic frame count, cmavo causality, judri binding causality, judri gated bridge, brivi lock, frame necessity, actual bridge transfer, held-out adversarial prompt audit, adversarial training augmentation, semantic coverage augmentation`
- Frozen/forbidden drift: `fixed typed-slot layout as main result, decorative frame occupancy, phrase accuracy as canonical accuracy, mass deletion of M20 control assets`
- Promotion basis: `strict_accuracy, bridi_trace_exact_accuracy, lock_pass_rate, actual_bridge_transfer_score`
- Primary metrics: `strict_accuracy, bridi_trace_exact_accuracy, gismu_accuracy, judri_binding_accuracy`
- Guardrail metrics: `cmavo_causal_delta, judri_causal_delta, judri_bridge_gate_active_mean, judri_bridge_gate_silenced_predicate_energy_mean, adversarial_strict_accuracy, adversarial_worst_surface_accuracy, semantic_coverage_strict_accuracy, semantic_coverage_worst_surface_accuracy, brivi_lock_violation_rate, accuracy_per_trace_token`
- Historical comparison families: `M19, M20`
- Required test contracts: `m21.dynamic_frame_count, m21.bridi_reconstruction, m21.cmavo_causality, m21.judri_binding, m21.judri_gated_bridge, m21.pointer_necessity, m21.m19_gauntlet_port, m21.frame_necessity, m21.actual_bridge_transfer, m21.adversarial_heldout, m21.adversarial_augmentation, m21.semantic_coverage`
- Automatic comparison targets: `M21, M20, M19, M19.31, M20.1.F, M21.1.A, M21.1.B, M21.1.C, M21.1.D, M21.1.E, M21.1.F, M21.1.G, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m21/run_m21_actual_bridge_suite.py, scripts/m21/run_m21_adversarial_audit.py, scripts/m21/run_m21_dynamic_bridi_suite.py, scripts/m21/run_m21_lock_suite.py, scripts/m21/run_m21_synthetic_assay_suite.py`
- DAGs: `airflow/dags/m21/lojban_m21_dynamic_bridi_dag.py`

## 24. M22

- Kind: `major_series`
- Layer: `semantic_generalization_substrate`
- Objective: keep the M21 dynamic bridi Q-former fixed and test whether expanded semantic surfaces generalize beyond memorized synthetic templates without damaging clean accuracy or judri causality.
- Entry count: `1`
- Runnable rows: `1`
- Question boundary: semantic coverage generalization over the M21 dynamic bridi substrate
- Thesis: keep the M21 dynamic bridi Q-former fixed and test whether expanded semantic surfaces generalize beyond memorized synthetic templates without damaging clean accuracy or judri causality.
- Selected upstream: `M21.1.O`
- Inherits: `dynamic bridi Q-former, judri-gated bridge, pointer necessity evidence, semantic coverage audit harness`
- Reopens: `semantic surface diversity, role-chain generalization, reserved adversarial audit coverage`
- Rejects: `new architecture family before semantic generalization is exhausted, hyperbolic geometry as primary fix`
- IDs: `M22`
- Aliases: `M22`
- Allowed axes: `semantic training surface family, role-chain generalization, polarity reframe templates, reserved adversarial audit surfaces, promotion tolerance against M21 control`
- Frozen/forbidden drift: `new geometry as the main intervention, new bridge architecture family, promoting smoke-only adversarial audits, sacrificing judri causal delta for surface accuracy`
- Promotion basis: `semantic_coverage_strict_accuracy, semantic_coverage_worst_surface_accuracy, m22_semantic_generalization_score, m22_semantic_strict_delta_vs_m21_control`
- Primary metrics: `semantic_coverage_strict_accuracy, semantic_coverage_worst_surface_accuracy, m22_semantic_generalization_score`
- Guardrail metrics: `strict_accuracy, bridi_trace_exact_accuracy, judri_causal_delta, m22_clean_accuracy_drop_vs_m21_control, m22_judri_delta_drop_vs_m21_control, m22_candidate_cell_count, m22_candidate_cells_present`
- Historical comparison families: `M21`
- Required test contracts: `m22.semantic_coverage_generalization`
- Automatic comparison targets: `M22, M21, M20, M21.1.H, M21.1.I, M21.1.J, M21.1.K, M21.1.L, M21.1.M, M21.1.N, M21.1.O`
- Baseline manifest: `docs/baselines/m_series_bridge_baseline_manifest.json`
- Scripts: `scripts/m22/run_m22_semantic_generalization.py`
- DAGs: `airflow/dags/m22/lojban_m22_semantic_generalization_dag.py`

## 25. Control Plane

- Kind: `control_plane`
- Layer: `control_plane`
- Objective: backfill, normalize, aggregate, and render the full ablation program so historical and modern runs live in one auditable surface.
- Entry count: `0`
- Runnable rows: `1`
- Question boundary: program governance and reproducibility
- Thesis: a coherent research program needs one canonical control plane for lineage, aggregation, and orchestration.
- Allowed axes: `history backfill, program map rendering, suite aggregation`
- Frozen/forbidden drift: `silent family creation, implicit lineage jumps`
- Promotion basis: `manifest completeness, path coherence, family coverage`
- Primary metrics: `entry_count, family_count`
- Guardrail metrics: `historical_gap_count`
- Docs: `docs/ABLATION_HISTORY_FULL.md, docs/ABLATION_PROGRAM_MAP.md`
- Scripts: `scripts/control_plane/build_ablation_program_map.py, scripts/control_plane/build_ablation_program_spine.py, scripts/control_plane/run_ablation_history_backfill.py, scripts/m_bridge/run_m_bridge_ablation_test_suite.py`
- DAGs: `airflow/dags/control_plane/lojban_ablation_history_backfill_dag.py, airflow/dags/control_plane/lojban_ablation_program_spine_dag.py, airflow/dags/m_bridge/lojban_m_bridge_ablation_test_suite_dag.py`
