# Full Ablation History

- Source manifest: `artifacts/runs/telemetry/raw/ablation/hypercube/ablation_history_backfill/ablation_history_backfill_m_excavation_20260329/ablation_history_manifest.json`
- Generated from run: `ablation_history_backfill_m_excavation_20260329`
- Total canonical entries: `145`
- Artifact-backed entries: `126`
- Runnable entries: `44`

This document lists every ablation point currently tracked in the unified ledger. Each item includes a brief description, provenance status, and the best-known metric surface when one exists.

## Group Index

- `core_matrix`: `6`
- `control_duel`: `2`
- `h_series`: `8`
- `h5_bridge`: `4`
- `j_series`: `5`
- `l_series`: `21`
- `a_to_g_matrix`: `6`
- `phase5_train_ablation`: `7`
- `phase5_objective_ablation`: `7`
- `m_track`: `78`
- `historical_gap`: `1`

## core_matrix

Earliest documented core controls and challengers before later reruns and telemetry consolidation.

### legacy_core_matrix (6)

#### Babel Expansion

- Canonical ID: `legacy.core.e`
- Aliases: `E, Core/E`
- Lookup aliases: `legacy:E, core:E`
- Brief: test unconstrained vocabulary expansion
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: negative control
- Best-known metrics: held_out_accuracy=`0.167000`

#### Base Model (No Adapter)

- Canonical ID: `legacy.core.a`
- Aliases: `A, Core/A`
- Lookup aliases: `legacy:A, core:A`
- Brief: core baseline control without a symbolic adapter
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: base control
- Best-known metrics: held_out_accuracy=`0.167000`
- Notes: Documented as the foundational control group in the canonical ledger.

#### KV Handoff

- Canonical ID: `legacy.core.c`
- Aliases: `C, Core/C`
- Lookup aliases: `legacy:C, core:C`
- Brief: test direct KV handoff without explicit topology
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: negative bridge ablation
- Best-known metrics: held_out_accuracy=`0.104000`

#### Rigid Symbolic (Phase 5)

- Canonical ID: `legacy.core.b`
- Aliases: `B, Core/B, Run B`
- Lookup aliases: `legacy:B, core:B, run:B`
- Brief: control baseline for two-stage symbolic recovery
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: control baseline
- Best-known metrics: held_out_accuracy=`0.396000`
- Notes: Canonical control baseline before later bridge and reboot series.

#### Self-Correction

- Canonical ID: `legacy.core.f`
- Aliases: `F, Core/F`
- Lookup aliases: `legacy:F, core:F`
- Brief: measure rollback-heavy self-correction as an alternative to rigid topology
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: high-compute ablation
- Best-known metrics: held_out_accuracy=`0.312000`

#### True Coconut

- Canonical ID: `legacy.core.g`
- Aliases: `G, Core/G`
- Lookup aliases: `legacy:G, core:G`
- Brief: test continuous latent prefix without English output until the end
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: negative control
- Best-known metrics: held_out_accuracy=`0.104000`


## control_duel

Direct English-vs-Lojban control comparisons used to test whether medium alone explained performance.

### english_vs_lojban_duel (2)

#### Monolithic English CoT

- Canonical ID: `legacy.duel.english_cot`
- Aliases: `English CoT, Control Duel/English`
- Lookup aliases: `duel:english, control_duel:english`
- Brief: directly compare pure English chain-of-thought against rigid topology
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: language-medium control
- Best-known metrics: held_out_accuracy=`0.000000`

#### Rigid Lojban Topology

- Canonical ID: `legacy.duel.lojban_topology`
- Aliases: `Lojban Dual, Control Duel/Lojban`
- Lookup aliases: `duel:lojban, control_duel:lojban`
- Brief: directly compare rigid topology against English chain-of-thought
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: language-medium challenger
- Best-known metrics: held_out_accuracy=`0.417000`


## h_series

Mid-layer bridge and related H-family latent handoff experiments.

### h_series_artifact (4)

#### SwiGLU Mid-Layer Bridge (Non-Linear Alignment)

- Canonical ID: `h.series.h3`
- Aliases: `H3`
- Lookup aliases: `h:H3`
- Brief: artifact-backed H/H5 extension ablation or evaluation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: h-series family row
- Best-known metrics: held_out_accuracy=`0.000000`, final_answer_lift=`-0.166667`, geometry_retention=`0.934098`
- Scripts: `D:/lojbanhypop/scripts/true_coconut.py`

#### Dynamic Pointer Refactor Eval

- Canonical ID: `h.series.h5_dptr`
- Aliases: `H5-DPTR`
- Lookup aliases: `h:H5-DPTR`
- Brief: artifact-backed H/H5 extension ablation or evaluation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: h-series family row
- Best-known metrics: held_out_accuracy=`0.250000`
- Scripts: `D:/lojbanhypop/scripts/eval_h5_dynamic_pointer_refactor.py`

#### OOD Stress Test

- Canonical ID: `h.series.h5_ood`
- Aliases: `H5-OOD`
- Lookup aliases: `h:H5-OOD`
- Brief: artifact-backed H/H5 extension ablation or evaluation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: h-series family row
- Best-known metrics: held_out_accuracy=`0.750000`
- Scripts: `D:/lojbanhypop/scripts/eval_h5_ood_stress.py`

#### Provenance Trace

- Canonical ID: `h.series.h5_prov`
- Aliases: `H5-PROV`
- Lookup aliases: `h:H5-PROV`
- Brief: artifact-backed H/H5 extension ablation or evaluation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: h-series family row
- Best-known metrics: geometry_retention=`0.002500`
- Scripts: `D:/lojbanhypop/scripts/trace_h5_provenance.py`

### legacy_h_series (4)

#### Deep Linear Bridge

- Canonical ID: `legacy.h.h2`
- Aliases: `H2`
- Lookup aliases: `legacy:H2`
- Brief: test deeper linear mid-layer bridge for geometry retention
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: mid-layer bridge ablation
- Best-known metrics: held_out_accuracy=`0.042000`, geometry_retention=`0.923000`

#### Deep SwiGLU Bridge

- Canonical ID: `legacy.h.h4`
- Aliases: `H4`
- Lookup aliases: `legacy:H4`
- Brief: test deeper non-linear mid-layer bridge with SwiGLU
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: mid-layer bridge ablation
- Best-known metrics: held_out_accuracy=`0.083000`, geometry_retention=`0.925000`

#### Linear Bridge

- Canonical ID: `legacy.h.h1`
- Aliases: `H1`
- Lookup aliases: `legacy:H1`
- Brief: inject learned topological latent space into middle English layers via linear bridge
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: mid-layer bridge ablation
- Best-known metrics: held_out_accuracy=`0.000000`, geometry_retention=`0.456000`

#### SwiGLU Non-Linear Bridge

- Canonical ID: `legacy.h.h3`
- Aliases: `H3`
- Lookup aliases: `legacy:H3`
- Brief: test non-linear mid-layer bridge with SwiGLU expansion
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: mid-layer bridge ablation
- Best-known metrics: held_out_accuracy=`0.000000`, geometry_retention=`0.934000`


## h5_bridge

Late H5 gearbox, boolean surgery, and forced-manifold bridge rows.

### legacy_h5_bridge (4)

#### H5.2a Gearbox Control

- Canonical ID: `legacy.h5.h5_2a`
- Aliases: `H5.2a, Gearbox Control`
- Lookup aliases: `legacy:H5.2a, h5:2a`
- Brief: monotonic pointer gearbox bridge without boolean surgery
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: slice-2 control
- Best-known metrics: held_out_accuracy=`0.375000`, logical_accuracy=`0.375000`, final_ce_loss=`13.240000`

#### H5.2b True Neuro-Symbolic Bridge

- Canonical ID: `legacy.h5.h5_2b`
- Aliases: `H5.2b, True Neuro-Symbolic`
- Lookup aliases: `legacy:H5.2b, h5:2b`
- Brief: slice-2 bridge with log-space boolean surgery enabled
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: slice-2 bridge intervention
- Best-known metrics: held_out_accuracy=`0.375000`, logical_accuracy=`0.375000`, final_ce_loss=`13.240000`, surgery_trigger_rate=`0.000000`

#### H5.4 Forced Boolean Manifold

- Canonical ID: `legacy.h5.h5_4`
- Aliases: `H5.4, Iron Collar`
- Lookup aliases: `legacy:H5.4, h5:4`
- Brief: force strict arity mask and boolean slot typing
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: forced manifold intervention
- Best-known metrics: held_out_accuracy=`0.000000`, logical_accuracy=`0.000000`, surgery_trigger_rate=`1.000000`

#### H5.5 Grounded Fine-Tune

- Canonical ID: `legacy.h5.h5_5`
- Aliases: `H5.5`
- Lookup aliases: `legacy:H5.5, h5:5`
- Brief: recover semantic decoding while preserving the iron collar manifold
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`doc_only`
- Baseline relation: semantic recovery
- Best-known metrics: held_out_accuracy=`1.000000`, logical_accuracy=`1.000000`


## j_series

Data invariance, adversarial synthesis, and acceptance-diagnostic series.

### j_series (5)

#### Graph Target (Factor Schema)

- Canonical ID: `j.series.j_1`
- Normalized ID: `M1.1`
- Taxonomy: major=`1` minor=`1` cell=`None`
- Aliases: `J-1, M1.1`
- Lookup aliases: `j:J-1, M1.1`
- Brief: J-series invariance, curriculum, or adversarial synthesis ablation/eval
- Architectural question: `M1`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: j-series row
- Best-known metrics: surgery_trigger_rate=`1.000000`
- Scripts: `D:/lojbanhypop/scripts/eval_j_1.py`

#### Operator Curriculum Build

- Canonical ID: `j.series.j_4`
- Normalized ID: `M1.4`
- Taxonomy: major=`1` minor=`4` cell=`None`
- Aliases: `J-4, M1.4`
- Lookup aliases: `j:J-4, M1.4`
- Brief: J-series invariance, curriculum, or adversarial synthesis ablation/eval
- Architectural question: `M1`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: j-series row
- Scripts: `D:/lojbanhypop/scripts/eval_j_4.py`

#### Paraphrase Explosion (Invariance)

- Canonical ID: `j.series.j_2`
- Normalized ID: `M1.2`
- Taxonomy: major=`1` minor=`2` cell=`None`
- Aliases: `J-2, M1.2`
- Lookup aliases: `j:J-2, M1.2`
- Brief: J-series invariance, curriculum, or adversarial synthesis ablation/eval
- Architectural question: `M1`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: j-series row
- Scripts: `D:/lojbanhypop/scripts/eval_j_2.py`

#### Stop-Grad Isolation Gate

- Canonical ID: `j.series.j_3`
- Normalized ID: `M1.3`
- Taxonomy: major=`1` minor=`3` cell=`None`
- Aliases: `J-3, M1.3`
- Lookup aliases: `j:J-3, M1.3`
- Brief: J-series invariance, curriculum, or adversarial synthesis ablation/eval
- Architectural question: `M1`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: j-series row
- Best-known metrics: surgery_trigger_rate=`1.000000`
- Scripts: `D:/lojbanhypop/scripts/eval_j_3.py, D:/lojbanhypop/scripts/train_h5_persistent_vq_advisor.py`

#### Adversarial Synthesis (Scope/Foil)

- Canonical ID: `j.series.j_5`
- Normalized ID: `M1.5`
- Taxonomy: major=`1` minor=`5` cell=`None`
- Aliases: `J-5, M1.5`
- Lookup aliases: `j:J-5, M1.5`
- Brief: J-series invariance, curriculum, or adversarial synthesis ablation/eval
- Architectural question: `M1`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: j-series row
- Best-known metrics: surgery_trigger_rate=`1.000000`
- Scripts: `D:/lojbanhypop/scripts/eval_j_5.py`


## l_series

Constraint-optimized Lagrangian training lineage, including the L-series charter, L6 branch, and L-rooted M3+/M4/M5 branch work.

### l_series_branch_telemetry (17)

#### Baseline Gate Run (L6-C + curriculum)

- Canonical ID: `l.branch.m3.m3_0`
- Normalized ID: `M3.0`
- Taxonomy: major=`3` minor=`0` cell=`None`
- Aliases: `M3.0`
- Lookup aliases: `M3+:M3.0`
- Brief: telemetry-rooted M3+ family row
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3+ row

#### Binding Bootcamp (unbound focus)

- Canonical ID: `l.branch.m3.m3_1`
- Normalized ID: `M3.1`
- Taxonomy: major=`3` minor=`1` cell=`None`
- Aliases: `M3.1`
- Lookup aliases: `M3+:M3.1`
- Brief: telemetry-rooted M3+ family row
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3+ row

#### Compression Activation (tier C pressure)

- Canonical ID: `l.branch.m3.m3_4`
- Normalized ID: `M3.4`
- Taxonomy: major=`3` minor=`4` cell=`None`
- Aliases: `M3.4`
- Lookup aliases: `M3+:M3.4`
- Brief: telemetry-rooted M3+ family row
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3+ row

#### Depth Ramp (1->4 curriculum pressure)

- Canonical ID: `l.branch.m3.m3_2`
- Normalized ID: `M3.2`
- Taxonomy: major=`3` minor=`2` cell=`None`
- Aliases: `M3.2`
- Lookup aliases: `M3+:M3.2`
- Brief: telemetry-rooted M3+ family row
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3+ row

#### Truth Discrimination (foil-sensitive tier B)

- Canonical ID: `l.branch.m3.m3_3`
- Normalized ID: `M3.3`
- Taxonomy: major=`3` minor=`3` cell=`None`
- Aliases: `M3.3`
- Lookup aliases: `M3+:M3.3`
- Brief: telemetry-rooted M3+ family row
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3+ row

#### Ablated anchor (swap-test disabled)

- Canonical ID: `l.branch.m3_5.m3_5_c`
- Aliases: `M3.5.C`
- Lookup aliases: `M3.5:M3.5.C`
- Brief: telemetry-rooted M3.5 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.5 row

#### Asymmetric Baseline (forced non-commutativity)

- Canonical ID: `l.branch.m3_5.m3_5_a`
- Aliases: `M3.5.A`
- Lookup aliases: `M3.5:M3.5.A`
- Brief: telemetry-rooted M3.5 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.5 row

#### Symmetry-aware crucible (invariant vs foil)

- Canonical ID: `l.branch.m3_5.m3_5_b`
- Aliases: `M3.5.B`
- Lookup aliases: `M3.5:M3.5.B`
- Brief: telemetry-rooted M3.5 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.5 row

#### M3.6 M3.6.A

- Canonical ID: `l.branch.m3_6.m3_6_a`
- Aliases: `M3.6.M3.6.A, M3.6.A`
- Lookup aliases: `M3.6.M3.6.A`
- Brief: telemetry-rooted M3.6 ablation cell
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.6 cell

#### M3.6 M3.6.B

- Canonical ID: `l.branch.m3_6.m3_6_b`
- Aliases: `M3.6.M3.6.B, M3.6.B`
- Lookup aliases: `M3.6.M3.6.B`
- Brief: telemetry-rooted M3.6 ablation cell
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.6 cell

#### M3.6 M3.6.C

- Canonical ID: `l.branch.m3_6.m3_6_c`
- Aliases: `M3.6.M3.6.C, M3.6.C`
- Lookup aliases: `M3.6.M3.6.C`
- Brief: telemetry-rooted M3.6 ablation cell
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.6 cell

#### Control (no shadow pressure)

- Canonical ID: `l.branch.m3_7.m3_7_a`
- Aliases: `M3.7.A`
- Lookup aliases: `M3.7:M3.7.A`
- Brief: telemetry-rooted M3.7 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.7 row

#### Family separation + rolling shadow

- Canonical ID: `l.branch.m3_7.m3_7_c`
- Aliases: `M3.7.C`
- Lookup aliases: `M3.7:M3.7.C`
- Brief: telemetry-rooted M3.7 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.7 row

#### Paraphrase shadow alignment

- Canonical ID: `l.branch.m3_7.m3_7_b`
- Aliases: `M3.7.B`
- Lookup aliases: `M3.7:M3.7.B`
- Brief: telemetry-rooted M3.7 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.7 row

#### Domain reuse reward

- Canonical ID: `l.branch.m3_8.m3_8_b`
- Aliases: `M3.8.B`
- Lookup aliases: `M3.8:M3.8.B`
- Brief: telemetry-rooted M3.8 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.8 row

#### Family clustering objective

- Canonical ID: `l.branch.m3_8.m3_8_c`
- Aliases: `M3.8.C`
- Lookup aliases: `M3.8:M3.8.C`
- Brief: telemetry-rooted M3.8 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.8 row

#### Operator entropy regularization

- Canonical ID: `l.branch.m3_8.m3_8_a`
- Aliases: `M3.8.A`
- Lookup aliases: `M3.8:M3.8.A`
- Brief: telemetry-rooted M3.8 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.8 row

### l_series_charter (1)

#### L-Series Lexicographic Augmented Lagrangian Controller

- Canonical ID: `l.series.charter`
- Aliases: `L-Series, Lagrangian Series`
- Lookup aliases: `l_series, L-series`
- Brief: replace static weighted-loss blending with lexicographic augmented Lagrangian control over Tier A/B/C constraints
- Provenance: evidence=`doc_reported` confidence=`low` reproducibility=`doc_only`
- Baseline relation: series charter
- Scripts: `scripts/train_l_series_mvs.py`
- DAGs: `airflow/dags/lojban_l_series_dag.py`
- Notes: Documented in docs/L_SERIES.md as the main lexicographic constraint-control phase. | Serves as the umbrella lineage for L6 and the early M3+/M4/M5 branch experiments rooted in runs/l_series.

### l_series_l6 (3)

#### Scope Drill + TierB Force + Soft Constraint Identity Audit

- Canonical ID: `l.series.l6.l6_a`
- Normalized ID: `M2.1`
- Taxonomy: major=`2` minor=`1` cell=`None`
- Aliases: `L6-A, M2.A, M2.1`
- Lookup aliases: `l6:L6-A, M2.A, M2.1`
- Brief: L6 branch ablation over scope drills and Tier B/C forcing
- Architectural question: `M2`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: l6 branch cell
- Scripts: `scripts/run_l6_ablation_branch.py, scripts/train_l_series_mvs.py`

#### Scope Drill Only

- Canonical ID: `l.series.l6.l6_b`
- Normalized ID: `M2.2`
- Taxonomy: major=`2` minor=`2` cell=`None`
- Aliases: `L6-B, M2.B, M2.2`
- Lookup aliases: `l6:L6-B, M2.B, M2.2`
- Brief: L6 branch ablation over scope drills and Tier B/C forcing
- Architectural question: `M2`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: l6 branch cell
- Scripts: `scripts/run_l6_ablation_branch.py, scripts/train_l_series_mvs.py`

#### TierB Force + Soft Constraint Audit Only

- Canonical ID: `l.series.l6.l6_c`
- Normalized ID: `M2.3`
- Taxonomy: major=`2` minor=`3` cell=`None`
- Aliases: `L6-C, M2.C, M2.3`
- Lookup aliases: `l6:L6-C, M2.C, M2.3`
- Brief: L6 branch ablation over scope drills and Tier B/C forcing
- Architectural question: `M2`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: l6 branch cell
- Scripts: `scripts/run_l6_ablation_branch.py, scripts/train_l_series_mvs.py`


## a_to_g_matrix

Artifact-backed reruns of the legacy A-G matrix under the modern telemetry stack.

### a_to_g_matrix (6)

#### Babel Bridge (Projected latent handoff)

- Canonical ID: `a_to_g.e`
- Aliases: `E, A-G/E`
- Lookup aliases: `a_to_g:E`
- Brief: artifact-backed rerun of the A-G coconut ablation matrix
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`doc_only`
- Baseline relation: a-to-g matrix cell
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`

#### Coconut Fusion (Latent KV Handoff)

- Canonical ID: `a_to_g.c`
- Aliases: `C, A-G/C`
- Lookup aliases: `a_to_g:C`
- Brief: artifact-backed rerun of the A-G coconut ablation matrix
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: a-to-g matrix cell
- Best-known metrics: held_out_accuracy=`0.000000`, logical_accuracy=`0.041667`, final_answer_lift=`-0.166667`, symbolic_lift=`0.041667`
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`

#### Control (English CoT -> English)

- Canonical ID: `a_to_g.a`
- Aliases: `A, A-G/A`
- Lookup aliases: `a_to_g:A`
- Brief: artifact-backed rerun of the A-G coconut ablation matrix
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: a-to-g matrix cell
- Best-known metrics: held_out_accuracy=`0.166667`, logical_accuracy=`0.000000`
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`

#### Enhanced Constraint Text-to-Text (No Handoff)

- Canonical ID: `a_to_g.b_2`
- Aliases: `B.2, A-G/B.2`
- Lookup aliases: `a_to_g:B.2`
- Brief: artifact-backed rerun of the A-G coconut ablation matrix
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: a-to-g matrix cell
- Best-known metrics: held_out_accuracy=`0.000000`, logical_accuracy=`0.395833`, final_answer_lift=`-0.166667`, symbolic_lift=`0.395833`
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`

#### Legacy Text-to-Text (No Handoff)

- Canonical ID: `a_to_g.b_1`
- Aliases: `B.1, A-G/B.1`
- Lookup aliases: `a_to_g:B.1`
- Brief: artifact-backed rerun of the A-G coconut ablation matrix
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: a-to-g matrix cell
- Best-known metrics: held_out_accuracy=`0.437500`, logical_accuracy=`0.395833`, final_answer_lift=`0.270833`, symbolic_lift=`0.395833`
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`

#### NoPE Fusion (DroPE + latent handoff)

- Canonical ID: `a_to_g.d`
- Aliases: `D, A-G/D`
- Lookup aliases: `a_to_g:D`
- Brief: artifact-backed rerun of the A-G coconut ablation matrix
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: a-to-g matrix cell
- Best-known metrics: held_out_accuracy=`0.000000`, final_answer_lift=`0.000000`
- Scripts: `scripts/run_coconut_ablation_matrix.py`
- DAGs: `airflow/dags/lojban_ablation_matrix_dag.py`


## phase5_train_ablation

Training-stack ablations along the phase-5 path.

### phase5_train_ablation (7)

#### ablate_compositional_consistency_weight

- Canonical ID: `phase5.train.ablate_compositional_consistency_weight`
- Aliases: `ablate_compositional_consistency_weight`
- Lookup aliases: `phase5.train:ablate_compositional_consistency_weight`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`

#### ablate_compression_regularization_weight

- Canonical ID: `phase5.train.ablate_compression_regularization_weight`
- Aliases: `ablate_compression_regularization_weight`
- Lookup aliases: `phase5.train:ablate_compression_regularization_weight`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`

#### ablate_coverage_regularization_weight

- Canonical ID: `phase5.train.ablate_coverage_regularization_weight`
- Aliases: `ablate_coverage_regularization_weight`
- Lookup aliases: `phase5.train:ablate_coverage_regularization_weight`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`

#### ablate_roundtrip_consistency_weight

- Canonical ID: `phase5.train.ablate_roundtrip_consistency_weight`
- Aliases: `ablate_roundtrip_consistency_weight`
- Lookup aliases: `phase5.train:ablate_roundtrip_consistency_weight`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`

#### ablate_semantic_unambiguity_weight

- Canonical ID: `phase5.train.ablate_semantic_unambiguity_weight`
- Aliases: `ablate_semantic_unambiguity_weight`
- Lookup aliases: `phase5.train:ablate_semantic_unambiguity_weight`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`

#### baseline_no_phase5

- Canonical ID: `phase5.train.baseline_no_phase5`
- Aliases: `baseline_no_phase5`
- Lookup aliases: `phase5.train:baseline_no_phase5`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`

#### phase5_full

- Canonical ID: `phase5.train.phase5_full`
- Aliases: `phase5_full`
- Lookup aliases: `phase5.train:phase5_full`
- Brief: phase-5 training objective weight ablation
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: phase5 train ablation variant
- Scripts: `scripts/run_phase5_train_ablation.py`


## phase5_objective_ablation

Objective and loss-surface ablations along the phase-5 path.

### phase5_objective_ablation (7)

#### ablate_compositional_consistency_loss

- Canonical ID: `phase5.objective.ablate_compositional_consistency_loss`
- Aliases: `ablate_compositional_consistency_loss`
- Lookup aliases: `phase5.objective:ablate_compositional_consistency_loss`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`

#### ablate_compression_regularization_loss

- Canonical ID: `phase5.objective.ablate_compression_regularization_loss`
- Aliases: `ablate_compression_regularization_loss`
- Lookup aliases: `phase5.objective:ablate_compression_regularization_loss`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`

#### ablate_coverage_regularization_loss

- Canonical ID: `phase5.objective.ablate_coverage_regularization_loss`
- Aliases: `ablate_coverage_regularization_loss`
- Lookup aliases: `phase5.objective:ablate_coverage_regularization_loss`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`

#### ablate_roundtrip_consistency_loss

- Canonical ID: `phase5.objective.ablate_roundtrip_consistency_loss`
- Aliases: `ablate_roundtrip_consistency_loss`
- Lookup aliases: `phase5.objective:ablate_roundtrip_consistency_loss`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`

#### ablate_semantic_unambiguity_loss

- Canonical ID: `phase5.objective.ablate_semantic_unambiguity_loss`
- Aliases: `ablate_semantic_unambiguity_loss`
- Lookup aliases: `phase5.objective:ablate_semantic_unambiguity_loss`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`

#### baseline_no_phase5

- Canonical ID: `phase5.objective.baseline_no_phase5`
- Aliases: `baseline_no_phase5`
- Lookup aliases: `phase5.objective:baseline_no_phase5`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`

#### phase5_full

- Canonical ID: `phase5.objective.phase5_full`
- Aliases: `phase5_full`
- Lookup aliases: `phase5.objective:phase5_full`
- Brief: differentiate the contribution of individual phase-5 regularizer terms
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: objective-term ablation
- Scripts: `scripts/run_phase5_objective_ablation.py`


## m_track

Modern telemetry-rooted M-series work that is not historically rooted in L-series.

### m_archive_results (13)

#### M10 publication metrics

- Canonical ID: `m.track.m10_3`
- Normalized ID: `M10.3`
- Taxonomy: major=`10` minor=`3` cell=`None`
- Aliases: `M10.3, M10.publication, final_publication_metrics`
- Lookup aliases: `M10.3, M10.publication, m10:publication`
- Brief: publication-facing summary metrics for the late M10 line
- Architectural question: `M10`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m10 publication row
- Scripts: `scripts/m10/final_audit.py`

#### M6.6 directed AST final

- Canonical ID: `m.track.m6_6`
- Normalized ID: `M6.6`
- Taxonomy: major=`6` minor=`6` cell=`None`
- Aliases: `M6.6, RESULTS_M6_6_DIRECTED_AST_FINAL`
- Lookup aliases: `M6.6, m6:6`
- Brief: late M6 directed-AST expansion branch
- Architectural question: `M6`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m6 expansive variant
- Best-known metrics: held_out_accuracy=`0.300000`
- Scripts: `scripts/train_m6_logic_engine.py, scripts/eval_m6_logic_engine.py`

#### M6 severed bridge

- Canonical ID: `m.track.m6_0`
- Normalized ID: `M6.0`
- Taxonomy: major=`6` minor=`0` cell=`None`
- Aliases: `M6.0, RESULTS_M6_SEVERED_BRIDGE_20260314`
- Lookup aliases: `M6.0, m6:severed_bridge`
- Brief: direct logic-engine bridge baseline before later alignment and scratchpad variants
- Architectural question: `M6`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m6 baseline
- Best-known metrics: held_out_accuracy=`0.700000`
- Scripts: `scripts/train_m6_logic_engine.py, scripts/eval_m6_logic_engine.py`

#### M6.1 alignment 70acc

- Canonical ID: `m.track.m6_1`
- Normalized ID: `M6.1`
- Taxonomy: major=`6` minor=`1` cell=`None`
- Aliases: `M6.1, RESULTS_M6_1_ALIGNMENT_70ACC`
- Lookup aliases: `M6.1, m6:1`
- Brief: alignment-focused M6 variant with improved held-out accuracy
- Architectural question: `M6`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m6 aligned variant
- Best-known metrics: held_out_accuracy=`0.700000`
- Scripts: `scripts/train_m6_logic_engine.py, scripts/eval_m6_logic_engine.py`

#### M6.2 aligned 30acc

- Canonical ID: `m.track.m6_2`
- Normalized ID: `M6.2`
- Taxonomy: major=`6` minor=`2` cell=`None`
- Aliases: `M6.2, RESULTS_M6_2_ALIGNED_30ACC`
- Lookup aliases: `M6.2, m6:2`
- Brief: recalibrated M6 alignment follow-up with lower headline accuracy
- Architectural question: `M6`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m6 aligned variant
- Best-known metrics: held_out_accuracy=`0.300000`
- Scripts: `scripts/train_m6_logic_engine.py, scripts/eval_m6_logic_engine.py`

#### M6.3 scratchpad 35acc

- Canonical ID: `m.track.m6_3`
- Normalized ID: `M6.3`
- Taxonomy: major=`6` minor=`3` cell=`None`
- Aliases: `M6.3, RESULTS_M6_3_SCRATCHPAD_35ACC`
- Lookup aliases: `M6.3, m6:3`
- Brief: scratchpad-directed M6 branch before later dedicated scratchpad re-entry families
- Architectural question: `M6`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m6 scratchpad variant
- Best-known metrics: held_out_accuracy=`0.300000`
- Scripts: `scripts/train_m6_logic_engine.py, scripts/eval_m6_logic_engine.py`

#### M7 interleaved coprocessor

- Canonical ID: `m.track.m7_0`
- Normalized ID: `M7.0`
- Taxonomy: major=`7` minor=`0` cell=`None`
- Aliases: `M7.0, M7, RESULTS_M7_INTERLEAVED_COPROCESSOR`
- Lookup aliases: `M7.0, M7, m7:interleaved`
- Brief: interleaved coprocessor rollout after the M6 bridge family
- Architectural question: `M7`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m7 family row
- Best-known metrics: held_out_accuracy=`0.750000`
- Scripts: `scripts/train_m7_interleaved.py, scripts/eval_m7_interleaved.py`

#### M8 council of oracles

- Canonical ID: `m.track.m8_0`
- Normalized ID: `M8.0`
- Taxonomy: major=`8` minor=`0` cell=`None`
- Aliases: `M8.0, M8, RESULTS_M8_COUNCIL_OF_ORACLES`
- Lookup aliases: `M8.0, M8, m8:council`
- Brief: council-of-oracles composition family
- Architectural question: `M8`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m8 family row
- Best-known metrics: held_out_accuracy=`0.550000`
- Scripts: `scripts/train_m8_council.py, scripts/eval_m8_council.py`

#### M9 duel hypercube

- Canonical ID: `m.track.m9_1`
- Normalized ID: `M9.1`
- Taxonomy: major=`9` minor=`1` cell=`None`
- Aliases: `M9.1, M9.hypercube, RESULTS_M9_HYPERCUBE`
- Lookup aliases: `M9.1, M9.hypercube, m9:hypercube`
- Brief: hypercube duel comparison inside the M9 manifold regime
- Architectural question: `M9`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m9 hypercube row
- Scripts: `scripts/m9/eval_m9.py`

#### M9 provenance manifold audit

- Canonical ID: `m.track.m9_0`
- Normalized ID: `M9.0`
- Taxonomy: major=`9` minor=`0` cell=`None`
- Aliases: `M9.0, M9.audit, RESULTS_M9_AUDIT`
- Lookup aliases: `M9.0, M9.audit, m9:audit`
- Brief: audit the provenance-manifold system after phase synchronization
- Architectural question: `M9`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m9 audit row
- Best-known metrics: held_out_accuracy=`0.000000`
- Scripts: `scripts/m9/eval_m9.py`

#### M10 floor lock audit

- Canonical ID: `m.track.m10_2`
- Normalized ID: `M10.2`
- Taxonomy: major=`10` minor=`2` cell=`None`
- Aliases: `M10.2, M10.floor_lock, final_floor_lock`
- Lookup aliases: `M10.2, M10.floor_lock, m10:floor_lock`
- Brief: larger-sample M10 floor-lock evaluation
- Architectural question: `M10`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m10 floor-lock row
- Best-known metrics: held_out_accuracy=`0.780000`, macro_f1=`0.427924`
- Scripts: `scripts/m10/final_audit.py`

#### M10 final bridge audit

- Canonical ID: `m.track.m10_1`
- Normalized ID: `M10.1`
- Taxonomy: major=`10` minor=`1` cell=`None`
- Aliases: `M10.1, M10.final_bridge, final_bridge_audit`
- Lookup aliases: `M10.1, M10.final_bridge, m10:final_bridge`
- Brief: bridge-side final audit on the M10 translator/head stack
- Architectural question: `M10`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m10 final bridge row
- Best-known metrics: held_out_accuracy=`0.833333`, macro_f1=`0.418315`
- Scripts: `scripts/m10/final_audit.py`

#### M10 audit

- Canonical ID: `m.track.m10_0`
- Normalized ID: `M10.0`
- Taxonomy: major=`10` minor=`0` cell=`None`
- Aliases: `M10.0, M10.audit, RESULTS_M10_AUDIT`
- Lookup aliases: `M10.0, M10.audit, m10:audit`
- Brief: translator and English-head audit before final M11-native discriminative redirection
- Architectural question: `M10`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: m10 audit row
- Best-known metrics: held_out_accuracy=`0.600000`
- Scripts: `scripts/m10/final_audit.py`

### m_telemetry (65)

#### hypercube_ablation telemetry report

- Canonical ID: `m.track.hypercube_ablation`
- Aliases: `hypercube_ablation`
- Lookup aliases: `hypercube_ablation`
- Brief: telemetry-rooted hypercube_ablation family report
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: hypercube_ablation family report

#### Baseline Gate Run (L6-C + curriculum)

- Canonical ID: `m.track.m4.m3_0`
- Aliases: `M3.0`
- Lookup aliases: `M4:M3.0`
- Brief: telemetry-rooted M4 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M4 row

#### Binding Bootcamp (unbound focus)

- Canonical ID: `m.track.m4.m3_1`
- Aliases: `M3.1`
- Lookup aliases: `M4:M3.1`
- Brief: telemetry-rooted M4 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M4 row

#### Compression Activation (tier C pressure)

- Canonical ID: `m.track.m4.m3_4`
- Aliases: `M3.4`
- Lookup aliases: `M4:M3.4`
- Brief: telemetry-rooted M4 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M4 row

#### Depth Ramp (1->4 curriculum pressure)

- Canonical ID: `m.track.m4.m3_2`
- Aliases: `M3.2`
- Lookup aliases: `M4:M3.2`
- Brief: telemetry-rooted M4 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M4 row

#### Truth Discrimination (foil-sensitive tier B)

- Canonical ID: `m.track.m4.m3_3`
- Aliases: `M3.3`
- Lookup aliases: `M4:M3.3`
- Brief: telemetry-rooted M4 family row
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M4 row

#### M3.9 telemetry report

- Canonical ID: `m.track.m3_9`
- Normalized ID: `M3.9`
- Taxonomy: major=`3` minor=`9` cell=`None`
- Aliases: `M3.9`
- Lookup aliases: `M3.9`
- Brief: telemetry-rooted M3.9 family report
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.9 family report
- Scripts: `scripts/run_m3_9_primitive_probe.py`

#### M3.10 telemetry report

- Canonical ID: `m.track.m3_10`
- Normalized ID: `M3.10`
- Taxonomy: major=`3` minor=`10` cell=`None`
- Aliases: `M3.10`
- Lookup aliases: `M3.10`
- Brief: telemetry-rooted M3.10 family report
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.10 family report
- Scripts: `scripts/run_m3_10_ood_accuracy_probe.py`

#### M3.11 telemetry report

- Canonical ID: `m.track.m3_11`
- Normalized ID: `M3.11`
- Taxonomy: major=`3` minor=`11` cell=`None`
- Aliases: `M3.11`
- Lookup aliases: `M3.11`
- Brief: telemetry-rooted M3.11 family report
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.11 family report
- Scripts: `scripts/run_m3_11_winograd_failure_anatomy.py`

#### M3.14 A

- Canonical ID: `m.track.m3_14.a`
- Normalized ID: `M3.14.A`
- Taxonomy: major=`3` minor=`14` cell=`A`
- Aliases: `M3.14.A, A`
- Lookup aliases: `M3.14.A`
- Brief: telemetry-rooted M3.14 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.14 cell

#### M3.14 B

- Canonical ID: `m.track.m3_14.b`
- Normalized ID: `M3.14.B`
- Taxonomy: major=`3` minor=`14` cell=`B`
- Aliases: `M3.14.B, B`
- Lookup aliases: `M3.14.B`
- Brief: telemetry-rooted M3.14 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.14 cell

#### M3.14 C

- Canonical ID: `m.track.m3_14.c`
- Normalized ID: `M3.14.C`
- Taxonomy: major=`3` minor=`14` cell=`C`
- Aliases: `M3.14.C, C`
- Lookup aliases: `M3.14.C`
- Brief: telemetry-rooted M3.14 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.14 cell

#### M3.15 A

- Canonical ID: `m.track.m3_15.a`
- Normalized ID: `M3.15.A`
- Taxonomy: major=`3` minor=`15` cell=`A`
- Aliases: `M3.15.A, A`
- Lookup aliases: `M3.15.A`
- Brief: telemetry-rooted M3.15 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15 B

- Canonical ID: `m.track.m3_15.b`
- Normalized ID: `M3.15.B`
- Taxonomy: major=`3` minor=`15` cell=`B`
- Aliases: `M3.15.B, B`
- Lookup aliases: `M3.15.B`
- Brief: telemetry-rooted M3.15 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.001251`

#### M3.15 C

- Canonical ID: `m.track.m3_15.c`
- Normalized ID: `M3.15.C`
- Taxonomy: major=`3` minor=`15` cell=`C`
- Aliases: `M3.15.C, C`
- Lookup aliases: `M3.15.C`
- Brief: telemetry-rooted M3.15 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.001251`

#### M3.12 A

- Canonical ID: `m.track.m3_12.a`
- Normalized ID: `M3.12.A`
- Taxonomy: major=`3` minor=`12` cell=`A`
- Aliases: `M3.12.A, A`
- Lookup aliases: `M3.12.A`
- Brief: telemetry-rooted M3.12 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.12 cell

#### M3.12 B

- Canonical ID: `m.track.m3_12.b`
- Normalized ID: `M3.12.B`
- Taxonomy: major=`3` minor=`12` cell=`B`
- Aliases: `M3.12.B, B`
- Lookup aliases: `M3.12.B`
- Brief: telemetry-rooted M3.12 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.12 cell

#### M3.12 C

- Canonical ID: `m.track.m3_12.c`
- Normalized ID: `M3.12.C`
- Taxonomy: major=`3` minor=`12` cell=`C`
- Aliases: `M3.12.C, C`
- Lookup aliases: `M3.12.C`
- Brief: telemetry-rooted M3.12 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.12 cell

#### M3.15b A

- Canonical ID: `m.track.m3_15b.a`
- Normalized ID: `M3.15b.A`
- Taxonomy: major=`3` minor=`15b` cell=`A`
- Aliases: `M3.15b.A, A`
- Lookup aliases: `M3.15b.A`
- Brief: telemetry-rooted M3.15b ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15b cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15b B

- Canonical ID: `m.track.m3_15b.b`
- Normalized ID: `M3.15b.B`
- Taxonomy: major=`3` minor=`15b` cell=`B`
- Aliases: `M3.15b.B, B`
- Lookup aliases: `M3.15b.B`
- Brief: telemetry-rooted M3.15b ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15b cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.003670`

#### M3.15b C

- Canonical ID: `m.track.m3_15b.c`
- Normalized ID: `M3.15b.C`
- Taxonomy: major=`3` minor=`15b` cell=`C`
- Aliases: `M3.15b.C, C`
- Lookup aliases: `M3.15b.C`
- Brief: telemetry-rooted M3.15b ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15b cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15c A

- Canonical ID: `m.track.m3_15c.a`
- Normalized ID: `M3.15c.A`
- Taxonomy: major=`3` minor=`15c` cell=`A`
- Aliases: `M3.15c.A, A`
- Lookup aliases: `M3.15c.A`
- Brief: telemetry-rooted M3.15c ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15c cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15c B

- Canonical ID: `m.track.m3_15c.b`
- Normalized ID: `M3.15c.B`
- Taxonomy: major=`3` minor=`15c` cell=`B`
- Aliases: `M3.15c.B, B`
- Lookup aliases: `M3.15c.B`
- Brief: telemetry-rooted M3.15c ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15c cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15c C

- Canonical ID: `m.track.m3_15c.c`
- Normalized ID: `M3.15c.C`
- Taxonomy: major=`3` minor=`15c` cell=`C`
- Aliases: `M3.15c.C, C`
- Lookup aliases: `M3.15c.C`
- Brief: telemetry-rooted M3.15c ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15c cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15d A

- Canonical ID: `m.track.m3_15d.a`
- Normalized ID: `M3.15d.A`
- Taxonomy: major=`3` minor=`15d` cell=`A`
- Aliases: `M3.15d.A, A`
- Lookup aliases: `M3.15d.A`
- Brief: telemetry-rooted M3.15d ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15d cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15d B

- Canonical ID: `m.track.m3_15d.b`
- Normalized ID: `M3.15d.B`
- Taxonomy: major=`3` minor=`15d` cell=`B`
- Aliases: `M3.15d.B, B`
- Lookup aliases: `M3.15d.B`
- Brief: telemetry-rooted M3.15d ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15d cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15d C

- Canonical ID: `m.track.m3_15d.c`
- Normalized ID: `M3.15d.C`
- Taxonomy: major=`3` minor=`15d` cell=`C`
- Aliases: `M3.15d.C, C`
- Lookup aliases: `M3.15d.C`
- Brief: telemetry-rooted M3.15d ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15d cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.15d D

- Canonical ID: `m.track.m3_15d.d`
- Normalized ID: `M3.15d.D`
- Taxonomy: major=`3` minor=`15d` cell=`D`
- Aliases: `M3.15d.D, D`
- Lookup aliases: `M3.15d.D`
- Brief: telemetry-rooted M3.15d ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.15d cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.16 A

- Canonical ID: `m.track.m3_16.a`
- Normalized ID: `M3.16.A`
- Taxonomy: major=`3` minor=`16` cell=`A`
- Aliases: `M3.16.A, A`
- Lookup aliases: `M3.16.A`
- Brief: telemetry-rooted M3.16 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.16 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### M3.16 B

- Canonical ID: `m.track.m3_16.b`
- Normalized ID: `M3.16.B`
- Taxonomy: major=`3` minor=`16` cell=`B`
- Aliases: `M3.16.B, B`
- Lookup aliases: `M3.16.B`
- Brief: telemetry-rooted M3.16 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.16 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000680`

#### M3.16 C

- Canonical ID: `m.track.m3_16.c`
- Normalized ID: `M3.16.C`
- Taxonomy: major=`3` minor=`16` cell=`C`
- Aliases: `M3.16.C, C`
- Lookup aliases: `M3.16.C`
- Brief: telemetry-rooted M3.16 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.16 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000072`

#### M3.16 D

- Canonical ID: `m.track.m3_16.d`
- Normalized ID: `M3.16.D`
- Taxonomy: major=`3` minor=`16` cell=`D`
- Aliases: `M3.16.D, D`
- Lookup aliases: `M3.16.D`
- Brief: telemetry-rooted M3.16 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.16 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000124`

#### M4.0 telemetry report

- Canonical ID: `m.track.m4_0`
- Normalized ID: `M4.0`
- Taxonomy: major=`4` minor=`0` cell=`None`
- Aliases: `M4.0`
- Lookup aliases: `M4.0`
- Brief: telemetry-rooted M4.0 family report
- Architectural question: `M4`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M4.0 family report
- Scripts: `scripts/run_m4_0_semantic_probe.py`

#### M4.2 telemetry report

- Canonical ID: `m.track.m4_2`
- Normalized ID: `M4.2`
- Taxonomy: major=`4` minor=`2` cell=`None`
- Aliases: `M4.2`
- Lookup aliases: `M4.2`
- Brief: telemetry-rooted M4.2 family report
- Architectural question: `M4`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M4.2 family report
- Scripts: `scripts/run_m4_2_predicate_grounding.py`

#### Reuse-oriented control

- Canonical ID: `m.track.m5_0.a`
- Normalized ID: `M5.0.A`
- Taxonomy: major=`5` minor=`0` cell=`A`
- Aliases: `M5.A`
- Lookup aliases: `M5:M5.A`
- Brief: telemetry-rooted M5 family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5 row

#### Selective lexical adversary + family clustering

- Canonical ID: `m.track.m5_0.c`
- Normalized ID: `M5.0.C`
- Taxonomy: major=`5` minor=`0` cell=`C`
- Aliases: `M5.C`
- Lookup aliases: `M5:M5.C`
- Brief: telemetry-rooted M5 family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5 row

#### Selective lexical adversary + reuse

- Canonical ID: `m.track.m5_0.b`
- Normalized ID: `M5.0.B`
- Taxonomy: major=`5` minor=`0` cell=`B`
- Aliases: `M5.B`
- Lookup aliases: `M5:M5.B`
- Brief: telemetry-rooted M5 family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5 row

#### Add counterfactual invariance

- Canonical ID: `m.track.m5_1.n2`
- Normalized ID: `M5.1.N2`
- Taxonomy: major=`5` minor=`1` cell=`N2`
- Aliases: `M5.N2`
- Lookup aliases: `M5.padded_nary_family:M5.N2`
- Brief: telemetry-rooted M5.padded_nary_family family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5.padded_nary_family row

#### Core + uniformity + GRL

- Canonical ID: `m.track.m5_1.n1`
- Normalized ID: `M5.1.N1`
- Taxonomy: major=`5` minor=`1` cell=`N1`
- Aliases: `M5.N1`
- Lookup aliases: `M5.padded_nary_family:M5.N1`
- Brief: telemetry-rooted M5.padded_nary_family family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5.padded_nary_family row

#### Full padded crucible

- Canonical ID: `m.track.m5_1.n3`
- Normalized ID: `M5.1.N3`
- Taxonomy: major=`5` minor=`1` cell=`N3`
- Aliases: `M5.N3`
- Lookup aliases: `M5.padded_nary_family:M5.N3`
- Brief: telemetry-rooted M5.padded_nary_family family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5.padded_nary_family row

#### Padded core only

- Canonical ID: `m.track.m5_1.n0`
- Normalized ID: `M5.1.N0`
- Taxonomy: major=`5` minor=`1` cell=`N0`
- Aliases: `M5.N0`
- Lookup aliases: `M5.padded_nary_family:M5.N0`
- Brief: telemetry-rooted M5.padded_nary_family family row
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M5.padded_nary_family row

#### M5.2.autoregressive_chain.run telemetry report

- Canonical ID: `m.track.m5_2`
- Normalized ID: `M5.2`
- Taxonomy: major=`5` minor=`2` cell=`None`
- Aliases: `M5.2.autoregressive_chain.run`
- Lookup aliases: `M5.2.autoregressive_chain.run`
- Brief: telemetry-rooted M5.2.autoregressive_chain.run family report
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M5.2.autoregressive_chain.run family report
- Scripts: `scripts/run_m5_2_autoregressive_chain.py`

#### M5.3.masked_pair_chain.run telemetry report

- Canonical ID: `m.track.m5_3`
- Normalized ID: `M5.3`
- Taxonomy: major=`5` minor=`3` cell=`None`
- Aliases: `M5.3.masked_pair_chain.run`
- Lookup aliases: `M5.3.masked_pair_chain.run`
- Brief: telemetry-rooted M5.3.masked_pair_chain.run family report
- Architectural question: `M5`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M5.3.masked_pair_chain.run family report
- Scripts: `scripts/run_m5_3_masked_pair_chain.py`

#### control no re-entry

- Canonical ID: `m.track.m3_17.a`
- Normalized ID: `M3.17.A`
- Taxonomy: major=`3` minor=`17` cell=`A`
- Aliases: `M3.17.A, A`
- Lookup aliases: `M3.17.A`
- Brief: telemetry-rooted M3.17 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.17 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`

#### direct residual re-encoder

- Canonical ID: `m.track.m3_17.d`
- Normalized ID: `M3.17.D`
- Taxonomy: major=`3` minor=`17` cell=`D`
- Aliases: `M3.17.D, D`
- Lookup aliases: `M3.17.D`
- Brief: telemetry-rooted M3.17 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.17 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000427`

#### single return-state bottleneck

- Canonical ID: `m.track.m3_17.b`
- Normalized ID: `M3.17.B`
- Taxonomy: major=`3` minor=`17` cell=`B`
- Aliases: `M3.17.B, B`
- Lookup aliases: `M3.17.B`
- Brief: telemetry-rooted M3.17 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.17 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`-12.486698`

#### three return-state bottleneck

- Canonical ID: `m.track.m3_17.c`
- Normalized ID: `M3.17.C`
- Taxonomy: major=`3` minor=`17` cell=`C`
- Aliases: `M3.17.C, C`
- Lookup aliases: `M3.17.C`
- Brief: telemetry-rooted M3.17 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.17 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`-8.375755`

#### control no advisor

- Canonical ID: `m.track.m3_18.a`
- Normalized ID: `M3.18.A`
- Taxonomy: major=`3` minor=`18` cell=`A`
- Aliases: `M3.18.A, A`
- Lookup aliases: `M3.18.A`
- Brief: telemetry-rooted M3.18 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.18 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000000`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### frozen multi-return token bundle

- Canonical ID: `m.track.m3_18.c`
- Normalized ID: `M3.18.C`
- Taxonomy: major=`3` minor=`18` cell=`C`
- Aliases: `M3.18.C, C`
- Lookup aliases: `M3.18.C`
- Brief: telemetry-rooted M3.18 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.18 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`-8.691725`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`0.850000`

#### frozen single return token

- Canonical ID: `m.track.m3_18.b`
- Normalized ID: `M3.18.B`
- Taxonomy: major=`3` minor=`18` cell=`B`
- Aliases: `M3.18.B, B`
- Lookup aliases: `M3.18.B`
- Brief: telemetry-rooted M3.18 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.18 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`-8.378815`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### hybrid token plus residual translator

- Canonical ID: `m.track.m3_18.e`
- Normalized ID: `M3.18.E`
- Taxonomy: major=`3` minor=`18` cell=`E`
- Aliases: `M3.18.E, E`
- Lookup aliases: `M3.18.E`
- Brief: telemetry-rooted M3.18 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.18 cell
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`-8.879877`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`0.850000`

#### learned residual continuation vector

- Canonical ID: `m.track.m3_18.d`
- Normalized ID: `M3.18.D`
- Taxonomy: major=`3` minor=`18` cell=`D`
- Aliases: `M3.18.D, D`
- Lookup aliases: `M3.18.D`
- Brief: telemetry-rooted M3.18 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.18 cell
- Inherits from: `M3.17.D`
- Best-known metrics: held_out_accuracy=`0.500000`, intervention_effect_on_gold=`0.000060`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### M3.19 D0

- Canonical ID: `m.track.m3_19.d0`
- Normalized ID: `M3.19.D0`
- Taxonomy: major=`3` minor=`19` cell=`D0`
- Aliases: `M3.19.D0, D0`
- Lookup aliases: `M3.19.D0`
- Brief: telemetry-rooted M3.19 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.19 cell
- Inherits from: `M3.18.D`
- Best-known metrics: held_out_accuracy=`0.000000`, intervention_effect_on_gold=`0.000010`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### M3.19 D1

- Canonical ID: `m.track.m3_19.d1`
- Normalized ID: `M3.19.D1`
- Taxonomy: major=`3` minor=`19` cell=`D1`
- Aliases: `M3.19.D1, D1`
- Lookup aliases: `M3.19.D1`
- Brief: telemetry-rooted M3.19 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.19 cell
- Best-known metrics: held_out_accuracy=`0.000000`, intervention_effect_on_gold=`-0.000002`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### M3.19 D2

- Canonical ID: `m.track.m3_19.d2`
- Normalized ID: `M3.19.D2`
- Taxonomy: major=`3` minor=`19` cell=`D2`
- Aliases: `M3.19.D2, D2`
- Lookup aliases: `M3.19.D2`
- Brief: telemetry-rooted M3.19 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.19 cell
- Best-known metrics: held_out_accuracy=`0.000000`, intervention_effect_on_gold=`-0.000002`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### M3.19 D3

- Canonical ID: `m.track.m3_19.d3`
- Normalized ID: `M3.19.D3`
- Taxonomy: major=`3` minor=`19` cell=`D3`
- Aliases: `M3.19.D3, D3`
- Lookup aliases: `M3.19.D3`
- Brief: telemetry-rooted M3.19 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M3.19 cell
- Best-known metrics: held_out_accuracy=`0.000000`, intervention_effect_on_gold=`-0.000002`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### relaxed residual scratchpad

- Canonical ID: `m.track.m14.c`
- Normalized ID: `M14.C`
- Taxonomy: major=`14` minor=`None` cell=`C`
- Aliases: `M14.C, C`
- Lookup aliases: `M14.C`
- Brief: telemetry-rooted M14 ablation cell
- Architectural question: `M14`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M14 cell
- Inherits from: `M3.19.D0, M11`
- Best-known metrics: held_out_accuracy=`1.000000`, intervention_effect_on_gold=`0.000269`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### scratchpad-only control

- Canonical ID: `m.track.m14.a`
- Normalized ID: `M14.A`
- Taxonomy: major=`14` minor=`None` cell=`A`
- Aliases: `M14.A, A`
- Lookup aliases: `M14.A`
- Brief: telemetry-rooted M14 ablation cell
- Architectural question: `M14`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M14 cell
- Best-known metrics: held_out_accuracy=`1.000000`, intervention_effect_on_gold=`0.000000`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### severance-threshold residual scratchpad

- Canonical ID: `m.track.m14.d`
- Normalized ID: `M14.D`
- Taxonomy: major=`14` minor=`None` cell=`D`
- Aliases: `M14.D, D`
- Lookup aliases: `M14.D`
- Brief: telemetry-rooted M14 ablation cell
- Architectural question: `M14`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M14 cell
- Best-known metrics: held_out_accuracy=`1.000000`, intervention_effect_on_gold=`0.000264`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### strict residual scratchpad

- Canonical ID: `m.track.m14.b`
- Normalized ID: `M14.B`
- Taxonomy: major=`14` minor=`None` cell=`B`
- Aliases: `M14.B, B`
- Lookup aliases: `M14.B`
- Brief: telemetry-rooted M14 ablation cell
- Architectural question: `M14`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M14 cell
- Best-known metrics: held_out_accuracy=`1.000000`, intervention_effect_on_gold=`-0.000150`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### token-only scratchpad baseline

- Canonical ID: `m.track.m14.e`
- Normalized ID: `M14.E`
- Taxonomy: major=`14` minor=`None` cell=`E`
- Aliases: `M14.E, E`
- Lookup aliases: `M14.E`
- Brief: telemetry-rooted M14 ablation cell
- Architectural question: `M14`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`runnable`
- Baseline relation: M14 cell
- Best-known metrics: held_out_accuracy=`1.000000`, intervention_effect_on_gold=`-0.000282`, resume_first_token_accuracy=`0.000000`, english_fluency_score=`1.000000`

#### M3.13 A

- Canonical ID: `m.track.m3_13.a`
- Normalized ID: `M3.13.A`
- Taxonomy: major=`3` minor=`13` cell=`A`
- Aliases: `M3.13.A, A`
- Lookup aliases: `M3.13.A`
- Brief: telemetry-rooted M3.13 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.13 cell

#### M3.13 B

- Canonical ID: `m.track.m3_13.b`
- Normalized ID: `M3.13.B`
- Taxonomy: major=`3` minor=`13` cell=`B`
- Aliases: `M3.13.B, B`
- Lookup aliases: `M3.13.B`
- Brief: telemetry-rooted M3.13 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.13 cell

#### M3.13 C

- Canonical ID: `m.track.m3_13.c`
- Normalized ID: `M3.13.C`
- Taxonomy: major=`3` minor=`13` cell=`C`
- Aliases: `M3.13.C, C`
- Lookup aliases: `M3.13.C`
- Brief: telemetry-rooted M3.13 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.13 cell

#### M3.13 D

- Canonical ID: `m.track.m3_13.d`
- Normalized ID: `M3.13.D`
- Taxonomy: major=`3` minor=`13` cell=`D`
- Aliases: `M3.13.D, D`
- Lookup aliases: `M3.13.D`
- Brief: telemetry-rooted M3.13 ablation cell
- Architectural question: `M3`
- Provenance: evidence=`artifact` confidence=`high` reproducibility=`artifact_only`
- Baseline relation: M3.13 cell


## historical_gap

Known missing artifacts, orphaned checkpoints, and preserved gaps.

### historical_gap (1)

#### Two-stage phase-5 baseline restoration path

- Canonical ID: `restoration.phase5.two_stage_baseline`
- Aliases: `Restoration Path, Lost Asset`
- Lookup aliases: `restoration:phase5_baseline`
- Brief: document the missing production baseline asset and the restoration command path
- Provenance: evidence=`doc_reported` confidence=`medium` reproducibility=`orphaned`
- Baseline relation: restoration target
- Best-known metrics: held_out_accuracy=`0.396000`, logical_accuracy=`0.417000`
- Scripts: `scripts/run_phase5_two_stage_recovery.py, scripts/mine_compositional_anchors.py`
- Notes: Doc-reported missing asset for the original two-stage control baseline. | Serves as gap handling rather than a confirmed artifact-backed ablation run.


## historical_gaps

### missing.phase5_two_stage_stage2

- Kind: `missing_artifact`
- Path: `runs/phase5_two_stage_recovery_anchors/20260224_225142/stage2_phase5`
- Source: `docs/history/reports/AUDIT_REPORT.md`
- Description: Original Run B control baseline asset reported missing in audit report.

### orphaned.h3_projection_weights

- Kind: `orphaned_checkpoint`
- Path: `runs/projections/swiglu_midlayer_bridge_h3_exp4.pt`
- Source: `docs/history/reports/AUDIT_REPORT.md`
- Description: Audit report describes H3/H4 projection weights as mathematically orphaned relative to the deleted adapter manifold.
