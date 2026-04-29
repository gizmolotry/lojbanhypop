# Retrospective Run Report

- Generated: `2026-04-14`
- Workspace: `D:\lojbanhypop`
- Purpose: run as much of the historical and modern ablation program as is realistically recoverable today, then summarize the actual runnable state of the codebase.

## Executive Summary

The repository is now in a meaningfully better state than it was at the start of this pass:

- there is a unified runnable legacy grid surface rather than a pile of one-off entrypoints
- the widest recoverable legacy family set can now be launched from one runner and one Airflow DAG
- optional legacy training lanes that had broken pathing were repaired
- the expanded retrospective run completed across both the core legacy surfaces and the heavier optional training lanes

What is true after this run:

- `A-G` is recoverable in a reduced runnable form
- `H/H5/J` is runnable
- `L6` is runnable in recovered form
- `Phase-5 objective` is runnable
- `Phase-5 train` is runnable
- `English CoT duel` is runnable

What is still not true:

- we still cannot honestly claim that the entire historical program is fully rerunnable end to end
- not every legacy artifact has the missing checkpoints or adapters needed for historically exact reproduction
- the modern `M` families are only partially rerun in this pass; for most of them, the current source of truth remains the unified ledger and prior telemetry artifacts

## Fresh Execution Artifacts

Primary unified legacy retrospective:

- [legacy_ablation_grid_manifest.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\legacy_grid\legacy_grid_retrospective_20260413\legacy_ablation_grid_manifest.json)
- [legacy_ablation_grid_summary.md](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\legacy_grid\legacy_grid_retrospective_20260413\legacy_ablation_grid_summary.md)

Supporting fresh probes:

- [phase5 live probe manifest](D:\lojbanhypop\runs\phase5_train_ablation\live_probe\20260414_002219\ablation_manifest.json)
- [english duel live probe manifest](D:\lojbanhypop\runs\english_cot_control_duel\live_probe\20260414_002219\english_cot_control_manifest.json)

Program-wide control-plane surfaces used for retrospective context:

- [ABLATION_PROGRAM_SPINE.md](D:\lojbanhypop\docs\ABLATION_PROGRAM_SPINE.md)
- [m_bridge_ablation_suite_manifest.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m_bridge_ablation_test_suite\full_program_probe_20260413\m_bridge_ablation_suite_manifest.json)

## What Was Newly Fixed

These were code-level recoverability fixes, not just reports:

- [run_legacy_ablation_grid.py](D:\lojbanhypop\scripts\legacy\run_legacy_ablation_grid.py)
  Added one unified legacy runner with lane-level execution and aggregate-only rebuild support.

- [lojban_legacy_ablation_grid_dag.py](D:\lojbanhypop\airflow\dags\legacy\lojban_legacy_ablation_grid_dag.py)
  Added one Airflow DAG for the widest recoverable legacy grid.

- [lojban_ablation_master_spine_dag.py](D:\lojbanhypop\airflow\dags\control_plane\lojban_ablation_master_spine_dag.py)
  Registered the new legacy grid surface into the master orchestration spine.

- [run_phase5_train_ablation.py](D:\lojbanhypop\scripts\legacy\run_phase5_train_ablation.py)
  Fixed moved `train_lora.py` pathing after the repo cleanup.

- [run_english_cot_control_duel.py](D:\lojbanhypop\scripts\legacy\run_english_cot_control_duel.py)
  Fixed moved dataset-builder and trainer paths after the repo cleanup.

- [eval_h5_ood_stress.py](D:\lojbanhypop\scripts\legacy\eval_h5_ood_stress.py)
  Patched legacy imports so the H5 OOD surface runs again.

- [eval_h5_dynamic_pointer_refactor.py](D:\lojbanhypop\scripts\legacy\eval_h5_dynamic_pointer_refactor.py)
  Patched legacy imports so the dynamic pointer refactor eval runs again.

## Fresh Legacy Grid Results

### A-G

Artifact:

- [ablation_matrix.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\a_to_g\legacy_grid\legacy_grid_retrospective_20260413\20260414_002554\ablation_matrix.json)

Result:

- lane status: `ok`
- executed runs: `4`
- control base final accuracy: `0.3333`
- coconut handoff final accuracy: `0.0`
- NoPE handoff lift: `0.0`

Interpretation:

- the reduced recovered A-G lane is real and executable
- the generative handoff path remains non-competitive in this recovered run
- `B.2` and `E` are still missing extra assets, so this is not yet the full historical matrix

### H / H5 / J

Artifact:

- [run_h_series.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\legacy_grid\legacy_grid_retrospective_20260413\20260414_003217\run_h_series.json)

Result:

- lane status: `ok`
- executed runs: `12`
- `H1` handoff lift: `-0.3333`
- `H5-OOD` accuracy: `0.40`
- `J-1` schema valid rate: `1.0`
- `J-5` accepted foil-pair accuracy: `1.0`

Interpretation:

- the bridge-style H family is runnable but still empirically bad in this recovered setting
- the H5 extension diagnostics are alive again
- the J family is in much healthier shape than the older bridge families and behaves like a stable recovered diagnostic substrate

### L6

Artifact:

- [l6_ablation_manifest.json](D:\lojbanhypop\runs\l_series\l6_ablation\legacy_grid\legacy_grid_retrospective_20260413\20260414_004443\l6_ablation_manifest.json)

Result:

- lane status: `ok`
- executed rows: `3`
- mean scope constraint: `0.3432`
- best scope constraint: `0.3810`

Interpretation:

- the L6 branch is meaningfully runnable as a recovered training surface
- it is still a reduced reprobe, not a historically exact full-scale recreation

### Phase-5 Objective

Artifact:

- [phase5_objective_ablation.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\legacy_grid\legacy_grid_retrospective_20260413\phase5_objective_ablation.json)

Result:

- lane status: `ok`
- full total regularizer: `2.91397`
- dead term count: `0`
- dominant term: `semantic_unambiguity_loss`
- dominant term value: `56.22988`

Interpretation:

- this surface is cheap, deterministic, and healthy
- the semantic-unambiguity term remains the dominant force in the phase objective stack

### Phase-5 Train

Artifacts:

- initial storage-limited pass: [ablation_manifest.json](D:\lojbanhypop\runs\phase5_train_ablation\legacy_grid\legacy_grid_retrospective_20260413\20260414_004555\ablation_manifest.json)
- final successful rerun: [ablation_manifest.json](D:\lojbanhypop\runs\phase5_train_ablation\legacy_grid\legacy_grid_retrospective_20260413\20260414_112250\ablation_manifest.json)

Result:

- lane status: `ok`
- total variants: `9`
- successful variants: `9`
- failed variants: `0`

Interpretation:

- the code path is real
- the earlier partial sweep was blocked by storage pressure rather than training logic
- after freeing space, the full lane completed cleanly inside the same retrospective run id

### English CoT Duel

Artifacts:

- initial storage-limited pass: [english_cot_control_manifest.json](D:\lojbanhypop\runs\english_cot_control_duel\legacy_grid\legacy_grid_retrospective_20260413\20260414_004909\english_cot_control_manifest.json)
- final successful rerun: [english_cot_control_manifest.json](D:\lojbanhypop\runs\english_cot_control_duel\legacy_grid\legacy_grid_retrospective_20260413\20260414_112627\english_cot_control_manifest.json)

Result:

- lane status: `ok`
- dataset build: `ok`
- English CoT train: `ok`
- English CoT eval: `ok`
- Lojban reference eval: `ok`
- comparison:
  - `base_acc = 0.0`
  - `english_cot_adapter_acc = 0.0`
  - `lojban_adapter_acc = 0.0`

Interpretation:

- the lane is operationally healthy now
- at this tiny retrospective setting it produced no lift, but that is a result question rather than a recoverability question

## Modern Program State From The Control Plane

The unified control plane remains the best source of truth for the full historical-to-modern surface:

- [m_bridge_ablation_suite_manifest.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m_bridge_ablation_test_suite\full_program_probe_20260413\m_bridge_ablation_suite_manifest.json)

Key retrospective facts from that suite:

- total tracked entries in the backfilled history: `168`
- runnable-only entries in the history: `67`
- artifact-backed entries: `149`

Important modern findings already present in the ledger:

- `M11` remains the strongest positive result surface
  - headline accuracy: `0.85916`
  - floor-lock accuracy: `0.78`
  - publication mean accuracy: `0.77`

- no generative bridge family from `M3.15d` through `M3.19` has yet beaten the control path on a fully convincing basis in the ledger summary

- harmful bridge or return-path cells still cluster around:
  - `M3.17/B`
  - `M3.17/C`
  - `M3.18/B`
  - `M3.18/C`
  - `M3.18/E`

## M-Series Snapshot

The best single source for current family-level status is:

- [ABLATION_PROGRAM_SPINE.md](D:\lojbanhypop\docs\ABLATION_PROGRAM_SPINE.md)

The high-level picture today:

- `M1` derives from `J` and is runnable
- `M2` derives from `L` and is runnable in reduced form
- `M3` has broad historical coverage, but only part of the family is currently runnable end to end
- `M4` and `M5` have partial runnable surfaces
- `M11` is the strongest positive modern result family in the current ledger
- `M14` is chartered and modeled in the spine, but not yet a fresh rerun in this retrospective
- `M19` is currently the most operationally mature scratchpad-runway family in the modern stack

### M19 Current State

Primary report:

- [m19_isolation_grid_report.json](D:\lojbanhypop\artifacts\runs\telemetry\raw\ablation\hypercube\m19_isolation_grid\m19_isolation_grid_20260409_v2\m19_isolation_grid_report.json)

Useful currently backed results from `M19.3`:

- `A` `8Q / 64D / 8S`: accuracy `0.08`, avg tokens `32.0`
- `B` `8Q / 128D / 8S`: accuracy `0.30`, avg tokens `25.64`
- `C` `16Q / 64D / 8S`: accuracy `0.24`, avg tokens `22.16`
- `D` `8Q / 64D / 12S`: accuracy `0.22`, avg tokens `31.9`
- `E` `8Q / 128D / 12S`: accuracy `0.26`, avg tokens `19.88`
- `F` `16Q / 64D / 12S`: accuracy `0.24`, avg tokens `27.76`

Current interpretation:

- width helped more than extra queries in that stored grid
- the best raw accuracy cell in the stored report is `B`
- the best accuracy/token trade in that stored grid is `E`
- the attempted fresh rerun in this session only completed cell `A`, so the stored v2 grid remains the current trustworthy whole-grid M19 reference

## Coverage Verdict

### What Is Actually Runnable Today

- recovered `A-G` core subset
- recovered `H/H5/J`
- recovered `L6`
- `Phase-5 objective`
- `Phase-5 train`
- `English CoT duel`
- `M19` and several modern M-family surfaces through their existing runners and DAGs
- the control plane itself

### What Is Only Partially Runnable

- exact historical `A-G` full matrix because some specialty assets are still missing
- `L` beyond the recovered L6 surface
- broader `M3` and `M14` generative re-entry surfaces as a single easy rerun stack

### What Is Mostly Ledger-Backed Rather Than Freshly Re-executed In This Pass

- much of the pre-telemetry core matrix
- parts of `H` and `H5`
- large parts of `M3` through `M18`
- `M11` oracle results
- archived `L` branch extensions

## Main Scientific Takeaways

1. The old continuous bridge families are no longer just theoretically shaky; they are now operationally recoverable enough to show the same failure shape again under fresh execution.

2. The J-derived invariance and adversarial surfaces remain much healthier than the old generative handoff surfaces.

3. The L6 constraint branch is one of the more recoverable training-era legacy families.

4. The modern ledger still points to the same big story:
   native or discriminative manifold readout is strong
   generative direct bridge exposure remains fragile
   scratchpad and bounded re-entry families are the more plausible forward direction

5. Repo cleanliness and orchestration have improved enough that a senior engineer can now follow one coherent legacy grid path instead of spelunking through scattered scripts, but the historical recovery is still not complete.

## Practical Blockers

1. Some legacy specialty assets are still absent, so not every historical cell can be rerun exactly.

2. Several modern families are better represented by existing telemetry artifacts than by one-step "run everything" DAGs today.

3. The heavier training lanes are runnable now, but they remain more operationally fragile than the lightweight eval and aggregation surfaces.

## Recommended Next Steps

1. Extend the unified grid concept upward into the M families so the repo has:
   - one legacy grid DAG
   - one modern M-family grid DAG layer
   - one control-plane aggregate report over both

2. Backfill honest lane-status semantics into more family aggregators so "artifact exists" never gets mistaken for "run succeeded."

3. Continue recovering missing specialty assets for:
   - `A-G/B.2`
   - `A-G/E`
   - deeper L-branch family rows

4. Use the regained disk headroom to rerun a fuller modern follow-up, with the strongest candidates being:
   - fresh `M19` replications
   - formal `M14` implementation and sweep
   - a wider modern-family DAG layer beyond the legacy grid
