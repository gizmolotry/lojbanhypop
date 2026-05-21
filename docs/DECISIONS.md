# Decision Log

## 2026-05-21 - Local Branch Archive And Pruning
- Decision: Preserve historical local branch tips as annotated `archive/*` tags, then delete redundant local branch labels already contained by `codex/m21-judri-gated-bridge-20260517`.
- Why: Branches should represent active work, not a lab-notebook museum. Tags and ledgers preserve milestones without cluttering the branch picker.
- Impact: Local branches now keep only `master` and the active M21 branch. Scripts/DAGs/configs/docs were physically snapshotted under `archive/cleanup_snapshots/20260521_branch_cleanup/`, and four historical stashes were exported as patches before clearing.

## 2026-05-21 - Canonical Path Registry
- Decision: Centralize moved script/DAG path aliases in `src/lojban_evolution/control_plane/path_registry.py`.
- Why: DAGs, ledgers, and report builders had duplicated stale `scripts/run_*.py` path maps.
- Impact: Airflow `run_repo_script()` now canonicalizes legacy paths before execution; control-plane map/spine/backfill scripts use the shared registry.

## 2026-02-26 - Artifact Layout
- Decision: Default experiment outputs move to `artifacts/runs/` for core run scripts.
- Why: Keep generated data separate from source and reduce accidental commits.
- Impact: `scripts/legacy/run_experiment.py` and `scripts/run_phase_ablation.py` now default to `artifacts/runs`.

## 2026-02-26 - Reproducibility Manifest
- Decision: Each primary run writes `run_manifest.json` with args, git commit, dataset fingerprint, and output paths.
- Why: Make results auditable and easier to reproduce.
- Impact: Added `src/lojban_evolution/repro.py`; wired manifests into main run scripts.

## 2026-02-26 - CI Baseline
- Decision: Add GitHub Actions workflow to run `pytest -q` on push and pull request.
- Why: Catch breakage early and enforce a minimum quality gate.
- Impact: Added `.github/workflows/ci.yml`.
