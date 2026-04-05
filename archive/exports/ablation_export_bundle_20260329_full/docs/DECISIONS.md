# Decision Log

## 2026-02-26 - Artifact Layout
- Decision: Default experiment outputs move to `artifacts/runs/` for core run scripts.
- Why: Keep generated data separate from source and reduce accidental commits.
- Impact: `scripts/run_experiment.py` and `scripts/run_phase_ablation.py` now default to `artifacts/runs`.

## 2026-02-26 - Reproducibility Manifest
- Decision: Each primary run writes `run_manifest.json` with args, git commit, dataset fingerprint, and output paths.
- Why: Make results auditable and easier to reproduce.
- Impact: Added `src/lojban_evolution/repro.py`; wired manifests into main run scripts.

## 2026-02-26 - CI Baseline
- Decision: Add GitHub Actions workflow to run `pytest -q` on push and pull request.
- Why: Catch breakage early and enforce a minimum quality gate.
- Impact: Added `.github/workflows/ci.yml`.
