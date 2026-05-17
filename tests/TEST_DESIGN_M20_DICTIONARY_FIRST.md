# M20 Dictionary-First Predicate Induction Test Design

Current status: `src/lojban_evolution/m20`, `scripts/m20`, and M20 registry entries now exist. The executable tests focus on the synthetic dictionary-first trainer, six-cell M20.1 grid, registry/DAG ledger wiring, and CLI smoke.

## Recommended New Test Files

- `tests/test_m20_dictionary_first_dataset.py`
- `tests/test_m20_predicate_induction.py`
- `tests/test_m20_locks.py`
- `tests/test_m20_training_selection.py`
- `tests/test_m20_direct_unified_eval.py`
- `tests/test_m20_cli_smoke.py`

## Expected Public APIs Before Executable Tests

- `lojban_evolution.m20.dataset`: dataset row normalization, dictionary attachment, train/eval split purging.
- `lojban_evolution.m20.induction`: predicate candidate induction, dictionary-first filtering, predicate ID assignment.
- `lojban_evolution.m20.locks`: pure lock metric computation and pass/fail classification.
- `lojban_evolution.m20.training`: checkpoint scoring/selection using lock metrics.
- `lojban_evolution.m20.family`: registry metadata for scripts, DAGs, output roots, report names, and dataset defaults.
- `scripts/m20/*.py`: argparse-compatible CLI entrypoints with `--help`.

## Six Lock Test Cases

1. Dictionary precedence lock
   - Input: rows containing a valid dictionary predicate and a tempting non-dictionary paraphrase.
   - Assert: induction selects only dictionary-backed predicate IDs unless an explicit OOV path is enabled.
   - Assert: report metrics expose `dictionary_coverage`, `oov_predicate_rate`, and `dictionary_precedence_violation_rate`.

2. Predicate identity stability lock
   - Input: same predicate across renamed entities, flattened formatting, and reordered examples.
   - Assert: the canonical predicate ID is unchanged across transforms.
   - Assert: duplicate aliases collapse to one stable ID and do not create split-brain predicates.

3. Arity lock
   - Input: unary, binary, and ternary dictionary entries plus adversarial rows with missing/extra arguments.
   - Assert: induced predicate arity matches the dictionary arity.
   - Assert: malformed rows increment `arity_violation_rate` and are rejected or quarantined according to config.

4. Argument binding lock
   - Input: minimal pairs where predicate identity is fixed but argument order changes.
   - Assert: bound argument slots preserve order and role labels.
   - Assert: pointer/judri-style metrics distinguish swapped arguments from correct bindings.

5. Surface invariance lock
   - Input: entity anonymized, entity renamed, format flattened, and numeric-normalized rows following M19 kill-test style.
   - Assert: predicate IDs and arity remain stable across all surface transforms.
   - Assert: `entity_accuracy`, `entity_renamed_accuracy`, `format_accuracy`, and `numeric_accuracy` do not collapse relative to purged accuracy.

6. Leakage and shortcut lock
   - Input: train/eval overlaps, answer-masked prompts, and dictionary entries containing answer-like surface tokens.
   - Assert: overlap rows are separated from purged rows.
   - Assert: masked-prompt accuracy collapses while dictionary coverage remains measurable.
   - Assert: direct answer leakage produces a failing `leakage_status`.

## Ledger and DAG CLI Smoke

Add these to `tests/test_m20_cli_smoke.py` once scripts exist:

- `scripts/m20/run_m20_dictionary_first_suite.py --help`
  - Assert `usage:`, `--dictionary-path`, `--train-data-path`, `--eval-data-path`, `--output-root`.
- `scripts/m20/run_m20_predicate_induction.py --help`
  - Assert `usage:`, `--dictionary-path`, `--dataset-path`, `--output-path`.
- `scripts/m20/run_m20_lock_suite.py --help`
  - Assert `usage:`, `--induction-report`, `--purged-eval-path`, `--output-path`.
- `scripts/control_plane/run_direct_unified_eval.py --help`
  - Assert a future M20 report option exists, or that generic `--family-key M20`/surface report wiring is documented.

Add a registry smoke test once `lojban_evolution.m20.family` exists:

- Assert `M20_REGISTRY["M20"]["runner_scripts"]` contains `train`, `predicate_induction`, `lock_suite`, and `suite`.
- Assert `M20_REGISTRY["M20"]["dags"]` contains the same surfaces and paths under Airflow.
- Assert `output_roots` and `report_names` are present for ledger discovery.

## Synthetic Report Contract Tests

Add to `tests/test_m20_direct_unified_eval.py` once direct eval supports M20:

- Build synthetic benchmark, induction, lock-suite, replication, and ledger reports under a temp directory.
- Call `build_direct_unified_eval_manifest(family_key="M20", track="M20", ...)`.
- Assert contract rows are `available` for dictionary precedence, predicate identity, arity, argument binding, surface invariance, and leakage locks.
- Assert `headline_metrics` include lock pass rate, dictionary coverage, OOV rate, predicate identity stability, arity violation rate, argument binding accuracy, masked accuracy, and leakage status.

## Still Avoid

- No GPU tests; keep M20 tests synthetic and CPU-only like the existing M19 unit and CLI smoke tests.
- No promotion claims from phrase accuracy; `strict_accuracy` remains canonical.
- No M19 bridge imports inside M20 core modules.
