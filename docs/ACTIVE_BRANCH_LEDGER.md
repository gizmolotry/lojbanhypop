# Active Branch Ledger

Generated as part of the 2026-05-21 aggressive cleanup pass.

## Local Branches Kept At Pruning Time

These commits are the branch tips observed before the cleanup documentation and scaffold commits were created.

| Branch | Commit | Subject |
| --- | --- | --- |
| `codex/m21-judri-gated-bridge-20260517` | `8b67f15` | Stabilize cartography manifest provenance |
| `master` | `74a3cfb` | feat(m18): implement two-pass attention intervention and baseline Sapir-Whorf audit |

## Archived Branch Tips At Pruning Time

| Archive tag | Commit | Original subject |
| --- | --- | --- |
| `archive/codex-git-hygiene-cleanup` | `d41ee58` | feat(m9): implement decoupled harvester and forge microservices via ZeroMQ |
| `archive/codex-lineage-methodology-cleanup` | `d41ee58` | feat(m9): implement decoupled harvester and forge microservices via ZeroMQ |
| `archive/codex-m19-lojbanic-physics-retrofit-snapshot-20260429` | `ff258ad` | Add M19 pointer counterfactual suite |
| `archive/codex-m20-dictionary-first-20260514` | `ff258ad` | Add M19 pointer counterfactual suite |
| `archive/codex-m21-dynamic-bridi-20260515` | `ff258ad` | Add M19 pointer counterfactual suite |
| `archive/codex-m21-hyperbolic-bridi` | `d496837` | M21 hyperbolic tangent handoff and gauntlet ledger |
| `archive/codex-repo-layout-cleanup` | `d41ee58` | feat(m9): implement decoupled harvester and forge microservices via ZeroMQ |
| `archive/m18-sapir-whorf-victory` | `6ce32e1` | chore: massive cull of legacy series and professional rebrand |
| `archive/m19-neuro-symbolic-mainline` | `2a8fd03` | chore: cleanup redundant restorations, keeping legacy safe in archive/ |
| `archive/m6-neuro-symbolic-engine` | `5018510` | feat(m6): achieve 70 percent accuracy on severed bridge after 100 step alignment |
| `archive/m7-interleaved-coprocessor` | `18646a1` | feat(m7): implement the interleaved latent coprocessor with vector choke |
| `archive/m8-council-of-oracles` | `fff1ae7` | feat(m8): implement council of oracles with parallel latent broadcast and supreme judge resolution |
| `archive/m9-contrastive-nli-engine` | `74a3cfb` | feat(m18): implement two-pass attention intervention and baseline Sapir-Whorf audit |

## Safety Snapshot

A physical copy of runnable scripts, DAGs, selected docs, configs, and stash patches was saved before pruning:

```text
archive/cleanup_snapshots/20260521_branch_cleanup/
```

Snapshot contents:

- `payload/scripts/`
- `payload/airflow/dags/`
- `payload/configs/`
- selected project/cartography docs
- `stash_exports/`
- `cleanup_snapshot_manifest.json`

The snapshot is intentionally local-only and ignored by Git to avoid committing a duplicate warehouse of scripts.

## Local Branches Pruned

The following local branch labels were deleted after archive tags were created:

```text
m6_neuro_symbolic_engine
m7_interleaved_coprocessor
m8_council_of_oracles
m9_contrastive_nli_engine
m18_sapir_whorf_victory
m19_neuro_symbolic_mainline
codex/git-hygiene-cleanup
codex/lineage-methodology-cleanup
codex/repo-layout-cleanup
codex/m19-lojbanic-physics-retrofit-snapshot-20260429
codex/m20-dictionary-first-20260514
codex/m21-dynamic-bridi-20260515
codex/m21-hyperbolic-bridi
```

`master` was intentionally kept for compatibility.

## Stashes Cleared

Four historical stashes were exported to patch files and then cleared from Git stash state:

```text
archive/cleanup_snapshots/20260521_branch_cleanup/stash_exports/
```
