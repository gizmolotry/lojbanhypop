# Branch Strategy

This repo uses branches for active work and tags for historical milestones.

## Rule Of Thumb

- Branch = work may continue here.
- Tag = this exact historical point matters.
- Artifact = this is evidence from a run.
- Doc/ledger = this explains meaning.

Do not use branches as a lab notebook. If a branch is not expected to receive new commits this week, archive it as a tag and remove the local branch label.

## Active Branches

| Branch | Commit | Purpose |
| --- | --- | --- |
| `master` | legacy baseline | Compatibility/default branch anchor. Do not use as the current research mainline until explicitly promoted. |
| `codex/m21-judri-gated-bridge-20260517` | current HEAD | Active research/control-plane mainline after M21 cleanup and branch pruning. |

## Archive Tag Namespace

Historical branch tips are preserved under:

```text
archive/*
```

The archive tags are annotated. They preserve the old local branch tips before local branch pruning.

## Cleanup Procedure

1. Confirm the working tree is clean.
2. Confirm candidate branches are ancestors of the active branch with `git rev-list --left-right --count old...current`.
3. Create an annotated archive tag for each old branch tip.
4. Record tag mappings in `docs/ACTIVE_BRANCH_LEDGER.md`.
5. Delete only local branch names with `git branch -d`.
6. Do not delete remote branches in the same pass unless archive tags have been pushed and default-branch/PR expectations are checked.
7. Export stashes before clearing them.
8. Save a physical script/DAG snapshot under `archive/cleanup_snapshots/` before aggressive cleanup.

## Remote Branches

Remote branch cleanup is a separate operation. Local pruning does not delete remote branches.

## Stashes

Stashes are hidden state. Export them to patches before clearing:

```text
archive/cleanup_snapshots/<snapshot>/stash_exports/
```
