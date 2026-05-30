# Milestone Report Workspace

Populate this directory with:
- problem and method summary,
- baseline + initial HGPO plots/tables,
- reproducibility appendix (configs, seeds, run IDs),
- risk list and week 3-4 plan.

## Current SOTA

- **`sota_R13_goalcondFiLM.md`** — **80% pct_success** at R13 (goalcondFiLM + K=12 phase). Headline result; supersedes prior baseline. Full launch recipe, FiLM γ/β trajectory, and deprecated-path lessons captured inline.
- `sota_R12.md` — 73% pct_success at R12 (plain v3 recipe, K=8, no FiLM). Prior baseline; preserved for the +7pp delta narrative.

## Data Inventory (for post-hoc analysis)

- **`data_inventory.md`** — complete index of Modal volume artifacts (adapters, replay buffers, ckpts, manifests) + local orchestrator logs + schema docs for `train_log.json` / `replay.jsonl` / `ckpt.pt`. Includes analysis cookbook (eval curves, FiLM γ/β trajectories, per-task-type breakdowns, dead-K rates, SOTA vs baseline comparisons). Use this as the canonical entry point when generating final-report figures.
- **`logs_sota_R13/`** — archived local orchestrator logs (~30 MB, 6 files) for the 4-phase goalcondFiLM lineage + 2 deprecated runs preserved for forensic comparison.
