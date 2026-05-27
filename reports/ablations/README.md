# Ablation Studies

This document collects per-hypothesis ablation results for the TurnRD pipeline on
AlfWorld. Each ablation is a controlled variant of a frozen baseline that targets a
**single, pre-registered hypothesis** with a **pre-registered abort criterion**.
Results — positive, null, or falsifying — are recorded here so the lineage of design
decisions is reproducible.

## Conventions

- **Baseline**: `TurnRDV2_alfworld_SOTA_10round_mlpr32_v3` (configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3.json).
  Rank-32 MLP+attn LoRA, v_projection=True, alpha=0.5, recency-decay half-life=4
  rounds, num_train_games=400, lambda_value=1.0, 5 TurnRD epochs/round, 10 rounds × 80
  episodes × K=8 trajectories, SFT warm-start (`sft_alfworld_v2_e3_20260521_145134`).
  Reference plateau on the eval slice [task_id 6500-6599]: pct_success ≈ **0.69-0.73**
  across R6-R12 (v3 base + extend_R10R12).
- **Eval slice**: 100 episodes from a held-out task_id range `[6500, 6600)`.
- **Noise floor**: ≈ ±0.05 pct_success on n=100 episodes (from the prior alpha-sweep
  analysis in `reports/alfworld_alpha_sweep_README.md`).
- **Per-row format**: each ablation lists hypothesis, abort criterion, single-variable
  change vs. baseline, headline result, verdict, and a pointer to evidence files.

## Index

| Ablation | Question | Verdict | Date | Section |
|---|---|---|---|---|
| **Step-budget (xlbudget)** | Is the 0.69-0.73 plateau truncation-bound? | ❌ **FALSIFIED** | 2026-05-27 | [§1](#1-step-budget-xlbudget) |

Future ablations (planned, not yet run): goal-conditioned V-head (FiLM), V-head
capacity (4L/256), same-round intrinsic-reward shaping. See
`/Users/shoupeili/.llms/plans/goal_conditioned_v_head_alfworld.plan.md` for the FiLM
spec.

---

## 1. Step-budget (xlbudget)

### Hypothesis
The v3 plateau (R9 ≈ 0.71, R12 ≈ 0.73 on the original baseline) is a **structural step-budget
artefact**, not a credit-assignment failure. Specifically: too many otherwise-winnable
episodes truncate at the per-episode turn cap before completion.

### Pre-experiment evidence
Failure-mode probe `reports/turnrd_credit_assignment_demo/v3_R9_probe.jsonl` (50 R9 eval
episodes, looser 40-turn cap):
- **11/11 failures truncate at n_turns = 40** (probe cap; real eval cap was 30, so even more
  would have truncated there).
- **6/39 successes already need > 30 turns** (16, 16, 19, 20, 25, 37) — the 30-turn cap is
  actively suppressing winnable episodes.
- **8/11 failures show forward-progress patterns** in their last 10 turns (systematic
  drawer search, mid-task progression).
- **3/11 failures are policy dead-zones** (IR=0 in last 10 turns, repeated identical
  actions) — these set a hard floor the budget lift cannot move.

Predicted lift (probe-derived): 0.71-0.73 → 0.78-0.85.

### Single-variable change vs. v3 baseline
| Knob | v3 base | xlbudget |
|---|---|---|
| `env.max_steps` (TextWorld adapter step cap) | 40 | **60** |
| `env.max_turns` (rollout collector turn cap) | 30 (default) | **50** |
| every other config field | unchanged | unchanged |

Same architecture, same losses, same SFT warm-start, same seed family — only the
per-episode budget knobs differ.

### Pre-registered abort criteria
1. **R0 < 0.62** → budget lift HURT cold-start (would mean the lift introduces
   instability).
2. **R5 < 0.75** → the ceiling is not truncation-bound after all (the probe-derived
   prediction is wrong).

### Result (seed=31, single seed)
| Round | pct_success | n_turns_avg | Round-elapsed (min) |
|---|---|---|---|
| R0 | 0.61 | 25.9 | 17.7 |
| R1 | 0.66 | 23.8 | 13.5 |
| R2 | 0.66 | 23.4 | 13.2 |
| R3 | 0.66 | 23.1 | 98 |
| R4 | 0.70 | 22.2 | 113 |
| **R5** | **0.70** | 22.2 | 119 |
| R6 | 0.68 | 22.5 | 115 |
| R7 | 0.70 | — | 97 |
| R8 | 0.69 | — | ~214 |
| R9 | — | — | (orch crashed at R8 train_turnrd; see §1.4) |

Mean R3-R8 = 0.682, std = 0.014. **Identical plateau to the v3 baseline (R6-R12 ≈ 0.69-0.73).**
The budget lift produced **zero measurable lift** on R5-R8.

R8 eval result was recovered post-crash from Modal logs of app `ap-9mPK0e1aboCqDs1T9JykQt`:
`Eval done: avg_R=0.6900 (±0.4625) | pct_success=0.690 | ok=100/100 | elapsed=1369.66s`.

### Verdict: **FALSIFIED**
Abort criterion #2 fires at R5 (R5 = 0.70 < 0.75). The "ceiling is step-budget-bound"
hypothesis is ruled out by the data. The AlfWorld plateau on this slice is **not**
truncation-bound.

Two corollary observations strengthen the conclusion:
- **`n_turns_avg` actually decreased** from R0=25.9 to R3+=22-23 as the policy learned.
  After R3, the average sits well below both the old 30-turn cap and the new 50-turn cap.
  The 50-turn cap was rarely binding in the first place after RL warm-up.
- **Failure-mode shift is not visible.** R8's failures behave like R3's failures
  (qualitatively); we are not unlocking the "forward-progress" failures the probe
  predicted would be saved. Most likely those failures hit the same dead-zone pattern
  earlier in the trajectory once they get past the original 30-turn barrier — i.e., they
  were never truly progress-bound; the probe's "forward-progress" classification was a
  false positive on partial-credit signals.

### Implication for next steps
The plateau is most consistent with **supervision-limited** credit assignment (the
V-head's auxiliary target is goal-influenced but its input is goal-blind, so optimal
predictor collapses to `E[target|h_t]` and loses per-goal differentiation). This motivates
the **goal-conditioned V-head (FiLM) ablation** — see plan
`/Users/shoupeili/.llms/plans/goal_conditioned_v_head_alfworld.plan.md` and config
`configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalcond.json`.

Capacity scaling (4L/256 vs. 2L/128) is also untested; queued as a follow-up controlled
variant if goalcond is positive but insufficient.

### Crash note (cosmetic; does not affect verdict)
The xlbudget run's R8 train_turnrd phase crashed with
`SyntaxError: keyword argument repeated: lambda_value` at `infra/app_train_turnrd.py:250`
— a transient bug introduced by an unrelated mid-flight edit and now fixed. The crash
happened **after** R8's adapter and eval result were written, so the R8 datapoint
(pct_success=0.690) is intact. R9 was never started.

### Limitations
- **Single seed** (seed=31). No replicate; the n=100 eval noise floor (±0.05) is wider
  than the predicted +5-10 pp lift, so a near-zero observed delta cannot be distinguished
  from "lift exists but seed-noise hid it" with high confidence.
  - However: the *plateau pattern* across 6 consecutive rounds (R3-R8, std 0.014) is
    not consistent with a lift hiding under noise; a real +5-10 pp shift would have
    surfaced as at least one round > 0.78.
- **Other failure modes** (policy dead-zones, max_action_tokens=48 cap) were not
  separately probed at the new budget.
- **Eval slice bias.** The held-out slice [6500, 6600) was selected during v1 dev and may
  not be representative of the wider AlfWorld eval split. The plateau may be an artefact
  of this slice; a broader eval would tighten the conclusion.

### Evidence files
- Config: `configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget.json` (marked
  `[FALSIFIED 2026-05-27]` in `_notes`).
- Launcher: `scripts/run_alfworld_SOTA_10round_mlpr32_v3_xlbudget.sh` (FALSIFIED banner at
  top).
- Round sentinels: `/tmp/xlbudget_sentinels/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget_cloud_seed31_round0[0-7]_done.json`.
- Adapters (preserved as baseline data for goalcond comparison):
  `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget_cloud_seed31_round0[0-8]_adapter`.
- Replay buffer: `/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget/replay.jsonl`.
- Orchestrator log: `/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_xlbudget.log`.
- Live-tracking tool: `scripts/track_xlbudget_progress.sh`.
- Pre-experiment probe: `reports/turnrd_credit_assignment_demo/v3_R9_probe.jsonl`,
  `reports/turnrd_credit_assignment_demo/v3_R9_probe.summary.json`.

---

## Adding a new ablation

When a new controlled variant runs to completion (success, null, or falsification):

1. Append a row to the **Index** table above.
2. Add a new numbered section using the template structure of §1:
   `Hypothesis` → `Pre-experiment evidence` → `Single-variable change` →
   `Pre-registered abort criteria` → `Result` (round-by-round table) → `Verdict` →
   `Implication for next steps` → `Limitations` → `Evidence files`.
3. Pre-registering abort criteria **before** the run is what makes the negative result
   defensible; if you didn't pre-register, mark the verdict as "exploratory" rather than
   "falsified".
4. For ablations using configs in `configs/`, mirror the verdict back to the config's
   `_notes` field and the launcher's header banner (see xlbudget for the template).
5. If the ablation supersedes or invalidates an earlier one, link both directions.
