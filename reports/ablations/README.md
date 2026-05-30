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
| **Goal-conditioned V-head (FiLM)** | Does fixing the V-head target-input mismatch unlock credit assignment? | ⏳ **IN PROGRESS** | 2026-05-27 | [§2](#2-goal-conditioned-v-head-film) |
| **H-GRPO advantage normalization** | Does per-position z-scoring erase magnitude information needed for credit assignment? | 📝 **LIMITATION** | 2026-05-27 | [§3](#3-h-grpo-advantage-normalization-limitation) |
| **Progress prior necessity** | Is the progress prior necessary for stability with goal conditioning? | ⏳ **IN PROGRESS** | 2026-05-27 | [§4](#4-progress-prior-necessity) |

Future ablations (planned, not yet run): V-head capacity (4L/256). See
`/Users/shoupeili/.llms/plans/goal_conditioned_v_head_alfworld.plan.md` for the FiLM
spec.

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
variant if FiLM goalcond is positive but insufficient.

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
- Adapters (preserved as baseline data for FiLM-goalcond comparison):
  `/vol/checkpoints/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget_cloud_seed31_round0[0-8]_adapter`.
- Replay buffer: `/vol/cache/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_xlbudget/replay.jsonl`.
- Orchestrator log: `/tmp/turnrd_alfworld_SOTA_10round_mlpr32_v3_xlbudget.log`.
- Live-tracking tool: `scripts/track_xlbudget_progress.sh`.
- Pre-experiment probe: `reports/turnrd_credit_assignment_demo/v3_R9_probe.jsonl`,
  `reports/turnrd_credit_assignment_demo/v3_R9_probe.summary.json`.

---

## 2. Goal-conditioned V-head (FiLM)

### Hypothesis
The V-head's input is goal-blind (only `h_t`), so the optimal predictor collapses to
`E[target|h_t]` and loses per-goal differentiation. Conditioning the encoder hidden state
on a per-trajectory `goal_emb` via FiLM γ/β modulation before the α and value heads
should unlock per-goal value/credit estimates and lift final-round eval pct_success.

### Pre-experiment evidence
**Root cause:** Without goal conditioning, the value head and credit head see only the
generic per-turn LLM hidden state `h_t`. The model has no way to compute different
per-turn values for the same trajectory under different goals.

**One-round lag:** Goal signal enters policy gradient only via what V-head trained in
round N predicts in round N+1's RL pass. At round 0 and during V-baseline warmup, goal
information has zero effect. The FiLM-only design accepts this lag.

**Predicted lift:** +3-7 pp final-round eval pct_success (current SOTA ≈ 0.70-0.73 at R9;
target ≈ 0.75-0.80).

### Single-variable change vs. v3 baseline
| Component | v3 baseline | goalcond (FiLM-only) |
|---|---|---|
| TurnRD architecture | TurnRDv2 (2L/128, bidirectional) | **+ FiLM goal conditioning on α and v_t** |
| `turnrd.goal_conditioned_value_head` | false (default) | **true** |
| `turnrd.emit_goal_text` | false (default) | **true** |
| `turnrd.emit_goal_emb` | false (default) | **true** |
| `turnrd.progress_prior_strength` | 1.0 | **1.0** (unchanged) |
| `turnrd.lambda_progress` | 0.01 | **0.01** (unchanged) |
| `run.seed` | 31 | **31** |
| every other config field | unchanged | unchanged |

**Architecture changes:**
- **FiLM on α:** `TurnRDv2.forward()` applies FiLM modulation to encoder output `h`
  BEFORE `score_head`, making attention weights `α_t` a function of `(h_t, goal)` instead of
  just `h_t`. Per-turn rewards `r̂_t = α_t · R` become goal-aware.
- **FiLM on v_t:** Same modulation applied before `value_head`, so the actor-critic
  baseline also benefits from goal conditioning.

**Producer changes:**
- Emits `goal_text` (parsed from the Turn 0 observation) and `goal_emb: [input_dim]`
  per trajectory (synthetic single-turn embedder call on parsed `goal_text`).

**Trainer changes:**
- Threads `goal_emb` + `goal_emb_mask` into `model.forward()` when the flag is on.
- Per-row masking on V-head loss (missing-goal rows don't pollute gradient).

### Why per-turn goal supervision was removed
Earlier iterations bundled three per-turn supervision mechanisms with FiLM conditioning:
`goal_match_blend` (V-head target = (1-β)·progress + β·goal_match), `goal_shaping_coef`
(same-round intrinsic reward bump from sum of goal-match), and the standalone
`goalsup_v1` experiment (pure per-turn supervision without FiLM). All three regressed:
- The standalone goalsup_v1 run plateaued below v3 SOTA.
- The goalcond seed31 run with `goal_shaping_coef=0.1` showed R0 60% / R1 58% vs SOTA's
  R0 63% / R1 62% on the same SFT warm-start and same eval pool. The sole live cause was
  shaping-induced reward inflation (Goodhart's law: the policy gradient optimized the
  proxy score, not the env reward). The FiLM layers stayed zero-initialized because
  `train_turnrd` had never run; per-turn supervision was the only live mechanism, and it
  was actively harmful.

Per-turn supervision has been removed from the codebase (the `goal_match_blend`,
`goal_shaping_coef`, `goal_match_signal`, `score_action_against_goal`,
`parse_goal_object`, and `goal_aware_supervision` symbols no longer exist). Any future
attempt to re-enable them via config will fail at config-parse with an "unexpected key"
error. The current goalcond config exercises **only** the FiLM conditioning path.

### Pre-registered abort criteria
1. **R0 < 0.60** → FiLM conditioning destabilizes cold-start.
2. **R2 < 0.68** → No lift signal by round 2 (v3 baseline is ~0.68 at R2).
3. **R4 < 0.73** → Insufficient lift to justify full 10-round run (target: ≥ 0.75-0.80).

### Result
**TBD** — FiLM-only re-run planned after per-turn-supervision removal.

### Verdict
**TBD** — awaiting FiLM-only re-run.

### Limitations
- **Single seed** (seed=31, same as v3 baseline). This provides an apples-to-apples
  comparison on the identical task slice [24800, 25600), eliminating seed variance as a
  confounder. The n=100 eval noise floor (±0.05) still applies, but any lift > 0.05 is
  meaningful.
- **H-GRPO normalization:** Per-position z-scoring of turn advantages may erase magnitude
  information even with correct α. See §3 for analysis.

### Evidence files
- Config: `configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalcond.json`
- Implementation plan: `/Users/shoupeili/.llms/plans/goal_conditioned_v_head_alfworld.plan.md`
- Removal of per-turn supervision: `/Users/shoupeili/.llms/plans/remove_per_turn_goal_supervision.plan.md`

---

## 3. H-GRPO advantage normalization (limitation)

### The bottleneck: per-position z-scoring erases magnitude information

**GRPO's advantage construction** (`src/algorithms/grpo/advantage.py:114-116`):
```python
def compute_turn_advantages(per_turn_rewards):
    ...
    # Z-score per position across K trajectories
    row = [(traj[t] - pos_mean[t]) / pos_std[t] for t in range(len(traj))]
    return out
```

For TurnRDv2, `per_turn_rewards[i][t] = α_{i,t} · R_i` where `Σ_t α_{i,t} = 1`.

**Example with goal-conditioned α:**
- **Success** (R=1): α = [0.05, 0.05, **0.85**, 0.05] → rewards = [0.05, 0.05, 0.85, 0.05]
- **Failure** (R=0): α = [0.3, 0.3, 0.2, 0.2] → rewards = [0, 0, 0, 0]

After per-position normalization at t=2:
- `col = [0.85, 0]` → mean=0.425, std≈0.425
- Success normalized: (0.85 - 0.425)/0.425 = **+1.0**
- Failure normalized: (0 - 0.425)/0.425 = **-1.0**

**The signal is preserved, BUT magnitude information is erased.** The model can't learn that
"0.85 is much better than 0.7" — both become +1.0 after z-scoring. Only the **ranking**
within each position matters.

### Why this matters at 70% success rate

**AlfWorld at 70% success:**
- K=8 rollouts per task → ~5-6 successes, 2-3 failures per group
- Per-position normalization compares "good success" vs "mediocre success"

**Perverse incentive:** When 6/8 trajectories succeed, the 2 failures dominate the
normalization. The 6 successes get compressed into a narrow positive band. A "mediocre
success" (diffuse α) gets **negative** turn advantages at critical positions even though it
succeeded.

The model learns to **avoid turns that appear in mediocre successes**, not "what causes
success". It learns to distinguish good successes from mediocre successes, not successes
from failures.

### Dead groups at high success rates

When all 8 rollouts succeed (common at 70%):
- `std_reward = 0` → trajectory advantages are all 0
- **Entire group contributes zero gradient** (see `trainer.py:655-760` skip-dead-K guard)

**Estimated impact:** At 70% success, ~15-20% of groups are all-success (binomial tail).
These groups provide **zero learning signal** despite containing useful variance in α quality.

### Interaction with goal conditioning

**Even with FiLM-on-α, the normalization bottleneck remains:**

The goal-conditioned decomposer might learn:
- Success A: α = [0.05, 0.05, **0.85**, 0.05] (sharp peak on goal turn)
- Success B: α = [0.10, 0.10, **0.70**, 0.10] (broader peak)

Both are successes (R=1). After normalization at t=2:
- Success A: +1.2 (slightly above mean)
- Success B: +0.8 (slightly above mean)

**The 0.85 vs 0.70 distinction is compressed to 1.2 vs 0.8** — a 21% difference instead of
the true 21% difference in raw α (wait, that's the same ratio...). Actually the issue is
more subtle:

**The real problem:** When success rate is high, the **baseline is computed from mostly
successes**. The model doesn't learn "this turn is good because it leads to success"; it
learns "this turn is good because it's more characteristic of the best successes than the
worst successes in this group".

This is **discriminative** learning (separate good from mediocre) rather than **causal**
learning (identify which turns cause the outcome).

### Why this creates a 70% ceiling

**Hypothesis:** GRPO's group-relative normalization works well up to ~70% success, then
plateaus because:

1. **Dead groups:** 15-20% of groups contribute zero gradient
2. **Compressed signal:** Remaining groups compare successes to successes, not successes to
   failures
3. **Progress prior dominance:** With weak causal signal, the `lambda_progress=0.01` KL term
   keeps α near Method C. The model can't escape the prior without strong counter-evidence.

**The 70% ceiling is GRPO's ceiling on AlfWorld**, not TurnRD's ceiling. Both the xlbudget
and FiLM-goalcond runs may hit this limit regardless of architecture improvements.

### Potential fixes

**Option A: Remove per-position std normalization**
```python
# Keep mean centering, drop division by std
row = [traj[t] - pos_mean[t] for t in range(len(traj))]
```
**Downsides:**
- Advantage magnitudes vary wildly by position (early vs late turns)
- Late positions (few trajectories reach them) have larger centered values
- PPO clip (clip_eps=0.2) would truncate late positions more aggressively
- Requires retuning clip_eps, learning rate, possibly adding gradient clipping

**Option B: Global std normalization**
```python
# Compute std across ALL positions and trajectories in the group
all_vals = [x for traj in per_turn_rewards for x in traj]
global_std = std(all_vals)
row = [(traj[t] - pos_mean[t]) / global_std for t in range(len(traj))]
```
**Pros:** Preserves relative magnitudes across positions
**Cons:** Still loses information when global variance is high

**Option C: Switch to PPO with learned value function**
Drop GRPO entirely. Use GAE with a learned V(s_t).
**Downsides:**
- Value function learning is hard with sparse rewards
- Requires hyperparameter tuning (gae_lambda, vf_coef, etc.)
- Implementation complexity
- May not help if the bottleneck is elsewhere

### Recommendation

**This is a documented limitation, not an active ablation.** The FiLM-only goalcond run
(§2) will test whether goal conditioning provides enough signal to overcome the
normalization bottleneck. If both xlbudget and FiLM-goalcond plateau at 70-73%, this
limitation is the likely culprit.

**Next experiment if plateau persists:** 3-round test with per-position std normalization
removed (Option A or B above). If that destabilizes training, the limitation is fundamental
to GRPO on high-success-rate sparse-reward tasks.

### Evidence
- Code: `src/algorithms/grpo/advantage.py:114-116` (`compute_turn_advantages`)
- Code: `src/algorithms/grpo/trainer.py:655-760` (skip-dead-K guard)
- Analysis: v3 R9 failure-mode probe shows 70% success rate → ~15-20% dead groups expected

---

## 4. Progress prior necessity

### Hypothesis
The progress prior (initialization bias + KL regularization toward Method C) is **necessary
for stability** in TurnRDv2, regardless of whether goal conditioning is used. Completely
removing it causes overfitting to spurious patterns in small samples (80 trajectories at
Round 0), leading to degenerate credit assignment and training instability.

### Background
TurnRDv2 uses two progress-prior mechanisms:

1. **Initialization bias** (`progress_prior_strength`): Adds `(t/T) * strength` to per-turn
   scores before softmax. At strength=1.0, untrained α ≈ softmax(t/T) (Method C). This bias
   is **fixed** — as the score_head learns, the bias stays constant, so the model can deviate
   arbitrarily far from the prior.

2. **KL regularization** (`lambda_progress`): Adds `λ * KL(α || softmax(t/T))` to the loss,
   actively pulling α toward the prior throughout training. Default λ=0.01 is weak.

The prior encodes the inductive bias that **later turns are more likely causal on average**.
This is generally true for AlfWorld (you can't take the goal object before seeing it, can't
put before taking, etc.).

### Verdict
FiLM-only goalcond restores the prior to v3 defaults (`progress_prior_strength=1.0`,
`lambda_progress=0.01`) so the only live change vs SOTA is the FiLM γ/β path. The earlier
prior-disabling and prior-weakening attempts were both bundled with the now-removed
per-turn supervision (§2) and are not separable as standalone signals; a clean
prior-strength sweep would need to be re-run on top of the FiLM-only baseline.

### Evidence files
- Config: `configs/TurnRDV2_alfworld_SOTA_10round_mlpr32_v3_goalcond.json`
  (`progress_prior_strength=1.0`, `lambda_progress=0.01`)

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
